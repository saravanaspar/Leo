//! Synchronous story-batch reduction for aligned sparse recurrent models.
//!
//! Every worker starts from the same canonical checkpoint, trains one complete
//! document with worker-local runtime state, and returns parameter deltas. Fixed
//! tensors merge by aligned index; context entries merge by `(order, fingerprint)`
//! so unrelated collision slots are never averaged together.

use crate::metrics::TrainingStatistics;
use crate::model::Model;
use crate::runtime::ParameterChanges;
use crate::{digest_bytes, ArtifactDigest, LeoError, LeoResult};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseF32Delta {
    pub index: usize,
    pub delta: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextKeyDelta {
    pub order: usize,
    pub key: u64,
    pub embedding_delta: Vec<f32>,
    pub observation_increment: u32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatisticsDelta {
    pub processed_bytes: u64,
    pub processed_stories: u64,
    pub training_loss_sum: f64,
    pub training_targets: u64,
    pub active_neurons_sum: u64,
    pub active_neurons_peak: u64,
    pub synaptic_events: u64,
    pub numerical_rejections: u64,
    pub persistent_ticks: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseModelDelta {
    config_fingerprint: ArtifactDigest,
    base_revision: u64,
    revision_increment: u64,
    pub threshold: Vec<SparseF32Delta>,
    pub excitability: Vec<SparseF32Delta>,
    pub recurrent_weight: Vec<SparseF32Delta>,
    pub input_weight: Vec<SparseF32Delta>,
    pub output_weight: Vec<SparseF32Delta>,
    pub output_bias: Vec<SparseF32Delta>,
    pub context_output_weight: Vec<SparseF32Delta>,
    pub context: Vec<ContextKeyDelta>,
    pub statistics: StatisticsDelta,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackedModelValues {
    pub threshold: Vec<f32>,
    pub recurrent_weight: Vec<f32>,
    pub input_weight: Vec<f32>,
    /// CUDA-native row layout: one contiguous OUTPUT_CLASSES row per changed neuron.
    pub output_weight_by_neuron: Vec<f32>,
    pub output_bias: Option<Vec<f32>>,
    pub context_keys: Vec<u64>,
    pub context_observations: Vec<u32>,
    /// One contiguous embedding row per changed context slot.
    pub context_embeddings: Vec<f32>,
    pub context_output_weight: Option<Vec<f32>>,
    pub revision_increment: u64,
    pub statistics: StatisticsDelta,
}

/// Canonical sparse parameter image prepared once per logical batch and reused
/// for every CUDA replica/lane synchronization. The packet contains host-side
/// values in the same packed layouts consumed by the existing CUDA scatter
/// kernels, avoiding repeated canonical-model gathers for each device.
#[derive(Debug, Clone)]
pub struct PackedSparseModelUpdate {
    pub threshold_indices: Vec<u32>,
    pub threshold_values: Vec<f32>,
    pub recurrent_indices: Vec<u32>,
    pub recurrent_values: Vec<f32>,
    pub input_indices: Vec<u32>,
    pub input_values: Vec<f32>,
    pub output_neurons: Vec<u32>,
    /// CUDA-native layout: one contiguous OUTPUT_CLASSES row per neuron.
    pub output_values_by_neuron: Vec<f32>,
    pub context_slots: Vec<u32>,
    pub context_keys: Vec<u64>,
    pub context_observations: Vec<u32>,
    pub context_embeddings: Vec<f32>,
    pub output_bias: Option<Vec<f32>>,
    pub context_output_weights: Option<Vec<f32>>,
    pub generation: u64,
    pub parameter_revision: u64,
    pub statistics: TrainingStatistics,
}

impl PackedSparseModelUpdate {
    pub fn from_model(model: &Model, changes: &ParameterChanges) -> LeoResult<Self> {
        fn indices_u32(label: &str, values: &[usize]) -> LeoResult<Vec<u32>> {
            values
                .iter()
                .map(|&value| {
                    u32::try_from(value).map_err(|_| {
                        LeoError::internal(format!("{label} sparse index exceeds u32"))
                    })
                })
                .collect()
        }

        fn selected_values(label: &str, source: &[f32], indices: &[usize]) -> LeoResult<Vec<f32>> {
            indices
                .iter()
                .map(|&index| {
                    source.get(index).copied().ok_or_else(|| {
                        LeoError::internal(format!("{label} sparse index is outside the model"))
                    })
                })
                .collect()
        }

        let threshold_indices = indices_u32("threshold", &changes.threshold)?;
        let threshold_values =
            selected_values("threshold", &model.neurons.threshold, &changes.threshold)?;
        let recurrent_indices = indices_u32("recurrent", &changes.recurrent_weight)?;
        let recurrent_values = selected_values(
            "recurrent",
            &model.recurrent.weight,
            &changes.recurrent_weight,
        )?;
        let input_indices = indices_u32("input", &changes.input_weight)?;
        let input_values = selected_values("input", &model.input.weights, &changes.input_weight)?;

        let output_neurons = indices_u32("output neuron", &changes.output_neurons)?;
        let neuron_count = model.neuron_count();
        let mut output_values_by_neuron = Vec::with_capacity(
            changes
                .output_neurons
                .len()
                .saturating_mul(crate::symbols::OUTPUT_CLASSES),
        );
        for &neuron in &changes.output_neurons {
            if neuron >= neuron_count {
                return Err(LeoError::internal(
                    "output neuron sparse index is outside the model",
                ));
            }
            for output in 0..crate::symbols::OUTPUT_CLASSES {
                output_values_by_neuron.push(model.output.weights[output * neuron_count + neuron]);
            }
        }

        let context_slots = indices_u32("context slot", &changes.context_slots)?;
        let dim = model.config.context.embedding_dim;
        let mut context_keys = Vec::with_capacity(changes.context_slots.len());
        let mut context_observations = Vec::with_capacity(changes.context_slots.len());
        let mut context_embeddings =
            Vec::with_capacity(changes.context_slots.len().saturating_mul(dim));
        for &slot in &changes.context_slots {
            if slot >= model.context.keys.len() {
                return Err(LeoError::internal(
                    "context sparse slot is outside the model",
                ));
            }
            context_keys.push(model.context.keys[slot]);
            context_observations.push(model.context.observations[slot]);
            let start = slot * dim;
            context_embeddings.extend_from_slice(&model.context.embeddings[start..start + dim]);
        }

        Ok(Self {
            threshold_indices,
            threshold_values,
            recurrent_indices,
            recurrent_values,
            input_indices,
            input_values,
            output_neurons,
            output_values_by_neuron,
            context_slots,
            context_keys,
            context_observations,
            context_embeddings,
            output_bias: changes.output_bias_dirty.then(|| model.output.bias.clone()),
            context_output_weights: changes
                .context_output_dirty
                .then(|| model.context.output_weights.clone()),
            generation: model.generation,
            parameter_revision: model.parameter_revision,
            statistics: model.statistics.clone(),
        })
    }

    pub fn apply_to_model(&self, model: &mut Model) -> LeoResult<()> {
        fn overwrite_selected(
            label: &str,
            target: &mut [f32],
            indices: &[u32],
            values: &[f32],
        ) -> LeoResult<()> {
            if indices.len() != values.len() {
                return Err(LeoError::internal(format!(
                    "{label} packed sparse index/value counts differ"
                )));
            }
            for (&index, &value) in indices.iter().zip(values) {
                let slot = target.get_mut(index as usize).ok_or_else(|| {
                    LeoError::internal(format!("{label} packed sparse index is outside model"))
                })?;
                *slot = value;
            }
            Ok(())
        }

        overwrite_selected(
            "threshold",
            &mut model.neurons.threshold,
            &self.threshold_indices,
            &self.threshold_values,
        )?;
        overwrite_selected(
            "recurrent",
            &mut model.recurrent.weight,
            &self.recurrent_indices,
            &self.recurrent_values,
        )?;
        overwrite_selected(
            "input",
            &mut model.input.weights,
            &self.input_indices,
            &self.input_values,
        )?;

        let neuron_count = model.neuron_count();
        let expected_output_values = self
            .output_neurons
            .len()
            .saturating_mul(crate::symbols::OUTPUT_CLASSES);
        if self.output_values_by_neuron.len() != expected_output_values {
            return Err(LeoError::internal(
                "packed output rows do not match output neuron count",
            ));
        }
        for (row, &neuron) in self.output_neurons.iter().enumerate() {
            let neuron = neuron as usize;
            if neuron >= neuron_count {
                return Err(LeoError::internal(
                    "packed output neuron is outside the model",
                ));
            }
            for output in 0..crate::symbols::OUTPUT_CLASSES {
                model.output.weights[output * neuron_count + neuron] =
                    self.output_values_by_neuron[row * crate::symbols::OUTPUT_CLASSES + output];
            }
        }

        let dim = model.config.context.embedding_dim;
        if self.context_slots.len() != self.context_keys.len()
            || self.context_slots.len() != self.context_observations.len()
            || self.context_embeddings.len() != self.context_slots.len().saturating_mul(dim)
        {
            return Err(LeoError::internal(
                "packed context rows have invalid dimensions",
            ));
        }
        for (row, &slot) in self.context_slots.iter().enumerate() {
            let slot = slot as usize;
            if slot >= model.context.keys.len() {
                return Err(LeoError::internal(
                    "packed context slot is outside the model",
                ));
            }
            model.context.keys[slot] = self.context_keys[row];
            model.context.observations[slot] = self.context_observations[row];
            let start = slot * dim;
            let source = row * dim;
            model.context.embeddings[start..start + dim]
                .copy_from_slice(&self.context_embeddings[source..source + dim]);
        }
        if let Some(values) = &self.output_bias {
            if values.len() != model.output.bias.len() {
                return Err(LeoError::internal(
                    "packed output bias has invalid dimensions",
                ));
            }
            model.output.bias.copy_from_slice(values);
        }
        if let Some(values) = &self.context_output_weights {
            if values.len() != model.context.output_weights.len() {
                return Err(LeoError::internal(
                    "packed context output weights have invalid dimensions",
                ));
            }
            model.context.output_weights.copy_from_slice(values);
        }
        model.generation = self.generation;
        model.parameter_revision = self.parameter_revision;
        model.statistics = self.statistics.clone();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeMetrics {
    pub workers: usize,
    pub fixed_parameter_updates: usize,
    pub context_keys: usize,
}

impl SparseModelDelta {
    pub fn between(base: &Model, trained: &Model) -> LeoResult<Self> {
        ensure_aligned_models(base, trained)?;
        let slots_per_order = base.config.context.slots_per_order;
        let embedding_dim = base.config.context.embedding_dim;
        let mut base_context = HashMap::with_capacity(base.context.keys.len());
        for (slot, &key) in base.context.keys.iter().enumerate() {
            if key != 0 {
                base_context.insert((slot / slots_per_order + 1, key), slot);
            }
        }

        let mut context = Vec::new();
        for (slot, &key) in trained.context.keys.iter().enumerate() {
            if key == 0 {
                continue;
            }
            let order = slot / slots_per_order + 1;
            let trained_range = trained.context.embedding_range(slot, embedding_dim);
            let trained_embedding = &trained.context.embeddings[trained_range];
            let (base_embedding, base_observations) = match base_context.get(&(order, key)) {
                Some(&base_slot) => {
                    let range = base.context.embedding_range(base_slot, embedding_dim);
                    (
                        Some(&base.context.embeddings[range]),
                        base.context.observations[base_slot],
                    )
                }
                None => (None, 0),
            };
            let mut embedding_delta = Vec::with_capacity(embedding_dim);
            let mut changed = false;
            for dimension in 0..embedding_dim {
                let before = base_embedding.map_or(0.0, |values| values[dimension]);
                let delta = trained_embedding[dimension] - before;
                changed |= delta != 0.0;
                embedding_delta.push(delta);
            }
            let observation_increment =
                trained.context.observations[slot].saturating_sub(base_observations);
            if changed || observation_increment > 0 {
                context.push(ContextKeyDelta {
                    order,
                    key,
                    embedding_delta,
                    observation_increment,
                });
            }
        }

        let mut delta = Self {
            config_fingerprint: config_fingerprint(base),
            base_revision: base.parameter_revision,
            revision_increment: trained
                .parameter_revision
                .saturating_sub(base.parameter_revision),
            threshold: diff_f32(&base.neurons.threshold, &trained.neurons.threshold),
            excitability: diff_f32(&base.neurons.excitability, &trained.neurons.excitability),
            recurrent_weight: diff_f32(&base.recurrent.weight, &trained.recurrent.weight),
            input_weight: diff_f32(&base.input.weights, &trained.input.weights),
            output_weight: diff_f32(&base.output.weights, &trained.output.weights),
            output_bias: diff_f32(&base.output.bias, &trained.output.bias),
            context_output_weight: diff_f32(
                &base.context.output_weights,
                &trained.context.output_weights,
            ),
            context,
            statistics: statistics_delta(&base.statistics, &trained.statistics),
        };
        delta.canonicalize_order();
        Ok(delta)
    }

    pub fn between_tracked(
        base: &Model,
        trained: &Model,
        changes: &ParameterChanges,
    ) -> LeoResult<Self> {
        ensure_aligned_models(base, trained)?;
        let neuron_count = base.neuron_count();
        let mut output_weight = Vec::with_capacity(
            changes
                .output_neurons
                .len()
                .saturating_mul(crate::symbols::OUTPUT_CLASSES),
        );
        for &neuron in &changes.output_neurons {
            if neuron >= neuron_count {
                return Err(LeoError::internal(
                    "tracked output neuron is outside the model",
                ));
            }
            for output in 0..crate::symbols::OUTPUT_CLASSES {
                let index = output * neuron_count + neuron;
                let delta = trained.output.weights[index] - base.output.weights[index];
                if delta != 0.0 {
                    output_weight.push(SparseF32Delta { index, delta });
                }
            }
        }

        let mut delta = Self {
            config_fingerprint: config_fingerprint(base),
            base_revision: base.parameter_revision,
            revision_increment: trained
                .parameter_revision
                .saturating_sub(base.parameter_revision),
            threshold: diff_selected(
                &base.neurons.threshold,
                &trained.neurons.threshold,
                &changes.threshold,
            )?,
            excitability: Vec::new(),
            recurrent_weight: diff_selected(
                &base.recurrent.weight,
                &trained.recurrent.weight,
                &changes.recurrent_weight,
            )?,
            input_weight: diff_selected(
                &base.input.weights,
                &trained.input.weights,
                &changes.input_weight,
            )?,
            output_weight,
            output_bias: if changes.output_bias_dirty {
                diff_f32(&base.output.bias, &trained.output.bias)
            } else {
                Vec::new()
            },
            context_output_weight: if changes.context_output_dirty {
                diff_f32(
                    &base.context.output_weights,
                    &trained.context.output_weights,
                )
            } else {
                Vec::new()
            },
            context: context_deltas_for_slots(base, trained, &changes.context_slots)?,
            statistics: statistics_delta(&base.statistics, &trained.statistics),
        };
        delta.canonicalize_order();
        Ok(delta)
    }

    /// Build the same sparse worker delta as `between_tracked`, but from a
    /// device-side sparse snapshot instead of materializing a complete worker
    /// `Model` on the host. This is used by the exact CUDA logical-batch path:
    /// each story owns its learned parameter image on-device and only touched
    /// rows cross PCIe at the batch-end canonical mean barrier.
    pub fn from_tracked_values(
        base: &Model,
        changes: &ParameterChanges,
        values: TrackedModelValues,
    ) -> LeoResult<Self> {
        let TrackedModelValues {
            threshold,
            recurrent_weight,
            input_weight,
            output_weight_by_neuron,
            output_bias,
            context_keys,
            context_observations,
            context_embeddings,
            context_output_weight,
            revision_increment,
            statistics,
        } = values;

        fn selected_deltas(
            base: &[f32],
            indices: &[usize],
            values: &[f32],
            label: &str,
        ) -> LeoResult<Vec<SparseF32Delta>> {
            if indices.len() != values.len() {
                return Err(LeoError::internal(format!(
                    "tracked {label} value count does not match index count"
                )));
            }
            let mut result = Vec::with_capacity(indices.len());
            for (&index, &after) in indices.iter().zip(values) {
                let before = *base.get(index).ok_or_else(|| {
                    LeoError::internal(format!("tracked {label} index is outside the model"))
                })?;
                if !after.is_finite() {
                    return Err(LeoError::internal(format!(
                        "tracked {label} value is non-finite"
                    )));
                }
                let delta = after - before;
                if delta != 0.0 {
                    result.push(SparseF32Delta { index, delta });
                }
            }
            Ok(result)
        }

        let neuron_count = base.neuron_count();
        let expected_output_values = changes
            .output_neurons
            .len()
            .saturating_mul(crate::symbols::OUTPUT_CLASSES);
        if output_weight_by_neuron.len() != expected_output_values {
            return Err(LeoError::internal(
                "tracked output value count does not match changed neuron count",
            ));
        }
        let mut output_weight = Vec::with_capacity(expected_output_values);
        for (row, &neuron) in changes.output_neurons.iter().enumerate() {
            if neuron >= neuron_count {
                return Err(LeoError::internal(
                    "tracked output neuron is outside the model",
                ));
            }
            for output in 0..crate::symbols::OUTPUT_CLASSES {
                let host_index = output * neuron_count + neuron;
                let after = output_weight_by_neuron[row * crate::symbols::OUTPUT_CLASSES + output];
                if !after.is_finite() {
                    return Err(LeoError::internal(
                        "tracked output weight value is non-finite",
                    ));
                }
                let delta = after - base.output.weights[host_index];
                if delta != 0.0 {
                    output_weight.push(SparseF32Delta {
                        index: host_index,
                        delta,
                    });
                }
            }
        }

        let output_bias = match output_bias {
            Some(after) if changes.output_bias_dirty => {
                if after.len() != base.output.bias.len()
                    || after.iter().any(|value| !value.is_finite())
                {
                    return Err(LeoError::internal(
                        "tracked output bias snapshot is invalid",
                    ));
                }
                diff_f32(&base.output.bias, &after)
            }
            Some(_) => {
                return Err(LeoError::internal(
                    "tracked output bias supplied without dirty flag",
                ));
            }
            None if changes.output_bias_dirty => {
                return Err(LeoError::internal(
                    "tracked output bias dirty flag is missing its snapshot",
                ));
            }
            None => Vec::new(),
        };

        let context_output_weight = match context_output_weight {
            Some(after) if changes.context_output_dirty => {
                if after.len() != base.context.output_weights.len()
                    || after.iter().any(|value| !value.is_finite())
                {
                    return Err(LeoError::internal(
                        "tracked context output snapshot is invalid",
                    ));
                }
                diff_f32(&base.context.output_weights, &after)
            }
            Some(_) => {
                return Err(LeoError::internal(
                    "tracked context output supplied without dirty flag",
                ));
            }
            None if changes.context_output_dirty => {
                return Err(LeoError::internal(
                    "tracked context output dirty flag is missing its snapshot",
                ));
            }
            None => Vec::new(),
        };

        let dim = base.config.context.embedding_dim;
        if context_keys.len() != changes.context_slots.len()
            || context_observations.len() != changes.context_slots.len()
            || context_embeddings.len() != changes.context_slots.len().saturating_mul(dim)
        {
            return Err(LeoError::internal(
                "tracked context snapshot dimensions do not match changed slots",
            ));
        }
        let slots_per_order = base.config.context.slots_per_order;
        let mut context = Vec::with_capacity(changes.context_slots.len());
        for (row, &slot) in changes.context_slots.iter().enumerate() {
            if slot >= base.context.keys.len() {
                return Err(LeoError::internal(
                    "tracked context slot is outside the model",
                ));
            }
            let key = context_keys[row];
            if key == 0 {
                continue;
            }
            let order = slot / slots_per_order + 1;
            let base_slot = find_context_slot(base, order, key);
            let base_observations = base_slot.map_or(0, |value| base.context.observations[value]);
            let mut embedding_delta = Vec::with_capacity(dim);
            let mut changed = false;
            for dimension in 0..dim {
                let before = base_slot.map_or(0.0, |value| {
                    base.context.embeddings[value * dim + dimension]
                });
                let after = context_embeddings[row * dim + dimension];
                if !after.is_finite() {
                    return Err(LeoError::internal(
                        "tracked context embedding value is non-finite",
                    ));
                }
                let delta = after - before;
                changed |= delta != 0.0;
                embedding_delta.push(delta);
            }
            let observation_increment = context_observations[row].saturating_sub(base_observations);
            if changed || observation_increment > 0 {
                context.push(ContextKeyDelta {
                    order,
                    key,
                    embedding_delta,
                    observation_increment,
                });
            }
        }

        let mut delta = Self {
            config_fingerprint: config_fingerprint(base),
            base_revision: base.parameter_revision,
            revision_increment,
            threshold: selected_deltas(
                &base.neurons.threshold,
                &changes.threshold,
                &threshold,
                "threshold",
            )?,
            excitability: Vec::new(),
            recurrent_weight: selected_deltas(
                &base.recurrent.weight,
                &changes.recurrent_weight,
                &recurrent_weight,
                "recurrent weight",
            )?,
            input_weight: selected_deltas(
                &base.input.weights,
                &changes.input_weight,
                &input_weight,
                "input weight",
            )?,
            output_weight,
            output_bias,
            context_output_weight,
            context,
            statistics,
        };
        delta.canonicalize_order();
        Ok(delta)
    }

    fn canonicalize_order(&mut self) {
        self.threshold.sort_unstable_by_key(|update| update.index);
        self.excitability
            .sort_unstable_by_key(|update| update.index);
        self.recurrent_weight
            .sort_unstable_by_key(|update| update.index);
        self.input_weight
            .sort_unstable_by_key(|update| update.index);
        self.output_weight
            .sort_unstable_by_key(|update| update.index);
        self.output_bias.sort_unstable_by_key(|update| update.index);
        self.context_output_weight
            .sort_unstable_by_key(|update| update.index);
        self.context
            .sort_unstable_by_key(|update| (update.order, update.key));
    }
}

pub fn apply_mean_deltas(
    master: &mut Model,
    deltas: &[SparseModelDelta],
) -> LeoResult<MergeMetrics> {
    if deltas.is_empty() {
        return Err(LeoError::internal("cannot merge an empty worker batch"));
    }
    let fingerprint = config_fingerprint(master);
    if deltas
        .iter()
        .any(|delta| delta.config_fingerprint != fingerprint)
    {
        return Err(LeoError::internal(
            "worker delta configuration does not match the canonical model",
        ));
    }
    if deltas
        .iter()
        .any(|delta| delta.base_revision != master.parameter_revision)
    {
        return Err(LeoError::internal(
            "worker delta was not produced from the current canonical revision",
        ));
    }
    let worker_count = deltas.len();
    let mut fixed_parameter_updates = 0usize;

    fixed_parameter_updates += apply_sparse_mean(
        &mut master.neurons.threshold,
        deltas.iter().map(|delta| delta.threshold.as_slice()),
        worker_count,
    )?;
    fixed_parameter_updates += apply_sparse_mean(
        &mut master.neurons.excitability,
        deltas.iter().map(|delta| delta.excitability.as_slice()),
        worker_count,
    )?;
    fixed_parameter_updates += apply_sparse_mean(
        &mut master.recurrent.weight,
        deltas.iter().map(|delta| delta.recurrent_weight.as_slice()),
        worker_count,
    )?;
    fixed_parameter_updates += apply_sparse_mean(
        &mut master.input.weights,
        deltas.iter().map(|delta| delta.input_weight.as_slice()),
        worker_count,
    )?;
    fixed_parameter_updates += apply_sparse_mean(
        &mut master.output.weights,
        deltas.iter().map(|delta| delta.output_weight.as_slice()),
        worker_count,
    )?;
    fixed_parameter_updates += apply_sparse_mean(
        &mut master.output.bias,
        deltas.iter().map(|delta| delta.output_bias.as_slice()),
        worker_count,
    )?;
    fixed_parameter_updates += apply_sparse_mean(
        &mut master.context.output_weights,
        deltas
            .iter()
            .map(|delta| delta.context_output_weight.as_slice()),
        worker_count,
    )?;

    let context_keys = apply_context_deltas(master, deltas, worker_count)?;
    apply_statistics_deltas(&mut master.statistics, deltas);
    let revision_increment = deltas.iter().fold(0u64, |total, delta| {
        total.saturating_add(delta.revision_increment)
    });
    master.parameter_revision = master.parameter_revision.saturating_add(revision_increment);
    project_parameter_constraints_sparse(master, deltas)?;
    validate_sparse_merge(master, deltas)?;

    Ok(MergeMetrics {
        workers: worker_count,
        fixed_parameter_updates,
        context_keys,
    })
}

/// Return the exact canonical rows/slots that must be synchronized after a
/// sparse worker mean merge. Context keys can move to a different collision
/// slot during keyed merging, so both the worker-local source slots and the
/// final canonical destinations are included.
pub fn merged_parameter_changes(
    canonical: &mut Model,
    deltas: &[SparseModelDelta],
    raw_changes: &[ParameterChanges],
) -> LeoResult<ParameterChanges> {
    let neuron_count = canonical.neuron_count();
    let mut changes = ParameterChanges::default();
    for delta in deltas {
        changes
            .threshold
            .extend(delta.threshold.iter().map(|update| update.index));
        changes
            .recurrent_weight
            .extend(delta.recurrent_weight.iter().map(|update| update.index));
        changes
            .input_weight
            .extend(delta.input_weight.iter().map(|update| update.index));
        changes.output_neurons.extend(
            delta
                .output_weight
                .iter()
                .map(|update| update.index % neuron_count),
        );
        changes.output_bias_dirty |= !delta.output_bias.is_empty();
        changes.context_output_dirty |= !delta.context_output_weight.is_empty();
        for update in &delta.context {
            if let (Some(slot), _) =
                canonical.resolve_context_slot(update.order, update.key, false)?
            {
                changes.context_slots.push(slot);
            }
        }
    }

    for raw in raw_changes {
        changes.threshold.extend(raw.threshold.iter().copied());
        changes
            .recurrent_weight
            .extend(raw.recurrent_weight.iter().copied());
        changes
            .input_weight
            .extend(raw.input_weight.iter().copied());
        changes
            .output_neurons
            .extend(raw.output_neurons.iter().copied());
        changes
            .context_slots
            .extend(raw.context_slots.iter().copied());
        changes.output_bias_dirty |= raw.output_bias_dirty;
        changes.context_output_dirty |= raw.context_output_dirty;
    }

    fn dedup(values: &mut Vec<usize>) {
        values.sort_unstable();
        values.dedup();
    }
    dedup(&mut changes.threshold);
    dedup(&mut changes.recurrent_weight);
    dedup(&mut changes.input_weight);
    dedup(&mut changes.output_neurons);
    dedup(&mut changes.context_slots);
    Ok(changes)
}

fn ensure_aligned_models(base: &Model, trained: &Model) -> LeoResult<()> {
    if config_fingerprint(base) != config_fingerprint(trained)
        || base.recurrent.target_neuron != trained.recurrent.target_neuron
        || base.recurrent.target_branch != trained.recurrent.target_branch
        || base.recurrent.delay != trained.recurrent.delay
        || base.input.targets != trained.input.targets
        || base.input.branches != trained.input.branches
        || base.neurons.neuron_type != trained.neurons.neuron_type
    {
        return Err(LeoError::internal(
            "worker model topology is not aligned with the canonical checkpoint",
        ));
    }
    Ok(())
}

fn diff_f32(before: &[f32], after: &[f32]) -> Vec<SparseF32Delta> {
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter_map(|(index, (before, after))| {
            let delta = *after - *before;
            (delta != 0.0).then_some(SparseF32Delta { index, delta })
        })
        .collect()
}

fn diff_selected(
    before: &[f32],
    after: &[f32],
    indices: &[usize],
) -> LeoResult<Vec<SparseF32Delta>> {
    let mut result = Vec::with_capacity(indices.len());
    for &index in indices {
        if index >= before.len() || index >= after.len() {
            return Err(LeoError::internal(
                "tracked parameter index is outside the model",
            ));
        }
        let delta = after[index] - before[index];
        if delta != 0.0 {
            result.push(SparseF32Delta { index, delta });
        }
    }
    Ok(result)
}

fn find_context_slot(model: &Model, order: usize, key: u64) -> Option<usize> {
    let slots_per_order = model.config.context.slots_per_order;
    let base = (order - 1) * slots_per_order;
    let start = key as usize % slots_per_order;
    for offset in 0..model.config.context.probe_limit {
        let slot = base + (start + offset) % slots_per_order;
        if model.context.keys[slot] == key {
            return Some(slot);
        }
    }
    None
}

fn context_deltas_for_slots(
    base: &Model,
    trained: &Model,
    slots: &[usize],
) -> LeoResult<Vec<ContextKeyDelta>> {
    let embedding_dim = base.config.context.embedding_dim;
    let slots_per_order = base.config.context.slots_per_order;
    let mut result = Vec::with_capacity(slots.len());
    for &slot in slots {
        if slot >= trained.context.keys.len() {
            return Err(LeoError::internal(
                "tracked context slot is outside the model",
            ));
        }
        let key = trained.context.keys[slot];
        if key == 0 {
            continue;
        }
        let order = slot / slots_per_order + 1;
        let base_slot = find_context_slot(base, order, key);
        let base_observations = base_slot.map_or(0, |value| base.context.observations[value]);
        let trained_range = trained.context.embedding_range(slot, embedding_dim);
        let mut embedding_delta = Vec::with_capacity(embedding_dim);
        let mut changed = false;
        for dimension in 0..embedding_dim {
            let before = base_slot.map_or(0.0, |value| {
                base.context.embeddings[value * embedding_dim + dimension]
            });
            let delta = trained.context.embeddings[trained_range.start + dimension] - before;
            changed |= delta != 0.0;
            embedding_delta.push(delta);
        }
        let observation_increment =
            trained.context.observations[slot].saturating_sub(base_observations);
        if changed || observation_increment > 0 {
            result.push(ContextKeyDelta {
                order,
                key,
                embedding_delta,
                observation_increment,
            });
        }
    }
    Ok(result)
}

fn apply_sparse_mean<'a>(
    target: &mut [f32],
    groups: impl Iterator<Item = &'a [SparseF32Delta]>,
    worker_count: usize,
) -> LeoResult<usize> {
    let groups = groups.collect::<Vec<_>>();
    let mut positions = vec![0usize; groups.len()];
    let mut updates = 0usize;

    loop {
        let next_index = groups
            .iter()
            .zip(&positions)
            .filter_map(|(group, &position)| group.get(position).map(|update| update.index))
            .min();
        let Some(index) = next_index else {
            break;
        };
        if index >= target.len() {
            return Err(LeoError::internal("invalid sparse worker parameter delta"));
        }

        // Preserve the canonical logical-story reduction order exactly. Each
        // worker contributes at most once to a given aligned parameter because
        // SparseModelDelta is canonicalized before it reaches this reducer.
        let mut sum = 0.0f64;
        for (worker_index, group) in groups.iter().enumerate() {
            let position = positions[worker_index];
            if let Some(update) = group.get(position) {
                if update.index == index {
                    if !update.delta.is_finite() {
                        return Err(LeoError::internal("invalid sparse worker parameter delta"));
                    }
                    sum += update.delta as f64;
                    positions[worker_index] += 1;
                }
            }
        }

        target[index] += (sum / worker_count as f64) as f32;
        if !target[index].is_finite() {
            return Err(LeoError::internal("merged parameter became non-finite"));
        }
        updates = updates.saturating_add(1);
    }

    Ok(updates)
}

fn apply_context_deltas(
    master: &mut Model,
    deltas: &[SparseModelDelta],
    worker_count: usize,
) -> LeoResult<usize> {
    #[derive(Debug)]
    struct MergedContextDelta {
        order: usize,
        key: u64,
        embedding_sum: Vec<f64>,
        observations: u64,
    }

    let embedding_dim = master.config.context.embedding_dim;
    let groups = deltas
        .iter()
        .map(|delta| delta.context.as_slice())
        .collect::<Vec<_>>();
    let mut positions = vec![0usize; groups.len()];
    let mut merged = Vec::<MergedContextDelta>::new();

    loop {
        let next_key = groups
            .iter()
            .zip(&positions)
            .filter_map(|(group, &position)| {
                group.get(position).map(|update| (update.order, update.key))
            })
            .min();
        let Some((order, key)) = next_key else {
            break;
        };

        let mut embedding_sum = vec![0.0f64; embedding_dim];
        let mut observations = 0u64;
        for (worker_index, group) in groups.iter().enumerate() {
            let position = positions[worker_index];
            if let Some(update) = group.get(position) {
                if (update.order, update.key) == (order, key) {
                    if update.embedding_delta.len() != embedding_dim
                        || update.order == 0
                        || update.order > master.config.context.max_order
                        || update.key == 0
                        || update
                            .embedding_delta
                            .iter()
                            .any(|value| !value.is_finite())
                    {
                        return Err(LeoError::internal("invalid keyed context worker delta"));
                    }
                    for (sum, &value) in embedding_sum.iter_mut().zip(&update.embedding_delta) {
                        *sum += value as f64;
                    }
                    observations =
                        observations.saturating_add(u64::from(update.observation_increment));
                    positions[worker_index] += 1;
                }
            }
        }
        merged.push(MergedContextDelta {
            order,
            key,
            embedding_sum,
            observations,
        });
    }

    // Slot replacement priority remains identical to the previous reducer:
    // observation-heavy contexts first, then stable (order,key) tie-breaking.
    merged.sort_unstable_by(|left, right| {
        right
            .observations
            .cmp(&left.observations)
            .then_with(|| (left.order, left.key).cmp(&(right.order, right.key)))
    });
    for update in &merged {
        let (slot, _) = master.resolve_context_slot(update.order, update.key, true)?;
        let slot = slot.expect("allocating a valid context always returns a slot");
        let range = master.context.embedding_range(slot, embedding_dim);
        for (value, sum) in master.context.embeddings[range]
            .iter_mut()
            .zip(&update.embedding_sum)
        {
            *value += (*sum / worker_count as f64) as f32;
        }
        master.context.observations[slot] = master.context.observations[slot]
            .saturating_add(update.observations.min(u64::from(u32::MAX)) as u32);
    }
    Ok(merged.len())
}

fn statistics_delta(before: &TrainingStatistics, after: &TrainingStatistics) -> StatisticsDelta {
    StatisticsDelta {
        processed_bytes: after.processed_bytes.saturating_sub(before.processed_bytes),
        processed_stories: after
            .processed_stories
            .saturating_sub(before.processed_stories),
        training_loss_sum: after.training_loss_sum - before.training_loss_sum,
        training_targets: after
            .training_targets
            .saturating_sub(before.training_targets),
        active_neurons_sum: after
            .active_neurons_sum
            .saturating_sub(before.active_neurons_sum),
        active_neurons_peak: after.active_neurons_peak,
        synaptic_events: after.synaptic_events.saturating_sub(before.synaptic_events),
        numerical_rejections: after
            .numerical_rejections
            .saturating_sub(before.numerical_rejections),
        persistent_ticks: after
            .persistent_ticks
            .saturating_sub(before.persistent_ticks),
    }
}

fn apply_statistics_deltas(statistics: &mut TrainingStatistics, deltas: &[SparseModelDelta]) {
    for delta in deltas {
        statistics.processed_bytes = statistics
            .processed_bytes
            .saturating_add(delta.statistics.processed_bytes);
        statistics.processed_stories = statistics
            .processed_stories
            .saturating_add(delta.statistics.processed_stories);
        statistics.training_loss_sum += delta.statistics.training_loss_sum;
        statistics.training_targets = statistics
            .training_targets
            .saturating_add(delta.statistics.training_targets);
        statistics.active_neurons_sum = statistics
            .active_neurons_sum
            .saturating_add(delta.statistics.active_neurons_sum);
        statistics.active_neurons_peak = statistics
            .active_neurons_peak
            .max(delta.statistics.active_neurons_peak);
        statistics.synaptic_events = statistics
            .synaptic_events
            .saturating_add(delta.statistics.synaptic_events);
        statistics.numerical_rejections = statistics
            .numerical_rejections
            .saturating_add(delta.statistics.numerical_rejections);
        statistics.persistent_ticks = statistics
            .persistent_ticks
            .saturating_add(delta.statistics.persistent_ticks);
    }
}

fn project_parameter_constraints_sparse(
    model: &mut Model,
    deltas: &[SparseModelDelta],
) -> LeoResult<()> {
    let minimum = model.config.learning.weight_min;
    let maximum = model.config.learning.weight_max;
    let capacity = model.recurrent.capacity_per_neuron;
    let context_dim = model.config.context.embedding_dim;

    for delta in deltas {
        for update in &delta.threshold {
            let value = model
                .neurons
                .threshold
                .get_mut(update.index)
                .ok_or_else(|| LeoError::internal("merged threshold index is outside the model"))?;
            *value = (*value).clamp(0.05, 2.0);
        }
        for update in &delta.excitability {
            let value = model
                .neurons
                .excitability
                .get_mut(update.index)
                .ok_or_else(|| {
                    LeoError::internal("merged excitability index is outside the model")
                })?;
            *value = (*value).clamp(0.1, 4.0);
        }
        for update in &delta.recurrent_weight {
            let weight = model
                .recurrent
                .weight
                .get_mut(update.index)
                .ok_or_else(|| LeoError::internal("merged recurrent index is outside the model"))?;
            let source = update.index / capacity;
            *weight = (*weight).clamp(minimum, maximum);
            *weight = if model.neurons.neuron_type[source] == 0 {
                (*weight).max(0.0)
            } else {
                (*weight).min(0.0)
            };
        }
        for update in &delta.input_weight {
            let weight = model
                .input
                .weights
                .get_mut(update.index)
                .ok_or_else(|| LeoError::internal("merged input index is outside the model"))?;
            *weight = (*weight).clamp(0.0, maximum);
        }
        for update in &delta.output_weight {
            let weight =
                model.output.weights.get_mut(update.index).ok_or_else(|| {
                    LeoError::internal("merged output index is outside the model")
                })?;
            *weight = (*weight).clamp(minimum, maximum);
        }
        for update in &delta.context {
            if let (Some(slot), _) = model.resolve_context_slot(update.order, update.key, false)? {
                let start = slot * context_dim;
                for weight in &mut model.context.embeddings[start..start + context_dim] {
                    *weight = (*weight).clamp(minimum, maximum);
                }
            }
        }
        for update in &delta.context_output_weight {
            let weight = model
                .context
                .output_weights
                .get_mut(update.index)
                .ok_or_else(|| {
                    LeoError::internal("merged context output index is outside the model")
                })?;
            *weight = (*weight).clamp(minimum, maximum);
        }
    }
    Ok(())
}

fn validate_sparse_merge(model: &mut Model, deltas: &[SparseModelDelta]) -> LeoResult<()> {
    if !model.statistics.training_loss_sum.is_finite() {
        return Err(LeoError::internal(
            "merged model statistics contain a non-finite loss",
        ));
    }

    let minimum = model.config.learning.weight_min;
    let maximum = model.config.learning.weight_max;
    let capacity = model.recurrent.capacity_per_neuron;
    let context_dim = model.config.context.embedding_dim;

    for delta in deltas {
        for update in &delta.threshold {
            let value = model.neurons.threshold[update.index];
            if !value.is_finite() || !(0.05..=2.0).contains(&value) {
                return Err(LeoError::internal(
                    "merged threshold violates canonical parameter bounds",
                ));
            }
        }
        for update in &delta.excitability {
            let value = model.neurons.excitability[update.index];
            if !value.is_finite() || !(0.1..=4.0).contains(&value) {
                return Err(LeoError::internal(
                    "merged excitability violates canonical parameter bounds",
                ));
            }
        }
        for update in &delta.recurrent_weight {
            let weight = model.recurrent.weight[update.index];
            let source = update.index / capacity;
            let sign_valid = if model.neurons.neuron_type[source] == 0 {
                weight >= 0.0
            } else {
                weight <= 0.0
            };
            if !weight.is_finite() || weight < minimum || weight > maximum || !sign_valid {
                return Err(LeoError::internal(
                    "merged recurrent parameter violates model constraints",
                ));
            }
        }
        for update in &delta.input_weight {
            let weight = model.input.weights[update.index];
            if !weight.is_finite() || weight < 0.0 || weight > maximum {
                return Err(LeoError::internal(
                    "merged input parameter violates model constraints",
                ));
            }
        }
        for update in &delta.output_weight {
            let weight = model
                .output
                .weights
                .get(update.index)
                .copied()
                .ok_or_else(|| LeoError::internal("merged output index is outside the model"))?;
            if !weight.is_finite() || weight < minimum || weight > maximum {
                return Err(LeoError::internal(
                    "merged output weight violates canonical parameter bounds",
                ));
            }
        }
        for update in &delta.output_bias {
            if !model.output.bias[update.index].is_finite() {
                return Err(LeoError::internal("merged output bias is non-finite"));
            }
        }
        for update in &delta.context {
            if let (Some(slot), _) = model.resolve_context_slot(update.order, update.key, false)? {
                if model.context.keys[slot] == 0 && model.context.observations[slot] != 0 {
                    return Err(LeoError::internal(
                        "merged context slot has observations without a key",
                    ));
                }
                let start = slot * context_dim;
                if model.context.embeddings[start..start + context_dim]
                    .iter()
                    .any(|value| !value.is_finite() || *value < minimum || *value > maximum)
                {
                    return Err(LeoError::internal(
                        "merged context embedding violates canonical parameter bounds",
                    ));
                }
            }
        }
        for update in &delta.context_output_weight {
            let weight = model.context.output_weights[update.index];
            if !weight.is_finite() || weight < minimum || weight > maximum {
                return Err(LeoError::internal(
                    "merged context output weight violates canonical parameter bounds",
                ));
            }
        }
    }
    Ok(())
}

fn config_fingerprint(model: &Model) -> ArtifactDigest {
    digest_bytes(model.config.to_toml().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{
        apply_mean_deltas, statistics_delta, PackedSparseModelUpdate, SparseModelDelta,
        TrackedModelValues,
    };
    use crate::symbols::{BEGIN_DOCUMENT, END_DOCUMENT, OUTPUT_CLASSES};
    use crate::{Config, Model, ParameterChanges, Permission, Runtime};

    fn model() -> Model {
        let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
        Model::initialize(config).unwrap()
    }

    #[test]
    fn fixed_parameters_merge_as_a_synchronous_mean_delta() {
        let base = model();
        let mut first = base.clone();
        let mut second = base.clone();
        first.output.bias[0] += 0.2;
        second.output.bias[0] += 0.4;
        first.statistics.processed_stories += 1;
        second.statistics.processed_stories += 1;
        let deltas = [
            SparseModelDelta::between(&base, &first).unwrap(),
            SparseModelDelta::between(&base, &second).unwrap(),
        ];
        let mut master = base.clone();
        let metrics = apply_mean_deltas(&mut master, &deltas).unwrap();
        assert!((master.output.bias[0] - (base.output.bias[0] + 0.3)).abs() < 1.0e-6);
        assert_eq!(master.statistics.processed_stories, 2);
        assert_eq!(metrics.workers, 2);
    }

    #[test]
    fn tracked_delta_matches_full_delta_for_one_worker_story() {
        let base = model();
        let mut runtime = Runtime::new(base.clone()).unwrap();
        runtime.enable_parameter_tracking();
        runtime.begin_document();
        runtime
            .step(BEGIN_DOCUMENT, Some(b'A' as u32), Permission::Training)
            .unwrap();
        runtime
            .step(b'A' as u32, Some(END_DOCUMENT), Permission::Training)
            .unwrap();
        runtime.finish_document();

        let full = SparseModelDelta::between(&base, runtime.model()).unwrap();
        let tracked =
            SparseModelDelta::between_tracked(&base, runtime.model(), &runtime.parameter_changes())
                .unwrap();
        assert_eq!(tracked, full);
    }

    #[test]
    fn device_sparse_snapshot_matches_tracked_worker_delta() {
        let base = model();
        let mut runtime = Runtime::new(base.clone()).unwrap();
        runtime.enable_parameter_tracking();
        runtime.begin_document();
        for (symbol, target) in [
            (BEGIN_DOCUMENT, Some(b'A' as u32)),
            (b'A' as u32, Some(b'B' as u32)),
            (b'B' as u32, Some(END_DOCUMENT)),
        ] {
            runtime.step(symbol, target, Permission::Training).unwrap();
        }
        runtime.finish_document();

        let trained = runtime.model();
        let changes = runtime.parameter_changes();
        let neuron_count = base.neuron_count();
        let output_classes = crate::symbols::OUTPUT_CLASSES;
        let mut output_weight_by_neuron =
            Vec::with_capacity(changes.output_neurons.len().saturating_mul(output_classes));
        for &neuron in &changes.output_neurons {
            for output in 0..output_classes {
                output_weight_by_neuron
                    .push(trained.output.weights[output * neuron_count + neuron]);
            }
        }
        let dim = base.config.context.embedding_dim;
        let mut context_keys = Vec::with_capacity(changes.context_slots.len());
        let mut context_observations = Vec::with_capacity(changes.context_slots.len());
        let mut context_embeddings =
            Vec::with_capacity(changes.context_slots.len().saturating_mul(dim));
        for &slot in &changes.context_slots {
            context_keys.push(trained.context.keys[slot]);
            context_observations.push(trained.context.observations[slot]);
            let start = slot * dim;
            context_embeddings.extend_from_slice(&trained.context.embeddings[start..start + dim]);
        }

        let values = TrackedModelValues {
            threshold: changes
                .threshold
                .iter()
                .map(|&index| trained.neurons.threshold[index])
                .collect(),
            recurrent_weight: changes
                .recurrent_weight
                .iter()
                .map(|&index| trained.recurrent.weight[index])
                .collect(),
            input_weight: changes
                .input_weight
                .iter()
                .map(|&index| trained.input.weights[index])
                .collect(),
            output_weight_by_neuron,
            output_bias: changes
                .output_bias_dirty
                .then(|| trained.output.bias.clone()),
            context_keys,
            context_observations,
            context_embeddings,
            context_output_weight: changes
                .context_output_dirty
                .then(|| trained.context.output_weights.clone()),
            revision_increment: trained
                .parameter_revision
                .saturating_sub(base.parameter_revision),
            statistics: statistics_delta(&base.statistics, &trained.statistics),
        };

        let expected = SparseModelDelta::between_tracked(&base, trained, &changes).unwrap();
        let observed = SparseModelDelta::from_tracked_values(&base, &changes, values).unwrap();
        assert_eq!(observed, expected);
    }

    #[test]
    fn output_weights_merge_in_logical_worker_order_without_tree_maps() {
        let base = model();
        let neuron = 3usize;
        let output = 7usize;
        let index = output * base.neuron_count() + neuron;
        let mut first = base.clone();
        let mut second = base.clone();
        first.output.weights[index] += 0.2;
        second.output.weights[index] -= 0.1;
        let deltas = [
            SparseModelDelta::between(&base, &first).unwrap(),
            SparseModelDelta::between(&base, &second).unwrap(),
        ];
        assert_eq!(deltas[0].output_weight.len(), 1);
        assert_eq!(deltas[1].output_weight.len(), 1);
        assert_eq!(deltas[0].output_weight[0].index, index);

        let mut master = base.clone();
        apply_mean_deltas(&mut master, &deltas).unwrap();
        let expected = base.output.weights[index] + 0.05;
        assert!((master.output.weights[index] - expected).abs() < 1.0e-6);
    }

    #[test]
    fn packed_sparse_update_reproduces_selected_canonical_rows() {
        let base = model();
        let mut canonical = base.clone();
        canonical.neurons.threshold[0] += 0.125;
        canonical.recurrent.weight[0] *= 0.5;
        canonical.input.weights[0] += 0.03125;
        let neuron_count = canonical.neuron_count();
        for output in 0..OUTPUT_CLASSES {
            canonical.output.weights[output * neuron_count] += output as f32 * 1.0e-6;
        }
        canonical.output.bias[0] += 0.25;
        canonical.context.keys[0] = 42;
        canonical.context.observations[0] = 7;
        canonical.context.embeddings[0] = -0.125;
        canonical.context.output_weights[0] += 0.0625;
        canonical.parameter_revision = canonical.parameter_revision.saturating_add(3);
        canonical.statistics.processed_stories = 9;

        let changes = ParameterChanges {
            threshold: vec![0],
            recurrent_weight: vec![0],
            input_weight: vec![0],
            output_neurons: vec![0],
            context_slots: vec![0],
            output_bias_dirty: true,
            context_output_dirty: true,
        };
        let packet = PackedSparseModelUpdate::from_model(&canonical, &changes).unwrap();
        let mut replica = base.clone();
        packet.apply_to_model(&mut replica).unwrap();

        assert_eq!(replica.neurons.threshold[0], canonical.neurons.threshold[0]);
        assert_eq!(replica.recurrent.weight[0], canonical.recurrent.weight[0]);
        assert_eq!(replica.input.weights[0], canonical.input.weights[0]);
        for output in 0..OUTPUT_CLASSES {
            assert_eq!(
                replica.output.weights[output * replica.neuron_count()],
                canonical.output.weights[output * canonical.neuron_count()]
            );
        }
        assert_eq!(replica.output.bias, canonical.output.bias);
        assert_eq!(replica.context.keys[0], canonical.context.keys[0]);
        assert_eq!(
            replica.context.observations[0],
            canonical.context.observations[0]
        );
        assert_eq!(
            replica.context.embeddings[0],
            canonical.context.embeddings[0]
        );
        assert_eq!(
            replica.context.output_weights,
            canonical.context.output_weights
        );
        assert_eq!(replica.parameter_revision, canonical.parameter_revision);
        assert_eq!(replica.statistics.processed_stories, 9);
    }

    #[test]
    fn stale_worker_deltas_are_rejected() {
        let base = model();
        let mut trained = base.clone();
        trained.output.bias[0] += 0.1;
        let delta = SparseModelDelta::between(&base, &trained).unwrap();
        let mut advanced_master = base;
        advanced_master.parameter_revision += 1;
        let error = apply_mean_deltas(&mut advanced_master, &[delta]).unwrap_err();
        assert!(error.message().contains("current canonical revision"));
    }

    #[test]
    fn context_entries_merge_by_fingerprint_not_collision_slot() {
        let base = model();
        let slots = base.config.context.slots_per_order;
        let first_key = 1u64;
        let second_key = first_key + slots as u64;
        let mut first = base.clone();
        let mut second = base.clone();
        let first_slot = first
            .resolve_context_slot(1, first_key, true)
            .unwrap()
            .0
            .unwrap();
        let second_slot = second
            .resolve_context_slot(1, second_key, true)
            .unwrap()
            .0
            .unwrap();
        let dim = base.config.context.embedding_dim;
        first.context.embeddings[first_slot * dim] = 0.4;
        second.context.embeddings[second_slot * dim] = -0.2;
        first.context.observations[first_slot] = 1;
        second.context.observations[second_slot] = 1;
        let deltas = [
            SparseModelDelta::between(&base, &first).unwrap(),
            SparseModelDelta::between(&base, &second).unwrap(),
        ];
        let mut master = base;
        apply_mean_deltas(&mut master, &deltas).unwrap();
        let keys = &master.context.keys[..slots];
        assert!(keys.contains(&first_key));
        assert!(keys.contains(&second_key));
    }
}
