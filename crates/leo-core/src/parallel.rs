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
use std::collections::{BTreeMap, HashMap};

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
    project_parameter_constraints(master);
    master.validate()?;

    Ok(MergeMetrics {
        workers: worker_count,
        fixed_parameter_updates,
        context_keys,
    })
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
            return Err(LeoError::internal("tracked context slot is outside the model"));
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
    let mut sums = BTreeMap::<usize, f64>::new();
    for group in groups {
        for update in group {
            if update.index >= target.len() || !update.delta.is_finite() {
                return Err(LeoError::internal("invalid sparse worker parameter delta"));
            }
            *sums.entry(update.index).or_default() += update.delta as f64;
        }
    }
    for (&index, &sum) in &sums {
        target[index] += (sum / worker_count as f64) as f32;
        if !target[index].is_finite() {
            return Err(LeoError::internal("merged parameter became non-finite"));
        }
    }
    Ok(sums.len())
}

fn apply_context_deltas(
    master: &mut Model,
    deltas: &[SparseModelDelta],
    worker_count: usize,
) -> LeoResult<usize> {
    let embedding_dim = master.config.context.embedding_dim;
    let mut merged = BTreeMap::<(usize, u64), (Vec<f64>, u64)>::new();
    for delta in deltas {
        for update in &delta.context {
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
            let entry = merged
                .entry((update.order, update.key))
                .or_insert_with(|| (vec![0.0; embedding_dim], 0));
            for (sum, value) in entry.0.iter_mut().zip(&update.embedding_delta) {
                *sum += *value as f64;
            }
            entry.1 = entry
                .1
                .saturating_add(u64::from(update.observation_increment));
        }
    }

    let mut ordered = merged.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| {
        right
            .1
             .1
            .cmp(&left.1 .1)
            .then_with(|| left.0.cmp(&right.0))
    });
    for ((order, key), (embedding_sum, observations)) in &ordered {
        let (slot, _) = master.resolve_context_slot(*order, *key, true)?;
        let slot = slot.expect("allocating a valid context always returns a slot");
        let range = master.context.embedding_range(slot, embedding_dim);
        for (value, sum) in master.context.embeddings[range]
            .iter_mut()
            .zip(embedding_sum)
        {
            *value += (*sum / worker_count as f64) as f32;
        }
        master.context.observations[slot] = master.context.observations[slot]
            .saturating_add((*observations).min(u64::from(u32::MAX)) as u32);
    }
    Ok(ordered.len())
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

fn project_parameter_constraints(model: &mut Model) {
    let minimum = model.config.learning.weight_min;
    let maximum = model.config.learning.weight_max;
    for threshold in &mut model.neurons.threshold {
        *threshold = threshold.clamp(0.05, 2.0);
    }
    for excitability in &mut model.neurons.excitability {
        *excitability = excitability.clamp(0.1, 4.0);
    }
    let capacity = model.recurrent.capacity_per_neuron;
    for (slot, weight) in model.recurrent.weight.iter_mut().enumerate() {
        let source = slot / capacity;
        *weight = (*weight).clamp(minimum, maximum);
        *weight = if model.neurons.neuron_type[source] == 0 {
            (*weight).max(0.0)
        } else {
            (*weight).min(0.0)
        };
    }
    for weight in &mut model.input.weights {
        *weight = (*weight).clamp(0.0, maximum);
    }
    for weight in &mut model.output.weights {
        *weight = (*weight).clamp(minimum, maximum);
    }
    for weight in &mut model.context.embeddings {
        *weight = (*weight).clamp(minimum, maximum);
    }
    for weight in &mut model.context.output_weights {
        *weight = (*weight).clamp(minimum, maximum);
    }
}

fn config_fingerprint(model: &Model) -> ArtifactDigest {
    digest_bytes(model.config.to_toml().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{apply_mean_deltas, SparseModelDelta};
    use crate::symbols::{BEGIN_DOCUMENT, END_DOCUMENT};
    use crate::{Config, Model, Permission, Runtime};

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
