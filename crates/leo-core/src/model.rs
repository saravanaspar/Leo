//! Persistent model parameters and deterministic sparse initialization.
//!
//! Per-document membrane, activation, branch, fatigue, refractory, eligibility,
//! and delay-ring state are executor-owned (`Runtime` on CPU, device buffers on
//! CUDA); checkpoints contain only parameters meaningful across documents and
//! process restarts.

use crate::metrics::TrainingStatistics;
use crate::rng::SplitMix64;
use crate::symbols::{OUTPUT_CLASSES, SYMBOL_COUNT};
use crate::{Config, LeoError, LeoResult};
use std::collections::HashSet;

pub const BRANCH_EXCITATORY: u8 = 0;
pub const BRANCH_INHIBITORY: u8 = 1;
pub const BRANCH_TEMPORAL: u8 = 2;
pub const BRANCH_CONTEXT_GATE: u8 = 3;

#[derive(Debug, Clone)]
pub struct NeuronParameters {
    pub threshold: Vec<f32>,
    pub excitability: Vec<f32>,
    pub neuron_type: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SynapseArrays {
    pub capacity_per_neuron: usize,
    pub target_neuron: Vec<u32>,
    pub target_branch: Vec<u8>,
    pub delay: Vec<u8>,
    pub weight: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct InputProjection {
    pub fanout: usize,
    pub targets: Vec<u32>,
    pub branches: Vec<u8>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ContextProjection {
    pub keys: Vec<u64>,
    pub embeddings: Vec<f32>,
    pub observations: Vec<u32>,
    /// Shared projection in output-major layout: [output][embedding].
    pub output_weights: Vec<f32>,
}

impl ContextProjection {
    pub fn embedding_range(&self, slot: usize, embedding_dim: usize) -> std::ops::Range<usize> {
        let start = slot * embedding_dim;
        start..start + embedding_dim
    }
}

#[derive(Debug, Clone)]
pub struct OutputProjection {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub config: Config,
    pub generation: u64,
    pub parameter_revision: u64,
    pub neurons: NeuronParameters,
    pub recurrent: SynapseArrays,
    pub input: InputProjection,
    pub output: OutputProjection,
    pub context: ContextProjection,
    pub statistics: TrainingStatistics,
}

impl Model {
    pub fn initialize(config: Config) -> LeoResult<Self> {
        config.validate()?;
        let neuron_count = config.model.neuron_count;
        let capacity = config.model.synapses_per_neuron;
        let synapse_count = neuron_count
            .checked_mul(capacity)
            .ok_or_else(|| LeoError::internal("synapse array size overflow"))?;
        let output_weight_count = OUTPUT_CLASSES
            .checked_mul(neuron_count)
            .ok_or_else(|| LeoError::internal("output array size overflow"))?;
        let context_slots = config
            .context
            .max_order
            .checked_mul(config.context.slots_per_order)
            .ok_or_else(|| LeoError::internal("context slot count overflow"))?;
        let context_embedding_count = context_slots
            .checked_mul(config.context.embedding_dim)
            .ok_or_else(|| LeoError::internal("context embedding count overflow"))?;
        let context_output_count = OUTPUT_CLASSES
            .checked_mul(config.context.embedding_dim)
            .ok_or_else(|| LeoError::internal("context output projection overflow"))?;
        let mut rng = SplitMix64::new(config.model.seed);

        let mut neuron_type = vec![0u8; neuron_count];
        let inhibitory_count =
            ((1.0 - config.model.excitatory_fraction) * neuron_count as f32).round() as usize;
        let mut neuron_ids: Vec<usize> = (0..neuron_count).collect();
        rng.shuffle(&mut neuron_ids);
        for &neuron in neuron_ids.iter().take(inhibitory_count) {
            neuron_type[neuron] = 1;
        }

        let mut neurons = NeuronParameters {
            threshold: vec![0.0; neuron_count],
            excitability: vec![1.0; neuron_count],
            neuron_type,
        };
        for threshold in &mut neurons.threshold {
            *threshold = rng.range_f32(0.45, 0.55);
        }

        let mut recurrent = SynapseArrays {
            capacity_per_neuron: capacity,
            target_neuron: vec![0; synapse_count],
            target_branch: vec![BRANCH_EXCITATORY; synapse_count],
            delay: vec![1; synapse_count],
            weight: vec![0.0; synapse_count],
        };
        initialize_recurrent(&config, &mut rng, &neurons.neuron_type, &mut recurrent)?;

        let input = initialize_input(&config, &mut rng)?;
        let mut output_weights = vec![0.0; output_weight_count];
        for weight in &mut output_weights {
            *weight = rng.range_f32(-0.005, 0.005);
        }
        let mut context_output_weights = vec![0.0; context_output_count];
        for weight in &mut context_output_weights {
            *weight = rng.range_f32(-0.005, 0.005);
        }

        Ok(Self {
            config,
            generation: 0,
            parameter_revision: 0,
            neurons,
            recurrent,
            input,
            output: OutputProjection {
                weights: output_weights,
                bias: vec![0.0; OUTPUT_CLASSES],
            },
            context: ContextProjection {
                keys: vec![0; context_slots],
                embeddings: vec![0.0; context_embedding_count],
                observations: vec![0; context_slots],
                output_weights: context_output_weights,
            },
            statistics: TrainingStatistics::default(),
        })
    }

    pub fn neuron_count(&self) -> usize {
        self.config.model.neuron_count
    }

    pub fn synapse_slot_range(&self, neuron: usize) -> std::ops::Range<usize> {
        let start = neuron * self.recurrent.capacity_per_neuron;
        start..start + self.recurrent.capacity_per_neuron
    }

    /// Resolve an exact context fingerprint inside its fixed-capacity order partition.
    ///
    /// Returns the selected slot and the number of probes performed. Allocation is
    /// deterministic and clears the latent embedding when a weak slot is replaced.
    pub fn resolve_context_slot(
        &mut self,
        order: usize,
        key: u64,
        allocate: bool,
    ) -> LeoResult<(Option<usize>, usize)> {
        if order == 0 || order > self.config.context.max_order || key == 0 {
            return Err(LeoError::internal("invalid context order or fingerprint"));
        }
        let slots_per_order = self.config.context.slots_per_order;
        let probe_limit = self.config.context.probe_limit;
        let base = (order - 1) * slots_per_order;
        let start = key as usize % slots_per_order;
        let mut empty_slot = None;
        let mut weakest_slot = base + start;
        let mut weakest_observations = u32::MAX;
        let mut probes = 0usize;

        for offset in 0..probe_limit {
            let slot = base + (start + offset) % slots_per_order;
            probes = probes.saturating_add(1);
            let existing_key = self.context.keys[slot];
            if existing_key == key {
                return Ok((Some(slot), probes));
            }
            if existing_key == 0 && empty_slot.is_none() {
                empty_slot = Some(slot);
            }
            let observations = self.context.observations[slot];
            if observations < weakest_observations {
                weakest_observations = observations;
                weakest_slot = slot;
            }
        }

        if !allocate {
            return Ok((None, probes));
        }
        let slot = empty_slot.unwrap_or(weakest_slot);
        if self.context.keys[slot] != key {
            self.context.keys[slot] = key;
            self.context.observations[slot] = 0;
            let range = self
                .context
                .embedding_range(slot, self.config.context.embedding_dim);
            self.context.embeddings[range].fill(0.0);
        }
        Ok((Some(slot), probes))
    }

    pub fn validate(&self) -> LeoResult<()> {
        self.config.validate()?;
        let neuron_count = self.config.model.neuron_count;
        let capacity = self.config.model.synapses_per_neuron;
        let synapse_count = neuron_count
            .checked_mul(capacity)
            .ok_or_else(|| LeoError::internal("synapse array size overflow"))?;
        let input_count = SYMBOL_COUNT
            .checked_mul(self.config.model.input_fanout)
            .ok_or_else(|| LeoError::internal("input projection size overflow"))?;
        let output_count = OUTPUT_CLASSES
            .checked_mul(neuron_count)
            .ok_or_else(|| LeoError::internal("output projection size overflow"))?;
        let context_slots = self
            .config
            .context
            .max_order
            .checked_mul(self.config.context.slots_per_order)
            .ok_or_else(|| LeoError::internal("context slot count overflow"))?;
        let context_embedding_count = context_slots
            .checked_mul(self.config.context.embedding_dim)
            .ok_or_else(|| LeoError::internal("context embedding size overflow"))?;
        let context_output_count = OUTPUT_CLASSES
            .checked_mul(self.config.context.embedding_dim)
            .ok_or_else(|| LeoError::internal("context output projection size overflow"))?;

        let neuron_lengths = [
            self.neurons.threshold.len(),
            self.neurons.excitability.len(),
            self.neurons.neuron_type.len(),
        ];
        if neuron_lengths.iter().any(|length| *length != neuron_count)
            || self.recurrent.capacity_per_neuron != capacity
            || self.recurrent.target_neuron.len() != synapse_count
            || self.recurrent.target_branch.len() != synapse_count
            || self.recurrent.delay.len() != synapse_count
            || self.recurrent.weight.len() != synapse_count
            || self.input.fanout != self.config.model.input_fanout
            || self.input.targets.len() != input_count
            || self.input.branches.len() != input_count
            || self.input.weights.len() != input_count
            || self.output.weights.len() != output_count
            || self.output.bias.len() != OUTPUT_CLASSES
            || self.context.keys.len() != context_slots
            || self.context.embeddings.len() != context_embedding_count
            || self.context.observations.len() != context_slots
            || self.context.output_weights.len() != context_output_count
        {
            return Err(LeoError::internal("model tensor dimensions are incompatible"));
        }

        let float_vectors: [&[f32]; 8] = [
            &self.neurons.threshold,
            &self.neurons.excitability,
            &self.recurrent.weight,
            &self.input.weights,
            &self.output.weights,
            &self.output.bias,
            &self.context.embeddings,
            &self.context.output_weights,
        ];
        if float_vectors
            .iter()
            .flat_map(|values| values.iter())
            .any(|value| !value.is_finite())
            || !self.statistics.training_loss_sum.is_finite()
        {
            return Err(LeoError::internal("model contains NaN or infinite values"));
        }
        if self
            .context
            .keys
            .iter()
            .zip(&self.context.observations)
            .any(|(key, observations)| *key == 0 && *observations != 0)
        {
            return Err(LeoError::internal(
                "empty context slots cannot contain observations",
            ));
        }
        if self.neurons.neuron_type.iter().any(|kind| *kind > 1) {
            return Err(LeoError::internal(
                "neuron types must be excitatory or inhibitory",
            ));
        }

        for source in 0..neuron_count {
            for slot in self.synapse_slot_range(source) {
                let target = self.recurrent.target_neuron[slot] as usize;
                let branch = self.recurrent.target_branch[slot];
                let delay = self.recurrent.delay[slot];
                let weight = self.recurrent.weight[slot];
                if target >= neuron_count {
                    return Err(LeoError::internal(format!(
                        "recurrent slot {slot} targets neuron {target} outside the model"
                    )));
                }
                if branch > BRANCH_CONTEXT_GATE || !matches!(delay, 1 | 2 | 4 | 8) {
                    return Err(LeoError::internal(format!(
                        "recurrent slot {slot} has invalid branch or delay"
                    )));
                }
                if weight < self.config.learning.weight_min
                    || weight > self.config.learning.weight_max
                {
                    return Err(LeoError::internal(format!(
                        "recurrent slot {slot} has invalid weight"
                    )));
                }
                if self.neurons.neuron_type[source] == 0 {
                    if weight < 0.0 || branch == BRANCH_INHIBITORY {
                        return Err(LeoError::internal(format!(
                            "excitatory source {source} has incompatible slot {slot}"
                        )));
                    }
                } else if weight > 0.0 || branch != BRANCH_INHIBITORY {
                    return Err(LeoError::internal(format!(
                        "inhibitory source {source} has incompatible slot {slot}"
                    )));
                }
            }
        }

        for slot in 0..input_count {
            if self.input.targets[slot] as usize >= neuron_count
                || self.input.branches[slot] == BRANCH_INHIBITORY
                || self.input.branches[slot] > BRANCH_CONTEXT_GATE
                || self.input.weights[slot] < 0.0
                || self.input.weights[slot] > self.config.learning.weight_max
            {
                return Err(LeoError::internal(format!(
                    "input projection slot {slot} has invalid state"
                )));
            }
        }
        Ok(())
    }
}

fn initialize_recurrent(
    config: &Config,
    rng: &mut SplitMix64,
    neuron_type: &[u8],
    recurrent: &mut SynapseArrays,
) -> LeoResult<()> {
    let neuron_count = config.model.neuron_count;
    let block_size = config.model.neurons_per_block;
    let local_count = (config.model.synapses_per_neuron * 3) / 4;

    for (source, &source_type) in neuron_type.iter().enumerate() {
        let source_block = source / block_size;
        let mut used = HashSet::with_capacity(config.model.synapses_per_neuron * 2);
        for local_slot in 0..config.model.synapses_per_neuron {
            let slot = source * recurrent.capacity_per_neuron + local_slot;
            let prefer_local = local_slot < local_count;
            let mut selected = None;
            for _ in 0..128 {
                let target = if prefer_local {
                    let block_delta = rng.range_usize(3) as isize - 1;
                    let block = (source_block as isize + block_delta)
                        .clamp(0, config.model.block_count as isize - 1)
                        as usize;
                    block * block_size + rng.range_usize(block_size)
                } else {
                    rng.range_usize(neuron_count)
                };
                let branch = if source_type == 0 {
                    match rng.range_usize(3) {
                        0 => BRANCH_EXCITATORY,
                        1 => BRANCH_TEMPORAL,
                        _ => BRANCH_CONTEXT_GATE,
                    }
                } else {
                    BRANCH_INHIBITORY
                };
                if target == source && branch != BRANCH_TEMPORAL {
                    continue;
                }
                if used.insert((target, branch)) {
                    selected = Some((target, branch));
                    break;
                }
            }
            let (target, branch) = selected.ok_or_else(|| {
                LeoError::internal(format!(
                    "could not initialize unique synapse for neuron {source}"
                ))
            })?;
            recurrent.target_neuron[slot] = target as u32;
            recurrent.target_branch[slot] = branch;
            recurrent.delay[slot] = sample_delay(rng);
            recurrent.weight[slot] = if source_type == 0 {
                rng.range_f32(0.01, 0.08)
            } else {
                rng.range_f32(-0.08, -0.01)
            };
        }
    }
    Ok(())
}

fn initialize_input(config: &Config, rng: &mut SplitMix64) -> LeoResult<InputProjection> {
    let fanout = config.model.input_fanout;
    let count = SYMBOL_COUNT
        .checked_mul(fanout)
        .ok_or_else(|| LeoError::internal("input projection size overflow"))?;
    let mut targets = vec![0u32; count];
    let mut branches = vec![BRANCH_EXCITATORY; count];
    let mut weights = vec![0.0; count];

    for symbol in 0..SYMBOL_COUNT {
        let mut used = HashSet::with_capacity(fanout * 2);
        for index in 0..fanout {
            let target = loop {
                let candidate = rng.range_usize(config.model.neuron_count);
                if used.insert(candidate) {
                    break candidate;
                }
            };
            let offset = symbol * fanout + index;
            targets[offset] = target as u32;
            branches[offset] = match index % 8 {
                6 => BRANCH_TEMPORAL,
                7 => BRANCH_CONTEXT_GATE,
                _ => BRANCH_EXCITATORY,
            };
            weights[offset] = rng.range_f32(0.05, 0.20);
        }
    }

    Ok(InputProjection {
        fanout,
        targets,
        branches,
        weights,
    })
}

fn sample_delay(rng: &mut SplitMix64) -> u8 {
    match rng.range_usize(100) {
        0..=49 => 1,
        50..=74 => 2,
        75..=89 => 4,
        _ => 8,
    }
}
