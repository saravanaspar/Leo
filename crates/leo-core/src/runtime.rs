//! Sparse, recurrent, delayed-event runtime with forward-only local learning.

use crate::learning::{
    analog_rank_surrogate, bounded_delta, branch_jacobian, inhibitory_homeostasis_delta,
};
use crate::model::{
    Model, BRANCH_CONTEXT_GATE, BRANCH_EXCITATORY, BRANCH_INHIBITORY, BRANCH_TEMPORAL,
};
use crate::symbols::{
    is_input_symbol, output_index_to_symbol, output_symbol_to_index, END_DOCUMENT_OUTPUT_INDEX,
    OUTPUT_CLASSES,
};
use crate::{LeoError, LeoResult, Permission, StepMetrics};

#[derive(Debug, Clone, Copy)]
struct Event {
    target_neuron: u32,
    target_branch: u8,
    value: f32,
    synapse_slot: u32,
    presynaptic_activation: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct ActivationSelectionMetrics {
    suprathreshold_neurons: usize,
    block_selected_neurons: usize,
    global_cap_clipped_neurons: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct LearningUpdateMetrics {
    eligibility_abs_sum: f64,
    eligibility_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ParameterChanges {
    pub threshold: Vec<usize>,
    pub recurrent_weight: Vec<usize>,
    pub input_weight: Vec<usize>,
    pub output_neurons: Vec<usize>,
    pub context_slots: Vec<usize>,
    pub output_bias_dirty: bool,
    pub context_output_dirty: bool,
}

#[derive(Debug, Clone)]
pub struct WorkerState {
    membrane: Vec<f32>,
    activation: Vec<f32>,
    fatigue: Vec<f32>,
    refractory_until: Vec<u64>,
    branches: Vec<f32>,
}

impl WorkerState {
    fn new(neuron_count: usize, branch_count: usize) -> Self {
        Self {
            membrane: vec![0.0; neuron_count],
            activation: vec![0.0; neuron_count],
            fatigue: vec![0.0; neuron_count],
            refractory_until: vec![0; neuron_count],
            branches: vec![0.0; branch_count],
        }
    }
}
impl LearningUpdateMetrics {
    fn mean_abs_eligibility(self) -> f32 {
        if self.eligibility_count == 0 {
            0.0
        } else {
            (self.eligibility_abs_sum / self.eligibility_count as f64) as f32
        }
    }
}

#[inline]
fn decay_for_elapsed(base: f32, elapsed: u64) -> f32 {
    match elapsed {
        0 => 1.0,
        1 => base,
        _ => base.powi(elapsed.min(i32::MAX as u64) as i32),
    }
}

fn transpose_output_weights(model: &Model) -> Vec<f32> {
    let neuron_count = model.neuron_count();
    let mut transposed = vec![0.0; neuron_count * OUTPUT_CLASSES];
    for output in 0..OUTPUT_CLASSES {
        let model_row = output * neuron_count;
        for neuron in 0..neuron_count {
            transposed[neuron * OUTPUT_CLASSES + output] = model.output.weights[model_row + neuron];
        }
    }
    transposed
}

fn rotated_tie_key(neuron: usize, tick_rotation: usize, neuron_count: usize) -> usize {
    (neuron + neuron_count - tick_rotation) % neuron_count
}

#[inline]
fn context_survives_dropout(seed: u64, tick: u64, symbol: u32, dropout_rate: f32) -> bool {
    if dropout_rate <= 0.0 {
        return true;
    }
    let mut value = seed
        ^ tick.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(symbol).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let draw = (value >> 40) as f32 / (1u32 << 24) as f32;
    draw >= dropout_rate
}

pub struct Runtime {
    model: Model,
    state: WorkerState,
    delay_ring: Vec<Vec<Event>>,
    due_events: Vec<Event>,
    current_tick: u64,
    plasticity_phase: usize,
    touched: Vec<usize>,
    touched_mark: Vec<bool>,
    active: Vec<usize>,
    previous_active: Vec<usize>,
    selected_mark: Vec<bool>,
    selection_surrogate: Vec<f32>,
    branch_last_tick: Vec<u64>,
    neuron_last_tick: Vec<u64>,
    adaptation_state: [Vec<f32>; 3],

    // Recurrent forward eligibility state. All arrays are runtime-only and sparse-indexed.
    recurrent_branch_sensitivity: Vec<f32>,
    recurrent_membrane_sensitivity: Vec<f32>,
    recurrent_fatigue_sensitivity: Vec<f32>,
    recurrent_adaptation_sensitivity: [Vec<f32>; 3],
    eligibility: Vec<f32>,
    eligibility_last_tick: Vec<u64>,
    eligible_recurrent_slots: Vec<usize>,
    eligible_recurrent_mark: Vec<bool>,

    // Input projections use the same forward sensitivity rule.
    input_branch_sensitivity: Vec<f32>,
    input_membrane_sensitivity: Vec<f32>,
    input_fatigue_sensitivity: Vec<f32>,
    input_adaptation_sensitivity: [Vec<f32>; 3],
    input_eligibility: Vec<f32>,
    input_eligibility_last_tick: Vec<u64>,
    eligible_input_slots: Vec<usize>,
    eligible_input_mark: Vec<bool>,

    logits: Vec<f32>,
    probabilities: Vec<f32>,
    errors: Vec<f32>,
    learning_signal: Vec<f32>,
    learning_destinations: Vec<usize>,
    learning_destination_mark: Vec<bool>,
    output_weights_by_neuron: Vec<f32>,
    output_cache_dirty: bool,
    block_candidates: Vec<Vec<(usize, f32)>>,
    block_selection_cutoff: Vec<f32>,
    population_selection_cutoff: f32,
    context_history: Vec<u32>,
    active_context_slots: Vec<usize>,
    active_context_scales: Vec<f32>,
    context_latent: Vec<f32>,
    context_latent_gradient: Vec<f32>,
    active_context_probes: usize,
    context_applied_this_step: bool,
    /// Smoothed rank-cutoff proxy retained for observability only.
    population_inhibition: f32,
    track_parameter_changes: bool,
    changed_threshold: Vec<usize>,
    changed_threshold_mark: Vec<bool>,
    changed_recurrent_weight: Vec<usize>,
    changed_recurrent_weight_mark: Vec<bool>,
    changed_input_weight: Vec<usize>,
    changed_input_weight_mark: Vec<bool>,
    changed_output_neurons: Vec<usize>,
    changed_output_neuron_mark: Vec<bool>,
    changed_context_slots: Vec<usize>,
    changed_context_slot_mark: Vec<bool>,
    output_bias_dirty: bool,
    context_output_dirty: bool,
    persistent_document_activity: bool,
    learning_trace_enabled: bool,
}

impl Runtime {
    pub fn new(model: Model) -> LeoResult<Self> {
        model.validate()?;
        let neuron_count = model.neuron_count();
        let synapse_count = model.recurrent.weight.len();
        let input_count = model.input.weights.len();
        let recurrent_trace_count = synapse_count;
        let input_trace_count = input_count;
        let max_delay = model
            .recurrent
            .delay
            .iter()
            .copied()
            .max()
            .unwrap_or(8)
            .max(8) as usize;
        let ring_size = max_delay + 1;
        let block_count = model.config.model.block_count;
        let neurons_per_block = model.config.model.neurons_per_block;
        let branch_count = neuron_count * model.config.model.branches_per_neuron;
        let output_weights_by_neuron = transpose_output_weights(&model);
        let context_order = model.config.context.max_order;
        let context_embedding_dim = model.config.context.embedding_dim;
        Ok(Self {
            model,
            state: WorkerState::new(neuron_count, branch_count),
            delay_ring: (0..ring_size).map(|_| Vec::new()).collect(),
            due_events: Vec::new(),
            current_tick: 0,
            plasticity_phase: 0,
            touched: Vec::with_capacity(neuron_count.min(4096)),
            touched_mark: vec![false; neuron_count],
            active: Vec::with_capacity(neuron_count.min(4096)),
            previous_active: Vec::with_capacity(neuron_count.min(4096)),
            selected_mark: vec![false; neuron_count],
            selection_surrogate: vec![0.0; neuron_count],
            branch_last_tick: vec![0; neuron_count],
            neuron_last_tick: vec![0; neuron_count],
            adaptation_state: [
                vec![0.0; neuron_count],
                vec![0.0; neuron_count],
                vec![0.0; neuron_count],
            ],
            recurrent_branch_sensitivity: vec![0.0; recurrent_trace_count],
            recurrent_membrane_sensitivity: vec![0.0; recurrent_trace_count],
            recurrent_fatigue_sensitivity: vec![0.0; recurrent_trace_count],
            recurrent_adaptation_sensitivity: [
                vec![0.0; recurrent_trace_count],
                vec![0.0; recurrent_trace_count],
                vec![0.0; recurrent_trace_count],
            ],
            eligibility: vec![0.0; synapse_count],
            eligibility_last_tick: vec![0; synapse_count],
            eligible_recurrent_slots: Vec::with_capacity(recurrent_trace_count.min(4096)),
            eligible_recurrent_mark: vec![false; recurrent_trace_count],
            input_branch_sensitivity: vec![0.0; input_trace_count],
            input_membrane_sensitivity: vec![0.0; input_trace_count],
            input_fatigue_sensitivity: vec![0.0; input_trace_count],
            input_adaptation_sensitivity: [
                vec![0.0; input_trace_count],
                vec![0.0; input_trace_count],
                vec![0.0; input_trace_count],
            ],
            input_eligibility: vec![0.0; input_trace_count],
            input_eligibility_last_tick: vec![0; input_trace_count],
            eligible_input_slots: Vec::with_capacity(input_trace_count.min(4096)),
            eligible_input_mark: vec![false; input_trace_count],
            logits: vec![0.0; OUTPUT_CLASSES],
            probabilities: vec![0.0; OUTPUT_CLASSES],
            errors: vec![0.0; OUTPUT_CLASSES],
            learning_signal: vec![0.0; neuron_count],
            learning_destinations: Vec::with_capacity(neuron_count.min(4096)),
            learning_destination_mark: vec![false; neuron_count],
            output_weights_by_neuron,
            output_cache_dirty: false,
            block_candidates: (0..block_count)
                .map(|_| Vec::with_capacity(neurons_per_block))
                .collect(),
            block_selection_cutoff: vec![0.0; block_count],
            population_selection_cutoff: 0.0,
            context_history: Vec::with_capacity(context_order),
            active_context_slots: Vec::with_capacity(context_order),
            active_context_scales: Vec::with_capacity(context_order),
            context_latent: vec![0.0; context_embedding_dim],
            context_latent_gradient: vec![0.0; context_embedding_dim],
            active_context_probes: 0,
            context_applied_this_step: false,
            population_inhibition: 0.0,
            track_parameter_changes: false,
            changed_threshold: Vec::new(),
            changed_threshold_mark: Vec::new(),
            changed_recurrent_weight: Vec::new(),
            changed_recurrent_weight_mark: Vec::new(),
            changed_input_weight: Vec::new(),
            changed_input_weight_mark: Vec::new(),
            changed_output_neurons: Vec::new(),
            changed_output_neuron_mark: Vec::new(),
            changed_context_slots: Vec::new(),
            changed_context_slot_mark: Vec::new(),
            output_bias_dirty: false,
            context_output_dirty: false,
            persistent_document_activity: false,
            learning_trace_enabled: false,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut Model {
        // Callers may replace or mutate the serialized output matrix (for example when
        // restoring a best checkpoint). Rebuild the runtime-local neuron-major cache
        // before the next step.
        self.output_cache_dirty = true;
        &mut self.model
    }

    pub fn enable_parameter_tracking(&mut self) {
        if self.track_parameter_changes {
            return;
        }
        self.track_parameter_changes = true;
        self.changed_threshold_mark = vec![false; self.model.neuron_count()];
        self.changed_recurrent_weight_mark = vec![false; self.model.recurrent.weight.len()];
        self.changed_input_weight_mark = vec![false; self.model.input.weights.len()];
        self.changed_output_neuron_mark = vec![false; self.model.neuron_count()];
        self.changed_context_slot_mark = vec![false; self.model.context.keys.len()];
    }

    pub fn parameter_changes(&self) -> ParameterChanges {
        ParameterChanges {
            threshold: self.changed_threshold.clone(),
            recurrent_weight: self.changed_recurrent_weight.clone(),
            input_weight: self.changed_input_weight.clone(),
            output_neurons: self.changed_output_neurons.clone(),
            context_slots: self.changed_context_slots.clone(),
            output_bias_dirty: self.output_bias_dirty,
            context_output_dirty: self.context_output_dirty,
        }
    }

    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }

    pub fn active_neurons(&self) -> &[usize] {
        &self.active
    }

    pub fn activation(&self, neuron: usize) -> f32 {
        self.state.activation.get(neuron).copied().unwrap_or(0.0)
    }

    fn ensure_output_weight_cache(&mut self) {
        if self.output_cache_dirty {
            self.output_weights_by_neuron = transpose_output_weights(&self.model);
            self.output_cache_dirty = false;
        }
    }

    pub fn begin_document(&mut self) {
        self.reset_transient_state();
    }

    pub fn finish_document(&mut self) {
        if self.persistent_document_activity {
            self.model.statistics.processed_stories =
                self.model.statistics.processed_stories.saturating_add(1);
        }
        self.reset_transient_state();
    }

    pub fn reset_transient_state(&mut self) {
        self.state.membrane.fill(0.0);
        self.state.activation.fill(0.0);
        self.state.fatigue.fill(0.0);
        self.state.refractory_until.fill(0);
        self.state.branches.fill(0.0);
        for state in &mut self.adaptation_state {
            state.fill(0.0);
        }
        for bucket in &mut self.delay_ring {
            bucket.clear();
        }
        self.due_events.clear();

        self.recurrent_branch_sensitivity.fill(0.0);
        self.recurrent_membrane_sensitivity.fill(0.0);
        self.recurrent_fatigue_sensitivity.fill(0.0);
        for state in &mut self.recurrent_adaptation_sensitivity {
            state.fill(0.0);
        }
        self.eligibility.fill(0.0);
        self.eligibility_last_tick.fill(self.current_tick);
        self.eligible_recurrent_mark.fill(false);
        self.eligible_recurrent_slots.clear();

        self.input_branch_sensitivity.fill(0.0);
        self.input_membrane_sensitivity.fill(0.0);
        self.input_fatigue_sensitivity.fill(0.0);
        for state in &mut self.input_adaptation_sensitivity {
            state.fill(0.0);
        }
        self.input_eligibility.fill(0.0);
        self.input_eligibility_last_tick.fill(self.current_tick);
        self.eligible_input_mark.fill(false);
        self.eligible_input_slots.clear();

        self.branch_last_tick.fill(self.current_tick);
        self.neuron_last_tick.fill(self.current_tick);
        self.touched_mark.fill(false);
        self.selected_mark.fill(false);
        self.selection_surrogate.fill(0.0);
        self.touched.clear();
        self.active.clear();
        self.previous_active.clear();
        for block in &mut self.block_candidates {
            block.clear();
        }
        self.block_selection_cutoff.fill(0.0);
        self.population_selection_cutoff = 0.0;
        self.context_history.clear();
        self.active_context_slots.clear();
        self.active_context_scales.clear();
        self.context_latent.fill(0.0);
        self.context_latent_gradient.fill(0.0);
        self.active_context_probes = 0;
        self.context_applied_this_step = false;
        self.population_inhibition = 0.0;
        self.persistent_document_activity = false;
        self.learning_trace_enabled = false;
        self.plasticity_phase = 0;
    }

    pub fn step(
        &mut self,
        symbol: u32,
        target: Option<u32>,
        permission: Permission,
    ) -> LeoResult<StepMetrics> {
        if !is_input_symbol(symbol) {
            return Err(LeoError::internal(format!(
                "invalid input symbol: {symbol}"
            )));
        }
        let target_index = target
            .map(|value| {
                output_symbol_to_index(value)
                    .ok_or_else(|| LeoError::internal(format!("invalid output target: {value}")))
            })
            .transpose()?;
        self.ensure_output_weight_cache();
        self.learning_trace_enabled = !matches!(permission, Permission::Frozen);

        self.start_tick();
        self.deliver_due_events();
        self.inject_symbol(symbol);
        let context_enabled = matches!(permission, Permission::Frozen)
            || context_survives_dropout(
                self.model.config.model.seed,
                self.current_tick,
                symbol,
                self.model.config.context.dropout_rate,
            );
        self.activate_context(
            symbol,
            !matches!(permission, Permission::Frozen) && context_enabled,
            context_enabled,
        );
        let selection = self.select_active_neurons();
        if self.learning_trace_enabled {
            self.update_forward_eligibilities();
        }
        self.apply_post_activation_dynamics();
        let emitted_events = self.emit_recurrent_events();
        let neural_loss = self.compute_output_probabilities(target_index);
        let predicted_symbol = output_index_to_symbol(self.argmax_probability())
            .expect("argmax index must be a valid output class");
        let loss = target_index.map(|index| -self.probabilities[index].max(1.0e-12).ln());
        let context_loss_gain = match (neural_loss, loss) {
            (Some(neural), Some(combined)) => neural - combined,
            _ => 0.0,
        };

        let mut learning_metrics = LearningUpdateMetrics::default();
        if let Some(target_output_index) = target_index {
            let strength = permission.strength(&self.model.config);
            let target_weight = if target_output_index == END_DOCUMENT_OUTPUT_INDEX {
                self.model.config.learning.end_document_weight
            } else {
                1.0
            };
            if strength > 0.0 {
                learning_metrics =
                    self.apply_learning(target_output_index, strength, target_weight)?;
                self.model.parameter_revision = self.model.parameter_revision.saturating_add(1);
            }
        }

        let active_neurons = self.active.len();
        if !matches!(permission, Permission::Frozen) {
            self.persistent_document_activity = true;
            self.update_homeostasis(active_neurons);
            self.update_statistics(symbol, loss, emitted_events);
        }
        self.finish_tick();

        let output_madds = active_neurons
            .saturating_mul(OUTPUT_CLASSES)
            .saturating_add(
                self.active_context_slots
                    .len()
                    .saturating_mul(self.model.config.context.embedding_dim),
            )
            .saturating_add(if self.active_context_slots.is_empty() {
                0
            } else {
                self.model
                    .config
                    .context
                    .embedding_dim
                    .saturating_mul(OUTPUT_CLASSES)
            });

        Ok(StepMetrics {
            loss,
            predicted_symbol,
            active_neurons,
            emitted_events,
            suprathreshold_neurons: selection.suprathreshold_neurons,
            block_selected_neurons: selection.block_selected_neurons,
            global_cap_clipped_neurons: selection.global_cap_clipped_neurons,
            population_inhibition: self.population_inhibition,
            context_cells: self.active_context_slots.len(),
            context_probes: self.active_context_probes,
            context_applied: self.context_applied_this_step,
            context_loss_gain,
            output_madds,
            eligible_recurrent_synapses: self.eligible_recurrent_slots.len(),
            eligible_input_synapses: self.eligible_input_slots.len(),
            mean_abs_eligibility: learning_metrics.mean_abs_eligibility(),
        })
    }

    fn start_tick(&mut self) {
        for &neuron in &self.touched {
            self.touched_mark[neuron] = false;
            self.selection_surrogate[neuron] = 0.0;
        }
        self.touched.clear();

        std::mem::swap(&mut self.active, &mut self.previous_active);
        self.active.clear();
        for &neuron in &self.previous_active {
            self.state.activation[neuron] = 0.0;
            self.selected_mark[neuron] = false;
        }
        let mut previous_index = 0usize;
        while previous_index < self.previous_active.len() {
            let neuron = self.previous_active[previous_index];
            self.mark_touched(neuron);
            previous_index += 1;
        }
        for block in &mut self.block_candidates {
            block.clear();
        }
        self.block_selection_cutoff.fill(0.0);
        self.population_selection_cutoff = 0.0;
    }

    fn finish_tick(&mut self) {
        self.current_tick = self.current_tick.saturating_add(1);
    }

    fn mark_touched(&mut self, neuron: usize) {
        if self.touched_mark[neuron] {
            return;
        }
        self.bring_neuron_to_current_tick(neuron);
        self.touched_mark[neuron] = true;
        self.touched.push(neuron);
    }

    fn bring_neuron_to_current_tick(&mut self, neuron: usize) {
        let branch_elapsed = self
            .current_tick
            .saturating_sub(self.branch_last_tick[neuron]);
        if branch_elapsed > 0 {
            let decay = decay_for_elapsed(self.model.config.dynamics.branch_decay, branch_elapsed);
            let base = neuron * 4;
            for branch in 0..4 {
                self.state.branches[base + branch] *= decay;
            }
            self.branch_last_tick[neuron] = self.current_tick;
        }

        let neuron_elapsed = self
            .current_tick
            .saturating_sub(self.neuron_last_tick[neuron]);
        if neuron_elapsed > 0 {
            self.state.membrane[neuron] *=
                decay_for_elapsed(self.model.config.dynamics.membrane_decay, neuron_elapsed);
            self.state.fatigue[neuron] *=
                decay_for_elapsed(self.model.config.dynamics.fatigue_decay, neuron_elapsed);
            let decays = self.model.config.dynamics.adaptation_decays();
            for (state, decay) in self.adaptation_state.iter_mut().zip(decays) {
                state[neuron] *= decay_for_elapsed(decay, neuron_elapsed);
            }
            self.neuron_last_tick[neuron] = self.current_tick;
        }
    }

    fn deliver_due_events(&mut self) {
        let bucket = (self.current_tick as usize) % self.delay_ring.len();
        std::mem::swap(&mut self.due_events, &mut self.delay_ring[bucket]);
        let mut event_index = 0usize;
        while event_index < self.due_events.len() {
            let event = self.due_events[event_index];
            if self.learning_trace_enabled {
                self.activate_recurrent_trace(
                    event.synapse_slot as usize,
                    event.presynaptic_activation,
                );
            }
            self.add_to_branch(
                event.target_neuron as usize,
                event.target_branch,
                event.value,
            );
            event_index += 1;
        }
        self.due_events.clear();
    }

    fn inject_symbol(&mut self, symbol: u32) {
        let fanout = self.model.input.fanout;
        let start = symbol as usize * fanout;
        for offset in start..start + fanout {
            let target = self.model.input.targets[offset] as usize;
            let branch = self.model.input.branches[offset];
            let value = self.model.input.weights[offset];
            if self.learning_trace_enabled {
                self.activate_input_trace(offset, 1.0);
            }
            self.add_to_branch(target, branch, value);
        }
    }

    fn activate_context(&mut self, symbol: u32, allocate: bool, enabled: bool) {
        let max_order = self.model.config.context.max_order;
        if self.context_history.len() == max_order {
            self.context_history.remove(0);
        }
        self.context_history.push(symbol);
        self.active_context_slots.clear();
        self.active_context_scales.clear();
        self.context_latent.fill(0.0);
        self.context_latent_gradient.fill(0.0);
        self.active_context_probes = 0;
        self.context_applied_this_step = enabled;
        if !enabled {
            return;
        }

        let available = self.context_history.len().min(max_order);
        let confidence_target = self.model.config.context.confidence_observations as f32;
        let mut scale_sum = 0.0f32;
        for order in 1..=available {
            let key = self.context_key(order);
            let Some(slot) = self.resolve_context_slot(order, key, allocate) else {
                continue;
            };
            let observations = self.model.context.observations[slot] as f32;
            let confidence = observations / (observations + confidence_target);
            let scale = confidence * order as f32;
            self.active_context_slots.push(slot);
            self.active_context_scales.push(scale);
            scale_sum += scale;
        }
        if scale_sum > 0.0 {
            for scale in &mut self.active_context_scales {
                *scale /= scale_sum;
            }
        }
    }

    fn context_key(&self, order: usize) -> u64 {
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = 0xcbf29ce484222325u64 ^ order as u64;
        let start = self.context_history.len() - order;
        for symbol in &self.context_history[start..] {
            for byte in symbol.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash.max(1)
    }

    fn resolve_context_slot(&mut self, order: usize, key: u64, allocate: bool) -> Option<usize> {
        let (slot, probes) = self
            .model
            .resolve_context_slot(order, key, allocate)
            .expect("runtime context order and fingerprint are valid");
        self.active_context_probes = self.active_context_probes.saturating_add(probes);
        slot
    }

    fn add_to_branch(&mut self, neuron: usize, branch: u8, value: f32) {
        if neuron >= self.model.neuron_count() || branch > BRANCH_CONTEXT_GATE {
            return;
        }
        self.mark_touched(neuron);
        let index = neuron * 4 + branch as usize;
        self.state.branches[index] += value;
    }

    fn activate_recurrent_trace(&mut self, slot: usize, presynaptic_activation: f32) {
        self.advance_recurrent_trace(slot);
        self.recurrent_branch_sensitivity[slot] += presynaptic_activation;
        if !self.eligible_recurrent_mark[slot] {
            self.eligible_recurrent_mark[slot] = true;
            self.eligible_recurrent_slots.push(slot);
        }
    }

    fn activate_input_trace(&mut self, slot: usize, symbol_activation: f32) {
        self.advance_input_trace(slot);
        self.input_branch_sensitivity[slot] += symbol_activation;
        if !self.eligible_input_mark[slot] {
            self.eligible_input_mark[slot] = true;
            self.eligible_input_slots.push(slot);
        }
    }

    fn advance_recurrent_trace(&mut self, slot: usize) {
        let elapsed = self
            .current_tick
            .saturating_sub(self.eligibility_last_tick[slot]);
        if elapsed == 0 {
            return;
        }
        self.recurrent_branch_sensitivity[slot] *=
            decay_for_elapsed(self.model.config.dynamics.branch_decay, elapsed);
        self.recurrent_membrane_sensitivity[slot] *=
            decay_for_elapsed(self.model.config.dynamics.membrane_decay, elapsed);
        self.recurrent_fatigue_sensitivity[slot] *=
            decay_for_elapsed(self.model.config.dynamics.fatigue_decay, elapsed);
        let decays = self.model.config.dynamics.adaptation_decays();
        for (state, decay) in self.recurrent_adaptation_sensitivity.iter_mut().zip(decays) {
            state[slot] *= decay_for_elapsed(decay, elapsed);
        }
        self.eligibility_last_tick[slot] = self.current_tick;
    }

    fn advance_input_trace(&mut self, slot: usize) {
        let elapsed = self
            .current_tick
            .saturating_sub(self.input_eligibility_last_tick[slot]);
        if elapsed == 0 {
            return;
        }
        self.input_branch_sensitivity[slot] *=
            decay_for_elapsed(self.model.config.dynamics.branch_decay, elapsed);
        self.input_membrane_sensitivity[slot] *=
            decay_for_elapsed(self.model.config.dynamics.membrane_decay, elapsed);
        self.input_fatigue_sensitivity[slot] *=
            decay_for_elapsed(self.model.config.dynamics.fatigue_decay, elapsed);
        let decays = self.model.config.dynamics.adaptation_decays();
        for (state, decay) in self.input_adaptation_sensitivity.iter_mut().zip(decays) {
            state[slot] *= decay_for_elapsed(decay, elapsed);
        }
        self.input_eligibility_last_tick[slot] = self.current_tick;
    }

    fn destination_branch_jacobian(&self, destination: usize, branch: u8) -> f32 {
        let base = destination * 4;
        branch_jacobian(
            branch,
            self.model.neurons.excitability[destination],
            self.state.branches[base + BRANCH_TEMPORAL as usize],
            self.state.branches[base + BRANCH_CONTEXT_GATE as usize],
        )
    }

    fn destination_surrogate(&self, destination: usize) -> f32 {
        self.selection_surrogate[destination]
    }

    #[inline]
    fn neuron_is_refractory(&self, neuron: usize) -> bool {
        self.state.refractory_until[neuron] > self.current_tick
    }

    fn cache_selection_surrogates(&mut self) {
        let block_size = self.model.config.model.neurons_per_block;
        let width = self.model.config.learning.surrogate_width;
        let gain = self.model.config.learning.surrogate_gain;
        for &destination in &self.touched {
            if self.neuron_is_refractory(destination) {
                self.selection_surrogate[destination] = 0.0;
                continue;
            }
            let margin =
                self.state.membrane[destination] - self.model.neurons.threshold[destination];
            let block = destination / block_size;
            let cutoff = self.block_selection_cutoff[block].max(self.population_selection_cutoff);
            self.selection_surrogate[destination] =
                analog_rank_surrogate(margin, cutoff, self.selected_mark[destination], width, gain);
        }
    }

    fn update_forward_eligibilities(&mut self) {
        let epsilon = self.model.config.learning.eligibility_epsilon;
        let mut index = 0usize;
        while index < self.eligible_recurrent_slots.len() {
            let slot = self.eligible_recurrent_slots[index];
            let keep = self.update_recurrent_forward_slot(slot, epsilon);
            if keep {
                index += 1;
            } else {
                self.eligible_recurrent_mark[slot] = false;
                self.eligible_recurrent_slots.swap_remove(index);
            }
        }

        index = 0;
        while index < self.eligible_input_slots.len() {
            let slot = self.eligible_input_slots[index];
            let keep = self.update_input_forward_slot(slot, epsilon);
            if keep {
                index += 1;
            } else {
                self.eligible_input_mark[slot] = false;
                self.eligible_input_slots.swap_remove(index);
            }
        }
    }

    fn update_recurrent_forward_slot(&mut self, slot: usize, epsilon: f32) -> bool {
        self.advance_recurrent_trace(slot);
        let destination = self.model.recurrent.target_neuron[slot] as usize;
        let branch = self.model.recurrent.target_branch[slot];
        let eligibility =
            if !self.touched_mark[destination] || self.neuron_is_refractory(destination) {
                self.eligibility[slot] * self.model.config.learning.eligibility_decay
            } else {
                let jacobian = self.destination_branch_jacobian(destination, branch);
                let surrogate = self.destination_surrogate(destination);
                let adaptation_sum = self
                    .recurrent_adaptation_sensitivity
                    .iter()
                    .map(|state| state[slot])
                    .sum::<f32>();
                let membrane_sensitivity = self.recurrent_membrane_sensitivity[slot]
                    + jacobian * self.recurrent_branch_sensitivity[slot]
                    - self.recurrent_fatigue_sensitivity[slot]
                    - adaptation_sum;
                self.recurrent_membrane_sensitivity[slot] = membrane_sensitivity;
                surrogate * membrane_sensitivity
            };
        self.eligibility[slot] = eligibility;

        if self.touched_mark[destination] && self.selected_mark[destination] {
            self.recurrent_fatigue_sensitivity[slot] +=
                self.model.config.dynamics.fatigue_gain * eligibility;
            let gains = self.model.config.dynamics.adaptation_gains();
            for (state, gain) in self.recurrent_adaptation_sensitivity.iter_mut().zip(gains) {
                state[slot] += gain * eligibility;
            }
        }

        let mut magnitude = self.recurrent_branch_sensitivity[slot]
            .abs()
            .max(self.recurrent_membrane_sensitivity[slot].abs())
            .max(self.recurrent_fatigue_sensitivity[slot].abs())
            .max(eligibility.abs());
        for state in &self.recurrent_adaptation_sensitivity {
            magnitude = magnitude.max(state[slot].abs());
        }
        magnitude > epsilon
    }

    fn update_input_forward_slot(&mut self, slot: usize, epsilon: f32) -> bool {
        self.advance_input_trace(slot);
        let destination = self.model.input.targets[slot] as usize;
        let branch = self.model.input.branches[slot];
        let eligibility =
            if !self.touched_mark[destination] || self.neuron_is_refractory(destination) {
                self.input_eligibility[slot] * self.model.config.learning.eligibility_decay
            } else {
                let jacobian = self.destination_branch_jacobian(destination, branch);
                let surrogate = self.destination_surrogate(destination);
                let adaptation_sum = self
                    .input_adaptation_sensitivity
                    .iter()
                    .map(|state| state[slot])
                    .sum::<f32>();
                let membrane_sensitivity = self.input_membrane_sensitivity[slot]
                    + jacobian * self.input_branch_sensitivity[slot]
                    - self.input_fatigue_sensitivity[slot]
                    - adaptation_sum;
                self.input_membrane_sensitivity[slot] = membrane_sensitivity;
                surrogate * membrane_sensitivity
            };
        self.input_eligibility[slot] = eligibility;

        if self.touched_mark[destination] && self.selected_mark[destination] {
            self.input_fatigue_sensitivity[slot] +=
                self.model.config.dynamics.fatigue_gain * eligibility;
            let gains = self.model.config.dynamics.adaptation_gains();
            for (state, gain) in self.input_adaptation_sensitivity.iter_mut().zip(gains) {
                state[slot] += gain * eligibility;
            }
        }

        let mut magnitude = self.input_branch_sensitivity[slot]
            .abs()
            .max(self.input_membrane_sensitivity[slot].abs())
            .max(self.input_fatigue_sensitivity[slot].abs())
            .max(eligibility.abs());
        for state in &self.input_adaptation_sensitivity {
            magnitude = magnitude.max(state[slot].abs());
        }
        magnitude > epsilon
    }

    fn apply_post_activation_dynamics(&mut self) {
        let gains = self.model.config.dynamics.adaptation_gains();
        let mut active_index = 0usize;
        while active_index < self.active.len() {
            let neuron = self.active[active_index];
            let activation = self.state.activation[neuron];
            self.state.fatigue[neuron] += self.model.config.dynamics.fatigue_gain * activation;
            for (state, gain) in self.adaptation_state.iter_mut().zip(gains) {
                state[neuron] += gain * activation;
            }
            self.state.membrane[neuron] -= self.model.neurons.threshold[neuron]
                * self.model.config.dynamics.membrane_reset_fraction;
            self.state.refractory_until[neuron] = self
                .current_tick
                .saturating_add(self.model.config.dynamics.refractory_ticks)
                .saturating_add(1);
            active_index += 1;
        }
    }

    fn select_active_neurons(&mut self) -> ActivationSelectionMetrics {
        let block_size = self.model.config.model.neurons_per_block;
        let neuron_count = self.model.neuron_count().max(1);
        let tick_rotation = (self.current_tick as usize) % neuron_count;
        let mut touched_index = 0usize;
        while touched_index < self.touched.len() {
            let neuron = self.touched[touched_index];
            if self.neuron_is_refractory(neuron) {
                touched_index += 1;
                continue;
            }
            let base = neuron * 4;
            let gate = self.state.branches[base + BRANCH_CONTEXT_GATE as usize].clamp(0.0, 1.0);
            let inhibitory = self.state.branches[base + BRANCH_INHIBITORY as usize];
            let net_evidence = self.state.branches[base + BRANCH_EXCITATORY as usize]
                + self.state.branches[base + BRANCH_TEMPORAL as usize] * gate
                + inhibitory;
            let adaptation = self
                .adaptation_state
                .iter()
                .map(|state| state[neuron])
                .sum::<f32>();
            // Excitability is a gain on evidence, not a tonic bias.
            let drive = self.model.neurons.excitability[neuron] * net_evidence
                - self.state.fatigue[neuron]
                - adaptation
                - self.population_inhibition;
            self.state.membrane[neuron] += drive;
            let activation = (self.state.membrane[neuron] - self.model.neurons.threshold[neuron])
                .clamp(0.0, 1.0);
            if activation > 0.0 {
                let block = neuron / block_size;
                self.block_candidates[block].push((neuron, activation));
            }
            touched_index += 1;
        }

        let suprathreshold_neurons = self
            .block_candidates
            .iter()
            .map(|block| block.len())
            .sum::<usize>();

        let block_limit = self.model.config.model.max_active_per_block;
        for block_index in 0..self.block_candidates.len() {
            let block = &mut self.block_candidates[block_index];
            if block.len() > block_limit {
                block.select_nth_unstable_by(block_limit, |left, right| {
                    right.1.total_cmp(&left.1).then_with(|| {
                        rotated_tie_key(left.0, tick_rotation, neuron_count).cmp(&rotated_tie_key(
                            right.0,
                            tick_rotation,
                            neuron_count,
                        ))
                    })
                });
                self.block_selection_cutoff[block_index] = block[block_limit].1;
                block.truncate(block_limit);
            }
            // Formula v2 keeps the exact local competition order even when the
            // global cap is provably non-binding and global ranking is skipped.
            block.sort_unstable_by(|left, right| {
                right.1.total_cmp(&left.1).then_with(|| {
                    rotated_tie_key(left.0, tick_rotation, neuron_count).cmp(&rotated_tie_key(
                        right.0,
                        tick_rotation,
                        neuron_count,
                    ))
                })
            });
            self.active.extend(block.iter().map(|(neuron, _)| *neuron));
        }

        let block_selected_neurons = self.active.len();
        let global_cap = self.model.config.model.max_active_global;
        let block_capacity = self
            .model
            .config
            .model
            .block_count
            .saturating_mul(block_limit);
        let redundant_global_topk = global_cap >= block_capacity;
        let global_cap_clipped_neurons = if redundant_global_topk {
            // No set member can be removed when the sum of all local winner
            // capacities is already <= the global cap. Preserve the same winner
            // set and deterministic block/local-rank order without an O(K log K)
            // global ranking pass.
            self.population_selection_cutoff = 0.0;
            0
        } else {
            self.active.sort_unstable_by(|left, right| {
                let right_activation = (self.state.membrane[*right]
                    - self.model.neurons.threshold[*right])
                    .clamp(0.0, 1.0);
                let left_activation = (self.state.membrane[*left]
                    - self.model.neurons.threshold[*left])
                    .clamp(0.0, 1.0);
                right_activation.total_cmp(&left_activation).then_with(|| {
                    rotated_tie_key(*left, tick_rotation, neuron_count).cmp(&rotated_tie_key(
                        *right,
                        tick_rotation,
                        neuron_count,
                    ))
                })
            });
            self.population_selection_cutoff = if self.active.len() > global_cap {
                let cutoff_neuron = self.active[global_cap];
                (self.state.membrane[cutoff_neuron] - self.model.neurons.threshold[cutoff_neuron])
                    .clamp(0.0, self.model.config.dynamics.population_inhibition_max)
            } else {
                0.0
            };
            let clipped = self.active.len().saturating_sub(global_cap);
            if clipped > 0 {
                self.active.truncate(global_cap);
            }
            clipped
        };
        self.update_population_inhibition(self.active.len());

        for &neuron in &self.active {
            let margin = self.state.membrane[neuron] - self.model.neurons.threshold[neuron];
            let activation = margin.clamp(0.0, 1.0);
            self.state.activation[neuron] = activation;
            self.selected_mark[neuron] = true;
        }

        self.cache_selection_surrogates();

        ActivationSelectionMetrics {
            suprathreshold_neurons,
            block_selected_neurons,
            global_cap_clipped_neurons,
        }
    }

    fn update_population_inhibition(&mut self, active_neurons: usize) {
        let activity = active_neurons as f32 / self.model.neuron_count().max(1) as f32;
        let error = activity - self.model.config.dynamics.target_activity;
        let rate = self.model.config.dynamics.population_inhibition_rate;
        self.population_inhibition = (self.population_inhibition + rate * error)
            .clamp(0.0, self.model.config.dynamics.population_inhibition_max);
    }

    fn emit_recurrent_events(&mut self) -> usize {
        let mut emitted = 0usize;
        let mut active_index = 0usize;
        while active_index < self.active.len() {
            let source = self.active[active_index];
            let activation = self.state.activation[source];
            let range = self.model.synapse_slot_range(source);
            for slot in range {
                let delay = self.model.recurrent.delay[slot].max(1) as usize;
                let bucket = (self.current_tick as usize + delay) % self.delay_ring.len();
                self.delay_ring[bucket].push(Event {
                    target_neuron: self.model.recurrent.target_neuron[slot],
                    target_branch: self.model.recurrent.target_branch[slot],
                    value: activation * self.model.recurrent.weight[slot],
                    synapse_slot: slot as u32,
                    presynaptic_activation: activation,
                });
                emitted += 1;
            }
            active_index += 1;
        }
        emitted
    }

    fn compute_output_probabilities(&mut self, target_index: Option<usize>) -> Option<f32> {
        self.logits.copy_from_slice(&self.model.output.bias);
        for &neuron in &self.active {
            let activation = self.state.activation[neuron];
            let cache_row = neuron * OUTPUT_CLASSES;
            for output in 0..OUTPUT_CLASSES {
                self.logits[output] +=
                    self.output_weights_by_neuron[cache_row + output] * activation;
            }
        }

        let neural_loss = target_index.map(|target| self.loss_from_logits(target));
        self.context_latent.fill(0.0);
        let embedding_dim = self.model.config.context.embedding_dim;
        if !self.active_context_slots.is_empty() {
            for (&slot, &scale) in self
                .active_context_slots
                .iter()
                .zip(self.active_context_scales.iter())
            {
                if scale <= 0.0 {
                    continue;
                }
                let row = slot * embedding_dim;
                for dimension in 0..embedding_dim {
                    self.context_latent[dimension] +=
                        self.model.context.embeddings[row + dimension] * scale;
                }
            }
            for output in 0..OUTPUT_CLASSES {
                let row = output * embedding_dim;
                let mut contribution = 0.0f32;
                for dimension in 0..embedding_dim {
                    contribution += self.model.context.output_weights[row + dimension]
                        * self.context_latent[dimension];
                }
                self.logits[output] += contribution;
            }
        }
        self.normalize_probabilities();
        neural_loss
    }

    fn loss_from_logits(&self, target: usize) -> f32 {
        let maximum = self
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum = self
            .logits
            .iter()
            .map(|logit| (*logit - maximum).exp())
            .sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            (OUTPUT_CLASSES as f32).ln()
        } else {
            maximum + sum.ln() - self.logits[target]
        }
    }

    fn normalize_probabilities(&mut self) {
        let maximum = self
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for output in 0..OUTPUT_CLASSES {
            let value = (self.logits[output] - maximum).exp();
            self.probabilities[output] = value;
            sum += value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            self.probabilities.fill(1.0 / OUTPUT_CLASSES as f32);
        } else {
            for probability in &mut self.probabilities {
                *probability /= sum;
            }
        }
    }

    fn argmax_probability(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn apply_learning(
        &mut self,
        target_output_index: usize,
        strength: f32,
        target_weight: f32,
    ) -> LeoResult<LearningUpdateMetrics> {
        self.errors.copy_from_slice(&self.probabilities);
        self.errors[target_output_index] -= 1.0;
        let supervised_strength = strength * target_weight;
        let (plasticity_scale, next_plasticity_phase) = self.model.config.learning.plasticity_step(
            self.plasticity_phase,
            target_output_index == END_DOCUMENT_OUTPUT_INDEX,
        );
        let neuron_count = self.model.neuron_count();

        // Destination membership is tick-local. Clear prior marks every
        // supervised step even when recurrent/input plasticity is deferred.
        for &neuron in &self.learning_destinations {
            self.learning_signal[neuron] = 0.0;
            self.learning_destination_mark[neuron] = false;
        }
        self.learning_destinations.clear();

        if plasticity_scale > 0.0 {
            let mut index = 0usize;
            while index < self.eligible_recurrent_slots.len() {
                let slot = self.eligible_recurrent_slots[index];
                let destination = self.model.recurrent.target_neuron[slot] as usize;
                self.add_learning_destination(destination);
                index += 1;
            }
            index = 0;
            while index < self.eligible_input_slots.len() {
                let slot = self.eligible_input_slots[index];
                let destination = self.model.input.targets[slot] as usize;
                self.add_learning_destination(destination);
                index += 1;
            }
            index = 0;
            while index < self.active.len() {
                let destination = self.active[index];
                self.add_learning_destination(destination);
                index += 1;
            }

            for &neuron in &self.learning_destinations {
                let cache_row = neuron * OUTPUT_CLASSES;
                let mut signal = 0.0f32;
                for output in 0..OUTPUT_CLASSES {
                    signal +=
                        self.output_weights_by_neuron[cache_row + output] * self.errors[output];
                }

                #[cfg(debug_assertions)]
                {
                    let mut reference = 0.0f32;
                    for output in 0..OUTPUT_CLASSES {
                        reference += self.model.output.weights[output * neuron_count + neuron]
                            * self.errors[output];
                    }
                    debug_assert!((reference - signal).abs() <= 1.0e-5);
                }
                self.learning_signal[neuron] = signal;
            }
        }

        // Direct readout/context supervision remains immediate on every target.
        // Formula v2 only consolidates recurrent/input credit assignment.
        let output_learning_rate = self.model.config.learning.output_learning_rate;
        let maximum_update = self.model.config.learning.max_update;
        let weight_min = self.model.config.learning.weight_min;
        let weight_max = self.model.config.learning.weight_max;
        let mut active_index = 0usize;
        while active_index < self.active.len() {
            let neuron = self.active[active_index];
            let activation = self.state.activation[neuron];
            let cache_row = neuron * OUTPUT_CLASSES;
            self.mark_output_neuron_changed(neuron);
            for output in 0..OUTPUT_CLASSES {
                let raw_delta =
                    -output_learning_rate * supervised_strength * self.errors[output] * activation;
                let delta = bounded_delta(raw_delta, maximum_update);
                let updated = (self.model.output.weights[output * neuron_count + neuron] + delta)
                    .clamp(weight_min, weight_max);
                self.model.output.weights[output * neuron_count + neuron] = updated;
                self.output_weights_by_neuron[cache_row + output] = updated;
            }
            active_index += 1;
        }
        if self.track_parameter_changes {
            self.output_bias_dirty = true;
        }
        for output in 0..OUTPUT_CLASSES {
            let raw_delta = -output_learning_rate * supervised_strength * self.errors[output];
            let delta = bounded_delta(raw_delta, maximum_update);
            self.model.output.bias[output] += delta;
            if !self.model.output.bias[output].is_finite() {
                return Err(LeoError::numerical(format!(
                    "output bias {output} became non-finite"
                )));
            }
        }
        if self.context_applied_this_step {
            self.update_context_projection(supervised_strength)?;
        }

        let mut metrics = LearningUpdateMetrics::default();
        if plasticity_scale > 0.0 {
            let consolidated_strength = supervised_strength * plasticity_scale;
            self.update_recurrent_synapses(consolidated_strength, &mut metrics)?;
            self.update_input_synapses(consolidated_strength, &mut metrics)?;
        }
        // Inhibitory homeostasis remains per-step; it is a local activity
        // regulator rather than delayed supervised credit assignment.
        self.apply_inhibitory_homeostasis(strength)?;
        self.plasticity_phase = next_plasticity_phase;
        Ok(metrics)
    }

    fn update_context_projection(&mut self, strength: f32) -> LeoResult<()> {
        if self.active_context_slots.is_empty() {
            return Ok(());
        }
        let learning_rate = self.model.config.context.learning_rate;
        let maximum_update = self.model.config.learning.max_update;
        let weight_min = self.model.config.learning.weight_min;
        let weight_max = self.model.config.learning.weight_max;
        let embedding_dim = self.model.config.context.embedding_dim;

        // Compute dL/d(context_latent) from the exact shared projection used
        // during the forward pass before mutating any projection weights.
        self.context_latent_gradient.fill(0.0);
        for output in 0..OUTPUT_CLASSES {
            let row = output * embedding_dim;
            for dimension in 0..embedding_dim {
                self.context_latent_gradient[dimension] +=
                    self.model.context.output_weights[row + dimension] * self.errors[output];
            }
        }

        // Shared latent-to-output projection learns structure across contexts.
        if self.track_parameter_changes {
            self.context_output_dirty = true;
        }
        for output in 0..OUTPUT_CLASSES {
            let row = output * embedding_dim;
            for dimension in 0..embedding_dim {
                let gradient = self.errors[output] * self.context_latent[dimension];
                let delta = bounded_delta(-learning_rate * strength * gradient, maximum_update);
                let updated = (self.model.context.output_weights[row + dimension] + delta)
                    .clamp(weight_min, weight_max);
                if !updated.is_finite() {
                    return Err(LeoError::numerical(
                        "non-finite context output update rejected",
                    ));
                }
                self.model.context.output_weights[row + dimension] = updated;
            }
        }

        // Each slot receives the exact chain-rule scale used in the forward sum:
        // dL/dE_slot = active_context_scale * dL/d(context_latent).
        let mut context_index = 0usize;
        while context_index < self.active_context_slots.len() {
            let slot = self.active_context_slots[context_index];
            let learning_scale = self.active_context_scales[context_index];
            let row = slot * embedding_dim;
            for dimension in 0..embedding_dim {
                let gradient = learning_scale * self.context_latent_gradient[dimension];
                let delta = bounded_delta(-learning_rate * strength * gradient, maximum_update);
                let updated = (self.model.context.embeddings[row + dimension] + delta)
                    .clamp(weight_min, weight_max);
                if !updated.is_finite() {
                    return Err(LeoError::internal(
                        "non-finite context embedding update rejected",
                    ));
                }
                self.model.context.embeddings[row + dimension] = updated;
            }
            self.mark_context_slot_changed(slot);
            self.model.context.observations[slot] =
                self.model.context.observations[slot].saturating_add(1);
            context_index += 1;
        }
        Ok(())
    }

    fn mark_changed(index: usize, marks: &mut [bool], changed: &mut Vec<usize>) {
        if !marks[index] {
            marks[index] = true;
            changed.push(index);
        }
    }

    fn mark_threshold_changed(&mut self, index: usize) {
        if self.track_parameter_changes {
            Self::mark_changed(
                index,
                &mut self.changed_threshold_mark,
                &mut self.changed_threshold,
            );
        }
    }

    fn mark_recurrent_weight_changed(&mut self, index: usize) {
        if self.track_parameter_changes {
            Self::mark_changed(
                index,
                &mut self.changed_recurrent_weight_mark,
                &mut self.changed_recurrent_weight,
            );
        }
    }

    fn mark_input_weight_changed(&mut self, index: usize) {
        if self.track_parameter_changes {
            Self::mark_changed(
                index,
                &mut self.changed_input_weight_mark,
                &mut self.changed_input_weight,
            );
        }
    }

    fn mark_output_neuron_changed(&mut self, neuron: usize) {
        if self.track_parameter_changes {
            Self::mark_changed(
                neuron,
                &mut self.changed_output_neuron_mark,
                &mut self.changed_output_neurons,
            );
        }
    }

    fn mark_context_slot_changed(&mut self, slot: usize) {
        if self.track_parameter_changes {
            Self::mark_changed(
                slot,
                &mut self.changed_context_slot_mark,
                &mut self.changed_context_slots,
            );
        }
    }

    fn add_learning_destination(&mut self, destination: usize) {
        if self.learning_destination_mark[destination] {
            return;
        }
        self.learning_destination_mark[destination] = true;
        self.learning_destinations.push(destination);
    }

    fn update_recurrent_synapses(
        &mut self,
        strength: f32,
        metrics: &mut LearningUpdateMetrics,
    ) -> LeoResult<()> {
        let learning_rate = self.model.config.learning.recurrent_learning_rate;
        let maximum_update = self.model.config.learning.max_update;
        let weight_min = self.model.config.learning.weight_min;
        let weight_max = self.model.config.learning.weight_max;

        let mut eligible_index = 0usize;
        while eligible_index < self.eligible_recurrent_slots.len() {
            let slot = self.eligible_recurrent_slots[eligible_index];
            eligible_index += 1;
            let destination = self.model.recurrent.target_neuron[slot] as usize;
            let eligibility = self.eligibility[slot];
            metrics.eligibility_abs_sum += eligibility.abs() as f64;
            metrics.eligibility_count += 1;
            let gradient = self.learning_signal[destination] * eligibility;
            let delta = bounded_delta(-learning_rate * strength * gradient, maximum_update);
            let old_weight = self.model.recurrent.weight[slot];
            let source = slot / self.model.recurrent.capacity_per_neuron;
            let source_type = self.model.neurons.neuron_type[source];
            let mut new_weight = (old_weight + delta).clamp(weight_min, weight_max);
            new_weight = if source_type == 0 {
                new_weight.max(0.0)
            } else {
                new_weight.min(0.0)
            };
            if !new_weight.is_finite() {
                self.model.statistics.numerical_rejections =
                    self.model.statistics.numerical_rejections.saturating_add(1);
                return Err(LeoError::numerical("non-finite recurrent update rejected"));
            }
            if new_weight != old_weight {
                self.mark_recurrent_weight_changed(slot);
            }
            self.model.recurrent.weight[slot] = new_weight;
        }
        Ok(())
    }

    fn update_input_synapses(
        &mut self,
        strength: f32,
        metrics: &mut LearningUpdateMetrics,
    ) -> LeoResult<()> {
        let learning_rate = self.model.config.learning.recurrent_learning_rate;
        let maximum_update = self.model.config.learning.max_update;
        let weight_max = self.model.config.learning.weight_max;

        let mut eligible_index = 0usize;
        while eligible_index < self.eligible_input_slots.len() {
            let slot = self.eligible_input_slots[eligible_index];
            eligible_index += 1;
            let destination = self.model.input.targets[slot] as usize;
            let eligibility = self.input_eligibility[slot];
            metrics.eligibility_abs_sum += eligibility.abs() as f64;
            metrics.eligibility_count += 1;
            let gradient = self.learning_signal[destination] * eligibility;
            let delta = bounded_delta(-learning_rate * strength * gradient, maximum_update);
            let old_weight = self.model.input.weights[slot];
            let new_weight = (old_weight + delta).clamp(0.0, weight_max);
            if !new_weight.is_finite() {
                return Err(LeoError::numerical("non-finite input update rejected"));
            }
            if new_weight != old_weight {
                self.mark_input_weight_changed(slot);
            }
            self.model.input.weights[slot] = new_weight;
        }
        Ok(())
    }

    fn apply_inhibitory_homeostasis(&mut self, strength: f32) -> LeoResult<()> {
        let desired_activity = self.model.config.dynamics.target_activity;
        let learning_rate = self.model.config.learning.inhibitory_learning_rate;
        let maximum_update = self.model.config.learning.max_update;
        let weight_min = self.model.config.learning.weight_min;

        let mut active_index = 0usize;
        while active_index < self.active.len() {
            let source = self.active[active_index];
            active_index += 1;
            if self.model.neurons.neuron_type[source] == 0 {
                continue;
            }
            let source_activation = self.state.activation[source];
            for slot in self.model.synapse_slot_range(source) {
                let target = self.model.recurrent.target_neuron[slot] as usize;
                let delta = inhibitory_homeostasis_delta(
                    source_activation,
                    if self.state.activation[target] > 0.0 {
                        1.0
                    } else {
                        0.0
                    },
                    desired_activity,
                    learning_rate,
                    strength,
                    maximum_update,
                );
                let old_weight = self.model.recurrent.weight[slot];
                let new_weight = (old_weight + delta).clamp(weight_min, 0.0);
                if !new_weight.is_finite() {
                    return Err(LeoError::numerical("non-finite inhibitory update rejected"));
                }
                if new_weight != old_weight {
                    self.mark_recurrent_weight_changed(slot);
                }
                self.model.recurrent.weight[slot] = new_weight;
            }
        }
        Ok(())
    }

    fn update_homeostasis(&mut self, active_neurons: usize) {
        let target = self.model.config.dynamics.target_activity;
        let rate = self.model.config.dynamics.threshold_homeostasis_rate;
        // Thresholds are persistent integrators of unbiased per-step activity error.
        // Runtime state remains disposable, while repeated updates make the target a
        // long-run average rather than a mandatory number of winners on each tick.
        let population_activity = active_neurons as f32 / self.model.neuron_count().max(1) as f32;
        let population_error = population_activity - target;
        let mut touched_index = 0usize;
        while touched_index < self.touched.len() {
            let neuron = self.touched[touched_index];
            let local_activity = if self.state.activation[neuron] > 0.0 {
                1.0
            } else {
                0.0
            };
            let local_error = local_activity - target;
            let threshold_delta = rate * (local_error + population_error);
            let old_threshold = self.model.neurons.threshold[neuron];
            let new_threshold = (old_threshold + threshold_delta).clamp(0.05, 2.0);
            if new_threshold != old_threshold {
                self.mark_threshold_changed(neuron);
            }
            self.model.neurons.threshold[neuron] = new_threshold;
            touched_index += 1;
        }
    }

    fn update_statistics(&mut self, symbol: u32, loss: Option<f32>, emitted_events: usize) {
        if symbol < 256 {
            self.model.statistics.processed_bytes =
                self.model.statistics.processed_bytes.saturating_add(1);
        }
        if let Some(value) = loss {
            self.model.statistics.training_loss_sum += value as f64;
            self.model.statistics.training_targets =
                self.model.statistics.training_targets.saturating_add(1);
        }
        self.model.statistics.active_neurons_sum = self
            .model
            .statistics
            .active_neurons_sum
            .saturating_add(self.active.len() as u64);
        self.model.statistics.active_neurons_peak = self
            .model
            .statistics
            .active_neurons_peak
            .max(self.active.len() as u64);
        self.model.statistics.synaptic_events = self
            .model
            .statistics
            .synaptic_events
            .saturating_add(emitted_events as u64);
        self.model.statistics.persistent_ticks =
            self.model.statistics.persistent_ticks.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{context_survives_dropout, Runtime};
    use crate::model::{BRANCH_EXCITATORY, BRANCH_INHIBITORY};
    use crate::symbols::output_symbol_to_index;
    use crate::{Config, Model, Permission};

    fn integrated_config() -> Config {
        let mut config = Config::default();
        config.model.name = "integrated-runtime-test".into();
        config.model.neuron_count = 8;
        config.model.block_count = 1;
        config.model.neurons_per_block = 8;
        config.model.synapses_per_neuron = 1;
        config.model.input_fanout = 1;
        config.model.max_active_per_block = 8;
        config.model.max_active_global = 8;
        config.dynamics.fatigue_gain = 1.0e-6;
        config.dynamics.refractory_ticks = 1;
        config.dynamics.target_activity = 0.5;
        config.dynamics.threshold_homeostasis_rate = 1.0e-8;
        config.learning.output_learning_rate = 1.0e-8;
        config.learning.recurrent_learning_rate = 0.1;
        config.learning.inhibitory_learning_rate = 1.0e-8;
        config.learning.eligibility_epsilon = 1.0e-7;
        config.learning.surrogate_width = 0.25;
        config.learning.surrogate_gain = 0.1;
        config.learning.max_update = 0.1;
        config.context.max_order = 3;
        config.context.slots_per_order = 1024;
        config.context.learning_rate = 0.1;
        config.context.confidence_observations = 1;
        config.context.dropout_rate = 0.0;
        config.validate().unwrap();
        config
    }

    fn causal_model(delay: u8) -> Model {
        let mut model = Model::initialize(integrated_config()).unwrap();
        model.neurons.threshold.fill(0.1);
        model.neurons.excitability.fill(1.0);
        model.recurrent.weight.fill(0.0);
        model.neurons.neuron_type[0] = 0;
        for slot in model.synapse_slot_range(0) {
            model.recurrent.target_branch[slot] = BRANCH_EXCITATORY;
        }
        model.input.weights.fill(0.0);
        model.input.branches.fill(BRANCH_EXCITATORY);
        model.output.weights.fill(0.0);
        model.output.bias.fill(0.0);

        model.recurrent.target_neuron[0] = 1;
        model.recurrent.target_branch[0] = BRANCH_EXCITATORY;
        model.recurrent.delay[0] = delay;
        model.recurrent.weight[0] = 0.5;

        let input_slot = b'A' as usize * model.input.fanout;
        model.input.targets[input_slot] = 0;
        model.input.weights[input_slot] = 1.0;
        let quiet_slot = b'B' as usize * model.input.fanout;
        model.input.targets[quiet_slot] = 7;
        model.input.weights[quiet_slot] = 0.0;

        let wrong = output_symbol_to_index(b'Z' as u32).unwrap();
        let neuron_count = model.neuron_count();
        model.output.weights[wrong * neuron_count + 1] = 1.0;
        model
    }

    #[test]
    fn frozen_execution_is_read_only() {
        let model = causal_model(2);
        let original_weight = model.recurrent.weight[0];
        let mut runtime = Runtime::new(model).unwrap();
        runtime.begin_document();
        for symbol in *b"ABBA" {
            runtime
                .step(symbol as u32, None, Permission::Frozen)
                .unwrap();
        }
        assert_eq!(runtime.model().recurrent.weight[0], original_weight);
        assert!(runtime.eligible_recurrent_slots.is_empty());
        assert!(runtime.eligible_input_slots.is_empty());
    }

    #[test]
    fn recurrent_credit_arrives_on_the_configured_delay() {
        for delay in [1u8, 2, 4, 8] {
            let mut runtime = Runtime::new(causal_model(delay)).unwrap();
            runtime.begin_document();
            let initial_weight = runtime.model().recurrent.weight[0];
            runtime
                .step(b'A' as u32, Some(b'Y' as u32), Permission::Training)
                .unwrap();
            assert_eq!(runtime.model().recurrent.weight[0], initial_weight);

            for _ in 1..delay {
                runtime
                    .step(b'B' as u32, Some(b'Y' as u32), Permission::Training)
                    .unwrap();
                assert_eq!(runtime.model().recurrent.weight[0], initial_weight);
            }
            runtime
                .step(b'B' as u32, Some(b'Y' as u32), Permission::Training)
                .unwrap();
            assert_ne!(runtime.model().recurrent.weight[0], initial_weight);
        }
    }

    #[test]
    fn input_trace_survives_refractory_tick_and_receives_later_error() {
        let mut model = causal_model(1);
        let input_slot = b'A' as usize * model.input.fanout;
        model.input.weights[input_slot] = 0.5;
        let wrong = output_symbol_to_index(b'Z' as u32).unwrap();
        let neuron_count = model.neuron_count();
        model.output.weights.fill(0.0);
        model.output.weights[wrong * neuron_count] = 1.0;
        let mut runtime = Runtime::new(model).unwrap();
        runtime.begin_document();
        runtime
            .step(b'A' as u32, None, Permission::Training)
            .unwrap();
        let before = runtime.model().input.weights[input_slot];
        runtime
            .step(b'B' as u32, Some(b'Y' as u32), Permission::Training)
            .unwrap();
        assert!(runtime.input_eligibility[input_slot] > 0.0);
        assert!(runtime.model().input.weights[input_slot] < before);
    }

    #[test]
    fn context_embedding_gradient_uses_the_exact_forward_scale() {
        let mut config = integrated_config();
        config.context.embedding_dim = 1;
        config.context.learning_rate = 0.1;
        config.learning.max_update = 1.0;
        config.validate().unwrap();
        let mut model = Model::initialize(config).unwrap();
        model.context.embeddings.fill(0.0);
        model.context.output_weights.fill(0.0);
        model.context.keys[0] = 1;
        model.context.keys[1] = 2;
        let output = output_symbol_to_index(b'X' as u32).unwrap();
        model.context.output_weights[output] = 1.0;
        let mut runtime = Runtime::new(model).unwrap();
        runtime.active_context_slots.extend([0, 1]);
        runtime.active_context_scales.extend([0.25, 0.75]);
        runtime.errors.fill(0.0);
        runtime.errors[output] = 1.0;
        runtime.update_context_projection(1.0).unwrap();

        let first = runtime.model().context.embeddings[0];
        let second = runtime.model().context.embeddings[1];
        assert!((first + 0.025).abs() < 1.0e-6);
        assert!((second + 0.075).abs() < 1.0e-6);
    }

    #[test]
    fn activity_target_does_not_force_winners_without_evidence() {
        let mut model = Model::initialize(integrated_config()).unwrap();
        model.recurrent.weight.fill(0.0);
        model.input.weights.fill(0.0);
        model.neurons.threshold.fill(0.5);
        let mut runtime = Runtime::new(model).unwrap();
        runtime.begin_document();
        let metrics = runtime.step(b'A' as u32, None, Permission::Frozen).unwrap();
        assert_eq!(metrics.active_neurons, 0);
        assert_eq!(metrics.suprathreshold_neurons, 0);
    }

    fn recurrent_memory_model() -> Model {
        let mut config = integrated_config();
        config.model.neuron_count = 64;
        config.model.block_count = 1;
        config.model.neurons_per_block = 64;
        config.model.synapses_per_neuron = 1;
        config.model.input_fanout = 1;
        config.model.max_active_per_block = 64;
        config.model.max_active_global = 64;
        config.dynamics.target_activity = 0.01;
        config.dynamics.membrane_reset_fraction = 1.0;
        config.dynamics.fatigue_gain = 1.0e-6;
        config.dynamics.adaptation_fast_gain = 1.0e-8;
        config.dynamics.adaptation_medium_gain = 1.0e-8;
        config.dynamics.adaptation_slow_gain = 1.0e-8;
        config.context.max_order = 8;
        config.context.embedding_dim = 8;
        config.context.slots_per_order = 64;
        config.context.probe_limit = 4;
        config.validate().unwrap();
        let mut model = Model::initialize(config).unwrap();
        model.neurons.threshold.fill(0.9);
        // Gain above one preserves a unit recurrent pulse across long hand-wired
        // chains without changing the production model's default excitability.
        model.neurons.excitability.fill(2.0);
        model.recurrent.weight.fill(0.0);
        model.input.weights.fill(0.0);
        model.output.weights.fill(0.0);
        model.output.bias.fill(0.0);
        model.context.embeddings.fill(0.0);
        model.context.output_weights.fill(0.0);
        model
    }

    fn configure_chain(model: &mut Model, source: usize, links: usize, delay: u8) -> usize {
        for offset in 0..links {
            let neuron = source + offset;
            let slot = model.synapse_slot_range(neuron).start;
            model.neurons.neuron_type[neuron] = 0;
            model.recurrent.target_neuron[slot] = (neuron + 1) as u32;
            model.recurrent.target_branch[slot] = BRANCH_EXCITATORY;
            model.recurrent.delay[slot] = delay;
            model.recurrent.weight[slot] = 1.0;
        }
        let endpoint = source + links;
        model.neurons.neuron_type[endpoint] = 0;
        let endpoint_slot = model.synapse_slot_range(endpoint).start;
        model.recurrent.target_branch[endpoint_slot] = BRANCH_EXCITATORY;
        model.recurrent.delay[endpoint_slot] = 1;
        model.recurrent.weight[endpoint_slot] = 0.0;
        endpoint
    }

    #[test]
    fn recurrent_only_state_preserves_branch_information_beyond_context_order() {
        for distance in [16usize, 32, 64, 128] {
            let mut model = recurrent_memory_model();
            let links = distance / 8;
            let endpoint_a = configure_chain(&mut model, 0, links, 8);
            let endpoint_b = configure_chain(&mut model, 32, links, 8);
            let input_a = b'A' as usize * model.input.fanout;
            let input_b = b'B' as usize * model.input.fanout;
            model.input.targets[input_a] = 0;
            model.input.targets[input_b] = 32;
            model.input.branches[input_a] = BRANCH_EXCITATORY;
            model.input.branches[input_b] = BRANCH_EXCITATORY;
            model.input.weights[input_a] = 1.0;
            model.input.weights[input_b] = 1.0;
            let neuron_count = model.neuron_count();
            let output_x = output_symbol_to_index(b'X' as u32).unwrap();
            let output_y = output_symbol_to_index(b'Y' as u32).unwrap();
            model.output.weights[output_x * neuron_count + endpoint_a] = 1.0;
            model.output.weights[output_y * neuron_count + endpoint_b] = 1.0;
            model.validate().unwrap();

            let predict = |runtime: &mut Runtime, prefix: u8| {
                runtime.begin_document();
                runtime
                    .step(prefix as u32, None, Permission::Frozen)
                    .unwrap();
                let mut predicted = 0u32;
                for _ in 0..distance {
                    predicted = runtime
                        .step(b'.' as u32, None, Permission::Frozen)
                        .unwrap()
                        .predicted_symbol;
                }
                predicted
            };
            let mut runtime = Runtime::new(model).unwrap();
            assert_eq!(
                predict(&mut runtime, b'A'),
                b'X' as u32,
                "distance {distance}"
            );
            assert_eq!(
                predict(&mut runtime, b'B'),
                b'Y' as u32,
                "distance {distance}"
            );
        }
    }

    #[test]
    fn recurrent_only_readout_learns_delayed_branches_beyond_frequency() {
        for distance in [16usize, 32, 64, 128] {
            let mut model = recurrent_memory_model();
            model.config.learning.output_learning_rate = 0.05;
            model.config.learning.max_update = 0.05;
            let links = distance / 8;
            configure_chain(&mut model, 0, links, 8);
            configure_chain(&mut model, 32, links, 8);
            let input_a = b'A' as usize * model.input.fanout;
            let input_b = b'B' as usize * model.input.fanout;
            model.input.targets[input_a] = 0;
            model.input.targets[input_b] = 32;
            model.input.branches[input_a] = BRANCH_EXCITATORY;
            model.input.branches[input_b] = BRANCH_EXCITATORY;
            model.input.weights[input_a] = 1.0;
            model.input.weights[input_b] = 1.0;
            model.validate().unwrap();
            let mut runtime = Runtime::new(model).unwrap();

            for _ in 0..64 {
                for (prefix, target) in [(b'A', b'X'), (b'B', b'Y')] {
                    runtime.begin_document();
                    runtime
                        .step(prefix as u32, None, Permission::Frozen)
                        .unwrap();
                    for _ in 0..distance - 1 {
                        runtime.step(b'.' as u32, None, Permission::Frozen).unwrap();
                    }
                    runtime
                        .step(b'.' as u32, Some(target as u32), Permission::Training)
                        .unwrap();
                    runtime.finish_document();
                }
            }

            for (prefix, expected) in [(b'A', b'X'), (b'B', b'Y')] {
                runtime.begin_document();
                runtime
                    .step(prefix as u32, None, Permission::Frozen)
                    .unwrap();
                let mut predicted = 0u32;
                for _ in 0..distance {
                    predicted = runtime
                        .step(b'.' as u32, None, Permission::Frozen)
                        .unwrap()
                        .predicted_symbol;
                }
                assert_eq!(predicted, expected as u32, "distance {distance}");
            }
        }
    }

    #[test]
    fn recurrent_only_ring_maintains_a_period_longer_than_context_order() {
        let mut model = recurrent_memory_model();
        let period = 16usize;
        let neuron_count = model.neuron_count();
        for neuron in 0..period {
            let slot = model.synapse_slot_range(neuron).start;
            model.neurons.neuron_type[neuron] = 0;
            model.recurrent.target_neuron[slot] = ((neuron + 1) % period) as u32;
            model.recurrent.target_branch[slot] = BRANCH_EXCITATORY;
            model.recurrent.delay[slot] = 1;
            model.recurrent.weight[slot] = 1.0;
            let output = output_symbol_to_index((b'A' + neuron as u8) as u32).unwrap();
            model.output.weights[output * neuron_count + neuron] = 1.0;
        }
        let input = b'!' as usize * model.input.fanout;
        model.input.targets[input] = 0;
        model.input.branches[input] = BRANCH_EXCITATORY;
        model.input.weights[input] = 1.0;
        model.validate().unwrap();
        let mut runtime = Runtime::new(model).unwrap();
        runtime.begin_document();
        let mut predictions = Vec::new();
        predictions.push(
            runtime
                .step(b'!' as u32, None, Permission::Frozen)
                .unwrap()
                .predicted_symbol,
        );
        for _ in 1..period * 2 {
            predictions.push(
                runtime
                    .step(b'.' as u32, None, Permission::Frozen)
                    .unwrap()
                    .predicted_symbol,
            );
        }
        assert_eq!(&predictions[..period], &predictions[period..period * 2]);
        assert!(predictions[..period]
            .windows(2)
            .all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn context_fingerprints_separate_colliding_buckets() {
        let mut config = integrated_config();
        config.context.max_order = 1;
        config.context.slots_per_order = 4;
        config.context.probe_limit = 4;
        config.validate().unwrap();
        let model = Model::initialize(config).unwrap();
        let mut runtime = Runtime::new(model).unwrap();

        let first = runtime.resolve_context_slot(1, 1, true).unwrap();
        let second = runtime.resolve_context_slot(1, 5, true).unwrap();

        assert_ne!(first, second);
        assert_eq!(runtime.model().context.keys[first], 1);
        assert_eq!(runtime.model().context.keys[second], 5);
    }

    #[test]
    fn temporal_context_learns_branching_prefixes_with_fixed_topology() {
        let mut config = integrated_config();
        config.learning.output_learning_rate = 1.0e-8;
        config.context.learning_rate = 0.5;
        config.learning.max_update = 0.1;
        config.validate().unwrap();

        let mut model = Model::initialize(config).unwrap();
        model.recurrent.weight.fill(0.0);
        model.input.weights.fill(0.0);
        model.output.weights.fill(0.0);
        model.output.bias.fill(0.0);
        model.neurons.threshold.fill(2.0);
        let fixed_synapses = model.recurrent.weight.len();
        let mut runtime = Runtime::new(model).unwrap();

        let stories: [&[u8]; 2] = [b"ABX", b"CBY"];
        for _ in 0..128 {
            for story in stories {
                runtime.begin_document();
                runtime
                    .step(
                        crate::symbols::BEGIN_DOCUMENT,
                        Some(story[0] as u32),
                        Permission::Training,
                    )
                    .unwrap();
                runtime
                    .step(story[0] as u32, Some(story[1] as u32), Permission::Training)
                    .unwrap();
                runtime
                    .step(story[1] as u32, Some(story[2] as u32), Permission::Training)
                    .unwrap();
                runtime
                    .step(
                        story[2] as u32,
                        Some(crate::symbols::END_DOCUMENT),
                        Permission::Training,
                    )
                    .unwrap();
                runtime.finish_document();
            }
        }

        let predict = |runtime: &mut Runtime, prefix: &[u8]| {
            runtime.begin_document();
            runtime
                .step(crate::symbols::BEGIN_DOCUMENT, None, Permission::Frozen)
                .unwrap();
            let mut prediction = crate::symbols::END_DOCUMENT;
            for byte in prefix {
                prediction = runtime
                    .step(*byte as u32, None, Permission::Frozen)
                    .unwrap()
                    .predicted_symbol;
            }
            prediction
        };

        assert_eq!(predict(&mut runtime, b"AB"), b'X' as u32);
        assert_eq!(predict(&mut runtime, b"CB"), b'Y' as u32);
        assert_eq!(runtime.model().recurrent.weight.len(), fixed_synapses);
    }

    #[test]
    fn context_dropout_is_deterministic_and_exposes_neural_steps() {
        let first = (0..1024)
            .filter(|tick| context_survives_dropout(1337, *tick, b'A' as u32, 0.5))
            .count();
        let second = (0..1024)
            .filter(|tick| context_survives_dropout(1337, *tick, b'A' as u32, 0.5))
            .count();
        assert_eq!(first, second);
        assert!((400..=624).contains(&first));
        assert!((0..32).all(|tick| context_survives_dropout(1337, tick, b'A' as u32, 0.0)));
    }

    #[test]
    fn end_document_target_receives_extra_bounded_weight() {
        let mut config = integrated_config();
        config.learning.output_learning_rate = 0.01;
        config.learning.end_document_weight = 4.0;
        config.learning.max_update = 0.02;
        config.context.dropout_rate = 0.0;
        let mut model = Model::initialize(config).unwrap();
        model.recurrent.weight.fill(0.0);
        model.input.weights.fill(0.0);
        model.output.weights.fill(0.0);
        model.output.bias.fill(0.0);
        model.neurons.threshold.fill(2.0);
        let mut runtime = Runtime::new(model).unwrap();
        runtime.begin_document();
        runtime
            .step(
                crate::symbols::BEGIN_DOCUMENT,
                Some(crate::symbols::END_DOCUMENT),
                Permission::Training,
            )
            .unwrap();
        let end_index = crate::symbols::END_DOCUMENT_OUTPUT_INDEX;
        assert!(runtime.model().output.bias[end_index] >= 0.019);
        assert!(runtime.model().output.bias[end_index] <= 0.02 + f32::EPSILON);
    }

    #[test]
    fn initialization_enforces_excitatory_and_inhibitory_routes() {
        let model = Model::initialize(integrated_config()).unwrap();
        for source in 0..model.neuron_count() {
            for slot in model.synapse_slot_range(source) {
                if model.neurons.neuron_type[source] == 0 {
                    assert!(model.recurrent.weight[slot] >= 0.0);
                    assert_ne!(model.recurrent.target_branch[slot], BRANCH_INHIBITORY);
                } else {
                    assert!(model.recurrent.weight[slot] <= 0.0);
                    assert_eq!(model.recurrent.target_branch[slot], BRANCH_INHIBITORY);
                }
            }
        }
    }

    #[test]
    fn adaptation_and_reset_are_part_of_every_runtime() {
        let mut model = Model::initialize(integrated_config()).unwrap();
        model.recurrent.weight.fill(0.0);
        model.input.weights.fill(0.0);
        model.neurons.threshold.fill(0.1);
        let slot = b'A' as usize * model.input.fanout;
        model.input.targets[slot] = 0;
        model.input.branches[slot] = BRANCH_EXCITATORY;
        model.input.weights[slot] = 1.0;
        let mut runtime = Runtime::new(model).unwrap();
        runtime.begin_document();
        let metrics = runtime.step(b'A' as u32, None, Permission::Frozen).unwrap();
        assert!(metrics.active_neurons > 0);
        assert!(runtime.state.membrane[0] < 1.0);
        assert!(runtime.adaptation_state.iter().any(|state| state[0] > 0.0));
    }
}
