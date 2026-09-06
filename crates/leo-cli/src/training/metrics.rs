//! Training/evaluation diagnostics shared by the training engine, validation, and CLI presentation.

use leo_core::StepMetrics;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ActivityDiagnostics {
    pub(crate) steps: u64,
    pub(crate) active_neurons_sum: u64,
    pub(crate) active_neurons_peak: u64,
    pub(crate) emitted_events: u64,
    pub(crate) suprathreshold_sum: u64,
    pub(crate) block_selected_sum: u64,
    pub(crate) cap_clipped_sum: u64,
    pub(crate) cap_hit_steps: u64,
    pub(crate) population_inhibition_sum: f64,
    pub(crate) population_inhibition_peak: f32,
    pub(crate) eligibility_sum: f64,
    pub(crate) eligibility_count: u64,
    pub(crate) context_cells_sum: u64,
    pub(crate) context_probes_sum: u64,
    pub(crate) context_applied_steps: u64,
    pub(crate) context_loss_gain_sum: f64,
    pub(crate) output_madds_sum: u64,
}

impl ActivityDiagnostics {
    pub(crate) fn record(&mut self, metrics: StepMetrics) {
        self.steps = self.steps.saturating_add(1);
        self.active_neurons_sum = self
            .active_neurons_sum
            .saturating_add(metrics.active_neurons as u64);
        self.active_neurons_peak = self.active_neurons_peak.max(metrics.active_neurons as u64);
        self.emitted_events = self
            .emitted_events
            .saturating_add(metrics.emitted_events as u64);
        self.suprathreshold_sum = self
            .suprathreshold_sum
            .saturating_add(metrics.suprathreshold_neurons as u64);
        self.block_selected_sum = self
            .block_selected_sum
            .saturating_add(metrics.block_selected_neurons as u64);
        self.cap_clipped_sum = self
            .cap_clipped_sum
            .saturating_add(metrics.global_cap_clipped_neurons as u64);
        if metrics.global_cap_clipped_neurons > 0 {
            self.cap_hit_steps = self.cap_hit_steps.saturating_add(1);
        }
        self.population_inhibition_sum += metrics.population_inhibition as f64;
        self.population_inhibition_peak = self
            .population_inhibition_peak
            .max(metrics.population_inhibition);
        let eligible = metrics
            .eligible_recurrent_synapses
            .saturating_add(metrics.eligible_input_synapses) as u64;
        self.eligibility_sum += metrics.mean_abs_eligibility as f64 * eligible as f64;
        self.eligibility_count = self.eligibility_count.saturating_add(eligible);
        self.context_cells_sum = self
            .context_cells_sum
            .saturating_add(metrics.context_cells as u64);
        self.context_probes_sum = self
            .context_probes_sum
            .saturating_add(metrics.context_probes as u64);
        if metrics.context_applied {
            self.context_applied_steps = self.context_applied_steps.saturating_add(1);
        }
        self.context_loss_gain_sum += metrics.context_loss_gain as f64;
        self.output_madds_sum = self
            .output_madds_sum
            .saturating_add(metrics.output_madds as u64);
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.steps = self.steps.saturating_add(other.steps);
        self.active_neurons_sum = self
            .active_neurons_sum
            .saturating_add(other.active_neurons_sum);
        self.active_neurons_peak = self.active_neurons_peak.max(other.active_neurons_peak);
        self.emitted_events = self.emitted_events.saturating_add(other.emitted_events);
        self.suprathreshold_sum = self
            .suprathreshold_sum
            .saturating_add(other.suprathreshold_sum);
        self.block_selected_sum = self
            .block_selected_sum
            .saturating_add(other.block_selected_sum);
        self.cap_clipped_sum = self.cap_clipped_sum.saturating_add(other.cap_clipped_sum);
        self.cap_hit_steps = self.cap_hit_steps.saturating_add(other.cap_hit_steps);
        self.population_inhibition_sum += other.population_inhibition_sum;
        self.population_inhibition_peak = self
            .population_inhibition_peak
            .max(other.population_inhibition_peak);
        self.eligibility_sum += other.eligibility_sum;
        self.eligibility_count = self
            .eligibility_count
            .saturating_add(other.eligibility_count);
        self.context_cells_sum = self
            .context_cells_sum
            .saturating_add(other.context_cells_sum);
        self.context_probes_sum = self
            .context_probes_sum
            .saturating_add(other.context_probes_sum);
        self.context_applied_steps = self
            .context_applied_steps
            .saturating_add(other.context_applied_steps);
        self.context_loss_gain_sum += other.context_loss_gain_sum;
        self.output_madds_sum = self.output_madds_sum.saturating_add(other.output_madds_sum);
    }

    pub(crate) fn mean_active(self, neuron_count: usize) -> f64 {
        if self.steps == 0 || neuron_count == 0 {
            0.0
        } else {
            self.active_neurons_sum as f64 / self.steps as f64 / neuron_count as f64
        }
    }

    pub(crate) fn events_per_step(self) -> f64 {
        self.emitted_events as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn mean_active_count(self) -> f64 {
        self.active_neurons_sum as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn mean_suprathreshold(self) -> f64 {
        self.suprathreshold_sum as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn mean_block_selected(self) -> f64 {
        self.block_selected_sum as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn cap_hit_fraction(self) -> f64 {
        self.cap_hit_steps as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn mean_population_inhibition(self) -> f64 {
        self.population_inhibition_sum / self.steps.max(1) as f64
    }

    pub(crate) fn mean_eligibility(self) -> f64 {
        self.eligibility_sum / self.eligibility_count.max(1) as f64
    }

    pub(crate) fn context_cells_per_step(self) -> f64 {
        self.context_cells_sum as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn context_cells_per_byte(self, processed_bytes: u64) -> f64 {
        self.context_cells_sum as f64 / processed_bytes.max(1) as f64
    }

    pub(crate) fn context_probes_per_step(self) -> f64 {
        self.context_probes_sum as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn context_probes_per_byte(self, processed_bytes: u64) -> f64 {
        self.context_probes_sum as f64 / processed_bytes.max(1) as f64
    }

    pub(crate) fn context_use_fraction(self) -> f64 {
        self.context_applied_steps as f64 / self.steps.max(1) as f64
    }

    pub(crate) fn output_madds_per_byte(self, processed_bytes: u64) -> f64 {
        self.output_madds_sum as f64 / processed_bytes.max(1) as f64
    }

    pub(crate) fn context_gain_bits_per_step(self) -> f64 {
        self.context_loss_gain_sum / self.steps.max(1) as f64 / std::f64::consts::LN_2
    }

    pub(crate) fn context_gain_bits_per_byte(self, processed_bytes: u64) -> f64 {
        self.context_loss_gain_sum / processed_bytes.max(1) as f64 / std::f64::consts::LN_2
    }

    pub(crate) fn output_madds_per_step(self) -> f64 {
        self.output_madds_sum as f64 / self.steps.max(1) as f64
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EvaluationMetrics {
    pub(crate) loss: f64,
    pub(crate) bits_per_byte: f64,
    pub(crate) accuracy: f64,
    pub(crate) targets: u64,
    pub(crate) processed_stories: u64,
    pub(crate) processed_bytes: u64,
    pub(crate) neuron_count: usize,
    pub(crate) activity: ActivityDiagnostics,
}

impl EvaluationMetrics {
    pub(crate) fn merge(&mut self, other: Self) {
        let previous_targets = self.targets;
        let combined_targets = previous_targets.saturating_add(other.targets);
        let combined_loss_sum =
            self.loss * previous_targets as f64 + other.loss * other.targets as f64;
        let combined_correct =
            self.accuracy * previous_targets as f64 + other.accuracy * other.targets as f64;
        self.targets = combined_targets;
        self.loss = combined_loss_sum / combined_targets.max(1) as f64;
        self.bits_per_byte = self.loss / std::f64::consts::LN_2;
        self.accuracy = combined_correct / combined_targets.max(1) as f64;
        self.processed_stories = self
            .processed_stories
            .saturating_add(other.processed_stories);
        self.processed_bytes = self.processed_bytes.saturating_add(other.processed_bytes);
        self.neuron_count = self.neuron_count.max(other.neuron_count);
        self.activity.add(other.activity);
    }
}

