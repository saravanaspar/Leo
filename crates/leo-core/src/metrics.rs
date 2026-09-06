//! Runtime and training measurements.

#[derive(Debug, Clone, Default)]
pub struct TrainingStatistics {
    pub processed_bytes: u64,
    pub processed_stories: u64,
    pub training_loss_sum: f64,
    pub training_targets: u64,
    pub active_neurons_sum: u64,
    pub active_neurons_peak: u64,
    pub synaptic_events: u64,
    pub numerical_rejections: u64,
    /// Number of non-frozen runtime steps represented by the persistent activity totals.
    pub persistent_ticks: u64,
}

impl TrainingStatistics {
    pub fn mean_training_loss(&self) -> f64 {
        if self.training_targets == 0 {
            0.0
        } else {
            self.training_loss_sum / self.training_targets as f64
        }
    }

    pub fn bits_per_byte(&self) -> f64 {
        self.mean_training_loss() / std::f64::consts::LN_2
    }

    pub fn mean_active_neurons_per_tick(&self) -> f64 {
        if self.persistent_ticks == 0 {
            0.0
        } else {
            self.active_neurons_sum as f64 / self.persistent_ticks as f64
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StepMetrics {
    pub loss: Option<f32>,
    pub predicted_symbol: u32,
    pub active_neurons: usize,
    pub emitted_events: usize,
    /// Neurons with positive activation before block-local and global competition.
    pub suprathreshold_neurons: usize,
    /// Candidates remaining after the block-local safety cap and before the global safety cap.
    pub block_selected_neurons: usize,
    /// Candidates removed by the global safety cap on this tick.
    pub global_cap_clipped_neurons: usize,
    /// Smoothed runtime-only activation cutoff retained for observability only.
    pub population_inhibition: f32,
    /// Number of learned temporal-context rows used for this prediction.
    pub context_cells: usize,
    /// Number of fixed-capacity context slots probed while resolving those rows.
    pub context_probes: usize,
    /// Whether latent temporal context was applied on this step.
    pub context_applied: bool,
    /// Reduction in target loss contributed by temporal context over neural output alone.
    pub context_loss_gain: f32,
    /// Actual multiply-add count for neural output and latent context projection.
    pub output_madds: usize,
    pub eligible_recurrent_synapses: usize,
    pub eligible_input_synapses: usize,
    pub mean_abs_eligibility: f32,
}
