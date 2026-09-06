//! Configuration for Leo's single integrated learning architecture.

use crate::{LeoError, LeoResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: ModelConfig,
    pub dynamics: DynamicsConfig,
    pub learning: LearningConfig,
    pub context: ContextConfig,
    pub replay: ReplayConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub seed: u64,
    pub neuron_count: usize,
    pub block_count: usize,
    pub neurons_per_block: usize,
    pub branches_per_neuron: usize,
    pub excitatory_fraction: f32,
    pub synapses_per_neuron: usize,
    pub input_fanout: usize,
    pub max_active_per_block: usize,
    pub max_active_global: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicsConfig {
    pub branch_decay: f32,
    pub membrane_decay: f32,
    pub fatigue_decay: f32,
    pub fatigue_gain: f32,
    pub refractory_ticks: u64,
    pub target_activity: f32,
    pub threshold_homeostasis_rate: f32,
    pub population_inhibition_rate: f32,
    pub population_inhibition_max: f32,
    pub adaptation_fast_decay: f32,
    pub adaptation_medium_decay: f32,
    pub adaptation_slow_decay: f32,
    pub adaptation_fast_gain: f32,
    pub adaptation_medium_gain: f32,
    pub adaptation_slow_gain: f32,
    /// Fraction of the firing threshold consumed after activation.
    pub membrane_reset_fraction: f32,
}

impl DynamicsConfig {
    pub fn adaptation_decays(&self) -> [f32; 3] {
        [
            self.adaptation_fast_decay,
            self.adaptation_medium_decay,
            self.adaptation_slow_decay,
        ]
    }

    pub fn adaptation_gains(&self) -> [f32; 3] {
        [
            self.adaptation_fast_gain,
            self.adaptation_medium_gain,
            self.adaptation_slow_gain,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningConfig {
    pub output_learning_rate: f32,
    pub recurrent_learning_rate: f32,
    pub inhibitory_learning_rate: f32,
    pub eligibility_decay: f32,
    pub eligibility_epsilon: f32,
    pub surrogate_width: f32,
    pub surrogate_gain: f32,
    pub max_update: f32,
    pub weight_min: f32,
    pub weight_max: f32,
    pub provisional_strength: f32,
    pub training_strength: f32,
    pub verified_strength: f32,
    /// Extra supervised weight for the rare end-of-document target.
    pub end_document_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextConfig {
    pub max_order: usize,
    pub embedding_dim: usize,
    pub slots_per_order: usize,
    pub probe_limit: usize,
    pub learning_rate: f32,
    pub confidence_observations: u32,
    /// Fraction of trainable steps that omit exact context from the forward pass.
    pub dropout_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    /// Fraction of supervised targets selected for surprise replay after each initial pass.
    pub fraction: f32,
    /// Preferred number of contiguous supervised targets in one replay segment.
    pub segment_bytes: usize,
    /// Number of full-document repeats used by the explicit `teach` command.
    pub teaching_replays: usize,
    /// Number of full-document repeats used by `teach --permission verified`.
    pub max_verified_replays: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub max_dataset_passes: usize,
    pub validate_every_bytes: u64,
    pub checkpoint_every_bytes: u64,
    pub early_stop_checks: usize,
    pub minimum_relative_improvement: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                name: "Leo".to_owned(),
                seed: 1337,
                neuron_count: 32_768,
                block_count: 128,
                neurons_per_block: 256,
                branches_per_neuron: 4,
                excitatory_fraction: 0.80,
                synapses_per_neuron: 48,
                input_fanout: 128,
                max_active_per_block: 8,
                max_active_global: 1_024,
            },
            dynamics: DynamicsConfig {
                branch_decay: 0.85,
                membrane_decay: 0.90,
                fatigue_decay: 0.97,
                fatigue_gain: 0.02,
                refractory_ticks: 1,
                target_activity: 0.02,
                threshold_homeostasis_rate: 0.0001,
                population_inhibition_rate: 0.05,
                population_inhibition_max: 2.0,
                adaptation_fast_decay: 0.95,
                adaptation_medium_decay: 0.99,
                adaptation_slow_decay: 0.999,
                adaptation_fast_gain: 0.005,
                adaptation_medium_gain: 0.001,
                adaptation_slow_gain: 0.0002,
                membrane_reset_fraction: 0.10,
            },
            learning: LearningConfig {
                output_learning_rate: 0.001,
                recurrent_learning_rate: 0.0002,
                inhibitory_learning_rate: 0.00002,
                eligibility_decay: 0.95,
                eligibility_epsilon: 1.0e-5,
                surrogate_width: 0.25,
                surrogate_gain: 0.10,
                max_update: 0.01,
                weight_min: -1.0,
                weight_max: 1.0,
                provisional_strength: 0.20,
                training_strength: 1.00,
                verified_strength: 1.00,
                end_document_weight: 4.0,
            },
            context: ContextConfig {
                max_order: 8,
                embedding_dim: 32,
                slots_per_order: 8192,
                probe_limit: 4,
                learning_rate: 0.05,
                confidence_observations: 4,
                dropout_rate: 0.5,
            },
            replay: ReplayConfig {
                fraction: 0.30,
                segment_bytes: 64,
                teaching_replays: 1,
                max_verified_replays: 4,
            },
            training: TrainingConfig {
                max_dataset_passes: 3,
                validate_every_bytes: 25_000_000,
                checkpoint_every_bytes: 50_000_000,
                early_stop_checks: 3,
                minimum_relative_improvement: 0.005,
            },
        }
    }
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> LeoResult<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_toml(&text)
    }

    pub fn from_toml(text: &str) -> LeoResult<Self> {
        let config = toml::from_str::<Self>(text)
            .map_err(|error| LeoError::configuration(format!("invalid TOML configuration: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> LeoResult<()> {
        if self.model.name.trim().is_empty() {
            return Err(LeoError::configuration("model.name must not be empty"));
        }
        if self.model.neuron_count == 0
            || self.model.block_count == 0
            || self.model.neurons_per_block == 0
        {
            return Err(LeoError::configuration("model dimensions must be positive"));
        }
        if self
            .model
            .block_count
            .checked_mul(self.model.neurons_per_block)
            != Some(self.model.neuron_count)
        {
            return Err(LeoError::configuration(
                "block_count * neurons_per_block must equal neuron_count without overflow",
            ));
        }
        if self.model.branches_per_neuron != 4 {
            return Err(LeoError::configuration(
                "Leo requires exactly four dendritic branches per neuron",
            ));
        }
        if !self.model.excitatory_fraction.is_finite()
            || !(0.0..1.0).contains(&self.model.excitatory_fraction)
        {
            return Err(LeoError::configuration(
                "excitatory_fraction must be finite and strictly between 0 and 1",
            ));
        }
        if self.model.synapses_per_neuron == 0 || self.model.input_fanout == 0 {
            return Err(LeoError::configuration(
                "synapses_per_neuron and input_fanout must be positive",
            ));
        }
        if self.model.synapses_per_neuron > self.model.neuron_count {
            return Err(LeoError::configuration(
                "synapses_per_neuron cannot exceed neuron_count",
            ));
        }
        if self.model.input_fanout > self.model.neuron_count {
            return Err(LeoError::configuration("input_fanout cannot exceed neuron_count"));
        }
        if self.model.max_active_per_block == 0
            || self.model.max_active_per_block > self.model.neurons_per_block
        {
            return Err(LeoError::configuration(
                "max_active_per_block must be in 1..=neurons_per_block",
            ));
        }
        if self.model.max_active_global == 0
            || self.model.max_active_global > self.model.neuron_count
        {
            return Err(LeoError::configuration(
                "max_active_global must be in 1..=neuron_count",
            ));
        }

        if self.dynamics.refractory_ticks == 0 {
            return Err(LeoError::configuration(
                "refractory_ticks must be positive in the integrated architecture",
            ));
        }

        for (name, value) in [
            ("branch_decay", self.dynamics.branch_decay),
            ("membrane_decay", self.dynamics.membrane_decay),
            ("fatigue_decay", self.dynamics.fatigue_decay),
            ("adaptation_fast_decay", self.dynamics.adaptation_fast_decay),
            (
                "adaptation_medium_decay",
                self.dynamics.adaptation_medium_decay,
            ),
            ("adaptation_slow_decay", self.dynamics.adaptation_slow_decay),
            ("eligibility_decay", self.learning.eligibility_decay),
        ] {
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(LeoError::configuration(format!(
                    "{name} must be finite and strictly between 0 and 1"
                )));
            }
        }
        for (name, value) in [
            ("fatigue_gain", self.dynamics.fatigue_gain),
            ("adaptation_fast_gain", self.dynamics.adaptation_fast_gain),
            (
                "adaptation_medium_gain",
                self.dynamics.adaptation_medium_gain,
            ),
            ("adaptation_slow_gain", self.dynamics.adaptation_slow_gain),
            (
                "threshold_homeostasis_rate",
                self.dynamics.threshold_homeostasis_rate,
            ),
            (
                "population_inhibition_rate",
                self.dynamics.population_inhibition_rate,
            ),
            ("output_learning_rate", self.learning.output_learning_rate),
            (
                "recurrent_learning_rate",
                self.learning.recurrent_learning_rate,
            ),
            (
                "inhibitory_learning_rate",
                self.learning.inhibitory_learning_rate,
            ),
            ("eligibility_epsilon", self.learning.eligibility_epsilon),
            ("surrogate_width", self.learning.surrogate_width),
            ("surrogate_gain", self.learning.surrogate_gain),
            ("max_update", self.learning.max_update),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(LeoError::configuration(format!("{name} must be finite and positive")));
            }
        }
        if !self.dynamics.target_activity.is_finite()
            || !(0.0..=1.0).contains(&self.dynamics.target_activity)
            || self.dynamics.target_activity == 0.0
        {
            return Err(LeoError::configuration(
                "target_activity must be finite and in (0, 1]",
            ));
        }
        if !self.dynamics.population_inhibition_max.is_finite()
            || self.dynamics.population_inhibition_max <= 0.0
        {
            return Err(LeoError::configuration(
                "population_inhibition_max must be finite and positive",
            ));
        }
        if !self.dynamics.membrane_reset_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.dynamics.membrane_reset_fraction)
            || self.dynamics.membrane_reset_fraction == 0.0
        {
            return Err(LeoError::configuration(
                "membrane_reset_fraction must be finite and in (0, 1]",
            ));
        }
        let target_active =
            (self.dynamics.target_activity * self.model.neuron_count as f32).ceil() as usize;
        if target_active > self.model.max_active_global {
            return Err(LeoError::configuration(
                "target_activity requires more neurons than max_active_global permits",
            ));
        }

        if self.learning.weight_min >= 0.0
            || self.learning.weight_max <= 0.0
            || self.learning.weight_min >= self.learning.weight_max
        {
            return Err(LeoError::configuration(
                "weight_min must be negative and weight_max must be positive",
            ));
        }
        if !self.learning.end_document_weight.is_finite()
            || !(1.0..=16.0).contains(&self.learning.end_document_weight)
        {
            return Err(LeoError::configuration(
                "end_document_weight must be finite and in [1, 16]",
            ));
        }

        for (name, strength) in [
            ("provisional_strength", self.learning.provisional_strength),
            ("training_strength", self.learning.training_strength),
            ("verified_strength", self.learning.verified_strength),
        ] {
            if !strength.is_finite() || !(0.0..=1.0).contains(&strength) || strength == 0.0 {
                return Err(LeoError::configuration(format!("{name} must be finite and in (0, 1]")));
            }
        }

        if self.context.max_order == 0
            || self.context.embedding_dim == 0
            || self.context.slots_per_order == 0
            || self.context.probe_limit == 0
            || self.context.confidence_observations == 0
        {
            return Err(LeoError::configuration(
                "all context-memory dimensions must be positive",
            ));
        }
        if self.context.embedding_dim > 256 {
            return Err(LeoError::configuration("context.embedding_dim cannot exceed 256"));
        }
        if self.context.max_order > 16 {
            return Err(LeoError::configuration("context.max_order cannot exceed 16"));
        }
        if self.context.probe_limit > self.context.slots_per_order {
            return Err(LeoError::configuration(
                "context.probe_limit cannot exceed slots_per_order",
            ));
        }
        if !self.context.dropout_rate.is_finite()
            || !(0.0..1.0).contains(&self.context.dropout_rate)
        {
            return Err(LeoError::configuration(
                "context.dropout_rate must be finite and in [0, 1)",
            ));
        }
        if !self.context.learning_rate.is_finite() || self.context.learning_rate <= 0.0 {
            return Err(LeoError::configuration(
                "context.learning_rate must be finite and positive",
            ));
        }
        let context_slots = self
            .context
            .max_order
            .checked_mul(self.context.slots_per_order)
            .ok_or_else(|| LeoError::configuration("context slot count overflow"))?;
        context_slots
            .checked_mul(self.context.embedding_dim)
            .ok_or_else(|| LeoError::configuration("context embedding size overflow"))?;
        self.context
            .embedding_dim
            .checked_mul(crate::symbols::OUTPUT_CLASSES)
            .ok_or_else(|| LeoError::configuration("context output projection size overflow"))?;
        if !self.replay.fraction.is_finite()
            || self.replay.fraction <= 0.0
            || self.replay.fraction > 1.0
        {
            return Err(LeoError::configuration(
                "replay.fraction must be finite and in the interval (0, 1]",
            ));
        }
        if self.replay.segment_bytes == 0
            || self.replay.teaching_replays == 0
            || self.replay.max_verified_replays == 0
        {
            return Err(LeoError::configuration("all replay limits must be positive"));
        }
        if self.training.max_dataset_passes == 0
            || self.training.validate_every_bytes == 0
            || self.training.checkpoint_every_bytes == 0
            || self.training.early_stop_checks == 0
        {
            return Err(LeoError::configuration("all training limits must be positive"));
        }
        if !self.training.minimum_relative_improvement.is_finite()
            || self.training.minimum_relative_improvement < 0.0
        {
            return Err(LeoError::configuration(
                "minimum_relative_improvement must be finite and nonnegative",
            ));
        }
        Ok(())
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("serializing a validated Leo configuration cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn integrated_configuration_round_trips() {
        let mut config = Config::default();
        config.dynamics.membrane_reset_fraction = 0.2;
        config.learning.inhibitory_learning_rate = 0.00003;
        config.context.max_order = 6;
        config.context.embedding_dim = 12;
        config.context.slots_per_order = 512;
        config.context.probe_limit = 3;
        config.context.dropout_rate = 0.25;
        config.learning.end_document_weight = 6.0;
        config.replay.segment_bytes = 32;

        let decoded = Config::from_toml(&config.to_toml()).expect("config should round trip");
        assert_eq!(decoded.dynamics.membrane_reset_fraction, 0.2);
        assert_eq!(decoded.learning.inhibitory_learning_rate, 0.00003);
        assert_eq!(decoded.context.max_order, 6);
        assert_eq!(decoded.context.embedding_dim, 12);
        assert_eq!(decoded.context.slots_per_order, 512);
        assert_eq!(decoded.context.probe_limit, 3);
        assert_eq!(decoded.context.dropout_rate, 0.25);
        assert_eq!(decoded.learning.end_document_weight, 6.0);
        assert_eq!(decoded.replay.segment_bytes, 32);
    }

    #[test]
    fn standard_toml_escaping_round_trips_model_names() {
        let mut config = Config::default();
        config.model.name = "Leo # \"v1\"\nreference".to_owned();
        let encoded = config.to_toml();
        let decoded = Config::from_toml(&encoded).expect("escaped TOML should round trip");
        assert_eq!(decoded.model.name, config.model.name);
    }

    #[test]
    fn unknown_architecture_switches_are_rejected() {
        let text = Config::default().to_toml() + "\n[learning]\nunexpected_field = true\n";
        assert!(Config::from_toml(&text).is_err());
    }
}
