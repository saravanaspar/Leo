//! Forward-only local learning helpers for Leo's integrated architecture.

use crate::model::{BRANCH_CONTEXT_GATE, BRANCH_EXCITATORY, BRANCH_INHIBITORY, BRANCH_TEMPORAL};

/// Straight-through derivative for Leo's analog activation plus a small rank-boundary term.
#[inline]
pub fn analog_rank_surrogate(
    margin: f32,
    selection_cutoff: f32,
    selected: bool,
    width: f32,
    gain: f32,
) -> f32 {
    let direct = if selected && margin > 0.0 && margin < 1.0 {
        1.0
    } else {
        0.0
    };
    if width <= 0.0 || gain <= 0.0 {
        return direct;
    }
    let boundary = (1.0 - (margin - selection_cutoff).abs() / width).clamp(0.0, 1.0);
    direct + gain * boundary
}

/// Derivative of destination drive with respect to one branch accumulator.
/// Inhibitory branch values are already negative, so their derivative is positive.
#[inline]
pub fn branch_jacobian(
    branch: u8,
    excitability: f32,
    temporal_accumulator: f32,
    gate_accumulator: f32,
) -> f32 {
    let gate = gate_accumulator.clamp(0.0, 1.0);
    match branch {
        BRANCH_EXCITATORY => excitability,
        BRANCH_TEMPORAL => excitability * gate,
        BRANCH_INHIBITORY => excitability,
        BRANCH_CONTEXT_GATE => {
            let gate_derivative = if gate_accumulator > 0.0 && gate_accumulator < 1.0 {
                1.0
            } else {
                0.0
            };
            excitability * temporal_accumulator * gate_derivative
        }
        _ => 0.0,
    }
}

#[inline]
pub fn bounded_delta(value: f32, maximum: f32) -> f32 {
    value.clamp(-maximum, maximum)
}

/// Local inhibitory homeostasis: overactive targets make inhibitory weights more negative;
/// underactive targets release inhibition gradually.
#[inline]
pub fn inhibitory_homeostasis_delta(
    source_activation: f32,
    target_activity: f32,
    desired_activity: f32,
    learning_rate: f32,
    strength: f32,
    maximum_update: f32,
) -> f32 {
    let activity_error = target_activity - desired_activity;
    bounded_delta(
        -learning_rate * strength * source_activation * activity_error,
        maximum_update,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        analog_rank_surrogate, bounded_delta, branch_jacobian, inhibitory_homeostasis_delta,
    };
    use crate::model::{
        BRANCH_CONTEXT_GATE, BRANCH_EXCITATORY, BRANCH_INHIBITORY, BRANCH_TEMPORAL,
    };

    #[test]
    fn analog_surrogate_preserves_selected_unsaturated_gradient() {
        assert!((analog_rank_surrogate(0.5, 0.4, true, 0.25, 0.1) - 1.06).abs() <= 1.0e-6);
        assert_eq!(analog_rank_surrogate(1.0, 0.4, true, 0.25, 0.1), 0.0);
    }

    #[test]
    fn branch_jacobian_matches_integrated_branch_semantics() {
        assert_eq!(branch_jacobian(BRANCH_EXCITATORY, 1.0, 0.5, 0.5), 1.0);
        assert_eq!(branch_jacobian(BRANCH_TEMPORAL, 1.0, 0.5, 0.5), 0.5);
        assert_eq!(branch_jacobian(BRANCH_INHIBITORY, 1.0, 0.5, 0.5), 1.0);
        assert_eq!(branch_jacobian(BRANCH_CONTEXT_GATE, 1.0, 0.5, 0.5), 0.5);
    }

    #[test]
    fn inhibitory_homeostasis_pushes_in_the_correct_direction() {
        assert!(inhibitory_homeostasis_delta(1.0, 0.10, 0.02, 0.01, 1.0, 0.1) < 0.0);
        assert!(inhibitory_homeostasis_delta(1.0, 0.00, 0.02, 0.01, 1.0, 0.1) > 0.0);
    }

    #[test]
    fn update_clipping_is_symmetric() {
        assert_eq!(bounded_delta(1.0, 0.1), 0.1);
        assert_eq!(bounded_delta(-1.0, 0.1), -0.1);
    }
}
