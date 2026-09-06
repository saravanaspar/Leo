use leo_core::model::BRANCH_EXCITATORY;
use leo_core::symbols::{BEGIN_DOCUMENT, END_DOCUMENT};
use leo_core::{Config, Model, Permission, Runtime};

fn test_config() -> Config {
    Config::from_toml(include_str!("../../../configs/test.toml")).unwrap()
}

fn manual_output_probabilities(runtime: &Runtime) -> Vec<f32> {
    let model = runtime.model();
    let neuron_count = model.neuron_count();
    let mut logits = model.output.bias.clone();
    for &neuron in runtime.active_neurons() {
        let activation = runtime.activation(neuron);
        for (output, logit) in logits.iter_mut().enumerate() {
            *logit += model.output.weights[output * neuron_count + neuron] * activation;
        }
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .into_iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>();
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

fn assert_probabilities_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn frozen_document_is_read_only() {
    let model = Model::initialize(test_config()).unwrap();
    let original = model.clone();
    let mut runtime = Runtime::new(model).unwrap();
    runtime.begin_document();
    runtime
        .step(BEGIN_DOCUMENT, Some(b'A' as u32), Permission::Frozen)
        .unwrap();
    runtime
        .step(b'A' as u32, Some(END_DOCUMENT), Permission::Frozen)
        .unwrap();
    runtime.finish_document();

    assert_eq!(runtime.model().recurrent.weight, original.recurrent.weight);
    assert_eq!(runtime.model().input.weights, original.input.weights);
    assert_eq!(runtime.model().output.weights, original.output.weights);
    assert_eq!(runtime.model().output.bias, original.output.bias);
    assert_eq!(runtime.model().context.keys, original.context.keys);
    assert_eq!(
        runtime.model().context.embeddings,
        original.context.embeddings
    );
    assert_eq!(
        runtime.model().context.output_weights,
        original.context.output_weights
    );
    assert_eq!(
        runtime.model().context.observations,
        original.context.observations
    );
    assert_eq!(runtime.model().statistics.processed_bytes, 0);
    assert_eq!(runtime.model().statistics.processed_stories, 0);
}

#[test]
fn training_document_updates_persistent_statistics() {
    let model = Model::initialize(test_config()).unwrap();
    let mut runtime = Runtime::new(model).unwrap();
    runtime.begin_document();
    runtime
        .step(BEGIN_DOCUMENT, Some(b'A' as u32), Permission::Training)
        .unwrap();
    runtime
        .step(b'A' as u32, Some(END_DOCUMENT), Permission::Training)
        .unwrap();
    runtime.finish_document();

    assert_eq!(runtime.model().statistics.processed_bytes, 1);
    assert_eq!(runtime.model().statistics.processed_stories, 1);
    assert_eq!(runtime.model().statistics.training_targets, 2);
    assert_eq!(runtime.model().statistics.persistent_ticks, 2);
}

#[test]
fn sparse_selection_respects_the_global_cap() {
    let mut config = test_config();
    config.model.neuron_count = 64;
    config.model.block_count = 1;
    config.model.neurons_per_block = 64;
    config.model.input_fanout = 64;
    config.model.max_active_per_block = 64;
    config.model.max_active_global = 8;
    config.dynamics.target_activity = 0.02;
    config.validate().unwrap();

    let mut model = Model::initialize(config).unwrap();
    model.recurrent.weight.fill(0.0);
    model.input.weights.fill(0.0);
    model.neurons.threshold.fill(0.1);
    let start = b'A' as usize * model.input.fanout;
    for (neuron, slot) in (start..start + model.input.fanout).enumerate() {
        model.input.targets[slot] = neuron as u32;
        model.input.branches[slot] = BRANCH_EXCITATORY;
        model.input.weights[slot] = 1.0;
    }

    let mut runtime = Runtime::new(model).unwrap();
    runtime.begin_document();
    let metrics = runtime.step(b'A' as u32, None, Permission::Frozen).unwrap();
    assert!(metrics.active_neurons <= 8);
    assert!(metrics.active_neurons > 0);
}

#[test]
fn output_cache_matches_checkpoint_layout_and_rebuilds_after_mutation() {
    let mut model = Model::initialize(test_config()).unwrap();
    model.neurons.threshold.fill(0.1);
    model.input.branches.fill(BRANCH_EXCITATORY);
    model.input.weights.fill(1.0);

    let mut runtime = Runtime::new(model).unwrap();
    runtime.begin_document();
    runtime
        .step(BEGIN_DOCUMENT, None, Permission::Frozen)
        .unwrap();
    assert!(!runtime.active_neurons().is_empty());
    assert_probabilities_close(
        runtime.probabilities(),
        &manual_output_probabilities(&runtime),
    );

    let neuron_count = runtime.model().neuron_count();
    {
        let model = runtime.model_mut();
        for neuron in 0..neuron_count {
            model.output.weights[neuron] += 0.25;
        }
    }
    runtime.begin_document();
    runtime
        .step(BEGIN_DOCUMENT, None, Permission::Frozen)
        .unwrap();
    assert_probabilities_close(
        runtime.probabilities(),
        &manual_output_probabilities(&runtime),
    );
}
