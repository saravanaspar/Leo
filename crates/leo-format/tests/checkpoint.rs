use leo_core::symbols::BEGIN_DOCUMENT;
use leo_core::{Config, Model, Permission, Runtime};
use leo_format::{checkpoint_hash, commit_model_rolling, load_model, save_model_atomic};
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn checkpoint_round_trip_preserves_tensors() {
    let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
    let mut model = Model::initialize(config).unwrap();
    model.parameter_revision = 23;
    model.statistics.persistent_ticks = 17;
    model.statistics.active_neurons_sum = 51;
    model.input.weights[0] = 0.75;
    model.context.keys[0] = 42;
    model.context.embeddings[0] = 0.5;
    model.context.output_weights[0] = -0.25;
    model.context.observations[0] = 7;
    let path = temporary_path("roundtrip.pscls");
    save_model_atomic(&path, &model, false).unwrap();
    let before = checkpoint_hash(&path).unwrap();
    let loaded = load_model(&path).unwrap();
    assert_eq!(loaded.generation, model.generation);
    assert_eq!(loaded.parameter_revision, 23);
    assert_eq!(
        loaded.recurrent.target_neuron,
        model.recurrent.target_neuron
    );
    assert_eq!(loaded.output.weights, model.output.weights);
    assert_eq!(loaded.input.weights, model.input.weights);
    assert_eq!(loaded.context.keys, model.context.keys);
    assert_eq!(loaded.context.embeddings, model.context.embeddings);
    assert_eq!(loaded.context.output_weights, model.context.output_weights);
    assert_eq!(loaded.context.observations, model.context.observations);
    assert_eq!(loaded.statistics.persistent_ticks, 17);
    assert_eq!(loaded.statistics.mean_active_neurons_per_tick(), 3.0);
    assert_eq!(checkpoint_hash(&path).unwrap(), before);
    fs::remove_file(path).unwrap();
}

#[test]
fn rolling_checkpoint_keeps_only_current_and_previous_model() {
    let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
    let mut model = Model::initialize(config).unwrap();
    let path = temporary_path("rolling.pscls");
    save_model_atomic(&path, &model, false).unwrap();

    for revision in 1..=4 {
        model.parameter_revision = revision;
        commit_model_rolling(&path, &mut model, 1).unwrap();
    }

    assert_eq!(load_model(&path).unwrap().generation, 4);
    assert_eq!(load_model(&path).unwrap().parameter_revision, 4);
    assert!(!generation_path_for_test(&path, 0).exists());
    assert!(!generation_path_for_test(&path, 1).exists());
    assert!(!generation_path_for_test(&path, 2).exists());
    assert!(generation_path_for_test(&path, 3).exists());

    fs::remove_file(generation_path_for_test(&path, 3)).unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn checkpoint_excludes_worker_local_runtime_state() {
    let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
    let mut model = Model::initialize(config).unwrap();
    model.neurons.threshold.fill(0.1);
    model.input.weights.fill(1.0);
    let mut runtime = Runtime::new(model).unwrap();
    runtime.begin_document();
    runtime
        .step(BEGIN_DOCUMENT, None, Permission::Frozen)
        .unwrap();
    assert!(!runtime.active_neurons().is_empty());

    let path = temporary_path("worker-state.pscls");
    save_model_atomic(&path, runtime.model(), false).unwrap();
    let loaded = load_model(&path).unwrap();
    let resumed = Runtime::new(loaded).unwrap();
    assert!(resumed.active_neurons().is_empty());
    assert!((0..resumed.model().neuron_count()).all(|neuron| resumed.activation(neuron) == 0.0));
    fs::remove_file(path).unwrap();
}

#[test]
fn corrupted_checkpoint_is_rejected() {
    let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
    let model = Model::initialize(config).unwrap();
    let path = temporary_path("corrupt.pscls");
    save_model_atomic(&path, &model, false).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[32] ^= 0xFF;
    fs::write(&path, bytes).unwrap();
    assert!(load_model(&path).is_err());
    fs::remove_file(path).unwrap();
}

fn generation_path_for_test(path: &Path, generation: u64) -> PathBuf {
    PathBuf::from(format!("{}.gen{:06}", path.display(), generation))
}

fn temporary_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("leo-{}-{name}", std::process::id()))
}
