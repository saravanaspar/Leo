//! Frozen validation/evaluation services used by training and `leo eval`.

use super::metrics::{ActivityDiagnostics, EvaluationMetrics};
use leo_core::symbols::{BEGIN_DOCUMENT, END_DOCUMENT};
use leo_core::{available_gpu_devices, BackendKind, BackendRuntime, LeoError, LeoResult, Model, Permission, StepMetrics};
use leo_data::PreparedDataset;
use std::thread;

pub(crate) fn evaluate_model(
    model: Model,
    dataset: &PreparedDataset,
    story_limit: Option<usize>,
    backend: BackendKind,
) -> LeoResult<EvaluationMetrics> {
    let limit = story_limit.unwrap_or(dataset.len()).min(dataset.len());

    let gpu_devices = if backend == BackendKind::Gpu {
        available_gpu_devices()
            .unwrap_or(1)
            .max(1)
            .min(limit.max(1))
    } else {
        1
    };

    if backend == BackendKind::Gpu && gpu_devices > 1 && limit > 1 {
        let mut runtimes = Vec::with_capacity(gpu_devices);
        for device_index in 0..gpu_devices {
            runtimes.push(BackendRuntime::new_gpu_on_device(
                model.clone(),
                device_index,
            )?);
        }

        let mut handles = Vec::with_capacity(gpu_devices);
        for (shard_index, runtime) in runtimes.into_iter().enumerate() {
            let shard_dataset = dataset.try_clone().map_err(|error| {
                LeoError::dataset(format!("cannot clone verified evaluation dataset: {error}"))
            })?;
            handles.push(thread::spawn(move || {
                evaluate_model_shard(
                    runtime,
                    shard_dataset,
                    limit,
                    shard_index,
                    gpu_devices,
                )
            }));
        }

        let mut combined = EvaluationMetrics {
            neuron_count: model.neuron_count(),
            ..EvaluationMetrics::default()
        };
        for handle in handles {
            let shard = handle
                .join()
                .map_err(|_| LeoError::internal("evaluation GPU worker panicked"))??;
            combined.merge(shard);
        }
        return Ok(combined);
    }

    evaluate_model_shard(
        BackendRuntime::new(model, backend)?,
        dataset.try_clone().map_err(|error| {
            LeoError::dataset(format!("cannot clone verified evaluation dataset: {error}"))
        })?,
        limit,
        0,
        1,
    )
}

fn evaluate_model_shard(
    mut runtime: BackendRuntime,
    mut dataset: PreparedDataset,
    limit: usize,
    shard_index: usize,
    shard_count: usize,
) -> LeoResult<EvaluationMetrics> {
    let neuron_count = runtime.model().neuron_count();
    let mut loss_sum = 0.0f64;
    let mut targets = 0u64;
    let mut correct = 0u64;
    let mut processed_bytes = 0u64;
    let mut activity = ActivityDiagnostics::default();
    let mut processed_stories = 0u64;

    for story_index in (shard_index..limit).step_by(shard_count) {
        let story = dataset
            .story(story_index)
            .map_err(|error| LeoError::dataset(format!("dataset error: {error}")))?;
        if story.is_empty() {
            continue;
        }

        let mut steps = Vec::with_capacity(story.len() + 1);
        steps.push((BEGIN_DOCUMENT, Some(story[0] as u32)));
        steps.extend(
            story
                .windows(2)
                .map(|window| (window[0] as u32, Some(window[1] as u32))),
        );
        steps.push((
            *story.last().expect("nonempty story") as u32,
            Some(END_DOCUMENT),
        ));

        runtime.begin_document()?;
        let metrics = runtime.training_step_batch(&steps, Permission::Frozen)?;
        for (step_metrics, (_, target)) in metrics.into_iter().zip(&steps) {
            record_evaluation_step(
                step_metrics,
                target.expect("evaluation steps always have targets"),
                &mut loss_sum,
                &mut targets,
                &mut correct,
                &mut activity,
            );
        }
        runtime.finish_document()?;

        processed_bytes = processed_bytes.saturating_add(story.len() as u64);
        processed_stories = processed_stories.saturating_add(1);
        if processed_stories % 25 == 0 {
            eprintln!(
                "{{\"event\":\"evaluation_progress\",\"shard\":{},\"shards\":{},\"stories\":{}}}",
                shard_index, shard_count, processed_stories,
            );
        }
    }

    let mean_loss = loss_sum / targets.max(1) as f64;
    Ok(EvaluationMetrics {
        loss: mean_loss,
        bits_per_byte: mean_loss / std::f64::consts::LN_2,
        accuracy: correct as f64 / targets.max(1) as f64,
        targets,
        processed_stories,
        processed_bytes,
        neuron_count,
        activity,
    })
}

fn record_evaluation_step(
    metrics: StepMetrics,
    target: u32,
    loss_sum: &mut f64,
    targets: &mut u64,
    correct: &mut u64,
    activity: &mut ActivityDiagnostics,
) {
    if let Some(loss) = metrics.loss {
        *loss_sum += loss as f64;
        *targets = (*targets).saturating_add(1);
        if metrics.predicted_symbol == target {
            *correct = (*correct).saturating_add(1);
        }
    }
    activity.record(metrics);
}


pub(crate) fn print_prediction_evaluation(event: &str, metrics: &EvaluationMetrics) {
    let context_gain = metrics
        .activity
        .context_gain_bits_per_byte(metrics.processed_bytes);
    println!(
        "{{\"event\":\"{}\",\"loss\":{},\"bits_per_byte\":{},\"neural_only_bits_per_byte\":{},\"accuracy\":{},\"targets\":{},\"stories\":{},\"bytes\":{},\"active_fraction\":{},\"active_neurons_mean\":{},\"active_neurons_peak\":{},\"suprathreshold_mean\":{},\"block_selected_mean\":{},\"cap_hit_fraction\":{},\"population_inhibition_mean\":{},\"population_inhibition_peak\":{},\"recurrent_events_per_byte\":{},\"context_cells_per_byte\":{},\"context_probes_per_byte\":{},\"context_use_fraction\":{},\"output_madds_per_byte\":{},\"context_gain_bits_per_byte\":{},\"eligibility_mean\":{}}}",
        event,
        metrics.loss,
        metrics.bits_per_byte,
        metrics.bits_per_byte + context_gain,
        metrics.accuracy,
        metrics.targets,
        metrics.processed_stories,
        metrics.processed_bytes,
        metrics.activity.mean_active(metrics.neuron_count),
        metrics.activity.mean_active_count(),
        metrics.activity.active_neurons_peak,
        metrics.activity.mean_suprathreshold(),
        metrics.activity.mean_block_selected(),
        metrics.activity.cap_hit_fraction(),
        metrics.activity.mean_population_inhibition(),
        metrics.activity.population_inhibition_peak,
        metrics.activity.emitted_events as f64 / metrics.processed_bytes.max(1) as f64,
        metrics.activity.context_cells_per_byte(metrics.processed_bytes),
        metrics.activity.context_probes_per_byte(metrics.processed_bytes),
        metrics.activity.context_use_fraction(),
        metrics.activity.output_madds_per_byte(metrics.processed_bytes),
        context_gain,
        metrics.activity.mean_eligibility(),
    );
}

pub(crate) fn print_learning_quality(training: &EvaluationMetrics, held_out: &EvaluationMetrics) {
    println!(
        "{{\"event\":\"learning_quality\",\"frozen_training_bits_per_byte\":{},\"held_out_bits_per_byte\":{},\"generalization_gap\":{},\"frozen_training_accuracy\":{},\"held_out_accuracy\":{},\"held_out_context_gain_bits_per_byte\":{}}}",
        training.bits_per_byte,
        held_out.bits_per_byte,
        held_out.bits_per_byte - training.bits_per_byte,
        training.accuracy,
        held_out.accuracy,
        held_out.activity.context_gain_bits_per_byte(held_out.processed_bytes),
    );
}


pub(crate) fn update_best_validation(
    current_model: &Model,
    validation: &EvaluationMetrics,
    best: &mut Option<(f64, Model)>,
) {
    if best
        .as_ref()
        .map(|(best_loss, _)| validation.loss < *best_loss)
        .unwrap_or(true)
    {
        *best = Some((validation.loss, current_model.clone()));
    }
}

pub(crate) fn meaningful_improvement(previous: f64, current: f64, minimum_relative: f64) -> bool {
    if !previous.is_finite() {
        return true;
    }
    current < previous && (previous - current) / previous.abs().max(1.0e-12) >= minimum_relative
}

