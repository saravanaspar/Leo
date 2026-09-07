//! End-to-end training lifecycle. CLI parsing stays in `main.rs`; this module
//! owns dataset/run identity, resume transactions, validation cadence, early
//! stopping, checkpoint scheduling, and final best-model restoration.

use super::dataset::{open_dataset, shuffled_story_order, StoryBatchPrefetcher};
use super::resume::{
    advance_checkpoint_deadline, cleanup_training_resume_artifacts, commit_training_checkpoint,
    load_resume_best_model, load_training_resume_for_model,
    persist_training_resume_for_saved_model, resume_state_matches_model,
    save_training_resume_state, training_resume_path, validate_training_resume_identity,
    TrainingResumeState, TRAINING_CHECKPOINT_INTERVAL, TRAINING_CHECKPOINT_PREVIOUS_GENERATIONS,
};
use super::validation::{
    evaluate_model, meaningful_improvement, print_prediction_evaluation, update_best_validation,
};
use super::{configured_multi_gpu_devices, TrainingEngine, GPU_REFERENCE_WORKERS};
use crate::json_escape;
use leo_core::semantics::{EXECUTION_SEMANTICS_NAME, LEO_RELEASE_VERSION, TRAINING_POLICY_NAME};
use leo_core::{BackendKind, BackendRuntime, LeoError, LeoResult, Permission};
use leo_format::load_model;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TrainRequest<'a> {
    pub(crate) model_path: &'a str,
    pub(crate) train_bytes: &'a str,
    pub(crate) train_index: &'a str,
    pub(crate) validation_paths: Option<(&'a str, &'a str)>,
    pub(crate) fresh_run: bool,
    pub(crate) requested_backend: BackendKind,
    pub(crate) passes: Option<usize>,
    pub(crate) story_limit: Option<usize>,
    pub(crate) workers: Option<usize>,
    pub(crate) max_training_bytes: Option<u64>,
    pub(crate) validation_story_limit: Option<usize>,
    pub(crate) log_every: usize,
}

pub(crate) fn run_training(request: TrainRequest<'_>) -> LeoResult<()> {
    let model_path = request.model_path;
    if request.fresh_run {
        cleanup_training_resume_artifacts(model_path)?;
    }

    let dataset = open_dataset(request.train_bytes, request.train_index, "training")?;
    let resume_path = training_resume_path(model_path);
    let loaded_model = load_model(model_path)?;
    let existing_resume_state = load_training_resume_for_model(model_path, &loaded_model)?;
    let mut runtime = BackendRuntime::new(loaded_model, request.requested_backend)?;
    let backend = runtime.resolved_backend();
    let passes = request
        .passes
        .unwrap_or(runtime.model().config.training.max_dataset_passes);
    let story_limit = request
        .story_limit
        .unwrap_or(dataset.len())
        .min(dataset.len());
    let workers = request.workers.unwrap_or_else(|| {
        if backend == BackendKind::Gpu {
            GPU_REFERENCE_WORKERS
        } else {
            1
        }
    });
    let max_training_bytes = request.max_training_bytes;
    let multi_gpu_devices = configured_multi_gpu_devices(backend, workers.min(story_limit.max(1)))?;
    let multi_gpu_enabled = multi_gpu_devices > 1;
    if passes == 0 || story_limit == 0 || workers == 0 || max_training_bytes == Some(0) {
        return Err(LeoError::usage(
            "training requires at least one pass, one story, and one worker",
        ));
    }

    let validation_dataset = if let Some((valid_bytes, valid_index)) = request.validation_paths {
        Some(open_dataset(valid_bytes, valid_index, "validation")?)
    } else {
        None
    };
    let validation_dataset_id = validation_dataset
        .as_ref()
        .map(|dataset| dataset.identity().dataset_id);
    let validation_story_limit = request.validation_story_limit;
    let log_every = request.log_every.max(1);
    let seed = runtime.model().config.model.seed;
    let started = Instant::now();
    let mut stopped_early = false;
    let mut stopped_by_byte_limit = false;

    let synchronization = if backend == BackendKind::Gpu && workers > 1 {
        "gpu_story_mean_exact_v1"
    } else {
        "one_story_per_worker"
    };

    let train_dataset_id = dataset.identity().dataset_id;

    let resumed = existing_resume_state.is_some();
    let mut resume_state = if let Some(state) = existing_resume_state {
        validate_training_resume_identity(
            &state,
            train_dataset_id,
            dataset.len(),
            story_limit,
            passes,
            workers,
            backend,
            synchronization,
            max_training_bytes,
            validation_dataset_id,
            validation_story_limit,
        )?;
        state
    } else {
        TrainingResumeState {
            model_generation: runtime.model().generation,
            model_parameter_revision: runtime.model().parameter_revision,
            model_processed_stories: runtime.model().statistics.processed_stories,
            train_dataset_id,
            dataset_len: dataset.len(),
            story_limit,
            passes,
            workers,
            backend,
            synchronization: synchronization.to_owned(),
            max_training_bytes,
            validation_dataset_id,
            validation_story_limit,
            next_pass: 0,
            next_position: 0,
            presentations: 0,
            input_bytes_seen: 0,
            last_validation_processed_bytes: runtime.model().statistics.processed_bytes,
            patience_best: f64::INFINITY,
            stale_checks: 0,
            best_validation_loss: None,
            best_parameter_revision: None,
            best_checkpoint_digest: None,
            finalized: false,
        }
    };

    if resumed && resume_state.finalized {
        cleanup_training_resume_artifacts(model_path)?;
        println!(
            "{{\"event\":\"training_resume_cleanup\",\"generation\":{}}}",
            runtime.model().generation,
        );
        return Ok(());
    }

    let mut best_validation = load_resume_best_model(model_path, &resume_state)?;
    let mut patience_best = resume_state.patience_best;
    let mut stale_checks = resume_state.stale_checks;
    let mut presentations = resume_state.presentations;
    let mut input_bytes_seen = resume_state.input_bytes_seen;
    let mut last_validation = resume_state.last_validation_processed_bytes;
    let mut last_checkpoint_processed_bytes = runtime.model().statistics.processed_bytes;
    let mut next_periodic_checkpoint = TRAINING_CHECKPOINT_INTERVAL;
    let start_pass = resume_state.next_pass;
    let start_position = resume_state.next_position;

    if !resumed {
        save_training_resume_state(&resume_path, &resume_state)?;
    } else {
        println!(
            "{{\"event\":\"training_resumed\",\"pass\":{},\"story\":{},\"stories\":{},\"presentations\":{},\"input_bytes_seen\":{},\"generation\":{}}}",
            start_pass.saturating_add(1),
            start_position,
            story_limit,
            presentations,
            input_bytes_seen,
            runtime.model().generation,
        );
    }

    println!(
        "{{\"event\":\"training_start\",\"leo_version\":\"{}\",\"training_policy\":\"{}\",\"execution_semantics\":\"{}\",\"dataset_id\":\"{}\",\"requested_backend\":\"{}\",\"backend\":\"{}\",\"passes\":{},\"stories_per_pass\":{},\"workers\":{},\"synchronization\":\"{}\",\"gpu_devices\":{},\"multi_gpu_experimental\":{},\"multi_gpu_exact_data_parallel\":{},\"architecture\":\"fixed_sparse_recurrent_latent_context\",\"gpu_native_story_batch\":{},\"context_embedding_dim\":{},\"context_dropout_rate\":{},\"end_document_weight\":{},\"replay_fraction\":{},\"replay_segment_targets\":{}}}",
        LEO_RELEASE_VERSION,
        TRAINING_POLICY_NAME,
        EXECUTION_SEMANTICS_NAME,
        train_dataset_id,
        request.requested_backend,
        backend,
        passes,
        story_limit,
        workers,
        synchronization,
        multi_gpu_devices,
        multi_gpu_enabled,
        multi_gpu_enabled,
        backend == BackendKind::Gpu && workers > 1,
        runtime.model().config.context.embedding_dim,
        runtime.model().config.context.dropout_rate,
        runtime.model().config.learning.end_document_weight,
        runtime.model().config.replay.fraction,
        runtime.model().config.replay.segment_bytes,
    );
    println!(
        "{{\"event\":\"checkpoint_policy\",\"periodic_seconds\":{},\"retained_model_copies\":{},\"byte_safety_interval\":{},\"resume_state\":\"{}\"}}",
        TRAINING_CHECKPOINT_INTERVAL.as_secs(),
        TRAINING_CHECKPOINT_PREVIOUS_GENERATIONS + 1,
        runtime.model().config.training.checkpoint_every_bytes,
        json_escape(&resume_path.to_string_lossy()),
    );

    if multi_gpu_enabled {
        eprintln!(
            "{{\"event\":\"multi_gpu_training\",\"devices\":{},\"semantics\":\"flat_story_delta_mean\",\"exact_logical_batch_mean\":true,\"replay_parallel\":false}}",
            multi_gpu_devices,
        );
    }
    let mut training_engine = TrainingEngine::new(runtime.model(), multi_gpu_devices)?;

    'passes: for pass in start_pass..passes {
        let order = shuffled_story_order(story_limit, seed, pass);
        let mut position = if pass == start_pass {
            start_position
        } else {
            0
        };
        let prefetcher = StoryBatchPrefetcher::spawn(&dataset, order, "training")?;
        if position < story_limit {
            let remaining_bytes =
                max_training_bytes.map(|limit| limit.saturating_sub(input_bytes_seen));
            prefetcher.request(position, workers, remaining_bytes)?;
        }
        while position < story_limit {
            let (stories, next_position, batch_input_bytes) = prefetcher.receive()?;
            if stories.is_empty() {
                stopped_by_byte_limit = max_training_bytes.is_some();
                break 'passes;
            }
            let next_input_bytes_seen = input_bytes_seen.saturating_add(batch_input_bytes);
            if next_position < story_limit {
                let remaining_bytes =
                    max_training_bytes.map(|limit| limit.saturating_sub(next_input_bytes_seen));
                prefetcher.request(next_position, workers, remaining_bytes)?;
            }
            input_bytes_seen = next_input_bytes_seen;
            let batch_started = Instant::now();
            let report =
                training_engine.train_batch(&mut runtime, stories, Permission::Training)?;
            let batch_seconds = batch_started.elapsed().as_secs_f64();
            let previous_presentations = presentations;
            presentations = presentations.saturating_add(report.stories);
            position = next_position;

            resume_state.next_pass = pass;
            resume_state.next_position = position;
            resume_state.presentations = presentations;
            resume_state.input_bytes_seen = input_bytes_seen;

            let crossed_log_boundary =
                previous_presentations / log_every != presentations / log_every;
            if crossed_log_boundary || position == story_limit {
                println!(
                    "{{\"event\":\"training_progress\",\"pass\":{},\"story\":{},\"stories\":{},\"presentations\":{},\"workers\":{},\"loss\":{},\"bits_per_byte\":{},\"active_fraction\":{},\"recurrent_events_per_step\":{},\"context_cells_per_step\":{},\"context_probes_per_step\":{},\"context_use_fraction\":{},\"output_madds_per_step\":{},\"context_gain_bits_per_step\":{},\"steps_per_second\":{},\"execution_steps_per_second\":{},\"replay_segments\":{},\"replay_steps\":{},\"replay_prefix_steps\":{},\"replay_execution_steps\":{},\"merged_fixed_updates\":{},\"merged_context_keys\":{},\"elapsed_seconds\":{}}}",
                    pass + 1,
                    position,
                    story_limit,
                    presentations,
                    report.merge.workers,
                    report.mean_loss,
                    report.mean_loss / std::f64::consts::LN_2,
                    report.activity.mean_active(runtime.model().neuron_count()),
                    report.activity.events_per_step(),
                    report.activity.context_cells_per_step(),
                    report.activity.context_probes_per_step(),
                    report.activity.context_use_fraction(),
                    report.activity.output_madds_per_step(),
                    report.activity.context_gain_bits_per_step(),
                    report.activity.steps as f64 / batch_seconds.max(1.0e-9),
                    report.activity.steps.saturating_add(report.replay_prefix_steps) as f64
                        / batch_seconds.max(1.0e-9),
                    report.replay_segments,
                    report.replay_steps,
                    report.replay_prefix_steps,
                    report.replay_steps.saturating_add(report.replay_prefix_steps),
                    report.merge.fixed_parameter_updates,
                    report.merge.context_keys,
                    started.elapsed().as_secs_f64(),
                );
            }

            let processed_bytes = runtime.model().statistics.processed_bytes;
            let elapsed = started.elapsed();
            let byte_checkpoint_due = processed_bytes
                .saturating_sub(last_checkpoint_processed_bytes)
                >= runtime.model().config.training.checkpoint_every_bytes;
            let periodic_checkpoint_due = elapsed >= next_periodic_checkpoint;
            let mut checkpoint_saved_this_batch = false;
            if byte_checkpoint_due || periodic_checkpoint_due {
                commit_training_checkpoint(
                    model_path,
                    &mut runtime,
                    &mut resume_state,
                    &best_validation,
                )?;
                checkpoint_saved_this_batch = true;
                last_checkpoint_processed_bytes = processed_bytes;
                if periodic_checkpoint_due {
                    next_periodic_checkpoint = advance_checkpoint_deadline(
                        next_periodic_checkpoint,
                        elapsed,
                        TRAINING_CHECKPOINT_INTERVAL,
                    );
                }
                println!(
                    "{{\"event\":\"checkpoint_saved\",\"reason\":\"{}\",\"generation\":{},\"processed_bytes\":{},\"retained_model_copies\":{},\"resume_pass\":{},\"resume_story\":{},\"elapsed_seconds\":{}}}",
                    if periodic_checkpoint_due { "periodic" } else { "byte_safety" },
                    runtime.model().generation,
                    processed_bytes,
                    TRAINING_CHECKPOINT_PREVIOUS_GENERATIONS + 1,
                    pass + 1,
                    position,
                    elapsed.as_secs_f64(),
                );
            }

            if let Some(validation_dataset) = validation_dataset.as_ref() {
                if processed_bytes.saturating_sub(last_validation)
                    >= runtime.model().config.training.validate_every_bytes
                {
                    runtime.synchronize_model()?;
                    let validation = evaluate_model(
                        runtime.model().clone(),
                        validation_dataset,
                        validation_story_limit,
                        backend,
                    )?;
                    update_best_validation(runtime.model(), &validation, &mut best_validation);
                    print_prediction_evaluation("validation", &validation);
                    last_validation = processed_bytes;
                    resume_state.last_validation_processed_bytes = last_validation;
                    if checkpoint_saved_this_batch
                        && resume_state_matches_model(&resume_state, runtime.model())
                    {
                        persist_training_resume_for_saved_model(
                            model_path,
                            runtime.model(),
                            &mut resume_state,
                            &best_validation,
                        )?;
                    }
                }
            }

            if max_training_bytes.is_some_and(|limit| input_bytes_seen >= limit) {
                stopped_by_byte_limit = true;
                break 'passes;
            }
        }

        if let Some(validation_dataset) = validation_dataset.as_ref() {
            runtime.synchronize_model()?;
            let validation = evaluate_model(
                runtime.model().clone(),
                validation_dataset,
                validation_story_limit,
                backend,
            )?;
            update_best_validation(runtime.model(), &validation, &mut best_validation);
            print_prediction_evaluation("pass_validation", &validation);
            last_validation = runtime.model().statistics.processed_bytes;
            resume_state.last_validation_processed_bytes = last_validation;

            let required_improvement =
                runtime.model().config.training.minimum_relative_improvement as f64;
            if meaningful_improvement(patience_best, validation.loss, required_improvement) {
                patience_best = validation.loss;
                stale_checks = 0;
            } else {
                stale_checks = stale_checks.saturating_add(1);
            }
            resume_state.patience_best = patience_best;
            resume_state.stale_checks = stale_checks;
            if stale_checks >= runtime.model().config.training.early_stop_checks {
                stopped_early = true;
                break 'passes;
            }
        }

        resume_state.next_pass = pass.saturating_add(1);
        resume_state.next_position = 0;
        resume_state.presentations = presentations;
        resume_state.input_bytes_seen = input_bytes_seen;
        resume_state.last_validation_processed_bytes = last_validation;
        resume_state.patience_best = patience_best;
        resume_state.stale_checks = stale_checks;
        commit_training_checkpoint(
            model_path,
            &mut runtime,
            &mut resume_state,
            &best_validation,
        )?;
        last_checkpoint_processed_bytes = runtime.model().statistics.processed_bytes;
    }

    if let Some((_, mut best_model)) = best_validation.take() {
        let restored_parameter_revision = best_model.parameter_revision;
        best_model.generation = runtime.model().generation;
        // Restoring an older best parameter state is itself a new canonical
        // state transition. Keep the revision monotonic instead of assigning
        // the later trajectory's revision to older bytes.
        best_model.parameter_revision = runtime.model().parameter_revision.saturating_add(1);
        best_model.statistics = runtime.model().statistics.clone();
        eprintln!(
            "{{\"event\":\"best_model_restored\",\"source_parameter_revision\":{},\"new_parameter_revision\":{}}}",
            restored_parameter_revision,
            best_model.parameter_revision,
        );
        runtime = BackendRuntime::new(best_model, backend)?;
    }
    resume_state.finalized = true;
    resume_state.model_parameter_revision = runtime.model().parameter_revision;
    resume_state.model_processed_stories = runtime.model().statistics.processed_stories;
    let no_best_validation = None;
    commit_training_checkpoint(
        model_path,
        &mut runtime,
        &mut resume_state,
        &no_best_validation,
    )?;
    cleanup_training_resume_artifacts(model_path)?;

    println!(
        "{{\"event\":\"training_complete\",\"backend\":\"{}\",\"generation\":{},\"parameter_revision\":{},\"presentations\":{},\"input_bytes_seen\":{},\"processed_bytes\":{},\"processed_stories\":{},\"training_loss\":{},\"training_bits_per_byte\":{},\"active_fraction\":{},\"fixed_synapses\":{},\"context_slots\":{},\"context_embedding_dim\":{},\"stopped_early\":{},\"stopped_by_byte_limit\":{},\"elapsed_seconds\":{}}}",
        backend,
        runtime.model().generation,
        runtime.model().parameter_revision,
        presentations,
        input_bytes_seen,
        runtime.model().statistics.processed_bytes,
        runtime.model().statistics.processed_stories,
        runtime.model().statistics.mean_training_loss(),
        runtime.model().statistics.bits_per_byte(),
        runtime.model().statistics.mean_active_neurons_per_tick()
            / runtime.model().neuron_count() as f64,
        runtime.model().recurrent.weight.len(),
        runtime.model().context.observations.len(),
        runtime.model().config.context.embedding_dim,
        stopped_early,
        stopped_by_byte_limit,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}
