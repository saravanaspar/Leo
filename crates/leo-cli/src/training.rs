mod dataset;
mod lifecycle;
mod metrics;
mod resume;
mod validation;

pub(crate) use dataset::{open_dataset, StoryBatchPrefetcher};
#[cfg(test)]
pub(crate) use dataset::{read_story_batch, shuffled_story_order};
pub(crate) use lifecycle::{run_training, TrainRequest};
pub(crate) use metrics::ActivityDiagnostics;
#[cfg(test)]
pub(crate) use resume::{advance_checkpoint_deadline, TrainingResumeState};
pub(crate) use validation::{evaluate_model, print_learning_quality, print_prediction_evaluation};

use leo_core::symbols::{BEGIN_DOCUMENT, END_DOCUMENT};
use leo_core::{
    apply_mean_deltas, available_gpu_devices, merged_parameter_changes, BackendKind,
    BackendRuntime, LeoError, LeoResult, MergeMetrics, Model, ParameterChanges, Permission,
    SparseModelDelta, StepMetrics,
};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct ReplayDebugOptions {
    summary: bool,
    selection: bool,
    segments: bool,
    ranges: bool,
    timing: bool,
    segment_stride: usize,
}

impl ReplayDebugOptions {
    fn from_env() -> Self {
        let flag = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| {
                    let value = value.trim().to_ascii_lowercase();
                    !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "off" | "no")
                })
                .unwrap_or(false)
        };
        let general = flag("LEO_REPLAY_DEBUG");
        let timing_only = flag("LEO_REPLAY_DEBUG_TIMING");
        let ranges = flag("LEO_REPLAY_DEBUG_RANGES");
        let segments = flag("LEO_REPLAY_DEBUG_SEGMENTS");
        Self {
            summary: general || timing_only,
            selection: general || flag("LEO_REPLAY_DEBUG_SELECTION") || ranges,
            segments,
            ranges,
            timing: general || timing_only || segments,
            segment_stride: std::env::var("LEO_REPLAY_DEBUG_SEGMENT_STRIDE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(1),
        }
    }
}

fn replay_debug_options() -> ReplayDebugOptions {
    static OPTIONS: OnceLock<ReplayDebugOptions> = OnceLock::new();
    *OPTIONS.get_or_init(ReplayDebugOptions::from_env)
}

#[derive(Debug, Clone, Copy, Default)]
struct ReplayTiming {
    begin_ms: f64,
    prefix_build_ms: f64,
    prefix_execute_ms: f64,
    target_build_ms: f64,
    target_execute_ms: f64,
    reset_ms: f64,
    total_ms: f64,
}

impl ReplayTiming {
    fn add(&mut self, other: Self) {
        self.begin_ms += other.begin_ms;
        self.prefix_build_ms += other.prefix_build_ms;
        self.prefix_execute_ms += other.prefix_execute_ms;
        self.target_build_ms += other.target_build_ms;
        self.target_execute_ms += other.target_execute_ms;
        self.reset_ms += other.reset_ms;
        self.total_ms += other.total_ms;
    }
}

#[derive(Debug)]
struct ReplayRangeExecution {
    activity: ActivityDiagnostics,
    prefix_steps: u64,
    target_steps: u64,
    timing: ReplayTiming,
}

fn apply_replay_policy(
    runtime: &mut BackendRuntime,
    story: &[u8],
    losses: &[f32],
    permission: Permission,
    story_index: usize,
) -> LeoResult<ReplayTrainingReport> {
    let debug = replay_debug_options();
    let selection_started = (debug.timing || debug.selection).then(Instant::now);
    let fraction = runtime.model().config.replay.fraction;
    let segment_targets = runtime.model().config.replay.segment_bytes;
    let ranges = select_replay_ranges(losses, fraction, segment_targets);
    let selection_ms = selection_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let target_budget = if losses.is_empty() || fraction <= 0.0 || !fraction.is_finite() {
        0usize
    } else {
        ((losses.len() as f64) * (fraction.min(1.0) as f64))
            .ceil()
            .max(1.0) as usize
    };
    let selected_targets = ranges.iter().map(|range| range.len()).sum::<usize>();
    let estimated_prefix_steps = ranges.iter().map(|range| range.start as u64).sum::<u64>();

    if debug.selection {
        let finite = losses
            .iter()
            .copied()
            .filter(|loss| loss.is_finite())
            .collect::<Vec<_>>();
        let positive = finite
            .iter()
            .copied()
            .filter(|loss| *loss > 0.0)
            .collect::<Vec<_>>();
        let loss_mean = if finite.is_empty() {
            0.0
        } else {
            finite.iter().map(|loss| *loss as f64).sum::<f64>() / finite.len() as f64
        };
        let loss_max = finite.iter().copied().max_by(f32::total_cmp).unwrap_or(0.0);
        let positive_loss_min = positive
            .iter()
            .copied()
            .min_by(f32::total_cmp)
            .unwrap_or(0.0);
        let selected_losses = ranges
            .iter()
            .flat_map(|range| range.clone())
            .filter_map(|index| losses.get(index).copied())
            .filter(|loss| loss.is_finite())
            .collect::<Vec<_>>();
        let selected_loss_mean = if selected_losses.is_empty() {
            0.0
        } else {
            selected_losses.iter().map(|loss| *loss as f64).sum::<f64>()
                / selected_losses.len() as f64
        };
        let prefix_amplification = estimated_prefix_steps as f64 / selected_targets.max(1) as f64;
        eprintln!(
            "{{\"event\":\"replay_selection_debug\",\"story_index\":{},\"story_bytes\":{},\"loss_targets\":{},\"finite_losses\":{},\"positive_losses\":{},\"loss_mean\":{},\"loss_max\":{},\"positive_loss_min\":{},\"selected_loss_mean\":{},\"fraction\":{},\"segment_targets\":{},\"target_budget\":{},\"selected_segments\":{},\"selected_targets\":{},\"estimated_prefix_steps\":{},\"prefix_amplification\":{},\"selection_ms\":{}}}",
            story_index,
            story.len(),
            losses.len(),
            finite.len(),
            positive.len(),
            loss_mean,
            loss_max,
            positive_loss_min,
            selected_loss_mean,
            fraction,
            segment_targets,
            target_budget,
            ranges.len(),
            selected_targets,
            estimated_prefix_steps,
            prefix_amplification,
            selection_ms,
        );
        if debug.ranges {
            for (index, range) in ranges.iter().enumerate() {
                eprintln!(
                    "{{\"event\":\"replay_range_debug\",\"story_index\":{},\"segment_index\":{},\"start\":{},\"end\":{},\"targets\":{},\"estimated_prefix_steps\":{}}}",
                    story_index,
                    index,
                    range.start,
                    range.end,
                    range.len(),
                    range.start,
                );
            }
        }
    }

    let story_started = debug.timing.then(Instant::now);
    let mut activity = ActivityDiagnostics::default();
    let mut replay_steps = 0u64;
    let mut prefix_steps = 0u64;
    let mut timing = ReplayTiming::default();
    for (index, range) in ranges.iter().enumerate() {
        let cleanup_after = index + 1 == ranges.len();
        let execution = replay_target_range_impl(
            runtime,
            story,
            range.clone(),
            permission,
            cleanup_after,
            debug.timing,
        )?;
        replay_steps = replay_steps.saturating_add(execution.target_steps);
        prefix_steps = prefix_steps.saturating_add(execution.prefix_steps);
        timing.add(execution.timing);
        activity.add(execution.activity);

        if debug.segments && index % debug.segment_stride == 0 {
            eprintln!(
                "{{\"event\":\"replay_segment_debug\",\"story_index\":{},\"segment_index\":{},\"segment_count\":{},\"start\":{},\"end\":{},\"target_steps\":{},\"prefix_steps\":{},\"cleanup_after\":{},\"begin_ms\":{},\"prefix_build_ms\":{},\"prefix_execute_ms\":{},\"target_build_ms\":{},\"target_execute_ms\":{},\"reset_ms\":{},\"total_ms\":{}}}",
                story_index,
                index,
                ranges.len(),
                range.start,
                range.end,
                execution.target_steps,
                execution.prefix_steps,
                cleanup_after,
                execution.timing.begin_ms,
                execution.timing.prefix_build_ms,
                execution.timing.prefix_execute_ms,
                execution.timing.target_build_ms,
                execution.timing.target_execute_ms,
                execution.timing.reset_ms,
                execution.timing.total_ms,
            );
        }
    }
    if let Some(started) = story_started {
        timing.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    if debug.summary {
        eprintln!(
            "{{\"event\":\"replay_story_debug\",\"story_index\":{},\"story_bytes\":{},\"segments\":{},\"target_steps\":{},\"prefix_steps\":{},\"execution_steps\":{},\"selection_ms\":{},\"begin_ms\":{},\"prefix_build_ms\":{},\"prefix_execute_ms\":{},\"target_build_ms\":{},\"target_execute_ms\":{},\"reset_ms\":{},\"total_ms\":{},\"prefix_steps_per_second\":{},\"target_steps_per_second\":{}}}",
            story_index,
            story.len(),
            ranges.len(),
            replay_steps,
            prefix_steps,
            replay_steps.saturating_add(prefix_steps),
            selection_ms,
            timing.begin_ms,
            timing.prefix_build_ms,
            timing.prefix_execute_ms,
            timing.target_build_ms,
            timing.target_execute_ms,
            timing.reset_ms,
            timing.total_ms,
            if timing.prefix_execute_ms > 0.0 { prefix_steps as f64 / (timing.prefix_execute_ms / 1000.0) } else { 0.0 },
            if timing.target_execute_ms > 0.0 { replay_steps as f64 / (timing.target_execute_ms / 1000.0) } else { 0.0 },
        );
    }
    Ok(ReplayTrainingReport {
        activity,
        replay_segments: ranges.len(),
        replay_steps,
        prefix_steps,
        timing,
        selection_ms,
    })
}

#[derive(Debug)]
struct ReplayTrainingReport {
    pub(crate) activity: ActivityDiagnostics,
    replay_segments: usize,
    pub(crate) replay_steps: u64,
    pub(crate) prefix_steps: u64,
    timing: ReplayTiming,
    selection_ms: f64,
}

#[derive(Debug)]
pub(crate) struct DocumentTrainingPass {
    pub(crate) mean_loss: f64,
    losses: Vec<f32>,
    activity: ActivityDiagnostics,
}

#[derive(Debug)]
pub(crate) struct BatchTrainingReport {
    pub(crate) mean_loss: f64,
    pub(crate) targets: usize,
    pub(crate) stories: usize,
    pub(crate) activity: ActivityDiagnostics,
    pub(crate) replay_segments: usize,
    pub(crate) replay_steps: u64,
    pub(crate) replay_prefix_steps: u64,
    pub(crate) merge: MergeMetrics,
}

/// Deep training boundary used by the CLI. It owns execution coordination;
/// the canonical `BackendRuntime` remains explicit because checkpointing and
/// evaluation synchronize against that same model owner.
pub(crate) fn configured_multi_gpu_devices(
    backend: BackendKind,
    workers: usize,
) -> LeoResult<usize> {
    if backend != BackendKind::Gpu || workers == 0 {
        return Ok(1);
    }
    let requested = std::env::var("LEO_MULTI_GPU")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !requested || workers == 1 {
        return Ok(1);
    }

    let visible = available_gpu_devices()?;
    if visible < 2 {
        return Err(LeoError::backend(
            "LEO_MULTI_GPU requested, but fewer than two CUDA devices are visible",
        ));
    }
    Ok(visible.min(workers))
}

pub(crate) struct TrainingEngine {
    multi_gpu: Option<MultiGpuBatchTrainer>,
}

impl TrainingEngine {
    pub(crate) fn new(model: &Model, multi_gpu_devices: usize) -> LeoResult<Self> {
        let multi_gpu = if multi_gpu_devices > 1 {
            Some(MultiGpuBatchTrainer::new(model.clone(), multi_gpu_devices)?)
        } else {
            None
        };
        Ok(Self { multi_gpu })
    }

    pub(crate) fn train_batch(
        &mut self,
        canonical: &mut BackendRuntime,
        stories: Vec<Vec<u8>>,
        permission: Permission,
    ) -> LeoResult<BatchTrainingReport> {
        match self.multi_gpu.as_mut() {
            Some(trainer) => trainer.train_story_batch(canonical, stories, permission),
            None => train_story_batch(canonical, stories, permission),
        }
    }
}

// Multi-GPU replicas remain resident on their CUDA devices. Each device owns a
// contiguous subset of the logical story workers, but the canonical reducer
// always receives one delta per original story in original story order. This is
// Leo's data-parallel axis: device placement changes execution only; workers,
// denominator, and the canonical mean barrier do not change.
struct MultiGpuBatchTrainer {
    // Device 0 is the existing canonical CUDA runtime. Only devices 1..N-1
    // need extra full-model replicas, so a model that fits on one GPU does not
    // require two copies on GPU 0 merely to enable data parallelism.
    replicas: Vec<BackendRuntime>,
    device_count: usize,
}

struct MultiGpuShardResult {
    story_start: usize,
    story_losses: Vec<Vec<f32>>,
    activity: ActivityDiagnostics,
    story_deltas: Vec<SparseModelDelta>,
    story_changes: Vec<ParameterChanges>,
}

fn balanced_story_range(total: usize, shards: usize, shard_index: usize) -> std::ops::Range<usize> {
    let base = total / shards;
    let extra = total % shards;
    let start = shard_index * base + shard_index.min(extra);
    let len = base + usize::from(shard_index < extra);
    start..start + len
}

fn aggregate_device_story_metrics(
    story_metrics: Vec<Vec<StepMetrics>>,
) -> (Vec<Vec<f32>>, ActivityDiagnostics) {
    let mut story_losses = Vec::with_capacity(story_metrics.len());
    let mut activity = ActivityDiagnostics::default();

    for metrics_for_story in story_metrics {
        let mut losses = Vec::with_capacity(metrics_for_story.len());
        for metrics in metrics_for_story {
            if let Some(loss) = metrics.loss {
                losses.push(loss);
            }
            activity.record(metrics);
        }
        story_losses.push(losses);
    }

    (story_losses, activity)
}

fn run_multi_gpu_shard(
    worker: &mut BackendRuntime,
    shard: &[Vec<u8>],
    story_start: usize,
    permission: Permission,
) -> LeoResult<MultiGpuShardResult> {
    let report = worker
        .training_story_batch_deltas(shard, permission)?
        .ok_or_else(|| {
            LeoError::internal("multi-GPU shard did not enter GPU-native story batching")
        })?;

    if report.story_deltas.len() != shard.len()
        || report.story_changes.len() != shard.len()
        || report.story_metrics.len() != shard.len()
    {
        return Err(LeoError::internal(
            "multi-GPU device shard did not return one result per logical story",
        ));
    }

    let (story_losses, activity) = aggregate_device_story_metrics(report.story_metrics);

    Ok(MultiGpuShardResult {
        story_start,
        story_losses,
        activity,
        story_deltas: report.story_deltas,
        story_changes: report.story_changes,
    })
}

impl MultiGpuBatchTrainer {
    fn new(model: Model, device_count: usize) -> LeoResult<Self> {
        if device_count < 2 {
            return Err(LeoError::internal(
                "multi-GPU trainer requires at least two CUDA devices",
            ));
        }

        // The caller's canonical runtime already owns visible CUDA device 0.
        // Allocate replicas only on the remaining devices. This keeps the DDP
        // memory rule intuitive: the model must fit once on every participating
        // GPU; device 0 is not penalized with a second complete copy.
        let mut replicas = Vec::with_capacity(device_count.saturating_sub(1));
        for device_index in 1..device_count {
            replicas.push(BackendRuntime::new_gpu_on_device(
                model.clone(),
                device_index,
            )?);
        }

        Ok(Self {
            replicas,
            device_count,
        })
    }

    fn train_story_batch(
        &mut self,
        canonical: &mut BackendRuntime,
        stories: Vec<Vec<u8>>,
        permission: Permission,
    ) -> LeoResult<BatchTrainingReport> {
        let device_count = self.device_count.min(stories.len());
        if device_count < 2 {
            return train_story_batch(canonical, stories, permission);
        }
        let replica_count = device_count - 1;

        canonical.synchronize_model()?;
        let base_revision = canonical.model().parameter_revision;
        let base_model = canonical.model().clone();

        // Device 0 is the canonical runtime and participates directly in the
        // initial story pass. The retained-delta CUDA path leaves its canonical
        // parameters unchanged until the flat cross-device mean below.
        canonical.reset_transient_state()?;

        // Normally secondary replicas already equal canonical because sparse
        // synchronization happens at the end of every logical batch. Repair a
        // replica fully only after an out-of-band canonical revision change.
        for worker in &mut self.replicas[..replica_count] {
            if worker.model().parameter_revision != base_revision {
                *worker.model_mut()? = base_model.clone();
            }
            worker.reset_transient_state()?;
        }

        let results = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(device_count);

            let range = balanced_story_range(stories.len(), device_count, 0);
            let shard = &stories[range.clone()];
            let canonical_worker: &mut BackendRuntime = &mut *canonical;
            handles.push(scope.spawn(move || {
                run_multi_gpu_shard(canonical_worker, shard, range.start, permission)
            }));

            for (replica_index, worker) in self.replicas[..replica_count].iter_mut().enumerate() {
                let device_index = replica_index + 1;
                let range = balanced_story_range(stories.len(), device_count, device_index);
                let shard = &stories[range.clone()];
                handles
                    .push(scope.spawn(move || {
                        run_multi_gpu_shard(worker, shard, range.start, permission)
                    }));
            }

            let mut results = Vec::with_capacity(device_count);
            for handle in handles {
                results.push(
                    handle
                        .join()
                        .map_err(|_| LeoError::internal("multi-GPU training worker panicked"))??,
                );
            }
            Ok::<_, LeoError>(results)
        })?;

        // Thread handles are joined in device order, but sort explicitly by
        // canonical story position so future placement policies cannot change
        // FP32 reduction order accidentally.
        let mut results = results;
        results.sort_by_key(|result| result.story_start);

        let mut deltas = Vec::with_capacity(stories.len());
        let mut raw_changes = Vec::with_capacity(stories.len());
        let mut losses_by_story = Vec::with_capacity(stories.len());
        let mut activity = ActivityDiagnostics::default();
        let mut min_stories_per_device = usize::MAX;
        let mut max_stories_per_device = 0usize;

        for result in results {
            let shard_story_count = result.story_deltas.len();
            min_stories_per_device = min_stories_per_device.min(shard_story_count);
            max_stories_per_device = max_stories_per_device.max(shard_story_count);
            deltas.extend(result.story_deltas);
            raw_changes.extend(result.story_changes);
            losses_by_story.extend(result.story_losses);
            activity.add(result.activity);
        }

        if deltas.len() != stories.len()
            || raw_changes.len() != stories.len()
            || losses_by_story.len() != stories.len()
        {
            return Err(LeoError::internal(
                "multi-GPU flattened logical batch does not match story count",
            ));
        }

        // Aggregate loss in canonical story/target order after flattening so
        // device partitioning cannot change reporting reduction order.
        let mut loss_sum = 0.0f64;
        let mut targets = 0usize;
        for losses in &losses_by_story {
            for &loss in losses {
                loss_sum += loss as f64;
                targets = targets.saturating_add(1);
            }
        }

        // ----------------------------------------------------
        // One flat canonical mean across the original logical workers.
        // ----------------------------------------------------
        let merge_started = Instant::now();
        let (merged, sync_changes) = {
            let model = canonical.model_mut()?;
            let merged = apply_mean_deltas(model, &deltas)?;
            let sync_changes = merged_parameter_changes(model, &deltas, &raw_changes)?;
            (merged, sync_changes)
        };

        // `model_mut` deliberately marks a CUDA host mirror dirty. Because GPU
        // 0 already contains the old canonical image, upload only the sparse
        // rows changed by the flat mean instead of paying a full-model H2D copy
        // before replay. CPU/reference backends treat this as a no-op.
        canonical.commit_sparse_host_model(&sync_changes)?;
        canonical.reset_transient_state()?;
        let merge_seconds = merge_started.elapsed().as_secs_f64();

        // ----------------------------------------------------
        // Sparse canonical -> secondary GPU replicas, concurrently. Device 0
        // is already current from commit_sparse_host_model above.
        // ----------------------------------------------------
        let sync_started = Instant::now();
        let canonical_model = canonical.model();
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(replica_count);
            for worker in &mut self.replicas[..replica_count] {
                let changes = &sync_changes;
                handles.push(
                    scope.spawn(move || worker.synchronize_sparse_model(canonical_model, changes)),
                );
            }
            for handle in handles {
                handle.join().map_err(|_| {
                    LeoError::internal("multi-GPU sparse synchronization worker panicked")
                })??;
            }
            Ok::<_, LeoError>(())
        })?;
        let sync_seconds = sync_started.elapsed().as_secs_f64();

        eprintln!(
            "{{\"event\":\"multi_gpu_sparse_sync\",\"devices\":{},\"logical_stories\":{},\"stories_per_device_min\":{},\"stories_per_device_max\":{},\"exact_flat_story_mean\":true,\"canonical_device0_reused\":true,\"threshold\":{},\"recurrent\":{},\"input\":{},\"output_neurons\":{},\"context_slots\":{},\"output_bias\":{},\"context_output\":{},\"merge_ms\":{:.3},\"replica_sync_ms\":{:.3}}}",
            device_count,
            stories.len(),
            min_stories_per_device,
            max_stories_per_device,
            sync_changes.threshold.len(),
            sync_changes.recurrent_weight.len(),
            sync_changes.input_weight.len(),
            sync_changes.output_neurons.len(),
            sync_changes.context_slots.len(),
            sync_changes.output_bias_dirty,
            sync_changes.context_output_dirty,
            merge_seconds * 1000.0,
            sync_seconds * 1000.0,
        );

        // Replay stays in canonical story/range order. Until a true device-side
        // model-parallel replay executor exists, moving it across devices would
        // either alter parameter visibility or add a host/device barrier per
        // replay step. Preserve learning semantics and synchronize the resulting
        // sparse changes back to secondary data-parallel replicas afterward.
        canonical.enable_parameter_tracking();
        let replay = apply_batch_replay_policy(canonical, &stories, &losses_by_story, permission)?;
        activity.add(replay.activity);

        if replay.replay_steps > 0 {
            let replay_changes = canonical.parameter_changes()?;
            let canonical_model = canonical.model();
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(replica_count);
                for worker in &mut self.replicas[..replica_count] {
                    let changes = &replay_changes;
                    handles.push(
                        scope.spawn(move || {
                            worker.synchronize_sparse_model(canonical_model, changes)
                        }),
                    );
                }
                for handle in handles {
                    handle.join().map_err(|_| {
                        LeoError::internal("multi-GPU replay synchronization worker panicked")
                    })??;
                }
                Ok::<_, LeoError>(())
            })?;
        }

        Ok(BatchTrainingReport {
            mean_loss: loss_sum / targets.max(1) as f64,
            targets,
            stories: stories.len(),
            activity,
            replay_segments: replay.replay_segments,
            replay_steps: replay.replay_steps,
            replay_prefix_steps: replay.prefix_steps,
            merge: MergeMetrics {
                workers: stories.len(),
                fixed_parameter_updates: merged.fixed_parameter_updates,
                context_keys: merged.context_keys,
            },
        })
    }
}
fn apply_batch_replay_policy(
    runtime: &mut BackendRuntime,
    stories: &[Vec<u8>],
    losses_by_story: &[Vec<f32>],
    permission: Permission,
) -> LeoResult<ReplayTrainingReport> {
    if stories.len() != losses_by_story.len() {
        return Err(LeoError::internal(
            "replay story/loss batch length mismatch",
        ));
    }
    let debug = replay_debug_options();
    let batch_started = debug.timing.then(Instant::now);
    let mut combined = ReplayTrainingReport {
        activity: ActivityDiagnostics::default(),
        replay_segments: 0,
        replay_steps: 0,
        prefix_steps: 0,
        timing: ReplayTiming::default(),
        selection_ms: 0.0,
    };
    for (story_index, (story, losses)) in stories.iter().zip(losses_by_story).enumerate() {
        let replay = apply_replay_policy(runtime, story, losses, permission, story_index)?;
        combined.activity.add(replay.activity);
        combined.replay_segments = combined
            .replay_segments
            .saturating_add(replay.replay_segments);
        combined.replay_steps = combined.replay_steps.saturating_add(replay.replay_steps);
        combined.prefix_steps = combined.prefix_steps.saturating_add(replay.prefix_steps);
        combined.timing.add(replay.timing);
        combined.selection_ms += replay.selection_ms;
    }
    if let Some(started) = batch_started {
        combined.timing.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    if debug.summary {
        eprintln!(
            "{{\"event\":\"replay_batch_debug\",\"stories\":{},\"segments\":{},\"target_steps\":{},\"prefix_steps\":{},\"execution_steps\":{},\"selection_ms\":{},\"begin_ms\":{},\"prefix_build_ms\":{},\"prefix_execute_ms\":{},\"target_build_ms\":{},\"target_execute_ms\":{},\"reset_ms\":{},\"total_ms\":{},\"prefix_steps_per_second\":{},\"target_steps_per_second\":{}}}",
            stories.len(),
            combined.replay_segments,
            combined.replay_steps,
            combined.prefix_steps,
            combined.replay_steps.saturating_add(combined.prefix_steps),
            combined.selection_ms,
            combined.timing.begin_ms,
            combined.timing.prefix_build_ms,
            combined.timing.prefix_execute_ms,
            combined.timing.target_build_ms,
            combined.timing.target_execute_ms,
            combined.timing.reset_ms,
            combined.timing.total_ms,
            if combined.timing.prefix_execute_ms > 0.0 { combined.prefix_steps as f64 / (combined.timing.prefix_execute_ms / 1000.0) } else { 0.0 },
            if combined.timing.target_execute_ms > 0.0 { combined.replay_steps as f64 / (combined.timing.target_execute_ms / 1000.0) } else { 0.0 },
        );
    }
    Ok(combined)
}

pub(crate) fn train_story_batch(
    runtime: &mut BackendRuntime,
    stories: Vec<Vec<u8>>,
    permission: Permission,
) -> LeoResult<BatchTrainingReport> {
    if stories.is_empty() {
        return Err(LeoError::internal("training batch cannot be empty"));
    }

    if stories.len() == 1 {
        let initial = train_document_pass(runtime, &stories[0], permission)?;
        let replay = apply_replay_policy(runtime, &stories[0], &initial.losses, permission, 0)?;
        let mut activity = initial.activity;
        activity.add(replay.activity);
        return Ok(BatchTrainingReport {
            mean_loss: initial.mean_loss,
            targets: initial.losses.len(),
            stories: 1,
            activity,
            replay_segments: replay.replay_segments,
            replay_steps: replay.replay_steps,
            replay_prefix_steps: replay.prefix_steps,
            merge: MergeMetrics {
                workers: 1,
                fixed_parameter_updates: 0,
                context_keys: 0,
            },
        });
    }

    if runtime.resolved_backend() == BackendKind::Gpu {
        if let Some(device_report) = runtime.training_story_batch(&stories, permission)? {
            let mut activity = ActivityDiagnostics::default();
            let mut loss_sum = 0.0f64;
            let mut targets = 0usize;
            let mut losses_by_story = Vec::with_capacity(device_report.story_metrics.len());
            for story_metrics in device_report.story_metrics {
                let mut losses = Vec::with_capacity(story_metrics.len());
                for metrics in story_metrics {
                    if let Some(loss) = metrics.loss {
                        losses.push(loss);
                        loss_sum += loss as f64;
                        targets = targets.saturating_add(1);
                    }
                    activity.record(metrics);
                }
                losses_by_story.push(losses);
            }
            let replay =
                apply_batch_replay_policy(runtime, &stories, &losses_by_story, permission)?;
            activity.add(replay.activity);
            return Ok(BatchTrainingReport {
                mean_loss: loss_sum / targets.max(1) as f64,
                targets,
                stories: stories.len(),
                activity,
                replay_segments: replay.replay_segments,
                replay_steps: replay.replay_steps,
                replay_prefix_steps: replay.prefix_steps,
                merge: device_report.merge,
            });
        }
    }

    runtime.synchronize_model()?;
    let base = Arc::new(runtime.model().clone());
    let mut handles = Vec::with_capacity(stories.len());
    for story in stories {
        let worker_base = Arc::clone(&base);
        let mut worker = runtime.fork((*worker_base).clone())?;
        handles.push(thread::spawn(
            move || -> LeoResult<(Vec<u8>, SparseModelDelta, DocumentTrainingPass)> {
                worker.enable_parameter_tracking();
                let report = train_document_pass(&mut worker, &story, permission)?;
                let changes = worker.parameter_changes()?;
                let delta =
                    SparseModelDelta::between_tracked(&worker_base, worker.model(), &changes)?;
                Ok((story, delta, report))
            },
        ));
    }

    let mut trained_stories = Vec::with_capacity(handles.len());
    let mut deltas = Vec::with_capacity(handles.len());
    let mut reports = Vec::with_capacity(handles.len());
    for handle in handles {
        let (story, delta, report) = handle
            .join()
            .map_err(|_| LeoError::internal("story worker panicked"))??;
        trained_stories.push(story);
        deltas.push(delta);
        reports.push(report);
    }
    let merge = apply_mean_deltas(runtime.model_mut()?, &deltas)?;
    runtime.reset_transient_state()?;

    let mut activity = ActivityDiagnostics::default();
    let mut weighted_loss = 0.0f64;
    let mut targets = 0usize;
    let mut losses_by_story = Vec::with_capacity(reports.len());
    for report in reports {
        weighted_loss += report.mean_loss * report.losses.len() as f64;
        targets = targets.saturating_add(report.losses.len());
        activity.add(report.activity);
        losses_by_story.push(report.losses);
    }
    let replay =
        apply_batch_replay_policy(runtime, &trained_stories, &losses_by_story, permission)?;
    activity.add(replay.activity);

    Ok(BatchTrainingReport {
        mean_loss: weighted_loss / targets.max(1) as f64,
        targets,
        stories: deltas.len(),
        activity,
        replay_segments: replay.replay_segments,
        replay_steps: replay.replay_steps,
        replay_prefix_steps: replay.prefix_steps,
        merge,
    })
}

pub(crate) fn train_document_pass(
    runtime: &mut BackendRuntime,
    story: &[u8],
    permission: Permission,
) -> LeoResult<DocumentTrainingPass> {
    if story.is_empty() {
        return Ok(DocumentTrainingPass {
            mean_loss: 0.0,
            losses: Vec::new(),
            activity: ActivityDiagnostics::default(),
        });
    }

    let mut activity = ActivityDiagnostics::default();
    let mut losses = Vec::with_capacity(story.len() + 1);
    let mut loss_sum = 0.0f64;
    runtime.begin_document()?;

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
    for metrics in runtime.training_step_batch(&steps, permission)? {
        record_training_step(metrics, &mut losses, &mut loss_sum, &mut activity);
    }
    runtime.finish_document()?;

    Ok(DocumentTrainingPass {
        mean_loss: loss_sum / losses.len().max(1) as f64,
        losses,
        activity,
    })
}

fn record_training_step(
    metrics: StepMetrics,
    losses: &mut Vec<f32>,
    loss_sum: &mut f64,
    activity: &mut ActivityDiagnostics,
) {
    let loss = metrics.loss.unwrap_or(0.0);
    losses.push(loss);
    *loss_sum += loss as f64;
    activity.record(metrics);
}

pub(crate) fn select_replay_ranges(
    losses: &[f32],
    fraction: f32,
    segment_targets: usize,
) -> Vec<std::ops::Range<usize>> {
    if losses.is_empty() || !fraction.is_finite() || fraction <= 0.0 || segment_targets == 0 {
        return Vec::new();
    }

    let target_budget = ((losses.len() as f64) * (fraction.min(1.0) as f64))
        .ceil()
        .max(1.0) as usize;
    let mut positions = losses.iter().copied().enumerate().collect::<Vec<_>>();
    positions.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut selected = Vec::<std::ops::Range<usize>>::new();
    let mut selected_targets = 0usize;
    for (position, loss) in positions {
        if selected_targets >= target_budget {
            break;
        }
        if !loss.is_finite() || loss <= 0.0 {
            continue;
        }

        let remaining = target_budget - selected_targets;
        let width = segment_targets.min(remaining).min(losses.len());
        let mut start = position.saturating_sub(width / 2);
        let end = start.saturating_add(width).min(losses.len());
        start = end.saturating_sub(width);
        let candidate = start..end;
        if selected
            .iter()
            .any(|existing| ranges_overlap(existing, &candidate))
        {
            continue;
        }
        selected_targets = selected_targets.saturating_add(candidate.len());
        selected.push(candidate);
    }

    selected.sort_unstable_by_key(|range| range.start);
    selected
}

pub(crate) fn ranges_overlap(
    left: &std::ops::Range<usize>,
    right: &std::ops::Range<usize>,
) -> bool {
    left.start < right.end && right.start < left.end
}

pub(crate) fn replay_target_range(
    runtime: &mut BackendRuntime,
    story: &[u8],
    target_range: std::ops::Range<usize>,
    permission: Permission,
) -> LeoResult<ActivityDiagnostics> {
    Ok(replay_target_range_impl(runtime, story, target_range, permission, true, false)?.activity)
}

fn replay_target_range_impl(
    runtime: &mut BackendRuntime,
    story: &[u8],
    target_range: std::ops::Range<usize>,
    permission: Permission,
    cleanup_after: bool,
    collect_timing: bool,
) -> LeoResult<ReplayRangeExecution> {
    let total_started = collect_timing.then(Instant::now);
    let mut activity = ActivityDiagnostics::default();
    if story.is_empty() || target_range.is_empty() || target_range.start > story.len() {
        return Ok(ReplayRangeExecution {
            activity,
            prefix_steps: 0,
            target_steps: 0,
            timing: ReplayTiming::default(),
        });
    }
    let end = target_range.end.min(story.len() + 1);
    let mut timing = ReplayTiming::default();

    let begin_started = collect_timing.then(Instant::now);
    runtime.begin_document()?;
    timing.begin_ms = begin_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let mut next_target = target_range.start;
    let mut prefix_steps = 0u64;
    let mut target_steps = 0u64;
    if next_target == 0 {
        let target_started = collect_timing.then(Instant::now);
        for metrics in
            runtime.training_step_batch(&[(BEGIN_DOCUMENT, Some(story[0] as u32))], permission)?
        {
            activity.record(metrics);
            target_steps = target_steps.saturating_add(1);
        }
        timing.target_execute_ms += target_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        next_target = 1;
    } else {
        let prefix_build_started = collect_timing.then(Instant::now);
        let mut prefix = Vec::with_capacity(next_target);
        prefix.push((BEGIN_DOCUMENT, None));
        prefix.extend(
            story[..next_target.saturating_sub(1)]
                .iter()
                .map(|byte| (*byte as u32, None)),
        );
        prefix_steps = prefix.len() as u64;
        timing.prefix_build_ms = prefix_build_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let prefix_execute_started = collect_timing.then(Instant::now);
        runtime.advance_frozen_batch(&prefix)?;
        timing.prefix_execute_ms = prefix_execute_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
    }

    let target_build_started = collect_timing.then(Instant::now);
    let target_batch = (next_target..end)
        .map(|target_position| {
            let input = story[target_position - 1] as u32;
            let target = if target_position < story.len() {
                story[target_position] as u32
            } else {
                END_DOCUMENT
            };
            (input, Some(target))
        })
        .collect::<Vec<_>>();
    timing.target_build_ms = target_build_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let target_execute_started = collect_timing.then(Instant::now);
    for metrics in runtime.training_step_batch(&target_batch, permission)? {
        activity.record(metrics);
        target_steps = target_steps.saturating_add(1);
    }
    timing.target_execute_ms += target_execute_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if cleanup_after {
        let reset_started = collect_timing.then(Instant::now);
        runtime.reset_transient_state()?;
        timing.reset_ms = reset_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
    }
    timing.total_ms = total_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    Ok(ReplayRangeExecution {
        activity,
        prefix_steps,
        target_steps,
        timing,
    })
}

#[cfg(test)]
mod multi_gpu_partition_tests {
    use super::balanced_story_range;

    #[test]
    fn balanced_story_ranges_cover_logical_batch_in_order() {
        for total in 2usize..=16 {
            for shards in 2usize..=total {
                let ranges = (0..shards)
                    .map(|index| balanced_story_range(total, shards, index))
                    .collect::<Vec<_>>();
                assert_eq!(ranges.first().expect("range").start, 0);
                assert_eq!(ranges.last().expect("range").end, total);
                for pair in ranges.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start);
                }
                let lengths = ranges.iter().map(|range| range.len()).collect::<Vec<_>>();
                let min = *lengths.iter().min().expect("length");
                let max = *lengths.iter().max().expect("length");
                assert!(max - min <= 1);
                assert_eq!(lengths.iter().sum::<usize>(), total);
            }
        }
    }

    #[test]
    fn sixteen_devices_can_own_one_story_each() {
        let ranges = (0..16)
            .map(|index| balanced_story_range(16, 16, index))
            .collect::<Vec<_>>();
        assert!(ranges.iter().all(|range| range.len() == 1));
        for (story, range) in ranges.iter().enumerate() {
            assert_eq!(range.clone(), story..story + 1);
        }
    }

    #[test]
    fn uneven_device_count_preserves_story_order() {
        let ranges = (0..3)
            .map(|index| balanced_story_range(16, 3, index))
            .collect::<Vec<_>>();
        assert_eq!(ranges, vec![0..6, 6..11, 11..16]);
    }
}
