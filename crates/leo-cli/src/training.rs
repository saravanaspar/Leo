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
    apply_mean_deltas, apply_sum_deltas, available_gpu_devices, merged_parameter_changes,
    visible_gpu_device_compatibility, BackendKind, BackendRuntime, LeoError, LeoResult,
    MergeMetrics, Model, PackedSparseModelUpdate, ParameterChanges, Permission, SparseModelDelta,
    StepMetrics,
};
use std::sync::{mpsc, Arc, OnceLock};
use std::thread;
use std::time::Instant;

pub(crate) const GPU_REFERENCE_WORKERS: usize = 16;

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

/// Experimental replay executor that preserves the configured replay target
/// budget (30% in the reference configuration) but carries recurrent state
/// forward between selected ranges instead of rebuilding every range from byte
/// zero. This is intentionally opt-in until full TinyStories quality parity is
/// established because it changes replay-state semantics while keeping FP32 and
/// the supervised replay fraction unchanged.
fn replay_streaming_enabled() -> bool {
    std::env::var("LEO_REPLAY_STREAMING")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

/// Multi-GPU replay may change cross-story parameter visibility while never
/// changing the configured replay budget or FP32 arithmetic. Keep it opt-in
/// until a real training-quality A/B establishes equivalence or improvement;
/// LEO_MULTI_GPU_PARALLEL_REPLAY=1 enables the local-trajectory strategy.
fn multi_gpu_parallel_replay_enabled() -> bool {
    std::env::var("LEO_MULTI_GPU_PARALLEL_REPLAY")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
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

fn replay_step_at_position(
    story: &[u8],
    position: usize,
    supervised: bool,
) -> Option<(u32, Option<u32>)> {
    if story.is_empty() || position > story.len() {
        return None;
    }
    let input = if position == 0 {
        BEGIN_DOCUMENT
    } else {
        story[position - 1] as u32
    };
    let target = if supervised {
        Some(if position < story.len() {
            story[position] as u32
        } else {
            END_DOCUMENT
        })
    } else {
        None
    };
    Some((input, target))
}

fn replay_ranges_streaming(
    runtime: &mut BackendRuntime,
    story: &[u8],
    ranges: &[std::ops::Range<usize>],
    permission: Permission,
    collect_timing: bool,
) -> LeoResult<ReplayRangeExecution> {
    let total_started = collect_timing.then(Instant::now);
    let mut activity = ActivityDiagnostics::default();
    let mut timing = ReplayTiming::default();
    if story.is_empty() || ranges.is_empty() {
        return Ok(ReplayRangeExecution {
            activity,
            prefix_steps: 0,
            target_steps: 0,
            timing,
        });
    }

    let begin_started = collect_timing.then(Instant::now);
    runtime.begin_document()?;
    timing.begin_ms = begin_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // Build the complete streaming replay timeline once. Frozen positions keep
    // recurrent/context state moving between selected ranges; selected
    // positions remain supervised. CUDA consumes this mixed schedule in one
    // cooperative launch per 4096 timeline positions instead of bouncing
    // through the host at every range boundary.
    let build_started = collect_timing.then(Instant::now);
    let terminal = story.len() + 1;
    let mut cursor = 0usize;
    let mut schedule = Vec::new();
    let mut prefix_steps = 0u64;
    let mut target_steps = 0u64;

    for range in ranges {
        let start = range.start.min(terminal).max(cursor);
        let end = range.end.min(terminal).max(start);
        for position in cursor..start {
            if let Some(step) = replay_step_at_position(story, position, false) {
                schedule.push(step);
                prefix_steps = prefix_steps.saturating_add(1);
            }
        }
        for position in start..end {
            if let Some(step) = replay_step_at_position(story, position, true) {
                schedule.push(step);
                target_steps = target_steps.saturating_add(1);
            }
        }
        cursor = end;
    }
    timing.target_build_ms = build_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let execute_started = collect_timing.then(Instant::now);
    for metrics in runtime.replay_streaming_batch(&schedule, permission)? {
        activity.record(metrics);
    }
    // The device-resident schedule deliberately measures frozen + supervised
    // execution together; attribute it to target_execute_ms so the reported
    // replay target rate is an honest end-to-end rate rather than inventing a
    // split that no longer exists at the kernel level.
    timing.target_execute_ms = execute_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let reset_started = collect_timing.then(Instant::now);
    runtime.reset_transient_state()?;
    timing.reset_ms = reset_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
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
        emit_replay_range_debug(story_index, &ranges);
    }

    execute_replay_ranges(
        runtime,
        story,
        &ranges,
        permission,
        story_index,
        selection_ms,
    )
}

fn emit_replay_range_debug(story_index: usize, ranges: &[std::ops::Range<usize>]) {
    if !replay_debug_options().ranges {
        return;
    }
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

fn emit_device_replay_selection_debug(
    runtime: &BackendRuntime,
    story: &[u8],
    ranges: &[std::ops::Range<usize>],
    story_index: usize,
) {
    let debug = replay_debug_options();
    if !debug.selection {
        return;
    }
    let selected_targets = ranges.iter().map(|range| range.len()).sum::<usize>();
    let estimated_prefix_steps = ranges.iter().map(|range| range.start as u64).sum::<u64>();
    let prefix_amplification = estimated_prefix_steps as f64 / selected_targets.max(1) as f64;
    eprintln!(
        "{{\"event\":\"replay_device_selection_debug\",\"story_index\":{},\"story_bytes\":{},\"fraction\":{},\"segment_targets\":{},\"selected_segments\":{},\"selected_targets\":{},\"estimated_prefix_steps\":{},\"prefix_amplification\":{},\"selection_source\":\"device_postprocess\",\"losses_resident_on_device\":true}}",
        story_index,
        story.len(),
        runtime.model().config.replay.fraction,
        runtime.model().config.replay.segment_bytes,
        ranges.len(),
        selected_targets,
        estimated_prefix_steps,
        prefix_amplification,
    );
    emit_replay_range_debug(story_index, ranges);
}

fn execute_replay_ranges(
    runtime: &mut BackendRuntime,
    story: &[u8],
    ranges: &[std::ops::Range<usize>],
    permission: Permission,
    story_index: usize,
    selection_ms: f64,
) -> LeoResult<ReplayTrainingReport> {
    let debug = replay_debug_options();
    let estimated_prefix_steps = ranges.iter().map(|range| range.start as u64).sum::<u64>();
    let story_started = debug.timing.then(Instant::now);
    let mut activity = ActivityDiagnostics::default();
    let mut replay_steps = 0u64;
    let mut prefix_steps = 0u64;
    let mut timing = ReplayTiming::default();
    if replay_streaming_enabled() && !ranges.is_empty() {
        let execution = replay_ranges_streaming(runtime, story, ranges, permission, debug.timing)?;
        replay_steps = execution.target_steps;
        prefix_steps = execution.prefix_steps;
        timing.add(execution.timing);
        activity.add(execution.activity);
        if debug.segments {
            eprintln!(
                "{{\"event\":\"replay_streaming_debug\",\"story_index\":{},\"segments\":{},\"target_steps\":{},\"prefix_steps\":{},\"classic_estimated_prefix_steps\":{}}}",
                story_index,
                ranges.len(),
                replay_steps,
                prefix_steps,
                estimated_prefix_steps,
            );
        }
    } else {
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

impl ReplayTrainingReport {
    fn empty() -> Self {
        Self {
            activity: ActivityDiagnostics::default(),
            replay_segments: 0,
            replay_steps: 0,
            prefix_steps: 0,
            timing: ReplayTiming::default(),
            selection_ms: 0.0,
        }
    }

    fn add(&mut self, other: Self) {
        self.activity.add(other.activity);
        self.replay_segments = self.replay_segments.saturating_add(other.replay_segments);
        self.replay_steps = self.replay_steps.saturating_add(other.replay_steps);
        self.prefix_steps = self.prefix_steps.saturating_add(other.prefix_steps);
        self.timing.add(other.timing);
        self.selection_ms += other.selection_ms;
    }
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
    pub(crate) replay_seconds: f64,
    pub(crate) replay_sync_seconds: f64,
    pub(crate) merge: MergeMetrics,
}

/// Deep training boundary used by the CLI. It owns execution coordination;
/// the canonical `BackendRuntime` remains explicit because checkpointing and
/// evaluation synchronize against that same model owner.
pub(crate) fn multi_gpu_requested() -> bool {
    std::env::var("LEO_MULTI_GPU")
        .map(|value| {
            let value = value.trim();
            value == "1" || value.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

pub(crate) fn configured_multi_gpu_devices(
    backend: BackendKind,
    workers: usize,
) -> LeoResult<usize> {
    if backend != BackendKind::Gpu || workers == 0 {
        return Ok(1);
    }
    let requested = multi_gpu_requested();
    if !requested || workers == 1 {
        return Ok(1);
    }

    if let Ok(value) = std::env::var("LEO_CUDA_DEVICE") {
        let selected = value.trim().parse::<usize>().map_err(|_| {
            LeoError::backend("LEO_CUDA_DEVICE must be an integer when LEO_MULTI_GPU is enabled")
        })?;
        if selected != 0 {
            return Err(LeoError::backend(
                "exact LEO_MULTI_GPU requires the canonical runtime on visible CUDA device 0; use CUDA_VISIBLE_DEVICES to choose/reorder the device group",
            ));
        }
    }

    let visible = available_gpu_devices()?;
    if visible < 2 {
        return Err(LeoError::backend(
            "LEO_MULTI_GPU requested, but fewer than two CUDA devices are visible",
        ));
    }
    let device_count = visible.min(workers);
    let profiles = visible_gpu_device_compatibility()?;
    let selected = profiles.get(..device_count).ok_or_else(|| {
        LeoError::backend("CUDA compatibility probe returned fewer devices than cuDeviceGetCount")
    })?;
    if let Some(first) = selected.first() {
        if let Some(other) = selected.iter().skip(1).find(|profile| {
            profile.name != first.name
                || profile.compute_major != first.compute_major
                || profile.compute_minor != first.compute_minor
        }) {
            return Err(LeoError::backend(format!(
                "exact multi-GPU requires homogeneous CUDA devices; device {} is {} cc {}.{}, device {} is {} cc {}.{}",
                first.ordinal,
                first.name,
                first.compute_major,
                first.compute_minor,
                other.ordinal,
                other.name,
                other.compute_major,
                other.compute_minor,
            )));
        }
    }
    Ok(device_count)
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
    // Device 0 remains the caller-owned canonical runtime. Secondary CUDA
    // runtimes are created once inside persistent owner threads so CUDA
    // context ownership and host-thread creation are not paid per batch.
    replicas: Vec<MultiGpuReplicaWorker>,
    device_count: usize,
}

enum ReplicaCommand {
    Train {
        stories: Arc<Vec<Vec<u8>>>,
        range: std::ops::Range<usize>,
        permission: Permission,
        expected_revision: u64,
        repair_model: Option<Arc<Model>>,
        response: mpsc::Sender<LeoResult<MultiGpuShardResult>>,
    },
    Replay {
        stories: Arc<Vec<Vec<u8>>>,
        losses: Arc<Vec<Vec<f32>>>,
        range: std::ops::Range<usize>,
        permission: Permission,
        expected_revision: u64,
        base_model: Arc<Model>,
        response: mpsc::Sender<LeoResult<MultiGpuReplayShardResult>>,
    },
    Synchronize {
        update: Arc<PackedSparseModelUpdate>,
        response: mpsc::Sender<LeoResult<u64>>,
    },
    Shutdown,
}

struct MultiGpuReplicaWorker {
    device_index: usize,
    sender: mpsc::SyncSender<ReplicaCommand>,
    handle: Option<thread::JoinHandle<()>>,
    parameter_revision: u64,
}

impl MultiGpuReplicaWorker {
    fn spawn(model: Model, device_index: usize) -> LeoResult<Self> {
        let initial_revision = model.parameter_revision;
        let (sender, receiver) = mpsc::sync_channel::<ReplicaCommand>(2);
        let (init_sender, init_receiver) = mpsc::sync_channel::<LeoResult<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("leo-gpu-{device_index}"))
            .spawn(move || {
                let mut runtime = match BackendRuntime::new_gpu_on_device(model, device_index) {
                    Ok(runtime) => {
                        let _ = init_sender.send(Ok(()));
                        runtime
                    }
                    Err(error) => {
                        let _ = init_sender.send(Err(error));
                        return;
                    }
                };

                while let Ok(command) = receiver.recv() {
                    match command {
                        ReplicaCommand::Train {
                            stories,
                            range,
                            permission,
                            expected_revision,
                            repair_model,
                            response,
                        } => {
                            let result = (|| {
                                if runtime.model().parameter_revision != expected_revision {
                                    let repair = repair_model.ok_or_else(|| {
                                        LeoError::internal(format!(
                                            "multi-GPU replica {device_index} revision {} does not match canonical revision {expected_revision}",
                                            runtime.model().parameter_revision
                                        ))
                                    })?;
                                    *runtime.model_mut()? = (*repair).clone();
                                }
                                runtime.reset_transient_state()?;
                                run_multi_gpu_shard(
                                    &mut runtime,
                                    &stories[range.clone()],
                                    range.start,
                                    permission,
                                )
                            })();
                            let _ = response.send(result);
                        }
                        ReplicaCommand::Replay {
                            stories,
                            losses,
                            range,
                            permission,
                            expected_revision,
                            base_model,
                            response,
                        } => {
                            let result = (|| {
                                if runtime.model().parameter_revision != expected_revision {
                                    return Err(LeoError::internal(format!(
                                        "multi-GPU replay replica {device_index} revision {} does not match canonical revision {expected_revision}",
                                        runtime.model().parameter_revision
                                    )));
                                }
                                runtime.reset_transient_state()?;
                                run_multi_gpu_replay_shard(
                                    &mut runtime,
                                    &stories[range.clone()],
                                    &losses[range.clone()],
                                    range.start,
                                    permission,
                                    &base_model,
                                )
                            })();
                            let _ = response.send(result);
                        }
                        ReplicaCommand::Synchronize { update, response } => {
                            let result = runtime
                                .synchronize_packed_model(&update)
                                .map(|_| runtime.model().parameter_revision);
                            let _ = response.send(result);
                        }
                        ReplicaCommand::Shutdown => break,
                    }
                }
            })
            .map_err(|error| {
                LeoError::internal(format!(
                    "failed to spawn persistent multi-GPU worker for device {device_index}: {error}"
                ))
            })?;

        match init_receiver.recv() {
            Ok(Ok(())) => Ok(Self {
                device_index,
                sender,
                handle: Some(handle),
                parameter_revision: initial_revision,
            }),
            Ok(Err(error)) => {
                let _ = handle.join();
                Err(error)
            }
            Err(_) => {
                let _ = handle.join();
                Err(LeoError::internal(format!(
                    "multi-GPU worker for device {device_index} exited during initialization"
                )))
            }
        }
    }

    fn dispatch_train(
        &self,
        stories: Arc<Vec<Vec<u8>>>,
        range: std::ops::Range<usize>,
        permission: Permission,
        expected_revision: u64,
        repair_model: Option<Arc<Model>>,
    ) -> LeoResult<mpsc::Receiver<LeoResult<MultiGpuShardResult>>> {
        let (response, receiver) = mpsc::channel();
        self.sender
            .send(ReplicaCommand::Train {
                stories,
                range,
                permission,
                expected_revision,
                repair_model,
                response,
            })
            .map_err(|_| {
                LeoError::internal(format!(
                    "persistent multi-GPU worker for device {} is unavailable",
                    self.device_index
                ))
            })?;
        Ok(receiver)
    }

    fn dispatch_replay(
        &self,
        stories: Arc<Vec<Vec<u8>>>,
        losses: Arc<Vec<Vec<f32>>>,
        range: std::ops::Range<usize>,
        permission: Permission,
        expected_revision: u64,
        base_model: Arc<Model>,
    ) -> LeoResult<mpsc::Receiver<LeoResult<MultiGpuReplayShardResult>>> {
        let (response, receiver) = mpsc::channel();
        self.sender
            .send(ReplicaCommand::Replay {
                stories,
                losses,
                range,
                permission,
                expected_revision,
                base_model,
                response,
            })
            .map_err(|_| {
                LeoError::internal(format!(
                    "persistent multi-GPU replay worker for device {} is unavailable",
                    self.device_index
                ))
            })?;
        Ok(receiver)
    }

    fn dispatch_sync(
        &self,
        update: Arc<PackedSparseModelUpdate>,
    ) -> LeoResult<mpsc::Receiver<LeoResult<u64>>> {
        let (response, receiver) = mpsc::channel();
        self.sender
            .send(ReplicaCommand::Synchronize { update, response })
            .map_err(|_| {
                LeoError::internal(format!(
                    "persistent multi-GPU worker for device {} is unavailable",
                    self.device_index
                ))
            })?;
        Ok(receiver)
    }
}

impl Drop for MultiGpuReplicaWorker {
    fn drop(&mut self) {
        let _ = self.sender.send(ReplicaCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct MultiGpuShardResult {
    story_start: usize,
    story_work: usize,
    elapsed_seconds: f64,
    story_losses: Vec<Vec<f32>>,
    activity: ActivityDiagnostics,
    story_deltas: Vec<SparseModelDelta>,
    story_changes: Vec<ParameterChanges>,
}

struct MultiGpuReplayShardResult {
    story_start: usize,
    elapsed_seconds: f64,
    report: ReplayTrainingReport,
    delta: SparseModelDelta,
    changes: ParameterChanges,
}

struct MultiGpuReplayOutcome {
    report: ReplayTrainingReport,
    replay_seconds: f64,
    replay_sync_seconds: f64,
}

#[cfg(test)]
fn balanced_story_range(total: usize, shards: usize, shard_index: usize) -> std::ops::Range<usize> {
    let base = total / shards;
    let extra = total % shards;
    let start = shard_index * base + shard_index.min(extra);
    let len = base + usize::from(shard_index < extra);
    start..start + len
}

fn story_work(story: &[u8]) -> usize {
    if story.is_empty() {
        0
    } else {
        story.len().saturating_add(1)
    }
}

fn work_balanced_story_ranges(stories: &[Vec<u8>], shards: usize) -> Vec<std::ops::Range<usize>> {
    let shard_count = shards.min(stories.len());
    if shard_count == 0 {
        return Vec::new();
    }

    let mut ranges = Vec::with_capacity(shard_count);
    let mut start = 0usize;
    let mut remaining_work = stories.iter().map(|story| story_work(story)).sum::<usize>();

    for shard_index in 0..shard_count {
        let remaining_shards = shard_count - shard_index;
        if remaining_shards == 1 {
            ranges.push(start..stories.len());
            break;
        }

        let max_end = stories.len() - (remaining_shards - 1);
        let target = remaining_work.div_ceil(remaining_shards);
        let mut end = start;
        let mut assigned_work = 0usize;
        while end < max_end {
            let next_work = story_work(&stories[end]);
            if end > start && assigned_work >= target {
                break;
            }
            assigned_work = assigned_work.saturating_add(next_work);
            end += 1;
        }
        if end == start {
            end += 1;
            assigned_work = story_work(&stories[start]);
        }
        ranges.push(start..end);
        start = end;
        remaining_work = remaining_work.saturating_sub(assigned_work);
    }
    ranges
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
    let started = Instant::now();
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
        story_work: shard.iter().map(|story| story_work(story)).sum(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        story_losses,
        activity,
        story_deltas: report.story_deltas,
        story_changes: report.story_changes,
    })
}

fn run_multi_gpu_replay_shard(
    worker: &mut BackendRuntime,
    stories: &[Vec<u8>],
    losses_by_story: &[Vec<f32>],
    story_start: usize,
    permission: Permission,
    base_model: &Model,
) -> LeoResult<MultiGpuReplayShardResult> {
    if stories.len() != losses_by_story.len() {
        return Err(LeoError::internal(
            "multi-GPU replay shard story/loss length mismatch",
        ));
    }
    if worker.model().parameter_revision != base_model.parameter_revision {
        return Err(LeoError::internal(
            "multi-GPU replay shard did not start from the canonical revision",
        ));
    }
    let started = Instant::now();
    worker.enable_parameter_tracking();
    let report = apply_batch_replay_policy(worker, stories, losses_by_story, permission)?;
    let changes = worker.parameter_changes()?;
    let delta = SparseModelDelta::between_tracked(base_model, worker.model(), &changes)?;
    Ok(MultiGpuReplayShardResult {
        story_start,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        report,
        delta,
        changes,
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
        // Create secondary runtimes once on persistent owner threads. Keeping
        // each CUDA runtime on its owner thread avoids per-batch thread/context
        // setup and leaves device 0 free to execute the canonical shard.
        let mut replicas = Vec::with_capacity(device_count.saturating_sub(1));
        for device_index in 1..device_count {
            replicas.push(MultiGpuReplicaWorker::spawn(model.clone(), device_index)?);
        }

        Ok(Self {
            replicas,
            device_count,
        })
    }

    fn synchronize_replicas(&mut self, update: Arc<PackedSparseModelUpdate>) -> LeoResult<()> {
        // Synchronize every resident replica, not only devices active in a
        // short final batch. This prevents an inactive device from becoming
        // stale and requiring a full-model repair when a later batch grows.
        let mut receivers = Vec::with_capacity(self.replicas.len());
        for worker in &self.replicas {
            receivers.push(worker.dispatch_sync(Arc::clone(&update))?);
        }
        for (worker, receiver) in self.replicas.iter_mut().zip(receivers) {
            let revision = receiver.recv().map_err(|_| {
                LeoError::internal(format!(
                    "persistent multi-GPU synchronization worker {} exited",
                    worker.device_index
                ))
            })??;
            worker.parameter_revision = revision;
        }
        Ok(())
    }

    fn run_parallel_replay(
        &mut self,
        canonical: &mut BackendRuntime,
        stories: Arc<Vec<Vec<u8>>>,
        losses_by_story: Vec<Vec<f32>>,
        ranges: &[std::ops::Range<usize>],
        device_count: usize,
        permission: Permission,
    ) -> LeoResult<MultiGpuReplayOutcome> {
        if stories.len() != losses_by_story.len() || ranges.len() < device_count {
            return Err(LeoError::internal(
                "multi-GPU parallel replay inputs do not match device partition",
            ));
        }
        let active_replica_count = device_count.saturating_sub(1);
        let base_revision = canonical.model().parameter_revision;
        // One immutable host snapshot is shared by every owner thread when
        // constructing sparse local-replay deltas. This replaces N full model
        // clones and gives every replay shard exactly the same starting point.
        let base_model = Arc::new(canonical.model().clone());
        let losses = Arc::new(losses_by_story);
        let replay_started = Instant::now();

        let mut replica_receivers = Vec::with_capacity(active_replica_count);
        for (replica_index, worker) in self.replicas[..active_replica_count].iter().enumerate() {
            let device_index = replica_index + 1;
            replica_receivers.push(worker.dispatch_replay(
                Arc::clone(&stories),
                Arc::clone(&losses),
                ranges[device_index].clone(),
                permission,
                base_revision,
                Arc::clone(&base_model),
            )?);
        }

        let canonical_range = ranges[0].clone();
        let canonical_result = run_multi_gpu_replay_shard(
            canonical,
            &stories[canonical_range.clone()],
            &losses[canonical_range.clone()],
            canonical_range.start,
            permission,
            &base_model,
        )?;

        let mut results = Vec::with_capacity(device_count);
        results.push(canonical_result);
        for (replica_index, receiver) in replica_receivers.into_iter().enumerate() {
            let device_index = replica_index + 1;
            results.push(receiver.recv().map_err(|_| {
                LeoError::internal(format!(
                    "persistent multi-GPU replay worker {device_index} exited"
                ))
            })??);
        }
        let replay_seconds = replay_started.elapsed().as_secs_f64();
        results.sort_by_key(|result| result.story_start);

        let mut report = ReplayTrainingReport::empty();
        let mut deltas = Vec::with_capacity(results.len());
        let mut raw_changes = Vec::with_capacity(results.len());
        let mut min_device_seconds = f64::INFINITY;
        let mut max_device_seconds = 0.0f64;
        for result in results {
            min_device_seconds = min_device_seconds.min(result.elapsed_seconds);
            max_device_seconds = max_device_seconds.max(result.elapsed_seconds);
            report.add(result.report);
            deltas.push(result.delta);
            raw_changes.push(result.changes);
        }

        let sync_started = Instant::now();
        if report.replay_steps > 0 {
            // Device-local replay trajectories all started from base_model.
            // Restore device0's host image to that base, then add the local
            // sparse trajectories. A sum (not device-count mean) retains the
            // first-order update magnitude of executing every selected 30%
            // replay target while allowing cross-story replay concurrency.
            *canonical.model_mut()? = (*base_model).clone();
            let (replay_merge, replay_changes) = {
                let model = canonical.model_mut()?;
                let replay_merge = apply_sum_deltas(model, &deltas)?;
                let replay_changes = merged_parameter_changes(model, &deltas, &raw_changes)?;
                (replay_merge, replay_changes)
            };
            let replay_update = Arc::new(PackedSparseModelUpdate::from_model(
                canonical.model(),
                &replay_changes,
            )?);
            canonical.commit_packed_host_model(&replay_update)?;
            canonical.reset_transient_state()?;
            self.synchronize_replicas(Arc::clone(&replay_update))?;
            eprintln!(
                "{{\"event\":\"multi_gpu_parallel_replay\",\"devices\":{},\"replay_steps\":{},\"prefix_steps\":{},\"replay_seconds\":{:.6},\"device_seconds_min\":{:.6},\"device_seconds_max\":{:.6},\"reduction\":\"sum_local_trajectories\",\"fixed_parameter_updates\":{},\"context_keys\":{},\"fp32\":true}}",
                device_count,
                report.replay_steps,
                report.prefix_steps,
                replay_seconds,
                min_device_seconds,
                max_device_seconds,
                replay_merge.fixed_parameter_updates,
                replay_merge.context_keys,
            );
        } else {
            canonical.reset_transient_state()?;
        }
        let replay_sync_seconds = sync_started.elapsed().as_secs_f64();

        Ok(MultiGpuReplayOutcome {
            report,
            replay_seconds,
            replay_sync_seconds,
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
            // A short one-story batch still advances the canonical parameters.
            // The single-device path tracks and sparse-commits its complete
            // update into resident device-0 lanes. Reuse that tracked packet to
            // keep every secondary replica current as well.
            let report = train_story_batch(canonical, stories, permission)?;
            let changes = canonical.parameter_changes()?;
            let update = Arc::new(PackedSparseModelUpdate::from_model(
                canonical.model(),
                &changes,
            )?);
            self.synchronize_replicas(update)?;
            return Ok(report);
        }
        let active_replica_count = device_count - 1;

        canonical.synchronize_model()?;
        let base_revision = canonical.model().parameter_revision;
        canonical.reset_transient_state()?;

        let stories = Arc::new(stories);
        let ranges = work_balanced_story_ranges(stories.as_slice(), device_count);

        // Full repair is an exceptional path only. Build one host snapshot if
        // any active replica missed a sparse synchronization, then share it;
        // each device owns its independent model after cloning on its owner
        // thread. Steady-state batches allocate no full canonical clone.
        let needs_repair = self.replicas[..active_replica_count]
            .iter()
            .any(|worker| worker.parameter_revision != base_revision);
        let repair_model = needs_repair.then(|| Arc::new(canonical.model().clone()));

        // Dispatch secondary devices first so their GPU work overlaps device0.
        let mut replica_receivers = Vec::with_capacity(active_replica_count);
        for (replica_index, worker) in self.replicas[..active_replica_count].iter().enumerate() {
            let device_index = replica_index + 1;
            let range = ranges[device_index].clone();
            let repair = if worker.parameter_revision != base_revision {
                Some(Arc::clone(repair_model.as_ref().ok_or_else(|| {
                    LeoError::internal("multi-GPU replica repair snapshot was not prepared")
                })?))
            } else {
                None
            };
            replica_receivers.push(worker.dispatch_train(
                Arc::clone(&stories),
                range,
                permission,
                base_revision,
                repair,
            )?);
        }

        let canonical_range = ranges[0].clone();
        let canonical_result = run_multi_gpu_shard(
            canonical,
            &stories[canonical_range.clone()],
            canonical_range.start,
            permission,
        )?;

        let mut results = Vec::with_capacity(device_count);
        results.push(canonical_result);
        for (replica_index, receiver) in replica_receivers.into_iter().enumerate() {
            let device_index = replica_index + 1;
            results.push(receiver.recv().map_err(|_| {
                LeoError::internal(format!(
                    "persistent multi-GPU training worker {device_index} exited"
                ))
            })??);
        }

        // Restore canonical story order explicitly. Work-aware placement is an
        // execution detail and must never alter FP32/f64 reduction order.
        results.sort_by_key(|result| result.story_start);

        let mut deltas = Vec::with_capacity(stories.len());
        let mut raw_changes = Vec::with_capacity(stories.len());
        let mut losses_by_story = Vec::with_capacity(stories.len());
        let mut activity = ActivityDiagnostics::default();
        let mut min_stories_per_device = usize::MAX;
        let mut max_stories_per_device = 0usize;
        let mut min_work_per_device = usize::MAX;
        let mut max_work_per_device = 0usize;
        let mut min_device_seconds = f64::INFINITY;
        let mut max_device_seconds = 0.0f64;

        for result in results {
            let shard_story_count = result.story_deltas.len();
            min_stories_per_device = min_stories_per_device.min(shard_story_count);
            max_stories_per_device = max_stories_per_device.max(shard_story_count);
            min_work_per_device = min_work_per_device.min(result.story_work);
            max_work_per_device = max_work_per_device.max(result.story_work);
            min_device_seconds = min_device_seconds.min(result.elapsed_seconds);
            max_device_seconds = max_device_seconds.max(result.elapsed_seconds);
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

        // One flat canonical mean across the original logical workers.
        let merge_started = Instant::now();
        let (merged, sync_changes) = {
            let model = canonical.model_mut()?;
            let merged = apply_mean_deltas(model, &deltas)?;
            let sync_changes = merged_parameter_changes(model, &deltas, &raw_changes)?;
            (merged, sync_changes)
        };

        // Pack changed canonical rows exactly once. GPU0 commits the packet to
        // its canonical runtime and every resident story lane; all secondary
        // devices consume the same immutable packet on their owner threads.
        let sync_update = Arc::new(PackedSparseModelUpdate::from_model(
            canonical.model(),
            &sync_changes,
        )?);
        canonical.commit_packed_host_model(&sync_update)?;
        canonical.reset_transient_state()?;
        let merge_seconds = merge_started.elapsed().as_secs_f64();

        let sync_started = Instant::now();
        self.synchronize_replicas(Arc::clone(&sync_update))?;
        let sync_seconds = sync_started.elapsed().as_secs_f64();

        eprintln!(
            "{{\"event\":\"multi_gpu_sparse_sync\",\"devices\":{},\"logical_stories\":{},\"stories_per_device_min\":{},\"stories_per_device_max\":{},\"work_per_device_min\":{},\"work_per_device_max\":{},\"device_seconds_min\":{:.6},\"device_seconds_max\":{:.6},\"exact_flat_story_mean\":true,\"canonical_device0_reused\":true,\"persistent_replica_threads\":true,\"threshold\":{},\"recurrent\":{},\"input\":{},\"output_neurons\":{},\"context_slots\":{},\"output_bias\":{},\"context_output\":{},\"merge_ms\":{:.3},\"replica_sync_ms\":{:.3}}}",
            device_count,
            stories.len(),
            min_stories_per_device,
            max_stories_per_device,
            min_work_per_device,
            max_work_per_device,
            min_device_seconds,
            max_device_seconds,
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

        let (replay, replay_seconds, replay_sync_seconds) = if multi_gpu_parallel_replay_enabled() {
            let outcome = self.run_parallel_replay(
                canonical,
                Arc::clone(&stories),
                losses_by_story,
                &ranges,
                device_count,
                permission,
            )?;
            (
                outcome.report,
                outcome.replay_seconds,
                outcome.replay_sync_seconds,
            )
        } else {
            // Historical fallback: replay every story/range serially on
            // canonical device0 and synchronize the resulting sparse image.
            canonical.enable_parameter_tracking();
            let replay_started = Instant::now();
            let replay = apply_batch_replay_policy(
                canonical,
                stories.as_slice(),
                &losses_by_story,
                permission,
            )?;
            let replay_seconds = replay_started.elapsed().as_secs_f64();
            let replay_sync_started = Instant::now();
            if replay.replay_steps > 0 {
                let replay_changes = canonical.parameter_changes()?;
                let replay_update = Arc::new(PackedSparseModelUpdate::from_model(
                    canonical.model(),
                    &replay_changes,
                )?);
                canonical.synchronize_packed_model(&replay_update)?;
                self.synchronize_replicas(replay_update)?;
            }
            let replay_sync_seconds = replay_sync_started.elapsed().as_secs_f64();
            if replay.replay_steps > 0 {
                eprintln!(
                    "{{\"event\":\"multi_gpu_serial_replay\",\"devices\":{},\"replay_steps\":{},\"prefix_steps\":{},\"replay_seconds\":{:.6},\"replay_sync_seconds\":{:.6},\"parallel_replay\":false}}",
                    device_count,
                    replay.replay_steps,
                    replay.prefix_steps,
                    replay_seconds,
                    replay_sync_seconds,
                );
            }
            (replay, replay_seconds, replay_sync_seconds)
        };
        activity.add(replay.activity);

        Ok(BatchTrainingReport {
            mean_loss: loss_sum / targets.max(1) as f64,
            targets,
            stories: stories.len(),
            activity,
            replay_segments: replay.replay_segments,
            replay_steps: replay.replay_steps,
            replay_prefix_steps: replay.prefix_steps,
            replay_seconds,
            replay_sync_seconds,
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
    let mut combined = ReplayTrainingReport::empty();
    for (story_index, (story, losses)) in stories.iter().zip(losses_by_story).enumerate() {
        let replay = apply_replay_policy(runtime, story, losses, permission, story_index)?;
        combined.add(replay);
    }
    if let Some(started) = batch_started {
        combined.timing.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    emit_replay_batch_debug(stories.len(), &combined);
    Ok(combined)
}

fn emit_replay_batch_debug(story_count: usize, combined: &ReplayTrainingReport) {
    if !replay_debug_options().summary {
        return;
    }
    eprintln!(
        "{{\"event\":\"replay_batch_debug\",\"stories\":{},\"segments\":{},\"target_steps\":{},\"prefix_steps\":{},\"execution_steps\":{},\"selection_ms\":{},\"begin_ms\":{},\"prefix_build_ms\":{},\"prefix_execute_ms\":{},\"target_build_ms\":{},\"target_execute_ms\":{},\"reset_ms\":{},\"total_ms\":{},\"prefix_steps_per_second\":{},\"target_steps_per_second\":{}}}",
        story_count,
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

fn apply_batch_replay_ranges(
    runtime: &mut BackendRuntime,
    stories: &[Vec<u8>],
    ranges_by_story: &[Vec<std::ops::Range<usize>>],
    permission: Permission,
) -> LeoResult<ReplayTrainingReport> {
    if stories.len() != ranges_by_story.len() {
        return Err(LeoError::internal(
            "replay story/range batch length mismatch",
        ));
    }
    let debug = replay_debug_options();
    let batch_started = debug.timing.then(Instant::now);
    let mut combined = ReplayTrainingReport::empty();
    for (story_index, (story, ranges)) in stories.iter().zip(ranges_by_story).enumerate() {
        emit_device_replay_selection_debug(runtime, story, ranges, story_index);
        let replay = execute_replay_ranges(runtime, story, ranges, permission, story_index, 0.0)?;
        combined.add(replay);
    }
    if let Some(started) = batch_started {
        combined.timing.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    emit_replay_batch_debug(stories.len(), &combined);
    Ok(combined)
}

fn device_story_report_is_postprocessed(
    report: &leo_core::DeviceStoryBatchReport,
    story_count: usize,
) -> bool {
    report.story_metrics.iter().all(Vec::is_empty)
        && report.story_summaries.len() == story_count
        && report.replay_ranges.len() == story_count
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
        let track_gpu_update = runtime.resolved_backend() == BackendKind::Gpu;
        if track_gpu_update {
            runtime.enable_parameter_tracking();
        }
        let initial = train_document_pass(runtime, &stories[0], permission)?;
        let replay_started = Instant::now();
        let replay = apply_replay_policy(runtime, &stories[0], &initial.losses, permission, 0)?;
        let replay_seconds = replay_started.elapsed().as_secs_f64();
        let replay_sync_started = Instant::now();
        if track_gpu_update {
            let changes = runtime.parameter_changes()?;
            let update = PackedSparseModelUpdate::from_model(runtime.model(), &changes)?;
            runtime.synchronize_packed_model(&update)?;
        }
        let replay_sync_seconds = replay_sync_started.elapsed().as_secs_f64();
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
            replay_seconds,
            replay_sync_seconds,
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
            let device_postprocessed =
                device_story_report_is_postprocessed(&device_report, stories.len());

            let mut losses_by_story = Vec::new();
            if device_postprocessed {
                for summary in &device_report.story_summaries {
                    loss_sum += summary.training_loss_sum;
                    targets = targets.saturating_add(summary.training_targets as usize);
                    activity.steps = activity.steps.saturating_add(summary.training_targets);
                    activity.active_neurons_sum = activity
                        .active_neurons_sum
                        .saturating_add(summary.active_neurons_sum);
                    activity.active_neurons_peak = activity
                        .active_neurons_peak
                        .max(summary.active_neurons_peak);
                    activity.emitted_events = activity
                        .emitted_events
                        .saturating_add(summary.synaptic_events);
                    activity.context_applied_steps = activity
                        .context_applied_steps
                        .saturating_add(summary.context_applied_steps);
                }
            } else {
                losses_by_story = Vec::with_capacity(device_report.story_metrics.len());
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
            }
            // Replay advances the canonical device after the story-mean packet
            // has already synchronized all resident lanes. Track replay alone
            // and sparse-commit its changed rows back into those lanes so the
            // next batch does not require a full canonical restore.
            runtime.enable_parameter_tracking();
            let replay_started = Instant::now();
            let replay = if device_postprocessed {
                apply_batch_replay_ranges(
                    runtime,
                    &stories,
                    &device_report.replay_ranges,
                    permission,
                )?
            } else {
                apply_batch_replay_policy(runtime, &stories, &losses_by_story, permission)?
            };
            let replay_seconds = replay_started.elapsed().as_secs_f64();
            let replay_sync_started = Instant::now();
            if replay.replay_steps > 0 {
                let replay_changes = runtime.parameter_changes()?;
                let replay_update =
                    PackedSparseModelUpdate::from_model(runtime.model(), &replay_changes)?;
                runtime.synchronize_packed_model(&replay_update)?;
            }
            let replay_sync_seconds = replay_sync_started.elapsed().as_secs_f64();
            activity.add(replay.activity);
            return Ok(BatchTrainingReport {
                mean_loss: loss_sum / targets.max(1) as f64,
                targets,
                stories: stories.len(),
                activity,
                replay_segments: replay.replay_segments,
                replay_steps: replay.replay_steps,
                replay_prefix_steps: replay.prefix_steps,
                replay_seconds,
                replay_sync_seconds,
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
    let replay_started = Instant::now();
    let replay =
        apply_batch_replay_policy(runtime, &trained_stories, &losses_by_story, permission)?;
    let replay_seconds = replay_started.elapsed().as_secs_f64();
    activity.add(replay.activity);

    Ok(BatchTrainingReport {
        mean_loss: weighted_loss / targets.max(1) as f64,
        targets,
        stories: deltas.len(),
        activity,
        replay_segments: replay.replay_segments,
        replay_steps: replay.replay_steps,
        replay_prefix_steps: replay.prefix_steps,
        replay_seconds,
        replay_sync_seconds: 0.0,
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
mod device_story_postprocess_tests {
    use super::device_story_report_is_postprocessed;
    use leo_core::{DeviceStoryBatchReport, DeviceStorySummary, MergeMetrics, StepMetrics};

    #[test]
    fn one_empty_metrics_vector_per_story_uses_device_postprocess_results() {
        let story_count = 4;
        let mut report = DeviceStoryBatchReport {
            story_metrics: vec![Vec::new(); story_count],
            story_summaries: vec![DeviceStorySummary::default(); story_count],
            replay_ranges: vec![Vec::new(); story_count],
            merge: MergeMetrics::default(),
        };

        assert!(device_story_report_is_postprocessed(&report, story_count));

        report.story_metrics[2].push(StepMetrics::default());
        assert!(!device_story_report_is_postprocessed(&report, story_count));

        report.story_metrics[2].clear();
        report.story_summaries.pop();
        assert!(!device_story_report_is_postprocessed(&report, story_count));

        report.story_summaries.push(DeviceStorySummary::default());
        report.replay_ranges.pop();
        assert!(!device_story_report_is_postprocessed(&report, story_count));
    }
}

#[cfg(test)]
mod multi_gpu_partition_tests {
    use super::{balanced_story_range, story_work, work_balanced_story_ranges};

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
    fn work_balanced_ranges_cover_stories_and_reduce_straggler_work() {
        let stories = vec![
            vec![b'a'; 100],
            vec![b'b'; 1],
            vec![b'c'; 1],
            vec![b'd'; 1],
            vec![b'e'; 1],
            vec![b'f'; 1],
        ];
        let ranges = work_balanced_story_ranges(&stories, 2);
        assert_eq!(ranges.first().expect("range").start, 0);
        assert_eq!(ranges.last().expect("range").end, stories.len());
        assert_eq!(ranges[0].end, ranges[1].start);

        let work = ranges
            .iter()
            .map(|range| {
                stories[range.clone()]
                    .iter()
                    .map(|story| story_work(story))
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        let count_ranges = [
            balanced_story_range(stories.len(), 2, 0),
            balanced_story_range(stories.len(), 2, 1),
        ];
        let count_work = count_ranges
            .iter()
            .map(|range| {
                stories[range.clone()]
                    .iter()
                    .map(|story| story_work(story))
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert!(work.iter().max() <= count_work.iter().max());
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
