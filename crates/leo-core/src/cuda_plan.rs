//! CUDA execution planning, lightweight telemetry, and online autotuning.
//!
//! These choices are execution-only. The logical story batch, learning
//! equations, update barriers, replay policy, story order, and FP32 arithmetic
//! are fixed by the semantics contract. The tuner may change launch geometry
//! and physical lane chunking only.

use crate::{LeoError, LeoResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const PROFILE_SCHEMA: u32 = 4;
const OBSERVATIONS_PER_CANDIDATE: u32 = 2;
const PROFILE_SAMPLE_INTERVAL: u64 = 128;
const LANE_CANDIDATES: [usize; 6] = [8, 16, 32, 64, 96, 128];
const SPARSE_THREAD_CANDIDATES: [u32; 3] = [128, 256, 512];
const FUSED_WAVEFRONT_THREADS: u32 = 256;
static CACHE_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct CudaExecutionPlan {
    pub physical_lane_chunk: usize,
    pub sparse_apply_blocks: u32,
    pub sparse_apply_threads: u32,
    pub fused_wavefront_blocks: u32,
    pub fused_wavefront_threads: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaTuningLimits {
    pub multiprocessors: u32,
    pub max_threads_per_sm: u32,
    pub max_lanes: usize,
    /// Maximum cooperative grid blocks that can be resident concurrently for
    /// fused-wavefront kernels at 128/256/512 threads respectively.
    pub fused_blocks_128: u32,
    pub fused_blocks_256: u32,
    pub fused_blocks_512: u32,
    /// Compiled cooperative capacity of the production grouped persistent
    /// kernel at its exact 256-thread reduction shape.
    pub grouped_blocks_256: u32,
}

impl CudaTuningLimits {
    fn max_fused_blocks(self, threads: u32) -> u32 {
        match threads {
            128 => self.fused_blocks_128,
            256 => {
                if self.grouped_blocks_256 == 0 {
                    self.fused_blocks_256
                } else {
                    self.fused_blocks_256.min(self.grouped_blocks_256)
                }
            }
            512 => self.fused_blocks_512,
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaBatchTelemetry {
    pub h2d_ms: f32,
    pub compute_ms: f32,
    pub d2h_ms: f32,
    pub host_wait_ms: f32,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub theoretical_occupancy: f32,
    pub fused_launches: u64,
    pub direct_phase_launches: u64,
    pub graph_launches: u64,
}

impl CudaBatchTelemetry {
    pub(crate) fn h2d_gib_per_second(self) -> f64 {
        gib_per_second(self.h2d_bytes, self.h2d_ms)
    }

    pub(crate) fn d2h_gib_per_second(self) -> f64 {
        gib_per_second(self.d2h_bytes, self.d2h_ms)
    }

    fn dominant_host_visible_bottleneck(self) -> &'static str {
        let phases = [
            ("h2d", self.h2d_ms),
            ("compute", self.compute_ms),
            ("d2h", self.d2h_ms),
            ("host_wait", self.host_wait_ms),
        ];
        phases
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(name, _)| name)
            .unwrap_or("unknown")
    }
}

fn gib_per_second(bytes: u64, milliseconds: f32) -> f64 {
    if bytes == 0 || milliseconds <= 0.0 {
        0.0
    } else {
        (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (milliseconds as f64 / 1000.0)
    }
}

#[derive(Clone, Debug, Default)]
struct CandidateScore {
    weighted_work: f64,
    seconds: f64,
    observations: u32,
    telemetry: TelemetryAggregate,
}

impl CandidateScore {
    fn throughput(&self) -> f64 {
        if self.seconds <= 0.0 {
            0.0
        } else {
            self.weighted_work / self.seconds
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct TelemetryAggregate {
    samples: u32,
    h2d_ms: f64,
    compute_ms: f64,
    d2h_ms: f64,
    host_wait_ms: f64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    occupancy_sum: f64,
    fused_launches: u64,
    direct_phase_launches: u64,
    graph_launches: u64,
}

impl TelemetryAggregate {
    fn add(&mut self, sample: CudaBatchTelemetry) {
        self.samples = self.samples.saturating_add(1);
        self.h2d_ms += sample.h2d_ms as f64;
        self.compute_ms += sample.compute_ms as f64;
        self.d2h_ms += sample.d2h_ms as f64;
        self.host_wait_ms += sample.host_wait_ms as f64;
        self.h2d_bytes = self.h2d_bytes.saturating_add(sample.h2d_bytes);
        self.d2h_bytes = self.d2h_bytes.saturating_add(sample.d2h_bytes);
        self.occupancy_sum += sample.theoretical_occupancy as f64;
        self.fused_launches = self.fused_launches.saturating_add(sample.fused_launches);
        self.direct_phase_launches = self
            .direct_phase_launches
            .saturating_add(sample.direct_phase_launches);
        self.graph_launches = self.graph_launches.saturating_add(sample.graph_launches);
    }

    fn mean_occupancy(&self) -> f64 {
        self.occupancy_sum / self.samples.max(1) as f64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCandidateProgress {
    plan: CudaExecutionPlan,
    weighted_work: f64,
    seconds: f64,
    observations: u32,
    telemetry: TelemetryAggregate,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedCudaProfile {
    schema: u32,
    key: String,
    logical_lanes: usize,
    complete: bool,
    plan: CudaExecutionPlan,
    measured_work_per_second: f64,
    candidate_index: usize,
    progress: Vec<PersistedCandidateProgress>,
}

pub(crate) struct CudaExecutionTuner {
    key: String,
    profile_root: Option<PathBuf>,
    limits: CudaTuningLimits,
    active_logical_lanes: usize,
    plan: CudaExecutionPlan,
    candidates: Vec<CudaExecutionPlan>,
    candidate_index: usize,
    scores: BTreeMap<CudaExecutionPlan, CandidateScore>,
    tuning_complete: bool,
    batch_sequence: u64,
}

impl CudaExecutionTuner {
    pub(crate) fn new(key: String, limits: CudaTuningLimits) -> Self {
        let logical_lanes = limits.max_lanes.max(1);
        let plan = heuristic_plan(limits, logical_lanes);
        Self {
            key,
            profile_root: cache_root().map(|root| root.join("profiles")),
            limits,
            active_logical_lanes: 0,
            plan,
            candidates: Vec::new(),
            candidate_index: 0,
            scores: BTreeMap::new(),
            tuning_complete: false,
            batch_sequence: 0,
        }
    }

    pub(crate) fn plan_for(&mut self, logical_lanes: usize) -> CudaExecutionPlan {
        self.activate(logical_lanes);
        self.plan
    }

    /// Profiling is continuous while a plan is being tuned, then sampled at a
    /// low fixed cadence so regressions remain visible without taxing every
    /// production batch.
    pub(crate) fn should_profile_batch(&mut self) -> bool {
        self.batch_sequence = self.batch_sequence.saturating_add(1);
        !self.tuning_complete || self.batch_sequence % PROFILE_SAMPLE_INTERVAL == 0
    }

    /// Observe a complete logical story batch. `work_units` is executed story
    /// steps, so varying story lengths remain comparable. Selection is based on
    /// end-to-end throughput; telemetry explains *why* a plan wins or loses.
    pub(crate) fn observe_batch(
        &mut self,
        logical_lanes: usize,
        work_units: u64,
        elapsed: Duration,
        telemetry: Option<CudaBatchTelemetry>,
    ) {
        if logical_lanes == 0 || work_units == 0 || elapsed.is_zero() {
            return;
        }
        self.activate(logical_lanes);

        if let Some(sample) = telemetry {
            self.emit_telemetry(sample, work_units, elapsed);
        }
        if self.tuning_complete {
            return;
        }

        let observations = {
            let score = self.scores.entry(self.plan).or_default();
            score.weighted_work += work_units as f64;
            score.seconds += elapsed.as_secs_f64();
            score.observations = score.observations.saturating_add(1);
            if let Some(sample) = telemetry {
                score.telemetry.add(sample);
            }
            score.observations
        };
        if observations < OBSERVATIONS_PER_CANDIDATE {
            self.persist_progress_best_effort();
            return;
        }

        if self.candidate_index + 1 < self.candidates.len() {
            self.candidate_index += 1;
            self.plan = self.candidates[self.candidate_index];
            self.persist_progress_best_effort();
            return;
        }

        let best = self
            .scores
            .iter()
            .max_by(|left, right| {
                left.1
                    .throughput()
                    .partial_cmp(&right.1.throughput())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(plan, score)| (*plan, score.throughput()));
        if let Some((best_plan, throughput)) = best {
            self.plan = best_plan;
            self.tuning_complete = true;
            if let Err(error) = self.persist_profile(true, throughput) {
                emit_cache_error(&error);
            }
            self.emit_tuning_summary(throughput);
        }
    }

    fn activate(&mut self, logical_lanes: usize) {
        let logical_lanes = logical_lanes.clamp(1, self.limits.max_lanes.max(1));
        if self.active_logical_lanes == logical_lanes {
            return;
        }

        self.active_logical_lanes = logical_lanes;
        self.plan = heuristic_plan(self.limits, logical_lanes);
        self.candidates = build_candidates(self.limits, logical_lanes, self.plan);
        self.candidate_index = 0;
        self.scores.clear();
        self.tuning_complete = false;

        if let Some(profile) = self.load_profile(logical_lanes) {
            if profile.complete {
                self.plan = sanitize_plan(profile.plan, self.limits, logical_lanes);
                self.candidates = vec![self.plan];
                self.tuning_complete = true;
            } else {
                self.restore_progress(profile, logical_lanes);
            }
        }
    }

    fn profile_path(&self, logical_lanes: usize) -> Option<PathBuf> {
        self.profile_root
            .as_ref()
            .map(|root| root.join(format!("{}-lanes{logical_lanes}.toml", self.key)))
    }

    fn load_profile(&self, logical_lanes: usize) -> Option<PersistedCudaProfile> {
        let path = self.profile_path(logical_lanes)?;
        let text = fs::read_to_string(path).ok()?;
        let profile: PersistedCudaProfile = toml::from_str(&text).ok()?;
        (profile.schema == PROFILE_SCHEMA
            && profile.key == self.key
            && profile.logical_lanes == logical_lanes)
            .then_some(profile)
    }

    fn restore_progress(&mut self, profile: PersistedCudaProfile, logical_lanes: usize) {
        for saved in profile.progress {
            let plan = sanitize_plan(saved.plan, self.limits, logical_lanes);
            if !self.candidates.contains(&plan) {
                continue;
            }
            self.scores.insert(
                plan,
                CandidateScore {
                    weighted_work: saved.weighted_work,
                    seconds: saved.seconds,
                    observations: saved.observations.min(OBSERVATIONS_PER_CANDIDATE),
                    telemetry: saved.telemetry,
                },
            );
        }

        let restored_plan = sanitize_plan(profile.plan, self.limits, logical_lanes);
        self.candidate_index = self
            .candidates
            .iter()
            .position(|candidate| *candidate == restored_plan)
            .unwrap_or_else(|| {
                profile
                    .candidate_index
                    .min(self.candidates.len().saturating_sub(1))
            });
        self.plan = self.candidates[self.candidate_index];
        self.tuning_complete = false;
        eprintln!(
            "{{\"event\":\"cuda_autotune_resumed\",\"logical_lanes\":{},\"candidate_index\":{},\"candidate_count\":{},\"restored_scores\":{}}}",
            logical_lanes,
            self.candidate_index,
            self.candidates.len(),
            self.scores.len(),
        );
    }

    fn persist_progress_best_effort(&self) {
        if let Err(error) = self.persist_profile(false, 0.0) {
            emit_cache_error(&error);
        }
    }

    fn persist_profile(&self, complete: bool, throughput: f64) -> LeoResult<()> {
        let logical_lanes = self.active_logical_lanes.max(1);
        let Some(path) = self.profile_path(logical_lanes) else {
            return Ok(());
        };
        let progress = self
            .scores
            .iter()
            .map(|(plan, score)| PersistedCandidateProgress {
                plan: *plan,
                weighted_work: score.weighted_work,
                seconds: score.seconds,
                observations: score.observations,
                telemetry: score.telemetry.clone(),
            })
            .collect();
        let profile = PersistedCudaProfile {
            schema: PROFILE_SCHEMA,
            key: self.key.clone(),
            logical_lanes,
            complete,
            plan: self.plan,
            measured_work_per_second: throughput,
            candidate_index: self.candidate_index,
            progress,
        };
        let text = toml::to_string(&profile)
            .map_err(|error| LeoError::cuda(format!("could not encode CUDA profile: {error}")))?;
        atomic_write(&path, text.as_bytes())
    }

    fn emit_telemetry(&self, sample: CudaBatchTelemetry, work_units: u64, elapsed: Duration) {
        eprintln!(
            "{{\"event\":\"cuda_profile\",\"logical_lanes\":{},\"lane_chunk\":{},\"sparse_blocks\":{},\"sparse_threads\":{},\"fused_blocks\":{},\"fused_threads\":{},\"work_units\":{},\"wall_ms\":{},\"h2d_ms\":{},\"compute_ms\":{},\"d2h_ms\":{},\"host_wait_ms\":{},\"h2d_gib_s\":{},\"d2h_gib_s\":{},\"theoretical_occupancy\":{},\"fused_launches\":{},\"direct_phase_launches\":{},\"graph_launches\":{},\"host_visible_bottleneck\":\"{}\"}}",
            self.active_logical_lanes,
            self.plan.physical_lane_chunk,
            self.plan.sparse_apply_blocks,
            self.plan.sparse_apply_threads,
            self.plan.fused_wavefront_blocks,
            self.plan.fused_wavefront_threads,
            work_units,
            elapsed.as_secs_f64() * 1000.0,
            sample.h2d_ms,
            sample.compute_ms,
            sample.d2h_ms,
            sample.host_wait_ms,
            sample.h2d_gib_per_second(),
            sample.d2h_gib_per_second(),
            sample.theoretical_occupancy,
            sample.fused_launches,
            sample.direct_phase_launches,
            sample.graph_launches,
            sample.dominant_host_visible_bottleneck(),
        );
    }

    fn emit_tuning_summary(&self, throughput: f64) {
        let telemetry = self.scores.get(&self.plan).map(|score| &score.telemetry);
        eprintln!(
            "{{\"event\":\"cuda_autotune_complete\",\"logical_lanes\":{},\"candidate_count\":{},\"work_units_per_second\":{},\"lane_chunk\":{},\"sparse_blocks\":{},\"sparse_threads\":{},\"fused_blocks\":{},\"fused_threads\":{},\"mean_theoretical_occupancy\":{}}}",
            self.active_logical_lanes,
            self.candidates.len(),
            throughput,
            self.plan.physical_lane_chunk,
            self.plan.sparse_apply_blocks,
            self.plan.sparse_apply_threads,
            self.plan.fused_wavefront_blocks,
            self.plan.fused_wavefront_threads,
            telemetry.map(TelemetryAggregate::mean_occupancy).unwrap_or(0.0),
        );
    }
}

fn emit_cache_error(error: &LeoError) {
    eprintln!(
        "{{\"event\":\"cuda_autotune_cache_error\",\"message\":\"{}\"}}",
        error.message().replace('\\', "\\\\").replace('"', "\\\"")
    );
}

fn heuristic_plan(limits: CudaTuningLimits, logical_lanes: usize) -> CudaExecutionPlan {
    let sparse_threads = 256;
    let fused_threads = FUSED_WAVEFRONT_THREADS;
    CudaExecutionPlan {
        physical_lane_chunk: heuristic_lane_chunk(limits.multiprocessors, logical_lanes),
        sparse_apply_blocks: limits.multiprocessors.saturating_mul(2).max(1),
        sparse_apply_threads: sparse_threads,
        fused_wavefront_blocks: limits.max_fused_blocks(fused_threads),
        fused_wavefront_threads: fused_threads,
    }
}

fn sanitize_plan(
    mut plan: CudaExecutionPlan,
    limits: CudaTuningLimits,
    logical_lanes: usize,
) -> CudaExecutionPlan {
    plan.physical_lane_chunk = plan.physical_lane_chunk.clamp(1, logical_lanes.max(1));
    plan.sparse_apply_blocks = plan.sparse_apply_blocks.max(1);
    plan.sparse_apply_threads = nearest_thread_candidate(
        plan.sparse_apply_threads
            .min(limits.max_threads_per_sm.max(32)),
    );
    // The fused wavefront contains the forward reduction. Its 256-thread
    // reduction order is part of the FP32 execution contract, so it is not an
    // autotuning dimension. Grid width remains freely tunable.
    plan.fused_wavefront_threads = FUSED_WAVEFRONT_THREADS;
    let max_fused = limits.max_fused_blocks(FUSED_WAVEFRONT_THREADS);
    plan.fused_wavefront_blocks = if max_fused == 0 {
        0
    } else {
        plan.fused_wavefront_blocks.clamp(1, max_fused)
    };
    plan
}

fn build_candidates(
    limits: CudaTuningLimits,
    logical_lanes: usize,
    baseline: CudaExecutionPlan,
) -> Vec<CudaExecutionPlan> {
    // Exact v1 story batching no longer performs a per-byte sparse apply/reset,
    // so sparse_apply_* is legacy execution metadata rather than an active
    // tuning dimension. Search only dimensions that affect this fast path: the
    // number of concurrently staged story lanes and the cooperative fused-grid
    // width. This keeps online tuning short and makes persisted progress useful.
    let mut candidates = Vec::with_capacity(16);
    push_unique(&mut candidates, baseline);

    for lane in LANE_CANDIDATES {
        let mut plan = baseline;
        plan.physical_lane_chunk = lane.min(logical_lanes).max(1);
        push_unique(&mut candidates, plan);
    }

    let max_fused = limits.max_fused_blocks(FUSED_WAVEFRONT_THREADS);
    if max_fused > 0 {
        for divisor in [4u32, 2, 1] {
            let mut plan = baseline;
            plan.fused_wavefront_blocks = max_fused.div_ceil(divisor).max(1);
            push_unique(&mut candidates, plan);
        }

        // A small deterministic interaction set catches lane/grid coupling
        // without spending production batches on dimensions that are inactive
        // in the exact story-batch path.
        let half_grid = max_fused.div_ceil(2).max(1);
        for lane in LANE_CANDIDATES {
            let mut plan = baseline;
            plan.physical_lane_chunk = lane.min(logical_lanes).max(1);
            plan.fused_wavefront_blocks = half_grid;
            push_unique(&mut candidates, plan);
        }
    }

    candidates
        .into_iter()
        .map(|plan| sanitize_plan(plan, limits, logical_lanes))
        .fold(Vec::new(), |mut unique, plan| {
            push_unique(&mut unique, plan);
            unique
        })
}

fn push_unique(plans: &mut Vec<CudaExecutionPlan>, plan: CudaExecutionPlan) {
    if !plans.contains(&plan) {
        plans.push(plan);
    }
}

fn heuristic_lane_chunk(multiprocessors: u32, max_lanes: usize) -> usize {
    let target = (multiprocessors as usize).saturating_mul(2).clamp(16, 128);
    target.min(max_lanes.max(1))
}

fn nearest_thread_candidate(value: u32) -> u32 {
    *SPARSE_THREAD_CANDIDATES
        .iter()
        .min_by_key(|candidate| candidate.abs_diff(value))
        .unwrap_or(&256)
}

pub(crate) fn cache_root() -> Option<PathBuf> {
    if let Some(path) = env::var_os("LEO_CACHE_DIR") {
        if !path.is_empty() {
            return Some(PathBuf::from(path).join("cuda"));
        }
    }
    if let Some(path) = env::var_os("XDG_CACHE_HOME") {
        if !path.is_empty() {
            return Some(PathBuf::from(path).join("leo").join("cuda"));
        }
    }
    env::var_os("HOME")
        .filter(|path| !path.is_empty())
        .map(|home| PathBuf::from(home).join(".cache").join("leo").join("cuda"))
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> LeoResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| LeoError::cuda("CUDA cache path has no parent directory"))?;
    fs::create_dir_all(parent).map_err(|error| {
        LeoError::cuda(format!(
            "could not create CUDA cache {}: {error}",
            parent.display()
        ))
    })?;
    let sequence = CACHE_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp.{}.{}", std::process::id(), sequence));
    let mut file = fs::File::create(&tmp).map_err(|error| {
        LeoError::cuda(format!(
            "could not create CUDA cache {}: {error}",
            tmp.display()
        ))
    })?;
    file.write_all(bytes).map_err(|error| {
        LeoError::cuda(format!(
            "could not write CUDA cache {}: {error}",
            tmp.display()
        ))
    })?;
    file.sync_all().map_err(|error| {
        LeoError::cuda(format!(
            "could not sync CUDA cache {}: {error}",
            tmp.display()
        ))
    })?;
    drop(file);
    fs::rename(&tmp, path).map_err(|error| {
        let _ = fs::remove_file(&tmp);
        LeoError::cuda(format!(
            "could not publish CUDA cache {}: {error}",
            path.display()
        ))
    })?;
    #[cfg(unix)]
    {
        fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                LeoError::cuda(format!(
                    "could not sync CUDA cache directory {}: {error}",
                    parent.display()
                ))
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> CudaTuningLimits {
        CudaTuningLimits {
            multiprocessors: 80,
            max_threads_per_sm: 2048,
            max_lanes: 128,
            fused_blocks_128: 320,
            fused_blocks_256: 160,
            fused_blocks_512: 80,
            grouped_blocks_256: 160,
        }
    }

    #[test]
    fn heuristic_is_bounded_by_logical_capacity() {
        assert_eq!(heuristic_lane_chunk(80, 64), 64);
        assert_eq!(heuristic_lane_chunk(4, 128), 16);
        assert_eq!(heuristic_lane_chunk(1, 8), 8);
    }

    #[test]
    fn plan_never_changes_logical_lane_count() {
        let mut tuner = CudaExecutionTuner::new("test-no-cache".to_string(), limits());
        let plan = tuner.plan_for(37);
        assert!((1..=37).contains(&plan.physical_lane_chunk));
    }

    #[test]
    fn tuning_searches_only_dimensions_used_by_exact_story_batch() {
        let baseline = heuristic_plan(limits(), 64);
        let candidates = build_candidates(limits(), 64, baseline);
        assert!(candidates
            .iter()
            .any(|plan| plan.physical_lane_chunk != baseline.physical_lane_chunk));
        assert!(candidates
            .iter()
            .all(|plan| plan.sparse_apply_threads == baseline.sparse_apply_threads));
        assert!(candidates
            .iter()
            .all(|plan| plan.sparse_apply_blocks == baseline.sparse_apply_blocks));
        assert!(candidates
            .iter()
            .any(|plan| plan.fused_wavefront_blocks != baseline.fused_wavefront_blocks));
        assert!(candidates
            .iter()
            .all(|plan| plan.fused_wavefront_threads == FUSED_WAVEFRONT_THREADS));
    }

    #[test]
    fn partial_batches_do_not_share_a_profile_identity_with_full_batches() {
        let mut tuner = CudaExecutionTuner::new("test-lane-identity".to_string(), limits());
        let partial = tuner.plan_for(7);
        let full = tuner.plan_for(64);
        assert!(partial.physical_lane_chunk <= 7);
        assert!(full.physical_lane_chunk <= 64);
        assert_eq!(tuner.active_logical_lanes, 64);
    }
    #[test]
    fn incomplete_profile_resumes_candidate_progress() {
        let root = std::env::temp_dir().join(format!(
            "leo-cuda-plan-resume-{}-{}",
            std::process::id(),
            CACHE_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);

        let mut first = CudaExecutionTuner::new("resume-test".to_string(), limits());
        first.profile_root = Some(root.clone());
        let plan = first.plan_for(16);
        first.observe_batch(
            16,
            128,
            Duration::from_millis(10),
            Some(CudaBatchTelemetry::default()),
        );
        assert!(!first.tuning_complete);
        assert_eq!(
            first.scores.get(&plan).map(|score| score.observations),
            Some(1)
        );

        let mut resumed = CudaExecutionTuner::new("resume-test".to_string(), limits());
        resumed.profile_root = Some(root.clone());
        assert_eq!(resumed.plan_for(16), plan);
        assert_eq!(
            resumed.scores.get(&plan).map(|score| score.observations),
            Some(1)
        );
        resumed.observe_batch(16, 128, Duration::from_millis(10), None);
        assert_eq!(resumed.candidate_index, 1);
        assert_ne!(resumed.plan, plan);

        let _ = fs::remove_dir_all(root);
    }
}
