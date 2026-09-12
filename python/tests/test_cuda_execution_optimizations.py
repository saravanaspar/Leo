import math
import random
import runpy
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CudaExecutionOptimizationTests(unittest.TestCase):
    def test_exact_worklists_are_in_the_canonical_abi_and_hot_kernels(self):
        abi = (ROOT / "crates/leo-core/cuda_abi.def").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()

        for field in (
            "TOUCHED_LIST",
            "TOUCHED_COUNT",
            "LEARNING_DESTINATION_LIST",
            "LEARNING_DESTINATION_COUNT",
        ):
            self.assertIn(field, abi)
        for kernel in (
            "leo_shared_cache_surrogate_worklist",
            "leo_shared_learning_signals_worklist",
            "leo_shared_homeostasis_worklist",
        ):
            self.assertIn(kernel, cuda)
            self.assertIn(kernel, rust)
        self.assertNotIn("leo_shared_cache_surrogate_tiled", rust)

    def test_physical_batching_preserves_one_story_end_mean_barrier(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        planner = (ROOT / "crates/leo-core/src/cuda_plan.rs").read_text()
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()

        batch = cuda.split("fn train_story_batch_shared_device", 1)[1].split(
            "fn cuda_execution_profile_key", 1
        )[0]
        self.assertIn("physical_lane_chunk", batch)
        self.assertIn("snapshot_batch_lane_values", batch)
        self.assertIn("snapshot_batch_lane_context_values_batched", batch)
        self.assertIn("SparseModelDelta::from_tracked_values", batch)
        self.assertIn("launch_device_batch_fixed_merge", batch)
        self.assertIn("cuda_device_batch_merge_enabled()", batch)
        # The default reducer performs fixed numeric rows on-device and keeps
        # only the keyed context merge on the host. The complete historical
        # host mean remains behind LEO_CUDA_DEVICE_BATCH_MERGE=0.
        self.assertEqual(batch.count("apply_mean_deltas("), 2)
        self.assertNotIn("launch_sparse_apply_and_reset(", batch)
        self.assertIn("copy_canonical_parameters_to_batch_lane_async", batch)
        self.assertIn("model_revision", batch)
        self.assertIn("plan_never_changes_logical_lane_count", planner)
        self.assertNotIn("workers = execution_plan", cuda)

        # Unlike the old source-string-only gate, the GPU backend probe now
        # executes this production story-batch path against the CPU reference.
        self.assertIn("cuda_story_batch_conformance", main)
        self.assertIn("run_story_batch_conformance(&model, 0.0)", main)
        self.assertIn("run_story_batch_conformance(&model, 0.30)", main)

        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()
        lane_update = kernels.split("leo_shared_phase_post_deltas_lane", 1)[1].split(
            "leo_shared_phase_homeostasis_lane", 1
        )[0]
        for direct_update in (
            "leo_p_update_output",
            "leo_p_update_context",
            "leo_p_update_recurrent_weights",
            "leo_p_update_input_weights",
            "leo_p_inhibitory_homeostasis",
        ):
            self.assertIn(direct_update, lane_update)
        for cross_story_accumulator in (
            "leo_p_accumulate_output_delta",
            "leo_p_accumulate_context_delta",
            "leo_p_accumulate_recurrent_delta",
            "leo_p_accumulate_input_delta",
            "leo_p_accumulate_inhibitory_delta",
        ):
            self.assertNotIn(cross_story_accumulator, lane_update)

    def test_tuning_cache_and_async_transfers_are_execution_only(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        planner = (ROOT / "crates/leo-core/src/cuda_plan.rs").read_text()
        semantics = (ROOT / "docs/SEMANTICS.md").read_text()

        for marker in (
            "leo-ptx-v1",
            "cuMemcpyHtoDAsync",
            "cuMemcpyDtoHAsync",
            "cuMemcpyDtoDAsync",
            "cuMemAllocHost",
        ):
            self.assertIn(marker, cuda)
        self.assertIn("OBSERVATIONS_PER_CANDIDATE", planner)
        self.assertIn("PersistedCandidateProgress", planner)
        self.assertIn("persist_progress_best_effort", planner)
        self.assertIn("cuda_autotune_resumed", planner)
        self.assertIn("logical story batch selected by `--workers`", semantics)
        self.assertIn("not autotuning knobs", semantics)

    def test_dataset_prefetch_preserves_verified_artifact_path(self):
        dataset_module = (
            ROOT / "crates/leo-cli/src/training/dataset.rs"
        ).read_text()
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        data = (ROOT / "crates/leo-data/src/lib.rs").read_text()
        tdd = (ROOT / "docs/TDD.md").read_text()

        self.assertIn("StoryBatchPrefetcher", dataset_module)
        self.assertIn("sync_channel::<StoryBatchRequest>(1)", dataset_module)
        self.assertIn("dataset.try_clone()", dataset_module)
        self.assertNotIn("struct StoryBatchPrefetcher", main)
        self.assertIn("read_exact_at", data)
        self.assertIn("one-batch-ahead dataset prefetch", tdd)

    def test_fused_wavefront_and_dual_stream_pipeline_are_real(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        fused = kernels.split(
            'extern "C" __global__ void leo_shared_wavefront_fused', 1
        )[1].split(
            'extern "C" __global__ void leo_apply_shared_wavefront_deltas', 1
        )[0]
        for helper in (
            "leo_shared_phase_pre_lane",
            "leo_shared_phase_select_block",
            "leo_shared_phase_post_select_lane",
            "leo_shared_phase_cache_surrogate_lane",
            "leo_shared_phase_post_core_lane",
            "leo_shared_phase_learning_signals_lane",
            "leo_shared_phase_post_deltas_lane",
            "leo_shared_phase_homeostasis_lane",
            "leo_shared_phase_capture_lane",
        ):
            self.assertIn(helper, fused)
        self.assertIn("cooperative_groups::this_grid()", fused)
        self.assertGreaterEqual(fused.count("grid.sync()"), 8)
        self.assertIn("shared_wavefront_fused", cuda)
        self.assertIn("compute_stream", cuda)
        self.assertIn("transfer_stream", cuda)
        self.assertIn("cuStreamWaitEvent(upload ready)", cuda)
        self.assertIn("cuEventRecord(upload ready)", cuda)
        self.assertIn("cuEventElapsedTime", cuda)

    def test_autotuning_identity_and_hardware_profiling_are_complete(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        planner = (ROOT / "crates/leo-core/src/cuda_plan.rs").read_text()
        profiler = (ROOT / "scripts/profile_cuda.sh").read_text()

        for marker in (
            "cuDeviceGetName",
            "cuDeviceGetUuid",
            "cuDeviceGetPCIBusId",
            "cuDriverGetVersion",
            "cuDeviceTotalMem",
        ):
            self.assertIn(marker, cuda)
        self.assertIn("leo-cuda-plan-v4", cuda)
        self.assertIn("pci_bus_id", cuda)
        self.assertIn("cuFuncGetAttribute", cuda)
        self.assertIn("cuda_kernel_resources", cuda)
        self.assertIn("registers_per_thread", cuda)
        self.assertIn("local_bytes_per_thread", cuda)
        self.assertIn("cooperative_capacity_blocks", cuda)
        for field in ("sparse_apply_blocks", "sparse_apply_threads", "fused_wavefront_blocks"):
            self.assertIn(field, planner)
        self.assertIn("--set full", profiler)
        self.assertIn("--page details", profiler)
        self.assertIn("MemoryWorkloadAnalysis", profiler)
        self.assertIn("WarpStateStats", profiler)

        gpu_ci = (ROOT / ".github/workflows/gpu-ci.yml").read_text()
        gpu_check = (ROOT / "scripts/check_gpu.sh").read_text()
        self.assertIn("LEO_GPU_RUNNER", gpu_ci)
        self.assertIn("cuda_autotune_complete", gpu_check)
        self.assertIn("cuda_profile", gpu_check)
        self.assertIn("cuda_autotune_resumed", gpu_check)
        self.assertIn("cuda_story_batch_conformance", gpu_check)


    def test_replay_prefix_uses_frozen_state_fast_path_without_changing_replay_budget(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        self.assertIn("runtime.advance_frozen_batch(&prefix)?;", training)
        self.assertIn("fn advance_frozen_batch", backend)
        self.assertIn("leo_advance_frozen_persistent", cuda)
        self.assertIn("leo_advance_frozen_persistent", kernels)
        self.assertIn("leo_advance_frozen_cooperative", cuda)
        self.assertIn("leo_advance_frozen_cooperative", kernels)
        self.assertIn("frozen_grid_blocks", cuda)
        self.assertIn("launch_cooperative_exact", cuda)
        frozen_kernel = kernels.split(
            'extern "C" __global__ void leo_advance_frozen_persistent', 1
        )[0].rsplit("__device__ void leo_advance_frozen_story_block", 1)[1]
        self.assertIn("leo_p_context_advance_history", frozen_kernel)
        self.assertIn("leo_p_post_and_emit", frozen_kernel)
        self.assertNotIn("leo_p_forward", frozen_kernel)
        self.assertNotIn("leo_p_capture_training_step", frozen_kernel)
        self.assertNotIn("leo_p_cache_surrogate", frozen_kernel)
        cooperative = kernels.split(
            "leo_advance_frozen_cooperative_body", 1
        )[1].split(
            'extern "C" __global__ void leo_advance_frozen_cooperative', 1
        )[0]
        self.assertIn("cooperative_groups::this_grid()", cooperative)
        self.assertIn("leo_profile_phase_end", cooperative)
        self.assertIn("leo_p_select_model_block", cooperative)
        self.assertIn("leo_p_context_advance_history", cooperative)
        self.assertIn("leo_p_select_global", cooperative)
        self.assertIn("leo_p_post_and_emit_grid", cooperative)
        self.assertNotIn("leo_p_forward", cooperative)
        self.assertNotIn("leo_p_capture_training_step", cooperative)
        self.assertIn("runtime.model().config.replay.fraction", training)
        self.assertIn("runtime.model().config.replay.segment_bytes", training)
        self.assertIn("let cleanup_after = index + 1 == ranges.len();", training)
        self.assertIn("if cleanup_after {", training)

        benchmark = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        for field in (
            "base_training_targets",
            "replay_fraction",
            "replay_segments",
            "replay_steps",
            "replay_step_fraction",
        ):
            self.assertIn(field, benchmark)

    def test_sampled_cuda_phase_profiler_is_opt_in_and_math_neutral(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        self.assertIn("LEO_CUDA_PHASE_PROFILE", cuda)
        self.assertIn("LEO_CUDA_PHASE_PROFILE_STRIDE", cuda)
        self.assertIn("cuda_phase_profile", cuda)
        self.assertIn("shared_story_batch_production", cuda)
        self.assertIn("profiled_capacity_blocks", cuda)
        self.assertIn("phase_profile_counters", cuda)
        self.assertIn("phase_profile_sample_stride", cuda)
        self.assertIn("shared_wavefront_fused_profiled", cuda)
        self.assertIn("shared_wavefront_persistent_grouped_profiled", cuda)
        self.assertIn("shared_grouped_profile_blocks_256", cuda)
        self.assertIn('"leo_shared_wavefront_persistent_grouped_profiled"', cuda)
        self.assertNotIn("&& phase_profile_sample_stride.is_none()", cuda)

        normal_fused = kernels.split(
            'extern "C" __global__ void leo_shared_wavefront_fused', 1
        )[1].split("enum LeoCudaPhaseProfileCounter", 1)[0]
        self.assertNotIn("clock64()", normal_fused)
        self.assertNotIn("phase_profile_counters", normal_fused)

        profiled = kernels.split(
            'extern "C" __global__ void leo_shared_wavefront_fused_profiled', 1
        )[1].split(
            'extern "C" __global__ void leo_apply_shared_wavefront_deltas', 1
        )[0]
        persistent_profiled = kernels.split(
            "leo_shared_wavefront_persistent_grouped_body", 1
        )[1].split(
            'extern "C" __global__ void leo_shared_wavefront_persistent_grouped(', 1
        )[0]
        for phase in (
            "LEO_CUDA_PHASE_PROFILE_PRE",
            "LEO_CUDA_PHASE_PROFILE_SELECT",
            "LEO_CUDA_PHASE_PROFILE_POST_SELECT",
            "LEO_CUDA_PHASE_PROFILE_CACHE_SURROGATE",
            "LEO_CUDA_PHASE_PROFILE_POST_CORE",
            "LEO_CUDA_PHASE_PROFILE_LEARNING_SIGNALS",
            "LEO_CUDA_PHASE_PROFILE_POST_DELTAS",
            "LEO_CUDA_PHASE_PROFILE_HOMEOSTASIS",
            "LEO_CUDA_PHASE_PROFILE_CAPTURE",
        ):
            self.assertIn(phase, profiled)
        self.assertIn("clock64()", profiled)
        self.assertIn("grid.sync();", profiled)
        self.assertNotIn("atomicAdd", profiled)
        self.assertIn("leo_phase_profile_mark<PROFILED>", persistent_profiled)
        self.assertIn("LEO_CUDA_PHASE_PROFILE_SAMPLES", persistent_profiled)
        self.assertIn("leo_shared_phase_select_block_fast", persistent_profiled)
        self.assertIn(
            "fused_profile_blocks_256 >= grid_blocks",
            cuda,
        )
        self.assertIn("cuda_phase_profile_skipped", cuda)
        self.assertIn("profiled_kernel_cannot_match_production_grid", cuda)
        self.assertIn("sampled_grid_blocks_max", cuda)
        self.assertNotIn(
            "grid_blocks.min(coordinator.shared.fused_profile_blocks_256)",
            cuda,
        )


    def test_replay_supervised_path_uses_cooperative_grid_with_legacy_ab_control(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        cooperative = kernels.split("leo_train_cooperative_body", 1)[1].split(
            'extern "C" __global__ void leo_train_cooperative', 1
        )[0]
        self.assertIn("cooperative_groups::this_grid()", cooperative)
        self.assertIn("grid_thread", cooperative)
        self.assertIn("grid_stride", cooperative)
        for helper in (
            "leo_p_cache_surrogate_work",
            "leo_p_update_recurrent_eligibility_work",
            "leo_p_update_input_eligibility_work",
            "leo_p_post_and_emit_work",
            "leo_p_learning_signals_work",
            "leo_p_update_output_work",
            "leo_p_update_context_work",
            "leo_p_update_recurrent_weights_work",
            "leo_p_update_input_weights_work",
            "leo_p_inhibitory_homeostasis_work",
            "leo_p_homeostasis_work",
        ):
            self.assertIn(helper, cooperative)

        # Order-sensitive atomics/reductions remain block-0 phases.
        self.assertIn("if (blockIdx.x == 0U)", cooperative)
        self.assertIn("leo_p_deliver_events", cooperative)
        self.assertIn("leo_p_inject_symbol", cooperative)
        self.assertIn("leo_p_context_resolve", cooperative)
        self.assertIn("leo_p_select_global", cooperative)
        self.assertIn("leo_p_forward", cooperative)
        self.assertIn("leo_p_capture_training_step", cooperative)

        legacy = kernels.split(
            'extern "C" __global__ void leo_train_persistent', 1
        )[1]
        self.assertIn("if (blockIdx.x != 0U) return;", legacy)
        self.assertIn("LEO_CUDA_REPLAY_COOPERATIVE", cuda)
        self.assertIn("LEO_CUDA_REPLAY_BLOCKS", cuda)
        self.assertIn("LEO_CUDA_FROZEN_BLOCKS", cuda)
        self.assertIn("replay_grid_blocks", cuda)
        self.assertIn("replay_profile_grid_blocks", cuda)
        self.assertIn("profiled_kernel_cannot_match_production_grid", cuda)

    def test_replay_debugging_exposes_hidden_prefix_and_sampled_cuda_phases(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        lifecycle = (ROOT / "crates/leo-cli/src/training/lifecycle.rs").read_text()
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()
        docs = (ROOT / "docs/DEBUGGING.md").read_text()
        script = (ROOT / "scripts/debug_replay.sh").read_text()
        gpu_gate = (ROOT / "scripts/check_gpu.sh").read_text()

        for marker in (
            "replay_selection_debug",
            "replay_device_selection_debug",
            "replay_range_debug",
            "replay_segment_debug",
            "replay_story_debug",
            "replay_batch_debug",
            "LEO_REPLAY_DEBUG_SEGMENT_STRIDE",
        ):
            self.assertIn(marker, training)
        for marker in (
            "replay_prefix_steps",
            "replay_execution_steps",
            "replay_prefix_step_fraction",
            "execution_steps_with_prefix",
            "execution_steps_per_second",
        ):
            self.assertIn(marker, main)
        for marker in (
            "training_progress",
            "replay_prefix_steps",
            "replay_execution_steps",
            "execution_steps_per_second",
        ):
            self.assertIn(marker, lifecycle)
        for marker in (
            "cuda_debug_runtime",
            "driver_version",
            "pci_bus_id",
            "replay_profile_target_stride",
            "replay_profile_prefix_stride",
            "cuda_debug_memory",
            "cuda_debug_launch",
            "cuda_debug_chunk",
            "cuda_replay_kernel_profile",
            "cuda_frozen_kernel_profile",
            "LEO_CUDA_REPLAY_PROFILE_TARGET",
            "LEO_CUDA_REPLAY_PROFILE_PREFIX",
        ):
            self.assertIn(marker, cuda)
        self.assertIn("leo_train_cooperative_profiled", kernels)
        self.assertIn("leo_advance_frozen_cooperative_profiled", kernels)
        self.assertIn("LEO_CUDA_REPLAY_PROFILE", docs)
        self.assertIn("LEO_CUDA_REPLAY_COOPERATIVE", docs)
        self.assertIn("replay_device_selection_debug", docs)
        self.assertIn("LEO_CUDA_REPLAY_PROFILE", script)
        self.assertIn("CUDA replay diagnostics OK", gpu_gate)
        self.assertIn("replay_device_selection_debug", gpu_gate)
        self.assertIn("replay_prefix_steps", gpu_gate)
        self.assertIn("cuda_replay_kernel_profile", gpu_gate)

    def test_repository_gate_and_package_entrypoints_cover_all_reference_tests(self):
        check = (ROOT / "scripts/check.sh").read_text()
        makefile = (ROOT / "Makefile").read_text()
        data = (ROOT / "scripts/data.sh").read_text()
        ci = (ROOT / ".github/workflows/ci.yml").read_text()

        self.assertIn("python3 -m unittest discover -s python/tests -v", check)
        self.assertIn("python3 -m unittest discover -s tests -v", check)
        self.assertIn("bash ./scripts/check.sh", makefile)
        self.assertIn("huggingface_hub>=0.34,<2", data)
        self.assertNotIn("Install requirements.txt", data)
        self.assertIn("permissions:\n  contents: read", ci)
        self.assertNotIn("actions/checkout@v", ci)
        self.assertNotIn("actions/setup-python@v", ci)

    def test_training_lifecycle_is_owned_by_training_module(self):
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        lifecycle = (ROOT / "crates/leo-cli/src/training/lifecycle.rs").read_text()
        resume = (ROOT / "crates/leo-cli/src/training/resume.rs").read_text()
        validation = (ROOT / "crates/leo-cli/src/training/validation.rs").read_text()

        self.assertIn("pub(crate) fn run_training", lifecycle)
        self.assertIn("TrainingResumeState", resume)
        self.assertIn("pub(crate) fn evaluate_model", validation)
        self.assertNotIn("struct TrainingResumeState", main)
        self.assertNotIn("struct StoryBatchPrefetcher", main)
        self.assertNotIn("fn meaningful_improvement", main)


    def test_multi_gpu_flattens_story_deltas_before_one_canonical_mean(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        lifecycle = (ROOT / "crates/leo-cli/src/training/lifecycle.rs").read_text()
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()

        multi = training.split("struct MultiGpuBatchTrainer", 1)[1].split(
            "fn apply_batch_replay_policy", 1
        )[0]
        self.assertIn("balanced_story_range", multi)
        self.assertIn("training_story_batch_deltas", multi)
        self.assertIn("results.sort_by_key(|result| result.story_start)", multi)
        self.assertIn("deltas.extend(result.story_deltas)", multi)
        self.assertEqual(multi.count("apply_mean_deltas(model, &deltas)"), 1)
        self.assertIn("exact_flat_story_mean\\\":true", multi)
        self.assertNotIn("stories.len() % device_count", multi)
        self.assertNotIn("stories.len() / device_count < 2", multi)

        self.assertIn("DeviceStoryBatchDeltaReport", backend)
        self.assertIn("retain_story_deltas", cuda)
        self.assertIn("if retain_story_deltas", cuda)
        self.assertIn("snapshot_batch_lane_values_batched", cuda)
        self.assertIn("gpu_story_mean_exact_v1", lifecycle)
        self.assertIn("flat_story_delta_mean", lifecycle)

        # Benchmark must exercise the same distributed training engine; otherwise
        # scaling measurements would silently benchmark only GPU 0.
        benchmark = main.split("fn command_benchmark", 1)[1].split(
            "fn command_doctor", 1
        )[0]
        self.assertIn("configured_multi_gpu_devices", benchmark)
        self.assertIn("TrainingEngine::new", benchmark)
        self.assertIn("training_engine.train_batch", benchmark)
        self.assertIn("\\\"gpu_devices\\\"", benchmark)

    def test_multi_gpu_partition_supports_uneven_and_one_story_per_device(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()

        self.assertIn("fn balanced_story_range", training)
        self.assertIn("run_multi_gpu_shard", training)
        self.assertNotIn("if shard.len() == 1", training)
        self.assertIn("(stories.len() < 2 && !retain_story_deltas)", backend)
        self.assertIn("sixteen_devices_can_own_one_story_each", training)
        self.assertIn("uneven_device_count_preserves_story_order", training)

    def test_multi_gpu_reuses_canonical_device_zero_and_sparse_commits_mean(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()

        multi = training.split("struct MultiGpuBatchTrainer", 1)[1].split(
            "fn apply_batch_replay_policy", 1
        )[0]
        self.assertIn("Vec::with_capacity(device_count.saturating_sub(1))", multi)
        self.assertIn("for device_index in 1..device_count", multi)
        self.assertNotIn("for device_index in 0..device_count", multi)
        self.assertIn("MultiGpuReplicaWorker::spawn", multi)
        self.assertIn("run_multi_gpu_shard(\n            canonical,", multi)
        self.assertIn("canonical.commit_packed_host_model(&sync_update)?", multi)
        self.assertIn(r'canonical_device0_reused\":true', multi)
        self.assertIn(r'persistent_replica_threads\":true', multi)
        self.assertIn("fn synchronize_packed_model", backend)
        self.assertIn("fn commit_packed_host_model", backend)
        self.assertIn("pub(crate) fn synchronize_packed_model", cuda)
        self.assertIn("apply_packed_sparse_update", cuda)

    def test_explicit_multi_gpu_request_fails_instead_of_silent_single_gpu_fallback(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        lifecycle = (ROOT / "crates/leo-cli/src/training/lifecycle.rs").read_text()
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()

        self.assertIn("pub(crate) fn configured_multi_gpu_devices", training)
        self.assertIn("-> LeoResult<usize>", training)
        self.assertIn(
            "LEO_MULTI_GPU requested, but fewer than two CUDA devices are visible",
            training,
        )
        self.assertIn("configured_multi_gpu_devices(backend, workers.min(story_limit.max(1)))?", lifecycle)
        self.assertIn("configured_multi_gpu_devices(runtime.resolved_backend(), effective_workers)?", main)

    def test_multi_gpu_scaling_barrier_fixes_are_present(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        parallel = (ROOT / "crates/leo-core/src/parallel.rs").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()

        self.assertIn("struct MultiGpuReplicaWorker", training)
        self.assertIn("mpsc::sync_channel::<ReplicaCommand>(2)", training)
        self.assertNotIn("thread::scope", training)
        self.assertIn("work_balanced_story_ranges", training)
        self.assertIn("PackedSparseModelUpdate::from_model", training)
        self.assertIn("canonical.commit_packed_host_model(&sync_update)?", training)
        self.assertIn("self.synchronize_replicas(Arc::clone(&sync_update))?", training)
        self.assertIn("if device_count < 2", training)
        self.assertIn("canonical.enable_parameter_tracking()", training)
        self.assertIn("self.synchronize_replicas(update)?", training)
        self.assertIn("multi_gpu_serial_replay", training)
        self.assertIn("LEO_MULTI_GPU_PARALLEL_REPLAY", training)
        self.assertIn("run_parallel_replay", training)
        self.assertIn("apply_sum_deltas", training)
        self.assertIn("canonical.synchronize_packed_model(&replay_update)?", training)
        self.assertIn("runtime.synchronize_packed_model(&update)?", training)

        self.assertIn("struct PackedSparseModelUpdate", parallel)
        self.assertIn("fn apply_sparse_scaled", parallel)
        self.assertIn("pub fn apply_sum_deltas", parallel)
        self.assertNotIn("BTreeMap::<usize, f64>", parallel)
        self.assertNotIn("project_parameter_constraints(master);", parallel)
        self.assertIn("project_parameter_constraints_sparse(master, deltas)?", parallel)
        self.assertIn("validate_sparse_merge(master, deltas)?", parallel)

        self.assertIn("snapshot_batch_lane_values_batched", cuda)
        self.assertIn("batch snapshot counts", cuda)
        self.assertIn("fn read_batch_lane_change_counts", cuda)
        self.assertIn("let count_elements = lanes.len().saturating_mul(5);", cuda)
        self.assertIn("entire variable payload crosses PCIe in at most three D2H operations", cuda)
        self.assertIn("self.batch_snapshot_device_f32", cuda)
        self.assertIn("self.batch_snapshot_device_u32", cuda)
        self.assertIn("self.batch_snapshot_device_u64", cuda)
        self.assertIn("targets.extend(lanes.iter().map(|lane| lane.buffers))", cuda)
        self.assertIn("lane.model_revision = revision", cuda)
        self.assertIn("coordinator.commit_packed_host_model(&sync_update, lanes)?", cuda)
        self.assertNotIn("batch lane canonical restore", cuda)
        self.assertNotIn("upload_current_sparse_model", cuda)
        self.assertIn("fn synchronize_packed_model", backend)

    def test_scaling_benchmark_environment_is_hermetic(self):
        scaling = runpy.run_path(str(ROOT / "scripts/benchmark_multi_gpu.py"))
        benchmark_environment = scaling["benchmark_environment"]
        dirty = {
            "PATH": "/bin",
            "KEEP_ME": "yes",
            "LEO_MULTI_GPU": "1",
            "LEO_REPLAY_STREAMING": "1",
            "LEO_MULTI_GPU_PARALLEL_REPLAY": "1",
            "LEO_CUDA_SHARED_PERSISTENT": "0",
            "LEO_CUDA_SHARED_GROUPED": "0",
            "LEO_CUDA_DEVICE_BATCH_MERGE": "0",
            "LEO_CUDA_DEVICE_STORY_STEPS": "0",
            "LEO_CUDA_DEVICE_STORY_POSTPROCESS": "0",
            "LEO_CUDA_FULL_STEP_METRICS": "1",
            "LEO_CUDA_DEBUG_LAUNCHES": "1",
            "LEO_CUDA_REPLAY_PROFILE_STRIDE": "1",
            "LEO_CUDA_PHASE_PROFILE": "1",
            "LEO_REPLAY_DEBUG_SEGMENTS": "1",
        }

        clean = benchmark_environment(dirty)
        self.assertEqual(clean["KEEP_ME"], "yes")
        self.assertEqual(clean["PATH"], "/bin")
        self.assertFalse(any(key.startswith("LEO_") for key in clean))

        legacy = benchmark_environment(dirty, legacy_execution=True)
        self.assertEqual(legacy["KEEP_ME"], "yes")
        self.assertEqual(legacy["LEO_CUDA_SHARED_PERSISTENT"], "0")
        self.assertEqual(legacy["LEO_CUDA_SHARED_GROUPED"], "0")
        self.assertEqual(legacy["LEO_CUDA_DEVICE_BATCH_MERGE"], "0")
        self.assertNotIn("LEO_REPLAY_STREAMING", legacy)
        self.assertNotIn("LEO_MULTI_GPU_PARALLEL_REPLAY", legacy)
        self.assertNotIn("LEO_CUDA_FULL_STEP_METRICS", legacy)
        self.assertNotIn("LEO_CUDA_DEBUG_LAUNCHES", legacy)

    def test_multi_gpu_exactness_and_reproducibility_gates_are_present(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        lifecycle = (ROOT / "crates/leo-cli/src/training/lifecycle.rs").read_text()
        resume = (ROOT / "crates/leo-cli/src/training/resume.rs").read_text()
        validation = (ROOT / "crates/leo-cli/src/training/validation.rs").read_text()
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        scaling = (ROOT / "scripts/benchmark_multi_gpu.py").read_text()
        gpu_gate = (ROOT / "scripts/check_gpu.sh").read_text()

        self.assertIn("GPU_REFERENCE_WORKERS: usize = 16", training)
        self.assertIn("visible_gpu_device_compatibility", training)
        self.assertIn("exact multi-GPU requires homogeneous CUDA devices", training)
        self.assertIn('"gpu_story_mean_exact_v1"', lifecycle)
        self.assertIn("synchronization_modes_equivalent", resume)
        self.assertIn("configured_multi_gpu_devices(backend, limit.max(1))?", validation)
        self.assertIn("training_state_hash(runtime.model())", main)
        self.assertIn("training_state_sha256", main)
        self.assertIn("selectors.DefaultSelector", scaling)
        self.assertIn("gpu_heartbeat", scaling)
        self.assertIn("training_state_sha256", scaling)
        self.assertIn("benchmark_environment", scaling)
        self.assertIn("LEO_REPLAY_STREAMING", scaling)
        self.assertIn("LEO_MULTI_GPU_PARALLEL_REPLAY", scaling)
        self.assertIn("--legacy-execution", scaling)
        self.assertIn("unset LEO_REPLAY_STREAMING", gpu_gate)
        self.assertIn("unset LEO_MULTI_GPU_PARALLEL_REPLAY", gpu_gate)
        self.assertIn("unset LEO_CUDA_FULL_STEP_METRICS", gpu_gate)
        self.assertIn("--counts 1,2", gpu_gate)
        self.assertIn("final-state conformance gate", gpu_gate)

    def test_persistent_selection_skips_exactly_untouched_blocks(self):
        cuda = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        mark_touched = cuda.split(
            "__device__ __forceinline__ void leo_p_mark_touched", 1
        )[1].split("__device__ __forceinline__ void leo_p_mark_learning_destination", 1)[0]
        self.assertIn("LEO_P_BLOCK_CUTOFF", mark_touched)
        self.assertIn("atomicExch(&block_cutoff[neuron / cfg->neurons_per_block], 1.0f)", mark_touched)

        start_tick = cuda.split("__device__ void leo_p_start_tick", 1)[1].split(
            "__device__ void leo_p_deliver_events", 1
        )[0]
        self.assertIn("block_cutoff[block] = 0.0f", start_tick)
        self.assertIn("winner_value[index] = -1.0f", start_tick)

        select_block = cuda.split(
            "__device__ __forceinline__ void leo_p_select_model_block", 1
        )[1].split("__device__ void leo_p_select_blocks", 1)[0]
        self.assertIn("if (block_cutoff[model_block] == 0.0f) return", select_block)
        self.assertIn("block_cutoff[model_block] = cutoff", select_block)


    def test_persistent_shared_fast_path_keeps_full_debug_fallback(self):
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        persistent = kernels.split(
            'extern "C" __global__ void leo_shared_wavefront_persistent', 1
        )[1].split("enum LeoCudaPhaseProfileCounter", 1)[0]
        self.assertIn(
            "for (unsigned int step_index = 0U; step_index < max_step_count; ++step_index)",
            persistent,
        )
        self.assertGreaterEqual(persistent.count("grid.sync();"), 9)
        for helper in (
            "leo_shared_phase_select_block_fast",
            "leo_shared_phase_post_core_lane_fast",
            "leo_shared_phase_post_deltas_lane_fast",
            "leo_shared_phase_capture_lane_fast",
        ):
            self.assertIn(helper, persistent)

        self.assertIn("struct CudaFastTrainingStepRecord", rust)
        self.assertIn("LEO_CUDA_SHARED_PERSISTENT", rust)
        self.assertIn("LEO_CUDA_FULL_STEP_METRICS", rust)
        self.assertIn("shared_persistent_enabled", rust)
        self.assertIn("leo_train_cooperative_fast", rust)
        self.assertIn("leo_shared_wavefront_persistent", rust)
        self.assertIn("LeoFastTrainingStepRecord", kernels)
        self.assertIn("leo_train_cooperative_fast", kernels)

        fast_capture = kernels.split(
            "leo_p_capture_training_step_fast", 1
        )[1].split("__device__ void leo_p_capture_training_step", 1)[0]
        self.assertIn("record.loss", fast_capture)
        self.assertIn("record.active_count", fast_capture)
        self.assertIn("record.error_code", fast_capture)
        self.assertIn("records[record_index] = record", fast_capture)
        self.assertNotIn("neural_only_loss", fast_capture)
        self.assertNotIn("predicted_index", fast_capture)

    def test_fast_training_path_skips_diagnostic_only_global_atomics(self):
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        select_impl = kernels.split("leo_p_select_model_block_impl", 1)[1].split(
            "__device__ __forceinline__ void leo_p_select_model_block(", 1
        )[0]
        self.assertIn("if (DETAILED_METRICS &&", select_impl)
        self.assertIn("atomicAdd(&counters->suprathreshold", select_impl)
        self.assertIn("atomicAdd(&counters->block_selected", select_impl)

        recurrent_impl = kernels.split(
            "leo_p_update_recurrent_eligibility_work_impl", 1
        )[1].split("__device__ void leo_p_update_recurrent_eligibility_work(", 1)[0]
        input_impl = kernels.split(
            "leo_p_update_input_eligibility_work_impl", 1
        )[1].split("__device__ void leo_p_update_input_eligibility_work(", 1)[0]
        recurrent_weights = kernels.split(
            "leo_p_update_recurrent_weights_work_impl", 1
        )[1].split("__device__ void leo_p_update_recurrent_weights_work(", 1)[0]
        input_weights = kernels.split(
            "leo_p_update_input_weights_work_impl", 1
        )[1].split("__device__ void leo_p_update_input_weights_work(", 1)[0]

        for body, counter in (
            (recurrent_impl, "eligible_recurrent"),
            (input_impl, "eligible_input"),
            (recurrent_weights, "learning_eligibility_abs_sum"),
            (input_weights, "learning_eligibility_abs_sum"),
        ):
            self.assertIn("DETAILED_METRICS", body)
            self.assertIn(counter, body)

        cooperative = kernels.split("leo_train_cooperative_body", 1)[1].split(
            'extern "C" __global__ void leo_train_cooperative', 1
        )[0]
        self.assertIn("if (FAST_METRICS)", cooperative)
        self.assertIn("leo_p_select_model_block_fast", cooperative)
        self.assertIn("leo_p_update_recurrent_eligibility_work_fast", cooperative)
        self.assertIn("leo_p_update_input_eligibility_work_fast", cooperative)
        self.assertIn("leo_p_update_recurrent_weights_work_fast", cooperative)
        self.assertIn("leo_p_update_input_weights_work_fast", cooperative)

    def test_legacy_eligibility_kernels_do_not_reference_template_metric_flag(self):
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        recurrent = kernels.split(
            'extern "C" __global__ void leo_update_recurrent_eligibility(', 1
        )[1].split('extern "C" __global__ void leo_update_input_eligibility(', 1)[0]
        input_body = kernels.split(
            'extern "C" __global__ void leo_update_input_eligibility(', 1
        )[1].split('extern "C" __global__ void leo_update_recurrent_weights(', 1)[0]

        self.assertNotIn("DETAILED_METRICS", recurrent)
        self.assertNotIn("DETAILED_METRICS", input_body)
        self.assertIn("atomicAdd(&counters->eligible_recurrent, 1U);", recurrent)
        self.assertIn("atomicAdd(&counters->eligible_input, 1U);", input_body)

    def test_streaming_replay_is_opt_in_and_classic_replay_remains_default(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        flag = training.split("fn replay_streaming_enabled", 1)[1].split(
            "#[derive(Debug, Clone, Copy, Default)]", 1
        )[0]
        self.assertIn("LEO_REPLAY_STREAMING", flag)
        self.assertIn(".unwrap_or(false)", flag)

        streaming = training.split("fn replay_ranges_streaming", 1)[1].split(
            "fn apply_replay_policy", 1
        )[0]
        self.assertIn("runtime.begin_document()?", streaming)
        self.assertIn("runtime.replay_streaming_batch(&schedule, permission)?", streaming)
        self.assertIn("schedule.push(step)", streaming)
        self.assertIn("runtime.reset_transient_state()?", streaming)

        self.assertIn("fn replay_streaming_batch", backend)
        self.assertIn("pub(crate) fn replay_streaming_batch", rust)
        self.assertIn("target_index: -2", rust)
        self.assertIn("const bool frozen_advance = step.target_index == -2", kernels)
        self.assertIn("leo_p_context_advance_history", kernels)

        policy = training.split("fn apply_replay_policy", 1)[1].split(
            "fn apply_batch_replay_policy", 1
        )[0]
        self.assertIn("replay_streaming_enabled()", policy)
        self.assertIn("replay_ranges_streaming", policy)
        self.assertIn("replay_target_range_impl", policy)
        self.assertIn("runtime.model().config.replay.fraction", policy)
        self.assertIn("runtime.model().config.replay.segment_bytes", policy)

        gpu_gate = (ROOT / "scripts/check_gpu.sh").read_text()
        self.assertIn("LEO_CUDA_SHARED_PERSISTENT=0", gpu_gate)
        self.assertIn("LEO_CUDA_SHARED_GROUPED=0", gpu_gate)
        self.assertIn("LEO_CUDA_DEVICE_BATCH_MERGE=0", gpu_gate)
        self.assertIn("Optimized-vs-legacy CUDA exact-state gate OK", gpu_gate)
        self.assertIn("LEO_CUDA_DEBUG_LAUNCHES=1", gpu_gate)
        self.assertIn("optimized_kernel=", gpu_gate)
        self.assertIn("device_batch_merge_observed", gpu_gate)
        self.assertIn("device_batch_merge=true", gpu_gate)
        self.assertIn("device_story_steps_observed", gpu_gate)
        self.assertIn("device_story_steps=true", gpu_gate)
        self.assertIn("device_story_postprocess_observed", gpu_gate)
        self.assertIn("device_story_postprocess=true", gpu_gate)
        self.assertIn("training_state_sha256", gpu_gate)

    def test_grouped_shared_wavefront_parallelizes_each_logical_lane(self):
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        planner = (ROOT / "crates/leo-core/src/cuda_plan.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        grouped = kernels.split(
            "leo_shared_wavefront_persistent_grouped_body", 1
        )[1].split(
            "leo_shared_wavefront_persistent_grouped(", 1
        )[0]
        self.assertIn("lane = blockIdx.x % lane_count", grouped)
        self.assertIn("lane_block = blockIdx.x / lane_count", grouped)
        self.assertIn("blocks_in_lane", grouped)
        self.assertIn("lane_step_count", grouped)
        self.assertIn("cooperative_groups::this_grid()", grouped)
        self.assertIn("step_index < max_step_count", grouped)
        self.assertIn("active = lane_valid && step_index < lane_step_count", grouped)
        self.assertNotIn("leo_lane_group_barrier", kernels)
        # Cache-surrogate work and next-eligibility count reset are independent
        # and now share one visibility boundary. Keep the exact expected count
        # so accidental synchronization changes remain visible to this gate.
        self.assertEqual(
            grouped.count("cooperative_groups::this_grid().sync();"), 17
        )
        self.assertIn("Count reset is independent of surrogate calculation", grouped)
        self.assertNotIn("cooperative_groups::grid_group grid", grouped)
        self.assertNotIn("rec_current_list = nullptr", grouped)
        self.assertIn(
            "__launch_bounds__(256, 2)\nleo_shared_wavefront_persistent_grouped(",
            kernels,
        )
        self.assertIn(
            "__launch_bounds__(256, 2)\nleo_shared_wavefront_persistent_grouped_profiled(",
            kernels,
        )
        self.assertIn("lane_thread", grouped)
        self.assertIn("lane_stride", grouped)
        self.assertIn("leo_p_forward_context_latent_work", grouped)
        self.assertIn("leo_p_forward_logits_work", grouped)
        self.assertIn("leo_p_update_recurrent_weights_work_fast", grouped)
        self.assertIn("leo_p_update_input_weights_work_fast", grouped)
        self.assertIn("leo_p_homeostasis_work", grouped)
        self.assertIn("let grid_blocks = grouped_budget;", rust)
        self.assertIn("extra_lane_blocks", rust)
        self.assertIn("LEO_CUDA_SHARED_GROUPED", rust)
        self.assertIn("shared_grouped_blocks_256", rust)
        self.assertIn("grouped_blocks_256", planner)

        gpu_gate = (ROOT / "scripts/check_gpu.sh").read_text()
        self.assertIn("Full-capacity grouped exact-state gate OK", gpu_gate)
        self.assertIn("blocks_per_lane_max", gpu_gate)
        self.assertIn("Grouped residency gate OK", gpu_gate)

    def test_grouped_cta_remainder_distribution_covers_each_model_block_once(self):
        for grid_blocks in (16, 17, 31, 48, 55, 56, 112):
            for lane_count in (1, 2, 7, 8, 16):
                if grid_blocks < lane_count:
                    continue
                for lane in range(lane_count):
                    blocks_in_lane = (grid_blocks + lane_count - 1 - lane) // lane_count
                    covered = []
                    for lane_block in range(blocks_in_lane):
                        covered.extend(range(lane_block, 128, blocks_in_lane))
                    self.assertEqual(sorted(covered), list(range(128)))
                    self.assertEqual(len(covered), len(set(covered)))

    def test_device_replay_packed_order_reconstructs_host_ranges(self):
        def f32(value):
            return struct.unpack("<f", struct.pack("<f", value))[0]

        def host_ranges(losses, fraction, segment_targets):
            if not losses or not math.isfinite(fraction) or fraction <= 0.0 or segment_targets == 0:
                return []
            fraction = f32(fraction)
            target_budget = max(1, math.ceil(len(losses) * min(fraction, 1.0)))
            positions = [
                (position, loss)
                for position, loss in enumerate(losses)
                if math.isfinite(loss) and loss > 0.0
            ]
            positions.sort(key=lambda item: (-item[1], item[0]))
            selected = []
            selected_targets = 0
            for position, _loss in positions:
                if selected_targets >= target_budget:
                    break
                remaining = target_budget - selected_targets
                width = min(segment_targets, remaining, len(losses))
                start = max(0, position - width // 2)
                end = min(len(losses), start + width)
                start = end - width
                candidate = (start, end)
                if any(left < end and start < right for left, right in selected):
                    continue
                selected.append(candidate)
                selected_targets += width
            return sorted(selected)

        def device_ranges(losses, fraction, segment_targets):
            fraction = f32(fraction)
            if not losses or not math.isfinite(fraction) or fraction <= 0.0 or segment_targets == 0:
                return []
            keys = []
            for position, loss in enumerate(losses):
                if not math.isfinite(loss) or loss <= 0.0:
                    continue
                bits = struct.unpack("<I", struct.pack("<f", loss))[0]
                keys.append(((bits << 32) | (0xFFFFFFFF - position), position))
            keys.sort(reverse=True)
            target_budget = max(1, math.ceil(len(losses) * min(fraction, 1.0)))
            selected = []
            selected_targets = 0
            for _key, position in keys:
                if selected_targets >= target_budget:
                    break
                remaining = target_budget - selected_targets
                width = min(segment_targets, remaining, len(losses))
                start = max(0, position - width // 2)
                end = min(len(losses), start + width)
                start = end - width
                if any(left < end and start < right for left, right in selected):
                    continue
                selected.append((start, end))
                selected_targets += width
            return sorted(selected)

        rng = random.Random(1337)
        for length in (1, 2, 7, 48, 127, 512):
            for fraction in (0.0, 0.01, 0.3, 1.0):
                for segment_targets in (1, 7, 48):
                    losses = [f32(rng.random() * 9.0) for _ in range(length)]
                    if length >= 7:
                        losses[0] = f32(1.0)
                        losses[1] = f32(1.0)
                        losses[2] = 0.0
                        losses[3] = -0.0
                        losses[4] = float("inf")
                        losses[5] = float("nan")
                    self.assertEqual(
                        device_ranges(losses, fraction, segment_targets),
                        host_ranges(losses, fraction, segment_targets),
                    )

    def test_device_story_step_builder_preserves_schedule_contract(self):
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        builder = kernels.split('extern "C" __global__ void leo_build_shared_story_steps', 1)[1].split(
            'extern "C" __global__ void leo_postprocess_shared_story_records', 1
        )[0]
        dropout = kernels.split("bool leo_context_survives_dropout_exact", 1)[1].split(
            'extern "C" __global__ void leo_build_shared_story_steps', 1
        )[0]
        self.assertIn("symbol = 256U; // BEGIN_DOCUMENT", builder)
        self.assertIn("target_index = (int)bytes[0]", builder)
        self.assertIn("target_index = 256; // END_DOCUMENT_OUTPUT_INDEX", builder)
        self.assertIn("strength * target_weight", builder)
        self.assertIn("0x9e3779b97f4a7c15ULL", dropout)
        self.assertIn("0xbf58476d1ce4e5b9ULL", dropout)
        self.assertIn("0x94d049bb133111ebULL", dropout)
        self.assertIn("16777216.0f", dropout)
        self.assertIn("cuda_device_story_steps_enabled", rust)
        self.assertIn("LEO_CUDA_DEVICE_STORY_STEPS", rust)
        self.assertIn("launch_device_story_step_builder", rust)
        self.assertIn("batch_story_bytes", rust)
        self.assertIn("host_batch_story_bytes", rust)

    def test_device_story_postprocess_keeps_losses_and_replay_selection_on_gpu(self):
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        self.assertIn("leo_build_shared_story_steps", kernels)
        self.assertIn("leo_context_survives_dropout_exact", kernels)
        self.assertIn("leo_postprocess_shared_story_records", kernels)
        self.assertIn("leo_replay_loss_record", kernels)
        self.assertIn("LeoFastStorySummary", kernels)
        self.assertIn("LeoReplayRange", kernels)
        self.assertIn("leo_bitonic_sort_selection_keys", kernels)
        self.assertIn("LEO_DEVICE_REPLAY_MAX_STEPS 4096U", kernels)
        self.assertIn("cuda_device_story_steps_enabled", rust)
        self.assertIn("LEO_CUDA_DEVICE_STORY_STEPS", rust)
        self.assertIn("launch_device_story_step_builder", rust)
        self.assertIn("cuda_device_story_postprocess_enabled", rust)
        self.assertIn("LEO_CUDA_DEVICE_STORY_POSTPROCESS", rust)
        self.assertIn("launch_device_story_postprocess", rust)
        self.assertIn("read_device_story_postprocess", rust)
        self.assertIn("batch_story_summaries", rust)
        self.assertIn("batch_replay_ranges", rust)
        self.assertIn("DeviceStorySummary", backend)
        self.assertIn("device_postprocessed", training)
        self.assertIn("apply_batch_replay_ranges", training)
        device_ranges = training.split("fn apply_batch_replay_ranges", 1)[1].split(
            "fn device_story_report_is_postprocessed", 1
        )[0]
        self.assertIn("emit_device_replay_selection_debug", device_ranges)
        self.assertIn("emit_replay_batch_debug", device_ranges)
        self.assertIn("losses_resident_on_device", training)
        # The CPU selector remains the exact fallback for debug, long stories,
        # multi-GPU retained deltas, and explicit execution A/B.
        self.assertIn("select_replay_ranges(losses, fraction, segment_targets)", training)

    def test_device_batch_merge_marks_touched_rows_even_when_mean_cancels(self):
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        merge = kernels.split(
            'extern "C" __global__ void leo_merge_shared_lane_fixed_parameters', 1
        )[1].split('extern "C" __global__ void leo_apply_shared_wavefront_deltas', 1)[0]
        self.assertIn("bool lane_changed = false", merge)
        self.assertIn("lane_changed |= delta != 0.0f", merge)
        self.assertIn("if (lane_changed)", merge)
        self.assertIn("leo_mark_changed", merge)
        self.assertIn("LEO_CUDA_DEVICE_BATCH_MERGE", rust)
        self.assertIn(r'\"scope\":\"device_batch_merge', rust)
        self.assertIn("leo_merge_shared_lane_fixed_parameters", rust)
        self.assertIn("snapshot_batch_lane_context_values_batched", rust)
        # Device merge must include every input symbol: 256 bytes plus the
        # BEGIN_DOCUMENT and END_DOCUMENT control symbols.
        self.assertIn("#define LEO_SYMBOLS 258", kernels)
        self.assertIn(
            "const unsigned int input_len = LEO_SYMBOLS * cfg->input_fanout;",
            merge,
        )

    def test_power_of_two_indexing_and_short_decay_have_exact_fast_paths(self):
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()
        self.assertIn("leo_fast_mod_u32", kernels)
        self.assertIn("leo_fast_mod_u64_u32", kernels)
        self.assertIn("value & (modulus - 1U)", kernels)
        decay = kernels.split("float leo_decay", 1)[1].split(
            "__device__ __forceinline__ unsigned int leo_rotated_key", 1
        )[0]
        self.assertIn("if (elapsed <= 7ULL)", decay)
        self.assertIn("const float base2 = base * base", decay)
        self.assertIn("const float base4 = base2 * base2", decay)
        self.assertIn("while (exponent != 0ULL)", decay)

    def test_cuda_document_reset_keeps_public_sync_contract_and_replay_deferred_fast_path(self):
        rust = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        begin = rust.split("pub(crate) fn begin_document", 1)[1].split(
            "pub(crate) fn begin_document_deferred_for_replay", 1
        )[0]
        deferred = rust.split(
            "pub(crate) fn begin_document_deferred_for_replay", 1
        )[1].split("pub(crate) fn finish_document", 1)[0]
        reset = rust.split("fn enqueue_transient_reset", 1)[1].split(
            "pub(crate) fn step", 1
        )[0]
        replay = training.split("fn replay_target_range_impl", 1)[1].split(
            "#[cfg(test)]", 1
        )[0]

        # General backend callers retain the historical synchronous reset API.
        self.assertIn("self.reset_transient_state()", begin)
        self.assertNotIn("self.enqueue_transient_reset()", begin)
        # Only classic replay may queue the reset and rely on same-stream ordering.
        self.assertIn("self.enqueue_transient_reset()", deferred)
        self.assertIn("fn begin_document_deferred_for_replay", backend)
        self.assertIn("if collect_timing {", replay)
        self.assertIn("runtime.begin_document()?;", replay)
        self.assertIn("runtime.begin_document_deferred_for_replay()?;", replay)

        # These payloads are hidden behind counts or are rebuilt by start_tick.
        for redundant in (
            "b.active,",
            "b.active_value,",
            "b.block_winner_neuron,",
            "b.block_winner_value,",
            "b.ring_source,",
            "b.ring_activation,",
            "b.ring_weight,",
            "b.active_context_slots,",
            "b.active_context_scales,",
        ):
            self.assertNotIn(redundant, reset)
        for required in (
            "b.active_count,",
            "b.ring_count,",
            "b.active_context_count,",
            "b.recurrent_eligibility,",
            "b.input_eligibility,",
        ):
            self.assertIn(required, reset)

    def test_profiled_homeostasis_uses_production_sync_schedule(self):
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()
        grouped = kernels.split("leo_shared_wavefront_persistent_grouped_body", 1)[1].split(
            'extern "C" __global__ void\n__launch_bounds__(256, 2)\nleo_shared_wavefront_persistent_grouped',
            1,
        )[0]
        replay = kernels.split("leo_train_cooperative_body", 1)[1].split(
            'extern "C" __global__ void leo_train_cooperative(', 1
        )[0]

        self.assertNotIn(
            "if (PROFILED) cooperative_groups::this_grid().sync();", grouped
        )
        self.assertNotIn(
            "LEO_CUDA_REPLAY_PROFILE_INHIBITORY, phase_started", replay
        )
        self.assertIn(
            "The HOMEOSTASIS counter below measures the combined interval", replay
        )

    def test_replay_target_zero_fuses_begin_step_into_one_ordered_target_batch(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        replay = training.split("fn replay_target_range_impl", 1)[1].split(
            "#[cfg(test)]", 1
        )[0]

        self.assertIn("if next_target == 0 {", replay)
        self.assertIn(
            "target_batch.push((BEGIN_DOCUMENT, Some(story[0] as u32)));", replay
        )
        self.assertIn("next_target = 1;", replay)
        self.assertEqual(replay.count("runtime.training_step_batch(&target_batch"), 1)
        self.assertNotIn(
            "training_step_batch(&[(BEGIN_DOCUMENT, Some(story[0] as u32))]", replay
        )

    def test_parallel_multi_gpu_replay_is_opt_in_and_keeps_budget_fp32(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        flag = training.split("fn multi_gpu_parallel_replay_enabled", 1)[1].split(
            "#[derive(Debug, Clone, Copy, Default)]", 1
        )[0]
        self.assertIn("LEO_MULTI_GPU_PARALLEL_REPLAY", flag)
        self.assertIn(".unwrap_or(false)", flag)
        self.assertIn("run_parallel_replay", training)
        self.assertIn("apply_sum_deltas", training)
        self.assertIn("sum_local_trajectories", training)
        self.assertIn('"fp32\\\":true', training)
        self.assertIn("runtime.model().config.replay.fraction", training)


if __name__ == "__main__":
    unittest.main()
