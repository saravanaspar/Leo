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
        self.assertIn("SparseModelDelta::from_tracked_values", batch)
        self.assertEqual(batch.count("apply_mean_deltas("), 1)
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
        self.assertIn("fused_wavefront_only", cuda)
        self.assertIn("profiled_capacity_blocks", cuda)
        self.assertIn("phase_profile_counters", cuda)
        self.assertIn("phase_profile_sample_stride", cuda)
        self.assertIn("shared_wavefront_fused_profiled", cuda)

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
        self.assertIn("LEO_CUDA_REPLAY_PROFILE", script)
        self.assertIn("CUDA replay diagnostics OK", gpu_gate)
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
        self.assertIn("do NOT first build a device-local mean", cuda)
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
        self.assertIn("runtime.synchronize_packed_model(&replay_update)?", training)
        self.assertIn("runtime.synchronize_packed_model(&update)?", training)

        self.assertIn("struct PackedSparseModelUpdate", parallel)
        self.assertIn("fn apply_sparse_mean", parallel)
        self.assertNotIn("BTreeMap::<usize, f64>", parallel)
        self.assertNotIn("project_parameter_constraints(master);", parallel)
        self.assertIn("project_parameter_constraints_sparse(master, deltas)?", parallel)
        self.assertIn("validate_sparse_merge(master, deltas)?", parallel)

        self.assertIn("snapshot_batch_lane_values_batched", cuda)
        self.assertIn("batch snapshot counts", cuda)
        self.assertIn("let count_elements = lane_count.saturating_mul(5);", cuda)
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


if __name__ == "__main__":
    unittest.main()
