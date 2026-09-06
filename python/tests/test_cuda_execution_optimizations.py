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

    def test_physical_batching_does_not_replace_logical_mean_batch(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        planner = (ROOT / "crates/leo-core/src/cuda_plan.rs").read_text()

        self.assertIn("physical_lane_chunk", cuda)
        self.assertIn("1.0f32 / lanes.len() as f32", cuda)
        self.assertIn("launch_sparse_apply_and_reset", cuda)
        self.assertIn("plan_never_changes_logical_lane_count", planner)
        self.assertNotIn("workers = execution_plan", cuda)

    def test_tuning_cache_async_transfers_and_graphs_are_execution_only(self):
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        planner = (ROOT / "crates/leo-core/src/cuda_plan.rs").read_text()
        semantics = (ROOT / "docs/SEMANTICS.md").read_text()

        for marker in (
            "leo-ptx-v1",
            "cuMemcpyHtoDAsync",
            "cuMemcpyDtoHAsync",
            "cuMemAllocHost",
            "cuStreamBeginCapture",
            "cuGraphLaunch",
            "capture_sparse_apply_graph",
        ):
            self.assertIn(marker, cuda)
        self.assertIn("OBSERVATIONS_PER_CANDIDATE", planner)
        self.assertIn("persist_profile", planner)
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
        self.assertIn("leo-cuda-plan-v3", cuda)
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
        self.assertIn("graph_launches", gpu_check)


    def test_replay_prefix_uses_frozen_state_fast_path_without_changing_replay_budget(self):
        training = (ROOT / "crates/leo-cli/src/training.rs").read_text()
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text()
        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text()
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()

        self.assertIn("runtime.advance_frozen_batch(&prefix)?;", training)
        self.assertIn("fn advance_frozen_batch", backend)
        self.assertIn("leo_advance_frozen_persistent", cuda)
        self.assertIn("leo_advance_frozen_persistent", kernels)
        frozen_kernel = kernels.split(
            'extern "C" __global__ void leo_advance_frozen_persistent', 1
        )[0].rsplit("__device__ void leo_advance_frozen_story_block", 1)[1]
        self.assertIn("leo_p_context_advance_history", frozen_kernel)
        self.assertIn("leo_p_post_and_emit", frozen_kernel)
        self.assertNotIn("leo_p_forward", frozen_kernel)
        self.assertNotIn("leo_p_capture_training_step", frozen_kernel)
        self.assertNotIn("leo_p_cache_surrogate", frozen_kernel)
        self.assertIn("runtime.model().config.replay.fraction", training)
        self.assertIn("runtime.model().config.replay.segment_bytes", training)
        self.assertIn("let cleanup_after = index + 1 == ranges.len();", training)
        self.assertIn("if cleanup_after {", training)

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



if __name__ == "__main__":
    unittest.main()
