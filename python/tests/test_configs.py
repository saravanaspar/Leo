from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
REQUIRED_CONFIGS = {"test.toml", "probe.toml", "tinystories.toml"}
EXPECTED_FIELDS = {
    "model": {
        "name", "seed", "neuron_count", "block_count", "neurons_per_block",
        "branches_per_neuron", "excitatory_fraction",
        "synapses_per_neuron", "input_fanout", "max_active_per_block",
        "max_active_global",
    },
    "dynamics": {
        "branch_decay", "membrane_decay", "fatigue_decay", "fatigue_gain",
        "refractory_ticks", "target_activity", "threshold_homeostasis_rate",
        "population_inhibition_rate", "population_inhibition_max",
        "adaptation_fast_decay", "adaptation_medium_decay", "adaptation_slow_decay",
        "adaptation_fast_gain", "adaptation_medium_gain", "adaptation_slow_gain",
        "membrane_reset_fraction",
    },
    "learning": {
        "output_learning_rate", "recurrent_learning_rate", "inhibitory_learning_rate",
        "eligibility_decay", "eligibility_epsilon", "surrogate_width",
        "surrogate_gain", "max_update", "weight_min", "weight_max",
        "provisional_strength", "training_strength", "verified_strength",
        "end_document_weight",
    },
    "context": {
        "embedding_dim", "max_order", "slots_per_order", "probe_limit", "learning_rate",
        "confidence_observations", "dropout_rate",
    },
    "replay": {
        "fraction", "segment_bytes", "teaching_replays", "max_verified_replays",
    },
    "training": {
        "max_dataset_passes", "validate_every_bytes", "checkpoint_every_bytes",
        "early_stop_checks", "minimum_relative_improvement",
    },
}


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


class IntegratedConfigTests(unittest.TestCase):
    def test_release_metadata_is_v1_0_1_without_semantic_schema_bump(self) -> None:
        with (ROOT / "Cargo.toml").open("rb") as handle:
            workspace = tomllib.load(handle)
        self.assertEqual(workspace["workspace"]["package"]["version"], "1.0.1")

        lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        for package in ("leo-cli", "leo-core", "leo-data", "leo-format"):
            self.assertIn(f'name = "{package}"\nversion = "1.0.1"', lock)

        self.assertIn('version: "1.0.1"', (ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        self.assertIn("# Leo v1.0.1", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn(
            "| Leo release | 1.0.1 |",
            (ROOT / "docs/SEMANTICS.md").read_text(encoding="utf-8"),
        )

        semantics = (ROOT / "crates/leo-core/src/semantics.rs").read_text(encoding="utf-8")
        for contract in (
            "MODEL_SCHEMA_VERSION",
            "TRAINING_POLICY_VERSION",
            "EXECUTION_SEMANTICS_VERSION",
            "DATASET_SCHEMA_VERSION",
            "CUDA_ABI_VERSION",
            "CHECKPOINT_SCHEMA_VERSION",
        ):
            self.assertIn(f"pub const {contract}: u32 = 1;", semantics)

    def test_only_supported_configs_exist(self) -> None:
        names = {path.name for path in CONFIG_ROOT.glob("*.toml")}
        self.assertEqual(names, REQUIRED_CONFIGS)

    def test_configs_use_the_exact_integrated_schema(self) -> None:
        for name in sorted(REQUIRED_CONFIGS):
            config = load(CONFIG_ROOT / name)
            self.assertEqual(set(config), set(EXPECTED_FIELDS), name)
            for section, fields in EXPECTED_FIELDS.items():
                self.assertEqual(set(config[section]), fields, f"{name}:{section}")

            model = config["model"]
            self.assertEqual(
                int(model["block_count"]) * int(model["neurons_per_block"]),
                int(model["neuron_count"]),
            )
            self.assertGreater(float(config["replay"]["fraction"]), 0.0)
            self.assertLessEqual(float(config["replay"]["fraction"]), 1.0)
            self.assertGreater(int(config["context"]["max_order"]), 0)
            self.assertGreater(int(config["context"]["slots_per_order"]), 0)
            self.assertGreater(int(config["context"]["probe_limit"]), 0)



    def test_learning_state_is_parameter_only(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "crates").rglob("*.rs")
        )
        for obsolete in (
            "normalized_update_benefit",
            "usefulness_decay",
            "minimum_plasticity",
            "maximum_stability",
            "recurrent.plasticity",
            "recurrent.stability",
            "recurrent.usefulness",
            "input.plasticity",
            "input.stability",
            "input.usefulness",
        ):
            self.assertNotIn(obsolete, source)

    def test_full_context_capacity_uses_latent_rows(self) -> None:
        full = load(CONFIG_ROOT / "tinystories.toml")
        context = full["context"]
        self.assertEqual(int(context["embedding_dim"]), 32)
        self.assertEqual(int(context["slots_per_order"]), 65536)
        self.assertEqual(int(context["max_order"]), 8)

    def test_core_formula_and_parallel_contracts_are_present(self) -> None:
        runtime = (ROOT / "crates/leo-core/src/runtime.rs").read_text(encoding="utf-8")
        model = (ROOT / "crates/leo-core/src/model.rs").read_text(encoding="utf-8")
        parallel = (ROOT / "crates/leo-core/src/parallel.rs").read_text(encoding="utf-8")
        cli = (ROOT / "crates/leo-cli/src/main.rs").read_text(encoding="utf-8")
        checkpoint = (ROOT / "crates/leo-format/src/lib.rs").read_text(encoding="utf-8")

        self.assertIn("let learning_scale = self.active_context_scales[context_index];", runtime)
        self.assertNotIn("order_weight_sum", runtime)
        self.assertIn("if activation > 0.0", runtime)
        self.assertIn("threshold_delta = rate * (local_error + population_error)", runtime)
        self.assertIn("pub struct WorkerState", runtime)
        self.assertIn("pub struct ContextProjection", model)
        self.assertIn("pub embeddings: Vec<f32>", model)
        self.assertIn("pub output_weights: Vec<f32>", model)
        self.assertIn("between_tracked", parallel)
        self.assertIn("worker delta was not produced from the current canonical revision", parallel)
        self.assertIn("context_deltas_for_slots", parallel)
        self.assertIn("training_benchmark", cli)
        self.assertIn("projected_seconds_1gb_2_epochs", cli)
        self.assertIn('const MAGIC: &[u8; 8] = b"PSCLS100";', checkpoint)
        self.assertIn("SemanticsContract", checkpoint)
        self.assertIn("digest_bytes", checkpoint)

        persistent_struct = model.split("pub struct Model", 1)[1].split("impl Model", 1)[0]
        for transient in ("membrane", "activation", "fatigue", "refractory", "eligibility", "delay_ring"):
            self.assertNotIn(transient, persistent_struct)

    def test_backend_switch_contract_is_present_without_formula_fork(self) -> None:
        backend = (ROOT / "crates/leo-core/src/backend.rs").read_text(encoding="utf-8")
        cli = (ROOT / "crates/leo-cli/src/main.rs").read_text(encoding="utf-8")
        train = (ROOT / "scripts/train.sh").read_text(encoding="utf-8")
        evaluate = (ROOT / "scripts/evaluate.sh").read_text(encoding="utf-8")

        cuda = (ROOT / "crates/leo-core/src/cuda.rs").read_text(encoding="utf-8")
        kernels = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text(encoding="utf-8")
        self.assertFalse((ROOT / "crates/leo-core/src/accelerator.rs").exists())
        self.assertIn("pub trait RuntimeBackend: Send", backend)
        self.assertIn("pub struct BackendRuntime", backend)
        self.assertIn("impl RuntimeBackend for Runtime", backend)
        self.assertIn("struct GpuRuntime", backend)
        self.assertIn("CudaRuntime", backend)
        self.assertIn("device_resident_model: true", backend)
        self.assertIn("batched_documents: true", backend)
        self.assertIn("training_story_batch", backend)
        self.assertIn("GPU_STORY_BATCH_MAX_LANES", cuda)
        # Device-to-device copies are execution-only synchronization for the
        # exact lane-private GPU batch; they do not introduce a learning formula fork.
        self.assertIn("cuMemcpyDtoDAsync", cuda)
        self.assertIn("copy_canonical_parameters_to_batch_lane_async", cuda)
        self.assertIn("nvrtcCompileProgram", cuda)
        self.assertIn("cuLaunchKernel", cuda)
        self.assertIn("include_str!(\"cuda_kernels.cu\")", cuda)
        self.assertIn("cuda_abi_generated.rs", cuda)
        self.assertIn("leo_cuda_abi.h", cuda)
        self.assertIn("const float* population_inhibition", kernels)
        self.assertIn("recurrent_next_eligible_list", cuda)
        self.assertIn("input_next_eligible_list", cuda)
        self.assertNotIn("DenseAccelerator", backend + cuda)
        for kernel in (
            "leo_deliver_events",
            "leo_select_blocks",
            "leo_update_recurrent_eligibility",
            "leo_forward",
            "leo_learning_signals",
            "leo_update_output",
            "leo_update_context",
            "leo_update_recurrent_weights",
            "leo_update_input_weights",
            "leo_homeostasis",
            "leo_train_story_batch",
        ):
            self.assertIn(kernel, kernels)
        self.assertIn("fork_for_model", cuda)
        self.assertIn("fork_for_model", backend)
        self.assertIn("Arc<SharedCuda>", cuda)
        self.assertNotIn("GPU execution kernels are not linked", backend)
        self.assertIn("BackendRuntime", (ROOT / "docs/TDD.md").read_text(encoding="utf-8"))
        self.assertIn('"backend" => command_backend', cli)
        self.assertIn('value("backend")', cli)
        self.assertIn("LEO_BACKEND", cli)
        self.assertIn('--backend "$BACKEND"', train)
        self.assertIn('--backend "$BACKEND"', evaluate)

        runtime = (ROOT / "crates/leo-core/src/runtime.rs").read_text(encoding="utf-8")
        self.assertNotIn("BackendKind", runtime)
        self.assertNotIn("BackendRuntime", runtime)

    def test_only_supported_scripts_exist(self) -> None:
        scripts = {path.name for path in (ROOT / "scripts").glob("*.sh")}
        self.assertEqual(
            scripts,
            {
                "check.sh",
                "check_gpu.sh",
                "data.sh",
                "train.sh",
                "evaluate.sh",
                "profile_cuda.sh",
                "debug_replay.sh",
            },
        )

    def test_dynamic_topology_code_is_absent(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "crates").rglob("*.rs")
        )
        for obsolete in (
            "GrowthConfig",
            ".growth",
            "last_growth_byte",
            "initial_synapses_per_neuron",
            "reserve_synapses_per_neuron",
            "recurrent.enabled",
            "new_synapses",
            "self_feed",
        ):
            self.assertNotIn(obsolete, source)

    def test_sizes_are_ordered_for_test_probe_and_training(self) -> None:
        sizes = [
            int(load(CONFIG_ROOT / name)["model"]["neuron_count"])
            for name in ("test.toml", "probe.toml", "tinystories.toml")
        ]
        self.assertEqual(sizes, [128, 4096, 32768])


if __name__ == "__main__":
    unittest.main()
