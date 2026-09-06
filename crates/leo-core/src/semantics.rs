//! Versioned contracts that define what a Leo training run means.
//!
//! These versions are intentionally independent from the crate release version.
//! Execution code may become faster without changing these values; a value is
//! bumped only when the corresponding persisted or learning contract changes.

pub const LEO_RELEASE_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MODEL_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_POLICY_VERSION: u32 = 1;
pub const EXECUTION_SEMANTICS_VERSION: u32 = 1;
pub const DATASET_SCHEMA_VERSION: u32 = 1;
pub const CUDA_ABI_VERSION: u32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;

pub const TRAINING_POLICY_NAME: &str = "bounded-surprise-replay-v1";
pub const EXECUTION_SEMANTICS_NAME: &str = "leo-exact-fp32-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsContract {
    pub model_schema: u32,
    pub training_policy: u32,
    pub execution_semantics: u32,
    pub dataset_schema: u32,
    pub cuda_abi: u32,
}

impl SemanticsContract {
    pub const CURRENT: Self = Self {
        model_schema: MODEL_SCHEMA_VERSION,
        training_policy: TRAINING_POLICY_VERSION,
        execution_semantics: EXECUTION_SEMANTICS_VERSION,
        dataset_schema: DATASET_SCHEMA_VERSION,
        cuda_abi: CUDA_ABI_VERSION,
    };

    pub const fn as_array(self) -> [u32; 5] {
        [
            self.model_schema,
            self.training_policy,
            self.execution_semantics,
            self.dataset_schema,
            self.cuda_abi,
        ]
    }
}
