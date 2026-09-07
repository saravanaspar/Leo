//! Leo's byte-level sparse recurrent learning runtime.

pub mod artifact;
pub mod backend;
pub mod config;
#[cfg(target_os = "linux")]
mod cuda;
#[cfg(target_os = "linux")]
mod cuda_plan;
pub mod learning;
pub mod metrics;
pub mod model;
pub mod parallel;
pub mod permission;
pub mod rng;
pub mod runtime;
pub mod semantics;
pub mod symbols;
pub mod utf8;

use std::fmt::{Display, Formatter};

pub use artifact::{digest_bytes, digest_file, ArtifactDigest, Sha256, DIGEST_ALGORITHM};
pub use backend::{
    available_gpu_devices, BackendCapabilities, BackendKind, BackendRuntime, DeviceAccess,
    DeviceBufferSpec, DeviceModelLayout, DeviceScalarType, DeviceStoryBatchReport, RuntimeBackend,
};
pub use config::Config;
pub use metrics::{StepMetrics, TrainingStatistics};
pub use model::Model;
pub use parallel::{apply_mean_deltas, merged_parameter_changes, MergeMetrics, SparseModelDelta};
pub use permission::Permission;
pub use runtime::{ParameterChanges, Runtime};
pub use semantics::SemanticsContract;

pub type LeoResult<T> = Result<T, LeoError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeoErrorKind {
    Usage,
    Configuration,
    Dataset,
    CheckpointCorrupt,
    CheckpointIncompatible,
    Backend,
    Cuda,
    Numerical,
    Io,
    Internal,
}

#[derive(Debug, Clone)]
pub struct LeoError {
    kind: LeoErrorKind,
    message: String,
}

impl LeoError {
    pub fn new(kind: LeoErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Usage, message)
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Configuration, message)
    }

    pub fn dataset(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Dataset, message)
    }

    pub fn checkpoint_corrupt(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::CheckpointCorrupt, message)
    }

    pub fn checkpoint_incompatible(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::CheckpointIncompatible, message)
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Backend, message)
    }

    pub fn cuda(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Cuda, message)
    }

    pub fn numerical(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Numerical, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(LeoErrorKind::Internal, message)
    }

    pub const fn kind(&self) -> LeoErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for LeoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for LeoError {}

impl From<std::io::Error> for LeoError {
    fn from(error: std::io::Error) -> Self {
        Self::new(LeoErrorKind::Io, error.to_string())
    }
}
