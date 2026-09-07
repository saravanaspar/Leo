//! Backend-neutral runtime boundary.
//!
//! CPU and CUDA executors share this object-safe boundary. The CPU runtime is
//! the numerical reference; the CUDA executor implements the same equations
//! with persistent model and transient recurrent/eligibility state on device.
//! Both executors use the same serialized checkpoint representation.

use crate::parallel::{MergeMetrics, SparseModelDelta};
use crate::runtime::ParameterChanges;
use crate::{LeoError, LeoResult, Model, Permission, Runtime, StepMetrics};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Auto,
    Cpu,
    Gpu,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }

    pub fn is_available(self) -> bool {
        match self {
            Self::Auto | Self::Cpu => true,
            Self::Gpu => gpu_device_count().is_ok(),
        }
    }
}

impl Display for BackendKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = LeoError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err(LeoError::configuration(format!(
                "invalid backend '{value}'; expected auto, cpu, or gpu"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub available: bool,
    pub training: bool,
    pub evaluation: bool,
    pub generation: bool,
    pub device_resident_model: bool,
    pub batched_documents: bool,
}

impl BackendCapabilities {
    pub const fn cpu() -> Self {
        Self {
            available: true,
            training: true,
            evaluation: true,
            generation: true,
            device_resident_model: false,
            batched_documents: false,
        }
    }

    pub fn gpu() -> Self {
        let available = gpu_device_count().is_ok();
        Self {
            available,
            training: available,
            evaluation: available,
            generation: available,
            device_resident_model: available,
            batched_documents: available,
        }
    }

    const fn active_gpu() -> Self {
        Self {
            available: true,
            training: true,
            evaluation: true,
            generation: true,
            device_resident_model: true,
            batched_documents: true,
        }
    }
}

#[cfg(target_os = "linux")]
fn gpu_device_count() -> LeoResult<usize> {
    crate::cuda::CudaRuntime::probe()
}

#[cfg(not(target_os = "linux"))]
fn gpu_device_count() -> LeoResult<usize> {
    Err(LeoError::backend(
        "the CUDA backend is supported only on Linux builds",
    ))
}

pub fn available_gpu_devices() -> LeoResult<usize> {
    gpu_device_count()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceScalarType {
    F32,
    U32,
    U64,
    U8,
}

impl DeviceScalarType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::U64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBufferSpec {
    pub name: &'static str,
    pub scalar_type: DeviceScalarType,
    pub elements: usize,
    pub access: DeviceAccess,
}

impl DeviceBufferSpec {
    pub fn bytes(&self) -> usize {
        self.elements.saturating_mul(self.scalar_type.byte_width())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceModelLayout {
    pub buffers: Vec<DeviceBufferSpec>,
    pub total_bytes: usize,
    pub read_only_bytes: usize,
    pub read_write_bytes: usize,
}

impl DeviceModelLayout {
    pub fn for_model(model: &Model) -> Self {
        use DeviceAccess::{ReadOnly, ReadWrite};
        use DeviceScalarType::{F32, U32, U64, U8};

        let buffers = vec![
            DeviceBufferSpec {
                name: "neurons.threshold",
                scalar_type: F32,
                elements: model.neurons.threshold.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "neurons.excitability",
                scalar_type: F32,
                elements: model.neurons.excitability.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "neurons.neuron_type",
                scalar_type: U8,
                elements: model.neurons.neuron_type.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "recurrent.target_neuron",
                scalar_type: U32,
                elements: model.recurrent.target_neuron.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "recurrent.target_branch",
                scalar_type: U8,
                elements: model.recurrent.target_branch.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "recurrent.delay",
                scalar_type: U8,
                elements: model.recurrent.delay.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "recurrent.weight",
                scalar_type: F32,
                elements: model.recurrent.weight.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "input.targets",
                scalar_type: U32,
                elements: model.input.targets.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "input.branches",
                scalar_type: U8,
                elements: model.input.branches.len(),
                access: ReadOnly,
            },
            DeviceBufferSpec {
                name: "input.weights",
                scalar_type: F32,
                elements: model.input.weights.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "output.weights",
                scalar_type: F32,
                elements: model.output.weights.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "output.bias",
                scalar_type: F32,
                elements: model.output.bias.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "context.keys",
                scalar_type: U64,
                elements: model.context.keys.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "context.embeddings",
                scalar_type: F32,
                elements: model.context.embeddings.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "context.observations",
                scalar_type: U32,
                elements: model.context.observations.len(),
                access: ReadWrite,
            },
            DeviceBufferSpec {
                name: "context.output_weights",
                scalar_type: F32,
                elements: model.context.output_weights.len(),
                access: ReadWrite,
            },
        ];
        let mut total_bytes = 0usize;
        let mut read_only_bytes = 0usize;
        let mut read_write_bytes = 0usize;
        for buffer in &buffers {
            let bytes = buffer.bytes();
            total_bytes = total_bytes.saturating_add(bytes);
            match buffer.access {
                ReadOnly => read_only_bytes = read_only_bytes.saturating_add(bytes),
                ReadWrite => read_write_bytes = read_write_bytes.saturating_add(bytes),
            }
        }
        Self {
            buffers,
            total_bytes,
            read_only_bytes,
            read_write_bytes,
        }
    }
}

#[derive(Debug)]
pub struct DeviceStoryBatchReport {
    pub story_metrics: Vec<Vec<StepMetrics>>,
    pub merge: MergeMetrics,
}

/// GPU-native logical-story results retained before the device-local mean.
/// Multi-device data parallelism uses this additive interface to flatten every
/// story delta back into canonical worker order and apply one exact mean.
#[derive(Debug)]
pub struct DeviceStoryBatchDeltaReport {
    pub story_metrics: Vec<Vec<StepMetrics>>,
    pub merge: MergeMetrics,
    pub story_deltas: Vec<SparseModelDelta>,
    pub story_changes: Vec<ParameterChanges>,
}

pub trait RuntimeBackend: Send {
    fn backend_kind(&self) -> BackendKind;
    fn capabilities(&self) -> BackendCapabilities;
    fn fork(&self, model: Model) -> LeoResult<Box<dyn RuntimeBackend>>;
    fn model(&self) -> &Model;
    fn synchronize_model(&mut self) -> LeoResult<()>;
    fn model_mut(&mut self) -> LeoResult<&mut Model>;
    fn enable_parameter_tracking(&mut self);
    fn parameter_changes(&mut self) -> LeoResult<ParameterChanges>;

    fn synchronize_sparse_model(
        &mut self,
        canonical: &Model,
        _changes: &ParameterChanges,
    ) -> LeoResult<()> {
        *self.model_mut()? = canonical.clone();
        Ok(())
    }

    /// Commit sparse values from this backend's already-updated host model to
    /// its resident device image. CPU/reference backends need no action because
    /// their host model is the executor state.
    fn commit_sparse_host_model(&mut self, _changes: &ParameterChanges) -> LeoResult<()> {
        Ok(())
    }
    fn probabilities(&self) -> &[f32];
    fn active_neurons(&self) -> &[usize];
    fn activation(&self, neuron: usize) -> f32;
    fn begin_document(&mut self) -> LeoResult<()>;
    fn finish_document(&mut self) -> LeoResult<()>;
    fn reset_transient_state(&mut self) -> LeoResult<()>;
    fn step(
        &mut self,
        input_symbol: u32,
        target_symbol: Option<u32>,
        permission: Permission,
    ) -> LeoResult<StepMetrics>;

    /// Training-only batch path. CPU backends preserve the reference semantics by
    /// dispatching through `step`; GPU backends may queue multiple dependent
    /// steps and materialize compact metrics at a synchronization boundary.
    fn training_step_batch(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        steps
            .iter()
            .map(|(input, target)| self.step(*input, *target, permission))
            .collect()
    }

    /// Advance transient document state through unsupervised frozen steps.
    /// The default path preserves the reference implementation by executing
    /// ordinary frozen steps and discarding their metrics. GPU backends may
    /// skip output-only work that cannot affect the next recurrent state.
    fn advance_frozen_batch(&mut self, steps: &[(u32, Option<u32>)]) -> LeoResult<()> {
        self.training_step_batch(steps, Permission::Frozen)?;
        Ok(())
    }

    /// Optional GPU-native whole-story batch execution. Implementations return
    /// `None` when they do not support device-resident independent story lanes.
    fn training_story_batch(
        &mut self,
        _stories: &[Vec<u8>],
        _permission: Permission,
    ) -> LeoResult<Option<DeviceStoryBatchReport>> {
        Ok(None)
    }

    /// Optional GPU-native whole-story batch execution that retains each
    /// private story delta instead of exposing only the device-local mean.
    /// The default keeps third-party/runtime implementations source-compatible.
    fn training_story_batch_deltas(
        &mut self,
        _stories: &[Vec<u8>],
        _permission: Permission,
    ) -> LeoResult<Option<DeviceStoryBatchDeltaReport>> {
        Ok(None)
    }
}

impl RuntimeBackend for Runtime {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::cpu()
    }

    fn fork(&self, model: Model) -> LeoResult<Box<dyn RuntimeBackend>> {
        Ok(Box::new(Runtime::new(model)?))
    }

    fn model(&self) -> &Model {
        Runtime::model(self)
    }

    fn synchronize_model(&mut self) -> LeoResult<()> {
        Ok(())
    }

    fn model_mut(&mut self) -> LeoResult<&mut Model> {
        Ok(Runtime::model_mut(self))
    }

    fn enable_parameter_tracking(&mut self) {
        Runtime::enable_parameter_tracking(self);
    }

    fn parameter_changes(&mut self) -> LeoResult<ParameterChanges> {
        Ok(Runtime::parameter_changes(self))
    }

    fn probabilities(&self) -> &[f32] {
        Runtime::probabilities(self)
    }

    fn active_neurons(&self) -> &[usize] {
        Runtime::active_neurons(self)
    }

    fn activation(&self, neuron: usize) -> f32 {
        Runtime::activation(self, neuron)
    }

    fn begin_document(&mut self) -> LeoResult<()> {
        Runtime::begin_document(self);
        Ok(())
    }

    fn finish_document(&mut self) -> LeoResult<()> {
        Runtime::finish_document(self);
        Ok(())
    }

    fn reset_transient_state(&mut self) -> LeoResult<()> {
        Runtime::reset_transient_state(self);
        Ok(())
    }

    fn step(
        &mut self,
        input_symbol: u32,
        target_symbol: Option<u32>,
        permission: Permission,
    ) -> LeoResult<StepMetrics> {
        Runtime::step(self, input_symbol, target_symbol, permission)
    }
}

#[cfg(target_os = "linux")]
struct GpuRuntime {
    runtime: crate::cuda::CudaRuntime,
    shared_batch_lanes: Vec<crate::cuda::CudaBatchLane>,
}

#[cfg(target_os = "linux")]
impl GpuRuntime {
    fn new(model: Model) -> LeoResult<Self> {
        let runtime = crate::cuda::CudaRuntime::new(model).map_err(|error| {
            LeoError::backend(format!(
                "GPU backend requested, but CUDA initialization failed: {}",
                error.message()
            ))
        })?;
        Ok(Self {
            runtime,
            shared_batch_lanes: Vec::new(),
        })
    }

    fn new_on_device(model: Model, device_index: usize) -> LeoResult<Self> {
        let runtime = crate::cuda::CudaRuntime::new_on_device(model, device_index).map_err(|error| {
            LeoError::backend(format!(
                "GPU backend requested on CUDA device {device_index}, but initialization failed: {}",
                error.message()
            ))
        })?;
        Ok(Self {
            runtime,
            shared_batch_lanes: Vec::new(),
        })
    }

    fn run_story_batch_cuda(
        &mut self,
        stories: &[Vec<u8>],
        permission: Permission,
        retain_story_deltas: bool,
    ) -> LeoResult<Option<crate::cuda::CudaStoryBatchReport>> {
        if stories.is_empty() || (stories.len() < 2 && !retain_story_deltas) {
            return Ok(None);
        }
        if stories.len() > crate::cuda::GPU_STORY_BATCH_MAX_LANES {
            return Err(LeoError::backend(format!(
                "GPU-native story batching supports at most {} stories per batch; got {}",
                crate::cuda::GPU_STORY_BATCH_MAX_LANES,
                stories.len()
            )));
        }

        while self.shared_batch_lanes.len() < stories.len() {
            let lane = self.runtime.allocate_shared_batch_lane()?;
            self.shared_batch_lanes.push(lane);
        }
        crate::cuda::train_story_batch_shared_device(
            &mut self.runtime,
            &mut self.shared_batch_lanes[..stories.len()],
            stories,
            permission,
            retain_story_deltas,
        )
        .map(Some)
    }
}

#[cfg(target_os = "linux")]
impl RuntimeBackend for GpuRuntime {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Gpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::active_gpu()
    }

    fn fork(&self, model: Model) -> LeoResult<Box<dyn RuntimeBackend>> {
        let runtime = self.runtime.fork_for_model(model).map_err(|error| {
            LeoError::backend(format!(
                "GPU backend worker initialization failed: {}",
                error.message()
            ))
        })?;
        Ok(Box::new(Self {
            runtime,
            shared_batch_lanes: Vec::new(),
        }))
    }

    fn model(&self) -> &Model {
        self.runtime.model()
    }

    fn synchronize_model(&mut self) -> LeoResult<()> {
        self.runtime.synchronize_model()
    }

    fn model_mut(&mut self) -> LeoResult<&mut Model> {
        self.runtime.model_mut()
    }

    fn enable_parameter_tracking(&mut self) {
        self.runtime.enable_parameter_tracking();
    }

    fn parameter_changes(&mut self) -> LeoResult<ParameterChanges> {
        self.runtime.parameter_changes()
    }

    fn synchronize_sparse_model(
        &mut self,
        canonical: &Model,
        changes: &ParameterChanges,
    ) -> LeoResult<()> {
        self.runtime.synchronize_sparse_model(canonical, changes)
    }

    fn commit_sparse_host_model(&mut self, changes: &ParameterChanges) -> LeoResult<()> {
        self.runtime.upload_current_sparse_model(changes)
    }

    fn probabilities(&self) -> &[f32] {
        self.runtime.probabilities()
    }

    fn active_neurons(&self) -> &[usize] {
        self.runtime.active_neurons()
    }

    fn activation(&self, neuron: usize) -> f32 {
        self.runtime.activation(neuron)
    }

    fn begin_document(&mut self) -> LeoResult<()> {
        self.runtime.begin_document()
    }

    fn finish_document(&mut self) -> LeoResult<()> {
        self.runtime.finish_document()
    }

    fn reset_transient_state(&mut self) -> LeoResult<()> {
        self.runtime.reset_transient_state()
    }

    fn step(
        &mut self,
        input_symbol: u32,
        target_symbol: Option<u32>,
        permission: Permission,
    ) -> LeoResult<StepMetrics> {
        self.runtime.step(input_symbol, target_symbol, permission)
    }

    fn training_step_batch(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        self.runtime.training_step_batch(steps, permission)
    }

    fn advance_frozen_batch(&mut self, steps: &[(u32, Option<u32>)]) -> LeoResult<()> {
        self.runtime.advance_frozen_batch(steps)
    }

    fn training_story_batch(
        &mut self,
        stories: &[Vec<u8>],
        permission: Permission,
    ) -> LeoResult<Option<DeviceStoryBatchReport>> {
        let Some(report) = self.run_story_batch_cuda(stories, permission, false)? else {
            return Ok(None);
        };
        Ok(Some(DeviceStoryBatchReport {
            story_metrics: report.story_metrics,
            merge: report.merge,
        }))
    }

    fn training_story_batch_deltas(
        &mut self,
        stories: &[Vec<u8>],
        permission: Permission,
    ) -> LeoResult<Option<DeviceStoryBatchDeltaReport>> {
        let Some(report) = self.run_story_batch_cuda(stories, permission, true)? else {
            return Ok(None);
        };
        Ok(Some(DeviceStoryBatchDeltaReport {
            story_metrics: report.story_metrics,
            merge: report.merge,
            story_deltas: report.story_deltas,
            story_changes: report.story_changes,
        }))
    }
}

pub struct BackendRuntime {
    requested: BackendKind,
    resolved: BackendKind,
    inner: Box<dyn RuntimeBackend>,
}

impl BackendRuntime {
    pub fn new(model: Model, requested: BackendKind) -> LeoResult<Self> {
        match requested {
            BackendKind::Auto | BackendKind::Cpu => {
                Self::from_backend(requested, Box::new(Runtime::new(model)?))
            }
            BackendKind::Gpu => {
                #[cfg(target_os = "linux")]
                {
                    Self::from_backend(requested, Box::new(GpuRuntime::new(model)?))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let _ = model;
                    Err(LeoError::backend(
                        "GPU backend requested, but the CUDA executor is supported only on Linux",
                    ))
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    pub fn new_gpu_on_device(model: Model, device_index: usize) -> LeoResult<Self> {
        Self::from_backend(
            BackendKind::Gpu,
            Box::new(GpuRuntime::new_on_device(model, device_index)?),
        )
    }

    #[cfg(not(target_os = "linux"))]
    pub fn new_gpu_on_device(_model: Model, _device_index: usize) -> LeoResult<Self> {
        Err(LeoError::backend(
            "GPU backend requested, but the CUDA executor is supported only on Linux",
        ))
    }

    pub fn from_backend(requested: BackendKind, inner: Box<dyn RuntimeBackend>) -> LeoResult<Self> {
        let resolved = inner.backend_kind();
        if resolved == BackendKind::Auto {
            return Err(LeoError::backend(
                "runtime backend implementations must resolve to cpu or gpu",
            ));
        }
        if requested != BackendKind::Auto && requested != resolved {
            return Err(LeoError::backend(format!(
                "requested backend {requested} does not match runtime backend {resolved}"
            )));
        }
        Ok(Self {
            requested,
            resolved,
            inner,
        })
    }

    pub fn requested_backend(&self) -> BackendKind {
        self.requested
    }

    pub fn resolved_backend(&self) -> BackendKind {
        self.resolved
    }

    pub fn capabilities(&self) -> BackendCapabilities {
        self.inner.capabilities()
    }

    pub fn fork(&self, model: Model) -> LeoResult<Self> {
        Self::from_backend(self.resolved, self.inner.fork(model)?)
    }

    pub fn model(&self) -> &Model {
        self.inner.model()
    }

    pub fn synchronize_model(&mut self) -> LeoResult<()> {
        self.inner.synchronize_model()
    }

    pub fn model_mut(&mut self) -> LeoResult<&mut Model> {
        self.inner.model_mut()
    }

    pub fn enable_parameter_tracking(&mut self) {
        self.inner.enable_parameter_tracking();
    }

    pub fn parameter_changes(&mut self) -> LeoResult<ParameterChanges> {
        self.inner.parameter_changes()
    }

    pub fn synchronize_sparse_model(
        &mut self,
        canonical: &Model,
        changes: &ParameterChanges,
    ) -> LeoResult<()> {
        self.inner.synchronize_sparse_model(canonical, changes)
    }

    pub fn commit_sparse_host_model(&mut self, changes: &ParameterChanges) -> LeoResult<()> {
        self.inner.commit_sparse_host_model(changes)
    }

    pub fn probabilities(&self) -> &[f32] {
        self.inner.probabilities()
    }

    pub fn active_neurons(&self) -> &[usize] {
        self.inner.active_neurons()
    }

    pub fn activation(&self, neuron: usize) -> f32 {
        self.inner.activation(neuron)
    }

    pub fn begin_document(&mut self) -> LeoResult<()> {
        self.inner.begin_document()
    }

    pub fn finish_document(&mut self) -> LeoResult<()> {
        self.inner.finish_document()
    }

    pub fn reset_transient_state(&mut self) -> LeoResult<()> {
        self.inner.reset_transient_state()
    }

    pub fn step(
        &mut self,
        input_symbol: u32,
        target_symbol: Option<u32>,
        permission: Permission,
    ) -> LeoResult<StepMetrics> {
        self.inner.step(input_symbol, target_symbol, permission)
    }

    pub fn training_step_batch(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        self.inner.training_step_batch(steps, permission)
    }

    pub fn advance_frozen_batch(&mut self, steps: &[(u32, Option<u32>)]) -> LeoResult<()> {
        self.inner.advance_frozen_batch(steps)
    }

    pub fn training_story_batch(
        &mut self,
        stories: &[Vec<u8>],
        permission: Permission,
    ) -> LeoResult<Option<DeviceStoryBatchReport>> {
        self.inner.training_story_batch(stories, permission)
    }

    pub fn training_story_batch_deltas(
        &mut self,
        stories: &[Vec<u8>],
        permission: Permission,
    ) -> LeoResult<Option<DeviceStoryBatchDeltaReport>> {
        self.inner.training_story_batch_deltas(stories, permission)
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendKind, BackendRuntime};
    use crate::symbols::BEGIN_DOCUMENT;
    use crate::{Config, Model, Permission, Runtime};
    use std::str::FromStr;

    fn tiny_model() -> Model {
        let mut config = Config::default();
        config.model.neuron_count = 32;
        config.model.block_count = 4;
        config.model.neurons_per_block = 8;
        config.model.branches_per_neuron = 4;
        config.model.synapses_per_neuron = 4;
        config.model.input_fanout = 4;
        config.model.max_active_per_block = 2;
        config.model.max_active_global = 8;
        config.context.max_order = 2;
        config.context.slots_per_order = 16;
        config.context.probe_limit = 2;
        config.context.embedding_dim = 4;
        Model::initialize(config).unwrap()
    }

    #[test]
    fn backend_names_parse_without_aliases() {
        assert_eq!(BackendKind::from_str("auto").unwrap(), BackendKind::Auto);
        assert_eq!(BackendKind::from_str("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::from_str("gpu").unwrap(), BackendKind::Gpu);
        assert!(BackendKind::from_str("cuda").is_err());
    }

    #[test]
    fn auto_resolves_to_cpu_in_the_reference_build() {
        let runtime = BackendRuntime::new(tiny_model(), BackendKind::Auto).unwrap();
        assert_eq!(runtime.requested_backend(), BackendKind::Auto);
        assert_eq!(runtime.resolved_backend(), BackendKind::Cpu);
    }

    #[test]
    fn gpu_selection_never_silently_uses_cpu() {
        match BackendRuntime::new(tiny_model(), BackendKind::Gpu) {
            Ok(runtime) => assert_eq!(runtime.resolved_backend(), BackendKind::Gpu),
            Err(error) => {
                assert!(
                    error.message().contains("CUDA") || error.message().contains("GPU backend")
                );
            }
        }
    }

    #[test]
    fn device_layout_covers_fixed_width_model_buffers() {
        let model = tiny_model();
        let layout = super::DeviceModelLayout::for_model(&model);
        assert_eq!(layout.buffers.len(), 16);
        assert!(layout.total_bytes > 0);
        assert_eq!(
            layout.total_bytes,
            layout.read_only_bytes + layout.read_write_bytes
        );
        assert!(layout
            .buffers
            .iter()
            .any(|buffer| buffer.name == "context.keys"));
    }

    #[test]
    fn backend_fork_preserves_the_selected_executor() {
        let model = tiny_model();
        let runtime = BackendRuntime::new(model.clone(), BackendKind::Cpu).unwrap();
        let fork = runtime.fork(model).unwrap();
        assert_eq!(fork.resolved_backend(), BackendKind::Cpu);
    }

    #[test]
    fn cpu_wrapper_preserves_step_semantics() {
        let model = tiny_model();
        let mut direct = Runtime::new(model.clone()).unwrap();
        let mut selected = BackendRuntime::new(model, BackendKind::Cpu).unwrap();
        direct.begin_document();
        selected.begin_document().unwrap();
        let direct_metrics = direct
            .step(BEGIN_DOCUMENT, Some(b'A' as u32), Permission::Frozen)
            .unwrap();
        let selected_metrics = selected
            .step(BEGIN_DOCUMENT, Some(b'A' as u32), Permission::Frozen)
            .unwrap();
        assert_eq!(
            direct_metrics.predicted_symbol,
            selected_metrics.predicted_symbol
        );
        assert_eq!(direct.probabilities(), selected.probabilities());
    }
}
