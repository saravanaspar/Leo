//! Full device-resident CUDA runtime for Leo.
//!
//! The serialized model remains identical to the CPU reference representation,
//! but the GPU executor owns persistent model tensors plus recurrent dynamics,
//! delayed-event state, context resolution, eligibility traces, and learning on
//! device. The host model is a checkpoint/merge mirror synchronized only at
//! explicit safe points.

use crate::cuda_plan::{
    atomic_write, cache_root, CudaBatchTelemetry, CudaExecutionPlan, CudaExecutionTuner,
    CudaTuningLimits,
};
use crate::parallel::{
    apply_mean_deltas, merged_parameter_changes, MergeMetrics, PackedSparseModelUpdate,
    SparseModelDelta, StatisticsDelta, TrackedModelValues,
};
use crate::runtime::ParameterChanges;
use crate::symbols::{
    is_input_symbol, output_index_to_symbol, output_symbol_to_index, BEGIN_DOCUMENT, END_DOCUMENT,
    END_DOCUMENT_OUTPUT_INDEX, OUTPUT_CLASSES,
};
use crate::{LeoError, LeoResult, Model, Permission, StepMetrics};
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::mem;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

const CUDA_SUCCESS: c_int = 0;
const CU_EVENT_DEFAULT: c_uint = 0;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const NVRTC_SUCCESS: c_int = 0;
const RTLD_NOW: c_int = 2;
const THREADS: c_uint = 256;
const FORWARD_THREADS: c_uint = 512;
const GLOBAL_SELECTION_THREADS: c_uint = 1024;
const PERSISTENT_THREADS: c_uint = 256;
const SHARED_BATCH_THREADS: c_uint = 256;
const TRAINING_STEP_BATCH_CAPACITY: usize = 4096;
pub(crate) const GPU_STORY_BATCH_MAX_LANES: usize = 128;
const CUDA_PHASE_PROFILE_COUNTER_COUNT: usize = 10;
const CUDA_PHASE_PROFILE_SAMPLES: usize = 0;
const CUDA_PHASE_PROFILE_PRE: usize = 1;
const CUDA_PHASE_PROFILE_SELECT: usize = 2;
const CUDA_PHASE_PROFILE_POST_SELECT: usize = 3;
const CUDA_PHASE_PROFILE_CACHE_SURROGATE: usize = 4;
const CUDA_PHASE_PROFILE_POST_CORE: usize = 5;
const CUDA_PHASE_PROFILE_LEARNING_SIGNALS: usize = 6;
const CUDA_PHASE_PROFILE_POST_DELTAS: usize = 7;
const CUDA_PHASE_PROFILE_HOMEOSTASIS: usize = 8;
const CUDA_PHASE_PROFILE_CAPTURE: usize = 9;
const CUDA_PHASE_PROFILE_DEFAULT_SAMPLE_STRIDE: u64 = 64;
const CUDA_REPLAY_PROFILE_COUNTER_COUNT: usize = 17;
const CUDA_FROZEN_PROFILE_COUNTER_COUNT: usize = 5;
const CUDA_REPLAY_PROFILE_DEFAULT_SAMPLE_STRIDE: u64 = 1;
const CUDA_REPLAY_PROFILE_SAMPLES: usize = 0;
const CUDA_REPLAY_PROFILE_PRE: usize = 1;
const CUDA_REPLAY_PROFILE_SELECT_BLOCKS: usize = 2;
const CUDA_REPLAY_PROFILE_SELECT_GLOBAL: usize = 3;
const CUDA_REPLAY_PROFILE_CACHE_SURROGATE: usize = 4;
const CUDA_REPLAY_PROFILE_RECURRENT_ELIGIBILITY: usize = 5;
const CUDA_REPLAY_PROFILE_INPUT_ELIGIBILITY: usize = 6;
const CUDA_REPLAY_PROFILE_POST_EMIT: usize = 7;
const CUDA_REPLAY_PROFILE_FORWARD: usize = 8;
const CUDA_REPLAY_PROFILE_LEARNING_SIGNALS: usize = 9;
const CUDA_REPLAY_PROFILE_OUTPUT_UPDATE: usize = 10;
const CUDA_REPLAY_PROFILE_CONTEXT_UPDATE: usize = 11;
const CUDA_REPLAY_PROFILE_RECURRENT_UPDATE: usize = 12;
const CUDA_REPLAY_PROFILE_INPUT_UPDATE: usize = 13;
const CUDA_REPLAY_PROFILE_INHIBITORY: usize = 14;
const CUDA_REPLAY_PROFILE_HOMEOSTASIS: usize = 15;
const CUDA_REPLAY_PROFILE_CAPTURE: usize = 16;
const CUDA_FROZEN_PROFILE_SAMPLES: usize = 0;
const CUDA_FROZEN_PROFILE_PRE: usize = 1;
const CUDA_FROZEN_PROFILE_SELECT_BLOCKS: usize = 2;
const CUDA_FROZEN_PROFILE_SELECT_GLOBAL: usize = 3;
const CUDA_FROZEN_PROFILE_POST_EMIT: usize = 4;
const RING_BUCKETS: usize = 9;
const MAX_BLOCK_WINNERS: usize = 64;
const MAX_GLOBAL_BLOCK_WINNERS: usize = 1024;
const SPARSE_LIST_WORK_ITEMS: usize = 65_536;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: c_int = 95;
const CUDA_KERNEL_BODY: &str = include_str!("cuda_kernels.cu");
const CUDA_ABI_HEADER: &str = include_str!(concat!(env!("OUT_DIR"), "/leo_cuda_abi.h"));
include!(concat!(env!("OUT_DIR"), "/cuda_abi_generated.rs"));

fn cuda_kernel_source() -> String {
    let mut source = String::with_capacity(CUDA_ABI_HEADER.len() + CUDA_KERNEL_BODY.len() + 1);
    source.push_str(CUDA_ABI_HEADER);
    source.push('\n');
    source.push_str(CUDA_KERNEL_BODY);
    source
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn env_positive_u32_opt(name: &str) -> Option<u32> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
}

fn env_positive_u64(name: &str, fallback: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(fallback)
}

#[derive(Debug, Clone, Copy, Default)]
struct CudaDebugOptions {
    summary: bool,
    memory: bool,
    chunks: bool,
    launches: bool,
    sync: bool,
    transfers: bool,
    state: bool,
}

impl CudaDebugOptions {
    fn from_env() -> Self {
        let summary = env_flag("LEO_CUDA_DEBUG");
        Self {
            summary,
            memory: summary || env_flag("LEO_CUDA_DEBUG_MEMORY"),
            chunks: env_flag("LEO_CUDA_DEBUG_CHUNKS"),
            launches: env_flag("LEO_CUDA_DEBUG_LAUNCHES"),
            sync: env_flag("LEO_CUDA_DEBUG_SYNC"),
            transfers: env_flag("LEO_CUDA_DEBUG_TRANSFERS"),
            state: env_flag("LEO_CUDA_DEBUG_STATE"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CudaReplayProfileOptions {
    target_stride: Option<u64>,
    prefix_stride: Option<u64>,
}

impl CudaReplayProfileOptions {
    fn from_env() -> Self {
        let both = env_flag("LEO_CUDA_REPLAY_PROFILE");
        let stride = env_positive_u64(
            "LEO_CUDA_REPLAY_PROFILE_STRIDE",
            CUDA_REPLAY_PROFILE_DEFAULT_SAMPLE_STRIDE,
        );
        Self {
            target_stride: (both || env_flag("LEO_CUDA_REPLAY_PROFILE_TARGET")).then_some(stride),
            prefix_stride: (both || env_flag("LEO_CUDA_REPLAY_PROFILE_PREFIX")).then_some(stride),
        }
    }
}

fn cuda_replay_cooperative_enabled() -> bool {
    !env::var("LEO_CUDA_REPLAY_COOPERATIVE")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn cuda_shared_persistent_enabled() -> bool {
    !env::var("LEO_CUDA_SHARED_PERSISTENT")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn cuda_shared_grouped_enabled() -> bool {
    !env::var("LEO_CUDA_SHARED_GROUPED")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn cuda_device_batch_merge_enabled() -> bool {
    !env::var("LEO_CUDA_DEVICE_BATCH_MERGE")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn cuda_full_step_metrics_enabled() -> bool {
    env_flag("LEO_CUDA_FULL_STEP_METRICS")
}

#[derive(Debug, Clone, Copy)]
struct CudaPhaseProfileReport {
    logical_lanes: usize,
    fused_launches_total: u64,
    sample_stride: u64,
    planned_fused_blocks: u32,
    geometry_skipped_samples: u64,
    sampled_grid_blocks_max: u32,
    skipped_grid_blocks_max: u32,
}

fn cuda_phase_profile_sample_stride() -> Option<u64> {
    env_flag("LEO_CUDA_PHASE_PROFILE").then(|| {
        env_positive_u64(
            "LEO_CUDA_PHASE_PROFILE_STRIDE",
            CUDA_PHASE_PROFILE_DEFAULT_SAMPLE_STRIDE,
        )
    })
}

impl CudaConfig {
    fn from_model(model: &Model) -> LeoResult<Self> {
        let c = &model.config;
        let to_u32 = |name: &str, value: usize| {
            u32::try_from(value)
                .map_err(|_| LeoError::cuda(format!("{name} exceeds CUDA u32 capacity")))
        };
        let refractory_ticks = u32::try_from(c.dynamics.refractory_ticks)
            .map_err(|_| LeoError::cuda("refractory_ticks exceeds CUDA u32 capacity"))?;
        Ok(Self {
            neuron_count: to_u32("neuron_count", c.model.neuron_count)?,
            block_count: to_u32("block_count", c.model.block_count)?,
            neurons_per_block: to_u32("neurons_per_block", c.model.neurons_per_block)?,
            branches_per_neuron: to_u32("branches_per_neuron", c.model.branches_per_neuron)?,
            synapses_per_neuron: to_u32("synapses_per_neuron", c.model.synapses_per_neuron)?,
            input_fanout: to_u32("input_fanout", c.model.input_fanout)?,
            max_active_per_block: to_u32("max_active_per_block", c.model.max_active_per_block)?,
            max_active_global: to_u32("max_active_global", c.model.max_active_global)?,
            context_max_order: to_u32("context.max_order", c.context.max_order)?,
            context_embedding_dim: to_u32("context.embedding_dim", c.context.embedding_dim)?,
            context_slots_per_order: to_u32("context.slots_per_order", c.context.slots_per_order)?,
            context_probe_limit: to_u32("context.probe_limit", c.context.probe_limit)?,
            context_confidence_observations: c.context.confidence_observations,
            refractory_ticks,
            branch_decay: c.dynamics.branch_decay,
            membrane_decay: c.dynamics.membrane_decay,
            fatigue_decay: c.dynamics.fatigue_decay,
            fatigue_gain: c.dynamics.fatigue_gain,
            target_activity: c.dynamics.target_activity,
            threshold_homeostasis_rate: c.dynamics.threshold_homeostasis_rate,
            population_inhibition_rate: c.dynamics.population_inhibition_rate,
            population_inhibition_max: c.dynamics.population_inhibition_max,
            adaptation_fast_decay: c.dynamics.adaptation_fast_decay,
            adaptation_medium_decay: c.dynamics.adaptation_medium_decay,
            adaptation_slow_decay: c.dynamics.adaptation_slow_decay,
            adaptation_fast_gain: c.dynamics.adaptation_fast_gain,
            adaptation_medium_gain: c.dynamics.adaptation_medium_gain,
            adaptation_slow_gain: c.dynamics.adaptation_slow_gain,
            membrane_reset_fraction: c.dynamics.membrane_reset_fraction,
            output_learning_rate: c.learning.output_learning_rate,
            recurrent_learning_rate: c.learning.recurrent_learning_rate,
            inhibitory_learning_rate: c.learning.inhibitory_learning_rate,
            eligibility_decay: c.learning.eligibility_decay,
            eligibility_epsilon: c.learning.eligibility_epsilon,
            surrogate_width: c.learning.surrogate_width,
            surrogate_gain: c.learning.surrogate_gain,
            max_update: c.learning.max_update,
            weight_min: c.learning.weight_min,
            weight_max: c.learning.weight_max,
            context_learning_rate: c.context.learning_rate,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaStepCounters {
    suprathreshold: u32,
    block_selected: u32,
    global_clipped: u32,
    context_cells: u32,
    context_probes: u32,
    eligible_recurrent: u32,
    eligible_input: u32,
    learning_eligibility_count: u32,
    learning_eligibility_abs_sum: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaTrainingStepRecord {
    loss: f32,
    neural_loss: f32,
    population_inhibition: f32,
    learning_eligibility_abs_sum: f32,
    predicted_index: u32,
    active_count: u32,
    suprathreshold: u32,
    block_selected: u32,
    global_clipped: u32,
    context_cells: u32,
    context_probes: u32,
    eligible_recurrent: u32,
    eligible_input: u32,
    learning_eligibility_count: u32,
    error_code: u32,
}

/// Production training only needs the target loss for replay selection, the
/// active count for coarse training statistics, and the sticky numerical error
/// code.  Keep the much larger `CudaTrainingStepRecord` for validation,
/// profiling and the reference/debug path, but avoid copying/computing all of
/// those diagnostics for every learned byte.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaFastTrainingStepRecord {
    loss: f32,
    active_count: u32,
    error_code: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaPersistentStep {
    symbol: u32,
    target_index: i32,
    context_enabled: u32,
    supervised_strength: f32,
}

type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CuUuid {
    bytes: [u8; 16],
}
type NvrtcProgram = *mut c_void;

type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> c_int;
type CuDeviceGetUuid = unsafe extern "C" fn(*mut CuUuid, CuDevice) -> c_int;
type CuDeviceGetPciBusId = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> c_int;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> c_int;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> c_int;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> c_int;
type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> c_int;
type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> c_int;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> c_int;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, c_uint) -> c_int;
type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> c_int;
type CuStreamSynchronize = unsafe extern "C" fn(CuStream) -> c_int;
type CuEventCreate = unsafe extern "C" fn(*mut CuEvent, c_uint) -> c_int;
type CuEventDestroy = unsafe extern "C" fn(CuEvent) -> c_int;
type CuEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> c_int;
type CuEventSynchronize = unsafe extern "C" fn(CuEvent) -> c_int;
type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> c_int;
type CuStreamWaitEvent = unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> c_int;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> c_int;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> c_int;
type CuModuleGetFunction = unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> c_int;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> c_int;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> c_int;
type CuMemAllocHost = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> c_int;
type CuMemsetD8 = unsafe extern "C" fn(CuDevicePtr, u8, usize) -> c_int;
type CuMemsetD32 = unsafe extern "C" fn(CuDevicePtr, u32, usize) -> c_int;
type CuMemsetD8Async = unsafe extern "C" fn(CuDevicePtr, u8, usize, CuStream) -> c_int;
type CuMemsetD32Async = unsafe extern "C" fn(CuDevicePtr, u32, usize, CuStream) -> c_int;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> c_int;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> c_int;
type CuMemcpyHtoDAsync = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> c_int;
type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> c_int;
type CuMemcpyDtoDAsync = unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize, CuStream) -> c_int;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;
type CuLaunchCooperativeKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CuStream,
    *mut *mut c_void,
) -> c_int;
type CuOccupancyMaxActiveBlocksPerMultiprocessor =
    unsafe extern "C" fn(*mut c_int, CuFunction, c_int, usize) -> c_int;
type CuGetErrorString = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;
type CuStreamBeginCapture = unsafe extern "C" fn(CuStream, c_int) -> c_int;
type CuStreamEndCapture = unsafe extern "C" fn(CuStream, *mut CuGraph) -> c_int;
type CuGraphInstantiateWithFlags = unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> c_int;
type CuGraphLaunch = unsafe extern "C" fn(CuGraphExec, CuStream) -> c_int;
type CuGraphDestroy = unsafe extern "C" fn(CuGraph) -> c_int;
type CuGraphExecDestroy = unsafe extern "C" fn(CuGraphExec) -> c_int;

type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type NvrtcCompileProgram = unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int;
type NvrtcGetPtxSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetPtx = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> c_int;
type NvrtcGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type NvrtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

struct DynamicLibrary {
    handle: *mut c_void,
}

unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}

impl DynamicLibrary {
    fn open(candidates: &[&str]) -> LeoResult<Self> {
        for candidate in candidates {
            let name = CString::new(*candidate)
                .map_err(|_| LeoError::cuda("invalid dynamic library name"))?;
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                return Ok(Self { handle });
            }
        }
        let detail = unsafe {
            let error = dlerror();
            if error.is_null() {
                "unknown dynamic loader error".to_owned()
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        };
        Err(LeoError::cuda(format!(
            "CUDA dynamic library unavailable (tried {}): {detail}",
            candidates.join(", ")
        )))
    }

    fn symbol(&self, names: &[&str]) -> LeoResult<*mut c_void> {
        for name in names {
            let symbol =
                CString::new(*name).map_err(|_| LeoError::cuda("invalid CUDA symbol name"))?;
            let address = unsafe { dlsym(self.handle, symbol.as_ptr()) };
            if !address.is_null() {
                return Ok(address);
            }
        }
        Err(LeoError::cuda(format!(
            "CUDA symbol unavailable: {}",
            names.join(" or ")
        )))
    }

    fn symbol_optional(&self, names: &[&str]) -> Option<*mut c_void> {
        for name in names {
            let Ok(symbol) = CString::new(*name) else {
                continue;
            };
            let address = unsafe { dlsym(self.handle, symbol.as_ptr()) };
            if !address.is_null() {
                return Some(address);
            }
        }
        None
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

macro_rules! load_function {
    ($library:expr, $names:expr, $type:ty) => {{
        let address = $library.symbol($names)?;
        unsafe { mem::transmute::<*mut c_void, $type>(address) }
    }};
}

macro_rules! load_optional_function {
    ($library:expr, $names:expr, $type:ty) => {{
        $library
            .symbol_optional($names)
            .map(|address| unsafe { mem::transmute::<*mut c_void, $type>(address) })
    }};
}

struct DriverFunctions {
    _library: DynamicLibrary,
    init: CuInit,
    driver_get_version: CuDriverGetVersion,
    device_get_name: CuDeviceGetName,
    device_get_uuid: Option<CuDeviceGetUuid>,
    device_get_pci_bus_id: CuDeviceGetPciBusId,
    device_total_mem: CuDeviceTotalMem,
    device_get_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_get_attribute: CuDeviceGetAttribute,
    ctx_create: CuCtxCreate,
    ctx_destroy: CuCtxDestroy,
    ctx_set_current: CuCtxSetCurrent,
    stream_create: CuStreamCreate,
    stream_destroy: CuStreamDestroy,
    stream_synchronize: CuStreamSynchronize,
    stream_wait_event: CuStreamWaitEvent,
    event_create: CuEventCreate,
    event_destroy: CuEventDestroy,
    event_record: CuEventRecord,
    event_synchronize: CuEventSynchronize,
    event_elapsed_time: CuEventElapsedTime,
    module_load_data: CuModuleLoadData,
    module_unload: CuModuleUnload,
    module_get_function: CuModuleGetFunction,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    mem_alloc_host: CuMemAllocHost,
    mem_free_host: CuMemFreeHost,
    memset_d8: CuMemsetD8,
    memset_d32: CuMemsetD32,
    memset_d8_async: CuMemsetD8Async,
    memset_d32_async: CuMemsetD32Async,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    memcpy_htod_async: CuMemcpyHtoDAsync,
    memcpy_dtoh_async: CuMemcpyDtoHAsync,
    memcpy_dtod_async: CuMemcpyDtoDAsync,
    launch_kernel: CuLaunchKernel,
    launch_cooperative_kernel: CuLaunchCooperativeKernel,
    occupancy_max_active_blocks_per_multiprocessor: CuOccupancyMaxActiveBlocksPerMultiprocessor,
    get_error_string: CuGetErrorString,
    stream_begin_capture: Option<CuStreamBeginCapture>,
    stream_end_capture: Option<CuStreamEndCapture>,
    graph_instantiate_with_flags: Option<CuGraphInstantiateWithFlags>,
    graph_launch: Option<CuGraphLaunch>,
    graph_destroy: Option<CuGraphDestroy>,
    graph_exec_destroy: Option<CuGraphExecDestroy>,
}

unsafe impl Send for DriverFunctions {}
unsafe impl Sync for DriverFunctions {}

impl DriverFunctions {
    fn load() -> LeoResult<Arc<Self>> {
        let library = DynamicLibrary::open(&["libcuda.so.1", "libcuda.so"])?;
        Ok(Arc::new(Self {
            init: load_function!(library, &["cuInit"], CuInit),
            driver_get_version: load_function!(
                library,
                &["cuDriverGetVersion"],
                CuDriverGetVersion
            ),
            device_get_name: load_function!(library, &["cuDeviceGetName"], CuDeviceGetName),
            device_get_uuid: load_optional_function!(
                library,
                &["cuDeviceGetUuid_v2", "cuDeviceGetUuid"],
                CuDeviceGetUuid
            ),
            device_get_pci_bus_id: load_function!(
                library,
                &["cuDeviceGetPCIBusId"],
                CuDeviceGetPciBusId
            ),
            device_total_mem: load_function!(
                library,
                &["cuDeviceTotalMem_v2", "cuDeviceTotalMem"],
                CuDeviceTotalMem
            ),
            device_get_count: load_function!(library, &["cuDeviceGetCount"], CuDeviceGetCount),
            device_get: load_function!(library, &["cuDeviceGet"], CuDeviceGet),
            device_get_attribute: load_function!(
                library,
                &["cuDeviceGetAttribute"],
                CuDeviceGetAttribute
            ),
            ctx_create: load_function!(library, &["cuCtxCreate_v2", "cuCtxCreate"], CuCtxCreate),
            ctx_destroy: load_function!(
                library,
                &["cuCtxDestroy_v2", "cuCtxDestroy"],
                CuCtxDestroy
            ),
            ctx_set_current: load_function!(library, &["cuCtxSetCurrent"], CuCtxSetCurrent),
            stream_create: load_function!(library, &["cuStreamCreate"], CuStreamCreate),
            stream_destroy: load_function!(
                library,
                &["cuStreamDestroy_v2", "cuStreamDestroy"],
                CuStreamDestroy
            ),
            stream_synchronize: load_function!(
                library,
                &["cuStreamSynchronize"],
                CuStreamSynchronize
            ),
            stream_wait_event: load_function!(library, &["cuStreamWaitEvent"], CuStreamWaitEvent),
            event_create: load_function!(library, &["cuEventCreate"], CuEventCreate),
            event_destroy: load_function!(
                library,
                &["cuEventDestroy_v2", "cuEventDestroy"],
                CuEventDestroy
            ),
            event_record: load_function!(library, &["cuEventRecord"], CuEventRecord),
            event_synchronize: load_function!(library, &["cuEventSynchronize"], CuEventSynchronize),
            event_elapsed_time: load_function!(
                library,
                &["cuEventElapsedTime"],
                CuEventElapsedTime
            ),
            module_load_data: load_function!(library, &["cuModuleLoadData"], CuModuleLoadData),
            module_unload: load_function!(library, &["cuModuleUnload"], CuModuleUnload),
            module_get_function: load_function!(
                library,
                &["cuModuleGetFunction"],
                CuModuleGetFunction
            ),
            mem_alloc: load_function!(library, &["cuMemAlloc_v2", "cuMemAlloc"], CuMemAlloc),
            mem_free: load_function!(library, &["cuMemFree_v2", "cuMemFree"], CuMemFree),
            mem_alloc_host: load_function!(
                library,
                &["cuMemAllocHost_v2", "cuMemAllocHost"],
                CuMemAllocHost
            ),
            mem_free_host: load_function!(library, &["cuMemFreeHost"], CuMemFreeHost),
            memset_d8: load_function!(library, &["cuMemsetD8_v2", "cuMemsetD8"], CuMemsetD8),
            memset_d32: load_function!(library, &["cuMemsetD32_v2", "cuMemsetD32"], CuMemsetD32),
            memset_d8_async: load_function!(library, &["cuMemsetD8Async"], CuMemsetD8Async),
            memset_d32_async: load_function!(library, &["cuMemsetD32Async"], CuMemsetD32Async),
            memcpy_htod: load_function!(
                library,
                &["cuMemcpyHtoD_v2", "cuMemcpyHtoD"],
                CuMemcpyHtoD
            ),
            memcpy_dtoh: load_function!(
                library,
                &["cuMemcpyDtoH_v2", "cuMemcpyDtoH"],
                CuMemcpyDtoH
            ),
            memcpy_htod_async: load_function!(
                library,
                &["cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync"],
                CuMemcpyHtoDAsync
            ),
            memcpy_dtoh_async: load_function!(
                library,
                &["cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync"],
                CuMemcpyDtoHAsync
            ),
            memcpy_dtod_async: load_function!(
                library,
                &["cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync"],
                CuMemcpyDtoDAsync
            ),
            launch_kernel: load_function!(library, &["cuLaunchKernel"], CuLaunchKernel),
            launch_cooperative_kernel: load_function!(
                library,
                &["cuLaunchCooperativeKernel"],
                CuLaunchCooperativeKernel
            ),
            occupancy_max_active_blocks_per_multiprocessor: load_function!(
                library,
                &["cuOccupancyMaxActiveBlocksPerMultiprocessor"],
                CuOccupancyMaxActiveBlocksPerMultiprocessor
            ),
            get_error_string: load_function!(library, &["cuGetErrorString"], CuGetErrorString),
            stream_begin_capture: load_optional_function!(
                library,
                &["cuStreamBeginCapture"],
                CuStreamBeginCapture
            ),
            stream_end_capture: load_optional_function!(
                library,
                &["cuStreamEndCapture"],
                CuStreamEndCapture
            ),
            graph_instantiate_with_flags: load_optional_function!(
                library,
                &["cuGraphInstantiateWithFlags"],
                CuGraphInstantiateWithFlags
            ),
            graph_launch: load_optional_function!(library, &["cuGraphLaunch"], CuGraphLaunch),
            graph_destroy: load_optional_function!(library, &["cuGraphDestroy"], CuGraphDestroy),
            graph_exec_destroy: load_optional_function!(
                library,
                &["cuGraphExecDestroy"],
                CuGraphExecDestroy
            ),
            _library: library,
        }))
    }

    fn check(&self, result: c_int, operation: &str) -> LeoResult<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut message = ptr::null();
        let detail = unsafe {
            if (self.get_error_string)(result, &mut message) == CUDA_SUCCESS && !message.is_null() {
                CStr::from_ptr(message).to_string_lossy().into_owned()
            } else {
                format!("CUDA error code {result}")
            }
        };
        Err(LeoError::cuda(format!("{operation} failed: {detail}")))
    }
}

struct NvrtcFunctions {
    _library: DynamicLibrary,
    create_program: NvrtcCreateProgram,
    compile_program: NvrtcCompileProgram,
    get_ptx_size: NvrtcGetPtxSize,
    get_ptx: NvrtcGetPtx,
    get_program_log_size: NvrtcGetProgramLogSize,
    get_program_log: NvrtcGetProgramLog,
    destroy_program: NvrtcDestroyProgram,
    get_error_string: NvrtcGetErrorString,
    version: NvrtcVersion,
}

impl NvrtcFunctions {
    fn load() -> LeoResult<Self> {
        let library = DynamicLibrary::open(&[
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
        ])?;
        Ok(Self {
            create_program: load_function!(library, &["nvrtcCreateProgram"], NvrtcCreateProgram),
            compile_program: load_function!(library, &["nvrtcCompileProgram"], NvrtcCompileProgram),
            get_ptx_size: load_function!(library, &["nvrtcGetPTXSize"], NvrtcGetPtxSize),
            get_ptx: load_function!(library, &["nvrtcGetPTX"], NvrtcGetPtx),
            get_program_log_size: load_function!(
                library,
                &["nvrtcGetProgramLogSize"],
                NvrtcGetProgramLogSize
            ),
            get_program_log: load_function!(library, &["nvrtcGetProgramLog"], NvrtcGetProgramLog),
            destroy_program: load_function!(library, &["nvrtcDestroyProgram"], NvrtcDestroyProgram),
            get_error_string: load_function!(
                library,
                &["nvrtcGetErrorString"],
                NvrtcGetErrorString
            ),
            version: load_function!(library, &["nvrtcVersion"], NvrtcVersion),
            _library: library,
        })
    }

    fn error_string(&self, result: c_int) -> String {
        let pointer = unsafe { (self.get_error_string)(result) };
        if pointer.is_null() {
            format!("NVRTC error code {result}")
        } else {
            unsafe { CStr::from_ptr(pointer).to_string_lossy().into_owned() }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct DeviceBuffer {
    pointer: CuDevicePtr,
    bytes: usize,
}

#[derive(Clone, Copy)]
struct PinnedHostBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

impl Default for PinnedHostBuffer {
    fn default() -> Self {
        Self {
            pointer: ptr::null_mut(),
            bytes: 0,
        }
    }
}

unsafe impl Send for PinnedHostBuffer {}

#[derive(Clone, Copy)]
struct KernelFunctions {
    start_tick: CuFunction,
    deliver_events: CuFunction,
    inject_symbol: CuFunction,
    context_resolve: CuFunction,
    select_blocks: CuFunction,
    select_global: CuFunction,
    cache_surrogate: CuFunction,
    update_recurrent_eligibility: CuFunction,
    update_input_eligibility: CuFunction,
    post_and_emit: CuFunction,
    forward: CuFunction,
    capture_training_step: CuFunction,
    train_persistent: CuFunction,
    train_cooperative: CuFunction,
    train_cooperative_fast: CuFunction,
    train_cooperative_profiled: CuFunction,
    advance_frozen_persistent: CuFunction,
    advance_frozen_cooperative: CuFunction,
    advance_frozen_cooperative_profiled: CuFunction,
    shared_wavefront_pre: CuFunction,
    shared_wavefront_fused: CuFunction,
    shared_wavefront_persistent: CuFunction,
    shared_wavefront_persistent_grouped: CuFunction,
    shared_wavefront_fused_profiled: CuFunction,
    shared_select_blocks: CuFunction,
    shared_post_select: CuFunction,
    shared_cache_surrogate_worklist: CuFunction,
    shared_post_core: CuFunction,
    shared_learning_signals_worklist: CuFunction,
    shared_post_deltas: CuFunction,
    shared_homeostasis_worklist: CuFunction,
    shared_capture_training_step: CuFunction,
    merge_shared_lane_fixed_parameters: CuFunction,
    apply_shared_wavefront_deltas: CuFunction,
    reset_shared_wavefront_deltas: CuFunction,
    learning_signals: CuFunction,
    update_output: CuFunction,
    update_context: CuFunction,
    update_recurrent_weights: CuFunction,
    update_input_weights: CuFunction,
    inhibitory_homeostasis: CuFunction,
    homeostasis: CuFunction,
    reset_last_ticks: CuFunction,
    gather_f32: CuFunction,
    gather_output: CuFunction,
    gather_context: CuFunction,
    scatter_f32: CuFunction,
    scatter_output: CuFunction,
    scatter_context: CuFunction,
    clear_changed_marks: CuFunction,
}

#[derive(Clone, Debug)]
struct CudaHardwareIdentity {
    device_ordinal: usize,
    name: String,
    uuid: String,
    pci_bus_id: String,
    total_memory_bytes: usize,
    driver_version: c_int,
}

fn query_cuda_hardware_identity(
    driver: &DriverFunctions,
    device: CuDevice,
    device_ordinal: usize,
) -> LeoResult<CudaHardwareIdentity> {
    let mut driver_version = 0;
    driver.check(
        unsafe { (driver.driver_get_version)(&mut driver_version) },
        "cuDriverGetVersion",
    )?;

    let mut name_bytes = [0 as c_char; 256];
    driver.check(
        unsafe {
            (driver.device_get_name)(name_bytes.as_mut_ptr(), name_bytes.len() as c_int, device)
        },
        "cuDeviceGetName",
    )?;
    let name = unsafe { CStr::from_ptr(name_bytes.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut pci_bus_bytes = [0 as c_char; 32];
    driver.check(
        unsafe {
            (driver.device_get_pci_bus_id)(
                pci_bus_bytes.as_mut_ptr(),
                pci_bus_bytes.len() as c_int,
                device,
            )
        },
        "cuDeviceGetPCIBusId",
    )?;
    let pci_bus_id = unsafe { CStr::from_ptr(pci_bus_bytes.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut total_memory_bytes = 0usize;
    driver.check(
        unsafe { (driver.device_total_mem)(&mut total_memory_bytes, device) },
        "cuDeviceTotalMem",
    )?;

    let uuid = if let Some(device_get_uuid) = driver.device_get_uuid {
        let mut raw = CuUuid::default();
        let result = unsafe { device_get_uuid(&mut raw, device) };
        if result == CUDA_SUCCESS {
            let mut encoded = String::with_capacity(raw.bytes.len() * 2);
            const HEX: &[u8; 16] = b"0123456789abcdef";
            for byte in raw.bytes {
                encoded.push(HEX[(byte >> 4) as usize] as char);
                encoded.push(HEX[(byte & 0x0f) as usize] as char);
            }
            encoded
        } else {
            // Older drivers may expose the symbol but reject UUID lookup for
            // specific virtualized devices. The rest of the identity remains
            // sufficiently specific and includes the device ordinal.
            "unavailable".to_string()
        }
    } else {
        "unavailable".to_string()
    };

    Ok(CudaHardwareIdentity {
        device_ordinal,
        name,
        uuid,
        pci_bus_id,
        total_memory_bytes,
        driver_version,
    })
}

fn cooperative_grid_capacity(
    driver: &DriverFunctions,
    kernel: CuFunction,
    cooperative_launch: bool,
    multiprocessor_count: c_int,
    threads: c_int,
    label: &str,
) -> LeoResult<u32> {
    if !cooperative_launch || multiprocessor_count <= 0 {
        return Ok(0);
    }
    let mut blocks_per_sm = 0;
    driver.check(
        unsafe {
            (driver.occupancy_max_active_blocks_per_multiprocessor)(
                &mut blocks_per_sm,
                kernel,
                threads,
                0,
            )
        },
        &format!("cuOccupancyMaxActiveBlocksPerMultiprocessor({label})"),
    )?;
    Ok((blocks_per_sm.max(0) as u32).saturating_mul(multiprocessor_count.max(0) as u32))
}

struct SharedCuda {
    driver: Arc<DriverFunctions>,
    context: CuContext,
    module: CuModule,
    kernels: KernelFunctions,
    persistent_grid_blocks: c_uint,
    replay_grid_blocks: c_uint,
    replay_fast_grid_blocks: c_uint,
    replay_profile_grid_blocks: c_uint,
    frozen_grid_blocks: c_uint,
    frozen_profile_grid_blocks: c_uint,
    compute_major: c_int,
    compute_minor: c_int,
    multiprocessor_count: u32,
    max_threads_per_sm: u32,
    cooperative_launch: bool,
    fused_blocks_128: u32,
    fused_blocks_256: u32,
    fused_blocks_512: u32,
    shared_persistent_blocks_256: u32,
    shared_grouped_blocks_256: u32,
    fused_profile_blocks_256: u32,
    hardware: CudaHardwareIdentity,
}

unsafe impl Send for SharedCuda {}
unsafe impl Sync for SharedCuda {}

struct CudaRuntimeEvents {
    upload_ready: Vec<CuEvent>,
    compute_done: CuEvent,
    h2d_start: CuEvent,
    h2d_end: CuEvent,
    compute_start: CuEvent,
    compute_end: CuEvent,
    d2h_start: CuEvent,
    d2h_end: CuEvent,
}

fn create_cuda_event(driver: &DriverFunctions, flags: c_uint, label: &str) -> LeoResult<CuEvent> {
    let mut event = ptr::null_mut();
    driver.check(unsafe { (driver.event_create)(&mut event, flags) }, label)?;
    Ok(event)
}

fn create_runtime_events(driver: &DriverFunctions) -> LeoResult<CudaRuntimeEvents> {
    let mut created = Vec::<CuEvent>::with_capacity(GPU_STORY_BATCH_MAX_LANES + 7);
    let result = (|| -> LeoResult<CudaRuntimeEvents> {
        let mut upload_ready = Vec::with_capacity(GPU_STORY_BATCH_MAX_LANES);
        for _ in 0..GPU_STORY_BATCH_MAX_LANES {
            let event = create_cuda_event(
                driver,
                CU_EVENT_DISABLE_TIMING,
                "cuEventCreate(upload ready)",
            )?;
            created.push(event);
            upload_ready.push(event);
        }
        let compute_done = create_cuda_event(
            driver,
            CU_EVENT_DISABLE_TIMING,
            "cuEventCreate(compute done)",
        )?;
        created.push(compute_done);
        let h2d_start = create_cuda_event(driver, CU_EVENT_DEFAULT, "cuEventCreate(h2d start)")?;
        created.push(h2d_start);
        let h2d_end = create_cuda_event(driver, CU_EVENT_DEFAULT, "cuEventCreate(h2d end)")?;
        created.push(h2d_end);
        let compute_start =
            create_cuda_event(driver, CU_EVENT_DEFAULT, "cuEventCreate(compute start)")?;
        created.push(compute_start);
        let compute_end =
            create_cuda_event(driver, CU_EVENT_DEFAULT, "cuEventCreate(compute end)")?;
        created.push(compute_end);
        let d2h_start = create_cuda_event(driver, CU_EVENT_DEFAULT, "cuEventCreate(d2h start)")?;
        created.push(d2h_start);
        let d2h_end = create_cuda_event(driver, CU_EVENT_DEFAULT, "cuEventCreate(d2h end)")?;
        created.push(d2h_end);
        Ok(CudaRuntimeEvents {
            upload_ready,
            compute_done,
            h2d_start,
            h2d_end,
            compute_start,
            compute_end,
            d2h_start,
            d2h_end,
        })
    })();
    if result.is_err() {
        for event in created {
            unsafe {
                (driver.event_destroy)(event);
            }
        }
    }
    result
}

impl Drop for SharedCuda {
    fn drop(&mut self) {
        if self.context.is_null() {
            return;
        }
        let _ = self.driver.check(
            unsafe { (self.driver.ctx_set_current)(self.context) },
            "cuCtxSetCurrent(shared drop)",
        );
        if !self.module.is_null() {
            unsafe {
                (self.driver.module_unload)(self.module);
            }
        }
        unsafe {
            (self.driver.ctx_destroy)(self.context);
        }
        self.context = ptr::null_mut();
    }
}

#[derive(Default, Clone, Copy)]
struct Buffers {
    config: DeviceBuffer,
    threshold: DeviceBuffer,
    excitability: DeviceBuffer,
    neuron_type: DeviceBuffer,
    recurrent_target: DeviceBuffer,
    recurrent_branch: DeviceBuffer,
    recurrent_delay: DeviceBuffer,
    recurrent_weight: DeviceBuffer,
    input_target: DeviceBuffer,
    input_branch: DeviceBuffer,
    input_weight: DeviceBuffer,
    output_weight: DeviceBuffer,
    output_bias: DeviceBuffer,
    context_keys: DeviceBuffer,
    context_embeddings: DeviceBuffer,
    context_observations: DeviceBuffer,
    context_output_weight: DeviceBuffer,

    membrane: DeviceBuffer,
    activation: DeviceBuffer,
    fatigue: DeviceBuffer,
    refractory_until: DeviceBuffer,
    branches: DeviceBuffer,
    branch_delta: DeviceBuffer,
    branch_last_tick: DeviceBuffer,
    neuron_last_tick: DeviceBuffer,
    adaptation_fast: DeviceBuffer,
    adaptation_medium: DeviceBuffer,
    adaptation_slow: DeviceBuffer,
    touched_epoch: DeviceBuffer,
    touched_list: DeviceBuffer,
    touched_count: DeviceBuffer,
    selected_epoch: DeviceBuffer,
    surrogate: DeviceBuffer,
    candidate_activation: DeviceBuffer,
    active: DeviceBuffer,
    active_value: DeviceBuffer,
    active_count: DeviceBuffer,
    block_winner_neuron: DeviceBuffer,
    block_winner_value: DeviceBuffer,
    block_cutoff: DeviceBuffer,
    population_cutoff: DeviceBuffer,
    population_inhibition: DeviceBuffer,

    recurrent_branch_sensitivity: DeviceBuffer,
    recurrent_membrane_sensitivity: DeviceBuffer,
    recurrent_fatigue_sensitivity: DeviceBuffer,
    recurrent_adaptation_fast_sensitivity: DeviceBuffer,
    recurrent_adaptation_medium_sensitivity: DeviceBuffer,
    recurrent_adaptation_slow_sensitivity: DeviceBuffer,
    recurrent_eligibility: DeviceBuffer,
    recurrent_last_tick: DeviceBuffer,
    recurrent_eligible_mark: DeviceBuffer,
    recurrent_eligible_list: DeviceBuffer,
    recurrent_eligible_count: DeviceBuffer,
    recurrent_next_eligible_list: DeviceBuffer,
    recurrent_next_eligible_count: DeviceBuffer,
    input_branch_sensitivity: DeviceBuffer,
    input_membrane_sensitivity: DeviceBuffer,
    input_fatigue_sensitivity: DeviceBuffer,
    input_adaptation_fast_sensitivity: DeviceBuffer,
    input_adaptation_medium_sensitivity: DeviceBuffer,
    input_adaptation_slow_sensitivity: DeviceBuffer,
    input_eligibility: DeviceBuffer,
    input_last_tick: DeviceBuffer,
    input_eligible_mark: DeviceBuffer,
    input_eligible_list: DeviceBuffer,
    input_eligible_count: DeviceBuffer,
    input_next_eligible_list: DeviceBuffer,
    input_next_eligible_count: DeviceBuffer,

    ring_count: DeviceBuffer,
    ring_source: DeviceBuffer,
    ring_activation: DeviceBuffer,
    ring_weight: DeviceBuffer,

    context_history: DeviceBuffer,
    context_history_count: DeviceBuffer,
    active_context_slots: DeviceBuffer,
    active_context_scales: DeviceBuffer,
    active_context_count: DeviceBuffer,
    context_latent: DeviceBuffer,
    context_gradient: DeviceBuffer,

    neural_logits: DeviceBuffer,
    logits: DeviceBuffer,
    probabilities: DeviceBuffer,
    errors: DeviceBuffer,
    learning_destination_epoch: DeviceBuffer,
    learning_destination_list: DeviceBuffer,
    learning_destination_count: DeviceBuffer,
    learning_signal: DeviceBuffer,
    counters: DeviceBuffer,
    error_flag: DeviceBuffer,
    training_step_records: DeviceBuffer,
    persistent_steps: DeviceBuffer,
    persistent_pointer_table: DeviceBuffer,
    batch_pointer_tables: DeviceBuffer,
    batch_step_buffers: DeviceBuffer,
    batch_step_counts: DeviceBuffer,
    batch_base_ticks: DeviceBuffer,
    batch_delta_pointer_table: DeviceBuffer,
    phase_profile_counters: DeviceBuffer,
    replay_profile_counters: DeviceBuffer,

    // Stage 3 shared-model synchronous mini-batch accumulators. These are
    // canonical-runtime buffers, not per-story state. Story blocks accumulate
    // constrained worker deltas here and a second kernel applies the mean
    // update before the next byte wavefront.
    batch_delta_threshold: DeviceBuffer,
    batch_delta_threshold_marks: DeviceBuffer,
    batch_delta_threshold_list: DeviceBuffer,
    batch_delta_threshold_count: DeviceBuffer,
    batch_delta_recurrent: DeviceBuffer,
    batch_delta_recurrent_marks: DeviceBuffer,
    batch_delta_recurrent_list: DeviceBuffer,
    batch_delta_recurrent_count: DeviceBuffer,
    batch_delta_input: DeviceBuffer,
    batch_delta_input_marks: DeviceBuffer,
    batch_delta_input_list: DeviceBuffer,
    batch_delta_input_count: DeviceBuffer,
    batch_delta_output: DeviceBuffer,
    batch_delta_output_marks: DeviceBuffer,
    batch_delta_output_list: DeviceBuffer,
    batch_delta_output_count: DeviceBuffer,
    batch_delta_output_bias: DeviceBuffer,
    batch_delta_context_embedding: DeviceBuffer,
    batch_delta_context_marks: DeviceBuffer,
    batch_delta_context_list: DeviceBuffer,
    batch_delta_context_count: DeviceBuffer,
    batch_delta_context_observations: DeviceBuffer,
    batch_delta_context_output: DeviceBuffer,

    changed_threshold_marks: DeviceBuffer,
    changed_threshold_list: DeviceBuffer,
    changed_threshold_count: DeviceBuffer,
    changed_recurrent_marks: DeviceBuffer,
    changed_recurrent_list: DeviceBuffer,
    changed_recurrent_count: DeviceBuffer,
    changed_input_marks: DeviceBuffer,
    changed_input_list: DeviceBuffer,
    changed_input_count: DeviceBuffer,
    changed_output_marks: DeviceBuffer,
    changed_output_list: DeviceBuffer,
    changed_output_count: DeviceBuffer,
    changed_context_marks: DeviceBuffer,
    changed_context_list: DeviceBuffer,
    changed_context_count: DeviceBuffer,

    gather_threshold: DeviceBuffer,
    gather_recurrent: DeviceBuffer,
    gather_input: DeviceBuffer,
    gather_output: DeviceBuffer,
    gather_context_keys: DeviceBuffer,
    gather_context_observations: DeviceBuffer,
    gather_context_embeddings: DeviceBuffer,
}

impl Buffers {
    fn all(&self) -> Vec<DeviceBuffer> {
        vec![
            self.config,
            self.threshold,
            self.excitability,
            self.neuron_type,
            self.recurrent_target,
            self.recurrent_branch,
            self.recurrent_delay,
            self.recurrent_weight,
            self.input_target,
            self.input_branch,
            self.input_weight,
            self.output_weight,
            self.output_bias,
            self.context_keys,
            self.context_embeddings,
            self.context_observations,
            self.context_output_weight,
            self.membrane,
            self.activation,
            self.fatigue,
            self.refractory_until,
            self.branches,
            self.branch_delta,
            self.branch_last_tick,
            self.neuron_last_tick,
            self.adaptation_fast,
            self.adaptation_medium,
            self.adaptation_slow,
            self.touched_epoch,
            self.touched_list,
            self.touched_count,
            self.selected_epoch,
            self.surrogate,
            self.candidate_activation,
            self.active,
            self.active_value,
            self.active_count,
            self.block_winner_neuron,
            self.block_winner_value,
            self.block_cutoff,
            self.population_cutoff,
            self.population_inhibition,
            self.recurrent_branch_sensitivity,
            self.recurrent_membrane_sensitivity,
            self.recurrent_fatigue_sensitivity,
            self.recurrent_adaptation_fast_sensitivity,
            self.recurrent_adaptation_medium_sensitivity,
            self.recurrent_adaptation_slow_sensitivity,
            self.recurrent_eligibility,
            self.recurrent_last_tick,
            self.recurrent_eligible_mark,
            self.recurrent_eligible_list,
            self.recurrent_eligible_count,
            self.recurrent_next_eligible_list,
            self.recurrent_next_eligible_count,
            self.input_branch_sensitivity,
            self.input_membrane_sensitivity,
            self.input_fatigue_sensitivity,
            self.input_adaptation_fast_sensitivity,
            self.input_adaptation_medium_sensitivity,
            self.input_adaptation_slow_sensitivity,
            self.input_eligibility,
            self.input_last_tick,
            self.input_eligible_mark,
            self.input_eligible_list,
            self.input_eligible_count,
            self.input_next_eligible_list,
            self.input_next_eligible_count,
            self.ring_count,
            self.ring_source,
            self.ring_activation,
            self.ring_weight,
            self.context_history,
            self.context_history_count,
            self.active_context_slots,
            self.active_context_scales,
            self.active_context_count,
            self.context_latent,
            self.context_gradient,
            self.neural_logits,
            self.logits,
            self.probabilities,
            self.errors,
            self.learning_destination_epoch,
            self.learning_destination_list,
            self.learning_destination_count,
            self.learning_signal,
            self.counters,
            self.error_flag,
            self.training_step_records,
            self.persistent_steps,
            self.persistent_pointer_table,
            self.batch_pointer_tables,
            self.batch_step_buffers,
            self.batch_step_counts,
            self.batch_base_ticks,
            self.batch_delta_pointer_table,
            self.phase_profile_counters,
            self.replay_profile_counters,
            self.batch_delta_threshold,
            self.batch_delta_threshold_marks,
            self.batch_delta_threshold_list,
            self.batch_delta_threshold_count,
            self.batch_delta_recurrent,
            self.batch_delta_recurrent_marks,
            self.batch_delta_recurrent_list,
            self.batch_delta_recurrent_count,
            self.batch_delta_input,
            self.batch_delta_input_marks,
            self.batch_delta_input_list,
            self.batch_delta_input_count,
            self.batch_delta_output,
            self.batch_delta_output_marks,
            self.batch_delta_output_list,
            self.batch_delta_output_count,
            self.batch_delta_output_bias,
            self.batch_delta_context_embedding,
            self.batch_delta_context_marks,
            self.batch_delta_context_list,
            self.batch_delta_context_count,
            self.batch_delta_context_observations,
            self.batch_delta_context_output,
            self.changed_threshold_marks,
            self.changed_threshold_list,
            self.changed_threshold_count,
            self.changed_recurrent_marks,
            self.changed_recurrent_list,
            self.changed_recurrent_count,
            self.changed_input_marks,
            self.changed_input_list,
            self.changed_input_count,
            self.changed_output_marks,
            self.changed_output_list,
            self.changed_output_count,
            self.changed_context_marks,
            self.changed_context_list,
            self.changed_context_count,
            self.gather_threshold,
            self.gather_recurrent,
            self.gather_input,
            self.gather_output,
            self.gather_context_keys,
            self.gather_context_observations,
            self.gather_context_embeddings,
        ]
    }
}

pub(crate) struct CudaRuntime {
    model: Model,
    shared: Arc<SharedCuda>,
    compute_stream: CuStream,
    transfer_stream: CuStream,
    events: CudaRuntimeEvents,
    buffers: Buffers,
    current_tick: u64,
    probabilities: Vec<f32>,
    neural_logits: Vec<f32>,
    active: Vec<usize>,
    activation: Vec<f32>,
    model_dirty: bool,
    persistent_document_activity: bool,
    track_parameter_changes: bool,
    accumulated_changes: ParameterChanges,
    accumulated_threshold_mark: Vec<bool>,
    accumulated_recurrent_mark: Vec<bool>,
    accumulated_input_mark: Vec<bool>,
    accumulated_output_mark: Vec<bool>,
    accumulated_context_mark: Vec<bool>,
    output_bias_dirty_since_sync: bool,
    context_output_dirty_since_sync: bool,
    execution_tuner: CudaExecutionTuner,
    host_batch_pointer_tables: PinnedHostBuffer,
    host_batch_step_buffers: PinnedHostBuffer,
    host_batch_step_counts: PinnedHostBuffer,
    host_batch_base_ticks: PinnedHostBuffer,
    host_batch_change_counts: PinnedHostBuffer,
    batch_snapshot_device_f32: DeviceBuffer,
    batch_snapshot_device_u32: DeviceBuffer,
    batch_snapshot_device_u64: DeviceBuffer,
    batch_snapshot_host_f32: PinnedHostBuffer,
    batch_snapshot_host_u32: PinnedHostBuffer,
    batch_snapshot_host_u64: PinnedHostBuffer,
    sparse_apply_graph: Option<CuGraphExec>,
    sparse_apply_graph_plan: Option<(u32, u32)>,
    sparse_apply_graph_disabled: bool,
    phase_profile_sample_stride: Option<u64>,
    replay_profile: CudaReplayProfileOptions,
    replay_profile_target_sequence: u64,
    replay_profile_prefix_sequence: u64,
    replay_cooperative_enabled: bool,
    shared_persistent_enabled: bool,
    full_step_metrics: bool,
    replay_grid_override: Option<u32>,
    frozen_grid_override: Option<u32>,
    debug: CudaDebugOptions,
}

unsafe impl Send for CudaRuntime {}

/// Exact logical-story lane for the shared CUDA wavefront executor.
///
/// Immutable topology/configuration tensors stay owned by the canonical
/// `CudaRuntime`. Each lane owns the mutable learned parameters plus its
/// recurrent/eligibility/context-history state so one story can never observe
/// another story's updates before the canonical batch-end mean barrier.
/// This is intentionally much lighter than cloning a complete CUDA runtime per
/// worker: static topology, launch infrastructure, scratch planning, streams,
/// modules, and the canonical model remain shared.
pub(crate) struct CudaBatchLane {
    shared: Arc<SharedCuda>,
    buffers: Buffers,
    current_tick: u64,
    /// Canonical parameter revision currently mirrored by the lane before story-local updates.
    model_revision: u64,
    host_pointer_table: PinnedHostBuffer,
    host_steps: PinnedHostBuffer,
    host_records: PinnedHostBuffer,
}

unsafe impl Send for CudaBatchLane {}

pub(crate) struct CudaStoryBatchReport {
    pub(crate) story_metrics: Vec<Vec<StepMetrics>>,
    pub(crate) merge: MergeMetrics,
    pub(crate) story_deltas: Vec<SparseModelDelta>,
    pub(crate) story_changes: Vec<ParameterChanges>,
}

#[derive(Clone, Copy, Debug, Default)]
struct SnapshotSpan {
    offset: usize,
    len: usize,
}

#[derive(Clone, Debug, Default)]
struct BatchLaneSnapshotLayout {
    threshold_indices: SnapshotSpan,
    recurrent_indices: SnapshotSpan,
    input_indices: SnapshotSpan,
    output_indices: SnapshotSpan,
    context_indices: SnapshotSpan,
    context_observations: SnapshotSpan,
    context_keys: SnapshotSpan,
    threshold_values: SnapshotSpan,
    recurrent_values: SnapshotSpan,
    input_values: SnapshotSpan,
    output_values: SnapshotSpan,
    context_embeddings: SnapshotSpan,
    output_bias: Option<SnapshotSpan>,
    context_output: Option<SnapshotSpan>,
}

#[derive(Clone, Debug, Default)]
struct BatchLaneContextSnapshotLayout {
    context_indices: SnapshotSpan,
    context_observations: SnapshotSpan,
    context_keys: SnapshotSpan,
    context_embeddings: SnapshotSpan,
}

impl CudaBatchLane {
    fn owned_buffers(&self) -> Vec<DeviceBuffer> {
        let b = self.buffers;
        vec![
            b.threshold,
            b.recurrent_weight,
            b.input_weight,
            b.output_weight,
            b.output_bias,
            b.context_keys,
            b.context_embeddings,
            b.context_observations,
            b.context_output_weight,
            b.changed_threshold_marks,
            b.changed_threshold_list,
            b.changed_threshold_count,
            b.changed_recurrent_marks,
            b.changed_recurrent_list,
            b.changed_recurrent_count,
            b.changed_input_marks,
            b.changed_input_list,
            b.changed_input_count,
            b.changed_output_marks,
            b.changed_output_list,
            b.changed_output_count,
            b.changed_context_marks,
            b.changed_context_list,
            b.changed_context_count,
            b.membrane,
            b.activation,
            b.fatigue,
            b.refractory_until,
            b.branches,
            b.branch_delta,
            b.branch_last_tick,
            b.neuron_last_tick,
            b.adaptation_fast,
            b.adaptation_medium,
            b.adaptation_slow,
            b.touched_epoch,
            b.touched_list,
            b.touched_count,
            b.selected_epoch,
            b.surrogate,
            b.candidate_activation,
            b.active,
            b.active_value,
            b.active_count,
            b.block_winner_neuron,
            b.block_winner_value,
            b.block_cutoff,
            b.population_cutoff,
            b.population_inhibition,
            b.recurrent_branch_sensitivity,
            b.recurrent_membrane_sensitivity,
            b.recurrent_fatigue_sensitivity,
            b.recurrent_adaptation_fast_sensitivity,
            b.recurrent_adaptation_medium_sensitivity,
            b.recurrent_adaptation_slow_sensitivity,
            b.recurrent_eligibility,
            b.recurrent_last_tick,
            b.recurrent_eligible_mark,
            b.recurrent_eligible_list,
            b.recurrent_eligible_count,
            b.recurrent_next_eligible_list,
            b.recurrent_next_eligible_count,
            b.input_branch_sensitivity,
            b.input_membrane_sensitivity,
            b.input_fatigue_sensitivity,
            b.input_adaptation_fast_sensitivity,
            b.input_adaptation_medium_sensitivity,
            b.input_adaptation_slow_sensitivity,
            b.input_eligibility,
            b.input_last_tick,
            b.input_eligible_mark,
            b.input_eligible_list,
            b.input_eligible_count,
            b.input_next_eligible_list,
            b.input_next_eligible_count,
            b.ring_count,
            b.ring_source,
            b.ring_activation,
            b.ring_weight,
            b.context_history,
            b.context_history_count,
            b.active_context_slots,
            b.active_context_scales,
            b.active_context_count,
            b.context_latent,
            b.context_gradient,
            b.neural_logits,
            b.logits,
            b.probabilities,
            b.errors,
            b.learning_destination_epoch,
            b.learning_destination_list,
            b.learning_destination_count,
            b.learning_signal,
            b.counters,
            b.error_flag,
            b.training_step_records,
            b.persistent_steps,
            b.persistent_pointer_table,
        ]
    }
}

impl Drop for CudaBatchLane {
    fn drop(&mut self) {
        let _ = self.shared.driver.check(
            unsafe { (self.shared.driver.ctx_set_current)(self.shared.context) },
            "cuCtxSetCurrent(batch lane drop)",
        );
        for buffer in self.owned_buffers() {
            if buffer.pointer != 0 {
                unsafe {
                    (self.shared.driver.mem_free)(buffer.pointer);
                }
            }
        }
        for buffer in [self.host_pointer_table, self.host_steps, self.host_records] {
            if !buffer.pointer.is_null() {
                unsafe {
                    (self.shared.driver.mem_free_host)(buffer.pointer);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaDeviceCompatibility {
    pub(crate) ordinal: usize,
    pub(crate) name: String,
    pub(crate) compute_major: i32,
    pub(crate) compute_minor: i32,
}

impl CudaRuntime {
    pub(crate) fn visible_device_compatibility() -> LeoResult<Vec<CudaDeviceCompatibility>> {
        let driver = DriverFunctions::load()?;
        driver.check(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut count = 0;
        driver.check(
            unsafe { (driver.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if count <= 0 {
            return Err(LeoError::cuda("CUDA driver reported no GPU devices"));
        }

        let mut profiles = Vec::with_capacity(count as usize);
        for ordinal in 0..count as usize {
            let mut device = 0;
            driver.check(
                unsafe { (driver.device_get)(&mut device, ordinal as c_int) },
                "cuDeviceGet",
            )?;
            let mut compute_major = 0;
            let mut compute_minor = 0;
            driver.check(
                unsafe {
                    (driver.device_get_attribute)(
                        &mut compute_major,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        device,
                    )
                },
                "cuDeviceGetAttribute(compute major)",
            )?;
            driver.check(
                unsafe {
                    (driver.device_get_attribute)(
                        &mut compute_minor,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        device,
                    )
                },
                "cuDeviceGetAttribute(compute minor)",
            )?;
            let hardware = query_cuda_hardware_identity(&driver, device, ordinal)?;
            profiles.push(CudaDeviceCompatibility {
                ordinal,
                name: hardware.name,
                compute_major,
                compute_minor,
            });
        }
        Ok(profiles)
    }

    pub(crate) fn probe() -> LeoResult<usize> {
        let driver = DriverFunctions::load()?;
        let _nvrtc = NvrtcFunctions::load()?;
        driver.check(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut count = 0;
        driver.check(
            unsafe { (driver.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if count <= 0 {
            return Err(LeoError::cuda("CUDA driver reported no GPU devices"));
        }
        Ok(count as usize)
    }

    pub(crate) fn new(model: Model) -> LeoResult<Self> {
        let requested_device = env::var("LEO_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        Self::new_on_device(model, requested_device)
    }

    pub(crate) fn new_on_device(model: Model, requested_device: usize) -> LeoResult<Self> {
        model.validate()?;
        validate_gpu_shape(&model)?;
        let driver = DriverFunctions::load()?;
        driver.check(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut device_count = 0;
        driver.check(
            unsafe { (driver.device_get_count)(&mut device_count) },
            "cuDeviceGetCount",
        )?;
        if device_count <= 0 {
            return Err(LeoError::cuda("CUDA driver reported no GPU devices"));
        }
        if requested_device >= device_count as usize {
            return Err(LeoError::cuda(format!(
                "CUDA device {requested_device} is outside available CUDA devices 0..{}",
                device_count - 1
            )));
        }
        let mut device = 0;
        driver.check(
            unsafe { (driver.device_get)(&mut device, requested_device as c_int) },
            "cuDeviceGet",
        )?;
        let mut compute_major = 0;
        let mut compute_minor = 0;
        driver.check(
            unsafe {
                (driver.device_get_attribute)(
                    &mut compute_major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(compute major)",
        )?;
        driver.check(
            unsafe {
                (driver.device_get_attribute)(
                    &mut compute_minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(compute minor)",
        )?;
        let mut multiprocessor_count = 0;
        driver.check(
            unsafe {
                (driver.device_get_attribute)(
                    &mut multiprocessor_count,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    device,
                )
            },
            "cuDeviceGetAttribute(multiprocessor count)",
        )?;
        let mut cooperative_launch = 0;
        driver.check(
            unsafe {
                (driver.device_get_attribute)(
                    &mut cooperative_launch,
                    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
                    device,
                )
            },
            "cuDeviceGetAttribute(cooperative launch)",
        )?;
        let mut max_threads_per_sm = 0;
        driver.check(
            unsafe {
                (driver.device_get_attribute)(
                    &mut max_threads_per_sm,
                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(max threads per multiprocessor)",
        )?;
        let hardware = query_cuda_hardware_identity(&driver, device, requested_device)?;
        let mut context = ptr::null_mut();
        driver.check(
            unsafe { (driver.ctx_create)(&mut context, 0, device) },
            "cuCtxCreate",
        )?;
        driver.check(
            unsafe { (driver.ctx_set_current)(context) },
            "cuCtxSetCurrent",
        )?;
        let ptx = compile_cuda_kernels(compute_major, compute_minor)?;
        let mut module = ptr::null_mut();
        driver.check(
            unsafe { (driver.module_load_data)(&mut module, ptx.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        let kernels = load_kernels(&driver, module)?;
        let fused_blocks_128 = cooperative_grid_capacity(
            &driver,
            kernels.shared_wavefront_fused,
            cooperative_launch != 0,
            multiprocessor_count,
            128,
            "fused wavefront 128",
        )?;
        let fused_blocks_256 = cooperative_grid_capacity(
            &driver,
            kernels.shared_wavefront_fused,
            cooperative_launch != 0,
            multiprocessor_count,
            256,
            "fused wavefront 256",
        )?;
        let fused_blocks_512 = cooperative_grid_capacity(
            &driver,
            kernels.shared_wavefront_fused,
            cooperative_launch != 0,
            multiprocessor_count,
            512,
            "fused wavefront 512",
        )?;
        let shared_persistent_blocks_256 = cooperative_grid_capacity(
            &driver,
            kernels.shared_wavefront_persistent,
            cooperative_launch != 0,
            multiprocessor_count,
            256,
            "persistent shared wavefront 256",
        )?;
        let shared_grouped_blocks_256 = cooperative_grid_capacity(
            &driver,
            kernels.shared_wavefront_persistent_grouped,
            cooperative_launch != 0,
            multiprocessor_count,
            256,
            "grouped persistent shared wavefront 256",
        )?;
        let fused_profile_blocks_256 = cooperative_grid_capacity(
            &driver,
            kernels.shared_wavefront_fused_profiled,
            cooperative_launch != 0,
            multiprocessor_count,
            256,
            "profiled fused wavefront 256",
        )?;
        let frozen_grid_blocks = cooperative_grid_capacity(
            &driver,
            kernels.advance_frozen_cooperative,
            cooperative_launch != 0,
            multiprocessor_count,
            PERSISTENT_THREADS as c_int,
            "cooperative frozen replay prefix",
        )?;
        let frozen_profile_grid_blocks = cooperative_grid_capacity(
            &driver,
            kernels.advance_frozen_cooperative_profiled,
            cooperative_launch != 0,
            multiprocessor_count,
            PERSISTENT_THREADS as c_int,
            "profiled cooperative frozen replay prefix",
        )?;
        let replay_grid_blocks = cooperative_grid_capacity(
            &driver,
            kernels.train_cooperative,
            cooperative_launch != 0,
            multiprocessor_count,
            PERSISTENT_THREADS as c_int,
            "cooperative replay trainer",
        )?;
        let replay_fast_grid_blocks = cooperative_grid_capacity(
            &driver,
            kernels.train_cooperative_fast,
            cooperative_launch != 0,
            multiprocessor_count,
            PERSISTENT_THREADS as c_int,
            "cooperative fast replay trainer",
        )?;
        let replay_profile_grid_blocks = cooperative_grid_capacity(
            &driver,
            kernels.train_cooperative_profiled,
            cooperative_launch != 0,
            multiprocessor_count,
            PERSISTENT_THREADS as c_int,
            "profiled cooperative replay trainer",
        )?;
        let persistent_grid_blocks = if cooperative_launch != 0 && multiprocessor_count > 0 {
            let mut active_blocks_per_sm = 0;
            driver.check(
                unsafe {
                    (driver.occupancy_max_active_blocks_per_multiprocessor)(
                        &mut active_blocks_per_sm,
                        kernels.train_persistent,
                        PERSISTENT_THREADS as c_int,
                        0,
                    )
                },
                "cuOccupancyMaxActiveBlocksPerMultiprocessor(persistent trainer)",
            )?;
            if active_blocks_per_sm > 0 {
                1
            } else {
                0
            }
        } else {
            0
        };
        let shared = Arc::new(SharedCuda {
            driver,
            context,
            module,
            kernels,
            persistent_grid_blocks,
            replay_grid_blocks,
            replay_fast_grid_blocks,
            replay_profile_grid_blocks,
            frozen_grid_blocks,
            frozen_profile_grid_blocks,
            compute_major,
            compute_minor,
            multiprocessor_count: multiprocessor_count.max(1) as u32,
            max_threads_per_sm: max_threads_per_sm.max(1) as u32,
            cooperative_launch: cooperative_launch != 0,
            fused_blocks_128,
            fused_blocks_256,
            fused_blocks_512,
            shared_persistent_blocks_256,
            shared_grouped_blocks_256,
            fused_profile_blocks_256,
            hardware,
        });
        let mut runtime = Self::with_shared(model, shared)?;
        runtime.upload_full_model()?;
        runtime.reset_transient_state()?;
        runtime.shared.driver.check(
            unsafe { (runtime.shared.driver.ctx_set_current)(ptr::null_mut()) },
            "cuCtxSetCurrent(detach)",
        )?;
        Ok(runtime)
    }

    fn with_shared(model: Model, shared: Arc<SharedCuda>) -> LeoResult<Self> {
        validate_gpu_shape(&model)?;
        let mut compute_stream = ptr::null_mut();
        shared.driver.check(
            unsafe { (shared.driver.stream_create)(&mut compute_stream, 0) },
            "cuStreamCreate(compute)",
        )?;
        let mut transfer_stream = ptr::null_mut();
        if let Err(error) = shared.driver.check(
            unsafe { (shared.driver.stream_create)(&mut transfer_stream, 0) },
            "cuStreamCreate(transfer)",
        ) {
            unsafe {
                (shared.driver.stream_destroy)(compute_stream);
            }
            return Err(error);
        }
        let events = match create_runtime_events(&shared.driver) {
            Ok(events) => events,
            Err(error) => {
                unsafe {
                    (shared.driver.stream_destroy)(transfer_stream);
                    (shared.driver.stream_destroy)(compute_stream);
                }
                return Err(error);
            }
        };
        let neuron_count = model.neuron_count();
        let recurrent_count = model.recurrent.weight.len();
        let input_count = model.input.weights.len();
        let context_count = model.context.keys.len();
        let execution_profile_key = cuda_execution_profile_key(&model, &shared);
        let execution_tuner = CudaExecutionTuner::new(
            execution_profile_key,
            CudaTuningLimits {
                multiprocessors: shared.multiprocessor_count,
                max_threads_per_sm: shared.max_threads_per_sm,
                max_lanes: GPU_STORY_BATCH_MAX_LANES,
                fused_blocks_128: shared.fused_blocks_128,
                fused_blocks_256: shared.fused_blocks_256,
                fused_blocks_512: shared.fused_blocks_512,
            },
        );
        let phase_profile_sample_stride = cuda_phase_profile_sample_stride();
        let replay_profile = CudaReplayProfileOptions::from_env();
        let replay_cooperative_enabled = cuda_replay_cooperative_enabled();
        let shared_persistent_enabled = cuda_shared_persistent_enabled();
        let full_step_metrics = cuda_full_step_metrics_enabled();
        let replay_grid_override = env_positive_u32_opt("LEO_CUDA_REPLAY_BLOCKS");
        let frozen_grid_override = env_positive_u32_opt("LEO_CUDA_FROZEN_BLOCKS");
        let debug = CudaDebugOptions::from_env();
        let mut runtime = Self {
            model,
            shared,
            compute_stream,
            transfer_stream,
            events,
            buffers: Buffers::default(),
            current_tick: 0,
            probabilities: vec![0.0; OUTPUT_CLASSES],
            neural_logits: vec![0.0; OUTPUT_CLASSES],
            active: Vec::new(),
            activation: vec![0.0; neuron_count],
            model_dirty: true,
            persistent_document_activity: false,
            track_parameter_changes: false,
            accumulated_changes: ParameterChanges::default(),
            accumulated_threshold_mark: vec![false; neuron_count],
            accumulated_recurrent_mark: vec![false; recurrent_count],
            accumulated_input_mark: vec![false; input_count],
            accumulated_output_mark: vec![false; neuron_count],
            accumulated_context_mark: vec![false; context_count],
            output_bias_dirty_since_sync: false,
            context_output_dirty_since_sync: false,
            execution_tuner,
            host_batch_pointer_tables: PinnedHostBuffer::default(),
            host_batch_step_buffers: PinnedHostBuffer::default(),
            host_batch_step_counts: PinnedHostBuffer::default(),
            host_batch_base_ticks: PinnedHostBuffer::default(),
            host_batch_change_counts: PinnedHostBuffer::default(),
            batch_snapshot_device_f32: DeviceBuffer::default(),
            batch_snapshot_device_u32: DeviceBuffer::default(),
            batch_snapshot_device_u64: DeviceBuffer::default(),
            batch_snapshot_host_f32: PinnedHostBuffer::default(),
            batch_snapshot_host_u32: PinnedHostBuffer::default(),
            batch_snapshot_host_u64: PinnedHostBuffer::default(),
            sparse_apply_graph: None,
            sparse_apply_graph_plan: None,
            sparse_apply_graph_disabled: false,
            phase_profile_sample_stride,
            replay_profile,
            replay_profile_target_sequence: 0,
            replay_profile_prefix_sequence: 0,
            replay_cooperative_enabled,
            shared_persistent_enabled,
            full_step_metrics,
            replay_grid_override,
            frozen_grid_override,
            debug,
        };
        runtime.make_current()?;
        runtime.allocate_buffers()?;
        runtime.initialize_static_pointer_tables()?;
        if runtime.debug.summary {
            eprintln!(
                "{{\"event\":\"cuda_debug_runtime\",\"device\":{},\"name\":\"{}\",\"uuid\":\"{}\",\"pci_bus_id\":\"{}\",\"driver_version\":{},\"cc\":\"{}.{}\",\"sm_count\":{},\"total_memory_bytes\":{},\"cooperative_launch\":{},\"replay_cooperative_enabled\":{},\"legacy_replay_grid\":{},\"replay_grid_capacity\":{},\"replay_profile_grid_capacity\":{},\"frozen_grid_capacity\":{},\"frozen_profile_grid_capacity\":{},\"replay_profile_target_stride\":{},\"replay_profile_prefix_stride\":{},\"replay_grid_override\":{},\"frozen_grid_override\":{},\"separate_transfer_stream\":true}}",
                runtime.shared.hardware.device_ordinal,
                runtime.shared.hardware.name.replace('\\', "\\\\").replace('\"', "\\\""),
                runtime.shared.hardware.uuid,
                runtime.shared.hardware.pci_bus_id,
                runtime.shared.hardware.driver_version,
                runtime.shared.compute_major,
                runtime.shared.compute_minor,
                runtime.shared.multiprocessor_count,
                runtime.shared.hardware.total_memory_bytes,
                runtime.shared.cooperative_launch,
                runtime.replay_cooperative_enabled,
                runtime.shared.persistent_grid_blocks,
                runtime.shared.replay_grid_blocks,
                runtime.shared.replay_profile_grid_blocks,
                runtime.shared.frozen_grid_blocks,
                runtime.shared.frozen_profile_grid_blocks,
                runtime.replay_profile.target_stride.unwrap_or(0),
                runtime.replay_profile.prefix_stride.unwrap_or(0),
                runtime.replay_grid_override.unwrap_or(0),
                runtime.frozen_grid_override.unwrap_or(0),
            );
        }
        if runtime.debug.memory {
            let allocated_bytes = runtime
                .buffers
                .all()
                .iter()
                .map(|buffer| buffer.bytes as u64)
                .sum::<u64>();
            eprintln!(
                "{{\"event\":\"cuda_debug_memory\",\"scope\":\"runtime\",\"allocated_device_bytes\":{},\"device_total_bytes\":{},\"allocation_fraction\":{}}}",
                allocated_bytes,
                runtime.shared.hardware.total_memory_bytes,
                allocated_bytes as f64 / runtime.shared.hardware.total_memory_bytes.max(1) as f64,
            );
        }
        Ok(runtime)
    }

    pub(crate) fn fork_for_model(&self, model: Model) -> LeoResult<Self> {
        self.make_current()?;
        let mut fork = Self::with_shared(model, Arc::clone(&self.shared))?;
        fork.upload_full_model()?;
        fork.reset_transient_state()?;
        fork.shared.driver.check(
            unsafe { (fork.shared.driver.ctx_set_current)(ptr::null_mut()) },
            "cuCtxSetCurrent(detach fork)",
        )?;
        Ok(fork)
    }

    // Sequential fallible allocation keeps partial-resource cleanup explicit and auditable.
    #[allow(clippy::field_reassign_with_default)]
    pub(crate) fn allocate_shared_batch_lane(&mut self) -> LeoResult<CudaBatchLane> {
        self.make_current()?;
        let n = self.model.neuron_count();
        let recurrent = self.model.recurrent.weight.len();
        let input = self.model.input.weights.len();
        let active = self.model.config.model.max_active_global;
        let block_winners = self
            .model
            .config
            .model
            .block_count
            .saturating_mul(self.model.config.model.max_active_per_block);
        let context_dim = self.model.config.context.embedding_dim;
        let max_order = self.model.config.context.max_order;
        let context_slots = max_order.saturating_mul(self.model.config.context.slots_per_order);
        let ring_records = RING_BUCKETS.saturating_mul(active);
        let ring_weights = ring_records.saturating_mul(self.model.recurrent.capacity_per_neuron);

        // Construct the owner before the first fallible allocation. If any
        // device or pinned-host allocation fails, `CudaBatchLane::drop` frees
        // every buffer acquired so far instead of leaking a partial lane.
        let mut lane = CudaBatchLane {
            shared: Arc::clone(&self.shared),
            buffers: Buffers::default(),
            current_tick: 0,
            model_revision: self.model.parameter_revision,
            host_pointer_table: PinnedHostBuffer::default(),
            host_steps: PinnedHostBuffer::default(),
            host_records: PinnedHostBuffer::default(),
        };
        let b = &mut lane.buffers;

        // Exact logical-batch semantics require every story to evolve its own
        // learned parameter image until the one canonical batch-end mean. Keep
        // topology/configuration shared; duplicate only mutable learned state.
        b.threshold = self.allocate_f32(n)?;
        b.recurrent_weight = self.allocate_f32(recurrent)?;
        b.input_weight = self.allocate_f32(input)?;
        b.output_weight = self.allocate_f32(n.saturating_mul(OUTPUT_CLASSES))?;
        b.output_bias = self.allocate_f32(OUTPUT_CLASSES)?;
        b.context_keys = self.allocate_u64(context_slots)?;
        b.context_embeddings = self.allocate_f32(context_slots.saturating_mul(context_dim))?;
        b.context_observations = self.allocate_u32(context_slots)?;
        b.context_output_weight = self.allocate_f32(OUTPUT_CLASSES.saturating_mul(context_dim))?;

        b.changed_threshold_marks = self.allocate_u32(n)?;
        b.changed_threshold_list = self.allocate_u32(n)?;
        b.changed_threshold_count = self.allocate_u32(1)?;
        b.changed_recurrent_marks = self.allocate_u32(recurrent)?;
        b.changed_recurrent_list = self.allocate_u32(recurrent)?;
        b.changed_recurrent_count = self.allocate_u32(1)?;
        b.changed_input_marks = self.allocate_u32(input)?;
        b.changed_input_list = self.allocate_u32(input)?;
        b.changed_input_count = self.allocate_u32(1)?;
        b.changed_output_marks = self.allocate_u32(n)?;
        b.changed_output_list = self.allocate_u32(n)?;
        b.changed_output_count = self.allocate_u32(1)?;
        b.changed_context_marks = self.allocate_u32(context_slots)?;
        b.changed_context_list = self.allocate_u32(context_slots)?;
        b.changed_context_count = self.allocate_u32(1)?;

        b.membrane = self.allocate_f32(n)?;
        b.activation = self.allocate_f32(n)?;
        b.fatigue = self.allocate_f32(n)?;
        b.refractory_until = self.allocate_u64(n)?;
        b.branches = self.allocate_f32(n.saturating_mul(4))?;
        b.branch_delta = self.allocate_f32(n.saturating_mul(4))?;
        b.branch_last_tick = self.allocate_u64(n)?;
        b.neuron_last_tick = self.allocate_u64(n)?;
        b.adaptation_fast = self.allocate_f32(n)?;
        b.adaptation_medium = self.allocate_f32(n)?;
        b.adaptation_slow = self.allocate_f32(n)?;
        b.touched_epoch = self.allocate_u64(n)?;
        b.touched_list = self.allocate_u32(n)?;
        b.touched_count = self.allocate_u32(1)?;
        b.selected_epoch = self.allocate_u64(n)?;
        b.surrogate = self.allocate_f32(n)?;
        b.candidate_activation = self.allocate_f32(n)?;
        b.active = self.allocate_u32(active)?;
        b.active_value = self.allocate_f32(active)?;
        b.active_count = self.allocate_u32(1)?;
        b.block_winner_neuron = self.allocate_u32(block_winners)?;
        b.block_winner_value = self.allocate_f32(block_winners)?;
        b.block_cutoff = self.allocate_f32(self.model.config.model.block_count)?;
        b.population_cutoff = self.allocate_f32(1)?;
        b.population_inhibition = self.allocate_f32(1)?;

        b.recurrent_branch_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_membrane_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_fatigue_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_adaptation_fast_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_adaptation_medium_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_adaptation_slow_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_eligibility = self.allocate_f32(recurrent)?;
        b.recurrent_last_tick = self.allocate_u64(recurrent)?;
        b.recurrent_eligible_mark = self.allocate_u32(recurrent)?;
        b.recurrent_eligible_list = self.allocate_u32(recurrent)?;
        b.recurrent_eligible_count = self.allocate_u32(1)?;
        b.recurrent_next_eligible_list = self.allocate_u32(recurrent)?;
        b.recurrent_next_eligible_count = self.allocate_u32(1)?;

        b.input_branch_sensitivity = self.allocate_f32(input)?;
        b.input_membrane_sensitivity = self.allocate_f32(input)?;
        b.input_fatigue_sensitivity = self.allocate_f32(input)?;
        b.input_adaptation_fast_sensitivity = self.allocate_f32(input)?;
        b.input_adaptation_medium_sensitivity = self.allocate_f32(input)?;
        b.input_adaptation_slow_sensitivity = self.allocate_f32(input)?;
        b.input_eligibility = self.allocate_f32(input)?;
        b.input_last_tick = self.allocate_u64(input)?;
        b.input_eligible_mark = self.allocate_u32(input)?;
        b.input_eligible_list = self.allocate_u32(input)?;
        b.input_eligible_count = self.allocate_u32(1)?;
        b.input_next_eligible_list = self.allocate_u32(input)?;
        b.input_next_eligible_count = self.allocate_u32(1)?;

        b.ring_count = self.allocate_u32(RING_BUCKETS)?;
        b.ring_source = self.allocate_u32(ring_records)?;
        b.ring_activation = self.allocate_f32(ring_records)?;
        b.ring_weight = self.allocate_f32(ring_weights)?;
        b.context_history = self.allocate_u32(max_order)?;
        b.context_history_count = self.allocate_u32(1)?;
        b.active_context_slots = self.allocate_u32(max_order)?;
        b.active_context_scales = self.allocate_f32(max_order)?;
        b.active_context_count = self.allocate_u32(1)?;
        b.context_latent = self.allocate_f32(context_dim)?;
        b.context_gradient = self.allocate_f32(context_dim)?;
        b.neural_logits = self.allocate_f32(OUTPUT_CLASSES)?;
        b.logits = self.allocate_f32(OUTPUT_CLASSES)?;
        b.probabilities = self.allocate_f32(OUTPUT_CLASSES)?;
        b.errors = self.allocate_f32(OUTPUT_CLASSES)?;
        b.learning_destination_epoch = self.allocate_u64(n)?;
        b.learning_destination_list = self.allocate_u32(n)?;
        b.learning_destination_count = self.allocate_u32(1)?;
        b.learning_signal = self.allocate_f32(n)?;
        b.counters = self.allocate_one::<CudaStepCounters>()?;
        b.error_flag = self.allocate_u32(1)?;
        b.training_step_records = self.allocate_bytes(
            TRAINING_STEP_BATCH_CAPACITY.saturating_mul(mem::size_of::<CudaTrainingStepRecord>()),
        )?;
        b.persistent_steps = self.allocate_bytes(
            TRAINING_STEP_BATCH_CAPACITY.saturating_mul(mem::size_of::<CudaPersistentStep>()),
        )?;
        b.persistent_pointer_table = self.allocate_u64(PERSISTENT_POINTER_COUNT)?;

        lane.host_pointer_table = self.allocate_pinned_host(
            PERSISTENT_POINTER_COUNT.saturating_mul(mem::size_of::<CuDevicePtr>()),
        )?;
        lane.host_steps = self.allocate_pinned_host(
            TRAINING_STEP_BATCH_CAPACITY.saturating_mul(mem::size_of::<CudaPersistentStep>()),
        )?;
        lane.host_records = self.allocate_pinned_host(
            TRAINING_STEP_BATCH_CAPACITY.saturating_mul(mem::size_of::<CudaTrainingStepRecord>()),
        )?;
        self.copy_canonical_parameters_to_batch_lane_async(&lane)?;
        self.reset_shared_batch_lane(&mut lane)?;
        self.refresh_shared_batch_pointer_table_async(&lane)?;
        self.synchronize()?;
        if self.debug.memory {
            let device_bytes = lane
                .buffers
                .all()
                .iter()
                .map(|buffer| buffer.bytes as u64)
                .sum::<u64>();
            let pinned_host_bytes = lane.host_pointer_table.bytes as u64
                + lane.host_steps.bytes as u64
                + lane.host_records.bytes as u64;
            eprintln!(
                "{{\"event\":\"cuda_debug_memory\",\"scope\":\"story_lane\",\"device_bytes\":{},\"pinned_host_bytes\":{},\"model_revision\":{}}}",
                device_bytes,
                pinned_host_bytes,
                lane.model_revision,
            );
        }
        Ok(lane)
    }

    fn copy_canonical_parameters_to_batch_lane_async(&self, lane: &CudaBatchLane) -> LeoResult<()> {
        let source = self.buffers;
        let target = lane.buffers;
        for (dst, src) in [
            (target.threshold, source.threshold),
            (target.recurrent_weight, source.recurrent_weight),
            (target.input_weight, source.input_weight),
            (target.output_weight, source.output_weight),
            (target.output_bias, source.output_bias),
            (target.context_keys, source.context_keys),
            (target.context_embeddings, source.context_embeddings),
            (target.context_observations, source.context_observations),
            (target.context_output_weight, source.context_output_weight),
        ] {
            self.copy_device_to_device_async_on(dst, src, self.transfer_stream)?;
        }
        Ok(())
    }

    fn reset_shared_batch_lane(&self, lane: &mut CudaBatchLane) -> LeoResult<()> {
        lane.current_tick = 0;
        let b = lane.buffers;
        for buffer in [
            b.membrane,
            b.activation,
            b.fatigue,
            b.refractory_until,
            b.branches,
            b.branch_delta,
            b.branch_last_tick,
            b.neuron_last_tick,
            b.adaptation_fast,
            b.adaptation_medium,
            b.adaptation_slow,
            b.touched_epoch,
            b.touched_count,
            b.selected_epoch,
            b.surrogate,
            b.candidate_activation,
            b.active_count,
            b.block_cutoff,
            b.population_cutoff,
            b.population_inhibition,
            b.changed_threshold_marks,
            b.changed_threshold_count,
            b.changed_recurrent_marks,
            b.changed_recurrent_count,
            b.changed_input_marks,
            b.changed_input_count,
            b.changed_output_marks,
            b.changed_output_count,
            b.changed_context_marks,
            b.changed_context_count,
            b.recurrent_branch_sensitivity,
            b.recurrent_membrane_sensitivity,
            b.recurrent_fatigue_sensitivity,
            b.recurrent_adaptation_fast_sensitivity,
            b.recurrent_adaptation_medium_sensitivity,
            b.recurrent_adaptation_slow_sensitivity,
            b.recurrent_eligibility,
            b.recurrent_last_tick,
            b.recurrent_eligible_mark,
            b.recurrent_eligible_count,
            b.recurrent_next_eligible_count,
            b.input_branch_sensitivity,
            b.input_membrane_sensitivity,
            b.input_fatigue_sensitivity,
            b.input_adaptation_fast_sensitivity,
            b.input_adaptation_medium_sensitivity,
            b.input_adaptation_slow_sensitivity,
            b.input_eligibility,
            b.input_last_tick,
            b.input_eligible_mark,
            b.input_eligible_count,
            b.input_next_eligible_count,
            b.ring_count,
            b.context_history,
            b.context_history_count,
            b.active_context_count,
            b.context_latent,
            b.context_gradient,
            b.neural_logits,
            b.logits,
            b.probabilities,
            b.errors,
            b.learning_destination_epoch,
            b.learning_destination_count,
            b.learning_signal,
            b.counters,
            b.error_flag,
        ] {
            self.memset_zero_async(buffer)?;
        }
        Ok(())
    }

    fn shared_batch_pointer_table(
        &self,
        lane: &CudaBatchLane,
    ) -> [CuDevicePtr; PERSISTENT_POINTER_COUNT] {
        let mut pointers = self.persistent_pointer_table();
        let b = lane.buffers;
        pointers[PersistentPointer::Threshold.index()] = b.threshold.pointer;
        pointers[PersistentPointer::RecurrentWeight.index()] = b.recurrent_weight.pointer;
        pointers[PersistentPointer::InputWeight.index()] = b.input_weight.pointer;
        pointers[PersistentPointer::OutputWeight.index()] = b.output_weight.pointer;
        pointers[PersistentPointer::OutputBias.index()] = b.output_bias.pointer;
        pointers[PersistentPointer::ContextKeys.index()] = b.context_keys.pointer;
        pointers[PersistentPointer::ContextEmbeddings.index()] = b.context_embeddings.pointer;
        pointers[PersistentPointer::ContextObservations.index()] = b.context_observations.pointer;
        pointers[PersistentPointer::ContextOutputWeight.index()] = b.context_output_weight.pointer;
        pointers[PersistentPointer::ChangedThresholdMarks.index()] =
            b.changed_threshold_marks.pointer;
        pointers[PersistentPointer::ChangedThresholdList.index()] =
            b.changed_threshold_list.pointer;
        pointers[PersistentPointer::ChangedThresholdCount.index()] =
            b.changed_threshold_count.pointer;
        pointers[PersistentPointer::ChangedRecurrentMarks.index()] =
            b.changed_recurrent_marks.pointer;
        pointers[PersistentPointer::ChangedRecurrentList.index()] =
            b.changed_recurrent_list.pointer;
        pointers[PersistentPointer::ChangedRecurrentCount.index()] =
            b.changed_recurrent_count.pointer;
        pointers[PersistentPointer::ChangedInputMarks.index()] = b.changed_input_marks.pointer;
        pointers[PersistentPointer::ChangedInputList.index()] = b.changed_input_list.pointer;
        pointers[PersistentPointer::ChangedInputCount.index()] = b.changed_input_count.pointer;
        pointers[PersistentPointer::ChangedOutputMarks.index()] = b.changed_output_marks.pointer;
        pointers[PersistentPointer::ChangedOutputList.index()] = b.changed_output_list.pointer;
        pointers[PersistentPointer::ChangedOutputCount.index()] = b.changed_output_count.pointer;
        pointers[PersistentPointer::ChangedContextMarks.index()] = b.changed_context_marks.pointer;
        pointers[PersistentPointer::ChangedContextList.index()] = b.changed_context_list.pointer;
        pointers[PersistentPointer::ChangedContextCount.index()] = b.changed_context_count.pointer;
        pointers[PersistentPointer::Membrane.index()] = b.membrane.pointer;
        pointers[PersistentPointer::Activation.index()] = b.activation.pointer;
        pointers[PersistentPointer::Fatigue.index()] = b.fatigue.pointer;
        pointers[PersistentPointer::RefractoryUntil.index()] = b.refractory_until.pointer;
        pointers[PersistentPointer::Branches.index()] = b.branches.pointer;
        pointers[PersistentPointer::BranchDelta.index()] = b.branch_delta.pointer;
        pointers[PersistentPointer::BranchLastTick.index()] = b.branch_last_tick.pointer;
        pointers[PersistentPointer::NeuronLastTick.index()] = b.neuron_last_tick.pointer;
        pointers[PersistentPointer::AdaptationFast.index()] = b.adaptation_fast.pointer;
        pointers[PersistentPointer::AdaptationMedium.index()] = b.adaptation_medium.pointer;
        pointers[PersistentPointer::AdaptationSlow.index()] = b.adaptation_slow.pointer;
        pointers[PersistentPointer::TouchedEpoch.index()] = b.touched_epoch.pointer;
        pointers[PersistentPointer::TouchedList.index()] = b.touched_list.pointer;
        pointers[PersistentPointer::TouchedCount.index()] = b.touched_count.pointer;
        pointers[PersistentPointer::SelectedEpoch.index()] = b.selected_epoch.pointer;
        pointers[PersistentPointer::Surrogate.index()] = b.surrogate.pointer;
        pointers[PersistentPointer::CandidateActivation.index()] = b.candidate_activation.pointer;
        pointers[PersistentPointer::Active.index()] = b.active.pointer;
        pointers[PersistentPointer::ActiveValue.index()] = b.active_value.pointer;
        pointers[PersistentPointer::ActiveCount.index()] = b.active_count.pointer;
        pointers[PersistentPointer::BlockWinnerNeuron.index()] = b.block_winner_neuron.pointer;
        pointers[PersistentPointer::BlockWinnerValue.index()] = b.block_winner_value.pointer;
        pointers[PersistentPointer::BlockCutoff.index()] = b.block_cutoff.pointer;
        pointers[PersistentPointer::PopulationCutoff.index()] = b.population_cutoff.pointer;
        pointers[PersistentPointer::PopulationInhibition.index()] = b.population_inhibition.pointer;
        pointers[PersistentPointer::RecBranchSensitivity.index()] =
            b.recurrent_branch_sensitivity.pointer;
        pointers[PersistentPointer::RecMembraneSensitivity.index()] =
            b.recurrent_membrane_sensitivity.pointer;
        pointers[PersistentPointer::RecFatigueSensitivity.index()] =
            b.recurrent_fatigue_sensitivity.pointer;
        pointers[PersistentPointer::RecAdaptationFastSensitivity.index()] =
            b.recurrent_adaptation_fast_sensitivity.pointer;
        pointers[PersistentPointer::RecAdaptationMediumSensitivity.index()] =
            b.recurrent_adaptation_medium_sensitivity.pointer;
        pointers[PersistentPointer::RecAdaptationSlowSensitivity.index()] =
            b.recurrent_adaptation_slow_sensitivity.pointer;
        pointers[PersistentPointer::RecEligibility.index()] = b.recurrent_eligibility.pointer;
        pointers[PersistentPointer::RecLastTick.index()] = b.recurrent_last_tick.pointer;
        pointers[PersistentPointer::RecEligibleMark.index()] = b.recurrent_eligible_mark.pointer;
        pointers[PersistentPointer::RecEligibleList.index()] = b.recurrent_eligible_list.pointer;
        pointers[PersistentPointer::RecEligibleCount.index()] = b.recurrent_eligible_count.pointer;
        pointers[PersistentPointer::RecNextEligibleList.index()] =
            b.recurrent_next_eligible_list.pointer;
        pointers[PersistentPointer::RecNextEligibleCount.index()] =
            b.recurrent_next_eligible_count.pointer;
        pointers[PersistentPointer::InputBranchSensitivity.index()] =
            b.input_branch_sensitivity.pointer;
        pointers[PersistentPointer::InputMembraneSensitivity.index()] =
            b.input_membrane_sensitivity.pointer;
        pointers[PersistentPointer::InputFatigueSensitivity.index()] =
            b.input_fatigue_sensitivity.pointer;
        pointers[PersistentPointer::InputAdaptationFastSensitivity.index()] =
            b.input_adaptation_fast_sensitivity.pointer;
        pointers[PersistentPointer::InputAdaptationMediumSensitivity.index()] =
            b.input_adaptation_medium_sensitivity.pointer;
        pointers[PersistentPointer::InputAdaptationSlowSensitivity.index()] =
            b.input_adaptation_slow_sensitivity.pointer;
        pointers[PersistentPointer::InputEligibility.index()] = b.input_eligibility.pointer;
        pointers[PersistentPointer::InputLastTick.index()] = b.input_last_tick.pointer;
        pointers[PersistentPointer::InputEligibleMark.index()] = b.input_eligible_mark.pointer;
        pointers[PersistentPointer::InputEligibleList.index()] = b.input_eligible_list.pointer;
        pointers[PersistentPointer::InputEligibleCount.index()] = b.input_eligible_count.pointer;
        pointers[PersistentPointer::InputNextEligibleList.index()] =
            b.input_next_eligible_list.pointer;
        pointers[PersistentPointer::InputNextEligibleCount.index()] =
            b.input_next_eligible_count.pointer;
        pointers[PersistentPointer::RingCount.index()] = b.ring_count.pointer;
        pointers[PersistentPointer::RingSource.index()] = b.ring_source.pointer;
        pointers[PersistentPointer::RingActivation.index()] = b.ring_activation.pointer;
        pointers[PersistentPointer::RingWeight.index()] = b.ring_weight.pointer;
        pointers[PersistentPointer::ContextHistory.index()] = b.context_history.pointer;
        pointers[PersistentPointer::ContextHistoryCount.index()] = b.context_history_count.pointer;
        pointers[PersistentPointer::ActiveContextSlots.index()] = b.active_context_slots.pointer;
        pointers[PersistentPointer::ActiveContextScales.index()] = b.active_context_scales.pointer;
        pointers[PersistentPointer::ActiveContextCount.index()] = b.active_context_count.pointer;
        pointers[PersistentPointer::ContextLatent.index()] = b.context_latent.pointer;
        pointers[PersistentPointer::ContextGradient.index()] = b.context_gradient.pointer;
        pointers[PersistentPointer::NeuralLogits.index()] = b.neural_logits.pointer;
        pointers[PersistentPointer::Logits.index()] = b.logits.pointer;
        pointers[PersistentPointer::Probabilities.index()] = b.probabilities.pointer;
        pointers[PersistentPointer::Errors.index()] = b.errors.pointer;
        pointers[PersistentPointer::LearningDestinationEpoch.index()] =
            b.learning_destination_epoch.pointer;
        pointers[PersistentPointer::LearningDestinationList.index()] =
            b.learning_destination_list.pointer;
        pointers[PersistentPointer::LearningDestinationCount.index()] =
            b.learning_destination_count.pointer;
        pointers[PersistentPointer::LearningSignal.index()] = b.learning_signal.pointer;
        pointers[PersistentPointer::Counters.index()] = b.counters.pointer;
        pointers[PersistentPointer::ErrorFlag.index()] = b.error_flag.pointer;
        pointers[PersistentPointer::TrainingStepRecords.index()] = b.training_step_records.pointer;
        pointers
    }

    fn refresh_shared_batch_pointer_table_async(&self, lane: &CudaBatchLane) -> LeoResult<()> {
        let pointer_table = self.shared_batch_pointer_table(lane);
        pinned_write(lane.host_pointer_table, &pointer_table)?;
        self.copy_pinned_to_device_async_on::<CuDevicePtr>(
            lane.buffers.persistent_pointer_table,
            lane.host_pointer_table,
            PERSISTENT_POINTER_COUNT,
            self.transfer_stream,
        )
    }

    fn batch_delta_pointer_table(&self) -> [CuDevicePtr; BATCH_DELTA_POINTER_COUNT] {
        let b = self.buffers;
        let mut pointers = [0; BATCH_DELTA_POINTER_COUNT];
        pointers[BatchDeltaPointer::Threshold.index()] = b.batch_delta_threshold.pointer;
        pointers[BatchDeltaPointer::ThresholdMarks.index()] = b.batch_delta_threshold_marks.pointer;
        pointers[BatchDeltaPointer::ThresholdList.index()] = b.batch_delta_threshold_list.pointer;
        pointers[BatchDeltaPointer::ThresholdCount.index()] = b.batch_delta_threshold_count.pointer;
        pointers[BatchDeltaPointer::Recurrent.index()] = b.batch_delta_recurrent.pointer;
        pointers[BatchDeltaPointer::RecurrentMarks.index()] = b.batch_delta_recurrent_marks.pointer;
        pointers[BatchDeltaPointer::RecurrentList.index()] = b.batch_delta_recurrent_list.pointer;
        pointers[BatchDeltaPointer::RecurrentCount.index()] = b.batch_delta_recurrent_count.pointer;
        pointers[BatchDeltaPointer::Input.index()] = b.batch_delta_input.pointer;
        pointers[BatchDeltaPointer::InputMarks.index()] = b.batch_delta_input_marks.pointer;
        pointers[BatchDeltaPointer::InputList.index()] = b.batch_delta_input_list.pointer;
        pointers[BatchDeltaPointer::InputCount.index()] = b.batch_delta_input_count.pointer;
        pointers[BatchDeltaPointer::Output.index()] = b.batch_delta_output.pointer;
        pointers[BatchDeltaPointer::OutputMarks.index()] = b.batch_delta_output_marks.pointer;
        pointers[BatchDeltaPointer::OutputList.index()] = b.batch_delta_output_list.pointer;
        pointers[BatchDeltaPointer::OutputCount.index()] = b.batch_delta_output_count.pointer;
        pointers[BatchDeltaPointer::OutputBias.index()] = b.batch_delta_output_bias.pointer;
        pointers[BatchDeltaPointer::ContextEmbedding.index()] =
            b.batch_delta_context_embedding.pointer;
        pointers[BatchDeltaPointer::ContextMarks.index()] = b.batch_delta_context_marks.pointer;
        pointers[BatchDeltaPointer::ContextList.index()] = b.batch_delta_context_list.pointer;
        pointers[BatchDeltaPointer::ContextCount.index()] = b.batch_delta_context_count.pointer;
        pointers[BatchDeltaPointer::ContextObservations.index()] =
            b.batch_delta_context_observations.pointer;
        pointers[BatchDeltaPointer::ContextOutput.index()] = b.batch_delta_context_output.pointer;
        pointers
    }

    pub(crate) fn model(&self) -> &Model {
        &self.model
    }

    pub(crate) fn model_mut(&mut self) -> LeoResult<&mut Model> {
        self.synchronize_model()?;
        self.model_dirty = true;
        Ok(&mut self.model)
    }

    pub(crate) fn synchronize_model(&mut self) -> LeoResult<()> {
        self.make_current()?;
        if self.model_dirty {
            return Ok(());
        }
        self.synchronize()?;
        self.sync_changed_model_to_host()?;
        Ok(())
    }

    pub(crate) fn enable_parameter_tracking(&mut self) {
        self.track_parameter_changes = true;
        self.accumulated_changes = ParameterChanges::default();
        self.accumulated_threshold_mark.fill(false);
        self.accumulated_recurrent_mark.fill(false);
        self.accumulated_input_mark.fill(false);
        self.accumulated_output_mark.fill(false);
        self.accumulated_context_mark.fill(false);
        self.output_bias_dirty_since_sync = false;
        self.context_output_dirty_since_sync = false;
    }

    pub(crate) fn parameter_changes(&mut self) -> LeoResult<ParameterChanges> {
        self.synchronize_model()?;
        Ok(self.accumulated_changes.clone())
    }

    /// Apply one host-packed canonical sparse update to this runtime and every
    /// resident logical-story lane. Packing is performed once by the training
    /// coordinator; on each device the index/value payload is uploaded once
    /// and fanned out to all lane parameter images with device-side scatters.
    /// This is the steady-state multi-GPU synchronization path and prevents a
    /// full canonical-to-lane model copy at the beginning of the next batch.
    pub(crate) fn synchronize_packed_model(
        &mut self,
        update: &PackedSparseModelUpdate,
        lanes: &mut [CudaBatchLane],
    ) -> LeoResult<()> {
        self.apply_packed_sparse_update(update, lanes, true)
    }

    /// Commit a canonical host merge to device 0 and its resident story lanes.
    /// Unlike replica synchronization this preserves parameter-tracking mode;
    /// replay enables tracking immediately after this batch barrier.
    pub(crate) fn commit_packed_host_model(
        &mut self,
        update: &PackedSparseModelUpdate,
        lanes: &mut [CudaBatchLane],
    ) -> LeoResult<()> {
        self.apply_packed_sparse_update(update, lanes, false)
    }

    fn apply_packed_sparse_update(
        &mut self,
        update: &PackedSparseModelUpdate,
        lanes: &mut [CudaBatchLane],
        clear_tracking: bool,
    ) -> LeoResult<()> {
        self.make_current()?;
        update.apply_to_model(&mut self.model)?;

        let scratch = self.buffers;
        let mut targets = Vec::with_capacity(lanes.len().saturating_add(1));
        targets.push(self.buffers);
        targets.extend(lanes.iter().map(|lane| lane.buffers));

        let scatter_f32 = |runtime: &CudaRuntime,
                           target: DeviceBuffer,
                           index_buffer: DeviceBuffer,
                           value_buffer: DeviceBuffer,
                           count: usize,
                           label: &str|
         -> LeoResult<()> {
            if count == 0 {
                return Ok(());
            }
            let mut target_ptr = target.pointer;
            let mut indices_ptr = index_buffer.pointer;
            let mut count = as_u32(label, count)?;
            let mut values_ptr = value_buffer.pointer;
            let mut params = [
                param(&mut target_ptr),
                param(&mut indices_ptr),
                param(&mut count),
                param(&mut values_ptr),
            ];
            runtime.launch(
                runtime.shared.kernels.scatter_f32,
                count as usize,
                THREADS,
                &mut params,
            )
        };

        if !update.threshold_indices.is_empty() {
            self.copy_to_device(scratch.changed_threshold_list, &update.threshold_indices)?;
            self.copy_to_device(scratch.gather_threshold, &update.threshold_values)?;
            for target in &targets {
                scatter_f32(
                    self,
                    target.threshold,
                    scratch.changed_threshold_list,
                    scratch.gather_threshold,
                    update.threshold_indices.len(),
                    "threshold count",
                )?;
            }
        }

        if !update.recurrent_indices.is_empty() {
            self.copy_to_device(scratch.changed_recurrent_list, &update.recurrent_indices)?;
            self.copy_to_device(scratch.gather_recurrent, &update.recurrent_values)?;
            for target in &targets {
                scatter_f32(
                    self,
                    target.recurrent_weight,
                    scratch.changed_recurrent_list,
                    scratch.gather_recurrent,
                    update.recurrent_indices.len(),
                    "recurrent count",
                )?;
            }
        }

        if !update.input_indices.is_empty() {
            self.copy_to_device(scratch.changed_input_list, &update.input_indices)?;
            self.copy_to_device(scratch.gather_input, &update.input_values)?;
            for target in &targets {
                scatter_f32(
                    self,
                    target.input_weight,
                    scratch.changed_input_list,
                    scratch.gather_input,
                    update.input_indices.len(),
                    "input count",
                )?;
            }
        }

        if !update.output_neurons.is_empty() {
            let expected = update.output_neurons.len().saturating_mul(OUTPUT_CLASSES);
            if update.output_values_by_neuron.len() != expected {
                return Err(LeoError::cuda(
                    "packed output value count does not match changed output rows",
                ));
            }
            self.copy_to_device(scratch.changed_output_list, &update.output_neurons)?;
            self.copy_to_device(scratch.gather_output, &update.output_values_by_neuron)?;
            for target in &targets {
                let mut target_ptr = target.output_weight.pointer;
                let mut indices_ptr = scratch.changed_output_list.pointer;
                let mut count = as_u32("output neuron count", update.output_neurons.len())?;
                let mut packed_ptr = scratch.gather_output.pointer;
                let mut params = [
                    param(&mut target_ptr),
                    param(&mut indices_ptr),
                    param(&mut count),
                    param(&mut packed_ptr),
                ];
                self.launch(
                    self.shared.kernels.scatter_output,
                    expected,
                    THREADS,
                    &mut params,
                )?;
            }
        }

        if !update.context_slots.is_empty() {
            let dim = self.model.config.context.embedding_dim;
            if update.context_keys.len() != update.context_slots.len()
                || update.context_observations.len() != update.context_slots.len()
                || update.context_embeddings.len() != update.context_slots.len().saturating_mul(dim)
            {
                return Err(LeoError::cuda(
                    "packed context rows have invalid dimensions",
                ));
            }
            self.copy_to_device(scratch.changed_context_list, &update.context_slots)?;
            self.copy_to_device(scratch.gather_context_keys, &update.context_keys)?;
            self.copy_to_device(
                scratch.gather_context_observations,
                &update.context_observations,
            )?;
            self.copy_to_device(
                scratch.gather_context_embeddings,
                &update.context_embeddings,
            )?;
            for target in &targets {
                let mut target_keys = target.context_keys.pointer;
                let mut target_observations = target.context_observations.pointer;
                let mut target_embeddings = target.context_embeddings.pointer;
                let mut embedding_dim = as_u32("context embedding dim", dim)?;
                let mut slots = scratch.changed_context_list.pointer;
                let mut count = as_u32("context slot count", update.context_slots.len())?;
                let mut source_keys = scratch.gather_context_keys.pointer;
                let mut source_observations = scratch.gather_context_observations.pointer;
                let mut source_embeddings = scratch.gather_context_embeddings.pointer;
                let mut params = [
                    param(&mut target_keys),
                    param(&mut target_observations),
                    param(&mut target_embeddings),
                    param(&mut embedding_dim),
                    param(&mut slots),
                    param(&mut count),
                    param(&mut source_keys),
                    param(&mut source_observations),
                    param(&mut source_embeddings),
                ];
                self.launch(
                    self.shared.kernels.scatter_context,
                    update
                        .context_embeddings
                        .len()
                        .max(update.context_slots.len()),
                    THREADS,
                    &mut params,
                )?;
            }
        }

        if let Some(values) = &update.output_bias {
            if values.len() != OUTPUT_CLASSES {
                return Err(LeoError::cuda("packed output bias has invalid dimensions"));
            }
            self.copy_to_device(scratch.output_bias, values)?;
            for lane in lanes.iter() {
                self.copy_device_to_device_async_on(
                    lane.buffers.output_bias,
                    scratch.output_bias,
                    self.compute_stream,
                )?;
            }
        }

        if let Some(values) = &update.context_output_weights {
            if values.len() != self.model.context.output_weights.len() {
                return Err(LeoError::cuda(
                    "packed context output weights have invalid dimensions",
                ));
            }
            self.copy_to_device(scratch.context_output_weight, values)?;
            for lane in lanes.iter() {
                self.copy_device_to_device_async_on(
                    lane.buffers.context_output_weight,
                    scratch.context_output_weight,
                    self.compute_stream,
                )?;
            }
        }

        self.synchronize()?;
        let revision = update.parameter_revision;
        for lane in lanes.iter_mut() {
            lane.model_revision = revision;
        }
        self.model_dirty = false;
        if clear_tracking {
            self.track_parameter_changes = false;
        }
        self.output_bias_dirty_since_sync = false;
        self.context_output_dirty_since_sync = false;
        Ok(())
    }

    pub(crate) fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }

    pub(crate) fn active_neurons(&self) -> &[usize] {
        &self.active
    }

    pub(crate) fn activation(&self, neuron: usize) -> f32 {
        self.activation.get(neuron).copied().unwrap_or(0.0)
    }

    pub(crate) fn begin_document(&mut self) -> LeoResult<()> {
        self.reset_transient_state()
    }

    pub(crate) fn finish_document(&mut self) -> LeoResult<()> {
        if self.persistent_document_activity {
            self.model.statistics.processed_stories =
                self.model.statistics.processed_stories.saturating_add(1);
        }
        self.reset_transient_state()
    }

    pub(crate) fn reset_transient_state(&mut self) -> LeoResult<()> {
        self.make_current()?;
        let b = self.buffers;
        for buffer in [
            b.membrane,
            b.activation,
            b.fatigue,
            b.refractory_until,
            b.branches,
            b.branch_delta,
            b.adaptation_fast,
            b.adaptation_medium,
            b.adaptation_slow,
            b.touched_epoch,
            b.touched_count,
            b.selected_epoch,
            b.surrogate,
            b.candidate_activation,
            b.active,
            b.active_value,
            b.active_count,
            b.block_winner_neuron,
            b.block_winner_value,
            b.block_cutoff,
            b.population_cutoff,
            b.population_inhibition,
            b.recurrent_branch_sensitivity,
            b.recurrent_membrane_sensitivity,
            b.recurrent_fatigue_sensitivity,
            b.recurrent_adaptation_fast_sensitivity,
            b.recurrent_adaptation_medium_sensitivity,
            b.recurrent_adaptation_slow_sensitivity,
            b.recurrent_eligibility,
            b.recurrent_eligible_mark,
            b.recurrent_eligible_count,
            b.recurrent_next_eligible_count,
            b.input_branch_sensitivity,
            b.input_membrane_sensitivity,
            b.input_fatigue_sensitivity,
            b.input_adaptation_fast_sensitivity,
            b.input_adaptation_medium_sensitivity,
            b.input_adaptation_slow_sensitivity,
            b.input_eligibility,
            b.input_eligible_mark,
            b.input_eligible_count,
            b.input_next_eligible_count,
            b.ring_count,
            b.ring_source,
            b.ring_activation,
            b.ring_weight,
            b.context_history,
            b.context_history_count,
            b.active_context_slots,
            b.active_context_scales,
            b.active_context_count,
            b.context_latent,
            b.context_gradient,
            b.neural_logits,
            b.logits,
            b.probabilities,
            b.errors,
            b.learning_destination_epoch,
            b.learning_destination_count,
            b.learning_signal,
            b.counters,
            b.error_flag,
        ] {
            self.memset_zero_async(buffer)?;
        }
        let neuron_count = self.model.neuron_count();
        let recurrent_count = self.model.recurrent.weight.len();
        let input_count = self.model.input.weights.len();
        let work = neuron_count.max(recurrent_count).max(input_count);
        let mut tick = self.current_tick;
        let mut neurons = as_u32("neuron count", neuron_count)?;
        let mut recurrent = as_u32("recurrent count", recurrent_count)?;
        let mut input = as_u32("input count", input_count)?;
        let mut branch_last = b.branch_last_tick.pointer;
        let mut neuron_last = b.neuron_last_tick.pointer;
        let mut recurrent_last = b.recurrent_last_tick.pointer;
        let mut input_last = b.input_last_tick.pointer;
        let mut parameters = [
            param(&mut tick),
            param(&mut neurons),
            param(&mut recurrent),
            param(&mut input),
            param(&mut branch_last),
            param(&mut neuron_last),
            param(&mut recurrent_last),
            param(&mut input_last),
        ];
        self.launch(
            self.shared.kernels.reset_last_ticks,
            work,
            THREADS,
            &mut parameters,
        )?;
        self.synchronize()?;
        self.active.clear();
        self.activation.fill(0.0);
        self.persistent_document_activity = false;
        Ok(())
    }

    pub(crate) fn step(
        &mut self,
        symbol: u32,
        target: Option<u32>,
        permission: Permission,
    ) -> LeoResult<StepMetrics> {
        if !is_input_symbol(symbol) {
            return Err(LeoError::cuda(format!("invalid input symbol: {symbol}")));
        }
        let target_index = target
            .map(|value| {
                output_symbol_to_index(value)
                    .ok_or_else(|| LeoError::cuda(format!("invalid output target: {value}")))
            })
            .transpose()?;
        self.make_current()?;
        if self.model_dirty {
            self.upload_full_model()?;
        }
        let learning_trace = !matches!(permission, Permission::Frozen);
        let context_enabled = matches!(permission, Permission::Frozen)
            || context_survives_dropout(
                self.model.config.model.seed,
                self.current_tick,
                symbol,
                self.model.config.context.dropout_rate,
            );
        let allocate_context = learning_trace && context_enabled;
        self.launch_step_prefix(symbol, learning_trace, context_enabled, allocate_context)?;
        let target_i32 = target_index.map(|index| index as i32).unwrap_or(-1);
        self.launch_forward(target_i32)?;

        let strength = permission.strength(&self.model.config);
        let mut learned = false;
        if let Some(target_output_index) = target_index {
            if strength > 0.0 {
                let target_weight = if target_output_index == END_DOCUMENT_OUTPUT_INDEX {
                    self.model.config.learning.end_document_weight
                } else {
                    1.0
                };
                self.launch_learning(strength, strength * target_weight, context_enabled)?;
                self.model.parameter_revision = self.model.parameter_revision.saturating_add(1);
                self.output_bias_dirty_since_sync = true;
                if context_enabled {
                    self.context_output_dirty_since_sync = true;
                }
                learned = true;
            }
        }
        if learning_trace {
            self.launch_homeostasis()?;
        }
        self.synchronize()?;
        let metrics = self.collect_step_metrics(target_index, context_enabled, learned)?;
        if learning_trace {
            self.persistent_document_activity = true;
            self.update_statistics(
                symbol,
                metrics.loss,
                metrics.active_neurons,
                metrics.emitted_events,
            );
        }
        self.current_tick = self.current_tick.saturating_add(1);
        Ok(metrics)
    }

    pub(crate) fn ensure_device_model_current(&mut self) -> LeoResult<()> {
        self.make_current()?;
        if self.model_dirty {
            self.upload_full_model()?;
        }
        Ok(())
    }

    pub(crate) fn advance_frozen_batch(&mut self, steps: &[(u32, Option<u32>)]) -> LeoResult<()> {
        if steps.is_empty() {
            return Ok(());
        }
        if steps.iter().any(|(_, target)| target.is_some()) {
            return Err(LeoError::cuda(
                "frozen state advance does not accept supervised targets",
            ));
        }

        self.make_current()?;
        if self.model_dirty {
            self.upload_full_model()?;
        }

        let debug_chunks =
            self.debug.chunks || self.debug.sync || self.debug.transfers || self.debug.state;
        let pointer_upload_started = debug_chunks.then(Instant::now);

        // Frozen replay does not mutate the pointer topology (no learning or
        // eligibility-list swapping), so upload the invariant pointer table
        // once for the whole prefix instead of once per 4096-step chunk.
        let pointer_table = self.persistent_pointer_table();
        self.copy_to_device(self.buffers.persistent_pointer_table, &pointer_table)?;
        let pointer_upload_ms = pointer_upload_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        for (chunk_index, chunk) in steps.chunks(TRAINING_STEP_BATCH_CAPACITY).enumerate() {
            let chunk_started = debug_chunks.then(Instant::now);
            let base_tick = self.current_tick;
            let build_started = debug_chunks.then(Instant::now);
            let mut device_steps = Vec::with_capacity(chunk.len());
            for &(symbol, _) in chunk {
                if !is_input_symbol(symbol) {
                    return Err(LeoError::cuda(format!(
                        "invalid input symbol in frozen state advance: {symbol}"
                    )));
                }
                device_steps.push(CudaPersistentStep {
                    symbol,
                    target_index: -1,
                    context_enabled: 0,
                    supervised_strength: 0.0,
                });
            }
            let build_ms = build_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let upload_started = debug_chunks.then(Instant::now);
            self.copy_to_device(self.buffers.persistent_steps, &device_steps)?;
            let upload_ms = upload_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let mut pointers = self.buffers.persistent_pointer_table.pointer;
            let mut persistent_steps = self.buffers.persistent_steps.pointer;
            let mut step_count = as_u32("frozen state advance step count", chunk.len())?;
            let mut tick = base_tick;

            let model_blocks = as_u32(
                "frozen replay model block count",
                self.model.config.model.block_count,
            )?;
            let frozen_blocks = self
                .frozen_grid_override
                .unwrap_or(self.shared.frozen_grid_blocks)
                .min(self.shared.frozen_grid_blocks.max(1))
                .min(model_blocks)
                .max(1);
            let profile_stride = self.replay_profile.prefix_stride;
            let profile_requested = profile_stride
                .is_some_and(|stride| self.replay_profile_prefix_sequence % stride == 0);
            self.replay_profile_prefix_sequence =
                self.replay_profile_prefix_sequence.saturating_add(1);
            let profile_sampled = profile_requested
                && self.shared.cooperative_launch
                && frozen_blocks > 1
                && self.shared.frozen_profile_grid_blocks >= frozen_blocks;

            if profile_requested && !profile_sampled {
                eprintln!(
                    "{{\"event\":\"cuda_frozen_kernel_profile_skipped\",\"reason\":\"profiled_kernel_cannot_match_production_grid\",\"requested_grid_blocks\":{},\"normal_capacity_blocks\":{},\"profiled_capacity_blocks\":{},\"step_count\":{}}}",
                    frozen_blocks,
                    self.shared.frozen_grid_blocks,
                    self.shared.frozen_profile_grid_blocks,
                    chunk.len(),
                );
            }

            if profile_sampled {
                self.clear_replay_profile_counters()?;
            }

            if self.debug.launches {
                let kernel = if profile_sampled {
                    "leo_advance_frozen_cooperative_profiled"
                } else if self.shared.cooperative_launch && frozen_blocks > 1 {
                    "leo_advance_frozen_cooperative"
                } else {
                    "leo_advance_frozen_persistent"
                };
                eprintln!(
                    "{{\"event\":\"cuda_debug_launch\",\"scope\":\"replay_prefix\",\"kernel\":\"{}\",\"grid_blocks\":{},\"threads\":{},\"step_count\":{},\"base_tick\":{},\"profiled\":{}}}",
                    kernel,
                    if self.shared.cooperative_launch && frozen_blocks > 1 { frozen_blocks } else { 1 },
                    PERSISTENT_THREADS,
                    chunk.len(),
                    base_tick,
                    profile_sampled,
                );
            }

            let kernel_started = debug_chunks.then(Instant::now);
            if profile_sampled {
                let mut profile_counters = self.buffers.replay_profile_counters.pointer;
                let mut parameters = [
                    param(&mut pointers),
                    param(&mut persistent_steps),
                    param(&mut step_count),
                    param(&mut tick),
                    param(&mut profile_counters),
                ];
                self.launch_cooperative_exact(
                    self.shared.kernels.advance_frozen_cooperative_profiled,
                    frozen_blocks,
                    PERSISTENT_THREADS,
                    &mut parameters,
                )?;
            } else {
                let mut parameters = [
                    param(&mut pointers),
                    param(&mut persistent_steps),
                    param(&mut step_count),
                    param(&mut tick),
                ];
                if self.shared.cooperative_launch && frozen_blocks > 1 {
                    self.launch_cooperative_exact(
                        self.shared.kernels.advance_frozen_cooperative,
                        frozen_blocks,
                        PERSISTENT_THREADS,
                        &mut parameters,
                    )?;
                } else {
                    self.launch_exact(
                        self.shared.kernels.advance_frozen_persistent,
                        1,
                        PERSISTENT_THREADS,
                        &mut parameters,
                    )?;
                }
            }
            self.synchronize()?;
            let kernel_sync_ms = kernel_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            if profile_sampled {
                self.emit_frozen_prefix_profile(frozen_blocks, profile_stride.unwrap_or(1))?;
            }

            self.current_tick = self.current_tick.saturating_add(chunk.len() as u64);
            if debug_chunks {
                let total_ms = chunk_started
                    .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                eprintln!(
                    "{{\"event\":\"cuda_debug_chunk\",\"scope\":\"replay_prefix\",\"chunk_index\":{},\"steps\":{},\"base_tick\":{},\"end_tick\":{},\"grid_blocks\":{},\"pointer_upload_ms\":{},\"build_ms\":{},\"step_upload_ms\":{},\"kernel_sync_ms\":{},\"total_ms\":{},\"parameter_revision\":{}}}",
                    chunk_index,
                    chunk.len(),
                    base_tick,
                    self.current_tick,
                    if self.shared.cooperative_launch && frozen_blocks > 1 { frozen_blocks } else { 1 },
                    if chunk_index == 0 { pointer_upload_ms } else { 0.0 },
                    build_ms,
                    upload_ms,
                    kernel_sync_ms,
                    total_ms,
                    self.model.parameter_revision,
                );
            }
        }

        Ok(())
    }

    pub(crate) fn training_step_batch(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        if self.shared.persistent_grid_blocks > 0 {
            return self.training_step_batch_persistent(steps, permission);
        }
        // Older CUDA devices without cooperative launch support keep the same
        // FP32 semantics through the per-step fallback. Persistent execution is
        // the v1 default whenever the hardware can provide the required grid barrier.
        self.training_step_batch_noncooperative(steps, permission)
    }

    /// Execute an interleaved frozen-prefix/supervised replay schedule without
    /// returning to the host at every selected range boundary. A target_index
    /// of -2 is an internal CUDA sentinel for a frozen transient-state advance;
    /// ordinary unsupervised training continues to use -1.
    pub(crate) fn replay_streaming_batch(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        if steps.is_empty() {
            return Ok(Vec::new());
        }

        let learning_trace = !matches!(permission, Permission::Frozen);
        let fast_metrics_requested = learning_trace && !self.full_step_metrics;
        let production_capacity = if fast_metrics_requested {
            self.shared.replay_fast_grid_blocks
        } else {
            self.shared.replay_grid_blocks
        };
        let cooperative_replay = self.replay_cooperative_enabled
            && self.shared.cooperative_launch
            && production_capacity > 0;

        if !cooperative_replay {
            // Capability fallback: preserve streaming semantics while still
            // batching contiguous frozen/target runs through the established
            // implementations.
            let mut metrics = Vec::new();
            let mut cursor = 0usize;
            while cursor < steps.len() {
                let supervised = steps[cursor].1.is_some();
                let mut end = cursor + 1;
                while end < steps.len() && steps[end].1.is_some() == supervised {
                    end += 1;
                }
                if supervised {
                    metrics.extend(self.training_step_batch(&steps[cursor..end], permission)?);
                } else {
                    self.advance_frozen_batch(&steps[cursor..end])?;
                }
                cursor = end;
            }
            return Ok(metrics);
        }

        self.make_current()?;
        if self.model_dirty {
            self.upload_full_model()?;
        }

        let strength = permission.strength(&self.model.config);
        let mut all_metrics =
            Vec::with_capacity(steps.iter().filter(|step| step.1.is_some()).count());
        let debug_chunks =
            self.debug.chunks || self.debug.sync || self.debug.transfers || self.debug.state;

        for (chunk_index, chunk) in steps.chunks(TRAINING_STEP_BATCH_CAPACITY).enumerate() {
            let chunk_started = debug_chunks.then(Instant::now);
            let base_tick = self.current_tick;
            let mut device_steps = Vec::with_capacity(chunk.len());
            let mut metadata = Vec::with_capacity(chunk.len());
            let mut trace_steps = 0usize;
            let mut learned_steps = 0u64;
            let mut context_learning = false;

            for (timeline_index, &(symbol, target)) in chunk.iter().enumerate() {
                if !is_input_symbol(symbol) {
                    return Err(LeoError::cuda(format!("invalid input symbol: {symbol}")));
                }
                let tick = base_tick.saturating_add(timeline_index as u64);
                let target_index = target
                    .map(|value| {
                        output_symbol_to_index(value).ok_or_else(|| {
                            LeoError::cuda(format!("invalid output target: {value}"))
                        })
                    })
                    .transpose()?;

                if let Some(target_output_index) = target_index {
                    let context_enabled = matches!(permission, Permission::Frozen)
                        || context_survives_dropout(
                            self.model.config.model.seed,
                            tick,
                            symbol,
                            self.model.config.context.dropout_rate,
                        );
                    let target_weight = if target_output_index == END_DOCUMENT_OUTPUT_INDEX {
                        self.model.config.learning.end_document_weight
                    } else {
                        1.0
                    };
                    let learned = strength > 0.0;
                    if learning_trace {
                        trace_steps += 1;
                    }
                    if learned {
                        learned_steps = learned_steps.saturating_add(1);
                        context_learning |= context_enabled;
                    }
                    device_steps.push(CudaPersistentStep {
                        symbol,
                        target_index: target_output_index as i32,
                        context_enabled: u32::from(context_enabled),
                        supervised_strength: strength * target_weight,
                    });
                    metadata.push(Some((
                        symbol,
                        target_output_index,
                        context_enabled,
                        learned,
                    )));
                } else {
                    device_steps.push(CudaPersistentStep {
                        symbol,
                        target_index: -2,
                        context_enabled: 1,
                        supervised_strength: 0.0,
                    });
                    metadata.push(None);
                }
            }

            let pointer_table = self.persistent_pointer_table();
            self.copy_to_device(self.buffers.persistent_pointer_table, &pointer_table)?;
            self.copy_to_device(self.buffers.persistent_steps, &device_steps)?;

            let mut pointers = self.buffers.persistent_pointer_table.pointer;
            let mut persistent_steps = self.buffers.persistent_steps.pointer;
            let mut step_count = as_u32("streaming replay step count", chunk.len())?;
            let mut tick = base_tick;
            let mut learning = u32::from(learning_trace);
            let mut raw_strength = strength;
            let model_blocks = as_u32(
                "streaming replay model block count",
                self.model.config.model.block_count,
            )?;
            let replay_blocks = self
                .replay_grid_override
                .unwrap_or(production_capacity)
                .min(production_capacity.max(1))
                .min(model_blocks)
                .max(1);
            let mut parameters = [
                param(&mut pointers),
                param(&mut persistent_steps),
                param(&mut step_count),
                param(&mut tick),
                param(&mut learning),
                param(&mut raw_strength),
            ];

            if self.debug.launches {
                eprintln!(
                    "{{\"event\":\"cuda_debug_launch\",\"scope\":\"streaming_replay_schedule\",\"kernel\":\"{}\",\"grid_blocks\":{},\"threads\":{},\"timeline_steps\":{},\"target_steps\":{},\"base_tick\":{}}}",
                    if fast_metrics_requested { "leo_train_cooperative_fast" } else { "leo_train_cooperative" },
                    replay_blocks,
                    PERSISTENT_THREADS,
                    chunk.len(),
                    trace_steps,
                    base_tick,
                );
            }

            self.launch_cooperative_exact(
                if fast_metrics_requested {
                    self.shared.kernels.train_cooperative_fast
                } else {
                    self.shared.kernels.train_cooperative
                },
                replay_blocks,
                PERSISTENT_THREADS,
                &mut parameters,
            )?;
            self.synchronize()?;

            let fast_records = if fast_metrics_requested {
                let mut records = vec![CudaFastTrainingStepRecord::default(); chunk.len()];
                self.copy_from_device_prefix(self.buffers.training_step_records, &mut records)?;
                Some(records)
            } else {
                None
            };
            let full_records = if fast_metrics_requested {
                None
            } else {
                let mut records = vec![CudaTrainingStepRecord::default(); chunk.len()];
                self.copy_from_device_prefix(self.buffers.training_step_records, &mut records)?;
                Some(records)
            };

            if learning_trace && trace_steps % 2 == 1 {
                std::mem::swap(
                    &mut self.buffers.recurrent_eligible_list,
                    &mut self.buffers.recurrent_next_eligible_list,
                );
                std::mem::swap(
                    &mut self.buffers.recurrent_eligible_count,
                    &mut self.buffers.recurrent_next_eligible_count,
                );
                std::mem::swap(
                    &mut self.buffers.input_eligible_list,
                    &mut self.buffers.input_next_eligible_list,
                );
                std::mem::swap(
                    &mut self.buffers.input_eligible_count,
                    &mut self.buffers.input_next_eligible_count,
                );
            }

            self.current_tick = self.current_tick.saturating_add(chunk.len() as u64);
            self.model.parameter_revision =
                self.model.parameter_revision.saturating_add(learned_steps);
            if learned_steps > 0 {
                self.output_bias_dirty_since_sync = true;
                if context_learning {
                    self.context_output_dirty_since_sync = true;
                }
            }

            for (record_index, meta) in metadata.into_iter().enumerate() {
                let Some((symbol, target_index, context_enabled, learned)) = meta else {
                    continue;
                };
                let metrics = if let Some(records) = fast_records.as_ref() {
                    let record = records[record_index];
                    if record.error_code != 0 {
                        self.model.statistics.numerical_rejections =
                            self.model.statistics.numerical_rejections.saturating_add(1);
                        return Err(LeoError::cuda(format!(
                            "CUDA streaming replay rejected a non-finite value (code {})",
                            record.error_code
                        )));
                    }
                    self.fast_training_record_to_metrics(
                        record,
                        Some(target_index),
                        context_enabled,
                    )
                } else {
                    let record = full_records.as_ref().expect("full replay records")[record_index];
                    if record.error_code != 0 {
                        self.model.statistics.numerical_rejections =
                            self.model.statistics.numerical_rejections.saturating_add(1);
                        return Err(LeoError::cuda(format!(
                            "CUDA streaming replay rejected a non-finite value (code {})",
                            record.error_code
                        )));
                    }
                    self.training_record_to_metrics(
                        record,
                        Some(target_index),
                        context_enabled,
                        learned,
                    )
                };
                if learning_trace {
                    self.persistent_document_activity = true;
                    self.update_statistics(
                        symbol,
                        metrics.loss,
                        metrics.active_neurons,
                        metrics.emitted_events,
                    );
                }
                all_metrics.push(metrics);
            }

            if debug_chunks {
                let total_ms = chunk_started
                    .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                eprintln!(
                    "{{\"event\":\"cuda_debug_chunk\",\"scope\":\"streaming_replay_schedule\",\"chunk_index\":{},\"timeline_steps\":{},\"target_steps\":{},\"base_tick\":{},\"end_tick\":{},\"grid_blocks\":{},\"total_ms\":{},\"parameter_revision\":{}}}",
                    chunk_index,
                    chunk.len(),
                    trace_steps,
                    base_tick,
                    self.current_tick,
                    replay_blocks,
                    total_ms,
                    self.model.parameter_revision,
                );
            }
        }

        Ok(all_metrics)
    }

    fn training_step_batch_noncooperative(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        if steps.is_empty() {
            return Ok(Vec::new());
        }
        self.make_current()?;
        if self.model_dirty {
            self.upload_full_model()?;
        }

        let learning_trace = !matches!(permission, Permission::Frozen);
        let strength = permission.strength(&self.model.config);
        let mut all_metrics = Vec::with_capacity(steps.len());

        for chunk in steps.chunks(TRAINING_STEP_BATCH_CAPACITY) {
            let mut metadata = Vec::with_capacity(chunk.len());

            for (record_index, &(symbol, target)) in chunk.iter().enumerate() {
                if !is_input_symbol(symbol) {
                    return Err(LeoError::cuda(format!("invalid input symbol: {symbol}")));
                }
                let target_index = target
                    .map(|value| {
                        output_symbol_to_index(value).ok_or_else(|| {
                            LeoError::cuda(format!("invalid output target: {value}"))
                        })
                    })
                    .transpose()?;
                let context_enabled = matches!(permission, Permission::Frozen)
                    || context_survives_dropout(
                        self.model.config.model.seed,
                        self.current_tick,
                        symbol,
                        self.model.config.context.dropout_rate,
                    );
                let allocate_context = learning_trace && context_enabled;
                self.launch_step_prefix(symbol, learning_trace, context_enabled, allocate_context)?;
                let target_i32 = target_index.map(|index| index as i32).unwrap_or(-1);
                self.launch_forward(target_i32)?;

                let mut learned = false;
                if let Some(target_output_index) = target_index {
                    if strength > 0.0 {
                        let target_weight = if target_output_index == END_DOCUMENT_OUTPUT_INDEX {
                            self.model.config.learning.end_document_weight
                        } else {
                            1.0
                        };
                        self.launch_learning(strength, strength * target_weight, context_enabled)?;
                        self.model.parameter_revision =
                            self.model.parameter_revision.saturating_add(1);
                        self.output_bias_dirty_since_sync = true;
                        if context_enabled {
                            self.context_output_dirty_since_sync = true;
                        }
                        learned = true;
                    }
                }
                if learning_trace {
                    self.launch_homeostasis()?;
                }
                self.launch_capture_training_step(target_i32, record_index)?;
                metadata.push((symbol, target_index, context_enabled, learned));
                self.current_tick = self.current_tick.saturating_add(1);
            }

            self.synchronize()?;
            let mut records = vec![CudaTrainingStepRecord::default(); chunk.len()];
            self.copy_from_device_prefix(self.buffers.training_step_records, &mut records)?;

            for (record, (symbol, target_index, context_enabled, learned)) in
                records.into_iter().zip(metadata)
            {
                if record.error_code != 0 {
                    self.model.statistics.numerical_rejections =
                        self.model.statistics.numerical_rejections.saturating_add(1);
                    return Err(LeoError::cuda(format!(
                        "CUDA learning update rejected a non-finite value (code {})",
                        record.error_code
                    )));
                }
                let metrics =
                    self.training_record_to_metrics(record, target_index, context_enabled, learned);
                if learning_trace {
                    self.persistent_document_activity = true;
                    self.update_statistics(
                        symbol,
                        metrics.loss,
                        metrics.active_neurons,
                        metrics.emitted_events,
                    );
                }
                all_metrics.push(metrics);
            }
        }

        Ok(all_metrics)
    }

    fn training_step_batch_persistent(
        &mut self,
        steps: &[(u32, Option<u32>)],
        permission: Permission,
    ) -> LeoResult<Vec<StepMetrics>> {
        if steps.is_empty() {
            return Ok(Vec::new());
        }
        self.make_current()?;
        if self.model_dirty {
            self.upload_full_model()?;
        }

        let learning_trace = !matches!(permission, Permission::Frozen);
        let strength = permission.strength(&self.model.config);
        let mut all_metrics = Vec::with_capacity(steps.len());
        let debug_chunks =
            self.debug.chunks || self.debug.sync || self.debug.transfers || self.debug.state;

        for (chunk_index, chunk) in steps.chunks(TRAINING_STEP_BATCH_CAPACITY).enumerate() {
            let chunk_started = debug_chunks.then(Instant::now);
            let base_tick = self.current_tick;
            let build_started = debug_chunks.then(Instant::now);
            let mut device_steps = Vec::with_capacity(chunk.len());
            let mut metadata = Vec::with_capacity(chunk.len());
            let mut learned_steps = 0u64;
            let mut context_learning = false;

            for (record_index, &(symbol, target)) in chunk.iter().enumerate() {
                if !is_input_symbol(symbol) {
                    return Err(LeoError::cuda(format!("invalid input symbol: {symbol}")));
                }
                let target_index = target
                    .map(|value| {
                        output_symbol_to_index(value).ok_or_else(|| {
                            LeoError::cuda(format!("invalid output target: {value}"))
                        })
                    })
                    .transpose()?;
                let tick = base_tick.saturating_add(record_index as u64);
                let context_enabled = matches!(permission, Permission::Frozen)
                    || context_survives_dropout(
                        self.model.config.model.seed,
                        tick,
                        symbol,
                        self.model.config.context.dropout_rate,
                    );
                let learned = target_index.is_some() && strength > 0.0;
                let supervised_strength = target_index.map_or(0.0, |target_output_index| {
                    let target_weight = if target_output_index == END_DOCUMENT_OUTPUT_INDEX {
                        self.model.config.learning.end_document_weight
                    } else {
                        1.0
                    };
                    strength * target_weight
                });
                if learned {
                    learned_steps = learned_steps.saturating_add(1);
                    context_learning |= context_enabled;
                }
                device_steps.push(CudaPersistentStep {
                    symbol,
                    target_index: target_index.map(|index| index as i32).unwrap_or(-1),
                    context_enabled: u32::from(context_enabled),
                    supervised_strength,
                });
                metadata.push((symbol, target_index, context_enabled, learned));
            }
            let build_ms = build_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let upload_started = debug_chunks.then(Instant::now);
            let pointer_table = self.persistent_pointer_table();
            self.copy_to_device(self.buffers.persistent_pointer_table, &pointer_table)?;
            self.copy_to_device(self.buffers.persistent_steps, &device_steps)?;
            let upload_ms = upload_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let mut pointers = self.buffers.persistent_pointer_table.pointer;
            let mut persistent_steps = self.buffers.persistent_steps.pointer;
            let mut step_count = as_u32("persistent step count", chunk.len())?;
            let mut tick = base_tick;
            let mut learning = u32::from(learning_trace);
            let mut raw_strength = strength;
            let model_blocks = as_u32(
                "replay training model block count",
                self.model.config.model.block_count,
            )?;
            let fast_metrics_requested = learning_trace && !self.full_step_metrics;
            let production_replay_capacity = if fast_metrics_requested {
                self.shared.replay_fast_grid_blocks
            } else {
                self.shared.replay_grid_blocks
            };
            let replay_blocks = self
                .replay_grid_override
                .unwrap_or(production_replay_capacity)
                .min(production_replay_capacity.max(1))
                .min(model_blocks)
                .max(1);
            let cooperative_replay = self.replay_cooperative_enabled
                && self.shared.cooperative_launch
                && production_replay_capacity > 0;
            let profile_stride = self.replay_profile.target_stride;
            let profile_requested = profile_stride
                .is_some_and(|stride| self.replay_profile_target_sequence % stride == 0);
            self.replay_profile_target_sequence =
                self.replay_profile_target_sequence.saturating_add(1);
            let profile_sampled = profile_requested
                && cooperative_replay
                && self.shared.replay_profile_grid_blocks >= replay_blocks;
            let use_fast_records = fast_metrics_requested && cooperative_replay && !profile_sampled;

            if profile_requested && !profile_sampled {
                eprintln!(
                    "{{\"event\":\"cuda_replay_kernel_profile_skipped\",\"reason\":\"profiled_kernel_cannot_match_production_grid\",\"requested_grid_blocks\":{},\"normal_capacity_blocks\":{},\"profiled_capacity_blocks\":{},\"step_count\":{}}}",
                    replay_blocks,
                    self.shared.replay_grid_blocks,
                    self.shared.replay_profile_grid_blocks,
                    chunk.len(),
                );
            }
            if profile_sampled {
                self.clear_replay_profile_counters()?;
            }

            if self.debug.launches {
                let kernel = if profile_sampled {
                    "leo_train_cooperative_profiled"
                } else if use_fast_records {
                    "leo_train_cooperative_fast"
                } else if cooperative_replay {
                    "leo_train_cooperative"
                } else {
                    "leo_train_persistent"
                };
                eprintln!(
                    "{{\"event\":\"cuda_debug_launch\",\"scope\":\"replay_targets\",\"kernel\":\"{}\",\"grid_blocks\":{},\"threads\":{},\"step_count\":{},\"learned_steps\":{},\"base_tick\":{},\"profiled\":{},\"cooperative_replay\":{}}}",
                    kernel,
                    if cooperative_replay { replay_blocks } else { self.shared.persistent_grid_blocks.max(1) },
                    PERSISTENT_THREADS,
                    chunk.len(),
                    learned_steps,
                    base_tick,
                    profile_sampled,
                    cooperative_replay,
                );
            }

            let kernel_started = debug_chunks.then(Instant::now);
            if profile_sampled {
                let mut profile_counters = self.buffers.replay_profile_counters.pointer;
                let mut parameters = [
                    param(&mut pointers),
                    param(&mut persistent_steps),
                    param(&mut step_count),
                    param(&mut tick),
                    param(&mut learning),
                    param(&mut raw_strength),
                    param(&mut profile_counters),
                ];
                self.launch_cooperative_exact(
                    self.shared.kernels.train_cooperative_profiled,
                    replay_blocks,
                    PERSISTENT_THREADS,
                    &mut parameters,
                )?;
            } else {
                let mut parameters = [
                    param(&mut pointers),
                    param(&mut persistent_steps),
                    param(&mut step_count),
                    param(&mut tick),
                    param(&mut learning),
                    param(&mut raw_strength),
                ];
                if cooperative_replay {
                    self.launch_cooperative_exact(
                        if use_fast_records {
                            self.shared.kernels.train_cooperative_fast
                        } else {
                            self.shared.kernels.train_cooperative
                        },
                        replay_blocks,
                        PERSISTENT_THREADS,
                        &mut parameters,
                    )?;
                } else {
                    self.launch_cooperative_exact(
                        self.shared.kernels.train_persistent,
                        self.shared.persistent_grid_blocks.max(1),
                        PERSISTENT_THREADS,
                        &mut parameters,
                    )?;
                }
            }
            self.synchronize()?;
            let kernel_sync_ms = kernel_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            if profile_sampled {
                self.emit_replay_target_profile(replay_blocks, profile_stride.unwrap_or(1))?;
            }

            let download_started = debug_chunks.then(Instant::now);
            let fast_records = if use_fast_records {
                let mut records = vec![CudaFastTrainingStepRecord::default(); chunk.len()];
                self.copy_from_device_prefix(self.buffers.training_step_records, &mut records)?;
                Some(records)
            } else {
                None
            };
            let full_records = if use_fast_records {
                None
            } else {
                let mut records = vec![CudaTrainingStepRecord::default(); chunk.len()];
                self.copy_from_device_prefix(self.buffers.training_step_records, &mut records)?;
                Some(records)
            };
            let download_ms = download_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            if learning_trace && chunk.len() % 2 == 1 {
                std::mem::swap(
                    &mut self.buffers.recurrent_eligible_list,
                    &mut self.buffers.recurrent_next_eligible_list,
                );
                std::mem::swap(
                    &mut self.buffers.recurrent_eligible_count,
                    &mut self.buffers.recurrent_next_eligible_count,
                );
                std::mem::swap(
                    &mut self.buffers.input_eligible_list,
                    &mut self.buffers.input_next_eligible_list,
                );
                std::mem::swap(
                    &mut self.buffers.input_eligible_count,
                    &mut self.buffers.input_next_eligible_count,
                );
            }
            self.current_tick = self.current_tick.saturating_add(chunk.len() as u64);
            self.model.parameter_revision =
                self.model.parameter_revision.saturating_add(learned_steps);
            if learned_steps > 0 {
                self.output_bias_dirty_since_sync = true;
                if context_learning {
                    self.context_output_dirty_since_sync = true;
                }
            }

            if let Some(records) = fast_records {
                for (record, (symbol, target_index, context_enabled, _learned)) in
                    records.into_iter().zip(metadata)
                {
                    if record.error_code != 0 {
                        self.model.statistics.numerical_rejections =
                            self.model.statistics.numerical_rejections.saturating_add(1);
                        return Err(LeoError::cuda(format!(
                            "CUDA persistent learning update rejected a non-finite value (code {})",
                            record.error_code
                        )));
                    }
                    let metrics =
                        self.fast_training_record_to_metrics(record, target_index, context_enabled);
                    self.persistent_document_activity = true;
                    self.update_statistics(
                        symbol,
                        metrics.loss,
                        metrics.active_neurons,
                        metrics.emitted_events,
                    );
                    all_metrics.push(metrics);
                }
            } else if let Some(records) = full_records {
                for (record, (symbol, target_index, context_enabled, learned)) in
                    records.into_iter().zip(metadata)
                {
                    if record.error_code != 0 {
                        self.model.statistics.numerical_rejections =
                            self.model.statistics.numerical_rejections.saturating_add(1);
                        return Err(LeoError::cuda(format!(
                            "CUDA persistent learning update rejected a non-finite value (code {})",
                            record.error_code
                        )));
                    }
                    let metrics = self.training_record_to_metrics(
                        record,
                        target_index,
                        context_enabled,
                        learned,
                    );
                    if learning_trace {
                        self.persistent_document_activity = true;
                        self.update_statistics(
                            symbol,
                            metrics.loss,
                            metrics.active_neurons,
                            metrics.emitted_events,
                        );
                    }
                    all_metrics.push(metrics);
                }
            }

            if debug_chunks {
                let total_ms = chunk_started
                    .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                eprintln!(
                    "{{\"event\":\"cuda_debug_chunk\",\"scope\":\"replay_targets\",\"chunk_index\":{},\"steps\":{},\"learned_steps\":{},\"base_tick\":{},\"end_tick\":{},\"grid_blocks\":{},\"cooperative_replay\":{},\"build_ms\":{},\"upload_ms\":{},\"kernel_sync_ms\":{},\"download_ms\":{},\"total_ms\":{},\"parameter_revision\":{}}}",
                    chunk_index,
                    chunk.len(),
                    learned_steps,
                    base_tick,
                    self.current_tick,
                    if cooperative_replay { replay_blocks } else { self.shared.persistent_grid_blocks.max(1) },
                    cooperative_replay,
                    build_ms,
                    upload_ms,
                    kernel_sync_ms,
                    download_ms,
                    total_ms,
                    self.model.parameter_revision,
                );
            }
        }

        Ok(all_metrics)
    }

    fn persistent_pointer_table(&self) -> [CuDevicePtr; PERSISTENT_POINTER_COUNT] {
        let b = self.buffers;
        let mut pointers = [0; PERSISTENT_POINTER_COUNT];
        pointers[PersistentPointer::Config.index()] = b.config.pointer;
        pointers[PersistentPointer::Threshold.index()] = b.threshold.pointer;
        pointers[PersistentPointer::Excitability.index()] = b.excitability.pointer;
        pointers[PersistentPointer::NeuronType.index()] = b.neuron_type.pointer;
        pointers[PersistentPointer::RecurrentTarget.index()] = b.recurrent_target.pointer;
        pointers[PersistentPointer::RecurrentBranch.index()] = b.recurrent_branch.pointer;
        pointers[PersistentPointer::RecurrentDelay.index()] = b.recurrent_delay.pointer;
        pointers[PersistentPointer::RecurrentWeight.index()] = b.recurrent_weight.pointer;
        pointers[PersistentPointer::InputTarget.index()] = b.input_target.pointer;
        pointers[PersistentPointer::InputBranch.index()] = b.input_branch.pointer;
        pointers[PersistentPointer::InputWeight.index()] = b.input_weight.pointer;
        pointers[PersistentPointer::OutputWeight.index()] = b.output_weight.pointer;
        pointers[PersistentPointer::OutputBias.index()] = b.output_bias.pointer;
        pointers[PersistentPointer::ContextKeys.index()] = b.context_keys.pointer;
        pointers[PersistentPointer::ContextEmbeddings.index()] = b.context_embeddings.pointer;
        pointers[PersistentPointer::ContextObservations.index()] = b.context_observations.pointer;
        pointers[PersistentPointer::ContextOutputWeight.index()] = b.context_output_weight.pointer;
        pointers[PersistentPointer::Membrane.index()] = b.membrane.pointer;
        pointers[PersistentPointer::Activation.index()] = b.activation.pointer;
        pointers[PersistentPointer::Fatigue.index()] = b.fatigue.pointer;
        pointers[PersistentPointer::RefractoryUntil.index()] = b.refractory_until.pointer;
        pointers[PersistentPointer::Branches.index()] = b.branches.pointer;
        pointers[PersistentPointer::BranchDelta.index()] = b.branch_delta.pointer;
        pointers[PersistentPointer::BranchLastTick.index()] = b.branch_last_tick.pointer;
        pointers[PersistentPointer::NeuronLastTick.index()] = b.neuron_last_tick.pointer;
        pointers[PersistentPointer::AdaptationFast.index()] = b.adaptation_fast.pointer;
        pointers[PersistentPointer::AdaptationMedium.index()] = b.adaptation_medium.pointer;
        pointers[PersistentPointer::AdaptationSlow.index()] = b.adaptation_slow.pointer;
        pointers[PersistentPointer::TouchedEpoch.index()] = b.touched_epoch.pointer;
        pointers[PersistentPointer::TouchedList.index()] = b.touched_list.pointer;
        pointers[PersistentPointer::TouchedCount.index()] = b.touched_count.pointer;
        pointers[PersistentPointer::SelectedEpoch.index()] = b.selected_epoch.pointer;
        pointers[PersistentPointer::Surrogate.index()] = b.surrogate.pointer;
        pointers[PersistentPointer::CandidateActivation.index()] = b.candidate_activation.pointer;
        pointers[PersistentPointer::Active.index()] = b.active.pointer;
        pointers[PersistentPointer::ActiveValue.index()] = b.active_value.pointer;
        pointers[PersistentPointer::ActiveCount.index()] = b.active_count.pointer;
        pointers[PersistentPointer::BlockWinnerNeuron.index()] = b.block_winner_neuron.pointer;
        pointers[PersistentPointer::BlockWinnerValue.index()] = b.block_winner_value.pointer;
        pointers[PersistentPointer::BlockCutoff.index()] = b.block_cutoff.pointer;
        pointers[PersistentPointer::PopulationCutoff.index()] = b.population_cutoff.pointer;
        pointers[PersistentPointer::PopulationInhibition.index()] = b.population_inhibition.pointer;
        pointers[PersistentPointer::RecBranchSensitivity.index()] =
            b.recurrent_branch_sensitivity.pointer;
        pointers[PersistentPointer::RecMembraneSensitivity.index()] =
            b.recurrent_membrane_sensitivity.pointer;
        pointers[PersistentPointer::RecFatigueSensitivity.index()] =
            b.recurrent_fatigue_sensitivity.pointer;
        pointers[PersistentPointer::RecAdaptationFastSensitivity.index()] =
            b.recurrent_adaptation_fast_sensitivity.pointer;
        pointers[PersistentPointer::RecAdaptationMediumSensitivity.index()] =
            b.recurrent_adaptation_medium_sensitivity.pointer;
        pointers[PersistentPointer::RecAdaptationSlowSensitivity.index()] =
            b.recurrent_adaptation_slow_sensitivity.pointer;
        pointers[PersistentPointer::RecEligibility.index()] = b.recurrent_eligibility.pointer;
        pointers[PersistentPointer::RecLastTick.index()] = b.recurrent_last_tick.pointer;
        pointers[PersistentPointer::RecEligibleMark.index()] = b.recurrent_eligible_mark.pointer;
        pointers[PersistentPointer::RecEligibleList.index()] = b.recurrent_eligible_list.pointer;
        pointers[PersistentPointer::RecEligibleCount.index()] = b.recurrent_eligible_count.pointer;
        pointers[PersistentPointer::RecNextEligibleList.index()] =
            b.recurrent_next_eligible_list.pointer;
        pointers[PersistentPointer::RecNextEligibleCount.index()] =
            b.recurrent_next_eligible_count.pointer;
        pointers[PersistentPointer::InputBranchSensitivity.index()] =
            b.input_branch_sensitivity.pointer;
        pointers[PersistentPointer::InputMembraneSensitivity.index()] =
            b.input_membrane_sensitivity.pointer;
        pointers[PersistentPointer::InputFatigueSensitivity.index()] =
            b.input_fatigue_sensitivity.pointer;
        pointers[PersistentPointer::InputAdaptationFastSensitivity.index()] =
            b.input_adaptation_fast_sensitivity.pointer;
        pointers[PersistentPointer::InputAdaptationMediumSensitivity.index()] =
            b.input_adaptation_medium_sensitivity.pointer;
        pointers[PersistentPointer::InputAdaptationSlowSensitivity.index()] =
            b.input_adaptation_slow_sensitivity.pointer;
        pointers[PersistentPointer::InputEligibility.index()] = b.input_eligibility.pointer;
        pointers[PersistentPointer::InputLastTick.index()] = b.input_last_tick.pointer;
        pointers[PersistentPointer::InputEligibleMark.index()] = b.input_eligible_mark.pointer;
        pointers[PersistentPointer::InputEligibleList.index()] = b.input_eligible_list.pointer;
        pointers[PersistentPointer::InputEligibleCount.index()] = b.input_eligible_count.pointer;
        pointers[PersistentPointer::InputNextEligibleList.index()] =
            b.input_next_eligible_list.pointer;
        pointers[PersistentPointer::InputNextEligibleCount.index()] =
            b.input_next_eligible_count.pointer;
        pointers[PersistentPointer::RingCount.index()] = b.ring_count.pointer;
        pointers[PersistentPointer::RingSource.index()] = b.ring_source.pointer;
        pointers[PersistentPointer::RingActivation.index()] = b.ring_activation.pointer;
        pointers[PersistentPointer::RingWeight.index()] = b.ring_weight.pointer;
        pointers[PersistentPointer::ContextHistory.index()] = b.context_history.pointer;
        pointers[PersistentPointer::ContextHistoryCount.index()] = b.context_history_count.pointer;
        pointers[PersistentPointer::ActiveContextSlots.index()] = b.active_context_slots.pointer;
        pointers[PersistentPointer::ActiveContextScales.index()] = b.active_context_scales.pointer;
        pointers[PersistentPointer::ActiveContextCount.index()] = b.active_context_count.pointer;
        pointers[PersistentPointer::ContextLatent.index()] = b.context_latent.pointer;
        pointers[PersistentPointer::ContextGradient.index()] = b.context_gradient.pointer;
        pointers[PersistentPointer::NeuralLogits.index()] = b.neural_logits.pointer;
        pointers[PersistentPointer::Logits.index()] = b.logits.pointer;
        pointers[PersistentPointer::Probabilities.index()] = b.probabilities.pointer;
        pointers[PersistentPointer::Errors.index()] = b.errors.pointer;
        pointers[PersistentPointer::LearningDestinationEpoch.index()] =
            b.learning_destination_epoch.pointer;
        pointers[PersistentPointer::LearningDestinationList.index()] =
            b.learning_destination_list.pointer;
        pointers[PersistentPointer::LearningDestinationCount.index()] =
            b.learning_destination_count.pointer;
        pointers[PersistentPointer::LearningSignal.index()] = b.learning_signal.pointer;
        pointers[PersistentPointer::Counters.index()] = b.counters.pointer;
        pointers[PersistentPointer::ErrorFlag.index()] = b.error_flag.pointer;
        pointers[PersistentPointer::TrainingStepRecords.index()] = b.training_step_records.pointer;
        pointers[PersistentPointer::ChangedThresholdMarks.index()] =
            b.changed_threshold_marks.pointer;
        pointers[PersistentPointer::ChangedThresholdList.index()] =
            b.changed_threshold_list.pointer;
        pointers[PersistentPointer::ChangedThresholdCount.index()] =
            b.changed_threshold_count.pointer;
        pointers[PersistentPointer::ChangedRecurrentMarks.index()] =
            b.changed_recurrent_marks.pointer;
        pointers[PersistentPointer::ChangedRecurrentList.index()] =
            b.changed_recurrent_list.pointer;
        pointers[PersistentPointer::ChangedRecurrentCount.index()] =
            b.changed_recurrent_count.pointer;
        pointers[PersistentPointer::ChangedInputMarks.index()] = b.changed_input_marks.pointer;
        pointers[PersistentPointer::ChangedInputList.index()] = b.changed_input_list.pointer;
        pointers[PersistentPointer::ChangedInputCount.index()] = b.changed_input_count.pointer;
        pointers[PersistentPointer::ChangedOutputMarks.index()] = b.changed_output_marks.pointer;
        pointers[PersistentPointer::ChangedOutputList.index()] = b.changed_output_list.pointer;
        pointers[PersistentPointer::ChangedOutputCount.index()] = b.changed_output_count.pointer;
        pointers[PersistentPointer::ChangedContextMarks.index()] = b.changed_context_marks.pointer;
        pointers[PersistentPointer::ChangedContextList.index()] = b.changed_context_list.pointer;
        pointers[PersistentPointer::ChangedContextCount.index()] = b.changed_context_count.pointer;
        pointers
    }

    fn launch_capture_training_step(
        &self,
        target_index: i32,
        record_index: usize,
    ) -> LeoResult<()> {
        let b = self.buffers;
        let mut probabilities = b.probabilities.pointer;
        let mut neural_logits = b.neural_logits.pointer;
        let mut active_count = b.active_count.pointer;
        let mut population_inhibition = b.population_inhibition.pointer;
        let mut counters = b.counters.pointer;
        let mut error_flag = b.error_flag.pointer;
        let mut target = target_index;
        let mut records = b.training_step_records.pointer;
        let mut index = as_u32("training step record index", record_index)?;
        let mut parameters = [
            param(&mut probabilities),
            param(&mut neural_logits),
            param(&mut active_count),
            param(&mut population_inhibition),
            param(&mut counters),
            param(&mut error_flag),
            param(&mut target),
            param(&mut records),
            param(&mut index),
        ];
        self.launch_exact(
            self.shared.kernels.capture_training_step,
            1,
            1,
            &mut parameters,
        )
    }

    fn training_record_to_metrics(
        &self,
        record: CudaTrainingStepRecord,
        target_index: Option<usize>,
        context_applied: bool,
        learned: bool,
    ) -> StepMetrics {
        let predicted_symbol = output_index_to_symbol(record.predicted_index as usize)
            .expect("CUDA probability index must map to an output symbol");
        let loss = target_index.map(|_| record.loss);
        let context_loss_gain = target_index
            .map(|_| record.neural_loss - record.loss)
            .unwrap_or(0.0);
        let active_count = record.active_count as usize;
        let context_cells = record.context_cells as usize;
        let output_madds = active_count
            .saturating_mul(OUTPUT_CLASSES)
            .saturating_add(context_cells.saturating_mul(self.model.config.context.embedding_dim))
            .saturating_add(if context_cells == 0 {
                0
            } else {
                self.model
                    .config
                    .context
                    .embedding_dim
                    .saturating_mul(OUTPUT_CLASSES)
            });
        let mean_abs_eligibility = if learned && record.learning_eligibility_count > 0 {
            record.learning_eligibility_abs_sum / record.learning_eligibility_count as f32
        } else {
            0.0
        };
        StepMetrics {
            loss,
            predicted_symbol,
            active_neurons: active_count,
            emitted_events: active_count.saturating_mul(self.model.recurrent.capacity_per_neuron),
            suprathreshold_neurons: record.suprathreshold as usize,
            block_selected_neurons: record.block_selected as usize,
            global_cap_clipped_neurons: record.global_clipped as usize,
            population_inhibition: record.population_inhibition,
            context_cells,
            context_probes: record.context_probes as usize,
            context_applied,
            context_loss_gain,
            output_madds,
            eligible_recurrent_synapses: record.eligible_recurrent as usize,
            eligible_input_synapses: record.eligible_input as usize,
            mean_abs_eligibility,
        }
    }

    fn fast_training_record_to_metrics(
        &self,
        record: CudaFastTrainingStepRecord,
        target_index: Option<usize>,
        context_applied: bool,
    ) -> StepMetrics {
        let active_count = record.active_count as usize;
        StepMetrics {
            loss: target_index.map(|_| record.loss),
            active_neurons: active_count,
            emitted_events: active_count.saturating_mul(self.model.recurrent.capacity_per_neuron),
            context_applied,
            ..StepMetrics::default()
        }
    }

    fn launch_step_prefix(
        &mut self,
        symbol: u32,
        learning_trace: bool,
        context_enabled: bool,
        allocate_context: bool,
    ) -> LeoResult<()> {
        let b = self.buffers;
        let neuron_count = self.model.neuron_count();
        let recurrent_count = self.model.recurrent.weight.len();
        let input_count = self.model.input.weights.len();
        let active_capacity = self.model.config.model.max_active_global;
        let config = b.config.pointer;
        let mut cfg = config;
        let mut tick = self.current_tick;
        let mut active = b.active.pointer;
        let mut active_count = b.active_count.pointer;
        let mut activation = b.activation.pointer;
        let mut touched_epoch = b.touched_epoch.pointer;
        let mut destination_epoch = b.learning_destination_epoch.pointer;
        let mut counters = b.counters.pointer;
        let mut start_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut active),
            param(&mut active_count),
            param(&mut activation),
            param(&mut touched_epoch),
            param(&mut destination_epoch),
            param(&mut counters),
        ];
        self.launch(
            self.shared.kernels.start_tick,
            neuron_count.max(active_capacity),
            THREADS,
            &mut start_params,
        )?;

        let mut recurrent_target = b.recurrent_target.pointer;
        let mut recurrent_branch = b.recurrent_branch.pointer;
        let mut recurrent_delay = b.recurrent_delay.pointer;
        let mut ring_count = b.ring_count.pointer;
        let mut ring_source = b.ring_source.pointer;
        let mut ring_activation = b.ring_activation.pointer;
        let mut ring_weight = b.ring_weight.pointer;
        let mut branch_delta = b.branch_delta.pointer;
        let mut learning = learning_trace;
        let mut rec_bs = b.recurrent_branch_sensitivity.pointer;
        let mut rec_ms = b.recurrent_membrane_sensitivity.pointer;
        let mut rec_fs = b.recurrent_fatigue_sensitivity.pointer;
        let mut rec_af = b.recurrent_adaptation_fast_sensitivity.pointer;
        let mut rec_am = b.recurrent_adaptation_medium_sensitivity.pointer;
        let mut rec_as = b.recurrent_adaptation_slow_sensitivity.pointer;
        let mut rec_e = b.recurrent_eligibility.pointer;
        let mut rec_last = b.recurrent_last_tick.pointer;
        let mut rec_mark = b.recurrent_eligible_mark.pointer;
        let mut rec_eligible_list = b.recurrent_eligible_list.pointer;
        let mut rec_eligible_count = b.recurrent_eligible_count.pointer;
        let mut deliver_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut recurrent_target),
            param(&mut recurrent_branch),
            param(&mut recurrent_delay),
            param(&mut ring_count),
            param(&mut ring_source),
            param(&mut ring_activation),
            param(&mut ring_weight),
            param(&mut branch_delta),
            param(&mut touched_epoch),
            param(&mut learning),
            param(&mut rec_bs),
            param(&mut rec_ms),
            param(&mut rec_fs),
            param(&mut rec_af),
            param(&mut rec_am),
            param(&mut rec_as),
            param(&mut rec_e),
            param(&mut rec_last),
            param(&mut rec_mark),
            param(&mut rec_eligible_list),
            param(&mut rec_eligible_count),
        ];
        self.launch(
            self.shared.kernels.deliver_events,
            active_capacity
                .saturating_mul(self.model.config.model.synapses_per_neuron)
                .saturating_mul(4),
            THREADS,
            &mut deliver_params,
        )?;

        let mut input_target = b.input_target.pointer;
        let mut input_branch = b.input_branch.pointer;
        let mut input_weight = b.input_weight.pointer;
        let mut input_bs = b.input_branch_sensitivity.pointer;
        let mut input_ms = b.input_membrane_sensitivity.pointer;
        let mut input_fs = b.input_fatigue_sensitivity.pointer;
        let mut input_af = b.input_adaptation_fast_sensitivity.pointer;
        let mut input_am = b.input_adaptation_medium_sensitivity.pointer;
        let mut input_as = b.input_adaptation_slow_sensitivity.pointer;
        let mut input_e = b.input_eligibility.pointer;
        let mut input_last = b.input_last_tick.pointer;
        let mut input_mark = b.input_eligible_mark.pointer;
        let mut input_eligible_list = b.input_eligible_list.pointer;
        let mut input_eligible_count = b.input_eligible_count.pointer;
        let mut symbol_value = symbol;
        let mut inject_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut symbol_value),
            param(&mut input_target),
            param(&mut input_branch),
            param(&mut input_weight),
            param(&mut branch_delta),
            param(&mut touched_epoch),
            param(&mut learning),
            param(&mut input_bs),
            param(&mut input_ms),
            param(&mut input_fs),
            param(&mut input_af),
            param(&mut input_am),
            param(&mut input_as),
            param(&mut input_e),
            param(&mut input_last),
            param(&mut input_mark),
            param(&mut input_eligible_list),
            param(&mut input_eligible_count),
        ];
        self.launch(
            self.shared.kernels.inject_symbol,
            self.model.config.model.input_fanout,
            THREADS,
            &mut inject_params,
        )?;

        let mut enabled = context_enabled;
        let mut allocate = allocate_context;
        let mut history = b.context_history.pointer;
        let mut history_count = b.context_history_count.pointer;
        let mut context_keys = b.context_keys.pointer;
        let mut context_embeddings = b.context_embeddings.pointer;
        let mut context_observations = b.context_observations.pointer;
        let mut context_slots = b.active_context_slots.pointer;
        let mut context_scales = b.active_context_scales.pointer;
        let mut context_count = b.active_context_count.pointer;
        let mut changed_context_marks = b.changed_context_marks.pointer;
        let mut changed_context_list = b.changed_context_list.pointer;
        let mut changed_context_count = b.changed_context_count.pointer;
        let mut context_params = [
            param(&mut cfg),
            param(&mut symbol_value),
            param(&mut enabled),
            param(&mut allocate),
            param(&mut history),
            param(&mut history_count),
            param(&mut context_keys),
            param(&mut context_embeddings),
            param(&mut context_observations),
            param(&mut context_slots),
            param(&mut context_scales),
            param(&mut context_count),
            param(&mut changed_context_marks),
            param(&mut changed_context_list),
            param(&mut changed_context_count),
            param(&mut counters),
        ];
        self.launch_exact(
            self.shared.kernels.context_resolve,
            1,
            1,
            &mut context_params,
        )?;

        let mut threshold = b.threshold.pointer;
        let mut excitability = b.excitability.pointer;
        let mut membrane = b.membrane.pointer;
        let mut fatigue = b.fatigue.pointer;
        let mut refractory = b.refractory_until.pointer;
        let mut branches = b.branches.pointer;
        let mut branch_last = b.branch_last_tick.pointer;
        let mut neuron_last = b.neuron_last_tick.pointer;
        let mut adaptation_fast = b.adaptation_fast.pointer;
        let mut adaptation_medium = b.adaptation_medium.pointer;
        let mut adaptation_slow = b.adaptation_slow.pointer;
        let mut population_inhibition = b.population_inhibition.pointer;
        let mut candidate = b.candidate_activation.pointer;
        let mut block_winner_neuron = b.block_winner_neuron.pointer;
        let mut block_winner_value = b.block_winner_value.pointer;
        let mut block_cutoff = b.block_cutoff.pointer;
        let mut select_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut threshold),
            param(&mut excitability),
            param(&mut membrane),
            param(&mut activation),
            param(&mut fatigue),
            param(&mut refractory),
            param(&mut branches),
            param(&mut branch_delta),
            param(&mut branch_last),
            param(&mut neuron_last),
            param(&mut adaptation_fast),
            param(&mut adaptation_medium),
            param(&mut adaptation_slow),
            param(&mut touched_epoch),
            param(&mut population_inhibition),
            param(&mut candidate),
            param(&mut block_winner_neuron),
            param(&mut block_winner_value),
            param(&mut block_cutoff),
            param(&mut counters),
        ];
        self.launch_exact(
            self.shared.kernels.select_blocks,
            as_u32("block count", self.model.config.model.block_count)?,
            as_u32(
                "neurons per block",
                self.model.config.model.neurons_per_block,
            )?,
            &mut select_params,
        )?;

        let mut active_value = b.active_value.pointer;
        let mut selected_epoch = b.selected_epoch.pointer;
        let mut population_cutoff = b.population_cutoff.pointer;
        let mut global_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut block_winner_neuron),
            param(&mut block_winner_value),
            param(&mut active),
            param(&mut active_value),
            param(&mut active_count),
            param(&mut activation),
            param(&mut selected_epoch),
            param(&mut destination_epoch),
            param(&mut population_cutoff),
            param(&mut population_inhibition),
            param(&mut counters),
        ];
        self.launch_exact(
            self.shared.kernels.select_global,
            1,
            GLOBAL_SELECTION_THREADS,
            &mut global_params,
        )?;

        let mut surrogate = b.surrogate.pointer;
        let mut surrogate_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut threshold),
            param(&mut membrane),
            param(&mut refractory),
            param(&mut touched_epoch),
            param(&mut selected_epoch),
            param(&mut block_cutoff),
            param(&mut population_cutoff),
            param(&mut surrogate),
        ];
        self.launch(
            self.shared.kernels.cache_surrogate,
            neuron_count,
            THREADS,
            &mut surrogate_params,
        )?;

        if learning_trace {
            let mut rec_next_eligible_list = b.recurrent_next_eligible_list.pointer;
            let mut rec_next_eligible_count = b.recurrent_next_eligible_count.pointer;
            let mut rec_elig_params = [
                param(&mut cfg),
                param(&mut tick),
                param(&mut recurrent_target),
                param(&mut recurrent_branch),
                param(&mut excitability),
                param(&mut branches),
                param(&mut refractory),
                param(&mut touched_epoch),
                param(&mut selected_epoch),
                param(&mut surrogate),
                param(&mut rec_bs),
                param(&mut rec_ms),
                param(&mut rec_fs),
                param(&mut rec_af),
                param(&mut rec_am),
                param(&mut rec_as),
                param(&mut rec_e),
                param(&mut rec_last),
                param(&mut rec_mark),
                param(&mut rec_eligible_list),
                param(&mut rec_eligible_count),
                param(&mut rec_next_eligible_list),
                param(&mut rec_next_eligible_count),
                param(&mut destination_epoch),
                param(&mut counters),
            ];
            self.memset_zero_async(b.recurrent_next_eligible_count)?;
            self.launch(
                self.shared.kernels.update_recurrent_eligibility,
                recurrent_count.min(SPARSE_LIST_WORK_ITEMS),
                THREADS,
                &mut rec_elig_params,
            )?;
            let mut input_next_eligible_list = b.input_next_eligible_list.pointer;
            let mut input_next_eligible_count = b.input_next_eligible_count.pointer;
            let mut input_elig_params = [
                param(&mut cfg),
                param(&mut tick),
                param(&mut input_target),
                param(&mut input_branch),
                param(&mut excitability),
                param(&mut branches),
                param(&mut refractory),
                param(&mut touched_epoch),
                param(&mut selected_epoch),
                param(&mut surrogate),
                param(&mut input_bs),
                param(&mut input_ms),
                param(&mut input_fs),
                param(&mut input_af),
                param(&mut input_am),
                param(&mut input_as),
                param(&mut input_e),
                param(&mut input_last),
                param(&mut input_mark),
                param(&mut input_eligible_list),
                param(&mut input_eligible_count),
                param(&mut input_next_eligible_list),
                param(&mut input_next_eligible_count),
                param(&mut destination_epoch),
                param(&mut counters),
            ];
            self.memset_zero_async(b.input_next_eligible_count)?;
            self.launch(
                self.shared.kernels.update_input_eligibility,
                input_count.min(SPARSE_LIST_WORK_ITEMS),
                THREADS,
                &mut input_elig_params,
            )?;
            std::mem::swap(
                &mut self.buffers.recurrent_eligible_list,
                &mut self.buffers.recurrent_next_eligible_list,
            );
            std::mem::swap(
                &mut self.buffers.recurrent_eligible_count,
                &mut self.buffers.recurrent_next_eligible_count,
            );
            std::mem::swap(
                &mut self.buffers.input_eligible_list,
                &mut self.buffers.input_next_eligible_list,
            );
            std::mem::swap(
                &mut self.buffers.input_eligible_count,
                &mut self.buffers.input_next_eligible_count,
            );
        }

        let mut recurrent_weight = b.recurrent_weight.pointer;
        let mut post_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut threshold),
            param(&mut recurrent_weight),
            param(&mut active),
            param(&mut active_value),
            param(&mut active_count),
            param(&mut membrane),
            param(&mut fatigue),
            param(&mut refractory),
            param(&mut adaptation_fast),
            param(&mut adaptation_medium),
            param(&mut adaptation_slow),
            param(&mut ring_count),
            param(&mut ring_source),
            param(&mut ring_activation),
            param(&mut ring_weight),
        ];
        self.launch(
            self.shared.kernels.post_and_emit,
            active_capacity,
            THREADS,
            &mut post_params,
        )?;
        Ok(())
    }

    fn launch_forward(&self, target_index: i32) -> LeoResult<()> {
        let b = self.buffers;
        let mut cfg = b.config.pointer;
        let mut output_weight = b.output_weight.pointer;
        let mut output_bias = b.output_bias.pointer;
        let mut context_embeddings = b.context_embeddings.pointer;
        let mut context_output_weight = b.context_output_weight.pointer;
        let mut active = b.active.pointer;
        let mut active_value = b.active_value.pointer;
        let mut active_count = b.active_count.pointer;
        let mut context_slots = b.active_context_slots.pointer;
        let mut context_scales = b.active_context_scales.pointer;
        let mut context_count = b.active_context_count.pointer;
        let mut target = target_index;
        let mut neural_logits = b.neural_logits.pointer;
        let mut logits = b.logits.pointer;
        let mut probabilities = b.probabilities.pointer;
        let mut errors = b.errors.pointer;
        let mut context_latent = b.context_latent.pointer;
        let mut context_gradient = b.context_gradient.pointer;
        let mut parameters = [
            param(&mut cfg),
            param(&mut output_weight),
            param(&mut output_bias),
            param(&mut context_embeddings),
            param(&mut context_output_weight),
            param(&mut active),
            param(&mut active_value),
            param(&mut active_count),
            param(&mut context_slots),
            param(&mut context_scales),
            param(&mut context_count),
            param(&mut target),
            param(&mut neural_logits),
            param(&mut logits),
            param(&mut probabilities),
            param(&mut errors),
            param(&mut context_latent),
            param(&mut context_gradient),
        ];
        self.launch_exact(
            self.shared.kernels.forward,
            1,
            FORWARD_THREADS,
            &mut parameters,
        )
    }

    fn launch_learning(
        &mut self,
        strength: f32,
        supervised_strength: f32,
        context_enabled: bool,
    ) -> LeoResult<()> {
        let b = self.buffers;
        let neuron_count = self.model.neuron_count();
        let recurrent_count = self.model.recurrent.weight.len();
        let input_count = self.model.input.weights.len();
        let mut cfg = b.config.pointer;
        let mut tick = self.current_tick;
        let mut output_weight = b.output_weight.pointer;
        let mut errors = b.errors.pointer;
        let mut destination_epoch = b.learning_destination_epoch.pointer;
        let mut learning_signal = b.learning_signal.pointer;
        let mut signal_params = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut output_weight),
            param(&mut errors),
            param(&mut destination_epoch),
            param(&mut learning_signal),
        ];
        self.launch(
            self.shared.kernels.learning_signals,
            neuron_count,
            THREADS,
            &mut signal_params,
        )?;

        let mut output_bias = b.output_bias.pointer;
        let mut active = b.active.pointer;
        let mut active_value = b.active_value.pointer;
        let mut active_count = b.active_count.pointer;
        let mut supervised = supervised_strength;
        let mut changed_output_marks = b.changed_output_marks.pointer;
        let mut changed_output_list = b.changed_output_list.pointer;
        let mut changed_output_count = b.changed_output_count.pointer;
        let mut error_flag = b.error_flag.pointer;
        let output_work = self
            .model
            .config
            .model
            .max_active_global
            .saturating_mul(OUTPUT_CLASSES)
            .saturating_add(OUTPUT_CLASSES);
        let mut output_params = [
            param(&mut cfg),
            param(&mut output_weight),
            param(&mut output_bias),
            param(&mut active),
            param(&mut active_value),
            param(&mut active_count),
            param(&mut errors),
            param(&mut supervised),
            param(&mut changed_output_marks),
            param(&mut changed_output_list),
            param(&mut changed_output_count),
            param(&mut error_flag),
        ];
        self.launch(
            self.shared.kernels.update_output,
            output_work,
            THREADS,
            &mut output_params,
        )?;

        if context_enabled {
            let mut context_output_weight = b.context_output_weight.pointer;
            let mut context_embeddings = b.context_embeddings.pointer;
            let mut context_observations = b.context_observations.pointer;
            let mut context_slots = b.active_context_slots.pointer;
            let mut context_scales = b.active_context_scales.pointer;
            let mut context_count = b.active_context_count.pointer;
            let mut context_latent = b.context_latent.pointer;
            let mut context_gradient = b.context_gradient.pointer;
            let mut changed_context_marks = b.changed_context_marks.pointer;
            let mut changed_context_list = b.changed_context_list.pointer;
            let mut changed_context_count = b.changed_context_count.pointer;
            let context_work = OUTPUT_CLASSES
                .saturating_mul(self.model.config.context.embedding_dim)
                .saturating_add(
                    self.model
                        .config
                        .context
                        .max_order
                        .saturating_mul(self.model.config.context.embedding_dim),
                );
            let mut context_params = [
                param(&mut cfg),
                param(&mut context_output_weight),
                param(&mut context_embeddings),
                param(&mut context_observations),
                param(&mut context_slots),
                param(&mut context_scales),
                param(&mut context_count),
                param(&mut context_latent),
                param(&mut context_gradient),
                param(&mut errors),
                param(&mut supervised),
                param(&mut changed_context_marks),
                param(&mut changed_context_list),
                param(&mut changed_context_count),
                param(&mut error_flag),
            ];
            self.launch(
                self.shared.kernels.update_context,
                context_work,
                THREADS,
                &mut context_params,
            )?;
        }

        let mut recurrent_target = b.recurrent_target.pointer;
        let mut neuron_type = b.neuron_type.pointer;
        let mut recurrent_weight = b.recurrent_weight.pointer;
        let mut rec_e = b.recurrent_eligibility.pointer;
        let mut rec_eligible_list = b.recurrent_eligible_list.pointer;
        let mut rec_eligible_count = b.recurrent_eligible_count.pointer;
        let mut changed_rec_marks = b.changed_recurrent_marks.pointer;
        let mut changed_rec_list = b.changed_recurrent_list.pointer;
        let mut changed_rec_count = b.changed_recurrent_count.pointer;
        let mut counters = b.counters.pointer;
        let mut recurrent_params = [
            param(&mut cfg),
            param(&mut recurrent_target),
            param(&mut neuron_type),
            param(&mut recurrent_weight),
            param(&mut rec_e),
            param(&mut rec_eligible_list),
            param(&mut rec_eligible_count),
            param(&mut learning_signal),
            param(&mut supervised),
            param(&mut changed_rec_marks),
            param(&mut changed_rec_list),
            param(&mut changed_rec_count),
            param(&mut counters),
            param(&mut error_flag),
        ];
        self.launch(
            self.shared.kernels.update_recurrent_weights,
            recurrent_count.min(SPARSE_LIST_WORK_ITEMS),
            THREADS,
            &mut recurrent_params,
        )?;

        let mut input_target = b.input_target.pointer;
        let mut input_weight = b.input_weight.pointer;
        let mut input_e = b.input_eligibility.pointer;
        let mut input_eligible_list = b.input_eligible_list.pointer;
        let mut input_eligible_count = b.input_eligible_count.pointer;
        let mut changed_input_marks = b.changed_input_marks.pointer;
        let mut changed_input_list = b.changed_input_list.pointer;
        let mut changed_input_count = b.changed_input_count.pointer;
        let mut input_params = [
            param(&mut cfg),
            param(&mut input_target),
            param(&mut input_weight),
            param(&mut input_e),
            param(&mut input_eligible_list),
            param(&mut input_eligible_count),
            param(&mut learning_signal),
            param(&mut supervised),
            param(&mut changed_input_marks),
            param(&mut changed_input_list),
            param(&mut changed_input_count),
            param(&mut counters),
            param(&mut error_flag),
        ];
        self.launch(
            self.shared.kernels.update_input_weights,
            input_count.min(SPARSE_LIST_WORK_ITEMS),
            THREADS,
            &mut input_params,
        )?;

        let mut activation = b.activation.pointer;
        let active_capacity = self.model.config.model.max_active_global;
        let mut raw_strength = strength;
        let mut inhibitory_params = [
            param(&mut cfg),
            param(&mut active),
            param(&mut active_value),
            param(&mut active_count),
            param(&mut activation),
            param(&mut neuron_type),
            param(&mut recurrent_target),
            param(&mut recurrent_weight),
            param(&mut raw_strength),
            param(&mut changed_rec_marks),
            param(&mut changed_rec_list),
            param(&mut changed_rec_count),
            param(&mut error_flag),
        ];
        self.launch(
            self.shared.kernels.inhibitory_homeostasis,
            active_capacity.saturating_mul(self.model.config.model.synapses_per_neuron),
            THREADS,
            &mut inhibitory_params,
        )?;
        Ok(())
    }

    fn launch_homeostasis(&self) -> LeoResult<()> {
        let b = self.buffers;
        let mut cfg = b.config.pointer;
        let mut tick = self.current_tick;
        let mut active_count = b.active_count.pointer;
        let mut activation = b.activation.pointer;
        let mut touched_epoch = b.touched_epoch.pointer;
        let mut threshold = b.threshold.pointer;
        let mut changed_marks = b.changed_threshold_marks.pointer;
        let mut changed_list = b.changed_threshold_list.pointer;
        let mut changed_count = b.changed_threshold_count.pointer;
        let mut parameters = [
            param(&mut cfg),
            param(&mut tick),
            param(&mut active_count),
            param(&mut activation),
            param(&mut touched_epoch),
            param(&mut threshold),
            param(&mut changed_marks),
            param(&mut changed_list),
            param(&mut changed_count),
        ];
        self.launch(
            self.shared.kernels.homeostasis,
            self.model.neuron_count(),
            THREADS,
            &mut parameters,
        )
    }

    fn collect_step_metrics(
        &mut self,
        target_index: Option<usize>,
        context_applied: bool,
        learned: bool,
    ) -> LeoResult<StepMetrics> {
        let b = self.buffers;
        let mut probabilities = [0.0f32; OUTPUT_CLASSES];
        self.copy_from_device(b.probabilities, &mut probabilities)?;
        self.probabilities.copy_from_slice(&probabilities);

        if target_index.is_some() {
            let mut neural_logits = [0.0f32; OUTPUT_CLASSES];
            self.copy_from_device(b.neural_logits, &mut neural_logits)?;
            self.neural_logits.copy_from_slice(&neural_logits);
        }
        let mut active_count = [0u32; 1];
        let mut context_count = [0u32; 1];
        let mut population_inhibition = [0.0f32; 1];
        let mut counters = [CudaStepCounters::default(); 1];
        let mut error_flag = [0u32; 1];
        self.copy_from_device(b.active_count, &mut active_count)?;
        self.copy_from_device(b.active_context_count, &mut context_count)?;
        self.copy_from_device(b.population_inhibition, &mut population_inhibition)?;
        self.copy_from_device(b.counters, &mut counters)?;
        self.copy_from_device(b.error_flag, &mut error_flag)?;
        if error_flag[0] != 0 {
            self.model.statistics.numerical_rejections =
                self.model.statistics.numerical_rejections.saturating_add(1);
            self.memset_zero(b.error_flag)?;
            return Err(LeoError::cuda(format!(
                "CUDA learning update rejected a non-finite value (code {})",
                error_flag[0]
            )));
        }
        let active_count = active_count[0] as usize;
        let mut active_u32 = vec![0u32; active_count];
        let mut active_value = vec![0.0f32; active_count];
        self.copy_from_device_prefix(b.active, &mut active_u32)?;
        self.copy_from_device_prefix(b.active_value, &mut active_value)?;
        for &neuron in &self.active {
            if let Some(value) = self.activation.get_mut(neuron) {
                *value = 0.0;
            }
        }
        self.active.clear();
        self.active.reserve(active_count);
        for (neuron, value) in active_u32.into_iter().zip(active_value) {
            let neuron = neuron as usize;
            self.active.push(neuron);
            if let Some(target) = self.activation.get_mut(neuron) {
                *target = value;
            }
        }
        let predicted_index = self
            .probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let predicted_symbol = output_index_to_symbol(predicted_index)
            .expect("CUDA probability index must map to an output symbol");
        let loss = target_index.map(|index| -self.probabilities[index].max(1.0e-12).ln());
        let neural_loss = target_index.map(|index| loss_from_logits(&self.neural_logits, index));
        let context_loss_gain = match (neural_loss, loss) {
            (Some(neural), Some(combined)) => neural - combined,
            _ => 0.0,
        };
        let counters = counters[0];
        let emitted_events = active_count.saturating_mul(self.model.recurrent.capacity_per_neuron);
        let context_cells = context_count[0] as usize;
        let output_madds = active_count
            .saturating_mul(OUTPUT_CLASSES)
            .saturating_add(context_cells.saturating_mul(self.model.config.context.embedding_dim))
            .saturating_add(if context_cells == 0 {
                0
            } else {
                self.model
                    .config
                    .context
                    .embedding_dim
                    .saturating_mul(OUTPUT_CLASSES)
            });
        let mean_abs_eligibility = if learned && counters.learning_eligibility_count > 0 {
            counters.learning_eligibility_abs_sum / counters.learning_eligibility_count as f32
        } else {
            0.0
        };
        Ok(StepMetrics {
            loss,
            predicted_symbol,
            active_neurons: active_count,
            emitted_events,
            suprathreshold_neurons: counters.suprathreshold as usize,
            block_selected_neurons: counters.block_selected as usize,
            global_cap_clipped_neurons: counters.global_clipped as usize,
            population_inhibition: population_inhibition[0],
            context_cells,
            context_probes: counters.context_probes as usize,
            context_applied,
            context_loss_gain,
            output_madds,
            eligible_recurrent_synapses: counters.eligible_recurrent as usize,
            eligible_input_synapses: counters.eligible_input as usize,
            mean_abs_eligibility,
        })
    }

    fn update_statistics(
        &mut self,
        symbol: u32,
        loss: Option<f32>,
        active_neurons: usize,
        emitted_events: usize,
    ) {
        if symbol < 256 {
            self.model.statistics.processed_bytes =
                self.model.statistics.processed_bytes.saturating_add(1);
        }
        if let Some(value) = loss {
            self.model.statistics.training_loss_sum += value as f64;
            self.model.statistics.training_targets =
                self.model.statistics.training_targets.saturating_add(1);
        }
        self.model.statistics.active_neurons_sum = self
            .model
            .statistics
            .active_neurons_sum
            .saturating_add(active_neurons as u64);
        self.model.statistics.active_neurons_peak = self
            .model
            .statistics
            .active_neurons_peak
            .max(active_neurons as u64);
        self.model.statistics.synaptic_events = self
            .model
            .statistics
            .synaptic_events
            .saturating_add(emitted_events as u64);
        self.model.statistics.persistent_ticks =
            self.model.statistics.persistent_ticks.saturating_add(1);
    }

    // Sequential fallible allocation keeps partial-resource cleanup explicit and auditable.
    #[allow(clippy::field_reassign_with_default)]
    fn allocate_buffers(&mut self) -> LeoResult<()> {
        let n = self.model.neuron_count();
        let recurrent = self.model.recurrent.weight.len();
        let input = self.model.input.weights.len();
        let active = self.model.config.model.max_active_global;
        let block_winners = self
            .model
            .config
            .model
            .block_count
            .saturating_mul(self.model.config.model.max_active_per_block);
        let contexts = self.model.context.keys.len();
        let context_dim = self.model.config.context.embedding_dim;
        let max_order = self.model.config.context.max_order;
        let ring_records = RING_BUCKETS.saturating_mul(active);
        let ring_weights = ring_records.saturating_mul(self.model.recurrent.capacity_per_neuron);
        let mut b = Buffers::default();
        b.config = self.allocate_one::<CudaConfig>()?;
        b.threshold = self.allocate_f32(n)?;
        b.excitability = self.allocate_f32(n)?;
        b.neuron_type = self.allocate_u8(n)?;
        b.recurrent_target = self.allocate_u32(recurrent)?;
        b.recurrent_branch = self.allocate_u8(recurrent)?;
        b.recurrent_delay = self.allocate_u8(recurrent)?;
        b.recurrent_weight = self.allocate_f32(recurrent)?;
        b.input_target = self.allocate_u32(input)?;
        b.input_branch = self.allocate_u8(input)?;
        b.input_weight = self.allocate_f32(input)?;
        b.output_weight = self.allocate_f32(n.saturating_mul(OUTPUT_CLASSES))?;
        b.output_bias = self.allocate_f32(OUTPUT_CLASSES)?;
        b.context_keys = self.allocate_u64(contexts)?;
        b.context_embeddings = self.allocate_f32(self.model.context.embeddings.len())?;
        b.context_observations = self.allocate_u32(contexts)?;
        b.context_output_weight = self.allocate_f32(self.model.context.output_weights.len())?;

        b.membrane = self.allocate_f32(n)?;
        b.activation = self.allocate_f32(n)?;
        b.fatigue = self.allocate_f32(n)?;
        b.refractory_until = self.allocate_u64(n)?;
        b.branches = self.allocate_f32(n.saturating_mul(4))?;
        b.branch_delta = self.allocate_f32(n.saturating_mul(4))?;
        b.branch_last_tick = self.allocate_u64(n)?;
        b.neuron_last_tick = self.allocate_u64(n)?;
        b.adaptation_fast = self.allocate_f32(n)?;
        b.adaptation_medium = self.allocate_f32(n)?;
        b.adaptation_slow = self.allocate_f32(n)?;
        b.touched_epoch = self.allocate_u64(n)?;
        b.touched_list = self.allocate_u32(n)?;
        b.touched_count = self.allocate_u32(1)?;
        b.selected_epoch = self.allocate_u64(n)?;
        b.surrogate = self.allocate_f32(n)?;
        b.candidate_activation = self.allocate_f32(n)?;
        b.active = self.allocate_u32(active)?;
        b.active_value = self.allocate_f32(active)?;
        b.active_count = self.allocate_u32(1)?;
        b.block_winner_neuron = self.allocate_u32(block_winners)?;
        b.block_winner_value = self.allocate_f32(block_winners)?;
        b.block_cutoff = self.allocate_f32(self.model.config.model.block_count)?;
        b.population_cutoff = self.allocate_f32(1)?;
        b.population_inhibition = self.allocate_f32(1)?;

        b.recurrent_branch_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_membrane_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_fatigue_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_adaptation_fast_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_adaptation_medium_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_adaptation_slow_sensitivity = self.allocate_f32(recurrent)?;
        b.recurrent_eligibility = self.allocate_f32(recurrent)?;
        b.recurrent_last_tick = self.allocate_u64(recurrent)?;
        b.recurrent_eligible_mark = self.allocate_u32(recurrent)?;
        b.recurrent_eligible_list = self.allocate_u32(recurrent)?;
        b.recurrent_eligible_count = self.allocate_u32(1)?;
        b.recurrent_next_eligible_list = self.allocate_u32(recurrent)?;
        b.recurrent_next_eligible_count = self.allocate_u32(1)?;
        b.input_branch_sensitivity = self.allocate_f32(input)?;
        b.input_membrane_sensitivity = self.allocate_f32(input)?;
        b.input_fatigue_sensitivity = self.allocate_f32(input)?;
        b.input_adaptation_fast_sensitivity = self.allocate_f32(input)?;
        b.input_adaptation_medium_sensitivity = self.allocate_f32(input)?;
        b.input_adaptation_slow_sensitivity = self.allocate_f32(input)?;
        b.input_eligibility = self.allocate_f32(input)?;
        b.input_last_tick = self.allocate_u64(input)?;
        b.input_eligible_mark = self.allocate_u32(input)?;
        b.input_eligible_list = self.allocate_u32(input)?;
        b.input_eligible_count = self.allocate_u32(1)?;
        b.input_next_eligible_list = self.allocate_u32(input)?;
        b.input_next_eligible_count = self.allocate_u32(1)?;

        b.ring_count = self.allocate_u32(RING_BUCKETS)?;
        b.ring_source = self.allocate_u32(ring_records)?;
        b.ring_activation = self.allocate_f32(ring_records)?;
        b.ring_weight = self.allocate_f32(ring_weights)?;
        b.context_history = self.allocate_u32(max_order)?;
        b.context_history_count = self.allocate_u32(1)?;
        b.active_context_slots = self.allocate_u32(max_order)?;
        b.active_context_scales = self.allocate_f32(max_order)?;
        b.active_context_count = self.allocate_u32(1)?;
        b.context_latent = self.allocate_f32(context_dim)?;
        b.context_gradient = self.allocate_f32(context_dim)?;
        b.neural_logits = self.allocate_f32(OUTPUT_CLASSES)?;
        b.logits = self.allocate_f32(OUTPUT_CLASSES)?;
        b.probabilities = self.allocate_f32(OUTPUT_CLASSES)?;
        b.errors = self.allocate_f32(OUTPUT_CLASSES)?;
        b.learning_destination_epoch = self.allocate_u64(n)?;
        b.learning_destination_list = self.allocate_u32(n)?;
        b.learning_destination_count = self.allocate_u32(1)?;
        b.learning_signal = self.allocate_f32(n)?;
        b.counters = self.allocate_one::<CudaStepCounters>()?;
        b.error_flag = self.allocate_u32(1)?;
        b.training_step_records = self.allocate_bytes(
            TRAINING_STEP_BATCH_CAPACITY.saturating_mul(mem::size_of::<CudaTrainingStepRecord>()),
        )?;
        b.persistent_steps = self.allocate_bytes(
            TRAINING_STEP_BATCH_CAPACITY.saturating_mul(mem::size_of::<CudaPersistentStep>()),
        )?;
        b.persistent_pointer_table = self.allocate_u64(PERSISTENT_POINTER_COUNT)?;
        b.batch_pointer_tables = self.allocate_u64(GPU_STORY_BATCH_MAX_LANES)?;
        b.batch_step_buffers = self.allocate_u64(GPU_STORY_BATCH_MAX_LANES)?;
        b.batch_step_counts = self.allocate_u32(GPU_STORY_BATCH_MAX_LANES)?;
        b.batch_base_ticks = self.allocate_u64(GPU_STORY_BATCH_MAX_LANES)?;
        b.batch_delta_pointer_table = self.allocate_u64(BATCH_DELTA_POINTER_COUNT)?;
        b.phase_profile_counters = self.allocate_u64(CUDA_PHASE_PROFILE_COUNTER_COUNT)?;
        b.replay_profile_counters = self.allocate_u64(CUDA_REPLAY_PROFILE_COUNTER_COUNT)?;

        b.batch_delta_threshold = self.allocate_f32(n)?;
        b.batch_delta_threshold_marks = self.allocate_u32(n)?;
        b.batch_delta_threshold_list = self.allocate_u32(n)?;
        b.batch_delta_threshold_count = self.allocate_u32(1)?;
        b.batch_delta_recurrent = self.allocate_f32(recurrent)?;
        b.batch_delta_recurrent_marks = self.allocate_u32(recurrent)?;
        b.batch_delta_recurrent_list = self.allocate_u32(recurrent)?;
        b.batch_delta_recurrent_count = self.allocate_u32(1)?;
        b.batch_delta_input = self.allocate_f32(input)?;
        b.batch_delta_input_marks = self.allocate_u32(input)?;
        b.batch_delta_input_list = self.allocate_u32(input)?;
        b.batch_delta_input_count = self.allocate_u32(1)?;
        b.batch_delta_output = self.allocate_f32(n.saturating_mul(OUTPUT_CLASSES))?;
        b.batch_delta_output_marks = self.allocate_u32(n)?;
        b.batch_delta_output_list = self.allocate_u32(n)?;
        b.batch_delta_output_count = self.allocate_u32(1)?;
        b.batch_delta_output_bias = self.allocate_f32(OUTPUT_CLASSES)?;
        b.batch_delta_context_embedding = self.allocate_f32(self.model.context.embeddings.len())?;
        b.batch_delta_context_marks = self.allocate_u32(contexts)?;
        b.batch_delta_context_list = self.allocate_u32(contexts)?;
        b.batch_delta_context_count = self.allocate_u32(1)?;
        b.batch_delta_context_observations = self.allocate_u32(contexts)?;
        b.batch_delta_context_output =
            self.allocate_f32(self.model.context.output_weights.len())?;

        b.changed_threshold_marks = self.allocate_u32(n)?;
        b.changed_threshold_list = self.allocate_u32(n)?;
        b.changed_threshold_count = self.allocate_u32(1)?;
        b.changed_recurrent_marks = self.allocate_u32(recurrent)?;
        b.changed_recurrent_list = self.allocate_u32(recurrent)?;
        b.changed_recurrent_count = self.allocate_u32(1)?;
        b.changed_input_marks = self.allocate_u32(input)?;
        b.changed_input_list = self.allocate_u32(input)?;
        b.changed_input_count = self.allocate_u32(1)?;
        b.changed_output_marks = self.allocate_u32(n)?;
        b.changed_output_list = self.allocate_u32(n)?;
        b.changed_output_count = self.allocate_u32(1)?;
        b.changed_context_marks = self.allocate_u32(contexts)?;
        b.changed_context_list = self.allocate_u32(contexts)?;
        b.changed_context_count = self.allocate_u32(1)?;

        b.gather_threshold = self.allocate_f32(n)?;
        b.gather_recurrent = self.allocate_f32(recurrent)?;
        b.gather_input = self.allocate_f32(input)?;
        b.gather_output = self.allocate_f32(n.saturating_mul(OUTPUT_CLASSES))?;
        b.gather_context_keys = self.allocate_u64(contexts)?;
        b.gather_context_observations = self.allocate_u32(contexts)?;
        b.gather_context_embeddings = self.allocate_f32(contexts.saturating_mul(context_dim))?;

        for buffer in [
            b.batch_delta_threshold,
            b.batch_delta_threshold_marks,
            b.batch_delta_threshold_count,
            b.batch_delta_recurrent,
            b.batch_delta_recurrent_marks,
            b.batch_delta_recurrent_count,
            b.batch_delta_input,
            b.batch_delta_input_marks,
            b.batch_delta_input_count,
            b.batch_delta_output,
            b.batch_delta_output_marks,
            b.batch_delta_output_count,
            b.batch_delta_output_bias,
            b.batch_delta_context_embedding,
            b.batch_delta_context_marks,
            b.batch_delta_context_count,
            b.batch_delta_context_observations,
            b.batch_delta_context_output,
            b.changed_threshold_marks,
            b.changed_threshold_count,
            b.changed_recurrent_marks,
            b.changed_recurrent_count,
            b.changed_input_marks,
            b.changed_input_count,
            b.changed_output_marks,
            b.changed_output_count,
            b.changed_context_marks,
            b.changed_context_count,
        ] {
            self.memset_zero(buffer)?;
        }
        self.host_batch_pointer_tables = self.allocate_pinned_host(
            GPU_STORY_BATCH_MAX_LANES.saturating_mul(mem::size_of::<CuDevicePtr>()),
        )?;
        self.host_batch_step_buffers = self.allocate_pinned_host(
            GPU_STORY_BATCH_MAX_LANES.saturating_mul(mem::size_of::<CuDevicePtr>()),
        )?;
        self.host_batch_step_counts = self.allocate_pinned_host(
            GPU_STORY_BATCH_MAX_LANES.saturating_mul(mem::size_of::<u32>()),
        )?;
        self.host_batch_base_ticks = self.allocate_pinned_host(
            GPU_STORY_BATCH_MAX_LANES.saturating_mul(mem::size_of::<u64>()),
        )?;
        self.host_batch_change_counts = self.allocate_pinned_host(
            GPU_STORY_BATCH_MAX_LANES
                .saturating_mul(5)
                .saturating_mul(mem::size_of::<u32>()),
        )?;
        self.buffers = b;
        Ok(())
    }

    fn initialize_static_pointer_tables(&self) -> LeoResult<()> {
        let persistent = self.persistent_pointer_table();
        let deltas = self.batch_delta_pointer_table();
        self.copy_to_device(self.buffers.persistent_pointer_table, &persistent)?;
        self.copy_to_device(self.buffers.batch_delta_pointer_table, &deltas)
    }

    fn upload_full_model(&mut self) -> LeoResult<()> {
        self.make_current()?;
        let config = CudaConfig::from_model(&self.model)?;
        let b = self.buffers;
        self.copy_to_device(b.config, std::slice::from_ref(&config))?;
        self.copy_to_device(b.threshold, &self.model.neurons.threshold)?;
        self.copy_to_device(b.excitability, &self.model.neurons.excitability)?;
        self.copy_to_device(b.neuron_type, &self.model.neurons.neuron_type)?;
        self.copy_to_device(b.recurrent_target, &self.model.recurrent.target_neuron)?;
        self.copy_to_device(b.recurrent_branch, &self.model.recurrent.target_branch)?;
        self.copy_to_device(b.recurrent_delay, &self.model.recurrent.delay)?;
        self.copy_to_device(b.recurrent_weight, &self.model.recurrent.weight)?;
        self.copy_to_device(b.input_target, &self.model.input.targets)?;
        self.copy_to_device(b.input_branch, &self.model.input.branches)?;
        self.copy_to_device(b.input_weight, &self.model.input.weights)?;
        let transposed = transpose_output_weights(&self.model);
        self.copy_to_device(b.output_weight, &transposed)?;
        self.copy_to_device(b.output_bias, &self.model.output.bias)?;
        self.copy_to_device(b.context_keys, &self.model.context.keys)?;
        self.copy_to_device(b.context_embeddings, &self.model.context.embeddings)?;
        self.copy_to_device(b.context_observations, &self.model.context.observations)?;
        self.copy_to_device(b.context_output_weight, &self.model.context.output_weights)?;
        self.synchronize()?;
        self.clear_device_changes_all()?;
        self.model_dirty = false;
        self.output_bias_dirty_since_sync = false;
        self.context_output_dirty_since_sync = false;
        Ok(())
    }

    fn accumulate_parameter_changes(&mut self, changes: &ParameterChanges) {
        if !self.track_parameter_changes {
            return;
        }
        accumulate_indices(
            &changes.threshold,
            &mut self.accumulated_threshold_mark,
            &mut self.accumulated_changes.threshold,
        );
        accumulate_indices(
            &changes.recurrent_weight,
            &mut self.accumulated_recurrent_mark,
            &mut self.accumulated_changes.recurrent_weight,
        );
        accumulate_indices(
            &changes.input_weight,
            &mut self.accumulated_input_mark,
            &mut self.accumulated_changes.input_weight,
        );
        accumulate_indices(
            &changes.output_neurons,
            &mut self.accumulated_output_mark,
            &mut self.accumulated_changes.output_neurons,
        );
        accumulate_indices(
            &changes.context_slots,
            &mut self.accumulated_context_mark,
            &mut self.accumulated_changes.context_slots,
        );
        self.accumulated_changes.output_bias_dirty |= changes.output_bias_dirty;
        self.accumulated_changes.context_output_dirty |= changes.context_output_dirty;
    }

    fn current_device_parameter_changes(
        &self,
        output_bias_dirty: bool,
        context_output_dirty: bool,
    ) -> LeoResult<ParameterChanges> {
        let b = &self.buffers;
        Ok(ParameterChanges {
            threshold: self
                .read_change_list(b.changed_threshold_list, b.changed_threshold_count)?,
            recurrent_weight: self
                .read_change_list(b.changed_recurrent_list, b.changed_recurrent_count)?,
            input_weight: self.read_change_list(b.changed_input_list, b.changed_input_count)?,
            output_neurons: self.read_change_list(b.changed_output_list, b.changed_output_count)?,
            context_slots: self
                .read_change_list(b.changed_context_list, b.changed_context_count)?,
            output_bias_dirty,
            context_output_dirty,
        })
    }

    fn sync_changed_model_to_host(&mut self) -> LeoResult<()> {
        let b = self.buffers;
        let threshold_indices =
            self.read_change_list(b.changed_threshold_list, b.changed_threshold_count)?;
        let recurrent_indices =
            self.read_change_list(b.changed_recurrent_list, b.changed_recurrent_count)?;
        let input_indices = self.read_change_list(b.changed_input_list, b.changed_input_count)?;
        let output_indices =
            self.read_change_list(b.changed_output_list, b.changed_output_count)?;
        let context_indices =
            self.read_change_list(b.changed_context_list, b.changed_context_count)?;

        if !threshold_indices.is_empty() {
            self.launch_gather_f32(
                b.threshold,
                b.changed_threshold_list,
                b.changed_threshold_count,
                b.gather_threshold,
                threshold_indices.len(),
            )?;
        }
        if !recurrent_indices.is_empty() {
            self.launch_gather_f32(
                b.recurrent_weight,
                b.changed_recurrent_list,
                b.changed_recurrent_count,
                b.gather_recurrent,
                recurrent_indices.len(),
            )?;
        }
        if !input_indices.is_empty() {
            self.launch_gather_f32(
                b.input_weight,
                b.changed_input_list,
                b.changed_input_count,
                b.gather_input,
                input_indices.len(),
            )?;
        }
        if !output_indices.is_empty() {
            let mut source = b.output_weight.pointer;
            let mut indices = b.changed_output_list.pointer;
            let mut count = b.changed_output_count.pointer;
            let mut output = b.gather_output.pointer;
            let mut params = [
                param(&mut source),
                param(&mut indices),
                param(&mut count),
                param(&mut output),
            ];
            self.launch(
                self.shared.kernels.gather_output,
                output_indices.len().saturating_mul(OUTPUT_CLASSES),
                THREADS,
                &mut params,
            )?;
        }
        if !context_indices.is_empty() {
            let mut keys = b.context_keys.pointer;
            let mut observations = b.context_observations.pointer;
            let mut embeddings = b.context_embeddings.pointer;
            let mut embedding_dim = as_u32(
                "context embedding dim",
                self.model.config.context.embedding_dim,
            )?;
            let mut indices = b.changed_context_list.pointer;
            let mut count = b.changed_context_count.pointer;
            let mut out_keys = b.gather_context_keys.pointer;
            let mut out_obs = b.gather_context_observations.pointer;
            let mut out_embeddings = b.gather_context_embeddings.pointer;
            let mut params = [
                param(&mut keys),
                param(&mut observations),
                param(&mut embeddings),
                param(&mut embedding_dim),
                param(&mut indices),
                param(&mut count),
                param(&mut out_keys),
                param(&mut out_obs),
                param(&mut out_embeddings),
            ];
            self.launch(
                self.shared.kernels.gather_context,
                context_indices
                    .len()
                    .saturating_mul(self.model.config.context.embedding_dim)
                    .max(context_indices.len()),
                THREADS,
                &mut params,
            )?;
        }
        self.synchronize()?;

        if !threshold_indices.is_empty() {
            let mut values = vec![0.0f32; threshold_indices.len()];
            self.copy_from_device_prefix(b.gather_threshold, &mut values)?;
            for (&index, value) in threshold_indices.iter().zip(values) {
                self.model.neurons.threshold[index] = value;
            }
        }
        if !recurrent_indices.is_empty() {
            let mut values = vec![0.0f32; recurrent_indices.len()];
            self.copy_from_device_prefix(b.gather_recurrent, &mut values)?;
            for (&index, value) in recurrent_indices.iter().zip(values) {
                self.model.recurrent.weight[index] = value;
            }
        }
        if !input_indices.is_empty() {
            let mut values = vec![0.0f32; input_indices.len()];
            self.copy_from_device_prefix(b.gather_input, &mut values)?;
            for (&index, value) in input_indices.iter().zip(values) {
                self.model.input.weights[index] = value;
            }
        }
        if !output_indices.is_empty() {
            let mut values = vec![0.0f32; output_indices.len().saturating_mul(OUTPUT_CLASSES)];
            self.copy_from_device_prefix(b.gather_output, &mut values)?;
            let neuron_count = self.model.neuron_count();
            for (row, &neuron) in output_indices.iter().enumerate() {
                for output in 0..OUTPUT_CLASSES {
                    self.model.output.weights[output * neuron_count + neuron] =
                        values[row * OUTPUT_CLASSES + output];
                }
            }
        }
        if self.output_bias_dirty_since_sync {
            let mut bias = vec![0.0f32; OUTPUT_CLASSES];
            self.copy_from_device(b.output_bias, &mut bias)?;
            self.model.output.bias = bias;
        }
        if !context_indices.is_empty() {
            let mut keys = vec![0u64; context_indices.len()];
            let mut observations = vec![0u32; context_indices.len()];
            let dim = self.model.config.context.embedding_dim;
            let mut embeddings = vec![0.0f32; context_indices.len().saturating_mul(dim)];
            self.copy_from_device_prefix(b.gather_context_keys, &mut keys)?;
            self.copy_from_device_prefix(b.gather_context_observations, &mut observations)?;
            self.copy_from_device_prefix(b.gather_context_embeddings, &mut embeddings)?;
            for (row, &slot) in context_indices.iter().enumerate() {
                self.model.context.keys[slot] = keys[row];
                self.model.context.observations[slot] = observations[row];
                let start = slot * dim;
                self.model.context.embeddings[start..start + dim]
                    .copy_from_slice(&embeddings[row * dim..(row + 1) * dim]);
            }
        }
        if self.context_output_dirty_since_sync {
            let mut values = vec![0.0f32; self.model.context.output_weights.len()];
            self.copy_from_device(b.context_output_weight, &mut values)?;
            self.model.context.output_weights = values;
        }

        if self.track_parameter_changes {
            accumulate_indices(
                &threshold_indices,
                &mut self.accumulated_threshold_mark,
                &mut self.accumulated_changes.threshold,
            );
            accumulate_indices(
                &recurrent_indices,
                &mut self.accumulated_recurrent_mark,
                &mut self.accumulated_changes.recurrent_weight,
            );
            accumulate_indices(
                &input_indices,
                &mut self.accumulated_input_mark,
                &mut self.accumulated_changes.input_weight,
            );
            accumulate_indices(
                &output_indices,
                &mut self.accumulated_output_mark,
                &mut self.accumulated_changes.output_neurons,
            );
            accumulate_indices(
                &context_indices,
                &mut self.accumulated_context_mark,
                &mut self.accumulated_changes.context_slots,
            );
            self.accumulated_changes.output_bias_dirty |= self.output_bias_dirty_since_sync;
            self.accumulated_changes.context_output_dirty |= self.context_output_dirty_since_sync;
        }

        self.clear_change_list(
            b.changed_threshold_marks,
            b.changed_threshold_list,
            b.changed_threshold_count,
            threshold_indices.len(),
        )?;
        self.clear_change_list(
            b.changed_recurrent_marks,
            b.changed_recurrent_list,
            b.changed_recurrent_count,
            recurrent_indices.len(),
        )?;
        self.clear_change_list(
            b.changed_input_marks,
            b.changed_input_list,
            b.changed_input_count,
            input_indices.len(),
        )?;
        self.clear_change_list(
            b.changed_output_marks,
            b.changed_output_list,
            b.changed_output_count,
            output_indices.len(),
        )?;
        self.clear_change_list(
            b.changed_context_marks,
            b.changed_context_list,
            b.changed_context_count,
            context_indices.len(),
        )?;
        self.synchronize()?;
        self.output_bias_dirty_since_sync = false;
        self.context_output_dirty_since_sync = false;
        Ok(())
    }

    fn read_batch_lane_change_counts(&mut self, lanes: &[CudaBatchLane]) -> LeoResult<Vec<u32>> {
        let count_elements = lanes.len().saturating_mul(5);
        if count_elements == 0 {
            return Ok(Vec::new());
        }
        self.ensure_batch_snapshot_capacity(0, count_elements, 0)?;
        self.record_event(
            self.events.compute_done,
            self.compute_stream,
            "cuEventRecord(batch snapshot counts ready)",
        )?;
        self.stream_wait_event(
            self.transfer_stream,
            self.events.compute_done,
            "cuStreamWaitEvent(batch snapshot counts ready)",
        )?;
        for (lane_index, lane) in lanes.iter().enumerate() {
            let b = lane.buffers;
            for (kind, count_buffer) in [
                b.changed_threshold_count,
                b.changed_recurrent_count,
                b.changed_input_count,
                b.changed_output_count,
                b.changed_context_count,
            ]
            .into_iter()
            .enumerate()
            {
                self.copy_device_to_device_async_on(
                    device_sub_buffer::<u32>(
                        self.batch_snapshot_device_u32,
                        lane_index * 5 + kind,
                        1,
                    )?,
                    count_buffer,
                    self.transfer_stream,
                )?;
            }
        }
        self.copy_device_to_pinned_async_on::<u32>(
            self.host_batch_change_counts,
            self.batch_snapshot_device_u32,
            count_elements,
            self.transfer_stream,
        )?;
        self.shared.driver.check(
            unsafe { (self.shared.driver.stream_synchronize)(self.transfer_stream) },
            "cuStreamSynchronize(batch snapshot counts)",
        )?;
        pinned_read_vec::<u32>(self.host_batch_change_counts, count_elements)
    }

    fn snapshot_batch_lane_values_batched(
        &mut self,
        lanes: &[CudaBatchLane],
        output_bias_dirty: &[bool],
        context_output_dirty: &[bool],
        revision_increment: &[u64],
        statistics: &[StatisticsDelta],
    ) -> LeoResult<Vec<(ParameterChanges, TrackedModelValues)>> {
        self.make_current()?;
        let lane_count = lanes.len();
        if output_bias_dirty.len() != lane_count
            || context_output_dirty.len() != lane_count
            || revision_increment.len() != lane_count
            || statistics.len() != lane_count
        {
            return Err(LeoError::cuda(
                "batched story snapshot metadata does not match lane count",
            ));
        }
        if lane_count == 0 {
            return Ok(Vec::new());
        }

        // A tiny first transfer sizes the variable sparse payload. The same
        // count helper is shared with the context-only fast batch snapshot.
        let counts = self.read_batch_lane_change_counts(lanes)?;

        fn take_span(cursor: &mut usize, len: usize) -> SnapshotSpan {
            let span = SnapshotSpan {
                offset: *cursor,
                len,
            };
            *cursor = (*cursor).saturating_add(len);
            span
        }

        let context_dim = self.model.config.context.embedding_dim;
        let context_output_len = self.model.context.output_weights.len();
        let mut u32_cursor = 0usize;
        let mut u64_cursor = 0usize;
        let mut f32_cursor = 0usize;
        let mut layouts = Vec::with_capacity(lane_count);
        for lane_index in 0..lane_count {
            let base = lane_index * 5;
            let threshold_count = counts[base] as usize;
            let recurrent_count = counts[base + 1] as usize;
            let input_count = counts[base + 2] as usize;
            let output_count = counts[base + 3] as usize;
            let context_count = counts[base + 4] as usize;
            layouts.push(BatchLaneSnapshotLayout {
                threshold_indices: take_span(&mut u32_cursor, threshold_count),
                recurrent_indices: take_span(&mut u32_cursor, recurrent_count),
                input_indices: take_span(&mut u32_cursor, input_count),
                output_indices: take_span(&mut u32_cursor, output_count),
                context_indices: take_span(&mut u32_cursor, context_count),
                context_observations: take_span(&mut u32_cursor, context_count),
                context_keys: take_span(&mut u64_cursor, context_count),
                threshold_values: take_span(&mut f32_cursor, threshold_count),
                recurrent_values: take_span(&mut f32_cursor, recurrent_count),
                input_values: take_span(&mut f32_cursor, input_count),
                output_values: take_span(
                    &mut f32_cursor,
                    output_count.saturating_mul(OUTPUT_CLASSES),
                ),
                context_embeddings: take_span(
                    &mut f32_cursor,
                    context_count.saturating_mul(context_dim),
                ),
                output_bias: output_bias_dirty[lane_index]
                    .then(|| take_span(&mut f32_cursor, OUTPUT_CLASSES)),
                context_output: context_output_dirty[lane_index]
                    .then(|| take_span(&mut f32_cursor, context_output_len)),
            });
        }

        self.ensure_batch_snapshot_capacity(f32_cursor, u32_cursor, u64_cursor)?;

        // Pack every sparse index list into the aggregate u32 device buffer.
        // These are device-local copies and can overlap the value-gather kernels
        // below; the context-observation spans occupy separate u32 regions.
        for (lane, layout) in lanes.iter().zip(&layouts) {
            let b = lane.buffers;
            for (span, source) in [
                (layout.threshold_indices, b.changed_threshold_list),
                (layout.recurrent_indices, b.changed_recurrent_list),
                (layout.input_indices, b.changed_input_list),
                (layout.output_indices, b.changed_output_list),
                (layout.context_indices, b.changed_context_list),
            ] {
                if span.len != 0 {
                    self.copy_device_to_device_async_on(
                        device_sub_buffer::<u32>(
                            self.batch_snapshot_device_u32,
                            span.offset,
                            span.len,
                        )?,
                        device_sub_buffer::<u32>(source, 0, span.len)?,
                        self.transfer_stream,
                    )?;
                }
            }
        }

        // Queue every gather first. Each lane writes to a disjoint region of
        // the aggregate device buffers, so no per-lane synchronization is
        // needed even though the canonical runtime owns the kernels/stream.
        for (lane, layout) in lanes.iter().zip(&layouts) {
            let b = lane.buffers;
            if layout.threshold_values.len != 0 {
                self.launch_gather_f32(
                    b.threshold,
                    b.changed_threshold_list,
                    b.changed_threshold_count,
                    device_sub_buffer::<f32>(
                        self.batch_snapshot_device_f32,
                        layout.threshold_values.offset,
                        layout.threshold_values.len,
                    )?,
                    layout.threshold_values.len,
                )?;
            }
            if layout.recurrent_values.len != 0 {
                self.launch_gather_f32(
                    b.recurrent_weight,
                    b.changed_recurrent_list,
                    b.changed_recurrent_count,
                    device_sub_buffer::<f32>(
                        self.batch_snapshot_device_f32,
                        layout.recurrent_values.offset,
                        layout.recurrent_values.len,
                    )?,
                    layout.recurrent_values.len,
                )?;
            }
            if layout.input_values.len != 0 {
                self.launch_gather_f32(
                    b.input_weight,
                    b.changed_input_list,
                    b.changed_input_count,
                    device_sub_buffer::<f32>(
                        self.batch_snapshot_device_f32,
                        layout.input_values.offset,
                        layout.input_values.len,
                    )?,
                    layout.input_values.len,
                )?;
            }
            if layout.output_indices.len != 0 {
                let mut source = b.output_weight.pointer;
                let mut indices = b.changed_output_list.pointer;
                let mut count = b.changed_output_count.pointer;
                let output_buffer = device_sub_buffer::<f32>(
                    self.batch_snapshot_device_f32,
                    layout.output_values.offset,
                    layout.output_values.len,
                )?;
                let mut output = output_buffer.pointer;
                let mut params = [
                    param(&mut source),
                    param(&mut indices),
                    param(&mut count),
                    param(&mut output),
                ];
                self.launch(
                    self.shared.kernels.gather_output,
                    layout.output_values.len,
                    THREADS,
                    &mut params,
                )?;
            }
            if layout.context_indices.len != 0 {
                let mut keys = b.context_keys.pointer;
                let mut observations = b.context_observations.pointer;
                let mut embeddings = b.context_embeddings.pointer;
                let mut embedding_dim = as_u32("context embedding dim", context_dim)?;
                let mut indices = b.changed_context_list.pointer;
                let mut count = b.changed_context_count.pointer;
                let key_buffer = device_sub_buffer::<u64>(
                    self.batch_snapshot_device_u64,
                    layout.context_keys.offset,
                    layout.context_keys.len,
                )?;
                let obs_buffer = device_sub_buffer::<u32>(
                    self.batch_snapshot_device_u32,
                    layout.context_observations.offset,
                    layout.context_observations.len,
                )?;
                let embedding_buffer = device_sub_buffer::<f32>(
                    self.batch_snapshot_device_f32,
                    layout.context_embeddings.offset,
                    layout.context_embeddings.len,
                )?;
                let mut out_keys = key_buffer.pointer;
                let mut out_obs = obs_buffer.pointer;
                let mut out_embeddings = embedding_buffer.pointer;
                let mut params = [
                    param(&mut keys),
                    param(&mut observations),
                    param(&mut embeddings),
                    param(&mut embedding_dim),
                    param(&mut indices),
                    param(&mut count),
                    param(&mut out_keys),
                    param(&mut out_obs),
                    param(&mut out_embeddings),
                ];
                self.launch(
                    self.shared.kernels.gather_context,
                    layout
                        .context_embeddings
                        .len
                        .max(layout.context_indices.len),
                    THREADS,
                    &mut params,
                )?;
            }
        }

        self.record_event(
            self.events.compute_done,
            self.compute_stream,
            "cuEventRecord(batch sparse gathers ready)",
        )?;
        self.stream_wait_event(
            self.transfer_stream,
            self.events.compute_done,
            "cuStreamWaitEvent(batch sparse gathers ready)",
        )?;

        // Bias/context-output rows are already contiguous, so stage them into
        // their reserved aggregate f32 spans with device-local copies. Then the
        // entire variable payload crosses PCIe in at most three D2H operations:
        // one f32 buffer, one u32 buffer, and one u64 buffer.
        for (lane, layout) in lanes.iter().zip(&layouts) {
            let b = lane.buffers;
            if let Some(span) = layout.output_bias {
                self.copy_device_to_device_async_on(
                    device_sub_buffer::<f32>(
                        self.batch_snapshot_device_f32,
                        span.offset,
                        span.len,
                    )?,
                    b.output_bias,
                    self.transfer_stream,
                )?;
            }
            if let Some(span) = layout.context_output {
                self.copy_device_to_device_async_on(
                    device_sub_buffer::<f32>(
                        self.batch_snapshot_device_f32,
                        span.offset,
                        span.len,
                    )?,
                    b.context_output_weight,
                    self.transfer_stream,
                )?;
            }
        }
        if f32_cursor != 0 {
            self.copy_device_to_pinned_async_on::<f32>(
                self.batch_snapshot_host_f32,
                self.batch_snapshot_device_f32,
                f32_cursor,
                self.transfer_stream,
            )?;
        }
        if u32_cursor != 0 {
            self.copy_device_to_pinned_async_on::<u32>(
                self.batch_snapshot_host_u32,
                self.batch_snapshot_device_u32,
                u32_cursor,
                self.transfer_stream,
            )?;
        }
        if u64_cursor != 0 {
            self.copy_device_to_pinned_async_on::<u64>(
                self.batch_snapshot_host_u64,
                self.batch_snapshot_device_u64,
                u64_cursor,
                self.transfer_stream,
            )?;
        }
        self.shared.driver.check(
            unsafe { (self.shared.driver.stream_synchronize)(self.transfer_stream) },
            "cuStreamSynchronize(batched sparse snapshot)",
        )?;

        let host_u32 = pinned_read_vec::<u32>(self.batch_snapshot_host_u32, u32_cursor)?;
        let host_u64 = pinned_read_vec::<u64>(self.batch_snapshot_host_u64, u64_cursor)?;
        let host_f32 = pinned_read_vec::<f32>(self.batch_snapshot_host_f32, f32_cursor)?;
        let usize_values = |span: SnapshotSpan| {
            host_u32[span.offset..span.offset + span.len]
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>()
        };
        let mut snapshots = Vec::with_capacity(lane_count);
        for lane_index in 0..lane_count {
            let layout = &layouts[lane_index];
            let changes = ParameterChanges {
                threshold: usize_values(layout.threshold_indices),
                recurrent_weight: usize_values(layout.recurrent_indices),
                input_weight: usize_values(layout.input_indices),
                output_neurons: usize_values(layout.output_indices),
                context_slots: usize_values(layout.context_indices),
                output_bias_dirty: output_bias_dirty[lane_index],
                context_output_dirty: context_output_dirty[lane_index],
            };
            let f32_values =
                |span: SnapshotSpan| host_f32[span.offset..span.offset + span.len].to_vec();
            let values = TrackedModelValues {
                threshold: f32_values(layout.threshold_values),
                recurrent_weight: f32_values(layout.recurrent_values),
                input_weight: f32_values(layout.input_values),
                output_weight_by_neuron: f32_values(layout.output_values),
                output_bias: layout.output_bias.map(f32_values),
                context_keys: host_u64[layout.context_keys.offset
                    ..layout.context_keys.offset + layout.context_keys.len]
                    .to_vec(),
                context_observations: host_u32[layout.context_observations.offset
                    ..layout.context_observations.offset + layout.context_observations.len]
                    .to_vec(),
                context_embeddings: f32_values(layout.context_embeddings),
                context_output_weight: layout.context_output.map(f32_values),
                revision_increment: revision_increment[lane_index],
                statistics: statistics[lane_index].clone(),
            };
            snapshots.push((changes, values));
        }

        for (lane, layout) in lanes.iter().zip(&layouts) {
            let b = lane.buffers;
            for (marks, list, count, changed_count) in [
                (
                    b.changed_threshold_marks,
                    b.changed_threshold_list,
                    b.changed_threshold_count,
                    layout.threshold_indices.len,
                ),
                (
                    b.changed_recurrent_marks,
                    b.changed_recurrent_list,
                    b.changed_recurrent_count,
                    layout.recurrent_indices.len,
                ),
                (
                    b.changed_input_marks,
                    b.changed_input_list,
                    b.changed_input_count,
                    layout.input_indices.len,
                ),
                (
                    b.changed_output_marks,
                    b.changed_output_list,
                    b.changed_output_count,
                    layout.output_indices.len,
                ),
                (
                    b.changed_context_marks,
                    b.changed_context_list,
                    b.changed_context_count,
                    layout.context_indices.len,
                ),
            ] {
                self.clear_change_list(marks, list, count, changed_count)?;
            }
        }
        self.synchronize()?;
        Ok(snapshots)
    }

    fn snapshot_batch_lane_context_values_batched(
        &mut self,
        lanes: &[CudaBatchLane],
        revision_increment: &[u64],
        statistics: &[StatisticsDelta],
    ) -> LeoResult<Vec<(ParameterChanges, TrackedModelValues)>> {
        self.make_current()?;
        let lane_count = lanes.len();
        if revision_increment.len() != lane_count || statistics.len() != lane_count {
            return Err(LeoError::cuda(
                "context-only batch snapshot metadata does not match lane count",
            ));
        }
        if lane_count == 0 {
            return Ok(Vec::new());
        }

        // We still read all five tiny counters so every lane's sparse mark
        // arrays can be cleared without copying the large fixed index/value
        // payload to the CPU. Only context keys/observations/embeddings cross
        // PCIe in the device-merge fast path.
        let counts = self.read_batch_lane_change_counts(lanes)?;
        let mut u32_cursor = 0usize;
        let mut u64_cursor = 0usize;
        let mut f32_cursor = 0usize;
        let mut layouts = Vec::with_capacity(lane_count);
        let context_dim = self.model.config.context.embedding_dim;
        for lane_index in 0..lane_count {
            let context_count = counts[lane_index * 5 + 4] as usize;
            let context_indices = SnapshotSpan {
                offset: u32_cursor,
                len: context_count,
            };
            u32_cursor = u32_cursor.saturating_add(context_count);
            let context_observations = SnapshotSpan {
                offset: u32_cursor,
                len: context_count,
            };
            u32_cursor = u32_cursor.saturating_add(context_count);
            let context_keys = SnapshotSpan {
                offset: u64_cursor,
                len: context_count,
            };
            u64_cursor = u64_cursor.saturating_add(context_count);
            let context_embeddings = SnapshotSpan {
                offset: f32_cursor,
                len: context_count.saturating_mul(context_dim),
            };
            f32_cursor = f32_cursor.saturating_add(context_embeddings.len);
            layouts.push(BatchLaneContextSnapshotLayout {
                context_indices,
                context_observations,
                context_keys,
                context_embeddings,
            });
        }
        self.ensure_batch_snapshot_capacity(f32_cursor, u32_cursor, u64_cursor)?;

        for (lane, layout) in lanes.iter().zip(&layouts) {
            if layout.context_indices.len == 0 {
                continue;
            }
            let b = lane.buffers;
            self.copy_device_to_device_async_on(
                device_sub_buffer::<u32>(
                    self.batch_snapshot_device_u32,
                    layout.context_indices.offset,
                    layout.context_indices.len,
                )?,
                device_sub_buffer::<u32>(b.changed_context_list, 0, layout.context_indices.len)?,
                self.transfer_stream,
            )?;

            let mut keys = b.context_keys.pointer;
            let mut observations = b.context_observations.pointer;
            let mut embeddings = b.context_embeddings.pointer;
            let mut embedding_dim = as_u32("context embedding dim", context_dim)?;
            let mut indices = b.changed_context_list.pointer;
            let mut count = b.changed_context_count.pointer;
            let key_buffer = device_sub_buffer::<u64>(
                self.batch_snapshot_device_u64,
                layout.context_keys.offset,
                layout.context_keys.len,
            )?;
            let obs_buffer = device_sub_buffer::<u32>(
                self.batch_snapshot_device_u32,
                layout.context_observations.offset,
                layout.context_observations.len,
            )?;
            let embedding_buffer = device_sub_buffer::<f32>(
                self.batch_snapshot_device_f32,
                layout.context_embeddings.offset,
                layout.context_embeddings.len,
            )?;
            let mut out_keys = key_buffer.pointer;
            let mut out_obs = obs_buffer.pointer;
            let mut out_embeddings = embedding_buffer.pointer;
            let mut params = [
                param(&mut keys),
                param(&mut observations),
                param(&mut embeddings),
                param(&mut embedding_dim),
                param(&mut indices),
                param(&mut count),
                param(&mut out_keys),
                param(&mut out_obs),
                param(&mut out_embeddings),
            ];
            self.launch(
                self.shared.kernels.gather_context,
                layout
                    .context_embeddings
                    .len
                    .max(layout.context_indices.len),
                THREADS,
                &mut params,
            )?;
        }

        self.record_event(
            self.events.compute_done,
            self.compute_stream,
            "cuEventRecord(batch context gathers ready)",
        )?;
        self.stream_wait_event(
            self.transfer_stream,
            self.events.compute_done,
            "cuStreamWaitEvent(batch context gathers ready)",
        )?;
        if f32_cursor != 0 {
            self.copy_device_to_pinned_async_on::<f32>(
                self.batch_snapshot_host_f32,
                self.batch_snapshot_device_f32,
                f32_cursor,
                self.transfer_stream,
            )?;
        }
        if u32_cursor != 0 {
            self.copy_device_to_pinned_async_on::<u32>(
                self.batch_snapshot_host_u32,
                self.batch_snapshot_device_u32,
                u32_cursor,
                self.transfer_stream,
            )?;
        }
        if u64_cursor != 0 {
            self.copy_device_to_pinned_async_on::<u64>(
                self.batch_snapshot_host_u64,
                self.batch_snapshot_device_u64,
                u64_cursor,
                self.transfer_stream,
            )?;
        }
        self.shared.driver.check(
            unsafe { (self.shared.driver.stream_synchronize)(self.transfer_stream) },
            "cuStreamSynchronize(context-only batch snapshot)",
        )?;

        let host_u32 = pinned_read_vec::<u32>(self.batch_snapshot_host_u32, u32_cursor)?;
        let host_u64 = pinned_read_vec::<u64>(self.batch_snapshot_host_u64, u64_cursor)?;
        let host_f32 = pinned_read_vec::<f32>(self.batch_snapshot_host_f32, f32_cursor)?;
        let mut snapshots = Vec::with_capacity(lane_count);
        for lane_index in 0..lane_count {
            let layout = &layouts[lane_index];
            let context_slots = host_u32[layout.context_indices.offset
                ..layout.context_indices.offset + layout.context_indices.len]
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>();
            let changes = ParameterChanges {
                context_slots,
                ..ParameterChanges::default()
            };
            let values = TrackedModelValues {
                threshold: Vec::new(),
                recurrent_weight: Vec::new(),
                input_weight: Vec::new(),
                output_weight_by_neuron: Vec::new(),
                output_bias: None,
                context_keys: host_u64[layout.context_keys.offset
                    ..layout.context_keys.offset + layout.context_keys.len]
                    .to_vec(),
                context_observations: host_u32[layout.context_observations.offset
                    ..layout.context_observations.offset + layout.context_observations.len]
                    .to_vec(),
                context_embeddings: host_f32[layout.context_embeddings.offset
                    ..layout.context_embeddings.offset + layout.context_embeddings.len]
                    .to_vec(),
                context_output_weight: None,
                revision_increment: revision_increment[lane_index],
                statistics: statistics[lane_index].clone(),
            };
            snapshots.push((changes, values));
        }

        for (lane_index, lane) in lanes.iter().enumerate() {
            let base = lane_index * 5;
            let b = lane.buffers;
            for (marks, list, count, changed_count) in [
                (
                    b.changed_threshold_marks,
                    b.changed_threshold_list,
                    b.changed_threshold_count,
                    counts[base] as usize,
                ),
                (
                    b.changed_recurrent_marks,
                    b.changed_recurrent_list,
                    b.changed_recurrent_count,
                    counts[base + 1] as usize,
                ),
                (
                    b.changed_input_marks,
                    b.changed_input_list,
                    b.changed_input_count,
                    counts[base + 2] as usize,
                ),
                (
                    b.changed_output_marks,
                    b.changed_output_list,
                    b.changed_output_count,
                    counts[base + 3] as usize,
                ),
                (
                    b.changed_context_marks,
                    b.changed_context_list,
                    b.changed_context_count,
                    counts[base + 4] as usize,
                ),
            ] {
                self.clear_change_list(marks, list, count, changed_count)?;
            }
        }
        self.synchronize()?;
        Ok(snapshots)
    }

    fn read_change_list(&self, list: DeviceBuffer, count: DeviceBuffer) -> LeoResult<Vec<usize>> {
        let mut host_count = [0u32; 1];
        self.copy_from_device(count, &mut host_count)?;
        let n = host_count[0] as usize;
        let mut indices = vec![0u32; n];
        self.copy_from_device_prefix(list, &mut indices)?;
        Ok(indices.into_iter().map(|value| value as usize).collect())
    }

    fn launch_gather_f32(
        &self,
        source: DeviceBuffer,
        indices: DeviceBuffer,
        count: DeviceBuffer,
        output: DeviceBuffer,
        work: usize,
    ) -> LeoResult<()> {
        let mut source_ptr = source.pointer;
        let mut indices_ptr = indices.pointer;
        let mut count_ptr = count.pointer;
        let mut output_ptr = output.pointer;
        let mut params = [
            param(&mut source_ptr),
            param(&mut indices_ptr),
            param(&mut count_ptr),
            param(&mut output_ptr),
        ];
        self.launch(self.shared.kernels.gather_f32, work, THREADS, &mut params)
    }

    fn clear_change_list(
        &self,
        marks: DeviceBuffer,
        list: DeviceBuffer,
        count: DeviceBuffer,
        work: usize,
    ) -> LeoResult<()> {
        if work == 0 {
            self.memset_zero(count)?;
            return Ok(());
        }
        let mut marks_ptr = marks.pointer;
        let mut list_ptr = list.pointer;
        let mut n = as_u32("changed parameter count", work)?;
        let mut count_ptr = count.pointer;
        let mut params = [
            param(&mut marks_ptr),
            param(&mut list_ptr),
            param(&mut n),
            param(&mut count_ptr),
        ];
        self.launch(
            self.shared.kernels.clear_changed_marks,
            work,
            THREADS,
            &mut params,
        )
    }

    // Retained for the legacy per-wavefront delta/graph capability. The exact
    // v1 logical story-batch path no longer calls it because canonical updates
    // happen once, after complete private story trajectories.
    #[allow(dead_code)]
    fn clear_batch_delta_accumulators_async(&self) -> LeoResult<()> {
        let b = self.buffers;
        for buffer in [
            b.batch_delta_threshold,
            b.batch_delta_threshold_marks,
            b.batch_delta_threshold_count,
            b.batch_delta_recurrent,
            b.batch_delta_recurrent_marks,
            b.batch_delta_recurrent_count,
            b.batch_delta_input,
            b.batch_delta_input_marks,
            b.batch_delta_input_count,
            b.batch_delta_output,
            b.batch_delta_output_marks,
            b.batch_delta_output_count,
            b.batch_delta_output_bias,
            b.batch_delta_context_embedding,
            b.batch_delta_context_marks,
            b.batch_delta_context_count,
            b.batch_delta_context_observations,
            b.batch_delta_context_output,
        ] {
            self.memset_zero_async(buffer)?;
        }
        Ok(())
    }

    fn clear_device_changes_all(&self) -> LeoResult<()> {
        let b = self.buffers;
        for buffer in [
            b.batch_delta_threshold,
            b.batch_delta_threshold_marks,
            b.batch_delta_threshold_count,
            b.batch_delta_recurrent,
            b.batch_delta_recurrent_marks,
            b.batch_delta_recurrent_count,
            b.batch_delta_input,
            b.batch_delta_input_marks,
            b.batch_delta_input_count,
            b.batch_delta_output,
            b.batch_delta_output_marks,
            b.batch_delta_output_count,
            b.batch_delta_output_bias,
            b.batch_delta_context_embedding,
            b.batch_delta_context_marks,
            b.batch_delta_context_count,
            b.batch_delta_context_observations,
            b.batch_delta_context_output,
            b.changed_threshold_marks,
            b.changed_threshold_count,
            b.changed_recurrent_marks,
            b.changed_recurrent_count,
            b.changed_input_marks,
            b.changed_input_count,
            b.changed_output_marks,
            b.changed_output_count,
            b.changed_context_marks,
            b.changed_context_count,
        ] {
            self.memset_zero(buffer)?;
        }
        Ok(())
    }

    fn make_current(&self) -> LeoResult<()> {
        self.shared.driver.check(
            unsafe { (self.shared.driver.ctx_set_current)(self.shared.context) },
            "cuCtxSetCurrent",
        )
    }

    fn allocate_one<T>(&self) -> LeoResult<DeviceBuffer> {
        self.allocate_bytes(mem::size_of::<T>())
    }

    fn allocate_f32(&self, elements: usize) -> LeoResult<DeviceBuffer> {
        self.allocate_bytes(elements.saturating_mul(mem::size_of::<f32>()))
    }

    fn allocate_u32(&self, elements: usize) -> LeoResult<DeviceBuffer> {
        self.allocate_bytes(elements.saturating_mul(mem::size_of::<u32>()))
    }

    fn allocate_u64(&self, elements: usize) -> LeoResult<DeviceBuffer> {
        self.allocate_bytes(elements.saturating_mul(mem::size_of::<u64>()))
    }

    fn allocate_u8(&self, elements: usize) -> LeoResult<DeviceBuffer> {
        self.allocate_bytes(elements)
    }

    fn allocate_bytes(&self, bytes: usize) -> LeoResult<DeviceBuffer> {
        let bytes = bytes.max(1);
        let mut pointer = 0;
        self.shared.driver.check(
            unsafe { (self.shared.driver.mem_alloc)(&mut pointer, bytes) },
            "cuMemAlloc",
        )?;
        Ok(DeviceBuffer { pointer, bytes })
    }

    fn allocate_pinned_host(&self, bytes: usize) -> LeoResult<PinnedHostBuffer> {
        let bytes = bytes.max(1);
        let mut pointer = ptr::null_mut();
        self.shared.driver.check(
            unsafe { (self.shared.driver.mem_alloc_host)(&mut pointer, bytes) },
            "cuMemAllocHost",
        )?;
        Ok(PinnedHostBuffer { pointer, bytes })
    }

    fn ensure_batch_snapshot_capacity(
        &mut self,
        f32_elements: usize,
        u32_elements: usize,
        u64_elements: usize,
    ) -> LeoResult<()> {
        let f32_bytes = f32_elements.saturating_mul(mem::size_of::<f32>());
        if f32_bytes > self.batch_snapshot_device_f32.bytes {
            let replacement = self.allocate_f32(f32_elements.max(1))?;
            let old = std::mem::replace(&mut self.batch_snapshot_device_f32, replacement);
            if old.pointer != 0 {
                self.shared.driver.check(
                    unsafe { (self.shared.driver.mem_free)(old.pointer) },
                    "cuMemFree(batch snapshot f32)",
                )?;
            }
        }
        if f32_bytes > self.batch_snapshot_host_f32.bytes {
            let replacement = self.allocate_pinned_host(f32_bytes.max(1))?;
            let old = std::mem::replace(&mut self.batch_snapshot_host_f32, replacement);
            if !old.pointer.is_null() {
                self.shared.driver.check(
                    unsafe { (self.shared.driver.mem_free_host)(old.pointer) },
                    "cuMemFreeHost(batch snapshot f32)",
                )?;
            }
        }

        let u32_bytes = u32_elements.saturating_mul(mem::size_of::<u32>());
        if u32_bytes > self.batch_snapshot_device_u32.bytes {
            let replacement = self.allocate_u32(u32_elements.max(1))?;
            let old = std::mem::replace(&mut self.batch_snapshot_device_u32, replacement);
            if old.pointer != 0 {
                self.shared.driver.check(
                    unsafe { (self.shared.driver.mem_free)(old.pointer) },
                    "cuMemFree(batch snapshot u32)",
                )?;
            }
        }
        if u32_bytes > self.batch_snapshot_host_u32.bytes {
            let replacement = self.allocate_pinned_host(u32_bytes.max(1))?;
            let old = std::mem::replace(&mut self.batch_snapshot_host_u32, replacement);
            if !old.pointer.is_null() {
                self.shared.driver.check(
                    unsafe { (self.shared.driver.mem_free_host)(old.pointer) },
                    "cuMemFreeHost(batch snapshot u32)",
                )?;
            }
        }

        let u64_bytes = u64_elements.saturating_mul(mem::size_of::<u64>());
        if u64_bytes > self.batch_snapshot_device_u64.bytes {
            let replacement = self.allocate_u64(u64_elements.max(1))?;
            let old = std::mem::replace(&mut self.batch_snapshot_device_u64, replacement);
            if old.pointer != 0 {
                self.shared.driver.check(
                    unsafe { (self.shared.driver.mem_free)(old.pointer) },
                    "cuMemFree(batch snapshot u64)",
                )?;
            }
        }
        if u64_bytes > self.batch_snapshot_host_u64.bytes {
            let replacement = self.allocate_pinned_host(u64_bytes.max(1))?;
            let old = std::mem::replace(&mut self.batch_snapshot_host_u64, replacement);
            if !old.pointer.is_null() {
                self.shared.driver.check(
                    unsafe { (self.shared.driver.mem_free_host)(old.pointer) },
                    "cuMemFreeHost(batch snapshot u64)",
                )?;
            }
        }
        Ok(())
    }

    fn copy_pinned_to_device_async_on<T>(
        &self,
        buffer: DeviceBuffer,
        host: PinnedHostBuffer,
        elements: usize,
        stream: CuStream,
    ) -> LeoResult<()> {
        let bytes = elements.saturating_mul(mem::size_of::<T>());
        if bytes > buffer.bytes || bytes > host.bytes {
            return Err(LeoError::cuda(format!(
                "CUDA pinned upload of {bytes} bytes exceeds capacity device={} host={}",
                buffer.bytes, host.bytes
            )));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.memcpy_htod_async)(
                    buffer.pointer,
                    host.pointer as *const c_void,
                    bytes,
                    stream,
                )
            },
            "cuMemcpyHtoDAsync",
        )
    }

    fn copy_device_to_pinned_async_on<T>(
        &self,
        host: PinnedHostBuffer,
        buffer: DeviceBuffer,
        elements: usize,
        stream: CuStream,
    ) -> LeoResult<()> {
        let bytes = elements.saturating_mul(mem::size_of::<T>());
        if bytes > buffer.bytes || bytes > host.bytes {
            return Err(LeoError::cuda(format!(
                "CUDA pinned download of {bytes} bytes exceeds capacity device={} host={}",
                buffer.bytes, host.bytes
            )));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.memcpy_dtoh_async)(host.pointer, buffer.pointer, bytes, stream)
            },
            "cuMemcpyDtoHAsync",
        )
    }

    fn copy_device_to_device_async_on(
        &self,
        target: DeviceBuffer,
        source: DeviceBuffer,
        stream: CuStream,
    ) -> LeoResult<()> {
        if target.bytes != source.bytes {
            return Err(LeoError::cuda(format!(
                "CUDA device copy size mismatch target={} source={}",
                target.bytes, source.bytes
            )));
        }
        if target.bytes == 0 {
            return Ok(());
        }
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.memcpy_dtod_async)(
                    target.pointer,
                    source.pointer,
                    target.bytes,
                    stream,
                )
            },
            "cuMemcpyDtoDAsync",
        )
    }

    fn record_event(&self, event: CuEvent, stream: CuStream, label: &str) -> LeoResult<()> {
        self.shared.driver.check(
            unsafe { (self.shared.driver.event_record)(event, stream) },
            label,
        )
    }

    fn stream_wait_event(&self, stream: CuStream, event: CuEvent, label: &str) -> LeoResult<()> {
        self.shared.driver.check(
            unsafe { (self.shared.driver.stream_wait_event)(stream, event, 0) },
            label,
        )
    }

    fn event_synchronize(&self, event: CuEvent, label: &str) -> LeoResult<()> {
        self.shared.driver.check(
            unsafe { (self.shared.driver.event_synchronize)(event) },
            label,
        )
    }

    fn event_elapsed_ms(&self, start: CuEvent, end: CuEvent, label: &str) -> LeoResult<f32> {
        let mut milliseconds = 0.0f32;
        self.shared.driver.check(
            unsafe { (self.shared.driver.event_elapsed_time)(&mut milliseconds, start, end) },
            label,
        )?;
        Ok(milliseconds)
    }

    fn memset_zero(&self, buffer: DeviceBuffer) -> LeoResult<()> {
        if buffer.pointer == 0 || buffer.bytes == 0 {
            return Ok(());
        }
        if buffer.bytes % 4 == 0 {
            self.shared.driver.check(
                unsafe { (self.shared.driver.memset_d32)(buffer.pointer, 0, buffer.bytes / 4) },
                "cuMemsetD32",
            )
        } else {
            self.shared.driver.check(
                unsafe { (self.shared.driver.memset_d8)(buffer.pointer, 0, buffer.bytes) },
                "cuMemsetD8",
            )
        }
    }

    fn memset_zero_async(&self, buffer: DeviceBuffer) -> LeoResult<()> {
        if buffer.pointer == 0 || buffer.bytes == 0 {
            return Ok(());
        }
        if buffer.bytes % 4 == 0 {
            self.shared.driver.check(
                unsafe {
                    (self.shared.driver.memset_d32_async)(
                        buffer.pointer,
                        0,
                        buffer.bytes / 4,
                        self.compute_stream,
                    )
                },
                "cuMemsetD32Async",
            )
        } else {
            self.shared.driver.check(
                unsafe {
                    (self.shared.driver.memset_d8_async)(
                        buffer.pointer,
                        0,
                        buffer.bytes,
                        self.compute_stream,
                    )
                },
                "cuMemsetD8Async",
            )
        }
    }

    fn copy_to_device<T>(&self, buffer: DeviceBuffer, values: &[T]) -> LeoResult<()> {
        let bytes = values.len().saturating_mul(mem::size_of::<T>());
        if bytes > buffer.bytes {
            return Err(LeoError::cuda(format!(
                "CUDA upload of {bytes} bytes exceeds buffer capacity {}",
                buffer.bytes
            )));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.memcpy_htod)(
                    buffer.pointer,
                    values.as_ptr().cast::<c_void>(),
                    bytes,
                )
            },
            "cuMemcpyHtoD",
        )
    }

    fn copy_from_device<T>(&self, buffer: DeviceBuffer, values: &mut [T]) -> LeoResult<()> {
        let bytes = values.len().saturating_mul(mem::size_of::<T>());
        if bytes > buffer.bytes {
            return Err(LeoError::cuda(format!(
                "CUDA download of {bytes} bytes exceeds buffer capacity {}",
                buffer.bytes
            )));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.memcpy_dtoh)(
                    values.as_mut_ptr().cast::<c_void>(),
                    buffer.pointer,
                    bytes,
                )
            },
            "cuMemcpyDtoH",
        )
    }

    fn copy_from_device_prefix<T>(&self, buffer: DeviceBuffer, values: &mut [T]) -> LeoResult<()> {
        self.copy_from_device(buffer, values)
    }

    fn clear_phase_profile_counters_async(&self) -> LeoResult<()> {
        self.memset_zero_async(self.buffers.phase_profile_counters)
    }

    fn emit_phase_profile(&self, report: CudaPhaseProfileReport) -> LeoResult<()> {
        let CudaPhaseProfileReport {
            logical_lanes,
            fused_launches_total,
            sample_stride,
            planned_fused_blocks,
            geometry_skipped_samples,
            sampled_grid_blocks_max,
            skipped_grid_blocks_max,
        } = report;
        let mut counters = [0u64; CUDA_PHASE_PROFILE_COUNTER_COUNT];
        self.copy_from_device(self.buffers.phase_profile_counters, &mut counters)?;
        let samples = counters[CUDA_PHASE_PROFILE_SAMPLES];
        if samples == 0 {
            let reason = if geometry_skipped_samples > 0 {
                "profiled_kernel_cannot_match_production_grid"
            } else if fused_launches_total == 0 {
                "no_fused_wavefront_launches"
            } else {
                "no_sampled_fused_wavefront_launches"
            };
            eprintln!(
                "{{\"event\":\"cuda_phase_profile_skipped\",\"reason\":\"{}\",\"logical_lanes\":{},\"sample_stride\":{},\"planned_fused_blocks\":{},\"normal_capacity_blocks\":{},\"profiled_capacity_blocks\":{},\"geometry_skipped_samples\":{},\"sampled_grid_blocks_max\":{},\"skipped_grid_blocks_max\":{},\"fused_launches_total\":{}}}",
                reason,
                logical_lanes,
                sample_stride,
                planned_fused_blocks,
                self.shared.fused_blocks_256,
                self.shared.fused_profile_blocks_256,
                geometry_skipped_samples,
                sampled_grid_blocks_max,
                skipped_grid_blocks_max,
                fused_launches_total,
            );
            return Ok(());
        }
        let total_cycles = counters[1..]
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add);
        let percent = |index: usize| -> f64 {
            if total_cycles == 0 {
                0.0
            } else {
                counters[index] as f64 * 100.0 / total_cycles as f64
            }
        };
        let cycles_per_sample = if samples == 0 {
            0.0
        } else {
            total_cycles as f64 / samples as f64
        };

        eprintln!(
            concat!(
                "{{\"event\":\"cuda_phase_profile\",",
                "\"scope\":\"fused_wavefront_only\",",
                "\"logical_lanes\":{},\"sample_stride\":{},",
                "\"planned_fused_blocks\":{},",
                "\"normal_capacity_blocks\":{},\"profiled_capacity_blocks\":{},",
                "\"geometry_skipped_samples\":{},",
                "\"sampled_grid_blocks_max\":{},\"skipped_grid_blocks_max\":{},",
                "\"sampled_fused_launches\":{},\"fused_launches_total\":{},",
                "\"total_profiled_cycles\":{},\"cycles_per_sample\":{},",
                "\"pre_cycles\":{},\"pre_pct\":{},",
                "\"select_cycles\":{},\"select_pct\":{},",
                "\"post_select_cycles\":{},\"post_select_pct\":{},",
                "\"cache_surrogate_cycles\":{},\"cache_surrogate_pct\":{},",
                "\"post_core_cycles\":{},\"post_core_pct\":{},",
                "\"learning_signals_cycles\":{},\"learning_signals_pct\":{},",
                "\"post_deltas_cycles\":{},\"post_deltas_pct\":{},",
                "\"homeostasis_cycles\":{},\"homeostasis_pct\":{},",
                "\"capture_cycles\":{},\"capture_pct\":{}}}"
            ),
            logical_lanes,
            sample_stride,
            planned_fused_blocks,
            self.shared.fused_blocks_256,
            self.shared.fused_profile_blocks_256,
            geometry_skipped_samples,
            sampled_grid_blocks_max,
            skipped_grid_blocks_max,
            samples,
            fused_launches_total,
            total_cycles,
            cycles_per_sample,
            counters[CUDA_PHASE_PROFILE_PRE],
            percent(CUDA_PHASE_PROFILE_PRE),
            counters[CUDA_PHASE_PROFILE_SELECT],
            percent(CUDA_PHASE_PROFILE_SELECT),
            counters[CUDA_PHASE_PROFILE_POST_SELECT],
            percent(CUDA_PHASE_PROFILE_POST_SELECT),
            counters[CUDA_PHASE_PROFILE_CACHE_SURROGATE],
            percent(CUDA_PHASE_PROFILE_CACHE_SURROGATE),
            counters[CUDA_PHASE_PROFILE_POST_CORE],
            percent(CUDA_PHASE_PROFILE_POST_CORE),
            counters[CUDA_PHASE_PROFILE_LEARNING_SIGNALS],
            percent(CUDA_PHASE_PROFILE_LEARNING_SIGNALS),
            counters[CUDA_PHASE_PROFILE_POST_DELTAS],
            percent(CUDA_PHASE_PROFILE_POST_DELTAS),
            counters[CUDA_PHASE_PROFILE_HOMEOSTASIS],
            percent(CUDA_PHASE_PROFILE_HOMEOSTASIS),
            counters[CUDA_PHASE_PROFILE_CAPTURE],
            percent(CUDA_PHASE_PROFILE_CAPTURE),
        );
        Ok(())
    }

    fn clear_replay_profile_counters(&self) -> LeoResult<()> {
        self.memset_zero(self.buffers.replay_profile_counters)
    }

    fn read_replay_profile_counters(&self) -> LeoResult<[u64; CUDA_REPLAY_PROFILE_COUNTER_COUNT]> {
        let mut counters = [0u64; CUDA_REPLAY_PROFILE_COUNTER_COUNT];
        self.copy_from_device(self.buffers.replay_profile_counters, &mut counters)?;
        Ok(counters)
    }

    fn emit_replay_target_profile(&self, grid_blocks: u32, sample_stride: u64) -> LeoResult<()> {
        let counters = self.read_replay_profile_counters()?;
        let samples = counters[CUDA_REPLAY_PROFILE_SAMPLES];
        let total_cycles = counters[1..CUDA_REPLAY_PROFILE_COUNTER_COUNT]
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add);
        let percent = |index: usize| -> f64 {
            if total_cycles == 0 {
                0.0
            } else {
                counters[index] as f64 * 100.0 / total_cycles as f64
            }
        };
        eprintln!(
            concat!(
                "{{\"event\":\"cuda_replay_kernel_profile\",",
                "\"scope\":\"supervised_replay_targets\",",
                "\"sample_stride\":{},\"samples\":{},\"grid_blocks\":{},",
                "\"normal_capacity_blocks\":{},\"profiled_capacity_blocks\":{},",
                "\"total_profiled_cycles\":{},\"cycles_per_step\":{},",
                "\"pre_cycles\":{},\"pre_pct\":{},",
                "\"select_blocks_cycles\":{},\"select_blocks_pct\":{},",
                "\"select_global_cycles\":{},\"select_global_pct\":{},",
                "\"cache_surrogate_cycles\":{},\"cache_surrogate_pct\":{},",
                "\"recurrent_eligibility_cycles\":{},\"recurrent_eligibility_pct\":{},",
                "\"input_eligibility_cycles\":{},\"input_eligibility_pct\":{},",
                "\"post_emit_cycles\":{},\"post_emit_pct\":{},",
                "\"forward_cycles\":{},\"forward_pct\":{},",
                "\"learning_signals_cycles\":{},\"learning_signals_pct\":{},",
                "\"output_update_cycles\":{},\"output_update_pct\":{},",
                "\"context_update_cycles\":{},\"context_update_pct\":{},",
                "\"recurrent_update_cycles\":{},\"recurrent_update_pct\":{},",
                "\"input_update_cycles\":{},\"input_update_pct\":{},",
                "\"inhibitory_cycles\":{},\"inhibitory_pct\":{},",
                "\"homeostasis_cycles\":{},\"homeostasis_pct\":{},",
                "\"capture_cycles\":{},\"capture_pct\":{}}}"
            ),
            sample_stride,
            samples,
            grid_blocks,
            self.shared.replay_grid_blocks,
            self.shared.replay_profile_grid_blocks,
            total_cycles,
            if samples == 0 {
                0.0
            } else {
                total_cycles as f64 / samples as f64
            },
            counters[CUDA_REPLAY_PROFILE_PRE],
            percent(CUDA_REPLAY_PROFILE_PRE),
            counters[CUDA_REPLAY_PROFILE_SELECT_BLOCKS],
            percent(CUDA_REPLAY_PROFILE_SELECT_BLOCKS),
            counters[CUDA_REPLAY_PROFILE_SELECT_GLOBAL],
            percent(CUDA_REPLAY_PROFILE_SELECT_GLOBAL),
            counters[CUDA_REPLAY_PROFILE_CACHE_SURROGATE],
            percent(CUDA_REPLAY_PROFILE_CACHE_SURROGATE),
            counters[CUDA_REPLAY_PROFILE_RECURRENT_ELIGIBILITY],
            percent(CUDA_REPLAY_PROFILE_RECURRENT_ELIGIBILITY),
            counters[CUDA_REPLAY_PROFILE_INPUT_ELIGIBILITY],
            percent(CUDA_REPLAY_PROFILE_INPUT_ELIGIBILITY),
            counters[CUDA_REPLAY_PROFILE_POST_EMIT],
            percent(CUDA_REPLAY_PROFILE_POST_EMIT),
            counters[CUDA_REPLAY_PROFILE_FORWARD],
            percent(CUDA_REPLAY_PROFILE_FORWARD),
            counters[CUDA_REPLAY_PROFILE_LEARNING_SIGNALS],
            percent(CUDA_REPLAY_PROFILE_LEARNING_SIGNALS),
            counters[CUDA_REPLAY_PROFILE_OUTPUT_UPDATE],
            percent(CUDA_REPLAY_PROFILE_OUTPUT_UPDATE),
            counters[CUDA_REPLAY_PROFILE_CONTEXT_UPDATE],
            percent(CUDA_REPLAY_PROFILE_CONTEXT_UPDATE),
            counters[CUDA_REPLAY_PROFILE_RECURRENT_UPDATE],
            percent(CUDA_REPLAY_PROFILE_RECURRENT_UPDATE),
            counters[CUDA_REPLAY_PROFILE_INPUT_UPDATE],
            percent(CUDA_REPLAY_PROFILE_INPUT_UPDATE),
            counters[CUDA_REPLAY_PROFILE_INHIBITORY],
            percent(CUDA_REPLAY_PROFILE_INHIBITORY),
            counters[CUDA_REPLAY_PROFILE_HOMEOSTASIS],
            percent(CUDA_REPLAY_PROFILE_HOMEOSTASIS),
            counters[CUDA_REPLAY_PROFILE_CAPTURE],
            percent(CUDA_REPLAY_PROFILE_CAPTURE),
        );
        Ok(())
    }

    fn emit_frozen_prefix_profile(&self, grid_blocks: u32, sample_stride: u64) -> LeoResult<()> {
        let counters = self.read_replay_profile_counters()?;
        let samples = counters[CUDA_FROZEN_PROFILE_SAMPLES];
        let total_cycles = counters[1..CUDA_FROZEN_PROFILE_COUNTER_COUNT]
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add);
        let percent = |index: usize| -> f64 {
            if total_cycles == 0 {
                0.0
            } else {
                counters[index] as f64 * 100.0 / total_cycles as f64
            }
        };
        eprintln!(
            concat!(
                "{{\"event\":\"cuda_frozen_kernel_profile\",",
                "\"scope\":\"replay_prefix_reconstruction\",",
                "\"sample_stride\":{},\"samples\":{},\"grid_blocks\":{},",
                "\"normal_capacity_blocks\":{},\"profiled_capacity_blocks\":{},",
                "\"total_profiled_cycles\":{},\"cycles_per_step\":{},",
                "\"pre_cycles\":{},\"pre_pct\":{},",
                "\"select_blocks_cycles\":{},\"select_blocks_pct\":{},",
                "\"select_global_cycles\":{},\"select_global_pct\":{},",
                "\"post_emit_cycles\":{},\"post_emit_pct\":{}}}"
            ),
            sample_stride,
            samples,
            grid_blocks,
            self.shared.frozen_grid_blocks,
            self.shared.frozen_profile_grid_blocks,
            total_cycles,
            if samples == 0 {
                0.0
            } else {
                total_cycles as f64 / samples as f64
            },
            counters[CUDA_FROZEN_PROFILE_PRE],
            percent(CUDA_FROZEN_PROFILE_PRE),
            counters[CUDA_FROZEN_PROFILE_SELECT_BLOCKS],
            percent(CUDA_FROZEN_PROFILE_SELECT_BLOCKS),
            counters[CUDA_FROZEN_PROFILE_SELECT_GLOBAL],
            percent(CUDA_FROZEN_PROFILE_SELECT_GLOBAL),
            counters[CUDA_FROZEN_PROFILE_POST_EMIT],
            percent(CUDA_FROZEN_PROFILE_POST_EMIT),
        );
        Ok(())
    }

    // Retained as a legacy execution capability for compatibility and future
    // experiments. Exact v1 story batching must not invoke a per-byte canonical
    // apply/reset barrier.
    #[allow(dead_code)]
    fn launch_sparse_apply_and_reset(&mut self, blocks: u32, threads: u32) -> LeoResult<bool> {
        let plan = (blocks.max(1), threads.max(32));
        if self.sparse_apply_graph_plan != Some(plan) {
            self.destroy_sparse_apply_graph();
            self.sparse_apply_graph_disabled = false;
        }
        if let Some(graph) = self.sparse_apply_graph {
            if let Some(launch) = self.shared.driver.graph_launch {
                self.shared.driver.check(
                    unsafe { launch(graph, self.compute_stream) },
                    "cuGraphLaunch(sparse apply)",
                )?;
                return Ok(true);
            }
            self.destroy_sparse_apply_graph();
        }

        if !self.sparse_apply_graph_disabled {
            if let Some(graph) = self.capture_sparse_apply_graph(plan.0, plan.1)? {
                self.sparse_apply_graph = Some(graph);
                self.sparse_apply_graph_plan = Some(plan);
                if let Some(launch) = self.shared.driver.graph_launch {
                    self.shared.driver.check(
                        unsafe { launch(graph, self.compute_stream) },
                        "cuGraphLaunch(sparse apply)",
                    )?;
                    return Ok(true);
                }
            } else {
                // Missing/unsupported graph APIs are an execution capability,
                // not an error. Avoid retrying capture every wavefront.
                self.sparse_apply_graph_disabled = true;
            }
        }
        self.launch_sparse_apply_and_reset_direct(plan.0, plan.1)?;
        Ok(false)
    }

    fn launch_device_batch_fixed_merge(&mut self, lane_count: usize) -> LeoResult<()> {
        if lane_count == 0 {
            return Ok(());
        }
        self.make_current()?;
        let mut model_pointers = self.buffers.persistent_pointer_table.pointer;
        let mut lane_pointer_addresses = self.buffers.batch_pointer_tables.pointer;
        let mut lane_count_value = as_u32("GPU device batch merge lane count", lane_count)?;
        let mut params = [
            param(&mut model_pointers),
            param(&mut lane_pointer_addresses),
            param(&mut lane_count_value),
        ];
        let work = self
            .model
            .neuron_count()
            .max(self.model.recurrent.weight.len())
            .max(self.model.input.weights.len())
            .max(self.model.context.output_weights.len())
            .max(OUTPUT_CLASSES);
        self.launch(
            self.shared.kernels.merge_shared_lane_fixed_parameters,
            work,
            THREADS,
            &mut params,
        )
    }

    fn launch_sparse_apply_and_reset_direct(&self, blocks: u32, threads: u32) -> LeoResult<()> {
        let mut model_pointers = self.buffers.persistent_pointer_table.pointer;
        let mut delta_pointers = self.buffers.batch_delta_pointer_table.pointer;
        let mut apply_parameters = [param(&mut model_pointers), param(&mut delta_pointers)];
        self.launch_exact(
            self.shared.kernels.apply_shared_wavefront_deltas,
            blocks.max(1),
            threads.max(32),
            &mut apply_parameters,
        )?;

        let mut delta_pointers = self.buffers.batch_delta_pointer_table.pointer;
        let mut reset_parameters = [param(&mut delta_pointers)];
        self.launch_exact(
            self.shared.kernels.reset_shared_wavefront_deltas,
            1,
            32,
            &mut reset_parameters,
        )
    }

    fn capture_sparse_apply_graph(
        &self,
        blocks: u32,
        threads: u32,
    ) -> LeoResult<Option<CuGraphExec>> {
        let driver = &self.shared.driver;
        let (Some(begin_capture), Some(end_capture), Some(instantiate), Some(graph_destroy)) = (
            driver.stream_begin_capture,
            driver.stream_end_capture,
            driver.graph_instantiate_with_flags,
            driver.graph_destroy,
        ) else {
            return Ok(None);
        };
        if driver.graph_launch.is_none() || driver.graph_exec_destroy.is_none() {
            return Ok(None);
        }

        // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1. Only this runtime thread
        // submits work to the stream while the tiny apply/reset graph is built.
        let begin = unsafe { begin_capture(self.compute_stream, 1) };
        if begin != CUDA_SUCCESS {
            return Ok(None);
        }
        if let Err(error) = self.launch_sparse_apply_and_reset_direct(blocks, threads) {
            let mut abandoned = ptr::null_mut();
            let _ = unsafe { end_capture(self.compute_stream, &mut abandoned) };
            if !abandoned.is_null() {
                unsafe { graph_destroy(abandoned) };
            }
            return Err(error);
        }

        let mut graph = ptr::null_mut();
        let end = unsafe { end_capture(self.compute_stream, &mut graph) };
        if end != CUDA_SUCCESS || graph.is_null() {
            if !graph.is_null() {
                unsafe { graph_destroy(graph) };
            }
            return Ok(None);
        }
        let mut executable = ptr::null_mut();
        let instantiate_result = unsafe { instantiate(&mut executable, graph, 0) };
        unsafe { graph_destroy(graph) };
        if instantiate_result != CUDA_SUCCESS || executable.is_null() {
            return Ok(None);
        }
        Ok(Some(executable))
    }

    fn destroy_sparse_apply_graph(&mut self) {
        let Some(graph) = self.sparse_apply_graph.take() else {
            self.sparse_apply_graph_plan = None;
            return;
        };
        if let Some(destroy) = self.shared.driver.graph_exec_destroy {
            unsafe {
                destroy(graph);
            }
        }
        self.sparse_apply_graph_plan = None;
    }

    fn synchronize(&self) -> LeoResult<()> {
        self.shared.driver.check(
            unsafe { (self.shared.driver.stream_synchronize)(self.compute_stream) },
            "cuStreamSynchronize",
        )
    }

    fn estimated_fused_occupancy(&self, plan: CudaExecutionPlan) -> f32 {
        if !self.shared.cooperative_launch
            || plan.fused_wavefront_blocks == 0
            || self.shared.multiprocessor_count == 0
            || self.shared.max_threads_per_sm == 0
        {
            return 0.0;
        }
        let blocks_per_sm = (plan.fused_wavefront_blocks as f32
            / self.shared.multiprocessor_count as f32)
            .ceil()
            .max(1.0);
        ((blocks_per_sm * plan.fused_wavefront_threads as f32)
            / self.shared.max_threads_per_sm as f32)
            .clamp(0.0, 1.0)
    }

    fn launch_cooperative_exact(
        &self,
        function: CuFunction,
        grid_x: c_uint,
        block_x: c_uint,
        parameters: &mut [*mut c_void],
    ) -> LeoResult<()> {
        if grid_x == 0 || block_x == 0 {
            return Err(LeoError::cuda("invalid CUDA cooperative launch dimensions"));
        }
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.launch_cooperative_kernel)(
                    function,
                    grid_x,
                    1,
                    1,
                    block_x,
                    1,
                    1,
                    0,
                    self.compute_stream,
                    parameters.as_mut_ptr(),
                )
            },
            "cuLaunchCooperativeKernel",
        )
    }

    fn launch(
        &self,
        function: CuFunction,
        work_items: usize,
        threads: c_uint,
        parameters: &mut [*mut c_void],
    ) -> LeoResult<()> {
        if work_items == 0 {
            return Ok(());
        }
        let grid_x = (work_items as u64).div_ceil(u64::from(threads)) as c_uint;
        self.launch_exact(function, grid_x, threads, parameters)
    }

    fn launch_exact(
        &self,
        function: CuFunction,
        grid_x: c_uint,
        block_x: c_uint,
        parameters: &mut [*mut c_void],
    ) -> LeoResult<()> {
        self.shared.driver.check(
            unsafe {
                (self.shared.driver.launch_kernel)(
                    function,
                    grid_x,
                    1,
                    1,
                    block_x,
                    1,
                    1,
                    0,
                    self.compute_stream,
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        let _ = self.shared.driver.check(
            unsafe { (self.shared.driver.ctx_set_current)(self.shared.context) },
            "cuCtxSetCurrent(runtime drop)",
        );
        if !self.compute_stream.is_null() {
            unsafe {
                (self.shared.driver.stream_synchronize)(self.compute_stream);
            }
        }
        if !self.transfer_stream.is_null() {
            unsafe {
                (self.shared.driver.stream_synchronize)(self.transfer_stream);
            }
        }
        self.destroy_sparse_apply_graph();
        for buffer in self.buffers.all() {
            if buffer.pointer != 0 {
                unsafe {
                    (self.shared.driver.mem_free)(buffer.pointer);
                }
            }
        }
        for buffer in [
            self.batch_snapshot_device_f32,
            self.batch_snapshot_device_u32,
            self.batch_snapshot_device_u64,
        ] {
            if buffer.pointer != 0 {
                unsafe {
                    (self.shared.driver.mem_free)(buffer.pointer);
                }
            }
        }
        for buffer in [
            self.host_batch_pointer_tables,
            self.host_batch_step_buffers,
            self.host_batch_step_counts,
            self.host_batch_base_ticks,
            self.host_batch_change_counts,
            self.batch_snapshot_host_f32,
            self.batch_snapshot_host_u32,
            self.batch_snapshot_host_u64,
        ] {
            if !buffer.pointer.is_null() {
                unsafe {
                    (self.shared.driver.mem_free_host)(buffer.pointer);
                }
            }
        }
        for event in self.events.upload_ready.drain(..) {
            if !event.is_null() {
                unsafe {
                    (self.shared.driver.event_destroy)(event);
                }
            }
        }
        for event in [
            self.events.compute_done,
            self.events.h2d_start,
            self.events.h2d_end,
            self.events.compute_start,
            self.events.compute_end,
            self.events.d2h_start,
            self.events.d2h_end,
        ] {
            if !event.is_null() {
                unsafe {
                    (self.shared.driver.event_destroy)(event);
                }
            }
        }
        if !self.transfer_stream.is_null() {
            unsafe {
                (self.shared.driver.stream_destroy)(self.transfer_stream);
            }
            self.transfer_stream = ptr::null_mut();
        }
        if !self.compute_stream.is_null() {
            unsafe {
                (self.shared.driver.stream_destroy)(self.compute_stream);
            }
            self.compute_stream = ptr::null_mut();
        }
    }
}

/// Exact v1 GPU logical story batch.
///
/// Stories share immutable topology/configuration but own recurrent state and
/// mutable learned parameters for their complete trajectories. Only after every
/// story finishes are sparse worker deltas reduced with the same canonical
/// batch-end mean rule as the CPU reference. Physical CUDA chunking therefore
/// changes launch geometry only, never parameter visibility or update order.
fn device_pointer_offset(pointer: CuDevicePtr, bytes: usize) -> LeoResult<CuDevicePtr> {
    pointer
        .checked_add(bytes as u64)
        .ok_or_else(|| LeoError::cuda("CUDA device pointer offset overflow"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WavefrontLaunchKind {
    Fused {
        phase_profile_sampled: bool,
        phase_profile_geometry_skipped: bool,
        grid_blocks: u32,
    },
    Direct {
        phase_launches: u64,
    },
}

#[derive(Clone, Copy)]
struct WavefrontChunkLaunch {
    lane_start: usize,
    physical_lane_count: usize,
    step_index: usize,
    model_block_count: u32,
    learning_trace: bool,
    strength: f32,
    batch_scale: f32,
    execution_plan: CudaExecutionPlan,
    phase_profile: bool,
}

#[derive(Clone, Copy)]
struct PersistentWavefrontChunkLaunch {
    lane_start: usize,
    physical_lane_count: usize,
    max_step_count: usize,
    model_block_count: u32,
    learning_trace: bool,
    strength: f32,
    batch_scale: f32,
    execution_plan: CudaExecutionPlan,
}

#[derive(Clone, Copy, Debug)]
struct PersistentWavefrontLaunch {
    grid_blocks: u32,
    blocks_per_lane: u32,
    grouped: bool,
}

fn launch_shared_wavefront_persistent_chunk(
    coordinator: &CudaRuntime,
    launch: PersistentWavefrontChunkLaunch,
) -> LeoResult<PersistentWavefrontLaunch> {
    let PersistentWavefrontChunkLaunch {
        lane_start,
        physical_lane_count,
        max_step_count,
        model_block_count,
        learning_trace,
        strength,
        batch_scale,
        execution_plan,
    } = launch;
    let lane_count = as_u32("GPU persistent physical lane chunk", physical_lane_count)?;
    let pointer_offset = lane_start.saturating_mul(mem::size_of::<CuDevicePtr>());
    let count_offset = lane_start.saturating_mul(mem::size_of::<u32>());
    let tick_offset = lane_start.saturating_mul(mem::size_of::<u64>());
    let pointer_tables_base = device_pointer_offset(
        coordinator.buffers.batch_pointer_tables.pointer,
        pointer_offset,
    )?;
    let step_buffers_base = device_pointer_offset(
        coordinator.buffers.batch_step_buffers.pointer,
        pointer_offset,
    )?;
    let step_counts_base =
        device_pointer_offset(coordinator.buffers.batch_step_counts.pointer, count_offset)?;
    let base_ticks_base =
        device_pointer_offset(coordinator.buffers.batch_base_ticks.pointer, tick_offset)?;

    let select_work = lane_count
        .checked_mul(model_block_count.max(1))
        .ok_or_else(|| LeoError::cuda("GPU persistent shared wavefront grid overflow"))?;
    let useful_blocks = select_work.max(lane_count).max(1);
    let planned_blocks = execution_plan
        .fused_wavefront_blocks
        .min(useful_blocks)
        .max(1);
    let grouped_budget = planned_blocks.min(coordinator.shared.shared_grouped_blocks_256);
    let grouped_blocks_per_lane = if cuda_shared_grouped_enabled() && lane_count > 0 {
        grouped_budget / lane_count
    } else {
        0
    };

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_buffers_ptr = step_buffers_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut max_step_count_value = as_u32("GPU persistent shared step count", max_step_count)?;
    let mut block_count_value = model_block_count;
    let mut learning_value = u32::from(learning_trace);
    let mut raw_strength = strength;
    let mut scale = batch_scale;
    let mut delta_pointers = coordinator.buffers.batch_delta_pointer_table.pointer;

    if grouped_blocks_per_lane > 0 {
        // Cooperative grids must keep every CTA resident at a grid barrier.
        // Launch an exact multiple of lane_count so every logical story owns
        // the same number of CTAs and lane-local work has a stable stride.
        let grid_blocks = lane_count
            .checked_mul(grouped_blocks_per_lane)
            .ok_or_else(|| LeoError::cuda("GPU grouped persistent grid overflow"))?;
        let mut blocks_per_lane_value = grouped_blocks_per_lane;
        let mut parameters = [
            param(&mut pointer_tables_ptr),
            param(&mut step_buffers_ptr),
            param(&mut step_counts_ptr),
            param(&mut base_ticks_ptr),
            param(&mut lane_count_value),
            param(&mut max_step_count_value),
            param(&mut block_count_value),
            param(&mut learning_value),
            param(&mut raw_strength),
            param(&mut scale),
            param(&mut delta_pointers),
            param(&mut blocks_per_lane_value),
        ];
        coordinator.launch_cooperative_exact(
            coordinator
                .shared
                .kernels
                .shared_wavefront_persistent_grouped,
            grid_blocks,
            SHARED_BATCH_THREADS,
            &mut parameters,
        )?;
        return Ok(PersistentWavefrontLaunch {
            grid_blocks,
            blocks_per_lane: grouped_blocks_per_lane,
            grouped: true,
        });
    }

    let grid_blocks = planned_blocks
        .min(coordinator.shared.shared_persistent_blocks_256)
        .max(1);
    let mut parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_buffers_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut max_step_count_value),
        param(&mut block_count_value),
        param(&mut learning_value),
        param(&mut raw_strength),
        param(&mut scale),
        param(&mut delta_pointers),
    ];
    coordinator.launch_cooperative_exact(
        coordinator.shared.kernels.shared_wavefront_persistent,
        grid_blocks,
        SHARED_BATCH_THREADS,
        &mut parameters,
    )?;
    Ok(PersistentWavefrontLaunch {
        grid_blocks,
        blocks_per_lane: 1,
        grouped: false,
    })
}

fn launch_shared_wavefront_chunk(
    coordinator: &CudaRuntime,
    launch: WavefrontChunkLaunch,
) -> LeoResult<WavefrontLaunchKind> {
    let WavefrontChunkLaunch {
        lane_start,
        physical_lane_count,
        step_index,
        model_block_count,
        learning_trace,
        strength,
        batch_scale,
        execution_plan,
        phase_profile,
    } = launch;
    let lane_count = as_u32("GPU physical lane chunk", physical_lane_count)?;
    let pointer_offset = lane_start.saturating_mul(mem::size_of::<CuDevicePtr>());
    let count_offset = lane_start.saturating_mul(mem::size_of::<u32>());
    let tick_offset = lane_start.saturating_mul(mem::size_of::<u64>());
    let pointer_tables_base = device_pointer_offset(
        coordinator.buffers.batch_pointer_tables.pointer,
        pointer_offset,
    )?;
    let step_buffers_base = device_pointer_offset(
        coordinator.buffers.batch_step_buffers.pointer,
        pointer_offset,
    )?;
    let step_counts_base =
        device_pointer_offset(coordinator.buffers.batch_step_counts.pointer, count_offset)?;
    let base_ticks_base =
        device_pointer_offset(coordinator.buffers.batch_base_ticks.pointer, tick_offset)?;
    let step = as_u32("GPU shared wavefront step", step_index)?;
    let learning = u32::from(learning_trace);

    // The fused kernel uses cooperative grid-wide barriers to preserve every
    // phase boundary from the fallback launch sequence. Keep its block size at
    // 256 so the forward-reduction order remains byte-for-byte compatible with
    // the established FP32 execution semantics.
    if coordinator.shared.cooperative_launch
        && execution_plan.fused_wavefront_blocks > 0
        && execution_plan.fused_wavefront_threads == SHARED_BATCH_THREADS
    {
        let select_work = lane_count
            .checked_mul(model_block_count.max(1))
            .ok_or_else(|| LeoError::cuda("GPU fused wavefront grid overflow"))?;
        let useful_blocks = select_work.max(lane_count).max(1);
        let grid_blocks = execution_plan
            .fused_wavefront_blocks
            .min(coordinator.shared.fused_blocks_256)
            .min(useful_blocks)
            .max(1);

        let mut pointer_tables_ptr = pointer_tables_base;
        let mut step_buffers_ptr = step_buffers_base;
        let mut step_counts_ptr = step_counts_base;
        let mut base_ticks_ptr = base_ticks_base;
        let mut lane_count_value = lane_count;
        let mut step_index_value = step;
        let mut block_count_value = model_block_count;
        let mut learning_value = learning;
        let mut raw_strength = strength;
        let mut scale = batch_scale;
        let mut delta_pointers = coordinator.buffers.batch_delta_pointer_table.pointer;

        // clock64 instrumentation increases register pressure and can reduce
        // cooperative residency. A sampled launch is comparable only when the
        // profiled kernel can run the exact same grid width as production; do
        // not silently shrink the grid just to obtain a sample.
        let phase_profile_sampled =
            phase_profile && coordinator.shared.fused_profile_blocks_256 >= grid_blocks;
        let phase_profile_geometry_skipped = phase_profile && !phase_profile_sampled;
        if phase_profile_sampled {
            let mut phase_profile_counters = coordinator.buffers.phase_profile_counters.pointer;
            let mut parameters = [
                param(&mut pointer_tables_ptr),
                param(&mut step_buffers_ptr),
                param(&mut step_counts_ptr),
                param(&mut base_ticks_ptr),
                param(&mut lane_count_value),
                param(&mut step_index_value),
                param(&mut block_count_value),
                param(&mut learning_value),
                param(&mut raw_strength),
                param(&mut scale),
                param(&mut delta_pointers),
                param(&mut phase_profile_counters),
            ];
            coordinator.launch_cooperative_exact(
                coordinator.shared.kernels.shared_wavefront_fused_profiled,
                grid_blocks,
                SHARED_BATCH_THREADS,
                &mut parameters,
            )?;
        } else {
            let mut parameters = [
                param(&mut pointer_tables_ptr),
                param(&mut step_buffers_ptr),
                param(&mut step_counts_ptr),
                param(&mut base_ticks_ptr),
                param(&mut lane_count_value),
                param(&mut step_index_value),
                param(&mut block_count_value),
                param(&mut learning_value),
                param(&mut raw_strength),
                param(&mut scale),
                param(&mut delta_pointers),
            ];
            coordinator.launch_cooperative_exact(
                coordinator.shared.kernels.shared_wavefront_fused,
                grid_blocks,
                SHARED_BATCH_THREADS,
                &mut parameters,
            )?;
        }
        return Ok(WavefrontLaunchKind::Fused {
            phase_profile_sampled,
            phase_profile_geometry_skipped,
            grid_blocks,
        });
    }

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_buffers_ptr = step_buffers_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut learning_value = learning;
    let mut pre_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_buffers_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
        param(&mut learning_value),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_wavefront_pre,
        lane_count,
        THREADS,
        &mut pre_parameters,
    )?;

    let select_grid = lane_count
        .checked_mul(model_block_count)
        .ok_or_else(|| LeoError::cuda("GPU shared select grid overflow"))?;
    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut block_count_value = model_block_count;
    let mut select_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
        param(&mut block_count_value),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_select_blocks,
        select_grid,
        THREADS,
        &mut select_parameters,
    )?;

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut post_select_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_post_select,
        lane_count,
        SHARED_BATCH_THREADS,
        &mut post_select_parameters,
    )?;

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut surrogate_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_cache_surrogate_worklist,
        lane_count,
        THREADS,
        &mut surrogate_parameters,
    )?;

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_buffers_ptr = step_buffers_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut learning_value = learning;
    let mut core_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_buffers_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
        param(&mut learning_value),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_post_core,
        lane_count,
        SHARED_BATCH_THREADS,
        &mut core_parameters,
    )?;

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_buffers_ptr = step_buffers_base;
    let mut step_counts_ptr = step_counts_base;
    let mut base_ticks_ptr = base_ticks_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut raw_strength = strength;
    let mut signal_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_buffers_ptr),
        param(&mut step_counts_ptr),
        param(&mut base_ticks_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
        param(&mut raw_strength),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_learning_signals_worklist,
        lane_count,
        THREADS,
        &mut signal_parameters,
    )?;

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_buffers_ptr = step_buffers_base;
    let mut step_counts_ptr = step_counts_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut learning_value = learning;
    let mut raw_strength = strength;
    let mut scale = batch_scale;
    let mut delta_pointers = coordinator.buffers.batch_delta_pointer_table.pointer;
    let mut delta_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_buffers_ptr),
        param(&mut step_counts_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
        param(&mut learning_value),
        param(&mut raw_strength),
        param(&mut scale),
        param(&mut delta_pointers),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_post_deltas,
        lane_count,
        SHARED_BATCH_THREADS,
        &mut delta_parameters,
    )?;

    if learning_trace {
        let mut pointer_tables_ptr = pointer_tables_base;
        let mut step_counts_ptr = step_counts_base;
        let mut base_ticks_ptr = base_ticks_base;
        let mut lane_count_value = lane_count;
        let mut step_index_value = step;
        let mut learning_value = learning;
        let mut scale = batch_scale;
        let mut delta_pointers = coordinator.buffers.batch_delta_pointer_table.pointer;
        let mut homeostasis_parameters = [
            param(&mut pointer_tables_ptr),
            param(&mut step_counts_ptr),
            param(&mut base_ticks_ptr),
            param(&mut lane_count_value),
            param(&mut step_index_value),
            param(&mut learning_value),
            param(&mut scale),
            param(&mut delta_pointers),
        ];
        coordinator.launch_exact(
            coordinator.shared.kernels.shared_homeostasis_worklist,
            lane_count,
            THREADS,
            &mut homeostasis_parameters,
        )?;
    }

    let mut pointer_tables_ptr = pointer_tables_base;
    let mut step_buffers_ptr = step_buffers_base;
    let mut step_counts_ptr = step_counts_base;
    let mut lane_count_value = lane_count;
    let mut step_index_value = step;
    let mut capture_parameters = [
        param(&mut pointer_tables_ptr),
        param(&mut step_buffers_ptr),
        param(&mut step_counts_ptr),
        param(&mut lane_count_value),
        param(&mut step_index_value),
    ];
    coordinator.launch_exact(
        coordinator.shared.kernels.shared_capture_training_step,
        lane_count,
        32,
        &mut capture_parameters,
    )?;
    Ok(WavefrontLaunchKind::Direct {
        phase_launches: if learning_trace { 9 } else { 8 },
    })
}

fn commit_shared_story_batch_update(
    coordinator: &mut CudaRuntime,
    lanes: &mut [CudaBatchLane],
    sync_changes: &ParameterChanges,
) -> LeoResult<()> {
    let sync_update = PackedSparseModelUpdate::from_model(&coordinator.model, sync_changes)?;
    coordinator.commit_packed_host_model(&sync_update, lanes)?;
    coordinator.accumulate_parameter_changes(sync_changes);
    Ok(())
}

fn combine_parameter_changes(
    mut fixed: ParameterChanges,
    context: ParameterChanges,
) -> ParameterChanges {
    fixed.context_slots = context.context_slots;
    fixed.output_bias_dirty |= context.output_bias_dirty;
    fixed.context_output_dirty |= context.context_output_dirty;
    fixed
}

pub(crate) fn train_story_batch_shared_device(
    coordinator: &mut CudaRuntime,
    lanes: &mut [CudaBatchLane],
    stories: &[Vec<u8>],
    permission: Permission,
    retain_story_deltas: bool,
) -> LeoResult<CudaStoryBatchReport> {
    if stories.is_empty() {
        return Err(LeoError::cuda("GPU shared story batch cannot be empty"));
    }
    if lanes.len() != stories.len() {
        return Err(LeoError::cuda(format!(
            "GPU shared story batch lane count {} does not match story count {}",
            lanes.len(),
            stories.len()
        )));
    }
    if lanes.len() > GPU_STORY_BATCH_MAX_LANES {
        return Err(LeoError::cuda(format!(
            "GPU shared story batch supports at most {GPU_STORY_BATCH_MAX_LANES} lanes; got {}",
            lanes.len()
        )));
    }

    coordinator.ensure_device_model_current()?;
    coordinator.make_current()?;
    let canonical_revision = coordinator.model.parameter_revision;
    for lane in lanes.iter_mut() {
        // Constant logical worker counts stay fully device-resident between
        // batches. A lane used after sitting out a prior batch is repaired once
        // from the current canonical image before story-local learning starts.
        if lane.model_revision != canonical_revision {
            coordinator.copy_canonical_parameters_to_batch_lane_async(lane)?;
        }
        coordinator.reset_shared_batch_lane(lane)?;
        // Keep the lane invalid until the successful batch-end mean has been
        // copied back. Any early CUDA error therefore forces a repair next use.
        lane.model_revision = u64::MAX;
    }

    let story_steps = stories
        .iter()
        .map(|story| {
            if story.is_empty() {
                return Vec::new();
            }
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
            steps
        })
        .collect::<Vec<_>>();
    let tuning_work_units = story_steps
        .iter()
        .map(|steps| steps.len() as u64)
        .sum::<u64>();
    let tuning_started = Instant::now();
    let execution_plan = coordinator.execution_tuner.plan_for(lanes.len());
    let physical_lane_chunk = execution_plan.physical_lane_chunk.max(1);

    let mut offsets = vec![0usize; lanes.len()];
    let mut story_metrics = story_steps
        .iter()
        .map(|steps| Vec::with_capacity(steps.len()))
        .collect::<Vec<_>>();
    let learning_trace = !matches!(permission, Permission::Frozen);
    let strength = permission.strength(&coordinator.model.config);
    // Retained in kernel launch ABI for compatibility with cached/source-tested
    // launch plumbing. Exact story-local learning no longer applies this scale
    // per byte; `apply_mean_deltas` performs the only 1/workers reduction.
    let batch_scale = 1.0f32 / lanes.len() as f32;
    let mut lane_learned_steps = vec![0u64; lanes.len()];
    let mut lane_context_learning = vec![false; lanes.len()];
    let mut lane_statistics = vec![StatisticsDelta::default(); lanes.len()];

    let profile_batch = coordinator.execution_tuner.should_profile_batch();
    let mut telemetry = CudaBatchTelemetry {
        theoretical_occupancy: coordinator.estimated_fused_occupancy(execution_plan),
        ..CudaBatchTelemetry::default()
    };
    let phase_profile_sample_stride = coordinator.phase_profile_sample_stride;
    let mut phase_profile_launch_sequence = 0u64;
    let mut phase_profile_geometry_skips = 0u64;
    let mut phase_profile_sampled_grid_max = 0u32;
    let mut phase_profile_skipped_grid_max = 0u32;
    if phase_profile_sample_stride.is_some() {
        coordinator.clear_phase_profile_counters_async()?;
    }

    while offsets
        .iter()
        .zip(&story_steps)
        .any(|(&offset, steps)| offset < steps.len())
    {
        let mut pointer_tables = Vec::with_capacity(lanes.len());
        let mut step_buffers = Vec::with_capacity(lanes.len());
        let mut step_counts = Vec::with_capacity(lanes.len());
        let mut base_ticks = Vec::with_capacity(lanes.len());
        let mut metadata = Vec::with_capacity(lanes.len());
        let mut max_chunk_steps = 0usize;
        let mut chunk_h2d_bytes = 0u64;
        let mut chunk_d2h_bytes = 0u64;

        // Prepare every pinned host lane first. Uploads are intentionally
        // deferred until physical groups are scheduled so the transfer stream
        // can stage group N+1 while the compute stream executes group N.
        for lane_index in 0..lanes.len() {
            let lane = &mut lanes[lane_index];
            let steps = &story_steps[lane_index];
            let start = offsets[lane_index];
            let end = start
                .saturating_add(TRAINING_STEP_BATCH_CAPACITY)
                .min(steps.len());
            let chunk = &steps[start..end];
            max_chunk_steps = max_chunk_steps.max(chunk.len());
            let base_tick = lane.current_tick;
            let mut device_steps = Vec::with_capacity(chunk.len());
            let mut lane_metadata = Vec::with_capacity(chunk.len());

            for (record_index, &(symbol, target)) in chunk.iter().enumerate() {
                if !is_input_symbol(symbol) {
                    return Err(LeoError::cuda(format!("invalid input symbol: {symbol}")));
                }
                let target_index = target
                    .map(|value| {
                        output_symbol_to_index(value).ok_or_else(|| {
                            LeoError::cuda(format!("invalid output target: {value}"))
                        })
                    })
                    .transpose()?;
                let tick = base_tick.saturating_add(record_index as u64);
                let context_enabled = matches!(permission, Permission::Frozen)
                    || context_survives_dropout(
                        coordinator.model.config.model.seed,
                        tick,
                        symbol,
                        coordinator.model.config.context.dropout_rate,
                    );
                let learned = target_index.is_some() && strength > 0.0;
                let supervised_strength = target_index.map_or(0.0, |target_output_index| {
                    let target_weight = if target_output_index == END_DOCUMENT_OUTPUT_INDEX {
                        coordinator.model.config.learning.end_document_weight
                    } else {
                        1.0
                    };
                    strength * target_weight
                });
                if learned {
                    lane_learned_steps[lane_index] =
                        lane_learned_steps[lane_index].saturating_add(1);
                    lane_context_learning[lane_index] |= context_enabled;
                }
                device_steps.push(CudaPersistentStep {
                    symbol,
                    target_index: target_index.map(|index| index as i32).unwrap_or(-1),
                    context_enabled: u32::from(context_enabled),
                    supervised_strength,
                });
                lane_metadata.push((symbol, target_index, context_enabled, learned));
            }

            pinned_write(lane.host_steps, &device_steps)?;
            chunk_h2d_bytes = chunk_h2d_bytes.saturating_add(
                (device_steps
                    .len()
                    .saturating_mul(mem::size_of::<CudaPersistentStep>())) as u64,
            );
            pointer_tables.push(lane.buffers.persistent_pointer_table.pointer);
            step_buffers.push(lane.buffers.persistent_steps.pointer);
            step_counts.push(as_u32("GPU shared story batch step count", chunk.len())?);
            base_ticks.push(base_tick);
            metadata.push(lane_metadata);
            offsets[lane_index] = end;
        }

        pinned_write(coordinator.host_batch_pointer_tables, &pointer_tables)?;
        pinned_write(coordinator.host_batch_step_buffers, &step_buffers)?;
        pinned_write(coordinator.host_batch_step_counts, &step_counts)?;
        pinned_write(coordinator.host_batch_base_ticks, &base_ticks)?;

        if profile_batch {
            coordinator.record_event(
                coordinator.events.h2d_start,
                coordinator.transfer_stream,
                "cuEventRecord(h2d start)",
            )?;
        }
        coordinator.copy_pinned_to_device_async_on::<CuDevicePtr>(
            coordinator.buffers.batch_pointer_tables,
            coordinator.host_batch_pointer_tables,
            pointer_tables.len(),
            coordinator.transfer_stream,
        )?;
        coordinator.copy_pinned_to_device_async_on::<CuDevicePtr>(
            coordinator.buffers.batch_step_buffers,
            coordinator.host_batch_step_buffers,
            step_buffers.len(),
            coordinator.transfer_stream,
        )?;
        coordinator.copy_pinned_to_device_async_on::<u32>(
            coordinator.buffers.batch_step_counts,
            coordinator.host_batch_step_counts,
            step_counts.len(),
            coordinator.transfer_stream,
        )?;
        coordinator.copy_pinned_to_device_async_on::<u64>(
            coordinator.buffers.batch_base_ticks,
            coordinator.host_batch_base_ticks,
            base_ticks.len(),
            coordinator.transfer_stream,
        )?;
        chunk_h2d_bytes = chunk_h2d_bytes
            .saturating_add((pointer_tables.len() * mem::size_of::<CuDevicePtr>()) as u64)
            .saturating_add((step_buffers.len() * mem::size_of::<CuDevicePtr>()) as u64)
            .saturating_add((step_counts.len() * mem::size_of::<u32>()) as u64)
            .saturating_add((base_ticks.len() * mem::size_of::<u64>()) as u64);

        let model_block_count = as_u32(
            "GPU shared model block count",
            coordinator.model.config.model.block_count,
        )?;
        let use_persistent_shared = coordinator.shared_persistent_enabled
            && learning_trace
            && !coordinator.full_step_metrics
            && phase_profile_sample_stride.is_none()
            && coordinator.shared.cooperative_launch
            && (coordinator.shared.shared_persistent_blocks_256 > 0
                || coordinator.shared.shared_grouped_blocks_256 > 0)
            && execution_plan.fused_wavefront_blocks > 0
            && execution_plan.fused_wavefront_threads == SHARED_BATCH_THREADS;
        let mut compute_timing_started = false;

        // Pipeline the first wavefront with the lane uploads. This creates real
        // copy/compute overlap while preserving the logical batch barrier: every
        // lane keeps its own learned parameter trajectory until the one
        // canonical story-batch mean is applied after all wavefronts complete.
        for (group_index, lane_start) in (0..lanes.len()).step_by(physical_lane_chunk).enumerate() {
            let physical_lane_count = physical_lane_chunk.min(lanes.len() - lane_start);
            for lane_index in lane_start..lane_start + physical_lane_count {
                let count = step_counts[lane_index] as usize;
                if count != 0 {
                    let lane = &lanes[lane_index];
                    coordinator.copy_pinned_to_device_async_on::<CudaPersistentStep>(
                        lane.buffers.persistent_steps,
                        lane.host_steps,
                        count,
                        coordinator.transfer_stream,
                    )?;
                }
            }
            let upload_ready = coordinator.events.upload_ready[group_index];
            coordinator.record_event(
                upload_ready,
                coordinator.transfer_stream,
                "cuEventRecord(upload ready)",
            )?;
            coordinator.stream_wait_event(
                coordinator.compute_stream,
                upload_ready,
                "cuStreamWaitEvent(upload ready)",
            )?;
            if profile_batch && !compute_timing_started {
                coordinator.record_event(
                    coordinator.events.compute_start,
                    coordinator.compute_stream,
                    "cuEventRecord(compute start)",
                )?;
                compute_timing_started = true;
            }
            if max_chunk_steps != 0 && use_persistent_shared {
                let group_max_steps = step_counts[lane_start..lane_start + physical_lane_count]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0) as usize;
                if group_max_steps != 0 {
                    let persistent_launch = launch_shared_wavefront_persistent_chunk(
                        coordinator,
                        PersistentWavefrontChunkLaunch {
                            lane_start,
                            physical_lane_count,
                            max_step_count: group_max_steps,
                            model_block_count,
                            learning_trace,
                            strength,
                            batch_scale,
                            execution_plan,
                        },
                    )?;
                    telemetry.fused_launches = telemetry.fused_launches.saturating_add(1);
                    if coordinator.debug.launches {
                        let kernel = if persistent_launch.grouped {
                            "leo_shared_wavefront_persistent_grouped"
                        } else {
                            "leo_shared_wavefront_persistent"
                        };
                        eprintln!(
                            "{{\"event\":\"cuda_debug_launch\",\"scope\":\"shared_story_batch\",\"kernel\":\"{}\",\"grid_blocks\":{},\"blocks_per_lane\":{},\"threads\":{},\"lane_start\":{},\"lane_count\":{},\"step_count\":{}}}",
                            kernel,
                            persistent_launch.grid_blocks,
                            persistent_launch.blocks_per_lane,
                            SHARED_BATCH_THREADS,
                            lane_start,
                            physical_lane_count,
                            group_max_steps,
                        );
                    }
                }
            } else if max_chunk_steps != 0 {
                let phase_profile = phase_profile_sample_stride
                    .is_some_and(|stride| phase_profile_launch_sequence % stride == 0);
                phase_profile_launch_sequence = phase_profile_launch_sequence.saturating_add(1);
                match launch_shared_wavefront_chunk(
                    coordinator,
                    WavefrontChunkLaunch {
                        lane_start,
                        physical_lane_count,
                        step_index: 0,
                        model_block_count,
                        learning_trace,
                        strength,
                        batch_scale,
                        execution_plan,
                        phase_profile,
                    },
                )? {
                    WavefrontLaunchKind::Fused {
                        phase_profile_sampled,
                        phase_profile_geometry_skipped,
                        grid_blocks,
                    } => {
                        telemetry.fused_launches += 1;
                        if phase_profile_sampled {
                            phase_profile_sampled_grid_max =
                                phase_profile_sampled_grid_max.max(grid_blocks);
                        }
                        if phase_profile_geometry_skipped {
                            phase_profile_geometry_skips =
                                phase_profile_geometry_skips.saturating_add(1);
                            phase_profile_skipped_grid_max =
                                phase_profile_skipped_grid_max.max(grid_blocks);
                        }
                    }
                    WavefrontLaunchKind::Direct { phase_launches } => {
                        telemetry.direct_phase_launches = telemetry
                            .direct_phase_launches
                            .saturating_add(phase_launches);
                    }
                }
            }
        }

        if profile_batch {
            coordinator.record_event(
                coordinator.events.h2d_end,
                coordinator.transfer_stream,
                "cuEventRecord(h2d end)",
            )?;
        }

        if !use_persistent_shared {
            for step_index in 1..max_chunk_steps {
                for lane_start in (0..lanes.len()).step_by(physical_lane_chunk) {
                    let physical_lane_count = physical_lane_chunk.min(lanes.len() - lane_start);
                    let phase_profile = phase_profile_sample_stride
                        .is_some_and(|stride| phase_profile_launch_sequence % stride == 0);
                    phase_profile_launch_sequence = phase_profile_launch_sequence.saturating_add(1);
                    match launch_shared_wavefront_chunk(
                        coordinator,
                        WavefrontChunkLaunch {
                            lane_start,
                            physical_lane_count,
                            step_index,
                            model_block_count,
                            learning_trace,
                            strength,
                            batch_scale,
                            execution_plan,
                            phase_profile,
                        },
                    )? {
                        WavefrontLaunchKind::Fused {
                            phase_profile_sampled,
                            phase_profile_geometry_skipped,
                            grid_blocks,
                        } => {
                            telemetry.fused_launches += 1;
                            if phase_profile_sampled {
                                phase_profile_sampled_grid_max =
                                    phase_profile_sampled_grid_max.max(grid_blocks);
                            }
                            if phase_profile_geometry_skipped {
                                phase_profile_geometry_skips =
                                    phase_profile_geometry_skips.saturating_add(1);
                                phase_profile_skipped_grid_max =
                                    phase_profile_skipped_grid_max.max(grid_blocks);
                            }
                        }
                        WavefrontLaunchKind::Direct { phase_launches } => {
                            telemetry.direct_phase_launches = telemetry
                                .direct_phase_launches
                                .saturating_add(phase_launches);
                        }
                    }
                }
            }
        }

        if profile_batch {
            coordinator.record_event(
                coordinator.events.compute_end,
                coordinator.compute_stream,
                "cuEventRecord(compute end)",
            )?;
        }
        coordinator.record_event(
            coordinator.events.compute_done,
            coordinator.compute_stream,
            "cuEventRecord(compute done)",
        )?;
        coordinator.stream_wait_event(
            coordinator.transfer_stream,
            coordinator.events.compute_done,
            "cuStreamWaitEvent(compute done)",
        )?;
        if profile_batch {
            coordinator.record_event(
                coordinator.events.d2h_start,
                coordinator.transfer_stream,
                "cuEventRecord(d2h start)",
            )?;
        }

        for lane_index in 0..lanes.len() {
            let count = step_counts[lane_index] as usize;
            if count == 0 {
                continue;
            }
            let lane = &lanes[lane_index];
            if use_persistent_shared {
                coordinator.copy_device_to_pinned_async_on::<CudaFastTrainingStepRecord>(
                    lane.host_records,
                    lane.buffers.training_step_records,
                    count,
                    coordinator.transfer_stream,
                )?;
                chunk_d2h_bytes = chunk_d2h_bytes.saturating_add(
                    (count.saturating_mul(mem::size_of::<CudaFastTrainingStepRecord>())) as u64,
                );
            } else {
                coordinator.copy_device_to_pinned_async_on::<CudaTrainingStepRecord>(
                    lane.host_records,
                    lane.buffers.training_step_records,
                    count,
                    coordinator.transfer_stream,
                )?;
                chunk_d2h_bytes = chunk_d2h_bytes.saturating_add(
                    (count.saturating_mul(mem::size_of::<CudaTrainingStepRecord>())) as u64,
                );
            }
        }
        coordinator.record_event(
            coordinator.events.d2h_end,
            coordinator.transfer_stream,
            "cuEventRecord(d2h end)",
        )?;
        let host_wait_started = Instant::now();
        coordinator.event_synchronize(coordinator.events.d2h_end, "cuEventSynchronize(d2h end)")?;
        let host_wait_ms = host_wait_started.elapsed().as_secs_f32() * 1000.0;

        if profile_batch {
            telemetry.h2d_ms += coordinator.event_elapsed_ms(
                coordinator.events.h2d_start,
                coordinator.events.h2d_end,
                "cuEventElapsedTime(h2d)",
            )?;
            if compute_timing_started {
                telemetry.compute_ms += coordinator.event_elapsed_ms(
                    coordinator.events.compute_start,
                    coordinator.events.compute_end,
                    "cuEventElapsedTime(compute)",
                )?;
            }
            telemetry.d2h_ms += coordinator.event_elapsed_ms(
                coordinator.events.d2h_start,
                coordinator.events.d2h_end,
                "cuEventElapsedTime(d2h)",
            )?;
            telemetry.host_wait_ms += host_wait_ms;
            telemetry.h2d_bytes = telemetry.h2d_bytes.saturating_add(chunk_h2d_bytes);
            telemetry.d2h_bytes = telemetry.d2h_bytes.saturating_add(chunk_d2h_bytes);
        }

        for lane_index in 0..lanes.len() {
            let lane = &mut lanes[lane_index];
            let count = step_counts[lane_index] as usize;
            if count == 0 {
                continue;
            }

            if learning_trace && count % 2 == 1 {
                std::mem::swap(
                    &mut lane.buffers.recurrent_eligible_list,
                    &mut lane.buffers.recurrent_next_eligible_list,
                );
                std::mem::swap(
                    &mut lane.buffers.recurrent_eligible_count,
                    &mut lane.buffers.recurrent_next_eligible_count,
                );
                std::mem::swap(
                    &mut lane.buffers.input_eligible_list,
                    &mut lane.buffers.input_next_eligible_list,
                );
                std::mem::swap(
                    &mut lane.buffers.input_eligible_count,
                    &mut lane.buffers.input_next_eligible_count,
                );
                // Shared-batch kernels dereference the lane-resident pointer
                // table on every wavefront. An odd chunk flips current/next
                // eligibility buffers, so refresh that table before the next
                // chunk just like the single-story persistent path does when it
                // rebuilds its pointer table per chunk.
                coordinator.refresh_shared_batch_pointer_table_async(lane)?;
            }
            lane.current_tick = lane.current_tick.saturating_add(count as u64);

            if use_persistent_shared {
                let records =
                    pinned_read_vec::<CudaFastTrainingStepRecord>(lane.host_records, count)?;
                for (record, (symbol, target_index, context_enabled, _learned)) in
                    records.into_iter().zip(metadata[lane_index].drain(..))
                {
                    if record.error_code != 0 {
                        coordinator.model.statistics.numerical_rejections = coordinator
                            .model
                            .statistics
                            .numerical_rejections
                            .saturating_add(1);
                        return Err(LeoError::cuda(format!(
                            "CUDA persistent shared story-batch learning update rejected a non-finite value (lane {lane_index}, code {})",
                            record.error_code
                        )));
                    }
                    let metrics = coordinator.fast_training_record_to_metrics(
                        record,
                        target_index,
                        context_enabled,
                    );
                    let statistics = &mut lane_statistics[lane_index];
                    if symbol < 256 {
                        statistics.processed_bytes = statistics.processed_bytes.saturating_add(1);
                    }
                    if let Some(value) = metrics.loss {
                        statistics.training_loss_sum += value as f64;
                        statistics.training_targets = statistics.training_targets.saturating_add(1);
                    }
                    statistics.active_neurons_sum = statistics
                        .active_neurons_sum
                        .saturating_add(metrics.active_neurons as u64);
                    statistics.active_neurons_peak = statistics
                        .active_neurons_peak
                        .max(metrics.active_neurons as u64);
                    statistics.synaptic_events = statistics
                        .synaptic_events
                        .saturating_add(metrics.emitted_events as u64);
                    statistics.persistent_ticks = statistics.persistent_ticks.saturating_add(1);
                    story_metrics[lane_index].push(metrics);
                }
            } else {
                let records = pinned_read_vec::<CudaTrainingStepRecord>(lane.host_records, count)?;
                for (record, (symbol, target_index, context_enabled, learned)) in
                    records.into_iter().zip(metadata[lane_index].drain(..))
                {
                    if record.error_code != 0 {
                        coordinator.model.statistics.numerical_rejections = coordinator
                            .model
                            .statistics
                            .numerical_rejections
                            .saturating_add(1);
                        return Err(LeoError::cuda(format!(
                            "CUDA shared story-batch learning update rejected a non-finite value (lane {lane_index}, code {})",
                            record.error_code
                        )));
                    }
                    let metrics = coordinator.training_record_to_metrics(
                        record,
                        target_index,
                        context_enabled,
                        learned,
                    );
                    if learning_trace {
                        let statistics = &mut lane_statistics[lane_index];
                        if symbol < 256 {
                            statistics.processed_bytes =
                                statistics.processed_bytes.saturating_add(1);
                        }
                        if let Some(value) = metrics.loss {
                            statistics.training_loss_sum += value as f64;
                            statistics.training_targets =
                                statistics.training_targets.saturating_add(1);
                        }
                        statistics.active_neurons_sum = statistics
                            .active_neurons_sum
                            .saturating_add(metrics.active_neurons as u64);
                        statistics.active_neurons_peak = statistics
                            .active_neurons_peak
                            .max(metrics.active_neurons as u64);
                        statistics.synaptic_events = statistics
                            .synaptic_events
                            .saturating_add(metrics.emitted_events as u64);
                        statistics.persistent_ticks = statistics.persistent_ticks.saturating_add(1);
                    }
                    story_metrics[lane_index].push(metrics);
                }
            }
        }
    }

    if learning_trace {
        for (statistics, steps) in lane_statistics.iter_mut().zip(&story_steps) {
            if !steps.is_empty() {
                statistics.processed_stories = statistics.processed_stories.saturating_add(1);
            }
        }
    }

    let output_bias_dirty = lane_learned_steps
        .iter()
        .map(|&steps| steps > 0)
        .collect::<Vec<_>>();
    let any_output_bias_dirty = output_bias_dirty.iter().copied().any(|dirty| dirty);
    let any_context_output_dirty = lane_context_learning.iter().copied().any(|dirty| dirty);

    // Multi-device data parallelism needs original per-story deltas so the
    // outer reducer can perform one flat logical-worker merge. Single-device
    // batches can avoid materializing the large fixed learned tensors on the
    // CPU: reduce them directly on the canonical GPU and transfer only context
    // hash rows, whose key-based collision semantics still require the host.
    let (merge, retained_deltas, retained_changes) = if retain_story_deltas {
        let snapshots = coordinator.snapshot_batch_lane_values_batched(
            lanes,
            &output_bias_dirty,
            &lane_context_learning,
            &lane_learned_steps,
            &lane_statistics,
        )?;
        let mut deltas = Vec::with_capacity(lanes.len());
        let mut raw_changes = Vec::with_capacity(lanes.len());
        for (changes, values) in snapshots {
            let delta =
                SparseModelDelta::from_tracked_values(&coordinator.model, &changes, values)?;
            raw_changes.push(changes);
            deltas.push(delta);
        }
        (
            MergeMetrics {
                workers: lanes.len(),
                fixed_parameter_updates: 0,
                context_keys: 0,
            },
            deltas,
            raw_changes,
        )
    } else if cuda_device_batch_merge_enabled() {
        // The dense device reducer parallelizes across parameters and loops
        // logical lanes in deterministic order inside each thread. This
        // removes the dominant fixed-weight D2H snapshot/CPU mean/H2D cycle.
        coordinator.launch_device_batch_fixed_merge(lanes.len())?;

        // Context slots are keyed hash-table entries: the same semantic key can
        // occupy different physical slots in different lanes. Preserve the
        // existing keyed merge while transferring only those sparse rows.
        let context_snapshots = coordinator.snapshot_batch_lane_context_values_batched(
            lanes,
            &lane_learned_steps,
            &lane_statistics,
        )?;

        let mut context_deltas = Vec::with_capacity(lanes.len());
        let mut context_raw_changes = Vec::with_capacity(lanes.len());
        for (changes, values) in context_snapshots {
            let delta =
                SparseModelDelta::from_tracked_values(&coordinator.model, &changes, values)?;
            context_raw_changes.push(changes);
            context_deltas.push(delta);
        }

        // Pull the already-merged fixed canonical rows back once so the host
        // model remains the source of truth for checkpointing/validation and
        // for constructing the sparse packet scattered to resident lanes.
        let fixed_changes = coordinator
            .current_device_parameter_changes(any_output_bias_dirty, any_context_output_dirty)?;
        coordinator.output_bias_dirty_since_sync |= any_output_bias_dirty;
        coordinator.context_output_dirty_since_sync |= any_context_output_dirty;
        coordinator.sync_changed_model_to_host()?;

        // Context-only deltas also carry revision/statistics accounting, so the
        // established reducer keeps those semantics unchanged while touching
        // no fixed learned arrays.
        let context_merge = apply_mean_deltas(&mut coordinator.model, &context_deltas)?;
        let context_changes = merged_parameter_changes(
            &mut coordinator.model,
            &context_deltas,
            &context_raw_changes,
        )?;
        let sync_changes = combine_parameter_changes(fixed_changes.clone(), context_changes);
        commit_shared_story_batch_update(coordinator, lanes, &sync_changes)?;

        let fixed_parameter_updates = fixed_changes
            .threshold
            .len()
            .saturating_add(fixed_changes.recurrent_weight.len())
            .saturating_add(fixed_changes.input_weight.len())
            .saturating_add(
                fixed_changes
                    .output_neurons
                    .len()
                    .saturating_mul(OUTPUT_CLASSES),
            )
            .saturating_add(if fixed_changes.output_bias_dirty {
                OUTPUT_CLASSES
            } else {
                0
            })
            .saturating_add(if fixed_changes.context_output_dirty {
                coordinator.model.context.output_weights.len()
            } else {
                0
            });
        (
            MergeMetrics {
                workers: lanes.len(),
                fixed_parameter_updates: fixed_parameter_updates
                    .saturating_add(context_merge.fixed_parameter_updates),
                context_keys: context_merge.context_keys,
            },
            Vec::new(),
            Vec::new(),
        )
    } else {
        // Compatibility path: materialize every touched fixed/context row and
        // execute the original host sparse mean. Keep this behind
        // LEO_CUDA_DEVICE_BATCH_MERGE=0 for A/B validation and rollback.
        let snapshots = coordinator.snapshot_batch_lane_values_batched(
            lanes,
            &output_bias_dirty,
            &lane_context_learning,
            &lane_learned_steps,
            &lane_statistics,
        )?;
        let mut deltas = Vec::with_capacity(lanes.len());
        let mut raw_changes = Vec::with_capacity(lanes.len());
        for (changes, values) in snapshots {
            let delta =
                SparseModelDelta::from_tracked_values(&coordinator.model, &changes, values)?;
            raw_changes.push(changes);
            deltas.push(delta);
        }
        let merge = apply_mean_deltas(&mut coordinator.model, &deltas)?;
        let sync_changes = merged_parameter_changes(&mut coordinator.model, &deltas, &raw_changes)?;
        commit_shared_story_batch_update(coordinator, lanes, &sync_changes)?;
        (merge, Vec::new(), Vec::new())
    };

    coordinator.execution_tuner.observe_batch(
        lanes.len(),
        tuning_work_units,
        tuning_started.elapsed(),
        profile_batch.then_some(telemetry),
    );
    if let Some(sample_stride) = phase_profile_sample_stride {
        coordinator.emit_phase_profile(CudaPhaseProfileReport {
            logical_lanes: lanes.len(),
            fused_launches_total: telemetry.fused_launches,
            sample_stride,
            planned_fused_blocks: execution_plan.fused_wavefront_blocks,
            geometry_skipped_samples: phase_profile_geometry_skips,
            sampled_grid_blocks_max: phase_profile_sampled_grid_max,
            skipped_grid_blocks_max: phase_profile_skipped_grid_max,
        })?;
    }
    coordinator.reset_transient_state()?;

    Ok(CudaStoryBatchReport {
        story_metrics,
        merge,
        story_deltas: retained_deltas,
        story_changes: retained_changes,
    })
}

fn cuda_execution_profile_key(model: &Model, shared: &SharedCuda) -> String {
    let source_digest = crate::digest_bytes(cuda_kernel_source().as_bytes());
    let material = format!(
        concat!(
            "leo-cuda-plan-v4|device={}|name={}|uuid={}|pci={}|vram={}|driver={}|",
            "cc={}.{}|sm={}|threads_sm={}|abi={}|semantics={}|source={}|",
            "n={}|blocks={}|npb={}|rec={}|input={}|ctx={}|ctxdim={}"
        ),
        shared.hardware.device_ordinal,
        shared.hardware.name,
        shared.hardware.uuid,
        shared.hardware.pci_bus_id,
        shared.hardware.total_memory_bytes,
        shared.hardware.driver_version,
        shared.compute_major,
        shared.compute_minor,
        shared.multiprocessor_count,
        shared.max_threads_per_sm,
        crate::semantics::CUDA_ABI_VERSION,
        crate::semantics::EXECUTION_SEMANTICS_VERSION,
        source_digest,
        model.neuron_count(),
        model.config.model.block_count,
        model.config.model.neurons_per_block,
        model.recurrent.weight.len(),
        model.input.weights.len(),
        model.context.keys.len(),
        model.config.context.embedding_dim,
    );
    crate::digest_bytes(material.as_bytes()).to_hex()
}

fn validate_gpu_shape(model: &Model) -> LeoResult<()> {
    if model.config.model.max_active_per_block > MAX_BLOCK_WINNERS {
        return Err(LeoError::cuda(format!(
            "GPU executor supports max_active_per_block <= {MAX_BLOCK_WINNERS}; got {}",
            model.config.model.max_active_per_block
        )));
    }
    let block_winners = model
        .config
        .model
        .block_count
        .checked_mul(model.config.model.max_active_per_block)
        .ok_or_else(|| LeoError::cuda("GPU block-winner count overflow"))?;
    if block_winners > MAX_GLOBAL_BLOCK_WINNERS {
        return Err(LeoError::cuda(format!(
            "GPU executor requires block_count * max_active_per_block <= {MAX_GLOBAL_BLOCK_WINNERS}; got {block_winners}"
        )));
    }
    if model.config.model.neurons_per_block > 1024 {
        return Err(LeoError::cuda(
            "GPU executor requires neurons_per_block <= 1024",
        ));
    }
    if model.config.model.max_active_global >= MAX_GLOBAL_BLOCK_WINNERS {
        // 1024 itself is valid only when there cannot be a 1025th winner; the
        // production layout has exactly 1024 block-local winners maximum.
        if block_winners > MAX_GLOBAL_BLOCK_WINNERS {
            return Err(LeoError::cuda(
                "GPU global active cap exceeds selection buffer",
            ));
        }
    }
    if model.config.context.max_order > 16 || model.config.context.embedding_dim > 256 {
        return Err(LeoError::cuda(
            "GPU executor context limits are max_order <= 16 and embedding_dim <= 256",
        ));
    }
    if model
        .recurrent
        .delay
        .iter()
        .any(|delay| !matches!(*delay, 1 | 2 | 4 | 8))
    {
        return Err(LeoError::cuda(
            "GPU executor requires recurrent delays to be one of 1, 2, 4, or 8",
        ));
    }
    Ok(())
}

fn transpose_output_weights(model: &Model) -> Vec<f32> {
    let neuron_count = model.neuron_count();
    let mut transposed = vec![0.0; neuron_count * OUTPUT_CLASSES];
    for output in 0..OUTPUT_CLASSES {
        let source = output * neuron_count;
        for neuron in 0..neuron_count {
            transposed[neuron * OUTPUT_CLASSES + output] = model.output.weights[source + neuron];
        }
    }
    transposed
}

fn accumulate_indices(indices: &[usize], marks: &mut [bool], changed: &mut Vec<usize>) {
    for &index in indices {
        if !marks[index] {
            marks[index] = true;
            changed.push(index);
        }
    }
}

#[inline]
fn context_survives_dropout(seed: u64, tick: u64, symbol: u32, dropout_rate: f32) -> bool {
    if dropout_rate <= 0.0 {
        return true;
    }
    let mut value = seed
        ^ tick.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(symbol).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let draw = (value >> 40) as f32 / (1u32 << 24) as f32;
    draw >= dropout_rate
}

fn loss_from_logits(logits: &[f32], target: usize) -> f32 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        (OUTPUT_CLASSES as f32).ln()
    } else {
        maximum + sum.ln() - logits[target]
    }
}

fn pinned_write<T: Copy>(buffer: PinnedHostBuffer, values: &[T]) -> LeoResult<()> {
    let bytes = values.len().saturating_mul(mem::size_of::<T>());
    if bytes > buffer.bytes {
        return Err(LeoError::cuda(format!(
            "CUDA pinned host write of {bytes} bytes exceeds capacity {}",
            buffer.bytes
        )));
    }
    if bytes != 0 {
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.pointer.cast::<u8>(),
                bytes,
            );
        }
    }
    Ok(())
}

fn pinned_read_vec<T: Copy + Default>(
    buffer: PinnedHostBuffer,
    elements: usize,
) -> LeoResult<Vec<T>> {
    let bytes = elements.saturating_mul(mem::size_of::<T>());
    if bytes > buffer.bytes {
        return Err(LeoError::cuda(format!(
            "CUDA pinned host read of {bytes} bytes exceeds capacity {}",
            buffer.bytes
        )));
    }
    let mut values = vec![T::default(); elements];
    if bytes != 0 {
        unsafe {
            ptr::copy_nonoverlapping(
                buffer.pointer.cast::<u8>(),
                values.as_mut_ptr().cast::<u8>(),
                bytes,
            );
        }
    }
    Ok(values)
}

fn device_sub_buffer<T>(
    buffer: DeviceBuffer,
    element_offset: usize,
    elements: usize,
) -> LeoResult<DeviceBuffer> {
    let byte_offset = element_offset.saturating_mul(mem::size_of::<T>());
    let bytes = elements.saturating_mul(mem::size_of::<T>());
    if byte_offset > buffer.bytes || bytes > buffer.bytes.saturating_sub(byte_offset) {
        return Err(LeoError::cuda(
            "CUDA snapshot device sub-buffer exceeds allocated capacity",
        ));
    }
    Ok(DeviceBuffer {
        pointer: device_pointer_offset(buffer.pointer, byte_offset)?,
        bytes,
    })
}

fn as_u32(name: &str, value: usize) -> LeoResult<u32> {
    u32::try_from(value).map_err(|_| LeoError::cuda(format!("{name} exceeds CUDA u32 capacity")))
}

fn param<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast::<c_void>()
}

fn load_kernels(driver: &DriverFunctions, module: CuModule) -> LeoResult<KernelFunctions> {
    Ok(KernelFunctions {
        start_tick: get_kernel(driver, module, "leo_start_tick")?,
        deliver_events: get_kernel(driver, module, "leo_deliver_events")?,
        inject_symbol: get_kernel(driver, module, "leo_inject_symbol")?,
        context_resolve: get_kernel(driver, module, "leo_context_resolve")?,
        select_blocks: get_kernel(driver, module, "leo_select_blocks")?,
        select_global: get_kernel(driver, module, "leo_select_global")?,
        cache_surrogate: get_kernel(driver, module, "leo_cache_surrogate")?,
        update_recurrent_eligibility: get_kernel(
            driver,
            module,
            "leo_update_recurrent_eligibility",
        )?,
        update_input_eligibility: get_kernel(driver, module, "leo_update_input_eligibility")?,
        post_and_emit: get_kernel(driver, module, "leo_post_and_emit")?,
        forward: get_kernel(driver, module, "leo_forward")?,
        capture_training_step: get_kernel(driver, module, "leo_capture_training_step")?,
        train_persistent: get_kernel(driver, module, "leo_train_persistent")?,
        train_cooperative: get_kernel(driver, module, "leo_train_cooperative")?,
        train_cooperative_fast: get_kernel(driver, module, "leo_train_cooperative_fast")?,
        train_cooperative_profiled: get_kernel(driver, module, "leo_train_cooperative_profiled")?,
        advance_frozen_persistent: get_kernel(driver, module, "leo_advance_frozen_persistent")?,
        advance_frozen_cooperative: get_kernel(driver, module, "leo_advance_frozen_cooperative")?,
        advance_frozen_cooperative_profiled: get_kernel(
            driver,
            module,
            "leo_advance_frozen_cooperative_profiled",
        )?,
        shared_wavefront_pre: get_kernel(driver, module, "leo_shared_wavefront_pre")?,
        shared_wavefront_fused: get_kernel(driver, module, "leo_shared_wavefront_fused")?,
        shared_wavefront_persistent: get_kernel(driver, module, "leo_shared_wavefront_persistent")?,
        shared_wavefront_persistent_grouped: get_kernel(
            driver,
            module,
            "leo_shared_wavefront_persistent_grouped",
        )?,
        shared_wavefront_fused_profiled: get_kernel(
            driver,
            module,
            "leo_shared_wavefront_fused_profiled",
        )?,
        shared_select_blocks: get_kernel(driver, module, "leo_shared_select_blocks")?,
        shared_post_select: get_kernel(driver, module, "leo_shared_post_select")?,
        shared_cache_surrogate_worklist: get_kernel(
            driver,
            module,
            "leo_shared_cache_surrogate_worklist",
        )?,
        shared_post_core: get_kernel(driver, module, "leo_shared_post_core")?,
        shared_learning_signals_worklist: get_kernel(
            driver,
            module,
            "leo_shared_learning_signals_worklist",
        )?,
        shared_post_deltas: get_kernel(driver, module, "leo_shared_post_deltas")?,
        shared_homeostasis_worklist: get_kernel(driver, module, "leo_shared_homeostasis_worklist")?,
        shared_capture_training_step: get_kernel(
            driver,
            module,
            "leo_shared_capture_training_step",
        )?,
        merge_shared_lane_fixed_parameters: get_kernel(
            driver,
            module,
            "leo_merge_shared_lane_fixed_parameters",
        )?,
        apply_shared_wavefront_deltas: get_kernel(
            driver,
            module,
            "leo_apply_shared_wavefront_deltas",
        )?,
        reset_shared_wavefront_deltas: get_kernel(
            driver,
            module,
            "leo_reset_shared_wavefront_deltas",
        )?,
        learning_signals: get_kernel(driver, module, "leo_learning_signals")?,
        update_output: get_kernel(driver, module, "leo_update_output")?,
        update_context: get_kernel(driver, module, "leo_update_context")?,
        update_recurrent_weights: get_kernel(driver, module, "leo_update_recurrent_weights")?,
        update_input_weights: get_kernel(driver, module, "leo_update_input_weights")?,
        inhibitory_homeostasis: get_kernel(driver, module, "leo_inhibitory_homeostasis")?,
        homeostasis: get_kernel(driver, module, "leo_homeostasis")?,
        reset_last_ticks: get_kernel(driver, module, "leo_reset_last_ticks")?,
        gather_f32: get_kernel(driver, module, "leo_gather_f32")?,
        gather_output: get_kernel(driver, module, "leo_gather_output")?,
        gather_context: get_kernel(driver, module, "leo_gather_context")?,
        scatter_f32: get_kernel(driver, module, "leo_scatter_f32")?,
        scatter_output: get_kernel(driver, module, "leo_scatter_output")?,
        scatter_context: get_kernel(driver, module, "leo_scatter_context")?,
        clear_changed_marks: get_kernel(driver, module, "leo_clear_changed_marks")?,
    })
}

fn get_kernel(driver: &DriverFunctions, module: CuModule, name: &str) -> LeoResult<CuFunction> {
    let name = CString::new(name).map_err(|_| LeoError::cuda("invalid CUDA kernel name"))?;
    let mut function = ptr::null_mut();
    driver.check(
        unsafe { (driver.module_get_function)(&mut function, module, name.as_ptr()) },
        "cuModuleGetFunction",
    )?;
    Ok(function)
}

fn compile_cuda_kernels(compute_major: c_int, compute_minor: c_int) -> LeoResult<Vec<u8>> {
    let nvrtc = NvrtcFunctions::load()?;
    let mut nvrtc_major = 0;
    let mut nvrtc_minor = 0;
    let version_result = unsafe { (nvrtc.version)(&mut nvrtc_major, &mut nvrtc_minor) };
    if version_result != NVRTC_SUCCESS {
        return Err(LeoError::cuda(format!(
            "nvrtcVersion failed: {}",
            nvrtc.error_string(version_result)
        )));
    }

    let source_text = cuda_kernel_source();
    let include_directory = cuda_include_directory().ok_or_else(|| {
        LeoError::cuda(
            "CUDA cooperative_groups.h was not found; set CUDA_HOME or CUDA_PATH to the CUDA toolkit root",
        )
    })?;
    let cooperative_groups_header = include_directory.join("cooperative_groups.h");
    let header_digest = crate::digest_file(&cooperative_groups_header)
        .map_err(|error| {
            LeoError::cuda(format!(
                "could not hash CUDA header {} for PTX cache identity: {error}",
                cooperative_groups_header.display()
            ))
        })?
        .to_hex();
    let source_digest = crate::digest_bytes(source_text.as_bytes());
    let cache_material = format!(
        "leo-ptx-v1|cc={compute_major}.{compute_minor}|nvrtc={nvrtc_major}.{nvrtc_minor}|abi={}|semantics={}|std=c++11|header={header_digest}|source={source_digest}",
        crate::semantics::CUDA_ABI_VERSION,
        crate::semantics::EXECUTION_SEMANTICS_VERSION,
    );
    let cache_key = crate::digest_bytes(cache_material.as_bytes()).to_hex();
    let cache_path = cache_root().map(|root| root.join("ptx").join(format!("{cache_key}.ptx")));

    if let Some(path) = cache_path.as_ref() {
        let digest_path = path.with_extension("ptx.sha256");
        if let (Ok(ptx), Ok(expected)) = (fs::read(path), fs::read_to_string(&digest_path)) {
            let actual = crate::digest_bytes(&ptx).to_hex();
            if !ptx.is_empty() && actual == expected.trim() {
                return Ok(ptx);
            }
        }
    }

    let ptx = compile_cuda_kernels_uncached(
        &nvrtc,
        &source_text,
        &include_directory,
        compute_major,
        compute_minor,
    )?;
    if let Some(path) = cache_path.as_ref() {
        if atomic_write(path, &ptx).is_ok() {
            let digest_path = path.with_extension("ptx.sha256");
            let digest = crate::digest_bytes(&ptx).to_hex();
            let _ = atomic_write(&digest_path, digest.as_bytes());
        }
    }
    Ok(ptx)
}

fn compile_cuda_kernels_uncached(
    nvrtc: &NvrtcFunctions,
    source_text: &str,
    include_directory: &std::path::Path,
    compute_major: c_int,
    compute_minor: c_int,
) -> LeoResult<Vec<u8>> {
    let source = CString::new(source_text)
        .map_err(|_| LeoError::cuda("CUDA kernel source contains an interior NUL"))?;
    let name = CString::new("leo_runtime.cu").expect("static CUDA source name is valid");
    let mut program = ptr::null_mut();
    let create = unsafe {
        (nvrtc.create_program)(
            &mut program,
            source.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    if create != NVRTC_SUCCESS {
        return Err(LeoError::cuda(format!(
            "nvrtcCreateProgram failed: {}",
            nvrtc.error_string(create)
        )));
    }
    let standard = CString::new("--std=c++11").expect("static NVRTC option is valid");
    let architecture = CString::new(format!(
        "--gpu-architecture=compute_{compute_major}{compute_minor}"
    ))
    .map_err(|_| LeoError::cuda("invalid CUDA compute capability"))?;
    let include_option = CString::new(format!("--include-path={}", include_directory.display()))
        .map_err(|_| LeoError::cuda("invalid CUDA include directory"))?;
    let options = [
        standard.as_ptr(),
        architecture.as_ptr(),
        include_option.as_ptr(),
    ];
    let compile =
        unsafe { (nvrtc.compile_program)(program, options.len() as c_int, options.as_ptr()) };
    if compile != NVRTC_SUCCESS {
        let log = nvrtc_program_log(nvrtc, program);
        unsafe {
            (nvrtc.destroy_program)(&mut program);
        }
        return Err(LeoError::cuda(format!(
            "NVRTC failed to compile Leo CUDA kernels: {}\n{}",
            nvrtc.error_string(compile),
            log
        )));
    }
    let mut ptx_size = 0usize;
    let size_result = unsafe { (nvrtc.get_ptx_size)(program, &mut ptx_size) };
    if size_result != NVRTC_SUCCESS || ptx_size == 0 {
        unsafe {
            (nvrtc.destroy_program)(&mut program);
        }
        return Err(LeoError::cuda(format!(
            "nvrtcGetPTXSize failed: {}",
            nvrtc.error_string(size_result)
        )));
    }
    let mut ptx = vec![0u8; ptx_size];
    let ptx_result = unsafe { (nvrtc.get_ptx)(program, ptx.as_mut_ptr().cast::<c_char>()) };
    unsafe {
        (nvrtc.destroy_program)(&mut program);
    }
    if ptx_result != NVRTC_SUCCESS {
        return Err(LeoError::cuda(format!(
            "nvrtcGetPTX failed: {}",
            nvrtc.error_string(ptx_result)
        )));
    }
    Ok(ptx)
}

fn cuda_include_directory() -> Option<PathBuf> {
    let mut roots = Vec::<PathBuf>::new();
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(value) = env::var(variable) {
            if !value.is_empty() {
                roots.push(PathBuf::from(value));
            }
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/opt/cuda"));
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("cuda-") {
                roots.push(entry.path());
            }
        }
    }
    roots
        .into_iter()
        .map(|root| root.join("include"))
        .find(|include| include.join("cooperative_groups.h").is_file())
}

fn nvrtc_program_log(nvrtc: &NvrtcFunctions, program: NvrtcProgram) -> String {
    let mut size = 0usize;
    let result = unsafe { (nvrtc.get_program_log_size)(program, &mut size) };
    if result != NVRTC_SUCCESS || size == 0 {
        return String::new();
    }
    let mut buffer = vec![0u8; size];
    let result = unsafe { (nvrtc.get_program_log)(program, buffer.as_mut_ptr().cast::<c_char>()) };
    if result != NVRTC_SUCCESS {
        return String::new();
    }
    if buffer.last().copied() == Some(0) {
        buffer.pop();
    }
    String::from_utf8_lossy(&buffer).into_owned()
}

#[cfg(test)]
mod tests {
    use super::cuda_kernel_source;

    #[test]
    fn cuda_source_contains_device_resident_runtime_kernels() {
        for name in [
            "leo_deliver_events",
            "leo_select_blocks",
            "leo_update_recurrent_eligibility",
            "leo_forward",
            "leo_update_recurrent_weights",
            "leo_homeostasis",
            "leo_train_persistent",
            "leo_train_cooperative_fast",
            "leo_train_story_batch",
            "leo_shared_wavefront_pre",
            "leo_shared_wavefront_persistent",
            "leo_shared_wavefront_fused_profiled",
            "leo_shared_select_blocks",
            "leo_shared_post_select",
            "leo_shared_cache_surrogate_worklist",
            "leo_shared_post_core",
            "leo_shared_learning_signals_worklist",
            "leo_shared_post_deltas",
            "leo_shared_homeostasis_worklist",
            "leo_shared_capture_training_step",
            "leo_apply_shared_wavefront_deltas",
            "leo_reset_shared_wavefront_deltas",
            "leo_scatter_f32",
            "leo_scatter_output",
            "leo_scatter_context",
        ] {
            assert!(cuda_kernel_source().contains(name));
        }
        assert!(cuda_kernel_source().contains("ring_weight"));
        assert!(cuda_kernel_source().contains("leo_context_resolve"));
    }
}
