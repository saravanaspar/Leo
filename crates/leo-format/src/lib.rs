//! Leo v1 tensor-native `.pscls` checkpoint reader, writer, generations, and rollback.
//!
//! The v1 wire format is intentionally strict: one schema, strong SHA-256
//! integrity metadata, no duplicate/overlapping sections, and an explicit Leo
//! semantics contract.  Pre-v1 checkpoint formats are not accepted by the main
//! runtime; v1.0.0 is a clean training baseline.

use leo_core::artifact::{digest_bytes, digest_file, ArtifactDigest, DIGEST_BYTES};
use leo_core::metrics::TrainingStatistics;
use leo_core::model::{
    ContextProjection, InputProjection, Model, NeuronParameters, OutputProjection, SynapseArrays,
};
use leo_core::semantics::{CHECKPOINT_SCHEMA_VERSION, SemanticsContract};
use leo_core::{Config, LeoError, LeoResult};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"PSCLS100";
const FORMAT_VERSION: u32 = CHECKPOINT_SCHEMA_VERSION;
const ENDIAN_MARKER: u32 = 0x0102_0304;
const HEADER_SIZE: usize = 128;
const DESCRIPTOR_SIZE: usize = 128;
const ALIGNMENT: usize = 64;
const HEADER_DIGEST_OFFSET: usize = 96;
const MAX_CHECKPOINT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_CONFIG_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
enum SectionKind {
    Config = 1,
    NeuronParameters = 2,
    RecurrentTargets = 3,
    RecurrentMetadata = 4,
    RecurrentWeights = 5,
    InputProjection = 6,
    OutputWeights = 7,
    OutputBias = 8,
    ContextProjection = 9,
    TrainingStatistics = 10,
    GenerationMetadata = 11,
    SemanticsContract = 12,
}

impl SectionKind {
    const ALL: [Self; 12] = [
        Self::Config,
        Self::NeuronParameters,
        Self::RecurrentTargets,
        Self::RecurrentMetadata,
        Self::RecurrentWeights,
        Self::InputProjection,
        Self::OutputWeights,
        Self::OutputBias,
        Self::ContextProjection,
        Self::TrainingStatistics,
        Self::GenerationMetadata,
        Self::SemanticsContract,
    ];

    fn from_u32(value: u32) -> LeoResult<Self> {
        match value {
            1 => Ok(Self::Config),
            2 => Ok(Self::NeuronParameters),
            3 => Ok(Self::RecurrentTargets),
            4 => Ok(Self::RecurrentMetadata),
            5 => Ok(Self::RecurrentWeights),
            6 => Ok(Self::InputProjection),
            7 => Ok(Self::OutputWeights),
            8 => Ok(Self::OutputBias),
            9 => Ok(Self::ContextProjection),
            10 => Ok(Self::TrainingStatistics),
            11 => Ok(Self::GenerationMetadata),
            12 => Ok(Self::SemanticsContract),
            _ => Err(LeoError::checkpoint_corrupt(format!(
                "unknown checkpoint section type: {value}"
            ))),
        }
    }

    const fn expected_dtype(self) -> DType {
        match self {
            Self::RecurrentWeights | Self::OutputWeights | Self::OutputBias => DType::F32,
            Self::GenerationMetadata => DType::U64,
            Self::SemanticsContract => DType::U32,
            _ => DType::Bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum DType {
    Bytes = 1,
    F32 = 2,
    U32 = 3,
    U64 = 4,
}

impl DType {
    fn from_u32(value: u32) -> LeoResult<Self> {
        match value {
            1 => Ok(Self::Bytes),
            2 => Ok(Self::F32),
            3 => Ok(Self::U32),
            4 => Ok(Self::U64),
            _ => Err(LeoError::checkpoint_corrupt(format!(
                "unknown checkpoint dtype: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
struct EncodedSection {
    kind: SectionKind,
    payload: Vec<u8>,
    dtype: DType,
    shape: [u64; 4],
}

#[derive(Debug, Clone)]
struct Descriptor {
    kind: SectionKind,
    schema_version: u32,
    dtype: DType,
    rank: u32,
    shape: [u64; 4],
    offset: u64,
    length: u64,
    alignment: u32,
    compression: u32,
    digest: ArtifactDigest,
}

#[derive(Clone, Copy)]
struct DecodedSection<'a> {
    descriptor: &'a Descriptor,
    payload: &'a [u8],
}

pub fn save_model_atomic(
    path: impl AsRef<Path>,
    model: &Model,
    retain_previous: bool,
) -> LeoResult<()> {
    model.validate()?;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    if retain_previous && path.exists() {
        let previous_generation = read_generation(path)?;
        let generation_path = generation_path(path, previous_generation);
        if !generation_path.exists() {
            fs::copy(path, &generation_path)?;
        }
    }

    let bytes = encode_model(model)?;
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut file = File::create(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
    Ok(())
}

pub fn commit_model(path: impl AsRef<Path>, model: &mut Model) -> LeoResult<()> {
    let previous_generation = model.generation;
    model.generation = model.generation.saturating_add(1);
    if let Err(error) = save_model_atomic(path, model, true) {
        model.generation = previous_generation;
        return Err(error);
    }
    Ok(())
}

pub fn commit_model_rolling(
    path: impl AsRef<Path>,
    model: &mut Model,
    keep_previous_generations: usize,
) -> LeoResult<()> {
    let path = path.as_ref();
    let previous_generation = model.generation;
    model.generation = model.generation.saturating_add(1);
    if let Err(error) = save_model_atomic(path, model, keep_previous_generations > 0) {
        model.generation = previous_generation;
        return Err(error);
    }
    prune_generations(path, keep_previous_generations)?;
    Ok(())
}

fn prune_generations(path: &Path, keep_previous_generations: usize) -> LeoResult<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| LeoError::checkpoint_corrupt(format!(
            "checkpoint path has no file name: {}",
            path.display()
        )))?
        .to_string_lossy();
    let prefix = format!("{file_name}.gen");
    let mut generations = Vec::new();
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(suffix) = name.strip_prefix(&prefix) else {
            continue;
        };
        if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(generation) = suffix.parse::<u64>() else {
            continue;
        };
        generations.push((generation, entry.path()));
    }
    generations.sort_unstable_by_key(|(generation, _)| *generation);
    let remove_count = generations.len().saturating_sub(keep_previous_generations);
    for (_, old_path) in generations.into_iter().take(remove_count) {
        fs::remove_file(old_path)?;
    }
    Ok(())
}

pub fn load_model(path: impl AsRef<Path>) -> LeoResult<Model> {
    let path = path.as_ref();
    let length = fs::metadata(path)?.len();
    if length > MAX_CHECKPOINT_BYTES {
        return Err(LeoError::checkpoint_corrupt(format!(
            "checkpoint is too large: {length} bytes"
        )));
    }
    let bytes = fs::read(path)?;
    decode_model(&bytes)
}

pub fn rollback_model(path: impl AsRef<Path>, generation: u64) -> LeoResult<Model> {
    let path = path.as_ref();
    let source = generation_path(path, generation);
    if !source.exists() {
        return Err(LeoError::checkpoint_corrupt(format!(
            "generation {generation} not found at {}",
            source.display()
        )));
    }
    let model = load_model(&source)?;
    save_model_atomic(path, &model, true)?;
    Ok(model)
}

pub fn read_generation(path: impl AsRef<Path>) -> LeoResult<u64> {
    let mut file = File::open(path)?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;
    validate_header(&header)?;
    Ok(u64::from_le_bytes(header[16..24].try_into().unwrap()))
}

pub fn checkpoint_hash(path: impl AsRef<Path>) -> LeoResult<ArtifactDigest> {
    Ok(digest_file(path)?)
}

pub fn checkpoint_digest(path: impl AsRef<Path>) -> LeoResult<ArtifactDigest> {
    checkpoint_hash(path)
}

fn generation_path(path: &Path, generation: u64) -> PathBuf {
    PathBuf::from(format!("{}.gen{:06}", path.display(), generation))
}

fn encode_model(model: &Model) -> LeoResult<Vec<u8>> {
    let semantics = encode_semantics_contract(SemanticsContract::CURRENT);
    let sections = vec![
        EncodedSection {
            kind: SectionKind::Config,
            payload: model.config.to_toml().into_bytes(),
            dtype: DType::Bytes,
            shape: [0, 0, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::NeuronParameters,
            payload: encode_neurons(&model.neurons),
            dtype: DType::Bytes,
            shape: [model.neuron_count() as u64, 0, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::RecurrentTargets,
            payload: encode_recurrent_targets(&model.recurrent),
            dtype: DType::Bytes,
            shape: [model.neuron_count() as u64, model.recurrent.capacity_per_neuron as u64, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::RecurrentMetadata,
            payload: encode_recurrent_metadata(&model.recurrent),
            dtype: DType::Bytes,
            shape: [model.neuron_count() as u64, model.recurrent.capacity_per_neuron as u64, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::RecurrentWeights,
            payload: encode_vec_f32(&model.recurrent.weight),
            dtype: DType::F32,
            shape: [model.recurrent.weight.len() as u64, 0, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::InputProjection,
            payload: encode_input(&model.input),
            dtype: DType::Bytes,
            shape: [leo_core::symbols::SYMBOL_COUNT as u64, model.input.fanout as u64, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::OutputWeights,
            payload: encode_vec_f32(&model.output.weights),
            dtype: DType::F32,
            shape: [leo_core::symbols::OUTPUT_CLASSES as u64, model.neuron_count() as u64, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::OutputBias,
            payload: encode_vec_f32(&model.output.bias),
            dtype: DType::F32,
            shape: [model.output.bias.len() as u64, 0, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::ContextProjection,
            payload: encode_context(&model.context),
            dtype: DType::Bytes,
            shape: [model.context.observations.len() as u64, model.config.context.embedding_dim as u64, leo_core::symbols::OUTPUT_CLASSES as u64, 0],
        },
        EncodedSection {
            kind: SectionKind::TrainingStatistics,
            payload: encode_statistics(&model.statistics),
            dtype: DType::Bytes,
            shape: [9, 0, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::GenerationMetadata,
            payload: encode_generation(model.generation, model.parameter_revision),
            dtype: DType::U64,
            shape: [2, 0, 0, 0],
        },
        EncodedSection {
            kind: SectionKind::SemanticsContract,
            payload: semantics,
            dtype: DType::U32,
            shape: [5, 0, 0, 0],
        },
    ];

    let table_end = HEADER_SIZE
        .checked_add(DESCRIPTOR_SIZE.saturating_mul(sections.len()))
        .ok_or_else(|| LeoError::checkpoint_corrupt("checkpoint descriptor table overflow"))?;
    let mut cursor = align_up(table_end, ALIGNMENT);
    let mut descriptors = Vec::with_capacity(sections.len());
    for section in &sections {
        let rank = rank_for_shape(section.shape);
        descriptors.push(Descriptor {
            kind: section.kind,
            schema_version: 1,
            dtype: section.dtype,
            rank,
            shape: section.shape,
            offset: cursor as u64,
            length: section.payload.len() as u64,
            alignment: ALIGNMENT as u32,
            compression: 0,
            digest: digest_bytes(&section.payload),
        });
        cursor = align_up(
            cursor
                .checked_add(section.payload.len())
                .ok_or_else(|| LeoError::checkpoint_corrupt("checkpoint size overflow"))?,
            ALIGNMENT,
        );
    }

    let config_descriptor = descriptors
        .iter()
        .find(|descriptor| descriptor.kind == SectionKind::Config)
        .ok_or_else(|| LeoError::checkpoint_corrupt("missing CONFIG section"))?;
    let mut output = vec![0u8; cursor];
    output[0..8].copy_from_slice(MAGIC);
    output[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    output[12..16].copy_from_slice(&ENDIAN_MARKER.to_le_bytes());
    output[16..24].copy_from_slice(&model.generation.to_le_bytes());
    output[24..28].copy_from_slice(&(sections.len() as u32).to_le_bytes());
    output[28..32].copy_from_slice(&0u32.to_le_bytes());
    output[32..40].copy_from_slice(&config_descriptor.offset.to_le_bytes());
    output[40..48].copy_from_slice(&config_descriptor.length.to_le_bytes());

    for (index, descriptor) in descriptors.iter().enumerate() {
        let start = HEADER_SIZE + index * DESCRIPTOR_SIZE;
        encode_descriptor(&mut output[start..start + DESCRIPTOR_SIZE], descriptor);
    }
    for (section, descriptor) in sections.iter().zip(descriptors.iter()) {
        let start = descriptor.offset as usize;
        output[start..start + section.payload.len()].copy_from_slice(&section.payload);
    }
    let header_digest = digest_bytes(&output[..HEADER_SIZE]);
    output[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + DIGEST_BYTES]
        .copy_from_slice(header_digest.as_bytes());
    Ok(output)
}

fn decode_model(bytes: &[u8]) -> LeoResult<Model> {
    if bytes.len() < HEADER_SIZE {
        return Err(LeoError::checkpoint_corrupt("checkpoint is smaller than the v1 header"));
    }
    validate_header(&bytes[..HEADER_SIZE])?;
    let generation = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let section_count = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
    if section_count != SectionKind::ALL.len() {
        return Err(LeoError::checkpoint_corrupt(format!(
            "invalid checkpoint section count {section_count}; expected {}",
            SectionKind::ALL.len()
        )));
    }
    if u32::from_le_bytes(bytes[28..32].try_into().unwrap()) != 0 {
        return Err(LeoError::checkpoint_corrupt("unsupported checkpoint header flags"));
    }
    let table_end = HEADER_SIZE
        .checked_add(section_count.saturating_mul(DESCRIPTOR_SIZE))
        .ok_or_else(|| LeoError::checkpoint_corrupt("descriptor table overflow"))?;
    if table_end > bytes.len() {
        return Err(LeoError::checkpoint_corrupt("truncated checkpoint descriptor table"));
    }
    let payload_region_start = align_up(table_end, ALIGNMENT);

    let mut descriptors = Vec::with_capacity(section_count);
    for index in 0..section_count {
        let start = HEADER_SIZE + index * DESCRIPTOR_SIZE;
        descriptors.push(decode_descriptor(&bytes[start..start + DESCRIPTOR_SIZE])?);
    }

    let mut ranges = Vec::with_capacity(section_count);
    for descriptor in &descriptors {
        validate_descriptor(descriptor, payload_region_start, bytes.len())?;
        let start = usize::try_from(descriptor.offset)
            .map_err(|_| LeoError::checkpoint_corrupt("section offset exceeds address space"))?;
        let length = usize::try_from(descriptor.length)
            .map_err(|_| LeoError::checkpoint_corrupt("section length exceeds address space"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| LeoError::checkpoint_corrupt("section range overflow"))?;
        ranges.push((start, end, descriptor.kind));
    }
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    let mut previous_end = payload_region_start;
    for (start, end, kind) in &ranges {
        if *start < previous_end {
            return Err(LeoError::checkpoint_corrupt(format!(
                "overlapping {:?} checkpoint section",
                kind
            )));
        }
        previous_end = *end;
    }

    let mut sections = BTreeMap::new();
    for descriptor in &descriptors {
        let start = descriptor.offset as usize;
        let end = start + descriptor.length as usize;
        let payload = &bytes[start..end];
        if digest_bytes(payload) != descriptor.digest {
            return Err(LeoError::checkpoint_corrupt(format!(
                "SHA-256 mismatch in {:?} section",
                descriptor.kind
            )));
        }
        if sections
            .insert(
                descriptor.kind,
                DecodedSection {
                    descriptor,
                    payload,
                },
            )
            .is_some()
        {
            return Err(LeoError::checkpoint_corrupt(format!(
                "duplicate {:?} checkpoint section",
                descriptor.kind
            )));
        }
    }
    for kind in SectionKind::ALL {
        if !sections.contains_key(&kind) {
            return Err(LeoError::checkpoint_corrupt(format!(
                "missing {:?} checkpoint section",
                kind
            )));
        }
    }

    let config_section = required(&sections, SectionKind::Config)?;
    if config_section.payload.len() > MAX_CONFIG_BYTES {
        return Err(LeoError::checkpoint_corrupt("CONFIG section exceeds 1 MiB"));
    }
    let config_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let config_length = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
    if config_section.descriptor.offset != config_offset
        || config_section.descriptor.length != config_length
    {
        return Err(LeoError::checkpoint_corrupt(
            "CONFIG header pointer does not match descriptor",
        ));
    }

    let contract = decode_semantics_contract(required(&sections, SectionKind::SemanticsContract)?.payload)?;
    if contract != SemanticsContract::CURRENT {
        return Err(LeoError::checkpoint_incompatible(format!(
            "checkpoint semantics {:?} do not match Leo v1.0.0 semantics {:?}",
            contract,
            SemanticsContract::CURRENT
        )));
    }

    let config_text = std::str::from_utf8(config_section.payload)
        .map_err(|error| LeoError::checkpoint_corrupt(format!("invalid CONFIG UTF-8: {error}")))?;
    let config = Config::from_toml(config_text)?;
    let neurons = decode_neurons(required(&sections, SectionKind::NeuronParameters)?.payload)?;
    let (capacity_per_neuron, target_neuron) =
        decode_recurrent_targets(required(&sections, SectionKind::RecurrentTargets)?.payload)?;
    let (target_branch, delay) =
        decode_recurrent_metadata(required(&sections, SectionKind::RecurrentMetadata)?.payload)?;
    let recurrent = SynapseArrays {
        capacity_per_neuron,
        target_neuron,
        target_branch,
        delay,
        weight: decode_vec_f32(required(&sections, SectionKind::RecurrentWeights)?.payload)?,
    };
    let input = decode_input(required(&sections, SectionKind::InputProjection)?.payload)?;
    let output = OutputProjection {
        weights: decode_vec_f32(required(&sections, SectionKind::OutputWeights)?.payload)?,
        bias: decode_vec_f32(required(&sections, SectionKind::OutputBias)?.payload)?,
    };
    let context = decode_context(required(&sections, SectionKind::ContextProjection)?.payload)?;
    let statistics = decode_statistics(required(&sections, SectionKind::TrainingStatistics)?.payload)?;
    let (section_generation, parameter_revision) =
        decode_generation(required(&sections, SectionKind::GenerationMetadata)?.payload)?;
    if section_generation != generation {
        return Err(LeoError::checkpoint_corrupt(
            "generation metadata does not match checkpoint header",
        ));
    }

    let model = Model {
        config,
        generation,
        parameter_revision,
        neurons,
        recurrent,
        input,
        output,
        context,
        statistics,
    };
    model.validate()?;
    validate_shapes(&sections, &model)?;
    Ok(model)
}

fn validate_header(header: &[u8]) -> LeoResult<()> {
    if header.len() != HEADER_SIZE {
        return Err(LeoError::checkpoint_corrupt("invalid checkpoint header size"));
    }
    if &header[0..8] != MAGIC {
        return Err(LeoError::checkpoint_incompatible(
            "unsupported checkpoint magic; Leo v1.0.0 accepts only PSCLS100 artifacts",
        ));
    }
    let format_version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if format_version != FORMAT_VERSION {
        return Err(LeoError::checkpoint_incompatible(format!(
            "unsupported checkpoint format version {format_version}; expected {FORMAT_VERSION}"
        )));
    }
    if u32::from_le_bytes(header[12..16].try_into().unwrap()) != ENDIAN_MARKER {
        return Err(LeoError::checkpoint_incompatible("unsupported checkpoint endianness"));
    }
    if header[48..HEADER_DIGEST_OFFSET].iter().any(|byte| *byte != 0) {
        return Err(LeoError::checkpoint_corrupt("checkpoint header reserved bytes are nonzero"));
    }
    let declared = digest_from_slice(&header[HEADER_DIGEST_OFFSET..HEADER_SIZE]);
    let mut normalized = header.to_vec();
    normalized[HEADER_DIGEST_OFFSET..HEADER_SIZE].fill(0);
    if digest_bytes(&normalized) != declared {
        return Err(LeoError::checkpoint_corrupt("checkpoint header SHA-256 mismatch"));
    }
    Ok(())
}

fn validate_descriptor(
    descriptor: &Descriptor,
    payload_region_start: usize,
    file_len: usize,
) -> LeoResult<()> {
    if descriptor.schema_version != 1 {
        return Err(LeoError::checkpoint_incompatible(format!(
            "unsupported schema version {} in {:?}",
            descriptor.schema_version, descriptor.kind
        )));
    }
    if descriptor.dtype != descriptor.kind.expected_dtype() {
        return Err(LeoError::checkpoint_corrupt(format!(
            "invalid dtype in {:?} section",
            descriptor.kind
        )));
    }
    if descriptor.rank != rank_for_shape(descriptor.shape) || descriptor.rank > 4 {
        return Err(LeoError::checkpoint_corrupt(format!(
            "invalid rank/shape in {:?} section",
            descriptor.kind
        )));
    }
    if descriptor.alignment as usize != ALIGNMENT {
        return Err(LeoError::checkpoint_corrupt(format!(
            "invalid alignment in {:?} section",
            descriptor.kind
        )));
    }
    if descriptor.compression != 0 {
        return Err(LeoError::checkpoint_incompatible(
            "compressed checkpoint sections are not supported in v1",
        ));
    }
    let start = usize::try_from(descriptor.offset)
        .map_err(|_| LeoError::checkpoint_corrupt("section offset exceeds address space"))?;
    let length = usize::try_from(descriptor.length)
        .map_err(|_| LeoError::checkpoint_corrupt("section length exceeds address space"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| LeoError::checkpoint_corrupt("section range overflow"))?;
    if start < payload_region_start || start % ALIGNMENT != 0 || end > file_len {
        return Err(LeoError::checkpoint_corrupt(format!(
            "invalid byte range for {:?} section",
            descriptor.kind
        )));
    }
    Ok(())
}

fn validate_shapes(
    sections: &BTreeMap<SectionKind, DecodedSection<'_>>,
    model: &Model,
) -> LeoResult<()> {
    let expected = [
        (SectionKind::Config, [0, 0, 0, 0]),
        (SectionKind::NeuronParameters, [model.neuron_count() as u64, 0, 0, 0]),
        (SectionKind::RecurrentTargets, [model.neuron_count() as u64, model.recurrent.capacity_per_neuron as u64, 0, 0]),
        (SectionKind::RecurrentMetadata, [model.neuron_count() as u64, model.recurrent.capacity_per_neuron as u64, 0, 0]),
        (SectionKind::RecurrentWeights, [model.recurrent.weight.len() as u64, 0, 0, 0]),
        (SectionKind::InputProjection, [leo_core::symbols::SYMBOL_COUNT as u64, model.input.fanout as u64, 0, 0]),
        (SectionKind::OutputWeights, [leo_core::symbols::OUTPUT_CLASSES as u64, model.neuron_count() as u64, 0, 0]),
        (SectionKind::OutputBias, [model.output.bias.len() as u64, 0, 0, 0]),
        (SectionKind::ContextProjection, [model.context.observations.len() as u64, model.config.context.embedding_dim as u64, leo_core::symbols::OUTPUT_CLASSES as u64, 0]),
        (SectionKind::TrainingStatistics, [9, 0, 0, 0]),
        (SectionKind::GenerationMetadata, [2, 0, 0, 0]),
        (SectionKind::SemanticsContract, [5, 0, 0, 0]),
    ];
    for (kind, shape) in expected {
        if required(sections, kind)?.descriptor.shape != shape {
            return Err(LeoError::checkpoint_corrupt(format!(
                "shape mismatch in {:?} section",
                kind
            )));
        }
    }
    Ok(())
}

fn required<'a>(
    sections: &'a BTreeMap<SectionKind, DecodedSection<'a>>,
    kind: SectionKind,
) -> LeoResult<DecodedSection<'a>> {
    sections
        .get(&kind)
        .copied()
        .ok_or_else(|| LeoError::checkpoint_corrupt(format!("missing {:?} section", kind)))
}

fn encode_descriptor(output: &mut [u8], descriptor: &Descriptor) {
    debug_assert_eq!(output.len(), DESCRIPTOR_SIZE);
    output.fill(0);
    output[0..4].copy_from_slice(&(descriptor.kind as u32).to_le_bytes());
    output[4..8].copy_from_slice(&descriptor.schema_version.to_le_bytes());
    output[8..12].copy_from_slice(&(descriptor.dtype as u32).to_le_bytes());
    output[12..16].copy_from_slice(&descriptor.rank.to_le_bytes());
    for (index, dimension) in descriptor.shape.iter().enumerate() {
        let start = 16 + index * 8;
        output[start..start + 8].copy_from_slice(&dimension.to_le_bytes());
    }
    output[48..56].copy_from_slice(&descriptor.offset.to_le_bytes());
    output[56..64].copy_from_slice(&descriptor.length.to_le_bytes());
    output[64..68].copy_from_slice(&descriptor.alignment.to_le_bytes());
    output[68..72].copy_from_slice(&descriptor.compression.to_le_bytes());
    output[72..104].copy_from_slice(descriptor.digest.as_bytes());
}

fn decode_descriptor(input: &[u8]) -> LeoResult<Descriptor> {
    if input.len() != DESCRIPTOR_SIZE {
        return Err(LeoError::checkpoint_corrupt("invalid checkpoint descriptor size"));
    }
    if input[104..128].iter().any(|byte| *byte != 0) {
        return Err(LeoError::checkpoint_corrupt("checkpoint descriptor reserved bytes are nonzero"));
    }
    let kind = SectionKind::from_u32(u32::from_le_bytes(input[0..4].try_into().unwrap()))?;
    let mut shape = [0u64; 4];
    for (index, dimension) in shape.iter_mut().enumerate() {
        let start = 16 + index * 8;
        *dimension = u64::from_le_bytes(input[start..start + 8].try_into().unwrap());
    }
    Ok(Descriptor {
        kind,
        schema_version: u32::from_le_bytes(input[4..8].try_into().unwrap()),
        dtype: DType::from_u32(u32::from_le_bytes(input[8..12].try_into().unwrap()))?,
        rank: u32::from_le_bytes(input[12..16].try_into().unwrap()),
        shape,
        offset: u64::from_le_bytes(input[48..56].try_into().unwrap()),
        length: u64::from_le_bytes(input[56..64].try_into().unwrap()),
        alignment: u32::from_le_bytes(input[64..68].try_into().unwrap()),
        compression: u32::from_le_bytes(input[68..72].try_into().unwrap()),
        digest: digest_from_slice(&input[72..104]),
    })
}

fn encode_generation(generation: u64, parameter_revision: u64) -> Vec<u8> {
    let mut output = Vec::with_capacity(16);
    put_u64(&mut output, generation);
    put_u64(&mut output, parameter_revision);
    output
}

fn decode_generation(input: &[u8]) -> LeoResult<(u64, u64)> {
    if input.len() != 16 {
        return Err(LeoError::checkpoint_corrupt("invalid generation metadata length"));
    }
    let mut reader = Reader::new(input);
    let generation = reader.u64()?;
    let parameter_revision = reader.u64()?;
    reader.finish()?;
    Ok((generation, parameter_revision))
}

fn encode_semantics_contract(contract: SemanticsContract) -> Vec<u8> {
    let mut output = Vec::with_capacity(20);
    for value in contract.as_array() {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn decode_semantics_contract(input: &[u8]) -> LeoResult<SemanticsContract> {
    if input.len() != 20 {
        return Err(LeoError::checkpoint_corrupt("invalid semantics contract length"));
    }
    let mut values = [0u32; 5];
    for (index, value) in values.iter_mut().enumerate() {
        let start = index * 4;
        *value = u32::from_le_bytes(input[start..start + 4].try_into().unwrap());
    }
    Ok(SemanticsContract {
        model_schema: values[0],
        training_policy: values[1],
        execution_semantics: values[2],
        dataset_schema: values[3],
        cuda_abi: values[4],
    })
}

fn rank_for_shape(shape: [u64; 4]) -> u32 {
    let mut rank = 0u32;
    let mut saw_zero = false;
    for dimension in shape {
        if dimension == 0 {
            saw_zero = true;
        } else if saw_zero {
            return 5;
        } else {
            rank += 1;
        }
    }
    rank
}

fn digest_from_slice(input: &[u8]) -> ArtifactDigest {
    debug_assert_eq!(input.len(), DIGEST_BYTES);
    let mut bytes = [0u8; DIGEST_BYTES];
    bytes.copy_from_slice(input);
    ArtifactDigest::from_array(bytes)
}
fn encode_neurons(state: &NeuronParameters) -> Vec<u8> {
    let mut output = Vec::new();
    put_vec_f32(&mut output, &state.threshold);
    put_vec_f32(&mut output, &state.excitability);
    put_vec_u8(&mut output, &state.neuron_type);
    output
}

fn decode_neurons(input: &[u8]) -> LeoResult<NeuronParameters> {
    let mut reader = Reader::new(input);
    let state = NeuronParameters {
        threshold: reader.vec_f32()?,
        excitability: reader.vec_f32()?,
        neuron_type: reader.vec_u8()?,
    };
    reader.finish()?;
    Ok(state)
}

fn encode_recurrent_targets(state: &SynapseArrays) -> Vec<u8> {
    let mut output = Vec::new();
    put_u64(&mut output, state.capacity_per_neuron as u64);
    put_vec_u32(&mut output, &state.target_neuron);
    output
}

fn decode_recurrent_targets(input: &[u8]) -> LeoResult<(usize, Vec<u32>)> {
    let mut reader = Reader::new(input);
    let capacity_per_neuron = reader.u64()? as usize;
    let target_neuron = reader.vec_u32()?;
    reader.finish()?;
    Ok((capacity_per_neuron, target_neuron))
}

fn encode_recurrent_metadata(state: &SynapseArrays) -> Vec<u8> {
    let mut output = Vec::new();
    put_vec_u8(&mut output, &state.target_branch);
    put_vec_u8(&mut output, &state.delay);
    output
}

type RecurrentMetadata = (Vec<u8>, Vec<u8>);

fn decode_recurrent_metadata(input: &[u8]) -> LeoResult<RecurrentMetadata> {
    let mut reader = Reader::new(input);
    let target_branch = reader.vec_u8()?;
    let delay = reader.vec_u8()?;
    reader.finish()?;
    Ok((target_branch, delay))
}

fn encode_input(input: &InputProjection) -> Vec<u8> {
    let mut output = Vec::new();
    put_u64(&mut output, input.fanout as u64);
    put_vec_u32(&mut output, &input.targets);
    put_vec_u8(&mut output, &input.branches);
    put_vec_f32(&mut output, &input.weights);
    output
}

fn decode_input(input: &[u8]) -> LeoResult<InputProjection> {
    let mut reader = Reader::new(input);
    let projection = InputProjection {
        fanout: reader.u64()? as usize,
        targets: reader.vec_u32()?,
        branches: reader.vec_u8()?,
        weights: reader.vec_f32()?,
    };
    reader.finish()?;
    Ok(projection)
}

fn encode_context(context: &ContextProjection) -> Vec<u8> {
    let mut output = Vec::new();
    put_vec_u64(&mut output, &context.keys);
    put_vec_f32(&mut output, &context.embeddings);
    put_vec_u32(&mut output, &context.observations);
    put_vec_f32(&mut output, &context.output_weights);
    output
}

fn decode_context(input: &[u8]) -> LeoResult<ContextProjection> {
    let mut reader = Reader::new(input);
    let context = ContextProjection {
        keys: reader.vec_u64()?,
        embeddings: reader.vec_f32()?,
        observations: reader.vec_u32()?,
        output_weights: reader.vec_f32()?,
    };
    reader.finish()?;
    Ok(context)
}

fn encode_statistics(stats: &TrainingStatistics) -> Vec<u8> {
    let mut output = Vec::new();
    put_u64(&mut output, stats.processed_bytes);
    put_u64(&mut output, stats.processed_stories);
    put_f64(&mut output, stats.training_loss_sum);
    put_u64(&mut output, stats.training_targets);
    put_u64(&mut output, stats.active_neurons_sum);
    put_u64(&mut output, stats.active_neurons_peak);
    put_u64(&mut output, stats.synaptic_events);
    put_u64(&mut output, stats.numerical_rejections);
    put_u64(&mut output, stats.persistent_ticks);
    output
}

fn decode_statistics(input: &[u8]) -> LeoResult<TrainingStatistics> {
    const FIELD_COUNT: usize = 9;
    if input.len() != FIELD_COUNT * 8 {
        return Err(LeoError::checkpoint_corrupt(format!(
            "invalid training statistics length: {}",
            input.len()
        )));
    }

    let mut reader = Reader::new(input);
    let processed_bytes = reader.u64()?;
    let stats = TrainingStatistics {
        processed_bytes,
        processed_stories: reader.u64()?,
        training_loss_sum: reader.f64()?,
        training_targets: reader.u64()?,
        active_neurons_sum: reader.u64()?,
        active_neurons_peak: reader.u64()?,
        synaptic_events: reader.u64()?,
        numerical_rejections: reader.u64()?,
        persistent_ticks: reader.u64()?,
    };
    reader.finish()?;
    Ok(stats)
}

fn encode_vec_f32(values: &[f32]) -> Vec<u8> {
    let mut output = Vec::new();
    put_vec_f32(&mut output, values);
    output
}

fn decode_vec_f32(input: &[u8]) -> LeoResult<Vec<f32>> {
    let mut reader = Reader::new(input);
    let values = reader.vec_f32()?;
    reader.finish()?;
    Ok(values)
}

fn put_vec_f32(output: &mut Vec<u8>, values: &[f32]) {
    put_u64(output, values.len() as u64);
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn put_vec_u64(output: &mut Vec<u8>, values: &[u64]) {
    put_u64(output, values.len() as u64);
    for value in values {
        put_u64(output, *value);
    }
}

fn put_vec_u32(output: &mut Vec<u8>, values: &[u32]) {
    put_u64(output, values.len() as u64);
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn put_vec_u8(output: &mut Vec<u8>, values: &[u8]) {
    put_u64(output, values.len() as u64);
    output.extend_from_slice(values);
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(output: &mut Vec<u8>, value: f64) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    fn take(&mut self, length: usize) -> LeoResult<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| LeoError::checkpoint_corrupt("checkpoint cursor overflow"))?;
        if end > self.input.len() {
            return Err(LeoError::checkpoint_corrupt("truncated checkpoint tensor"));
        }
        let output = &self.input[self.position..end];
        self.position = end;
        Ok(output)
    }

    fn u64(&mut self) -> LeoResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> LeoResult<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn vec_f32(&mut self) -> LeoResult<Vec<f32>> {
        let length = self.u64()? as usize;
        let bytes = self.take(
            length
                .checked_mul(4)
                .ok_or_else(|| LeoError::checkpoint_corrupt("vector overflow"))?,
        )?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn vec_u64(&mut self) -> LeoResult<Vec<u64>> {
        let length = self.u64()? as usize;
        let bytes = self.take(
            length
                .checked_mul(8)
                .ok_or_else(|| LeoError::checkpoint_corrupt("vector overflow"))?,
        )?;
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn vec_u32(&mut self) -> LeoResult<Vec<u32>> {
        let length = self.u64()? as usize;
        let bytes = self.take(
            length
                .checked_mul(4)
                .ok_or_else(|| LeoError::checkpoint_corrupt("vector overflow"))?,
        )?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn vec_u8(&mut self) -> LeoResult<Vec<u8>> {
        let length = self.u64()? as usize;
        Ok(self.take(length)?.to_vec())
    }

    fn finish(&self) -> LeoResult<()> {
        if self.position == self.input.len() {
            Ok(())
        } else {
            Err(LeoError::checkpoint_corrupt("unexpected trailing checkpoint data"))
        }
    }
}


fn align_up(value: usize, alignment: usize) -> usize {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_input, decode_model, decode_statistics, encode_model, put_u64, put_vec_u32,
        put_vec_u8, DESCRIPTOR_SIZE, HEADER_SIZE,
    };
    use leo_core::{Config, Model};

    fn sample_checkpoint() -> Vec<u8> {
        let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
        let model = Model::initialize(config).unwrap();
        encode_model(&model).unwrap()
    }

    #[test]
    fn duplicate_section_descriptors_are_rejected() {
        let mut bytes = sample_checkpoint();
        let first_kind = bytes[HEADER_SIZE..HEADER_SIZE + 4].to_vec();
        let second = HEADER_SIZE + DESCRIPTOR_SIZE;
        bytes[second..second + 4].copy_from_slice(&first_kind);
        assert!(decode_model(&bytes).is_err());
    }

    #[test]
    fn overlapping_section_ranges_are_rejected() {
        let mut bytes = sample_checkpoint();
        let first_offset = bytes[HEADER_SIZE + 48..HEADER_SIZE + 56].to_vec();
        let second = HEADER_SIZE + DESCRIPTOR_SIZE;
        bytes[second + 48..second + 56].copy_from_slice(&first_offset);
        assert!(decode_model(&bytes).is_err());
    }

    #[test]
    fn statistics_require_every_persisted_field() {
        let truncated = vec![0u8; 8 * 8];
        assert!(decode_statistics(&truncated).is_err());
    }

    #[test]
    fn input_projection_requires_complete_weight_tensor() {
        let mut truncated = Vec::new();
        put_u64(&mut truncated, 2);
        put_vec_u32(&mut truncated, &[0, 1, 2, 3]);
        put_vec_u8(&mut truncated, &[0, 0, 2, 3]);
        assert!(decode_input(&truncated).is_err());
    }
}
