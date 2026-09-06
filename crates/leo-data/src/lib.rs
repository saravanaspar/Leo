//! Versioned, self-validating prepared dataset artifacts.
//!
//! The `.bytes` file contains the exact concatenated UTF-8 story bytes.  The
//! `.idx` file carries a versioned header, strong content digests, provenance
//! digest, immutable DatasetId, and fixed-width records.

use leo_core::artifact::{digest_bytes, digest_file, ArtifactDigest, Sha256, DIGEST_BYTES};
use leo_core::semantics::DATASET_SCHEMA_VERSION;
use std::fs::File;
use std::io::{Read, Write};
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

pub const INDEX_MAGIC: &[u8; 8] = b"LEODATA1";
pub const INDEX_HEADER_SIZE: usize = 192;
pub const INDEX_RECORD_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoryIndex {
    pub offset: u64,
    pub length: u32,
    pub split_flags: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetIdentity {
    pub schema_version: u32,
    pub record_count: u64,
    pub bytes_length: u64,
    pub bytes_digest: ArtifactDigest,
    pub records_digest: ArtifactDigest,
    pub provenance_digest: ArtifactDigest,
    pub dataset_id: ArtifactDigest,
}

pub struct PreparedDataset {
    bytes: File,
    entries: Arc<[StoryIndex]>,
    identity: DatasetIdentity,
}

impl PreparedDataset {
    pub fn open(
        bytes_path: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let bytes_path = bytes_path.as_ref();
        let index_path = index_path.as_ref();
        let bytes = File::open(bytes_path)?;
        let bytes_length = bytes.metadata()?.len();
        let bytes_digest = digest_file(bytes_path)?;

        let mut index_file = File::open(index_path)?;
        let mut raw = Vec::new();
        index_file.read_to_end(&mut raw)?;
        if raw.len() < INDEX_HEADER_SIZE {
            return invalid_data("dataset index is smaller than the v1 header");
        }
        if &raw[0..8] != INDEX_MAGIC {
            return invalid_data("invalid dataset index magic; regenerate data for Leo v1.0.0");
        }
        let schema_version = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        if schema_version != DATASET_SCHEMA_VERSION {
            return invalid_data(format!(
                "unsupported dataset schema version {schema_version}; expected {DATASET_SCHEMA_VERSION}"
            ));
        }
        let header_size = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        if header_size != INDEX_HEADER_SIZE {
            return invalid_data(format!("invalid dataset index header size {header_size}"));
        }
        let record_count = u64::from_le_bytes(raw[16..24].try_into().unwrap());
        let declared_bytes_length = u64::from_le_bytes(raw[24..32].try_into().unwrap());
        let declared_bytes_digest = digest_from_slice(&raw[32..64]);
        let declared_records_digest = digest_from_slice(&raw[64..96]);
        let provenance_digest = digest_from_slice(&raw[96..128]);
        let declared_dataset_id = digest_from_slice(&raw[128..160]);
        let declared_header_digest = digest_from_slice(&raw[160..192]);

        let mut header_for_digest = raw[..INDEX_HEADER_SIZE].to_vec();
        header_for_digest[160..192].fill(0);
        if digest_bytes(&header_for_digest) != declared_header_digest {
            return invalid_data("dataset index header SHA-256 mismatch");
        }
        if declared_bytes_length != bytes_length {
            return invalid_data(format!(
                "dataset byte length mismatch: index declares {declared_bytes_length}, file has {bytes_length}"
            ));
        }
        if declared_bytes_digest != bytes_digest {
            return invalid_data("dataset bytes SHA-256 mismatch");
        }

        let records_len = usize::try_from(record_count)
            .ok()
            .and_then(|count| count.checked_mul(INDEX_RECORD_SIZE))
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "dataset record table overflow"))?;
        let expected_len = INDEX_HEADER_SIZE
            .checked_add(records_len)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "dataset index length overflow"))?;
        if raw.len() != expected_len {
            return invalid_data(format!(
                "dataset index length mismatch: expected {expected_len}, got {}",
                raw.len()
            ));
        }
        let record_bytes = &raw[INDEX_HEADER_SIZE..];
        if digest_bytes(record_bytes) != declared_records_digest {
            return invalid_data("dataset record-table SHA-256 mismatch");
        }
        let computed_dataset_id = compute_dataset_id(
            schema_version,
            record_count,
            bytes_length,
            bytes_digest,
            declared_records_digest,
            provenance_digest,
        );
        if computed_dataset_id != declared_dataset_id {
            return invalid_data("dataset ID does not match content and provenance");
        }

        let mut entries = Vec::with_capacity(record_count as usize);
        let mut previous_end = 0u64;
        for (record_index, chunk) in record_bytes.chunks_exact(INDEX_RECORD_SIZE).enumerate() {
            let entry = StoryIndex {
                offset: u64::from_le_bytes(chunk[0..8].try_into().unwrap()),
                length: u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                split_flags: u32::from_le_bytes(chunk[12..16].try_into().unwrap()),
            };
            if entry.length == 0 {
                return invalid_data(format!("dataset story {record_index} has zero length"));
            }
            let end = entry
                .offset
                .checked_add(entry.length as u64)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "dataset story range overflow"))?;
            if end > bytes_length {
                return invalid_data(format!("dataset story {record_index} extends past the bytes file"));
            }
            if entry.offset < previous_end {
                return invalid_data(format!("dataset story {record_index} overlaps a previous story"));
            }
            previous_end = end;
            entries.push(entry);
        }

        Ok(Self {
            bytes,
            entries: entries.into(),
            identity: DatasetIdentity {
                schema_version,
                record_count,
                bytes_length,
                bytes_digest,
                records_digest: declared_records_digest,
                provenance_digest,
                dataset_id: declared_dataset_id,
            },
        })
    }

    pub fn identity(&self) -> DatasetIdentity {
        self.identity
    }

    /// Clone an already-verified dataset handle without re-hashing the artifact.
    /// The cloned file descriptor points at the same verified bytes artifact and
    /// the immutable parsed index table is shared.
    pub fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self {
            bytes: self.bytes.try_clone()?,
            entries: Arc::clone(&self.entries),
            identity: self.identity,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entry(&self, index: usize) -> Option<StoryIndex> {
        self.entries.get(index).copied()
    }

    pub fn story(&mut self, index: usize) -> std::io::Result<Vec<u8>> {
        let entry = self.entries.get(index).copied().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "story index out of range")
        })?;
        let mut story = vec![0u8; entry.length as usize];
        #[cfg(unix)]
        self.bytes.read_exact_at(&mut story, entry.offset)?;
        #[cfg(not(unix))]
        {
            self.bytes.seek(SeekFrom::Start(entry.offset))?;
            self.bytes.read_exact(&mut story)?;
        }
        std::str::from_utf8(&story).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("story is not valid UTF-8: {error}"),
            )
        })?;
        Ok(story)
    }
}

pub fn compute_dataset_id(
    schema_version: u32,
    record_count: u64,
    bytes_length: u64,
    bytes_digest: ArtifactDigest,
    records_digest: ArtifactDigest,
    provenance_digest: ArtifactDigest,
) -> ArtifactDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"LEO-DATASET-ID\0");
    hasher.update(&schema_version.to_le_bytes());
    hasher.update(&record_count.to_le_bytes());
    hasher.update(&bytes_length.to_le_bytes());
    hasher.update(bytes_digest.as_bytes());
    hasher.update(records_digest.as_bytes());
    hasher.update(provenance_digest.as_bytes());
    hasher.finalize()
}

pub fn write_index(
    path: impl AsRef<Path>,
    bytes_path: impl AsRef<Path>,
    entries: &[StoryIndex],
    provenance_digest: ArtifactDigest,
) -> std::io::Result<DatasetIdentity> {
    let bytes_path = bytes_path.as_ref();
    let bytes_length = std::fs::metadata(bytes_path)?.len();
    let bytes_digest = digest_file(bytes_path)?;
    let mut records = Vec::with_capacity(entries.len().saturating_mul(INDEX_RECORD_SIZE));
    let mut previous_end = 0u64;
    for (index, entry) in entries.iter().enumerate() {
        if entry.length == 0 {
            return invalid_data(format!("dataset story {index} has zero length"));
        }
        let end = entry
            .offset
            .checked_add(entry.length as u64)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "dataset story range overflow"))?;
        if entry.offset < previous_end || end > bytes_length {
            return invalid_data(format!("invalid range for dataset story {index}"));
        }
        previous_end = end;
        records.extend_from_slice(&entry.offset.to_le_bytes());
        records.extend_from_slice(&entry.length.to_le_bytes());
        records.extend_from_slice(&entry.split_flags.to_le_bytes());
    }
    let records_digest = digest_bytes(&records);
    let record_count = entries.len() as u64;
    let dataset_id = compute_dataset_id(
        DATASET_SCHEMA_VERSION,
        record_count,
        bytes_length,
        bytes_digest,
        records_digest,
        provenance_digest,
    );
    let mut header = vec![0u8; INDEX_HEADER_SIZE];
    header[0..8].copy_from_slice(INDEX_MAGIC);
    header[8..12].copy_from_slice(&DATASET_SCHEMA_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(INDEX_HEADER_SIZE as u32).to_le_bytes());
    header[16..24].copy_from_slice(&record_count.to_le_bytes());
    header[24..32].copy_from_slice(&bytes_length.to_le_bytes());
    header[32..64].copy_from_slice(bytes_digest.as_bytes());
    header[64..96].copy_from_slice(records_digest.as_bytes());
    header[96..128].copy_from_slice(provenance_digest.as_bytes());
    header[128..160].copy_from_slice(dataset_id.as_bytes());
    let header_digest = digest_bytes(&header);
    header[160..192].copy_from_slice(header_digest.as_bytes());

    let path = path.as_ref();
    let mut file = File::create(path)?;
    file.write_all(&header)?;
    file.write_all(&records)?;
    file.sync_all()?;

    Ok(DatasetIdentity {
        schema_version: DATASET_SCHEMA_VERSION,
        record_count,
        bytes_length,
        bytes_digest,
        records_digest,
        provenance_digest,
        dataset_id,
    })
}

fn digest_from_slice(input: &[u8]) -> ArtifactDigest {
    debug_assert_eq!(input.len(), DIGEST_BYTES);
    let mut bytes = [0u8; DIGEST_BYTES];
    bytes.copy_from_slice(input);
    ArtifactDigest::from_array(bytes)
}

fn invalid_data<T>(message: impl Into<String>) -> std::io::Result<T> {
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{write_index, PreparedDataset, StoryIndex};
    use leo_core::{digest_bytes, ArtifactDigest};
    use std::fs;

    #[test]
    fn reads_story_by_offset_and_verifies_identity() {
        let base = std::env::temp_dir().join(format!("leo-data-{}", std::process::id()));
        let bytes = base.with_extension("bytes");
        let index = base.with_extension("idx");
        fs::write(&bytes, b"onetwo").unwrap();
        let provenance = digest_bytes(b"test provenance");
        let identity = write_index(
            &index,
            &bytes,
            &[
                StoryIndex {
                    offset: 0,
                    length: 3,
                    split_flags: 0,
                },
                StoryIndex {
                    offset: 3,
                    length: 3,
                    split_flags: 0,
                },
            ],
            provenance,
        )
        .unwrap();
        let mut dataset = PreparedDataset::open(&bytes, &index).unwrap();
        assert_eq!(dataset.identity(), identity);
        assert_eq!(dataset.story(1).unwrap(), b"two");
        fs::remove_file(bytes).unwrap();
        fs::remove_file(index).unwrap();
    }

    #[test]
    fn changed_middle_bytes_are_rejected() {
        let base = std::env::temp_dir().join(format!("leo-data-mutation-{}", std::process::id()));
        let bytes = base.with_extension("bytes");
        let index = base.with_extension("idx");
        fs::write(&bytes, b"abcdef").unwrap();
        write_index(
            &index,
            &bytes,
            &[StoryIndex {
                offset: 0,
                length: 6,
                split_flags: 0,
            }],
            ArtifactDigest::ZERO,
        )
        .unwrap();
        fs::write(&bytes, b"abcXef").unwrap();
        assert!(PreparedDataset::open(&bytes, &index).is_err());
        fs::remove_file(bytes).unwrap();
        fs::remove_file(index).unwrap();
    }
}
