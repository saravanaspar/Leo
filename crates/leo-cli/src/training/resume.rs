//! Atomic training-resume/checkpoint transaction state.

use leo_core::{ArtifactDigest, BackendKind, BackendRuntime, LeoError, LeoResult, Model};
use leo_format::{checkpoint_hash, commit_model_rolling, load_model, save_model_atomic};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) const TRAINING_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5 * 60);
pub(super) const TRAINING_CHECKPOINT_PREVIOUS_GENERATIONS: usize = 1;

const TRAINING_RESUME_MAGIC: &str = "LEOTRAIN100";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TrainingResumeState {
    pub(crate) model_generation: u64,
    pub(crate) model_parameter_revision: u64,
    pub(crate) model_processed_stories: u64,
    pub(crate) train_dataset_id: ArtifactDigest,
    pub(crate) dataset_len: usize,
    pub(crate) story_limit: usize,
    pub(crate) passes: usize,
    pub(crate) workers: usize,
    pub(crate) backend: BackendKind,
    pub(crate) synchronization: String,
    pub(crate) max_training_bytes: Option<u64>,
    pub(crate) validation_dataset_id: Option<ArtifactDigest>,
    pub(crate) validation_story_limit: Option<usize>,
    pub(crate) next_pass: usize,
    pub(crate) next_position: usize,
    pub(crate) presentations: usize,
    pub(crate) input_bytes_seen: u64,
    pub(crate) last_validation_processed_bytes: u64,
    pub(crate) patience_best: f64,
    pub(crate) stale_checks: usize,
    pub(crate) best_validation_loss: Option<f64>,
    pub(crate) best_parameter_revision: Option<u64>,
    pub(crate) best_checkpoint_digest: Option<ArtifactDigest>,
    pub(crate) finalized: bool,
}

impl TrainingResumeState {
    pub(crate) fn encode(&self) -> String {
        format!(
            concat!(
                "{}\n",
                "model_generation={}\n",
                "model_parameter_revision={}\n",
                "model_processed_stories={}\n",
                "train_dataset_id={}\n",
                "dataset_len={}\n",
                "story_limit={}\n",
                "passes={}\n",
                "workers={}\n",
                "backend={}\n",
                "synchronization={}\n",
                "max_training_bytes={}\n",
                "validation_dataset_id={}\n",
                "validation_story_limit={}\n",
                "next_pass={}\n",
                "next_position={}\n",
                "presentations={}\n",
                "input_bytes_seen={}\n",
                "last_validation_processed_bytes={}\n",
                "patience_best_bits={:016x}\n",
                "stale_checks={}\n",
                "best_validation_loss_bits={}\n",
                "best_parameter_revision={}\n",
                "best_checkpoint_digest={}\n",
                "finalized={}\n",
            ),
            TRAINING_RESUME_MAGIC,
            self.model_generation,
            self.model_parameter_revision,
            self.model_processed_stories,
            self.train_dataset_id,
            self.dataset_len,
            self.story_limit,
            self.passes,
            self.workers,
            self.backend,
            self.synchronization,
            encode_optional_u64(self.max_training_bytes),
            encode_optional_digest(self.validation_dataset_id),
            encode_optional_usize(self.validation_story_limit),
            self.next_pass,
            self.next_position,
            self.presentations,
            self.input_bytes_seen,
            self.last_validation_processed_bytes,
            self.patience_best.to_bits(),
            self.stale_checks,
            encode_optional_f64_bits(self.best_validation_loss),
            encode_optional_u64(self.best_parameter_revision),
            encode_optional_digest(self.best_checkpoint_digest),
            self.finalized,
        )
    }

    pub(crate) fn decode(text: &str) -> LeoResult<Self> {
        let mut lines = text.lines();
        if lines.next() != Some(TRAINING_RESUME_MAGIC) {
            return Err(LeoError::dataset(
                "invalid or obsolete training resume state header; start a fresh Leo v1.0.1 run",
            ));
        }
        let mut values = BTreeMap::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                LeoError::dataset(format!("invalid training resume state line: {line}"))
            })?;
            if values.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(LeoError::dataset(format!(
                    "duplicate training resume state field: {key}"
                )));
            }
        }
        let patience_bits =
            u64::from_str_radix(training_resume_value(&values, "patience_best_bits")?, 16)
                .map_err(|_| LeoError::dataset("invalid training resume patience bits"))?;
        Ok(Self {
            model_generation: training_resume_u64(&values, "model_generation")?,
            model_parameter_revision: training_resume_u64(&values, "model_parameter_revision")?,
            model_processed_stories: training_resume_u64(&values, "model_processed_stories")?,
            train_dataset_id: training_resume_digest(&values, "train_dataset_id")?,
            dataset_len: training_resume_usize(&values, "dataset_len")?,
            story_limit: training_resume_usize(&values, "story_limit")?,
            passes: training_resume_usize(&values, "passes")?,
            workers: training_resume_usize(&values, "workers")?,
            backend: training_resume_value(&values, "backend")?.parse()?,
            synchronization: training_resume_value(&values, "synchronization")?.to_owned(),
            max_training_bytes: decode_optional_u64(training_resume_value(
                &values,
                "max_training_bytes",
            )?)?,
            validation_dataset_id: decode_optional_digest(training_resume_value(
                &values,
                "validation_dataset_id",
            )?)?,
            validation_story_limit: decode_optional_usize(training_resume_value(
                &values,
                "validation_story_limit",
            )?)?,
            next_pass: training_resume_usize(&values, "next_pass")?,
            next_position: training_resume_usize(&values, "next_position")?,
            presentations: training_resume_usize(&values, "presentations")?,
            input_bytes_seen: training_resume_u64(&values, "input_bytes_seen")?,
            last_validation_processed_bytes: training_resume_u64(
                &values,
                "last_validation_processed_bytes",
            )?,
            patience_best: f64::from_bits(patience_bits),
            stale_checks: training_resume_usize(&values, "stale_checks")?,
            best_validation_loss: decode_optional_f64_bits(training_resume_value(
                &values,
                "best_validation_loss_bits",
            )?)?,
            best_parameter_revision: decode_optional_u64(training_resume_value(
                &values,
                "best_parameter_revision",
            )?)?,
            best_checkpoint_digest: decode_optional_digest(training_resume_value(
                &values,
                "best_checkpoint_digest",
            )?)?,
            finalized: training_resume_bool(&values, "finalized")?,
        })
    }
}

fn training_resume_value<'a>(
    values: &'a BTreeMap<String, String>,
    key: &str,
) -> LeoResult<&'a str> {
    values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| LeoError::internal(format!("training resume state is missing {key}")))
}

fn training_resume_u64(values: &BTreeMap<String, String>, key: &str) -> LeoResult<u64> {
    training_resume_value(values, key)?
        .parse::<u64>()
        .map_err(|_| LeoError::internal(format!("invalid training resume state value for {key}")))
}

fn training_resume_digest(
    values: &BTreeMap<String, String>,
    key: &str,
) -> LeoResult<ArtifactDigest> {
    ArtifactDigest::from_hex(training_resume_value(values, key)?).map_err(|error| {
        LeoError::dataset(format!("invalid training resume digest for {key}: {error}"))
    })
}

fn training_resume_usize(values: &BTreeMap<String, String>, key: &str) -> LeoResult<usize> {
    training_resume_value(values, key)?
        .parse::<usize>()
        .map_err(|_| LeoError::internal(format!("invalid training resume state value for {key}")))
}

fn training_resume_bool(values: &BTreeMap<String, String>, key: &str) -> LeoResult<bool> {
    match training_resume_value(values, key)? {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(LeoError::internal(format!(
            "invalid training resume state value for {key}"
        ))),
    }
}

fn encode_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn encode_optional_digest(value: Option<ArtifactDigest>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn encode_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn encode_optional_f64_bits(value: Option<f64>) -> String {
    value
        .map(|value| format!("{:016x}", value.to_bits()))
        .unwrap_or_else(|| "-".into())
}

fn decode_optional_u64(value: &str) -> LeoResult<Option<u64>> {
    if value == "-" {
        Ok(None)
    } else {
        value
            .parse::<u64>()
            .map(Some)
            .map_err(|_| LeoError::internal("invalid optional u64 in training resume state"))
    }
}

fn decode_optional_digest(value: &str) -> LeoResult<Option<ArtifactDigest>> {
    if value == "-" {
        Ok(None)
    } else {
        ArtifactDigest::from_hex(value).map(Some).map_err(|error| {
            LeoError::dataset(format!(
                "invalid optional digest in training resume state: {error}"
            ))
        })
    }
}

fn decode_optional_usize(value: &str) -> LeoResult<Option<usize>> {
    if value == "-" {
        Ok(None)
    } else {
        value
            .parse::<usize>()
            .map(Some)
            .map_err(|_| LeoError::internal("invalid optional usize in training resume state"))
    }
}

fn decode_optional_f64_bits(value: &str) -> LeoResult<Option<f64>> {
    if value == "-" {
        Ok(None)
    } else {
        u64::from_str_radix(value, 16)
            .map(|bits| Some(f64::from_bits(bits)))
            .map_err(|_| LeoError::internal("invalid optional f64 in training resume state"))
    }
}

pub(super) fn training_resume_path(model_path: &str) -> PathBuf {
    PathBuf::from(format!("{model_path}.trainstate"))
}

fn training_resume_pending_path(model_path: &str) -> PathBuf {
    PathBuf::from(format!("{model_path}.trainstate.pending"))
}

fn training_best_path(model_path: &str, parameter_revision: u64) -> PathBuf {
    PathBuf::from(format!(
        "{model_path}.trainbest.rev{parameter_revision:012}.pscls"
    ))
}

pub(super) fn save_training_resume_state(
    path: &Path,
    state: &TrainingResumeState,
) -> LeoResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut file = File::create(&temporary)?;
    file.write_all(state.encode().as_bytes())?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
    Ok(())
}

fn load_training_resume_state(path: &Path) -> LeoResult<TrainingResumeState> {
    let text = fs::read_to_string(path).map_err(|error| {
        LeoError::internal(format!("cannot read training resume state: {error}"))
    })?;
    TrainingResumeState::decode(&text)
}

pub(super) fn resume_state_matches_model(state: &TrainingResumeState, model: &Model) -> bool {
    state.model_generation == model.generation
        && state.model_parameter_revision == model.parameter_revision
        && state.model_processed_stories == model.statistics.processed_stories
}

fn promote_training_resume_pending(model_path: &str) -> LeoResult<()> {
    let pending = training_resume_pending_path(model_path);
    let stable = training_resume_path(model_path);
    match fs::rename(&pending, &stable) {
        Ok(()) => {}
        Err(first_error) if stable.exists() => {
            fs::remove_file(&stable)?;
            fs::rename(&pending, &stable).map_err(|second_error| {
                LeoError::internal(format!(
                    "cannot promote training resume state ({first_error}; {second_error})"
                ))
            })?;
        }
        Err(error) => return Err(error.into()),
    }
    if let Some(parent) = stable.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
    Ok(())
}

pub(super) fn load_training_resume_for_model(
    model_path: &str,
    model: &Model,
) -> LeoResult<Option<TrainingResumeState>> {
    let stable_path = training_resume_path(model_path);
    let pending_path = training_resume_pending_path(model_path);
    let stable = if stable_path.exists() {
        Some(load_training_resume_state(&stable_path)?)
    } else {
        None
    };
    let pending = if pending_path.exists() {
        Some(load_training_resume_state(&pending_path)?)
    } else {
        None
    };

    if let Some(state) = stable.as_ref() {
        if resume_state_matches_model(state, model) {
            if pending_path.exists() {
                fs::remove_file(&pending_path)?;
            }
            return Ok(stable);
        }
    }

    if let Some(state) = pending.as_ref() {
        if resume_state_matches_model(state, model) {
            promote_training_resume_pending(model_path)?;
            println!(
                "{{\"event\":\"training_resume_recovered\",\"generation\":{},\"reason\":\"completed_checkpoint_transaction\"}}",
                model.generation,
            );
            return Ok(pending);
        }
    }

    if stable.is_some() || pending.is_some() {
        return Err(LeoError::internal(
            "training resume state does not match the current checkpoint; restore the matching model or use --fresh-run to intentionally start a new run",
        ));
    }
    Ok(None)
}

fn synchronization_modes_equivalent(
    saved: &str,
    current: &str,
    backend: BackendKind,
    workers: usize,
) -> bool {
    if saved == current {
        return true;
    }
    if backend != BackendKind::Gpu || workers <= 1 {
        return false;
    }
    let exact_gpu_story_mean = |value: &str| {
        matches!(
            value,
            "gpu_story_mean_exact_v1"
                | "gpu_shared_wavefront_mean"
                | "gpu_multi_device_story_mean_exact"
        )
    };
    exact_gpu_story_mean(saved) && exact_gpu_story_mean(current)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_training_resume_identity(
    state: &TrainingResumeState,
    train_dataset_id: ArtifactDigest,
    dataset_len: usize,
    story_limit: usize,
    passes: usize,
    workers: usize,
    backend: BackendKind,
    synchronization: &str,
    max_training_bytes: Option<u64>,
    validation_dataset_id: Option<ArtifactDigest>,
    validation_story_limit: Option<usize>,
) -> LeoResult<()> {
    let mut mismatches = Vec::new();
    if state.train_dataset_id != train_dataset_id {
        mismatches.push("training dataset identity");
    }
    if state.dataset_len != dataset_len {
        mismatches.push("dataset length");
    }
    if state.story_limit != story_limit {
        mismatches.push("--max-stories");
    }
    if state.passes != passes {
        mismatches.push("--passes");
    }
    if state.workers != workers {
        mismatches.push("--workers");
    }
    if state.backend != backend {
        mismatches.push("resolved backend");
    }
    if !synchronization_modes_equivalent(&state.synchronization, synchronization, backend, workers)
    {
        mismatches.push("training synchronization mode");
    }
    if state.max_training_bytes != max_training_bytes {
        mismatches.push("--max-bytes");
    }
    if state.validation_dataset_id != validation_dataset_id {
        mismatches.push("validation dataset identity");
    }
    if state.validation_story_limit != validation_story_limit {
        mismatches.push("--validation-stories");
    }
    if state.next_pass > passes
        || (state.next_pass == passes && state.next_position != 0)
        || state.next_position > story_limit
    {
        mismatches.push("saved pass/story cursor");
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(LeoError::dataset(format!(
            "training resume state is incompatible with this command ({}); rerun the original command or pass --fresh-run to intentionally start a new run",
            mismatches.join(", ")
        )))
    }
}

pub(super) fn load_resume_best_model(
    model_path: &str,
    state: &TrainingResumeState,
) -> LeoResult<Option<(f64, Model)>> {
    match (
        state.best_validation_loss,
        state.best_parameter_revision,
        state.best_checkpoint_digest,
    ) {
        (None, None, None) => Ok(None),
        (Some(loss), Some(parameter_revision), Some(expected_hash)) => {
            let path = training_best_path(model_path, parameter_revision);
            if !path.exists() {
                return Err(LeoError::internal(format!(
                    "training resume best-validation snapshot is missing: {}",
                    path.display()
                )));
            }
            let actual_hash = checkpoint_hash(&path)?;
            if actual_hash != expected_hash {
                return Err(LeoError::internal(format!(
                    "training resume best-validation snapshot checksum mismatch: {}",
                    path.display()
                )));
            }
            let model = load_model(&path)?;
            if model.parameter_revision != parameter_revision {
                return Err(LeoError::internal(
                    "training resume best-validation snapshot revision mismatch",
                ));
            }
            Ok(Some((loss, model)))
        }
        _ => Err(LeoError::internal(
            "training resume state has incomplete best-validation metadata",
        )),
    }
}

fn persist_resume_best_model(
    model_path: &str,
    state: &mut TrainingResumeState,
    best_validation: &Option<(f64, Model)>,
) -> LeoResult<()> {
    if let Some((loss, best_model)) = best_validation {
        let path = training_best_path(model_path, best_model.parameter_revision);
        let already_persisted = path.exists()
            && state.best_parameter_revision == Some(best_model.parameter_revision)
            && state.best_checkpoint_digest.is_some();
        if !already_persisted {
            save_model_atomic(&path, best_model, false)?;
            state.best_checkpoint_digest = Some(checkpoint_hash(&path)?);
        }
        state.best_validation_loss = Some(*loss);
        state.best_parameter_revision = Some(best_model.parameter_revision);
    } else {
        state.best_validation_loss = None;
        state.best_parameter_revision = None;
        state.best_checkpoint_digest = None;
    }
    Ok(())
}

pub(super) fn persist_training_resume_for_saved_model(
    model_path: &str,
    model: &Model,
    state: &mut TrainingResumeState,
    best_validation: &Option<(f64, Model)>,
) -> LeoResult<()> {
    if !resume_state_matches_model(state, model) {
        return Err(LeoError::internal(
            "refusing to persist a training cursor ahead of the saved model checkpoint",
        ));
    }
    persist_resume_best_model(model_path, state, best_validation)?;
    save_training_resume_state(&training_resume_path(model_path), state)
}

pub(super) fn commit_training_checkpoint(
    model_path: &str,
    runtime: &mut BackendRuntime,
    state: &mut TrainingResumeState,
    best_validation: &Option<(f64, Model)>,
) -> LeoResult<()> {
    let previous_generation = runtime.model().generation;
    state.model_generation = previous_generation.saturating_add(1);
    state.model_parameter_revision = runtime.model().parameter_revision;
    state.model_processed_stories = runtime.model().statistics.processed_stories;
    persist_resume_best_model(model_path, state, best_validation)?;
    let pending_path = training_resume_pending_path(model_path);
    save_training_resume_state(&pending_path, state)?;

    if let Err(error) = commit_model_rolling(
        model_path,
        runtime.model_mut()?,
        TRAINING_CHECKPOINT_PREVIOUS_GENERATIONS,
    ) {
        let _ = fs::remove_file(&pending_path);
        state.model_generation = runtime.model().generation;
        return Err(error);
    }
    if !resume_state_matches_model(state, runtime.model()) {
        return Err(LeoError::internal(
            "checkpoint committed but training resume transaction did not match the saved model",
        ));
    }
    promote_training_resume_pending(model_path)
}

pub(super) fn cleanup_training_resume_artifacts(model_path: &str) -> LeoResult<()> {
    let state_path = training_resume_path(model_path);
    if state_path.exists() {
        fs::remove_file(&state_path)?;
    }
    let state_tmp = PathBuf::from(format!("{}.tmp", state_path.display()));
    if state_tmp.exists() {
        fs::remove_file(state_tmp)?;
    }
    let pending_path = training_resume_pending_path(model_path);
    if pending_path.exists() {
        fs::remove_file(&pending_path)?;
    }
    let pending_tmp = PathBuf::from(format!("{}.tmp", pending_path.display()));
    if pending_tmp.exists() {
        fs::remove_file(pending_tmp)?;
    }

    let model_path = Path::new(model_path);
    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = model_path
        .file_name()
        .ok_or_else(|| LeoError::internal("model path has no file name"))?
        .to_string_lossy();
    let prefix = format!("{file_name}.trainbest.rev");
    if parent.exists() {
        for entry in fs::read_dir(parent)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(&prefix) {
                fs::remove_file(entry.path())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn advance_checkpoint_deadline(
    mut deadline: Duration,
    elapsed: Duration,
    interval: Duration,
) -> Duration {
    while deadline <= elapsed {
        deadline += interval;
    }
    deadline
}

#[cfg(test)]
mod synchronization_mode_tests {
    use super::synchronization_modes_equivalent;
    use leo_core::BackendKind;

    #[test]
    fn exact_gpu_story_mean_resume_aliases_ignore_physical_gpu_count() {
        for saved in [
            "gpu_story_mean_exact_v1",
            "gpu_shared_wavefront_mean",
            "gpu_multi_device_story_mean_exact",
        ] {
            assert!(synchronization_modes_equivalent(
                saved,
                "gpu_story_mean_exact_v1",
                BackendKind::Gpu,
                16,
            ));
        }
    }

    #[test]
    fn experimental_device_mean_modes_are_not_treated_as_exact_aliases() {
        assert!(!synchronization_modes_equivalent(
            "gpu_experimental_device_mean",
            "gpu_story_mean_exact_v1",
            BackendKind::Gpu,
            16,
        ));
        assert!(!synchronization_modes_equivalent(
            "gpu_shared_wavefront_mean",
            "gpu_story_mean_exact_v1",
            BackendKind::Cpu,
            16,
        ));
    }
}
