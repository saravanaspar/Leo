//! Verified dataset access, deterministic story ordering, and one-batch-ahead prefetch.

use leo_core::rng::SplitMix64;
use leo_core::{LeoError, LeoResult};
use leo_data::PreparedDataset;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

pub(crate) fn shuffled_story_order(story_count: usize, seed: u64, pass: usize) -> Vec<usize> {
    let mut order = (0..story_count).collect::<Vec<_>>();
    SplitMix64::new(seed.wrapping_add(pass as u64).wrapping_add(1)).shuffle(&mut order);
    order
}

pub(crate) fn open_dataset(bytes: &str, index: &str, purpose: &str) -> LeoResult<PreparedDataset> {
    let dataset = PreparedDataset::open(bytes, index)
        .map_err(|error| LeoError::dataset(format!("dataset error during {purpose}: {error}")))?;
    if dataset.is_empty() {
        return Err(LeoError::dataset(format!(
            "dataset error: {purpose} dataset is empty"
        )));
    }
    Ok(dataset)
}

#[derive(Clone, Copy)]
struct StoryBatchRequest {
    position: usize,
    workers: usize,
    remaining_bytes: Option<u64>,
}

type StoryBatchResult = (Vec<Vec<u8>>, usize, u64);

/// One-batch-ahead dataset prefetcher. The worker owns a cloned handle to the
/// already-verified dataset, so prefetching never re-hashes the artifact and
/// never changes story order, logical batch size, or byte-limit semantics.
pub(crate) struct StoryBatchPrefetcher {
    requests: Option<SyncSender<StoryBatchRequest>>,
    results: Receiver<LeoResult<StoryBatchResult>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl StoryBatchPrefetcher {
    pub(crate) fn spawn(
        dataset: &PreparedDataset,
        story_order: Vec<usize>,
        purpose: &'static str,
    ) -> LeoResult<Self> {
        let mut dataset = dataset.try_clone().map_err(|error| {
            LeoError::internal(format!(
                "could not clone {purpose} dataset for prefetch: {error}"
            ))
        })?;
        let (request_tx, request_rx) = sync_channel::<StoryBatchRequest>(1);
        let (result_tx, result_rx) = sync_channel::<LeoResult<StoryBatchResult>>(1);
        let worker = thread::Builder::new()
            .name("leo-dataset-prefetch".to_owned())
            .spawn(move || {
                while let Ok(request) = request_rx.recv() {
                    let result = read_story_batch(
                        &mut dataset,
                        &story_order,
                        request.position,
                        request.workers,
                        request.remaining_bytes,
                        purpose,
                    );
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| {
                LeoError::internal(format!(
                    "could not start {purpose} dataset prefetcher: {error}"
                ))
            })?;
        Ok(Self {
            requests: Some(request_tx),
            results: result_rx,
            worker: Some(worker),
        })
    }

    pub(crate) fn request(
        &self,
        position: usize,
        workers: usize,
        remaining_bytes: Option<u64>,
    ) -> LeoResult<()> {
        self.requests
            .as_ref()
            .ok_or_else(|| LeoError::internal("dataset prefetcher is closed"))?
            .send(StoryBatchRequest {
                position,
                workers,
                remaining_bytes,
            })
            .map_err(|_| LeoError::internal("dataset prefetch worker stopped unexpectedly"))
    }

    pub(crate) fn receive(&self) -> LeoResult<StoryBatchResult> {
        self.results
            .recv()
            .map_err(|_| LeoError::internal("dataset prefetch worker stopped unexpectedly"))?
    }
}

impl Drop for StoryBatchPrefetcher {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) fn read_story_batch(
    dataset: &mut PreparedDataset,
    story_order: &[usize],
    position: usize,
    workers: usize,
    remaining_bytes: Option<u64>,
    purpose: &str,
) -> LeoResult<(Vec<Vec<u8>>, usize, u64)> {
    let batch_end = position.saturating_add(workers).min(story_order.len());
    let mut stories = Vec::with_capacity(batch_end.saturating_sub(position));
    let mut next_position = position;
    let mut input_bytes = 0u64;

    while next_position < batch_end {
        let story_index = story_order[next_position];
        let story_bytes = dataset
            .entry(story_index)
            .ok_or_else(|| LeoError::internal(format!("{purpose} story index is out of range")))?
            .length as u64;
        if remaining_bytes
            .is_some_and(|remaining| input_bytes.saturating_add(story_bytes) > remaining)
        {
            break;
        }
        let story = dataset
            .story(story_index)
            .map_err(|error| LeoError::internal(format!("{purpose} dataset error: {error}")))?;
        input_bytes = input_bytes.saturating_add(story_bytes);
        stories.push(story);
        next_position += 1;
    }

    Ok((stories, next_position, input_bytes))
}
