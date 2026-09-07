use leo_core::rng::SplitMix64;
use leo_core::symbols::{
    output_index_to_symbol, BEGIN_DOCUMENT, END_DOCUMENT, END_DOCUMENT_OUTPUT_INDEX, OUTPUT_CLASSES,
};
use leo_core::utf8::Utf8State;
use leo_core::{
    available_gpu_devices, ArtifactDigest, BackendCapabilities, BackendKind, BackendRuntime,
    Config, DeviceModelLayout, LeoError, LeoErrorKind, LeoResult, Model, Permission, Sha256,
};
use leo_format::{checkpoint_hash, commit_model, load_model, rollback_model, save_model_atomic};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::time::Instant;

mod training;

#[cfg(test)]
use training::{
    advance_checkpoint_deadline, read_story_batch, shuffled_story_order, TrainingResumeState,
};
use training::{
    configured_multi_gpu_devices, evaluate_model, open_dataset, print_learning_quality,
    print_prediction_evaluation, replay_target_range, run_training, train_document_pass,
    train_story_batch, ActivityDiagnostics, StoryBatchPrefetcher, TrainRequest, TrainingEngine,
    GPU_REFERENCE_WORKERS,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("leo: {error}");
        std::process::exit(exit_code(&error));
    }
}

fn run() -> LeoResult<()> {
    let mut raw = env::args().skip(1);
    let command = match raw.next() {
        Some(command) => command,
        None => {
            print_help();
            return Ok(());
        }
    };
    if is_help_command(&command) {
        print_help();
        return Ok(());
    }

    if !matches!(
        command.as_str(),
        "init"
            | "train"
            | "eval"
            | "prompt"
            | "teach"
            | "inspect"
            | "checkpoint"
            | "rollback"
            | "benchmark"
            | "backend"
    ) {
        return Err(LeoError::usage(format!("unknown command: {command}")));
    }

    let arguments = Arguments::parse_for(&command, raw.collect())?;
    if arguments.flag("help") {
        print_help();
        return Ok(());
    }

    match command.as_str() {
        "init" => command_init(&arguments),
        "train" => command_train(&arguments),
        "eval" => command_eval(&arguments),
        "prompt" => command_prompt(&arguments),
        "teach" => command_teach(&arguments),
        "inspect" => command_inspect(&arguments),
        "checkpoint" => command_checkpoint(&arguments),
        "rollback" => command_rollback(&arguments),
        "benchmark" => command_benchmark(&arguments),
        "backend" => command_backend(&arguments),
        _ => unreachable!("command validated before dispatch"),
    }
}

fn command_init(arguments: &Arguments) -> LeoResult<()> {
    let config_path = arguments.required("config")?;
    let output_path = arguments.required("output")?;
    let config = Config::from_file(config_path)?;
    let model = Model::initialize(config)?;
    save_model_atomic(output_path, &model, false)?;
    println!(
        "{{\"event\":\"initialized\",\"name\":\"{}\",\"parameter_revision\":{},\"neurons\":{},\"fixed_synapses\":{},\"context_slots\":{},\"context_embedding_dim\":{},\"model\":\"{}\"}}",
        json_escape(&model.config.model.name),
        model.parameter_revision,
        model.neuron_count(),
        model.recurrent.weight.len(),
        model.context.observations.len(),
        model.config.context.embedding_dim,
        json_escape(output_path),
    );
    Ok(())
}

fn command_train(arguments: &Arguments) -> LeoResult<()> {
    let validation_paths = paired_paths(arguments, "valid-bytes", "valid-index")?;
    run_training(TrainRequest {
        model_path: arguments.required("model")?,
        train_bytes: arguments.required("train-bytes")?,
        train_index: arguments.required("train-index")?,
        validation_paths,
        fresh_run: arguments.flag("fresh-run"),
        requested_backend: backend_from_arguments(arguments)?,
        passes: arguments.optional_usize("passes")?,
        story_limit: arguments.optional_usize("max-stories")?,
        workers: arguments.optional_usize("workers")?,
        max_training_bytes: arguments.optional_u64("max-bytes")?,
        validation_story_limit: arguments.optional_usize("validation-stories")?,
        log_every: arguments.optional_usize("log-every")?.unwrap_or(10),
    })
}

fn command_eval(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let bytes_path = arguments
        .value("bytes")
        .or_else(|| arguments.value("valid-bytes"))
        .ok_or_else(|| LeoError::internal("missing required option --bytes"))?;
    let index_path = arguments
        .value("index")
        .or_else(|| arguments.value("valid-index"))
        .ok_or_else(|| LeoError::internal("missing required option --index"))?;
    let story_limit = arguments.optional_usize("stories")?;
    let generation_story_limit = arguments
        .optional_usize("generation-stories")?
        .unwrap_or(20);
    let max_generation_bytes = arguments.optional_usize("max-bytes")?.unwrap_or(256);
    let model = load_model(model_path)?;
    let backend = backend_from_arguments(arguments)?;

    let held_out_dataset = open_dataset(bytes_path, index_path, "held-out evaluation")?;
    let held_out = evaluate_model(model.clone(), &held_out_dataset, story_limit, backend)?;
    print_prediction_evaluation("held_out_evaluation", &held_out);

    if let Some((train_bytes, train_index)) = paired_paths(arguments, "train-bytes", "train-index")?
    {
        let train_story_limit = arguments.optional_usize("train-stories")?;
        let training_dataset = open_dataset(train_bytes, train_index, "training evaluation")?;
        let training =
            evaluate_model(model.clone(), &training_dataset, train_story_limit, backend)?;
        print_prediction_evaluation("training_evaluation", &training);
        print_learning_quality(&training, &held_out);
    }

    let free_running = evaluate_free_running(
        model.clone(),
        bytes_path,
        index_path,
        story_limit,
        generation_story_limit,
        max_generation_bytes,
        backend,
    )?;
    println!(
        "{{\"event\":\"free_running_evaluation\",\"stories\":{},\"reference_bytes\":{},\"generated_bytes\":{},\"byte_accuracy\":{},\"mean_matching_prefix_fraction\":{},\"exact_completions\":{},\"natural_end_fraction\":{},\"mean_repetition_rate\":{}}}",
        free_running.stories,
        free_running.reference_bytes,
        free_running.generated_bytes,
        free_running.byte_accuracy(),
        free_running.mean_matching_prefix_fraction(),
        free_running.exact_completions,
        free_running.natural_end_fraction(),
        free_running.mean_repetition_rate(),
    );

    let prompt = arguments.value("prompt").unwrap_or("Once upon a time");
    let mut runtime = BackendRuntime::new(model, backend)?;
    let generation = generate_from_prefix(
        &mut runtime,
        prompt.as_bytes(),
        GenerationOptions {
            max_bytes: max_generation_bytes,
            temperature: arguments.optional_f32("temperature")?.unwrap_or(0.0),
            seed: arguments.optional_u64("seed")?.unwrap_or(1337),
        },
    )?;
    print_generation_evaluation("prompt_probe", prompt, &generation);
    Ok(())
}

fn command_prompt(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let prompt = arguments.required("text")?;
    let model = load_model(model_path)?;
    let model_seed = model.config.model.seed;
    let backend = backend_from_arguments(arguments)?;
    let mut runtime = BackendRuntime::new(model, backend)?;
    let generation = generate_from_prefix(
        &mut runtime,
        prompt.as_bytes(),
        GenerationOptions {
            max_bytes: arguments.optional_usize("max-bytes")?.unwrap_or(1000),
            temperature: arguments.optional_f32("temperature")?.unwrap_or(0.8),
            seed: arguments.optional_u64("seed")?.unwrap_or(model_seed),
        },
    )?;

    if arguments.flag("json") {
        print_generation_evaluation("generation", prompt, &generation);
    } else {
        println!("{prompt}{}", generation.text);
    }
    Ok(())
}

fn command_teach(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let text = arguments.required("text")?;
    let permission = Permission::parse(arguments.required("permission")?)?;
    if matches!(permission, Permission::Frozen) {
        return Err(LeoError::internal("teach requires a learning permission"));
    }

    let requested_backend = backend_from_arguments(arguments)?;
    let mut runtime = BackendRuntime::new(load_model(model_path)?, requested_backend)?;
    let before_hash = model_state_hash(runtime.model());
    let initial = train_document_pass(&mut runtime, text.as_bytes(), permission)?;

    let replay_count = if matches!(permission, Permission::Verified) {
        runtime.model().config.replay.max_verified_replays
    } else {
        runtime.model().config.replay.teaching_replays
    };
    let full_range = 0..text.len().saturating_add(1);
    let mut replay_activity = ActivityDiagnostics::default();
    for _ in 0..replay_count {
        replay_activity.add(replay_target_range(
            &mut runtime,
            text.as_bytes(),
            full_range.clone(),
            permission,
        )?);
    }

    runtime.synchronize_model()?;
    runtime.model().validate()?;
    commit_model(model_path, runtime.model_mut()?)?;
    let after_hash = model_state_hash(runtime.model());
    println!(
        "{{\"event\":\"teaching_committed\",\"generation\":{},\"permission\":\"{}\",\"initial_loss\":{},\"replays\":{},\"replay_steps\":{},\"changed_state\":{}}}",
        runtime.model().generation,
        permission_name(permission),
        initial.mean_loss,
        replay_count,
        replay_activity.steps,
        before_hash != after_hash,
    );
    Ok(())
}

fn command_inspect(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let model = load_model(model_path)?;
    let fixed_synapses = model.recurrent.weight.len();
    let observed_context_slots = model
        .context
        .observations
        .iter()
        .filter(|observations| **observations > 0)
        .count();
    let mean_threshold = mean_f32(&model.neurons.threshold);
    let mean_abs_weight = mean_abs_f32(model.recurrent.weight.iter().copied());
    let hash = checkpoint_hash(model_path)?;

    if arguments.flag("json") {
        println!(
            "{{\"event\":\"inspection\",\"name\":\"{}\",\"generation\":{},\"parameter_revision\":{},\"neurons\":{},\"fixed_synapses\":{},\"context_slots\":{},\"context_embedding_dim\":{},\"observed_context_slots\":{},\"mean_threshold\":{},\"mean_abs_recurrent_weight\":{},\"processed_bytes\":{},\"processed_stories\":{},\"training_bits_per_byte\":{},\"activity_fraction\":{},\"checkpoint_sha256\":\"{}\"}}",
            json_escape(&model.config.model.name),
            model.generation,
            model.parameter_revision,
            model.neuron_count(),
            fixed_synapses,
            model.context.observations.len(),
            model.config.context.embedding_dim,
            observed_context_slots,
            mean_threshold,
            mean_abs_weight,
            model.statistics.processed_bytes,
            model.statistics.processed_stories,
            model.statistics.bits_per_byte(),
            model.statistics.mean_active_neurons_per_tick() / model.neuron_count() as f64,
            hash,
        );
    } else {
        println!("name: {}", model.config.model.name);
        println!("generation: {}", model.generation);
        println!("parameter revision: {}", model.parameter_revision);
        println!("neurons: {}", model.neuron_count());
        println!("fixed synapses: {fixed_synapses}");
        println!("context slots: {}", model.context.observations.len());
        println!(
            "context embedding dim: {}",
            model.config.context.embedding_dim
        );
        println!("observed context slots: {observed_context_slots}");
        println!(
            "training bits/byte: {:.4}",
            model.statistics.bits_per_byte()
        );
        println!(
            "activity: {:.2}%",
            model.statistics.mean_active_neurons_per_tick() * 100.0 / model.neuron_count() as f64
        );
        println!("checkpoint SHA-256: {hash}");
    }
    Ok(())
}

#[derive(Debug)]
struct BatchConformanceResult {
    replay_fraction: f32,
    max_parameter_delta: f32,
    mean_loss_delta: f64,
    replay_segments: usize,
    replay_steps: u64,
    replay_prefix_steps: u64,
}

fn max_f32_slice_delta(left: &[f32], right: &[f32]) -> LeoResult<f32> {
    if left.len() != right.len() {
        return Err(LeoError::internal(
            "CPU/GPU conformance compared differently sized parameter arrays",
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(0.0f32, |maximum, (left, right)| {
            maximum.max((left - right).abs())
        }))
}

fn run_story_batch_conformance(
    base: &Model,
    replay_fraction: f32,
) -> LeoResult<BatchConformanceResult> {
    let mut conformance_model = base.clone();
    conformance_model.config.replay.fraction = replay_fraction;
    let stories = vec![
        b"red kite over a blue pond".to_vec(),
        b"small cat by a warm window".to_vec(),
        b"green boat on quiet water".to_vec(),
        b"little bird under the moon".to_vec(),
    ];

    let mut cpu = BackendRuntime::new(conformance_model.clone(), BackendKind::Cpu)?;
    let mut gpu = BackendRuntime::new(conformance_model, BackendKind::Gpu)?;
    let cpu_report = train_story_batch(&mut cpu, stories.clone(), Permission::Training)?;
    let gpu_report = train_story_batch(&mut gpu, stories, Permission::Training)?;
    cpu.synchronize_model()?;
    gpu.synchronize_model()?;

    if cpu_report.replay_segments != gpu_report.replay_segments
        || cpu_report.replay_steps != gpu_report.replay_steps
        || cpu_report.replay_prefix_steps != gpu_report.replay_prefix_steps
    {
        return Err(LeoError::internal(format!(
            "CUDA logical-batch replay parity failed at fraction {replay_fraction}: CPU segments/steps/prefix={}/{}/{}, GPU={}/{}/{}",
            cpu_report.replay_segments,
            cpu_report.replay_steps,
            cpu_report.replay_prefix_steps,
            gpu_report.replay_segments,
            gpu_report.replay_steps,
            gpu_report.replay_prefix_steps,
        )));
    }

    let cpu_model = cpu.model();
    let gpu_model = gpu.model();
    if cpu_model.parameter_revision != gpu_model.parameter_revision {
        return Err(LeoError::internal(format!(
            "CUDA logical-batch revision parity failed at replay fraction {replay_fraction}: CPU={}, GPU={}",
            cpu_model.parameter_revision, gpu_model.parameter_revision,
        )));
    }
    if cpu_model.context.keys != gpu_model.context.keys
        || cpu_model.context.observations != gpu_model.context.observations
    {
        return Err(LeoError::internal(format!(
            "CUDA logical-batch context identity parity failed at replay fraction {replay_fraction}"
        )));
    }

    let cpu_stats = &cpu_model.statistics;
    let gpu_stats = &gpu_model.statistics;
    if cpu_stats.processed_bytes != gpu_stats.processed_bytes
        || cpu_stats.processed_stories != gpu_stats.processed_stories
        || cpu_stats.training_targets != gpu_stats.training_targets
        || cpu_stats.active_neurons_sum != gpu_stats.active_neurons_sum
        || cpu_stats.active_neurons_peak != gpu_stats.active_neurons_peak
        || cpu_stats.synaptic_events != gpu_stats.synaptic_events
        || cpu_stats.numerical_rejections != gpu_stats.numerical_rejections
        || cpu_stats.persistent_ticks != gpu_stats.persistent_ticks
    {
        return Err(LeoError::internal(format!(
            "CUDA logical-batch statistics parity failed at replay fraction {replay_fraction}"
        )));
    }

    let mut max_parameter_delta = 0.0f32;
    for delta in [
        max_f32_slice_delta(&cpu_model.neurons.threshold, &gpu_model.neurons.threshold)?,
        max_f32_slice_delta(
            &cpu_model.neurons.excitability,
            &gpu_model.neurons.excitability,
        )?,
        max_f32_slice_delta(&cpu_model.recurrent.weight, &gpu_model.recurrent.weight)?,
        max_f32_slice_delta(&cpu_model.input.weights, &gpu_model.input.weights)?,
        max_f32_slice_delta(&cpu_model.output.weights, &gpu_model.output.weights)?,
        max_f32_slice_delta(&cpu_model.output.bias, &gpu_model.output.bias)?,
        max_f32_slice_delta(&cpu_model.context.embeddings, &gpu_model.context.embeddings)?,
        max_f32_slice_delta(
            &cpu_model.context.output_weights,
            &gpu_model.context.output_weights,
        )?,
    ] {
        max_parameter_delta = max_parameter_delta.max(delta);
    }
    let mean_loss_delta = (cpu_report.mean_loss - gpu_report.mean_loss).abs();
    let training_loss_sum_delta = (cpu_stats.training_loss_sum - gpu_stats.training_loss_sum).abs();
    const PARAMETER_TOLERANCE: f32 = 3.0e-3;
    const LOSS_TOLERANCE: f64 = 3.0e-3;
    if !max_parameter_delta.is_finite()
        || max_parameter_delta > PARAMETER_TOLERANCE
        || !mean_loss_delta.is_finite()
        || mean_loss_delta > LOSS_TOLERANCE
        || !training_loss_sum_delta.is_finite()
        || training_loss_sum_delta > LOSS_TOLERANCE * cpu_stats.training_targets.max(1) as f64
    {
        return Err(LeoError::internal(format!(
            "CUDA logical-batch numerical parity failed at replay fraction {replay_fraction}: max parameter delta={max_parameter_delta}, mean loss delta={mean_loss_delta}, training loss sum delta={training_loss_sum_delta}"
        )));
    }

    Ok(BatchConformanceResult {
        replay_fraction,
        max_parameter_delta,
        mean_loss_delta,
        replay_segments: gpu_report.replay_segments,
        replay_steps: gpu_report.replay_steps,
        replay_prefix_steps: gpu_report.replay_prefix_steps,
    })
}

fn command_backend(arguments: &Arguments) -> LeoResult<()> {
    let requested = backend_from_arguments(arguments)?;
    let gpu_devices = available_gpu_devices().unwrap_or(0);
    let (resolved, capabilities, execution_scope) = match requested {
        BackendKind::Auto | BackendKind::Cpu => (
            BackendKind::Cpu,
            BackendCapabilities::cpu(),
            "reference_sparse_cpu",
        ),
        BackendKind::Gpu => (
            BackendKind::Gpu,
            BackendCapabilities::gpu(),
            "cuda_shared_model_tiled_post_wavefront_learning",
        ),
    };

    if arguments.flag("json") {
        println!(
            "{{\"event\":\"backend\",\"requested\":\"{}\",\"resolved\":\"{}\",\"available\":{},\"training\":{},\"evaluation\":{},\"generation\":{},\"device_resident_model\":{},\"batched_documents\":{},\"cuda_devices\":{},\"execution_scope\":\"{}\"}}",
            requested,
            resolved,
            capabilities.available,
            capabilities.training,
            capabilities.evaluation,
            capabilities.generation,
            capabilities.device_resident_model,
            capabilities.batched_documents,
            gpu_devices,
            execution_scope,
        );
    } else {
        println!("requested backend: {requested}");
        println!("resolved backend: {resolved}");
        println!("available: {}", capabilities.available);
        println!("training: {}", capabilities.training);
        println!("evaluation: {}", capabilities.evaluation);
        println!("generation: {}", capabilities.generation);
        println!("cuda devices: {gpu_devices}");
        println!("execution scope: {execution_scope}");
        println!(
            "fully device-resident model: {}",
            capabilities.device_resident_model
        );
        println!(
            "batched-document executor: {}",
            capabilities.batched_documents
        );
    }

    if let Some(model_path) = arguments.value("model") {
        let model = load_model(model_path)?;
        if requested == BackendKind::Gpu {
            // Constructing the runtime compiles the NVRTC kernels, loads them through
            // the CUDA Driver API, allocates the complete device-resident model and
            // recurrent/eligibility runtime state, and uploads the checkpoint. A
            // successful probe therefore exercises the actual training executor.
            let mut gpu_runtime = BackendRuntime::new(model.clone(), BackendKind::Gpu)?;
            let mut cpu_runtime = BackendRuntime::new(model.clone(), BackendKind::Cpu)?;
            gpu_runtime.begin_document()?;
            cpu_runtime.begin_document()?;
            let probe = b"Once upon a time";
            let mut max_probability_delta = 0.0f32;
            for (index, byte) in probe.iter().copied().enumerate() {
                let input = if index == 0 {
                    BEGIN_DOCUMENT
                } else {
                    u32::from(probe[index - 1])
                };
                let target = Some(u32::from(byte));
                cpu_runtime.step(input, target, Permission::Frozen)?;
                gpu_runtime.step(input, target, Permission::Frozen)?;
                for (cpu, gpu) in cpu_runtime
                    .probabilities()
                    .iter()
                    .zip(gpu_runtime.probabilities())
                {
                    max_probability_delta = max_probability_delta.max((cpu - gpu).abs());
                }
            }
            cpu_runtime.finish_document()?;
            gpu_runtime.finish_document()?;
            if !max_probability_delta.is_finite() || max_probability_delta > 5.0e-4 {
                return Err(LeoError::internal(format!(
                    "CUDA forward parity failed: max probability delta {max_probability_delta}"
                )));
            }

            cpu_runtime.begin_document()?;
            gpu_runtime.begin_document()?;
            let mut max_training_probability_delta = 0.0f32;
            for (index, byte) in probe.iter().copied().enumerate() {
                let input = if index == 0 {
                    BEGIN_DOCUMENT
                } else {
                    u32::from(probe[index - 1])
                };
                let target = Some(u32::from(byte));
                cpu_runtime.step(input, target, Permission::Training)?;
                gpu_runtime.step(input, target, Permission::Training)?;
                for (cpu, gpu) in cpu_runtime
                    .probabilities()
                    .iter()
                    .zip(gpu_runtime.probabilities())
                {
                    max_training_probability_delta =
                        max_training_probability_delta.max((cpu - gpu).abs());
                }
            }
            cpu_runtime.finish_document()?;
            gpu_runtime.finish_document()?;
            if !max_training_probability_delta.is_finite()
                || max_training_probability_delta > 2.0e-3
            {
                return Err(LeoError::internal(format!(
                    "CUDA training parity failed: max probability delta {max_training_probability_delta}"
                )));
            }

            // The old probe stopped at one-step CPU/GPU parity and therefore
            // never exercised the production multi-story fast path. Run the
            // same logical batch through CPU reference and CUDA twice: once
            // without replay to isolate the batch-end mean barrier, then with
            // the locked v1 30% replay policy to cover replay selection/prefix
            // reconstruction and the resulting canonical model.
            let batch_without_replay = run_story_batch_conformance(&model, 0.0)?;
            let batch_with_replay = run_story_batch_conformance(&model, 0.30)?;

            if arguments.flag("json") {
                for result in [&batch_without_replay, &batch_with_replay] {
                    println!(
                        "{{\"event\":\"cuda_story_batch_conformance\",\"replay_fraction\":{},\"workers\":4,\"max_parameter_delta\":{},\"mean_loss_delta\":{},\"replay_segments\":{},\"replay_steps\":{},\"replay_prefix_steps\":{},\"ready\":true}}",
                        result.replay_fraction,
                        result.max_parameter_delta,
                        result.mean_loss_delta,
                        result.replay_segments,
                        result.replay_steps,
                        result.replay_prefix_steps,
                    );
                }
                println!(
                    "{{\"event\":\"cuda_kernel_probe\",\"resolved\":\"{}\",\"ready\":true,\"frozen_probe_steps\":{},\"frozen_max_probability_delta\":{},\"training_probe_steps\":{},\"training_max_probability_delta\":{}}}",
                    gpu_runtime.resolved_backend(),
                    probe.len(),
                    max_probability_delta,
                    probe.len(),
                    max_training_probability_delta,
                );
            } else {
                println!("CUDA kernel probe: ready");
                println!("CPU/GPU frozen max probability delta: {max_probability_delta}");
                println!(
                    "CPU/GPU training max probability delta: {max_training_probability_delta}"
                );
            }
        }
        let layout = DeviceModelLayout::for_model(&model);
        if arguments.flag("json") {
            println!(
                "{{\"event\":\"backend_model_layout\",\"buffers\":{},\"total_bytes\":{},\"read_only_bytes\":{},\"read_write_bytes\":{}}}",
                layout.buffers.len(),
                layout.total_bytes,
                layout.read_only_bytes,
                layout.read_write_bytes,
            );
        } else {
            println!("device buffers: {}", layout.buffers.len());
            println!("device bytes: {}", layout.total_bytes);
            println!("read-only bytes: {}", layout.read_only_bytes);
            println!("read-write bytes: {}", layout.read_write_bytes);
        }
    }
    Ok(())
}

fn command_checkpoint(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let mut model = load_model(model_path)?;
    commit_model(model_path, &mut model)?;
    println!(
        "{{\"event\":\"checkpoint\",\"generation\":{},\"checkpoint_sha256\":\"{}\"}}",
        model.generation,
        checkpoint_hash(model_path)?,
    );
    Ok(())
}

fn command_rollback(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let generation = arguments.required_u64("generation")?;
    let model = rollback_model(model_path, generation)?;
    println!(
        "{{\"event\":\"rollback\",\"generation\":{},\"checkpoint_sha256\":\"{}\"}}",
        model.generation,
        checkpoint_hash(model_path)?,
    );
    Ok(())
}

fn command_benchmark(arguments: &Arguments) -> LeoResult<()> {
    let model_path = arguments.required("model")?;
    let bytes_path = arguments.required("bytes")?;
    let index_path = arguments.required("index")?;
    let requested_stories = arguments.optional_usize("stories")?;
    let requested_backend = backend_from_arguments(arguments)?;

    if arguments.flag("train") {
        let mut runtime = BackendRuntime::new(load_model(model_path)?, requested_backend)?;
        let workers = arguments.optional_usize("workers")?.unwrap_or_else(|| {
            if runtime.resolved_backend() == BackendKind::Gpu {
                GPU_REFERENCE_WORKERS
            } else {
                1
            }
        });
        if workers == 0 {
            return Err(LeoError::internal(
                "training benchmark requires at least one worker",
            ));
        }
        let dataset = open_dataset(bytes_path, index_path, "training benchmark")?;
        let story_limit = requested_stories
            .unwrap_or(dataset.len())
            .min(dataset.len());
        let effective_workers = workers.min(story_limit.max(1));
        let multi_gpu_devices =
            configured_multi_gpu_devices(runtime.resolved_backend(), effective_workers)?;
        let mut training_engine = TrainingEngine::new(runtime.model(), multi_gpu_devices)?;
        let max_input_bytes = arguments.optional_u64("max-bytes")?;
        if max_input_bytes == Some(0) {
            return Err(LeoError::internal(
                "training benchmark --max-bytes must be positive",
            ));
        }
        let started = Instant::now();
        let mut position = 0usize;
        let mut input_bytes = 0u64;
        let mut activity = ActivityDiagnostics::default();
        let mut weighted_loss = 0.0f64;
        let mut targets = 0u64;
        let mut replay_segments = 0usize;
        let mut replay_steps = 0u64;
        let mut replay_prefix_steps = 0u64;
        let mut replay_seconds = 0.0f64;
        let mut replay_sync_seconds = 0.0f64;
        let story_order: Vec<usize> = (0..story_limit).collect();
        let prefetcher = StoryBatchPrefetcher::spawn(&dataset, story_order, "training benchmark")?;
        if position < story_limit {
            let remaining_bytes = max_input_bytes.map(|limit| limit.saturating_sub(input_bytes));
            prefetcher.request(position, workers, remaining_bytes)?;
        }
        while position < story_limit {
            let (stories, next_position, batch_input_bytes) = prefetcher.receive()?;
            if stories.is_empty() {
                break;
            }
            let next_input_bytes = input_bytes.saturating_add(batch_input_bytes);
            if next_position < story_limit {
                let remaining_bytes =
                    max_input_bytes.map(|limit| limit.saturating_sub(next_input_bytes));
                prefetcher.request(next_position, workers, remaining_bytes)?;
            }
            input_bytes = next_input_bytes;
            let report =
                training_engine.train_batch(&mut runtime, stories, Permission::Training)?;
            weighted_loss += report.mean_loss * report.targets as f64;
            targets = targets.saturating_add(report.targets as u64);
            replay_segments = replay_segments.saturating_add(report.replay_segments);
            replay_steps = replay_steps.saturating_add(report.replay_steps);
            replay_prefix_steps = replay_prefix_steps.saturating_add(report.replay_prefix_steps);
            replay_seconds += report.replay_seconds;
            replay_sync_seconds += report.replay_sync_seconds;
            activity.add(report.activity);
            position = next_position;
            if max_input_bytes.is_some_and(|limit| input_bytes >= limit) {
                break;
            }
        }
        if input_bytes == 0 {
            return Err(LeoError::internal(
                "training benchmark byte limit does not include one complete story",
            ));
        }
        let elapsed = started.elapsed().as_secs_f64();
        runtime.synchronize_model()?;
        let training_state_sha256 = training_state_hash(runtime.model());
        let projected_input_bytes = 2_000_000_000f64;
        let projected_seconds = elapsed / input_bytes.max(1) as f64 * projected_input_bytes;
        println!(
            "{{\"event\":\"training_benchmark\",\"model\":\"{}\",\"neurons\":{},\"fixed_synapses\":{},\"context_slots\":{},\"context_embedding_dim\":{},\"workers\":{},\"gpu_devices\":{},\"stories\":{},\"input_bytes\":{},\"training_steps\":{},\"base_training_targets\":{},\"replay_fraction\":{},\"replay_segments\":{},\"replay_steps\":{},\"replay_prefix_steps\":{},\"replay_execution_steps\":{},\"replay_step_fraction\":{},\"replay_prefix_step_fraction\":{},\"replay_seconds\":{},\"replay_sync_seconds\":{},\"replay_wall_fraction\":{},\"serial_replay_speedup_ceiling\":{},\"execution_steps_with_prefix\":{},\"seconds\":{},\"input_bytes_per_second\":{},\"steps_per_second\":{},\"execution_steps_per_second\":{},\"mean_loss\":{},\"bits_per_byte\":{},\"active_fraction\":{},\"recurrent_events_per_step\":{},\"context_cells_per_step\":{},\"context_probes_per_step\":{},\"output_madds_per_step\":{},\"training_state_sha256\":\"{}\",\"projected_seconds_1gb_2_epochs\":{},\"projected_days_1gb_2_epochs\":{}}}",
            json_escape(&runtime.model().config.model.name),
            runtime.model().neuron_count(),
            runtime.model().recurrent.weight.len(),
            runtime.model().context.keys.len(),
            runtime.model().config.context.embedding_dim,
            effective_workers,
            multi_gpu_devices,
            position,
            input_bytes,
            activity.steps,
            targets,
            runtime.model().config.replay.fraction,
            replay_segments,
            replay_steps,
            replay_prefix_steps,
            replay_steps.saturating_add(replay_prefix_steps),
            replay_steps as f64 / activity.steps.max(1) as f64,
            replay_prefix_steps as f64 / targets.max(1) as f64,
            replay_seconds,
            replay_sync_seconds,
            replay_seconds / elapsed.max(1.0e-9),
            if replay_seconds > 0.0 { elapsed / replay_seconds } else { 0.0 },
            activity.steps.saturating_add(replay_prefix_steps),
            elapsed,
            input_bytes as f64 / elapsed.max(1.0e-9),
            activity.steps as f64 / elapsed.max(1.0e-9),
            activity.steps.saturating_add(replay_prefix_steps) as f64 / elapsed.max(1.0e-9),
            weighted_loss / targets.max(1) as f64,
            weighted_loss / targets.max(1) as f64 / std::f64::consts::LN_2,
            activity.mean_active(runtime.model().neuron_count()),
            activity.events_per_step(),
            activity.context_cells_per_step(),
            activity.context_probes_per_step(),
            activity.output_madds_per_step(),
            training_state_sha256,
            projected_seconds,
            projected_seconds / 86_400.0,
        );
        return Ok(());
    }

    let model = load_model(model_path)?;
    let benchmark_dataset = open_dataset(bytes_path, index_path, "benchmark")?;
    let started = Instant::now();
    let metrics = evaluate_model(
        model,
        &benchmark_dataset,
        Some(requested_stories.unwrap_or(100)),
        requested_backend,
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{{\"event\":\"benchmark\",\"stories\":{},\"bytes\":{},\"seconds\":{},\"bytes_per_second\":{},\"recurrent_events_per_byte\":{},\"context_cells_per_byte\":{},\"context_probes_per_byte\":{},\"context_use_fraction\":{},\"output_madds_per_byte\":{},\"active_fraction\":{}}}",
        metrics.processed_stories,
        metrics.processed_bytes,
        elapsed,
        metrics.processed_bytes as f64 / elapsed.max(1.0e-9),
        metrics.activity.emitted_events as f64 / metrics.processed_bytes.max(1) as f64,
        metrics.activity.context_cells_per_byte(metrics.processed_bytes),
        metrics.activity.context_probes_per_byte(metrics.processed_bytes),
        metrics.activity.context_use_fraction(),
        metrics.activity.output_madds_per_byte(metrics.processed_bytes),
        metrics.activity.mean_active(metrics.neuron_count),
    );
    Ok(())
}

const MIN_GENERATED_BYTES_FOR_END: usize = 16;
const REPETITION_WINDOW_BYTES: usize = 96;
const REPEATED_TRIGRAM_PENALTY: f32 = 0.25;
const REPEATED_FOUR_GRAM_PENALTY: f32 = 0.05;
const IMMEDIATE_REPEAT_PENALTY: f32 = 0.75;
const REPEATED_RUN_PENALTY: f32 = 0.05;

#[derive(Debug, Clone, Copy)]
struct GenerationOptions {
    max_bytes: usize,
    temperature: f32,
    seed: u64,
}

#[derive(Debug, Clone)]
struct GenerationResult {
    text: String,
    bytes: Vec<u8>,
    natural_end: bool,
    repetition_rate: f64,
    printable_ratio: f64,
}

fn generate_from_prefix(
    runtime: &mut BackendRuntime,
    prefix: &[u8],
    options: GenerationOptions,
) -> LeoResult<GenerationResult> {
    std::str::from_utf8(prefix)
        .map_err(|error| LeoError::internal(format!("prompt is not valid UTF-8: {error}")))?;
    prime_document_prefix(runtime, prefix)?;
    let mut generator = SplitMix64::new(options.seed);
    let mut utf8 = Utf8State::default();
    let mut output = Vec::with_capacity(options.max_bytes);
    let mut generation_history = prefix.to_vec();
    let mut natural_end = false;

    for _ in 0..options.max_bytes {
        let symbol = select_output(
            runtime.probabilities(),
            utf8,
            options.temperature,
            &mut generator,
            &generation_history,
            output.len(),
        );
        if symbol == END_DOCUMENT {
            natural_end = true;
            break;
        }
        let byte = symbol as u8;
        if !utf8.push(byte) {
            return Err(LeoError::internal("UTF-8 guard selected an invalid byte"));
        }
        output.push(byte);
        generation_history.push(byte);
        runtime.step(symbol, None, Permission::Frozen)?;
        if generation_is_stuck(&output) {
            break;
        }
    }

    runtime.finish_document()?;
    trim_incomplete_utf8(&mut output);
    let text = String::from_utf8(output.clone())
        .map_err(|error| LeoError::internal(format!("generated invalid UTF-8: {error}")))?;
    Ok(GenerationResult {
        repetition_rate: repeated_ngram_rate(&output, 4),
        printable_ratio: printable_ratio(&output),
        text,
        bytes: output,
        natural_end,
    })
}

fn prime_document_prefix(runtime: &mut BackendRuntime, prefix: &[u8]) -> LeoResult<()> {
    runtime.begin_document()?;
    runtime.step(BEGIN_DOCUMENT, None, Permission::Frozen)?;
    for byte in prefix {
        runtime.step(*byte as u32, None, Permission::Frozen)?;
    }
    Ok(())
}

fn generation_is_stuck(output: &[u8]) -> bool {
    if output.len() >= 16
        && output[output.len() - 16..]
            .iter()
            .all(|byte| *byte == output[output.len() - 1])
    {
        return true;
    }
    if output.len() < 32 {
        return false;
    }
    let tail = &output[output.len() - 16..];
    (1..=8).any(|period| tail[..8] == tail[period..period + 8])
}

fn select_output(
    probabilities: &[f32],
    utf8: Utf8State,
    temperature: f32,
    generator: &mut SplitMix64,
    history: &[u8],
    generated_bytes: usize,
) -> u32 {
    if temperature <= 0.0 {
        let index = probabilities
            .iter()
            .copied()
            .enumerate()
            .take(OUTPUT_CLASSES)
            .filter(|(index, _)| allowed_output(*index, utf8, generated_bytes))
            .map(|(index, probability)| {
                (
                    index,
                    adjusted_generation_weight(index, probability, history),
                )
            })
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(index, _)| index)
            .unwrap_or(END_DOCUMENT_OUTPUT_INDEX);
        return output_index_to_symbol(index).expect("valid output index");
    }

    let inverse_temperature = 1.0 / temperature.max(0.05);
    let mut weights = [0.0f32; OUTPUT_CLASSES];
    let mut sum = 0.0f32;
    for (index, probability) in probabilities
        .iter()
        .copied()
        .enumerate()
        .take(OUTPUT_CLASSES)
    {
        if allowed_output(index, utf8, generated_bytes) {
            let adjusted = adjusted_generation_weight(index, probability, history);
            let weight = adjusted.max(1.0e-20).powf(inverse_temperature);
            weights[index] = weight;
            sum += weight;
        }
    }
    if !sum.is_finite() || sum <= 0.0 {
        return END_DOCUMENT;
    }
    let mut draw = generator.next_f32() * sum;
    for (index, weight) in weights.into_iter().enumerate() {
        draw -= weight;
        if draw <= 0.0 {
            return output_index_to_symbol(index).expect("valid sampled output index");
        }
    }
    END_DOCUMENT
}

fn adjusted_generation_weight(index: usize, probability: f32, history: &[u8]) -> f32 {
    if index == END_DOCUMENT_OUTPUT_INDEX {
        return probability;
    }
    let candidate = index as u8;
    let recent_start = history.len().saturating_sub(REPETITION_WINDOW_BYTES);
    let recent = &history[recent_start..];
    let mut adjustment = 1.0;
    if recent.last().copied() == Some(candidate) {
        adjustment *= IMMEDIATE_REPEAT_PENALTY;
    }
    if recent.len() >= 3
        && recent[recent.len() - 3..]
            .iter()
            .all(|byte| *byte == candidate)
    {
        adjustment *= REPEATED_RUN_PENALTY;
    }
    if completes_repeated_ngram(recent, candidate, 4) {
        adjustment *= REPEATED_FOUR_GRAM_PENALTY;
    } else if completes_repeated_ngram(recent, candidate, 3) {
        adjustment *= REPEATED_TRIGRAM_PENALTY;
    }
    probability * adjustment
}

fn completes_repeated_ngram(history: &[u8], candidate: u8, order: usize) -> bool {
    if order < 2 || history.len() < order {
        return false;
    }
    let prefix_length = order - 1;
    let suffix = &history[history.len() - prefix_length..];
    let last_existing_start = history.len() - order;
    (0..=last_existing_start).any(|start| {
        &history[start..start + prefix_length] == suffix
            && history[start + prefix_length] == candidate
    })
}

fn allowed_output(index: usize, utf8: Utf8State, generated_bytes: usize) -> bool {
    if index == END_DOCUMENT_OUTPUT_INDEX {
        return utf8.is_complete() && generated_bytes >= MIN_GENERATED_BYTES_FOR_END;
    }
    index < 256 && utf8.can_accept(index as u8)
}

#[derive(Debug, Default)]
struct FreeRunningMetrics {
    stories: u64,
    reference_bytes: u64,
    generated_bytes: u64,
    correct_bytes: u64,
    matching_prefix_fraction_sum: f64,
    exact_completions: u64,
    natural_ends: u64,
    repetition_rate_sum: f64,
}

impl FreeRunningMetrics {
    fn byte_accuracy(&self) -> f64 {
        self.correct_bytes as f64 / self.reference_bytes.max(1) as f64
    }

    fn mean_matching_prefix_fraction(&self) -> f64 {
        self.matching_prefix_fraction_sum / self.stories.max(1) as f64
    }

    fn natural_end_fraction(&self) -> f64 {
        self.natural_ends as f64 / self.stories.max(1) as f64
    }

    fn mean_repetition_rate(&self) -> f64 {
        self.repetition_rate_sum / self.stories.max(1) as f64
    }
}

fn evaluate_free_running(
    model: Model,
    bytes_path: &str,
    index_path: &str,
    prediction_story_limit: Option<usize>,
    generation_story_limit: usize,
    max_generation_bytes: usize,
    backend: BackendKind,
) -> LeoResult<FreeRunningMetrics> {
    let mut dataset = open_dataset(bytes_path, index_path, "free-running evaluation")?;
    let prediction_limit = prediction_story_limit
        .unwrap_or(dataset.len())
        .min(dataset.len());
    let limit = prediction_limit.min(generation_story_limit);
    let mut result = FreeRunningMetrics::default();
    let mut runtime = BackendRuntime::new(model.clone(), backend)?;

    for story_index in 0..limit {
        let story = dataset
            .story(story_index)
            .map_err(|error| LeoError::dataset(format!("dataset error: {error}")))?;
        if story.len() < 2 {
            continue;
        }
        let prefix_length = (story.len() / 4).clamp(1, 64).min(story.len() - 1);
        let reference = &story[prefix_length..];
        let generation = generate_from_prefix(
            &mut runtime,
            &story[..prefix_length],
            GenerationOptions {
                max_bytes: max_generation_bytes.min(reference.len().saturating_add(32)),
                temperature: 0.0,
                seed: model.config.model.seed ^ story_index as u64,
            },
        )?;
        let correct = generation
            .bytes
            .iter()
            .zip(reference)
            .filter(|(generated, expected)| generated == expected)
            .count();
        let matching_prefix = generation
            .bytes
            .iter()
            .zip(reference)
            .take_while(|(generated, expected)| generated == expected)
            .count();

        result.stories = result.stories.saturating_add(1);
        result.reference_bytes = result
            .reference_bytes
            .saturating_add(reference.len() as u64);
        result.generated_bytes = result
            .generated_bytes
            .saturating_add(generation.bytes.len() as u64);
        result.correct_bytes = result.correct_bytes.saturating_add(correct as u64);
        result.matching_prefix_fraction_sum +=
            matching_prefix as f64 / reference.len().max(1) as f64;
        result.repetition_rate_sum += generation.repetition_rate;
        if generation.natural_end {
            result.natural_ends = result.natural_ends.saturating_add(1);
        }
        if generation.bytes.as_slice() == reference && generation.natural_end {
            result.exact_completions = result.exact_completions.saturating_add(1);
        }
    }
    Ok(result)
}

fn print_generation_evaluation(event: &str, prompt: &str, generation: &GenerationResult) {
    println!(
        "{{\"event\":\"{}\",\"prompt\":\"{}\",\"continuation\":\"{}\",\"generated_bytes\":{},\"natural_end\":{},\"repetition_rate\":{},\"printable_ratio\":{},\"valid_utf8\":true}}",
        event,
        json_escape(prompt),
        json_escape(&generation.text),
        generation.bytes.len(),
        generation.natural_end,
        generation.repetition_rate,
        generation.printable_ratio,
    );
}

fn backend_from_arguments(arguments: &Arguments) -> LeoResult<BackendKind> {
    let value = arguments
        .value("backend")
        .map(str::to_owned)
        .or_else(|| env::var("LEO_BACKEND").ok())
        .unwrap_or_else(|| "auto".to_owned());
    value.parse()
}

fn paired_paths<'a>(
    arguments: &'a Arguments,
    bytes_key: &str,
    index_key: &str,
) -> LeoResult<Option<(&'a str, &'a str)>> {
    match (arguments.value(bytes_key), arguments.value(index_key)) {
        (Some(bytes), Some(index)) => Ok(Some((bytes, index))),
        (None, None) => Ok(None),
        _ => Err(LeoError::internal(format!(
            "--{bytes_key} and --{index_key} must be provided together"
        ))),
    }
}

fn model_state_hash(model: &Model) -> ArtifactDigest {
    let mut hash = Sha256::new();

    for value in model
        .recurrent
        .weight
        .iter()
        .chain(model.input.weights.iter())
        .chain(model.output.weights.iter())
        .chain(model.output.bias.iter())
        .chain(model.context.embeddings.iter())
        .chain(model.context.output_weights.iter())
        .chain(model.neurons.threshold.iter())
        .chain(model.neurons.excitability.iter())
    {
        hash.update(&value.to_bits().to_le_bytes());
    }
    for value in &model.recurrent.target_neuron {
        hash.update(&value.to_le_bytes());
    }
    hash.update(&model.recurrent.target_branch);
    hash.update(&model.recurrent.delay);
    for value in &model.input.targets {
        hash.update(&value.to_le_bytes());
    }
    hash.update(&model.input.branches);
    hash.update(&model.neurons.neuron_type);
    for value in &model.context.keys {
        hash.update(&value.to_le_bytes());
    }
    for value in &model.context.observations {
        hash.update(&value.to_le_bytes());
    }
    hash.finalize()
}

fn training_state_hash(model: &Model) -> ArtifactDigest {
    let mut hash = Sha256::new();
    hash.update(model_state_hash(model).as_bytes());
    hash.update(&model.generation.to_le_bytes());
    hash.update(&model.parameter_revision.to_le_bytes());
    hash.update(&model.statistics.processed_bytes.to_le_bytes());
    hash.update(&model.statistics.processed_stories.to_le_bytes());
    hash.update(&model.statistics.training_loss_sum.to_bits().to_le_bytes());
    hash.update(&model.statistics.training_targets.to_le_bytes());
    hash.update(&model.statistics.active_neurons_sum.to_le_bytes());
    hash.update(&model.statistics.active_neurons_peak.to_le_bytes());
    hash.update(&model.statistics.synaptic_events.to_le_bytes());
    hash.update(&model.statistics.numerical_rejections.to_le_bytes());
    hash.update(&model.statistics.persistent_ticks.to_le_bytes());
    hash.finalize()
}

fn mean_f32(values: &[f32]) -> f64 {
    values.iter().map(|value| *value as f64).sum::<f64>() / values.len().max(1) as f64
}

fn mean_abs_f32(values: impl Iterator<Item = f32>) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for value in values {
        sum += value.abs() as f64;
        count = count.saturating_add(1);
    }
    sum / count.max(1) as f64
}

fn repeated_ngram_rate(bytes: &[u8], width: usize) -> f64 {
    if width == 0 || bytes.len() < width {
        return 0.0;
    }
    let mut counts = HashMap::<Vec<u8>, usize>::new();
    let mut repeated = 0usize;
    for window in bytes.windows(width) {
        let count = counts.entry(window.to_vec()).or_insert(0);
        if *count > 0 {
            repeated = repeated.saturating_add(1);
        }
        *count = count.saturating_add(1);
    }
    repeated as f64 / (bytes.len() - width + 1) as f64
}

fn printable_ratio(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 1.0;
    }
    bytes
        .iter()
        .filter(|byte| byte.is_ascii_graphic() || byte.is_ascii_whitespace() || **byte >= 0x80)
        .count() as f64
        / bytes.len() as f64
}

fn trim_incomplete_utf8(bytes: &mut Vec<u8>) {
    if let Err(error) = std::str::from_utf8(bytes) {
        bytes.truncate(error.valid_up_to());
    }
}

fn permission_name(permission: Permission) -> &'static str {
    match permission {
        Permission::Frozen => "frozen",
        Permission::Provisional => "provisional",
        Permission::Training => "training",
        Permission::Verified => "verified",
    }
}

fn json_escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output
}

fn exit_code(error: &LeoError) -> i32 {
    match error.kind() {
        LeoErrorKind::Usage | LeoErrorKind::Configuration => 2,
        LeoErrorKind::CheckpointCorrupt | LeoErrorKind::CheckpointIncompatible => 4,
        LeoErrorKind::Numerical => 5,
        LeoErrorKind::Backend | LeoErrorKind::Cuda => 6,
        LeoErrorKind::Dataset => 7,
        LeoErrorKind::Io => 8,
        LeoErrorKind::Internal => 1,
    }
}

fn is_help_command(command: &str) -> bool {
    matches!(command, "help" | "--help" | "-h")
}

fn print_help() {
    println!(
        "Leo — sparse recurrent byte-learning system\n\n\
Usage:\n\
  leo init --config <FILE> --output <MODEL.pscls>\n\
  leo train --model <MODEL.pscls> --train-index <FILE> --train-bytes <FILE> [--valid-index <FILE> --valid-bytes <FILE>] [--passes N] [--max-stories N] [--max-bytes N] [--workers N] [--backend auto|cpu|gpu] [--fresh-run]\n\
  leo eval --model <MODEL.pscls> --index <FILE> --bytes <FILE> [--stories N] [--train-index <FILE> --train-bytes <FILE> --train-stories N] [--generation-stories 20] [--prompt \"Once upon a time\"] [--backend auto|cpu|gpu]\n\
  leo prompt --model <MODEL.pscls> --text <STORY_PREFIX> [--max-bytes 1000] [--temperature 0.8] [--seed N] [--backend auto|cpu|gpu] [--json]\n\
  leo teach --model <MODEL.pscls> --permission provisional|training|verified --text <LESSON> [--backend auto|cpu|gpu]\n\
  leo inspect --model <MODEL.pscls> [--json]\n\
  leo checkpoint --model <MODEL.pscls>\n\
  leo rollback --model <MODEL.pscls> --generation <N>\n\
  leo benchmark --model <MODEL.pscls> --index <FILE> --bytes <FILE> [--stories 100] [--train --max-bytes N --workers N] [--backend auto|cpu|gpu]\n\
  leo backend [--backend auto|cpu|gpu] [--model <MODEL.pscls>] [--json]\n"
    );
}

#[derive(Clone, Copy)]
struct CommandSchema {
    values: &'static [&'static str],
    flags: &'static [&'static str],
}

fn command_schema(command: &str) -> Option<CommandSchema> {
    let schema = match command {
        "init" => CommandSchema {
            values: &["config", "output"],
            flags: &[],
        },
        "train" => CommandSchema {
            values: &[
                "model",
                "train-bytes",
                "train-index",
                "valid-bytes",
                "valid-index",
                "passes",
                "max-stories",
                "max-bytes",
                "workers",
                "validation-stories",
                "log-every",
                "backend",
            ],
            flags: &["fresh-run"],
        },
        "eval" => CommandSchema {
            values: &[
                "model",
                "bytes",
                "index",
                "valid-bytes",
                "valid-index",
                "stories",
                "train-bytes",
                "train-index",
                "train-stories",
                "generation-stories",
                "max-bytes",
                "prompt",
                "temperature",
                "seed",
                "backend",
            ],
            flags: &[],
        },
        "prompt" => CommandSchema {
            values: &[
                "model",
                "text",
                "max-bytes",
                "temperature",
                "seed",
                "backend",
            ],
            flags: &["json"],
        },
        "teach" => CommandSchema {
            values: &["model", "permission", "text", "backend"],
            flags: &[],
        },
        "inspect" => CommandSchema {
            values: &["model"],
            flags: &["json"],
        },
        "checkpoint" => CommandSchema {
            values: &["model"],
            flags: &[],
        },
        "rollback" => CommandSchema {
            values: &["model", "generation"],
            flags: &[],
        },
        "benchmark" => CommandSchema {
            values: &[
                "model",
                "index",
                "bytes",
                "stories",
                "max-bytes",
                "workers",
                "backend",
            ],
            flags: &["train"],
        },
        "backend" => CommandSchema {
            values: &["backend", "model"],
            flags: &["json"],
        },
        _ => return None,
    };
    Some(schema)
}

struct Arguments {
    values: BTreeMap<String, String>,
    flags: HashSet<String>,
}

impl Arguments {
    fn parse_for(command: &str, raw: Vec<String>) -> LeoResult<Self> {
        let schema = command_schema(command)
            .ok_or_else(|| LeoError::usage(format!("unknown command: {command}")))?;
        let mut values = BTreeMap::new();
        let mut flags = HashSet::new();
        let mut index = 0usize;
        while index < raw.len() {
            let item = &raw[index];
            if item == "-h" || item == "--help" {
                if !flags.insert("help".to_owned()) {
                    return Err(LeoError::usage("duplicate option --help"));
                }
                index += 1;
                continue;
            }
            if !item.starts_with("--") || item == "--" {
                return Err(LeoError::usage(format!(
                    "invalid option or positional argument: {item}"
                )));
            }
            if item.contains('=') {
                return Err(LeoError::usage(format!(
                    "invalid option syntax: {item}; use --name value"
                )));
            }
            let key = item.trim_start_matches("--");
            if schema.flags.contains(&key) {
                if values.contains_key(key) || !flags.insert(key.to_owned()) {
                    return Err(LeoError::usage(format!("duplicate option --{key}")));
                }
                index += 1;
                continue;
            }
            if !schema.values.contains(&key) {
                return Err(LeoError::usage(format!(
                    "unknown option for {command}: --{key}"
                )));
            }
            if flags.contains(key) || values.contains_key(key) {
                return Err(LeoError::usage(format!("duplicate option --{key}")));
            }
            let value = raw
                .get(index + 1)
                .ok_or_else(|| LeoError::usage(format!("missing value for --{key}")))?;
            if value.starts_with("--") {
                return Err(LeoError::usage(format!("missing value for --{key}")));
            }
            values.insert(key.to_owned(), value.clone());
            index += 2;
        }
        Ok(Self { values, flags })
    }

    fn value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn required(&self, key: &str) -> LeoResult<&str> {
        self.value(key)
            .ok_or_else(|| LeoError::usage(format!("missing required option --{key}")))
    }

    fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    fn optional_usize(&self, key: &str) -> LeoResult<Option<usize>> {
        self.value(key)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| LeoError::usage(format!("invalid value for --{key}: {value}")))
            })
            .transpose()
    }

    fn optional_u64(&self, key: &str) -> LeoResult<Option<u64>> {
        self.value(key)
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|_| LeoError::usage(format!("invalid value for --{key}: {value}")))
            })
            .transpose()
    }

    fn required_u64(&self, key: &str) -> LeoResult<u64> {
        self.optional_u64(key)?
            .ok_or_else(|| LeoError::usage(format!("missing required option --{key}")))
    }

    fn optional_f32(&self, key: &str) -> LeoResult<Option<f32>> {
        self.value(key)
            .map(|value| {
                value
                    .parse::<f32>()
                    .map_err(|_| LeoError::usage(format!("invalid value for --{key}: {value}")))
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::training::{ranges_overlap, select_replay_ranges};
    use super::{
        adjusted_generation_weight, advance_checkpoint_deadline, allowed_output,
        backend_from_arguments, is_help_command, prime_document_prefix, read_story_batch,
        shuffled_story_order, train_story_batch, Arguments, StoryBatchPrefetcher,
        TrainingResumeState,
    };
    use leo_core::symbols::{BEGIN_DOCUMENT, END_DOCUMENT_OUTPUT_INDEX};
    use leo_core::utf8::Utf8State;
    use leo_core::{BackendKind, BackendRuntime, Config, Model, Permission, Runtime};
    use leo_data::{write_index, PreparedDataset, StoryIndex};
    use std::fs;
    use std::time::Duration;

    #[test]
    fn periodic_checkpoint_deadline_advances_without_catchup_bursts() {
        let interval = Duration::from_secs(300);
        assert_eq!(
            advance_checkpoint_deadline(interval, Duration::from_secs(300), interval),
            Duration::from_secs(600),
        );
        assert_eq!(
            advance_checkpoint_deadline(interval, Duration::from_secs(901), interval),
            Duration::from_secs(1200),
        );
    }

    #[test]
    fn help_aliases_are_recognized() {
        assert!(is_help_command("help"));
        assert!(is_help_command("--help"));
        assert!(is_help_command("-h"));
    }

    #[test]
    fn command_schema_accepts_flags_without_values() {
        let arguments = Arguments::parse_for(
            "benchmark",
            vec!["--train".to_owned(), "--workers".to_owned(), "4".to_owned()],
        )
        .unwrap();
        assert!(arguments.flag("train"));
        assert_eq!(arguments.optional_usize("workers").unwrap(), Some(4));
    }

    #[test]
    fn command_schema_rejects_unknown_and_duplicate_options() {
        assert!(
            Arguments::parse_for("train", vec!["--wrokers".to_owned(), "4".to_owned()],).is_err()
        );
        assert!(Arguments::parse_for(
            "train",
            vec![
                "--workers".to_owned(),
                "4".to_owned(),
                "--workers".to_owned(),
                "8".to_owned(),
            ],
        )
        .is_err());
    }

    #[test]
    fn training_resume_state_round_trip_preserves_exact_cursor() {
        let state = TrainingResumeState {
            model_generation: 85,
            model_parameter_revision: 44_944_803,
            model_processed_stories: 50_176,
            train_dataset_id: leo_core::ArtifactDigest::from_array([11; 32]),
            dataset_len: 2_119_489,
            story_limit: 40_000,
            passes: 2,
            workers: 90,
            backend: BackendKind::Gpu,
            synchronization: "gpu_shared_wavefront_mean".to_owned(),
            max_training_bytes: None,
            validation_dataset_id: Some(leo_core::ArtifactDigest::from_array([13; 32])),
            validation_story_limit: Some(3_000),
            next_pass: 1,
            next_position: 90,
            presentations: 40_090,
            input_bytes_seen: 35_000_000,
            last_validation_processed_bytes: 44_000_000,
            patience_best: 1.617466673931383,
            stale_checks: 0,
            best_validation_loss: Some(1.617466673931383),
            best_parameter_revision: Some(44_000_000),
            best_checkpoint_digest: Some(leo_core::ArtifactDigest::from_array([17; 32])),
            finalized: false,
        };
        let decoded = TrainingResumeState::decode(&state.encode()).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn backend_argument_uses_the_shared_backend_parser() {
        let arguments =
            Arguments::parse_for("backend", vec!["--backend".to_owned(), "gpu".to_owned()])
                .unwrap();
        assert_eq!(
            backend_from_arguments(&arguments).unwrap(),
            BackendKind::Gpu
        );
    }

    #[test]
    fn story_batches_honor_raw_byte_limits_without_splitting_records() {
        let base = std::env::temp_dir().join(format!("leo-cli-byte-batch-{}", std::process::id()));
        let bytes = base.with_extension("bytes");
        let index = base.with_extension("idx");
        fs::write(&bytes, b"onetwothree").unwrap();
        write_index(
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
                StoryIndex {
                    offset: 6,
                    length: 5,
                    split_flags: 0,
                },
            ],
            leo_core::ArtifactDigest::ZERO,
        )
        .unwrap();
        let mut dataset = PreparedDataset::open(&bytes, &index).unwrap();
        let order = [0usize, 1, 2];
        let (stories, next_position, input_bytes) =
            read_story_batch(&mut dataset, &order, 0, 3, Some(6), "test").unwrap();
        assert_eq!(stories, vec![b"one".to_vec(), b"two".to_vec()]);
        assert_eq!(next_position, 2);
        assert_eq!(input_bytes, 6);

        let (stories, next_position, input_bytes) =
            read_story_batch(&mut dataset, &order, 2, 1, Some(4), "test").unwrap();
        assert!(stories.is_empty());
        assert_eq!(next_position, 2);
        assert_eq!(input_bytes, 0);

        fs::remove_file(bytes).unwrap();
        fs::remove_file(index).unwrap();
    }

    #[test]
    fn prefetched_story_batch_matches_synchronous_reader() {
        let base =
            std::env::temp_dir().join(format!("leo-cli-prefetch-batch-{}", std::process::id()));
        let bytes = base.with_extension("bytes");
        let index = base.with_extension("idx");
        fs::write(&bytes, b"onetwothree").unwrap();
        write_index(
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
                StoryIndex {
                    offset: 6,
                    length: 5,
                    split_flags: 0,
                },
            ],
            leo_core::ArtifactDigest::ZERO,
        )
        .unwrap();

        let mut synchronous = PreparedDataset::open(&bytes, &index).unwrap();
        let order = vec![2usize, 0, 1];
        let expected = read_story_batch(&mut synchronous, &order, 0, 3, Some(8), "test").unwrap();

        let dataset = PreparedDataset::open(&bytes, &index).unwrap();
        let prefetcher = StoryBatchPrefetcher::spawn(&dataset, order, "test").unwrap();
        prefetcher.request(0, 3, Some(8)).unwrap();
        let actual = prefetcher.receive().unwrap();
        assert_eq!(actual, expected);
        drop(prefetcher);
        drop(dataset);
        drop(synchronous);

        fs::remove_file(bytes).unwrap();
        fs::remove_file(index).unwrap();
    }

    #[test]
    fn synchronous_story_batch_keeps_one_canonical_model() {
        let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
        let model = Model::initialize(config).unwrap();
        let mut runtime = BackendRuntime::new(model, BackendKind::Cpu).unwrap();
        let report = train_story_batch(
            &mut runtime,
            vec![b"one".to_vec(), b"two".to_vec()],
            Permission::Training,
        )
        .unwrap();
        assert_eq!(report.stories, 2);
        assert_eq!(report.merge.workers, 2);
        assert_eq!(runtime.model().statistics.processed_stories, 2);
    }

    #[test]
    fn shuffle_is_reproducible_and_complete() {
        let first = shuffled_story_order(100, 1337, 0);
        let second = shuffled_story_order(100, 1337, 0);
        assert_eq!(first, second);
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn replay_selects_high_loss_non_overlapping_segments() {
        let losses = vec![0.1, 4.0, 0.2, 0.1, 0.2, 5.0, 0.1, 0.1];
        let ranges = select_replay_ranges(&losses, 0.50, 3);
        assert_eq!(ranges.len(), 2);
        assert!(!ranges_overlap(&ranges[0], &ranges[1]));
        assert!(ranges.iter().any(|range| range.contains(&1)));
        assert!(ranges.iter().any(|range| range.contains(&5)));
    }

    #[test]
    fn repetition_penalty_demotes_seen_four_grams() {
        let history = b"the the ";
        let repeated = adjusted_generation_weight(b't' as usize, 0.6, history);
        let fresh = adjusted_generation_weight(b'x' as usize, 0.4, history);
        assert!(repeated < fresh);
    }

    #[test]
    fn ordinary_frequent_bytes_are_not_globally_suppressed() {
        let history = b"a calm day with a cat and a hat ";
        let space = adjusted_generation_weight(b' ' as usize, 0.6, history);
        let rare = adjusted_generation_weight(b'x' as usize, 0.4, history);
        assert!(space > rare);
    }

    #[test]
    fn repeated_runs_are_strongly_penalized() {
        let history = b"soooo";
        let repeated = adjusted_generation_weight(b'o' as usize, 0.8, history);
        let fresh = adjusted_generation_weight(b'n' as usize, 0.2, history);
        assert!(repeated < fresh);
    }

    #[test]
    fn end_document_requires_a_minimum_continuation() {
        let utf8 = Utf8State::default();
        assert!(!allowed_output(END_DOCUMENT_OUTPUT_INDEX, utf8, 15));
        assert!(allowed_output(END_DOCUMENT_OUTPUT_INDEX, utf8, 16));
    }

    #[test]
    fn prompt_prefix_matches_training_aligned_state() {
        let config = Config::from_toml(include_str!("../../../configs/test.toml")).unwrap();
        let model = Model::initialize(config).unwrap();
        let prefix = b"Once upon a time";

        let mut expected = Runtime::new(model.clone()).unwrap();
        expected.begin_document();
        expected
            .step(BEGIN_DOCUMENT, Some(prefix[0] as u32), Permission::Frozen)
            .unwrap();
        for window in prefix.windows(2) {
            expected
                .step(window[0] as u32, Some(window[1] as u32), Permission::Frozen)
                .unwrap();
        }
        expected
            .step(*prefix.last().unwrap() as u32, None, Permission::Frozen)
            .unwrap();

        let mut actual = BackendRuntime::new(model, BackendKind::Cpu).unwrap();
        prime_document_prefix(&mut actual, prefix).unwrap();
        assert_eq!(actual.probabilities(), expected.probabilities());
    }
}
