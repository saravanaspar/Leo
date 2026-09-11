#!/usr/bin/env python3
"""Live 1-vs-N GPU scaling benchmark for Leo's exact data-parallel trainer."""

import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leo", default="target/release/leo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bytes", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--stories", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated physical CUDA device IDs; defaults to nvidia-smi discovery",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="rerun a GPU-count case until CUDA autotuning is no longer present",
    )
    parser.add_argument(
        "--counts",
        default=None,
        help="comma-separated GPU counts to test (for example 1,2); defaults to powers of two plus all visible devices",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="number of clean measured runs per GPU count; report min/median/max",
    )
    parser.add_argument(
        "--legacy-execution",
        action="store_true",
        help=(
            "disable persistent/grouped/device-batch-merge execution optimizations "
            "for an exact legacy A/B; inherited LEO_* execution switches are otherwise ignored"
        ),
    )
    return parser.parse_args()


def discover_devices(explicit):
    if explicit:
        devices = [item.strip() for item in explicit.split(",") if item.strip()]
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if visible == "-1":
            devices = []
        else:
            devices = [item.strip() for item in visible.split(",") if item.strip()]
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not devices:
        raise RuntimeError("no CUDA devices found")
    return devices


def counts_to_test(device_count):
    counts = [1]
    value = 2
    while value <= device_count:
        counts.append(value)
        value *= 2
    if counts[-1] != device_count:
        counts.append(device_count)
    return counts


def gpu_heartbeat(devices):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                ",".join(devices),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "gpu telemetry unavailable"
    if result.returncode != 0:
        return "gpu telemetry unavailable"
    rows = []
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) == 4:
            rows.append(
                f"gpu{fields[0]}={fields[1]}% {fields[2]}/{fields[3]} MiB"
            )
    return "; ".join(rows) if rows else "gpu telemetry unavailable"


def benchmark_environment(base_env, legacy_execution=False):
    env = base_env.copy()
    explicit = {
        "LEO_MULTI_GPU",
        "LEO_CUDA_DEVICE",
        "LEO_REPLAY_STREAMING",
        "LEO_MULTI_GPU_PARALLEL_REPLAY",
        "LEO_CUDA_REPLAY_COOPERATIVE",
        "LEO_CUDA_SHARED_PERSISTENT",
        "LEO_CUDA_SHARED_GROUPED",
        "LEO_CUDA_DEVICE_BATCH_MERGE",
        "LEO_CUDA_DEVICE_STORY_STEPS",
        "LEO_CUDA_DEVICE_STORY_POSTPROCESS",
        "LEO_CUDA_FULL_STEP_METRICS",
        "LEO_CUDA_REPLAY_BLOCKS",
        "LEO_CUDA_FROZEN_BLOCKS",
    }
    prefixes = (
        "LEO_CUDA_DEBUG",
        "LEO_CUDA_REPLAY_PROFILE",
        "LEO_CUDA_PHASE_PROFILE",
        "LEO_REPLAY_DEBUG",
    )
    for key in list(env):
        if key in explicit or key.startswith(prefixes):
            env.pop(key, None)

    if legacy_execution:
        env["LEO_CUDA_SHARED_PERSISTENT"] = "0"
        env["LEO_CUDA_SHARED_GROUPED"] = "0"
        env["LEO_CUDA_DEVICE_BATCH_MERGE"] = "0"
    return env


def run_case(args, devices, count):
    visible = devices[:count]
    env = benchmark_environment(os.environ, legacy_execution=args.legacy_execution)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    env["LEO_MULTI_GPU"] = "1" if count > 1 else "0"

    cmd = [
        str(args.leo),
        "benchmark",
        "--train",
        "--model", str(args.model),
        "--bytes", str(args.bytes),
        "--index", str(args.index),
        "--stories", str(args.stories),
        "--workers", str(args.workers),
        "--backend", "gpu",
    ]

    print("\n" + "=" * 80, flush=True)
    print(f"{count} GPU(s): CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 80, flush=True)

    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    benchmark = None
    multi_events = []
    tuning_events = []
    started = time.monotonic()
    last_heartbeat = started

    def consume(line):
        nonlocal benchmark
        print(line, end="", flush=True)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if event.get("event") == "training_benchmark":
            benchmark = event
        elif event.get("event") in {"multi_gpu_training", "multi_gpu_sparse_sync"}:
            multi_events.append(event)
        # cuda_profile is low-cadence production telemetry even after tuning
        # completes. Only autotune lifecycle events contaminate throughput.
        if event.get("event") in {
            "cuda_autotune_resumed",
            "cuda_autotune_complete",
        }:
            tuning_events.append(event)

    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        while True:
            # proc.stdout is the only registered stream. Read it directly so
            # static type checkers do not treat SelectorKey.fileobj as an
            # int/HasFileno union while preserving the selector-driven wait.
            for _ in selector.select(timeout=1.0):
                line = proc.stdout.readline()
                if line:
                    consume(line)
            now = time.monotonic()
            if now - last_heartbeat >= 10:
                print(
                    f"[scaling heartbeat] {now-started:.1f}s | {gpu_heartbeat(visible)}",
                    flush=True,
                )
                last_heartbeat = now
            if proc.poll() is not None:
                for line in proc.stdout:
                    consume(line)
                break
    finally:
        selector.close()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{count}-GPU benchmark failed with exit code {rc}")
    if benchmark is None:
        raise RuntimeError(f"{count}-GPU benchmark emitted no training_benchmark event")
    if int(benchmark.get("gpu_devices", 1)) != count:
        raise RuntimeError(
            f"requested {count} GPUs but Leo reported gpu_devices={benchmark.get('gpu_devices')}"
        )
    if not benchmark.get("training_state_sha256"):
        raise RuntimeError("training benchmark emitted no final training_state_sha256")
    if count > 1 and not any(
        event.get("exact_flat_story_mean") is True for event in multi_events
    ):
        raise RuntimeError("multi-GPU run did not report exact_flat_story_mean=true")
    return benchmark, tuning_events


def run_clean_case(args, devices, count):
    for attempt in range(1, args.max_attempts + 1):
        benchmark, tuning_events = run_case(args, devices, count)
        if not tuning_events:
            if attempt > 1:
                print(f"{count}-GPU cache is warm after {attempt-1} tuning attempt(s).", flush=True)
            return benchmark
        print(
            f"{count}-GPU attempt {attempt} contained {len(tuning_events)} CUDA tuning event(s); "
            "discarding it as a throughput result and rerunning with the persisted cache.",
            flush=True,
        )
    raise RuntimeError(
        f"{count}-GPU CUDA autotuning did not become clean after {args.max_attempts} attempts"
    )


def semantic_signature(result):
    return {
        key: result.get(key)
        for key in (
            "stories",
            "input_bytes",
            "base_training_targets",
            "replay_segments",
            "replay_steps",
            "replay_prefix_steps",
            "training_state_sha256",
        )
    }


def main():
    args = parse_args()
    args.leo = Path(args.leo).resolve()
    args.model = Path(args.model).resolve()
    args.bytes = Path(args.bytes).resolve()
    args.index = Path(args.index).resolve()
    for path in (args.leo, args.model, args.bytes, args.index):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.workers <= 0 or args.stories <= 0:
        raise ValueError("--workers and --stories must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    devices = discover_devices(args.devices)
    max_devices = min(len(devices), args.workers)
    if args.counts:
        counts = sorted({int(value.strip()) for value in args.counts.split(",") if value.strip()})
        if not counts or counts[0] != 1 or any(value < 1 or value > max_devices for value in counts):
            raise ValueError(f"--counts must include 1 and stay within 1..{max_devices}")
    else:
        counts = counts_to_test(max_devices)
    print(f"Detected devices: {devices}", flush=True)
    print(f"Testing GPU counts: {counts}", flush=True)

    results = []
    for count in counts:
        cases = [run_clean_case(args, devices, count) for _ in range(args.repeats)]
        cases.sort(key=lambda item: float(item["steps_per_second"]))
        hashes = {case.get("training_state_sha256") for case in cases}
        if len(hashes) != 1:
            raise RuntimeError(f"{count}-GPU repeated runs produced different final state hashes")
        median = cases[len(cases) // 2]
        median["steps_per_second_min"] = float(cases[0]["steps_per_second"])
        median["steps_per_second_max"] = float(cases[-1]["steps_per_second"])
        results.append((count, median))

    base_count, base = results[0]
    assert base_count == 1
    base_rate = float(base["steps_per_second"])
    base_signature = semantic_signature(base)
    base_loss = float(base["mean_loss"])

    print("\n" + "=" * 96)
    print("LEO MULTI-GPU SCALING SUMMARY")
    print("=" * 96)
    print(
        f"{'GPUs':>5} {'steps/s':>14} {'speedup':>10} {'efficiency':>12} "
        f"{'seconds':>12} {'semantic':>10} {'loss_delta':>12}"
    )
    all_semantic = True
    for count, result in results:
        rate = float(result["steps_per_second"])
        speedup = rate / base_rate
        efficiency = speedup / count
        same = semantic_signature(result) == base_signature
        loss_delta = abs(float(result["mean_loss"]) - base_loss)
        state_equal = result.get("training_state_sha256") == base.get("training_state_sha256")
        semantic_ok = same and state_equal and loss_delta <= 1.0e-6
        all_semantic &= semantic_ok
        print(
            f"{count:5d} {rate:14.3f} {speedup:10.3f} {efficiency:12.3f} "
            f"{float(result['seconds']):12.3f} {str(semantic_ok):>10} {loss_delta:12.3g}"
        )

    print("\nBase semantic signature:", base_signature)
    print("Exact workload/final-state check:", "OK" if all_semantic else "MISMATCH")
    if not all_semantic:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
