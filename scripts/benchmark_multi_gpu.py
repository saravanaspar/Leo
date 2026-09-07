#!/usr/bin/env python3
"""Live 1-vs-N GPU scaling benchmark for Leo's exact data-parallel trainer."""

import argparse
import json
import os
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
    return parser.parse_args()


def discover_devices(explicit):
    if explicit:
        devices = [item.strip() for item in explicit.split(",") if item.strip()]
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


def run_case(args, devices, count):
    visible = devices[:count]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    env["LEO_MULTI_GPU"] = "1" if count > 1 else "0"

    # Keep final throughput measurements free of optional profilers/debuggers.
    for key in list(env):
        if key.startswith("LEO_CUDA_DEBUG") or key.startswith("LEO_CUDA_REPLAY_PROFILE"):
            env.pop(key, None)
        if key.startswith("LEO_REPLAY_DEBUG") or key.startswith("LEO_CUDA_PHASE_PROFILE"):
            env.pop(key, None)

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

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if event:
            if event.get("event") == "training_benchmark":
                benchmark = event
            elif event.get("event") in {"multi_gpu_training", "multi_gpu_sparse_sync"}:
                multi_events.append(event)
            if event.get("event") in {
                "cuda_profile",
                "cuda_autotune_resumed",
                "cuda_autotune_complete",
            }:
                tuning_events.append(event)
        now = time.monotonic()
        if now - last_heartbeat >= 10:
            print(f"[scaling heartbeat] {now-started:.1f}s", flush=True)
            last_heartbeat = now

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{count}-GPU benchmark failed with exit code {rc}")
    if benchmark is None:
        raise RuntimeError(f"{count}-GPU benchmark emitted no training_benchmark event")
    if int(benchmark.get("gpu_devices", 1)) != count:
        raise RuntimeError(
            f"requested {count} GPUs but Leo reported gpu_devices={benchmark.get('gpu_devices')}"
        )
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

    devices = discover_devices(args.devices)
    max_devices = min(len(devices), args.workers)
    counts = counts_to_test(max_devices)
    print(f"Detected devices: {devices}", flush=True)
    print(f"Testing GPU counts: {counts}", flush=True)

    results = []
    for count in counts:
        results.append((count, run_clean_case(args, devices, count)))

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
        semantic_ok = same and loss_delta <= 1.0e-6
        all_semantic &= semantic_ok
        print(
            f"{count:5d} {rate:14.3f} {speedup:10.3f} {efficiency:12.3f} "
            f"{float(result['seconds']):12.3f} {str(semantic_ok):>10} {loss_delta:12.3g}"
        )

    print("\nBase semantic signature:", base_signature)
    print("Semantic workload/loss check:", "OK" if all_semantic else "MISMATCH")
    if not all_semantic:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
