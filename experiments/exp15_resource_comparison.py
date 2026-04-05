#!/usr/bin/env python3
"""exp15: Backend resource usage comparison.

Compares CPU, RAM, GPU utilization between DeepStream and Custom backends
running the same video file. Drives the pipeline via REST API and polls
system metrics every second.

Usage:
    # Run with Custom backend (make dev && make start first)
    python experiments/exp15_resource_comparison.py --backend custom

    # Run with DeepStream backend (make dev BACKEND_TYPE=deepstream && make start first)
    python experiments/exp15_resource_comparison.py --backend deepstream

    # Compare results
    python experiments/exp15_resource_comparison.py --compare
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

API = "http://localhost:8000"
SOURCE = "/data/test_clips/drugmart_73_1min.mp4"
SITE_CONFIG = "backend/config/sites/drugmart.json"
OUTPUT_DIR = "experiments/backend_comparison"
CONTAINER_NAME = "vehicle_tracker-backend-1"
POLL_INTERVAL = 1.0  # seconds


def api(method, path, data=None):
    url = f"{API}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()}


def get_docker_stats():
    """Get CPU% and MEM usage from docker stats."""
    try:
        out = subprocess.check_output(
            ["docker", "stats", CONTAINER_NAME, "--no-stream",
             "--format", "{{.CPUPerc}}|{{.MemUsage}}"],
            text=True, timeout=5,
        ).strip()
        parts = out.split("|")
        cpu = float(parts[0].replace("%", ""))
        mem_str = parts[1].split("/")[0].strip()
        if "GiB" in mem_str:
            mem_mb = float(mem_str.replace("GiB", "")) * 1024
        elif "MiB" in mem_str:
            mem_mb = float(mem_str.replace("MiB", ""))
        else:
            mem_mb = 0
        return cpu, mem_mb
    except Exception:
        return 0.0, 0.0


def get_gpu_stats():
    """Get GPU memory, utilization, power from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_mem_mb": float(parts[0]),
            "gpu_util_pct": float(parts[1]),
            "gpu_power_w": float(parts[2]),
        }
    except Exception:
        return {"gpu_mem_mb": 0, "gpu_util_pct": 0, "gpu_power_w": 0}


def get_channel_phase(channel_id=0):
    try:
        data = api("GET", f"/channel/{channel_id}")
        return data.get("phase", "unknown")
    except Exception:
        return "unknown"


def run_benchmark(backend_name):
    print(f"\n{'='*60}")
    print(f"  exp15: Resource Benchmark — {backend_name.upper()}")
    print(f"{'='*60}\n")

    # Load site config
    with open(SITE_CONFIG) as f:
        site = json.load(f)

    # 1. Start pipeline
    print("[1/5] Starting pipeline...")
    resp = api("POST", "/pipeline/start")
    print(f"  → {resp}")

    # 2. Add channel
    print(f"[2/5] Adding channel: {SOURCE}")
    resp = api("POST", "/channel/add", {"source": SOURCE})
    channel_id = resp.get("channel_id", 0)
    print(f"  → channel_id={channel_id}")

    # Wait for stream to initialize
    time.sleep(3)

    # 3. Transition to Analytics
    print("[3/5] Starting Analytics...")
    phase_data = {
        "phase": "analytics",
        "roi_polygon": site["roi_polygon"],
        "entry_exit_lines": site["entry_exit_lines"],
    }
    resp = api("POST", f"/channel/{channel_id}/phase", phase_data)
    print(f"  → {resp}")

    # Wait for analytics to start
    time.sleep(2)

    # 4. Poll metrics
    print("[4/5] Polling metrics (1s interval)...")
    samples = []
    t_start = time.monotonic()

    while True:
        cpu, mem_mb = get_docker_stats()
        gpu = get_gpu_stats()
        elapsed = time.monotonic() - t_start

        sample = {
            "elapsed_s": round(elapsed, 1),
            "cpu_pct": cpu,
            "mem_mb": round(mem_mb, 1),
            **gpu,
        }
        samples.append(sample)

        phase = get_channel_phase(channel_id)
        sys.stdout.write(
            f"\r  t={elapsed:5.1f}s  CPU={cpu:5.1f}%  RAM={mem_mb:7.1f}MB  "
            f"GPU={gpu['gpu_mem_mb']:6.0f}MB/{gpu['gpu_util_pct']:3.0f}%  "
            f"Power={gpu['gpu_power_w']:5.1f}W  phase={phase}"
        )
        sys.stdout.flush()

        if phase == "review":
            print(f"\n  → Video finished (review phase) after {elapsed:.1f}s")
            break

        if elapsed > 120:
            print(f"\n  → Timeout after {elapsed:.1f}s")
            break

        time.sleep(POLL_INTERVAL)

    wall_time = time.monotonic() - t_start

    # 5. Collect final stats
    print("[5/5] Collecting results...")
    alerts = api("GET", f"/alerts?channel={channel_id}&limit=200")
    alert_count = len(alerts) if isinstance(alerts, list) else 0

    # Stop pipeline
    api("POST", "/pipeline/stop")

    # Compute summary
    def avg(key):
        vals = [s[key] for s in samples]
        return round(sum(vals) / len(vals), 1) if vals else 0

    def peak(key):
        vals = [s[key] for s in samples]
        return round(max(vals), 1) if vals else 0

    summary = {
        "backend": backend_name,
        "source": SOURCE,
        "wall_time_s": round(wall_time, 1),
        "samples": len(samples),
        "alert_count": alert_count,
        "avg_cpu_pct": avg("cpu_pct"),
        "peak_cpu_pct": peak("cpu_pct"),
        "avg_mem_mb": avg("mem_mb"),
        "peak_mem_mb": peak("mem_mb"),
        "avg_gpu_mem_mb": avg("gpu_mem_mb"),
        "peak_gpu_mem_mb": peak("gpu_mem_mb"),
        "avg_gpu_util_pct": avg("gpu_util_pct"),
        "peak_gpu_util_pct": peak("gpu_util_pct"),
        "avg_gpu_power_w": avg("gpu_power_w"),
        "peak_gpu_power_w": peak("gpu_power_w"),
    }

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{backend_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "samples": samples}, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Print summary
    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*40}")
    for k, v in summary.items():
        if k not in ("backend", "source", "samples"):
            print(f"  {k:<25} {v:>10}")

    return summary


def compare():
    print(f"\n{'='*60}")
    print(f"  exp15: Backend Resource Comparison")
    print(f"{'='*60}\n")

    results = {}
    for backend in ("deepstream", "custom"):
        path = os.path.join(OUTPUT_DIR, f"{backend}_metrics.json")
        if not os.path.exists(path):
            print(f"  Missing: {path} — run benchmark for {backend} first")
            return
        with open(path) as f:
            results[backend] = json.load(f)["summary"]

    ds = results["deepstream"]
    cu = results["custom"]

    rows = [
        ("Wall-clock time (s)", "wall_time_s"),
        ("Alert count", "alert_count"),
        ("Avg CPU %", "avg_cpu_pct"),
        ("Peak CPU %", "peak_cpu_pct"),
        ("Avg RAM (MB)", "avg_mem_mb"),
        ("Peak RAM (MB)", "peak_mem_mb"),
        ("Avg GPU memory (MB)", "avg_gpu_mem_mb"),
        ("Peak GPU memory (MB)", "peak_gpu_mem_mb"),
        ("Avg GPU utilization %", "avg_gpu_util_pct"),
        ("Peak GPU utilization %", "peak_gpu_util_pct"),
        ("Avg GPU power (W)", "avg_gpu_power_w"),
        ("Peak GPU power (W)", "peak_gpu_power_w"),
    ]

    # Print table
    print(f"| {'Metric':<25} | {'DeepStream':>12} | {'Custom':>12} |")
    print(f"|{'-'*27}|{'-'*14}|{'-'*14}|")
    for label, key in rows:
        print(f"| {label:<25} | {ds[key]:>12} | {cu[key]:>12} |")

    # Save markdown
    md_path = "experiments/exp15_results.md"
    with open(md_path, "w") as f:
        f.write("# exp15: Backend Resource Comparison\n\n")
        f.write(f"Source: `{ds['source']}`\n\n")
        f.write(f"| {'Metric':<25} | {'DeepStream':>12} | {'Custom':>12} |\n")
        f.write(f"|{'-'*27}|{'-'*14}|{'-'*14}|\n")
        for label, key in rows:
            f.write(f"| {label:<25} | {ds[key]:>12} | {cu[key]:>12} |\n")
    print(f"\nSaved to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend resource benchmark")
    parser.add_argument("--backend", choices=["custom", "deepstream"])
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.backend:
        run_benchmark(args.backend)
    else:
        parser.print_help()
