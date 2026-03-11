#!/usr/bin/env python3
"""
Atlas Runtime Estimator
========================
Collects data points to estimate how long a GPU/Trainium task will take.
Must run BEFORE gpu_runner.py to justify the requested runtime.

Usage:
    python runtime_estimator.py --task seedpolicy --instance g5.xlarge

Collects:
    - Dataset size (number of tasks/samples)
    - Per-sample estimated time (from paper or quick test)
    - Model size / memory requirement
    - Training epochs
    - Historical runs (if any)
    - Buffer factor (1.5x for safety)

Output: runtime estimate + cost estimate → saved to logs/
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SPOT_PRICES = {
    "g5.xlarge": 0.35,    # A10G 24GB
    "trn1.2xlarge": 0.50,  # 1 NeuronCore
}

LOG_DIR = Path.home() / ".openclaw" / "workspace" / "logs" / "gpu-runs"
HISTORY_FILE = LOG_DIR / "runtime-history.json"


def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {}


def save_history(history):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def estimate(task, instance_type, data_points):
    """Calculate runtime estimate from data points."""
    
    # Core calculation
    num_samples = data_points.get("num_samples", 0)
    time_per_sample_min = data_points.get("time_per_sample_min", 0)
    num_epochs = data_points.get("num_epochs", 1)
    setup_time_min = data_points.get("setup_time_min", 15)  # install deps, copy data
    
    # Raw estimate
    raw_min = (num_samples * time_per_sample_min * num_epochs) + setup_time_min
    
    # Buffer: 1.5x for unknowns (rate limits, retries, etc.)
    buffer = data_points.get("buffer_factor", 1.5)
    buffered_min = raw_min * buffer
    
    hours = buffered_min / 60
    cost = hours * SPOT_PRICES.get(instance_type, 0.50)
    
    return {
        "task": task,
        "instance_type": instance_type,
        "data_points": data_points,
        "raw_estimate_min": round(raw_min, 1),
        "buffer_factor": buffer,
        "buffered_estimate_min": round(buffered_min, 1),
        "estimated_hours": round(hours, 2),
        "estimated_cost_usd": round(cost, 2),
        "spot_rate_usd_hr": SPOT_PRICES.get(instance_type, 0.50),
        "recommended_max_hours": max(1, round(hours + 0.5)),  # Round up + 30min safety
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def print_estimate(est):
    print(f"{'='*60}")
    print(f"📊 Runtime Estimate: {est['task']}")
    print(f"{'='*60}")
    print(f"  Instance:     {est['instance_type']} (spot ${est['spot_rate_usd_hr']}/hr)")
    print()
    print(f"  Data points:")
    dp = est["data_points"]
    for k, v in dp.items():
        print(f"    {k}: {v}")
    print()
    print(f"  Raw estimate:      {est['raw_estimate_min']} min")
    print(f"  Buffer ({est['buffer_factor']}x):      {est['buffered_estimate_min']} min")
    print(f"  Estimated time:    {est['estimated_hours']}h")
    print(f"  Estimated cost:    ${est['estimated_cost_usd']}")
    print(f"  Recommended max:   {est['recommended_max_hours']}h")
    print(f"{'='*60}")
    
    # Compare with history if available
    history = load_history()
    if est["task"] in history:
        prev = history[est["task"]]
        if prev.get("actual_hours") is not None:
            print(f"\n  📈 Previous run: {prev['actual_hours']}h (estimated {prev['estimated_hours']}h)")
            accuracy = prev['actual_hours'] / prev['estimated_hours'] * 100 if prev['estimated_hours'] > 0 else 0
            print(f"  Estimation accuracy: {accuracy:.0f}%")
    
    return est


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Atlas Runtime Estimator")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--instance", required=True, help="Instance type")
    
    # Data points
    parser.add_argument("--samples", type=int, default=0, help="Number of samples/tasks to process")
    parser.add_argument("--time-per-sample", type=float, default=0, help="Estimated minutes per sample")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--setup-min", type=float, default=15, help="Setup time in minutes")
    parser.add_argument("--buffer", type=float, default=1.5, help="Safety buffer factor")
    parser.add_argument("--model-size-gb", type=float, default=0, help="Model size in GB (for memory check)")
    parser.add_argument("--gpu-memory-gb", type=float, default=24, help="Available GPU memory in GB")
    parser.add_argument("--note", default="", help="Additional context")
    
    # Record actual results
    parser.add_argument("--record-actual", type=float, help="Record actual runtime hours (post-run)")
    
    args = parser.parse_args()
    
    # Record actual results mode
    if args.record_actual is not None:
        history = load_history()
        if args.task in history:
            history[args.task]["actual_hours"] = args.record_actual
        else:
            history[args.task] = {
                "actual_hours": args.record_actual,
                "estimated_hours": 0,
                "instance_type": args.instance,
            }
        save_history(history)
        print(f"✅ Recorded: {args.task} actual runtime = {args.record_actual}h")
        return
    
    data_points = {
        "num_samples": args.samples,
        "time_per_sample_min": args.time_per_sample,
        "num_epochs": args.epochs,
        "setup_time_min": args.setup_min,
        "buffer_factor": args.buffer,
        "model_size_gb": args.model_size_gb,
        "gpu_memory_gb": args.gpu_memory_gb,
        "note": args.note,
    }
    
    # Memory check
    if args.model_size_gb > 0 and args.model_size_gb > args.gpu_memory_gb:
        print(f"⚠️  WARNING: Model ({args.model_size_gb}GB) may not fit in GPU memory ({args.gpu_memory_gb}GB)")
    
    est = estimate(args.task, args.instance, data_points)
    print_estimate(est)
    
    # Save estimate
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    est_file = LOG_DIR / f"estimate-{args.task}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
    est_file.write_text(json.dumps(est, indent=2))
    print(f"\n  Saved: {est_file}")
    
    # Save to history
    history = load_history()
    history[args.task] = {
        "estimated_hours": est["estimated_hours"],
        "actual_hours": None,
        "instance_type": args.instance,
        "timestamp": est["timestamp"],
    }
    save_history(history)


if __name__ == "__main__":
    main()
