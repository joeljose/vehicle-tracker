#!/usr/bin/env python3
"""Evaluate one or more trained checkpoints on the held-out test split.

Uses Ultralytics' built-in `model.val(split='test')` which runs inference
on data/splits/test/ and computes mAP50, mAP50-95, precision, recall
against the RF-DETR auto-labels in data/splits/test/labels/.

Why this matters: the val loss curves during training used the val split
for early stopping, and our 1000-frame comparison set OVERLAPS with the
training split. The test split is the only completely held-out data —
never backprop'd, never used for early stopping, never sampled into the
comparison set. It's the cleanest way to compare two trained checkpoints.

Important caveat: the "ground truth" in the test split is RF-DETR's
auto-labels, NOT human-verified annotations. The scores measure how well
each student imitates RF-DETR on unseen-to-them frames. They don't
measure real-world detection accuracy — we never built a human GT set.

Usage:
    python3 scripts/eval_on_test_split.py --runs yolov8s_rfdetr_v1 yolov8s_rfdetr_v1_cont

    # single run
    python3 scripts/eval_on_test_split.py --runs yolov8s_rfdetr_v1_cont
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402

RUNS_ROOT = REPO_ROOT / "runs" / "detect"
DATASET_YAML = REPO_ROOT / "data" / "dataset.yaml"


def run_val_on_test(run_name: str) -> dict:
    """Run ultralytics val(split='test') on the given run's best.pt.

    Returns a dict with mAP50, mAP50-95, precision, recall, and timing.
    """
    weights = RUNS_ROOT / run_name / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"ERROR: {weights} not found")

    from ultralytics import YOLO

    log(f"[{run_name}] loading {weights.name}...")
    model = YOLO(str(weights))

    log(f"[{run_name}] running validation on test split...")
    t0 = time.time()
    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        half=True,
        save_json=False,
        plots=False,
        verbose=False,
        # Put outputs in a temp subdir so we don't litter the real run dir
        project=str(RUNS_ROOT / run_name),
        name=f"test_eval_{int(time.time())}",
        exist_ok=True,
    )
    elapsed = time.time() - t0

    # Ultralytics' results object has .box with the metrics
    box = results.box
    return {
        "run": run_name,
        "weights": str(weights),
        "n_images": int(results.nt_per_class.sum()) if hasattr(results, "nt_per_class") else None,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "fitness": round(float(box.fitness()) if callable(getattr(box, "fitness", None)) else 0.0, 4),
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run names under runs/detect/ to evaluate",
    )
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        log(f"ERROR: {DATASET_YAML} not found. Run scripts/split_dataset.py first.")
        sys.exit(1)

    all_results: list[dict] = []
    for run_name in args.runs:
        result = run_val_on_test(run_name)
        all_results.append(result)

    log("")
    log("=" * 78)
    log("TEST SPLIT EVALUATION")
    log("=" * 78)
    log(
        f"{'run':<30} {'mAP50':>8} {'mAP50-95':>10} "
        f"{'precision':>11} {'recall':>9} {'sec':>6}"
    )
    log("-" * 78)
    for r in all_results:
        log(
            f"{r['run']:<30} {r['mAP50']:>8} {r['mAP50_95']:>10} "
            f"{r['precision']:>11} {r['recall']:>9} {r['elapsed_seconds']:>6}"
        )
    log("=" * 78)

    if len(all_results) >= 2:
        log("")
        log("Deltas (later runs vs first run):")
        base = all_results[0]
        for r in all_results[1:]:
            log(f"  {r['run']} vs {base['run']}:")
            for k in ("mAP50", "mAP50_95", "precision", "recall"):
                delta = r[k] - base[k]
                sign = "+" if delta >= 0 else ""
                log(f"    {k:<12} {base[k]:.4f} -> {r[k]:.4f}  ({sign}{delta:.4f})")


if __name__ == "__main__":
    main()
