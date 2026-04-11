#!/usr/bin/env python3
"""Run one teacher on the M8-P1.5 v2 comparison set.

Reads data/comparison/plan.json, runs the requested teacher on every image,
and saves raw per-frame predictions as JSON to
  data/comparison/predictions/{teacher}/{slug}/{stem}.json

Each JSON contains:
  {
    "image_hw": [H, W],
    "boxes_xywhn": [[cx, cy, w, h], ...],
    "confs": [c1, c2, ...],
    "class_ids": [0, 0, ...],          # all 0 post-collapse
    "inference_ms": <per-frame ms>
  }

Also appends per-teacher inference stats to
  data/comparison/predictions/{teacher}/_run_stats.json

Resumable: skips frames whose prediction JSON already exists.

Usage:
    python3 scripts/run_comparison.py --teacher yolov8x
    python3 scripts/run_comparison.py --teacher rfdetr_m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402
from teachers import get_teacher  # noqa: E402

COMPARISON_ROOT = REPO_ROOT / "data" / "comparison"
PLAN_PATH = COMPARISON_ROOT / "plan.json"
PREDICTIONS_ROOT = COMPARISON_ROOT / "predictions"

CONF_THRESHOLD = 0.3


def prediction_path(teacher: str, slug: str, stem: str) -> Path:
    return PREDICTIONS_ROOT / teacher / slug / f"{stem}.json"


def run_stats_path(teacher: str) -> Path:
    return PREDICTIONS_ROOT / teacher / "_run_stats.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher backend (yolov8x, yolo26x, yolo12x, rfdetr_m, grounding_dino_t)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if per-frame prediction JSON already exists",
    )
    args = parser.parse_args()

    if not PLAN_PATH.exists():
        log(f"ERROR: {PLAN_PATH} not found. Run scripts/build_comparison_set.py first.")
        sys.exit(1)

    plan = json.loads(PLAN_PATH.read_text())
    all_frames = plan["frames"]
    log(f"Plan has {len(all_frames)} frames")

    # Filter to pending (unless forced)
    pending: list[dict] = []
    for entry in all_frames:
        out = prediction_path(args.teacher, entry["site"], entry["stem"])
        if args.force or not out.exists():
            pending.append(entry)
    log(f"Pending: {len(pending)} frames ({len(all_frames) - len(pending)} already done)")

    if not pending:
        log("Nothing to do.")
        return

    log(f"Loading teacher: {args.teacher}")
    t0 = time.time()
    teacher = get_teacher(args.teacher)
    load_s = time.time() - t0
    log(f"  loaded in {load_s:.1f}s")

    image_paths = [
        REPO_ROOT / "data" / "comparison" / "images" / e["site"] / f"{e['stem']}.jpg"
        for e in pending
    ]

    log(f"Running inference on {len(image_paths)} frames...")
    t0 = time.time()
    detections = teacher.predict(image_paths, conf_threshold=CONF_THRESHOLD)
    total_s = time.time() - t0
    log(f"  inference done in {total_s:.1f}s ({total_s / len(image_paths) * 1000:.0f}ms/frame)")

    # Save per-frame JSON
    log(f"Writing {len(detections)} prediction files...")
    per_frame_ms = (total_s / len(image_paths)) * 1000
    for entry, det in zip(pending, detections):
        out = prediction_path(args.teacher, entry["site"], entry["stem"])
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_hw": list(det.image_hw),
            "boxes_xywhn": det.boxes_xywhn.tolist(),
            "confs": det.confs.tolist(),
            "class_ids": det.project_class_ids.tolist(),
            "inference_ms": per_frame_ms,
        }
        out.write_text(json.dumps(payload) + "\n")

    # Update run stats
    stats = {
        "teacher": args.teacher,
        "conf_threshold": CONF_THRESHOLD,
        "total_frames": len(image_paths),
        "load_seconds": round(load_s, 2),
        "inference_seconds_total": round(total_s, 2),
        "inference_ms_per_frame_mean": round(per_frame_ms, 2),
    }
    run_stats_path(args.teacher).write_text(json.dumps(stats, indent=2) + "\n")
    log(f"Run stats: {run_stats_path(args.teacher)}")
    log("Done.")


if __name__ == "__main__":
    main()
