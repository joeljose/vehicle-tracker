#!/usr/bin/env python3
"""Aggregate per-teacher comparison predictions into a single stats.json.

Reads data/comparison/predictions/{teacher}/{slug}/{stem}.json for every
teacher + frame, computes per-teacher summary statistics, and a pairwise
agreement metric against the YOLOv8x baseline. Writes
data/comparison/stats.json.

Run this after running run_comparison.py for every teacher (or at least
for YOLOv8x + any candidates you want stats for).

No ground truth is required — these are qualitative-support metrics only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402

COMPARISON_ROOT = REPO_ROOT / "data" / "comparison"
PLAN_PATH = COMPARISON_ROOT / "plan.json"
PREDICTIONS_ROOT = COMPARISON_ROOT / "predictions"
STATS_PATH = COMPARISON_ROOT / "stats.json"

# Post-bake-off we keep only the teachers that matter: the chosen
# teacher (rfdetr_m) and whatever student(s) we've trained. Historical
# bake-off results are preserved in training/experiments/teacher_comparison.md.
TEACHERS = [
    "rfdetr_m",
    "student_v1",
    "student_v1_cont",
]
# The baseline for pairwise agreement. For student evaluation the real
# question is "did the student inherit the teacher's knowledge?" so we
# compare against the teacher, not the COCO-pretrained YOLOv8x baseline.
BASELINE_TEACHER = "rfdetr_m"
IOU_MATCH = 0.5

# Histogram buckets (percent-of-image-area)
BOX_AREA_BUCKETS = [0, 0.5, 1, 2, 5, 10, 25, 100]   # 7 buckets
ASPECT_RATIO_BUCKETS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100]  # 7 buckets


def load_frame_prediction(teacher: str, slug: str, stem: str) -> dict | None:
    path = PREDICTIONS_ROOT / teacher / slug / f"{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def xywhn_to_xyxy_pixels(xywhn: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if len(xywhn) == 0:
        return np.zeros((0, 4))
    h, w = hw
    cx = xywhn[:, 0] * w
    cy = xywhn[:, 1] * h
    bw = xywhn[:, 2] * w
    bh = xywhn[:, 3] * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = a[:, None, :]
    bb = b[None, :, :]
    inter_x1 = np.maximum(aa[..., 0], bb[..., 0])
    inter_y1 = np.maximum(aa[..., 1], bb[..., 1])
    inter_x2 = np.minimum(aa[..., 2], bb[..., 2])
    inter_y2 = np.minimum(aa[..., 3], bb[..., 3])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (aa[..., 2] - aa[..., 0]) * (aa[..., 3] - aa[..., 1])
    area_b = (bb[..., 2] - bb[..., 0]) * (bb[..., 3] - bb[..., 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def bucketize(values: list[float], boundaries: list[float]) -> list[int]:
    """Right-open buckets: [b0, b1), [b1, b2), ..., last is [b_{n-1}, b_n]."""
    hist = [0] * (len(boundaries) - 1)
    for v in values:
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= v < boundaries[i + 1]:
                hist[i] += 1
                break
        else:
            if v == boundaries[-1]:
                hist[-1] += 1
    return hist


def compute_teacher_stats(teacher: str, plan_frames: list[dict]) -> dict | None:
    """Single-teacher summary. Returns None if no predictions exist."""
    per_site: dict[str, dict] = {}
    total_detections = 0
    frames_with_detections = 0
    frames_empty = 0
    frames_missing = 0
    detections_per_frame: list[int] = []
    mean_conf_accum = 0.0
    mean_conf_n = 0
    box_area_pcts: list[float] = []
    aspect_ratios: list[float] = []

    for entry in plan_frames:
        slug = entry["site"]
        stem = entry["stem"]
        pred = load_frame_prediction(teacher, slug, stem)
        if pred is None:
            frames_missing += 1
            continue
        n = len(pred.get("boxes_xywhn", []))
        total_detections += n
        detections_per_frame.append(n)
        site = per_site.setdefault(slug, {"total": 0, "frames": 0})
        site["total"] += n
        site["frames"] += 1

        if n == 0:
            frames_empty += 1
            continue
        frames_with_detections += 1

        confs = pred.get("confs", [])
        if confs:
            mean_conf_accum += sum(confs)
            mean_conf_n += len(confs)

        boxes = pred["boxes_xywhn"]
        for box in boxes:
            cx, cy, bw, bh = box
            box_area_pcts.append(bw * bh * 100.0)
            if bh > 0:
                aspect_ratios.append(bw / bh)

    if frames_with_detections == 0 and frames_empty == 0:
        return None

    # Fill in per-site averages
    for slug, s in per_site.items():
        s["per_frame_mean"] = round(s["total"] / max(s["frames"], 1), 2)

    run_stats_p = PREDICTIONS_ROOT / teacher / "_run_stats.json"
    run_stats = json.loads(run_stats_p.read_text()) if run_stats_p.exists() else {}

    stats = {
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "frames_empty": frames_empty,
        "frames_missing": frames_missing,
        "detections_per_frame_mean": round(float(np.mean(detections_per_frame)), 2) if detections_per_frame else 0,
        "detections_per_frame_median": int(np.median(detections_per_frame)) if detections_per_frame else 0,
        "detections_per_frame_p90": int(np.percentile(detections_per_frame, 90)) if detections_per_frame else 0,
        "detections_per_frame_max": int(max(detections_per_frame)) if detections_per_frame else 0,
        "per_site": per_site,
        "box_area_pct_buckets": BOX_AREA_BUCKETS,
        "box_area_pct_histogram": bucketize(box_area_pcts, BOX_AREA_BUCKETS),
        "aspect_ratio_buckets": ASPECT_RATIO_BUCKETS,
        "aspect_ratio_histogram": bucketize(aspect_ratios, ASPECT_RATIO_BUCKETS),
        "mean_conf": round(mean_conf_accum / max(mean_conf_n, 1), 3),
        "inference_seconds_total": run_stats.get("inference_seconds_total"),
        "inference_ms_per_frame_mean": run_stats.get("inference_ms_per_frame_mean"),
    }
    return stats


def compute_pairwise_agreement(candidate: str, plan_frames: list[dict]) -> dict | None:
    """Greedy IoU>=0.5 match of candidate boxes against the baseline per frame."""
    if candidate == BASELINE_TEACHER:
        return None
    matched = 0
    extra_in_candidate = 0
    missed_from_baseline = 0
    total_baseline = 0
    total_candidate = 0

    for entry in plan_frames:
        slug, stem = entry["site"], entry["stem"]
        base = load_frame_prediction(BASELINE_TEACHER, slug, stem)
        cand = load_frame_prediction(candidate, slug, stem)
        if base is None or cand is None:
            continue

        hw = tuple(base.get("image_hw") or cand.get("image_hw") or (1080, 1920))
        base_xyxy = xywhn_to_xyxy_pixels(np.asarray(base["boxes_xywhn"]), hw)
        cand_xyxy = xywhn_to_xyxy_pixels(np.asarray(cand["boxes_xywhn"]), hw)

        total_baseline += len(base_xyxy)
        total_candidate += len(cand_xyxy)

        if len(base_xyxy) == 0 and len(cand_xyxy) == 0:
            continue
        if len(base_xyxy) == 0:
            extra_in_candidate += len(cand_xyxy)
            continue
        if len(cand_xyxy) == 0:
            missed_from_baseline += len(base_xyxy)
            continue

        iou = iou_matrix(cand_xyxy, base_xyxy)  # (Nc, Nb)
        matched_base: set[int] = set()
        matched_cand: set[int] = set()
        # Greedy by max iou
        flat = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
        flat.sort(reverse=True)
        for score, i, j in flat:
            if score < IOU_MATCH:
                break
            if i in matched_cand or j in matched_base:
                continue
            matched += 1
            matched_cand.add(i)
            matched_base.add(j)
        extra_in_candidate += len(cand_xyxy) - len(matched_cand)
        missed_from_baseline += len(base_xyxy) - len(matched_base)

    return {
        "iou50_matched": matched,
        "iou50_match_rate_vs_baseline_boxes": round(matched / max(total_baseline, 1), 3),
        "extra_in_candidate": extra_in_candidate,
        "missed_from_baseline": missed_from_baseline,
        "total_baseline_boxes": total_baseline,
        "total_candidate_boxes": total_candidate,
    }


def main() -> None:
    if not PLAN_PATH.exists():
        log(f"ERROR: {PLAN_PATH} not found.")
        sys.exit(1)
    plan = json.loads(PLAN_PATH.read_text())
    plan_frames = plan["frames"]
    log(f"Aggregating over {len(plan_frames)} frames across {len(TEACHERS)} teachers")

    out: dict = {
        "comparison_set": {
            "total_frames": len(plan_frames),
            "seed": plan.get("seed"),
            "per_site": plan.get("per_site"),
        },
        "teachers": {},
        "pairwise_agreement_vs_baseline": {},
    }

    for teacher in TEACHERS:
        log(f"  {teacher}...")
        stats = compute_teacher_stats(teacher, plan_frames)
        if stats is None:
            log(f"    no predictions found; skipping")
            continue
        out["teachers"][teacher] = stats

    for teacher in TEACHERS:
        if teacher == BASELINE_TEACHER:
            continue
        if teacher not in out["teachers"]:
            continue
        pw = compute_pairwise_agreement(teacher, plan_frames)
        if pw is None:
            continue
        out["pairwise_agreement_vs_baseline"][teacher] = pw

    STATS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Wrote {STATS_PATH}")

    # Compact dashboard
    log("")
    log("=" * 70)
    log(f"{'teacher':<20} {'total':>8} {'per-frame':>10} {'mean-conf':>10} {'sec':>6}")
    log("-" * 70)
    for teacher in TEACHERS:
        s = out["teachers"].get(teacher)
        if s is None:
            continue
        log(
            f"{teacher:<20} "
            f"{s['total_detections']:>8} "
            f"{s['detections_per_frame_mean']:>10} "
            f"{s['mean_conf']:>10} "
            f"{s.get('inference_seconds_total', 0) or 0:>6}"
        )
    log("=" * 70)


if __name__ == "__main__":
    main()
