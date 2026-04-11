#!/usr/bin/env python3
"""Auto-label deduped frames with a configurable teacher backend.

Pass `--teacher <name>` to choose which model produces the pseudo-labels.
Available teachers (see scripts/teachers/__init__.py REGISTRY):
    yolov8x          (default — current baseline)
    yolo26x          (NMS-free, January 2026)
    yolo12x          (attention-centric, 2025)
    rfdetr_m         (Roboflow DETR, ICLR 2026, transformer)
    grounding_dino_t (HuggingFace zero-shot, text-prompted)

Each teacher returns project class 0 ("vehicle") under the M8-P1.5 v2
single-class collapse. Inference on all frames in `data/frames_dedup/{slug}/`,
writes YOLO-format `.txt` files.

Routing rules:

  - accepted = frame has at least one detection with conf >= ACCEPT_BEST (0.5)
  - flagged  = anything else with at least one detection
  - empty    = no detections (counted, no file written)

The LABEL FILE itself contains every detection above CONF_LOW (0.3),
regardless of which bin it lands in. The routing decides who looks at it,
not what's in it.

A spot-check review queue is emitted alongside the manifest
(`data/labels/review_queue_spotcheck.json`) — a uniform random sample of
accepted frames per site, used to sanity-check the accept bin overall.

Resumable: skips frames whose label already exists in either accepted/ or
flagged/ for the site.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Make `teachers` package importable when run as `python3 scripts/auto_label.py`
sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, load_video_sites, log, write_manifest  # noqa: E402

DEDUP_ROOT = REPO_ROOT / "data" / "frames_dedup"
LABELS_ROOT = REPO_ROOT / "data" / "labels"
ACCEPTED_ROOT = LABELS_ROOT / "accepted"
FLAGGED_ROOT = LABELS_ROOT / "flagged"
MANIFEST_PATH = LABELS_ROOT / "manifest.json"
REVIEW_QUEUE_SPOT_PATH = LABELS_ROOT / "review_queue_spotcheck.json"

PROJECT_NAMES = {0: "vehicle"}

CONF_LOW = 0.3  # inference floor — anything below this is dropped entirely
# Frame-level accept rule: at least one detection must clear ACCEPT_BEST.
# Under the single-class collapse there's no rare-class guard — every
# detection is just "vehicle" and a single threshold is sufficient.
ACCEPT_BEST = 0.5

DEFAULT_TEACHER = "yolov8x"

# Spot-check sampling: uniform random N accepted frames per site, no
# confidence filter. Borderline-accept frames are equally likely to be
# sampled — those are also error candidates and we want them in the mix.
SPOT_CHECK_PER_SITE = 50
SPOT_CHECK_SEED = 0


def empty_histogram() -> list[int]:
    # 7 buckets: [0.3-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0]
    return [0] * 7


def conf_bucket(c: float) -> int:
    idx = int((c - CONF_LOW) / 0.1)
    return max(0, min(6, idx))


def label_exists(stem: str, slug: str) -> bool:
    return (ACCEPTED_ROOT / slug / f"{stem}.txt").exists() or (
        FLAGGED_ROOT / slug / f"{stem}.txt").exists()


def gather_pending_frames(slug: str) -> list[Path]:
    site_dir = DEDUP_ROOT / slug
    if not site_dir.exists():
        return []
    return [
        p for p in sorted(site_dir.glob("frame_*.jpg")) if not label_exists(p.stem, slug)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        default=DEFAULT_TEACHER,
        help="Teacher backend (yolov8x, yolo26x, yolo12x, rfdetr_m, grounding_dino_t)",
    )
    args = parser.parse_args()

    sites = load_video_sites()

    # Heavy import after CLI sanity
    from teachers import get_teacher

    log(f"Loading teacher model: {args.teacher}")
    teacher = get_teacher(args.teacher)

    manifest_sites: dict[str, dict] = {}
    totals = {
        "frames_processed": 0,
        "frames_accepted": 0,
        "frames_flagged": 0,
        "frames_empty": 0,
        "class_counts": {name: 0 for name in PROJECT_NAMES.values()},
        "conf_histogram": empty_histogram(),
    }
    spot_check_candidates: dict[str, list[str]] = {}  # site -> all accepted paths (uniform sample pool)

    for slug in sites:
        site_dir = DEDUP_ROOT / slug
        if not site_dir.exists():
            log(f"  [warn] {slug}: dedup dir missing, skipping")
            continue

        (ACCEPTED_ROOT / slug).mkdir(parents=True, exist_ok=True)
        (FLAGGED_ROOT / slug).mkdir(parents=True, exist_ok=True)

        pending = gather_pending_frames(slug)
        site_total = sum(1 for _ in site_dir.glob("frame_*.jpg"))
        if not pending:
            log(f"  [skip] {slug}: all {site_total} frames already labeled")
            site_stats = recount_site(slug, site_total)
            manifest_sites[slug] = site_stats
            _accumulate(totals, site_stats)
            continue

        log(f"  [run]  {slug}: labeling {len(pending)}/{site_total} frames")
        site_stats = {
            "frames_processed": 0,
            "frames_accepted": 0,
            "frames_flagged": 0,
            "frames_empty": 0,
            "frames_accepted_with_rare": 0,
            "class_counts": {name: 0 for name in PROJECT_NAMES.values()},
            "conf_histogram": empty_histogram(),
        }

        detections_iter = teacher.predict(pending, conf_threshold=CONF_LOW)
        for frame_path, det in zip(pending, detections_iter):
            site_stats["frames_processed"] += 1
            if len(det.boxes_xywhn) == 0:
                site_stats["frames_empty"] += 1
                continue

            lines: list[str] = []
            kept_confs: list[float] = []
            for (cx, cy, w, h), conf, project_id in zip(
                det.boxes_xywhn, det.confs, det.project_class_ids
            ):
                pid = int(project_id)
                lines.append(
                    f"{pid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                kept_confs.append(float(conf))
                site_stats["class_counts"][PROJECT_NAMES[pid]] += 1
                site_stats["conf_histogram"][conf_bucket(float(conf))] += 1

            if not lines:
                site_stats["frames_empty"] += 1
                continue

            # Single-class routing: accept if the best detection is confident.
            if max(kept_confs) >= ACCEPT_BEST:
                target_root = ACCEPTED_ROOT
                site_stats["frames_accepted"] += 1
                accepted_rel = f"accepted/{slug}/{frame_path.stem}.txt"
                spot_check_candidates.setdefault(slug, []).append(accepted_rel)
            else:
                target_root = FLAGGED_ROOT
                site_stats["frames_flagged"] += 1

            label_path = target_root / slug / f"{frame_path.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")

        log(
            f"         {slug}: accepted={site_stats['frames_accepted']} "
            f"flagged={site_stats['frames_flagged']} "
            f"empty={site_stats['frames_empty']}"
        )
        manifest_sites[slug] = site_stats
        _accumulate(totals, site_stats)

    # Stratified random spot-check sample (per site) of high-confidence accepted frames.
    rng = random.Random(SPOT_CHECK_SEED)
    spot_check_sample: dict[str, list[str]] = {}
    for slug, candidates in spot_check_candidates.items():
        n = min(SPOT_CHECK_PER_SITE, len(candidates))
        spot_check_sample[slug] = sorted(rng.sample(candidates, n))
    spot_check_total = sum(len(v) for v in spot_check_sample.values())

    write_manifest(
        MANIFEST_PATH,
        {
            "stage": "auto_label",
            "teacher": args.teacher,
            "conf_low": CONF_LOW,
            "accept_best": ACCEPT_BEST,
            "class_names": PROJECT_NAMES,
            "conf_histogram_buckets": [
                "0.3-0.4",
                "0.4-0.5",
                "0.5-0.6",
                "0.6-0.7",
                "0.7-0.8",
                "0.8-0.9",
                "0.9-1.0",
            ],
            "review_queues": {
                "spotcheck": {
                    "path": REVIEW_QUEUE_SPOT_PATH.name,
                    "count": spot_check_total,
                    "per_site": SPOT_CHECK_PER_SITE,
                    "seed": SPOT_CHECK_SEED,
                    "sampling": "uniform random across all accepted frames per site",
                },
            },
            "total": totals,
            "sites": manifest_sites,
        },
    )
    REVIEW_QUEUE_SPOT_PATH.write_text(
        json.dumps(
            {
                "description": (
                    f"Uniform random sample (per site, n={SPOT_CHECK_PER_SITE}) "
                    "of accepted frames. Not biased toward high confidence — "
                    "borderline-accept frames are equally likely to surface, "
                    "since those are also error candidates. General sanity "
                    "check on the accept bin: catches false positives on road "
                    "signs/billboards AND threshold-borderline misclassifications."
                ),
                "per_site": SPOT_CHECK_PER_SITE,
                "seed": SPOT_CHECK_SEED,
                "count": spot_check_total,
                "by_site": spot_check_sample,
            },
            indent=2,
        )
        + "\n"
    )
    log(
        f"Done. Manifest: {MANIFEST_PATH}  "
        f"Spot-check sample: {spot_check_total} frames"
    )


def _accumulate(dst: dict, src: dict) -> None:
    for key in (
        "frames_processed",
        "frames_accepted",
        "frames_flagged",
        "frames_empty",
    ):
        dst[key] += src.get(key, 0)
    for cls_name, count in src["class_counts"].items():
        dst["class_counts"][cls_name] += count
    for i, c in enumerate(src["conf_histogram"]):
        dst["conf_histogram"][i] += c


def recount_site(slug: str, site_total: int) -> dict:
    """Build site stats from existing label files (used on full skip)."""
    accepted = sum(1 for _ in (ACCEPTED_ROOT / slug).glob("*.txt"))
    flagged = sum(1 for _ in (FLAGGED_ROOT / slug).glob("*.txt"))
    empty = max(0, site_total - accepted - flagged)
    class_counts = {name: 0 for name in PROJECT_NAMES.values()}
    hist = empty_histogram()
    for txt_file in list((ACCEPTED_ROOT / slug).glob("*.txt")) + list(
        (FLAGGED_ROOT / slug).glob("*.txt")
    ):
        for line in txt_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cid = int(parts[0])
            class_counts[PROJECT_NAMES.get(cid, "vehicle")] += 1
    return {
        "frames_processed": site_total,
        "frames_accepted": accepted,
        "frames_flagged": flagged,
        "frames_empty": empty,
        "class_counts": class_counts,
        "conf_histogram": hist,  # not reconstructable without confs
    }


if __name__ == "__main__":
    main()
