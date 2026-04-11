#!/usr/bin/env python3
"""Sanity check: load each teacher and run inference on a single frame.

Used after a Docker rebuild to confirm all wrappers + dependencies work
before running the full bake-off. Picks one frame from
data/frames_dedup/junction_741_73/ and reports detection counts per teacher.

Usage:
    python3 scripts/smoke_test_teachers.py [--teacher NAME] [--frame STEM]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402
from teachers import REGISTRY, get_teacher  # noqa: E402

DEFAULT_FRAME = "frame_004283"  # the gantry-fragmentation frame
DEFAULT_SITE = "junction_741_73"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=None, help="One specific teacher to test")
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    args = parser.parse_args()

    image_path = REPO_ROOT / "data" / "frames_dedup" / DEFAULT_SITE / f"{args.frame}.jpg"
    if not image_path.exists():
        log(f"ERROR: image not found: {image_path}")
        sys.exit(1)
    log(f"Test frame: {image_path}")

    teachers_to_test = [args.teacher] if args.teacher else list(REGISTRY)

    failures = []
    for name in teachers_to_test:
        log("=" * 60)
        log(f"Testing teacher: {name}")
        try:
            t0 = time.time()
            teacher = get_teacher(name)
            load_t = time.time() - t0
            log(f"  loaded in {load_t:.1f}s")

            t0 = time.time()
            results = teacher.predict([image_path], conf_threshold=0.3)
            infer_t = time.time() - t0
            log(f"  inference in {infer_t * 1000:.0f}ms")

            if not results:
                log("  WARNING: no results returned")
                failures.append((name, "empty results"))
                continue
            det = results[0]
            n = len(det.boxes_xywhn)
            log(f"  image_hw: {det.image_hw}")
            log(f"  detections: {n}")
            if n:
                from collections import Counter
                from teachers._base import PROJECT_NAMES
                cls_count = Counter(int(c) for c in det.project_class_ids)
                breakdown = ", ".join(
                    f"{PROJECT_NAMES.get(c, str(c))}={k}" for c, k in sorted(cls_count.items())
                )
                log(f"  class breakdown: {breakdown}")
                log(f"  conf range: {det.confs.min():.2f} .. {det.confs.max():.2f}")
            log(f"  PASS")
        except Exception as e:
            log(f"  FAIL: {type(e).__name__}: {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))

    log("=" * 60)
    if failures:
        log(f"FAILURES: {len(failures)}")
        for name, msg in failures:
            log(f"  {name}: {msg}")
        sys.exit(1)
    log(f"All {len(teachers_to_test)} teachers passed.")


if __name__ == "__main__":
    main()
