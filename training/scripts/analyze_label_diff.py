#!/usr/bin/env python3
"""Diff two auto_label manifests and print a comparison table.

Used in Phase C to compare the chosen teacher's full re-label run against
the YOLOv8x baseline. Reports class-count deltas, accepted/flagged shifts,
and teacher metadata.

Usage:
    python3 scripts/analyze_label_diff.py \
        --baseline data/labels/manifest_yolov8x_baseline.json \
        --candidate data/labels/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_int(n: int) -> str:
    return f"{n:>8,}"


def fmt_delta(new: int, old: int) -> str:
    delta = new - old
    pct = (delta / max(old, 1)) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:>+7,}  ({sign}{pct:+.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())

    print(f"Baseline:  {baseline.get('teacher', '?')}  ({args.baseline})")
    print(f"Candidate: {candidate.get('teacher', '?')}  ({args.candidate})")
    print()

    b_total = baseline["total"]
    c_total = candidate["total"]

    print("OVERALL")
    print("-" * 60)
    fields = [
        ("frames_processed", "frames processed"),
        ("frames_accepted", "frames accepted"),
        ("frames_flagged", "frames flagged"),
        ("frames_empty", "frames empty"),
        ("frames_accepted_with_rare", "  with rare class"),
    ]
    for key, label in fields:
        b = b_total.get(key, 0)
        c = c_total.get(key, 0)
        print(f"  {label:<25}  baseline={fmt_int(b)}  candidate={fmt_int(c)}  {fmt_delta(c, b)}")
    print()

    print("CLASS COUNTS")
    print("-" * 60)
    for cls in ["car", "truck", "bus", "motorcycle"]:
        b = b_total.get("class_counts", {}).get(cls, 0)
        c = c_total.get("class_counts", {}).get(cls, 0)
        print(f"  {cls:<10}  baseline={fmt_int(b)}  candidate={fmt_int(c)}  {fmt_delta(c, b)}")
    print()

    print("PER SITE")
    print("-" * 60)
    for slug in baseline.get("sites", {}):
        b_site = baseline["sites"][slug]
        c_site = candidate.get("sites", {}).get(slug, {})
        print(f"  {slug}")
        for key, label in [
            ("frames_accepted", "    accepted"),
            ("frames_flagged", "    flagged"),
            ("frames_accepted_with_rare", "    rare"),
        ]:
            b = b_site.get(key, 0)
            c = c_site.get(key, 0)
            print(f"  {label:<14}  baseline={fmt_int(b)}  candidate={fmt_int(c)}  {fmt_delta(c, b)}")
        for cls in ["car", "truck", "bus", "motorcycle"]:
            b = b_site.get("class_counts", {}).get(cls, 0)
            c = c_site.get("class_counts", {}).get(cls, 0)
            print(f"      {cls:<10}  baseline={fmt_int(b)}  candidate={fmt_int(c)}  {fmt_delta(c, b)}")
    print()


if __name__ == "__main__":
    main()
