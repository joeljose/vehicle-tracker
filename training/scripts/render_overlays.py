#!/usr/bin/env python3
"""Render bounding-box overlays for visual auto-label QA.

Picks N random labeled frames per site (from accepted+flagged), draws boxes
with class labels, and writes them to `data/inspect/{slug}/{stem}.jpg`.

Usage:
    python3 scripts/render_overlays.py --per-site 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from _common import REPO_ROOT, load_video_sites, log

DEDUP_ROOT = REPO_ROOT / "data" / "frames_dedup"
LABELS_ROOT = REPO_ROOT / "data" / "labels"
INSPECT_ROOT = REPO_ROOT / "data" / "inspect"

CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}
CLASS_COLORS = {
    0: (60, 200, 60),
    1: (60, 60, 220),
    2: (220, 160, 60),
    3: (220, 60, 220),
}


def find_labels(slug: str) -> list[Path]:
    accepted = list((LABELS_ROOT / "accepted" / slug).glob("*.txt"))
    flagged = list((LABELS_ROOT / "flagged" / slug).glob("*.txt"))
    return accepted + flagged


def draw_overlay(image_path: Path, label_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cid, str(cid))
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-site", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sites = load_video_sites()

    for slug in sites:
        labels = find_labels(slug)
        if not labels:
            log(f"  [skip] {slug}: no labels found")
            continue
        sample = rng.sample(labels, min(args.per_site, len(labels)))
        for label_path in sample:
            stem = label_path.stem
            image_path = DEDUP_ROOT / slug / f"{stem}.jpg"
            if not image_path.exists():
                continue
            out_path = INSPECT_ROOT / slug / f"{stem}.jpg"
            draw_overlay(image_path, label_path, out_path)
        log(f"  [done] {slug}: wrote {len(sample)} overlays to {INSPECT_ROOT / slug}")


if __name__ == "__main__":
    main()
