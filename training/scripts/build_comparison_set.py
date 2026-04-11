#!/usr/bin/env python3
"""Sample 1000 frames from the dedup pool into training/data/comparison/.

Per the M8-P1.5 v2 plan: all 734 741_73 frames (both the small pool and the
site where the fragmentation issue lives), plus 150 random Lytle South and
116 random Drugmart frames. Stages images as copies so downstream steps
(LS local-files storage, per-teacher inference) don't depend on the dedup
directory staying put.

Writes:
  data/comparison/plan.json
  data/comparison/images/{slug}/{stem}.jpg
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

# Make sibling module imports work when run as `python3 scripts/build_comparison_set.py`
sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402

FRAMES_DEDUP_ROOT = REPO_ROOT / "data" / "frames_dedup"
COMPARISON_ROOT = REPO_ROOT / "data" / "comparison"
COMPARISON_IMAGES = COMPARISON_ROOT / "images"
COMPARISON_PLAN = COMPARISON_ROOT / "plan.json"

# Deterministic per-site sample counts. junction_741_73 is small (734 total)
# AND the site with the known fragmentation issue, so we take all of it.
# Lytle + Drugmart are sampled randomly to hit 1000 total.
PER_SITE = {
    "junction_741_73": 734,             # all dedup frames
    "junction_741_lytle_south": 150,
    "junction_drugmart_73": 116,
}
TOTAL_EXPECTED = sum(PER_SITE.values())  # 1000

# Forced inclusions — the 3 known-fragmentation frames at 741_73.
# Since we're taking ALL 741_73 frames they'll be present anyway, but
# this also documents the intent for future maintainers.
FORCED_741_73 = ["frame_004283", "frame_011477", "frame_005976"]

SEED = 0


def build_plan() -> list[dict]:
    rng = random.Random(SEED)
    plan: list[dict] = []

    for slug, n in PER_SITE.items():
        site_dir = FRAMES_DEDUP_ROOT / slug
        if not site_dir.exists():
            log(f"  [warn] {slug}: dedup dir missing, skipping")
            continue
        all_stems = sorted(p.stem for p in site_dir.glob("frame_*.jpg"))
        if not all_stems:
            log(f"  [warn] {slug}: no frames found")
            continue

        if n >= len(all_stems):
            # Take everything (e.g. 741_73 where we sample 734/734)
            chosen = all_stems
            if n > len(all_stems):
                log(
                    f"  [note] {slug}: requested {n} but only {len(all_stems)} "
                    f"available; taking all"
                )
        elif slug == "junction_741_73":
            # Force the known fragmentation frames in, random-sample the rest
            forced = [s for s in FORCED_741_73 if s in all_stems]
            remaining = [s for s in all_stems if s not in forced]
            take = max(0, n - len(forced))
            chosen = forced + rng.sample(remaining, min(take, len(remaining)))
        else:
            chosen = rng.sample(all_stems, n)

        log(f"  {slug}: selected {len(chosen)} frames")
        for stem in chosen:
            plan.append({"site": slug, "stem": stem})

    return plan


def stage_images(plan: list[dict]) -> None:
    for entry in plan:
        slug = entry["site"]
        stem = entry["stem"]
        src = FRAMES_DEDUP_ROOT / slug / f"{stem}.jpg"
        dst = COMPARISON_IMAGES / slug / f"{stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def write_plan_manifest(plan: list[dict]) -> None:
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "per_site": PER_SITE,
        "total": len(plan),
        "forced_741_73": FORCED_741_73,
        "frames": [
            {
                "site": entry["site"],
                "stem": entry["stem"],
                "image": f"data/comparison/images/{entry['site']}/{entry['stem']}.jpg",
            }
            for entry in plan
        ],
    }
    COMPARISON_PLAN.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    log(f"Building comparison set (target total: {TOTAL_EXPECTED} frames)")
    plan = build_plan()
    log(f"Sampled {len(plan)} frames across {len(PER_SITE)} sites")
    log("Staging images into data/comparison/images/...")
    stage_images(plan)
    write_plan_manifest(plan)
    log(f"Done. Plan: {COMPARISON_PLAN}")


if __name__ == "__main__":
    main()
