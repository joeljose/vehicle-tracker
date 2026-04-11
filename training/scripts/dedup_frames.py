#!/usr/bin/env python3
"""Per-site perceptual deduplication of extracted frames.

Reads `data/frames/{slug}/`, runs imagededup PHash and removes near-duplicate
frames. Survivors are copied to `data/frames_dedup/{slug}/`. Canonical pick
per duplicate cluster is the lex-smallest filename (union-find over the
duplicate map).

Why PHash, not CNN: MobileNetV3 features encode SEMANTIC content ("traffic
intersection at daytime"). Every frame from a fixed camera looks semantically
identical to a CNN, so cosine similarity at any sane threshold collapses the
entire dataset into one cluster. PHash works on pixel-DCT structure — when
cars enter/leave/move, the hash actually changes. This is what we want.

Threshold is a Hamming distance on the 64-bit PHash. Lower = stricter:
  0  : pixel-identical only
  3  : effectively-identical (only sub-pixel jitter changed)
  5  : near-identical, only minor things changed
  10 : imagededup default — too lax for our case (drops frames where only
       a small car has moved across the junction)

We use 5 as a reasonable starting point. Tune by reading the manifest:
- if reduction is too aggressive (<60%), lower the threshold (3, then 2)
- if reduction is too weak (>95%), raise it (7, then 10)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from _common import REPO_ROOT, load_video_sites, log, write_manifest

FRAMES_ROOT = REPO_ROOT / "data" / "frames"
DEDUP_ROOT = REPO_ROOT / "data" / "frames_dedup"
MANIFEST_PATH = DEDUP_ROOT / "manifest.json"

# PHash Hamming distance threshold (max distance to consider as duplicate).
# 64-bit hash, so values are 0..64. See module docstring for tuning notes.
PHASH_MAX_DISTANCE = 5


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep lex-smallest as the root so the canonical pick is deterministic
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def dedup_site(site_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Returns (before, after)."""
    from imagededup.methods import PHash  # heavy import — keep local

    frames = sorted(p.name for p in site_dir.glob("frame_*.jpg"))
    before = len(frames)
    if before == 0:
        return 0, 0

    phash = PHash(verbose=False)
    duplicates = phash.find_duplicates(
        image_dir=str(site_dir),
        max_distance_threshold=PHASH_MAX_DISTANCE,
        scores=False,
    )

    uf = UnionFind(frames)
    for img, dups in duplicates.items():
        for d in dups:
            uf.union(img, d)

    keepers = sorted({uf.find(f) for f in frames})

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in keepers:
        src = site_dir / name
        dst = out_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
    return before, len(keepers)


def main() -> None:
    sites = load_video_sites()
    DEDUP_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_sites: dict[str, dict] = {}
    total_before = 0
    total_after = 0

    for slug in sites:
        site_dir = FRAMES_ROOT / slug
        out_dir = DEDUP_ROOT / slug
        if not site_dir.exists():
            log(f"  [warn] {slug}: source dir missing, skipping")
            continue

        existing = sum(1 for _ in out_dir.glob("frame_*.jpg")) if out_dir.exists() else 0
        if existing > 0:
            before = sum(1 for _ in site_dir.glob("frame_*.jpg"))
            after = existing
            log(f"  [skip] {slug}: {after} dedup frames already present")
        else:
            log(f"  [run]  {slug}: deduping...")
            before, after = dedup_site(site_dir, out_dir)
            log(f"         {slug}: {before} -> {after} ({100*(1-after/before):.1f}% reduction)")

        reduction_pct = round(100 * (1 - after / before), 2) if before else 0.0
        manifest_sites[slug] = {
            "before": before,
            "after": after,
            "reduction_pct": reduction_pct,
        }
        total_before += before
        total_after += after

    overall_reduction = (
        round(100 * (1 - total_after / total_before), 2) if total_before else 0.0
    )
    write_manifest(
        MANIFEST_PATH,
        {
            "stage": "dedup_frames",
            "method": "imagededup.PHash",
            "max_distance_threshold": PHASH_MAX_DISTANCE,
            "total": {
                "before": total_before,
                "after": total_after,
                "reduction_pct": overall_reduction,
            },
            "sites": manifest_sites,
        },
    )
    log(
        f"Done. {total_before} -> {total_after} "
        f"({overall_reduction}% reduction). Manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
