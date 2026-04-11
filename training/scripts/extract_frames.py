#!/usr/bin/env python3
"""Extract 1 FPS JPEG frames from site videos.

Reads `configs/video_sites.yaml`, runs ffmpeg per video, writes frames to
`data/frames/{site_slug}/frame_{N:06d}.jpg`. Resumable: a per-site
`.extracted_videos.json` records which videos have completed extraction so
re-runs skip them and only process new/interrupted videos.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from _common import (
    REPO_ROOT,
    iter_site_frames,  # noqa: F401  (kept for downstream scripts re-importing)
    load_video_sites,
    log,
    write_manifest,
)

FRAMES_ROOT = REPO_ROOT / "data" / "frames"
MANIFEST_PATH = FRAMES_ROOT / "manifest.json"


def count_frames(site_dir: Path) -> int:
    if not site_dir.exists():
        return 0
    return sum(1 for _ in site_dir.glob("frame_*.jpg"))


def video_duration_seconds(video: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(float(result.stdout.strip()))


def load_state(site_dir: Path) -> dict:
    state_path = site_dir / ".extracted_videos.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"videos": []}


def save_state(site_dir: Path, state: dict) -> None:
    state_path = site_dir / ".extracted_videos.json"
    state_path.write_text(json.dumps(state, indent=2))


def extract_one_video(video: Path, site_dir: Path, start_offset: int) -> None:
    """Run ffmpeg fps=1 extraction with start_number offset (1-indexed)."""
    site_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video),
        "-vf",
        "fps=1",
        "-q:v",
        "2",
        "-start_number",
        str(start_offset),
        str(site_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    sites = load_video_sites()
    log(f"Loaded {len(sites)} sites from video_sites.yaml")

    manifest_sites: dict[str, dict] = {}
    total_frames = 0

    for slug, site in sites.items():
        site_dir = FRAMES_ROOT / slug
        site_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(site_dir)
        done_set = set(state["videos"])

        next_offset = count_frames(site_dir) + 1
        new_count = 0
        for video in site.videos:
            if video.name in done_set:
                continue
            log(f"  [extract] {slug} <- {video.name}")
            extract_one_video(video, site_dir, next_offset)
            done_set.add(video.name)
            state["videos"] = sorted(done_set)
            save_state(site_dir, state)
            new_count += 1
            next_offset = count_frames(site_dir) + 1

        if new_count == 0:
            log(f"  [skip]    {slug}: all {len(site.videos)} videos already extracted")

        frame_count = count_frames(site_dir)
        expected = sum(video_duration_seconds(v) for v in site.videos)
        manifest_sites[slug] = {
            "display_name": site.display_name,
            "videos": [v.name for v in site.videos],
            "expected_frame_count": expected,
            "frame_count": frame_count,
        }
        total_frames += frame_count

    write_manifest(
        MANIFEST_PATH,
        {
            "stage": "extract_frames",
            "fps": 1,
            "total_frames": total_frames,
            "sites": manifest_sites,
        },
    )
    log(f"Done. Total frames: {total_frames}. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
