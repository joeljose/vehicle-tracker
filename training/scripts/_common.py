"""Shared helpers for training pipeline scripts."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path("/app")  # WORKDIR inside the training container
DEFAULT_VIDEO_SITES_YAML = REPO_ROOT / "configs" / "video_sites.yaml"
DEFAULT_VIDEO_SOURCE_DIR = Path("/data/site_videos")


@dataclass(frozen=True)
class SiteEntry:
    slug: str
    display_name: str
    videos: list[Path]


def log(msg: str) -> None:
    """Single-line stderr logger with UTC timestamp."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def load_video_sites(
    yaml_path: Path = DEFAULT_VIDEO_SITES_YAML,
    video_source_dir: Path = DEFAULT_VIDEO_SOURCE_DIR,
) -> dict[str, SiteEntry]:
    """Parse configs/video_sites.yaml and resolve video paths.

    Raises if any referenced video file is missing.
    """
    with yaml_path.open("r") as f:
        raw = yaml.safe_load(f)
    sites: dict[str, SiteEntry] = {}
    missing: list[str] = []
    for slug, entry in raw["sites"].items():
        videos: list[Path] = []
        for filename in entry["videos"]:
            p = video_source_dir / filename
            if not p.exists():
                missing.append(f"{slug}: {filename}")
            videos.append(p)
        sites[slug] = SiteEntry(
            slug=slug,
            display_name=entry["display_name"],
            videos=videos,
        )
    if missing:
        raise FileNotFoundError(
            "Missing source videos referenced in video_sites.yaml:\n  "
            + "\n  ".join(missing)
        )
    return sites


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Write a pretty-printed JSON manifest, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **data,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def iter_site_frames(frames_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (site_slug, frame_path) for every JPG under frames_root/<slug>/."""
    if not frames_root.exists():
        return
    for site_dir in sorted(frames_root.iterdir()):
        if not site_dir.is_dir():
            continue
        for frame in sorted(site_dir.glob("*.jpg")):
            yield site_dir.name, frame
