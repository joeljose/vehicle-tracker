"""Shared utilities used by both DeepStream and Custom pipeline adapters.

Extracts common logic to avoid duplication across backends:
- Snapshot serving (in-memory + disk fallback)
- Line crossing detection and alert building
- Lost track finalization and alert generation
- Entry/exit line parsing
- Snapshot directory cleanup
- Thread-safe callback dispatch
"""

import asyncio
import logging
import shutil
from pathlib import Path

import cv2

from backend.pipeline.direction import (
    LineSeg,
    check_line_crossing,
    classify_crossing,
)
from backend.pipeline.snapshot import BestPhotoTracker

logger = logging.getLogger(__name__)


# -- Snapshot serving --


def get_snapshot(
    states: dict,
    track_id: int,
    *,
    best_photo_attr: str = "best_photo",
) -> bytes | None:
    """Serve a snapshot JPEG for a track ID.

    Checks in-memory crops first (tracks still being processed),
    then falls back to disk (finalized tracks).

    Args:
        states: dict of channel_id → channel state objects.
        track_id: Track ID to look up.
        best_photo_attr: Attribute name for BestPhotoTracker on state.

    Returns:
        JPEG bytes or None.
    """
    # Try in-memory first (tracks still being processed)
    for state in states.values():
        best_photo = getattr(state, best_photo_attr, None)
        if best_photo:
            crop = best_photo.best_crops.get(track_id)
            if crop is not None:
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", bgr)
                return buf.tobytes()

    # Fall back to disk (finalized tracks)
    snapshots_root = Path("snapshots")
    for channel_dir in (
        snapshots_root.iterdir() if snapshots_root.exists() else []
    ):
        if channel_dir.is_dir():
            snapshot_path = channel_dir / f"{track_id}.jpg"
            if snapshot_path.exists():
                return snapshot_path.read_bytes()

    return None


# -- Line parsing --


def parse_lines(entry_exit_lines: dict) -> dict[str, LineSeg]:
    """Parse entry/exit line config (dict or LineSeg) to LineSeg objects.

    Accepts both raw dicts (from JSON config) and already-parsed LineSeg
    objects (idempotent).
    """
    if not entry_exit_lines:
        return {}
    lines = {}
    for arm_id, line_data in entry_exit_lines.items():
        if isinstance(line_data, LineSeg):
            lines[arm_id] = line_data
        elif isinstance(line_data, dict):
            lines[arm_id] = LineSeg(
                start=tuple(line_data["start"]),
                end=tuple(line_data["end"]),
                label=line_data.get("label", arm_id),
                junction_side=line_data.get("junction_side", "left"),
            )
    return lines


# -- Line crossing detection --


def check_crossings(
    *,
    track_id: int,
    prev: tuple[float, float],
    curr: tuple[float, float],
    lines: dict[str, LineSeg],
    active_tracks: dict[int, dict],
    roi_centroid: tuple[float, float] | None,
    frame_count: int,
    channel_id: int,
    alert_callback,
) -> None:
    """Check if a track crossed any entry/exit line and fire alerts.

    Args:
        track_id: Track ID.
        prev: Previous centroid (x, y).
        curr: Current centroid (x, y).
        lines: Parsed entry/exit lines.
        active_tracks: Dict of track_id → track state.
        roi_centroid: ROI polygon centroid for direction inference.
        frame_count: Current frame count.
        channel_id: Channel ID for alert metadata.
        alert_callback: Callable to invoke with alert dict.
    """
    for arm_id, line in lines.items():
        direction = check_line_crossing(line, prev, curr)
        if direction is None:
            continue

        crossing_type = classify_crossing(line, direction)
        ts = active_tracks.get(track_id)
        if ts is None:
            continue

        dsm = ts["dsm"]
        alert = dsm.on_crossing(
            arm_id,
            line.label,
            crossing_type,
            trajectory=ts["trajectory"].get_full(),
            arms=lines,
            roi_centroid=roi_centroid,
        )
        if alert:
            alert["track_id"] = track_id
            alert["frame"] = frame_count
            alert["label"] = ts["label"]
            alert["channel"] = channel_id
            alert["first_seen_frame"] = ts["first_frame"]
            alert["last_seen_frame"] = ts["last_frame"]
            alert["trajectory"] = ts["trajectory"].get_full()
            alert["per_frame_data"] = ts.get("per_frame_data", [])
            logger.info(
                "TRANSIT: Track #%d: %s -> %s (%s)",
                track_id,
                alert["entry_label"],
                alert["exit_label"],
                alert["method"],
            )
            alert_callback(alert)


# -- Lost track finalization --


def finalize_lost_track(
    *,
    track_id: int,
    stitcher_state: dict,
    lines: dict[str, LineSeg],
    roi_centroid: tuple[float, float] | None,
    frame_count: int,
    channel_id: int,
    best_photo: BestPhotoTracker | None,
    snapshot_dir: str,
    alert_callback,
    track_ended_callback,
) -> None:
    """Finalize a track that expired from the stitcher.

    Saves best photo, performs direction inference, and fires callbacks.

    Args:
        track_id: Track ID.
        stitcher_state: Stitcher state dict for this track.
        lines: Parsed entry/exit lines.
        roi_centroid: ROI polygon centroid.
        frame_count: Current frame count.
        channel_id: Channel ID.
        best_photo: BestPhotoTracker instance (or None).
        snapshot_dir: Directory to save snapshot JPEGs.
        alert_callback: Callable for transit alerts.
        track_ended_callback: Callable for track ended events.
    """
    dsm = stitcher_state["dsm"]
    trajectory = stitcher_state["trajectory"]
    trajectory_full = trajectory.get_full()
    label = stitcher_state.get("label", "unknown")
    lifetime = stitcher_state.get("lifetime", 0)

    # Save best photo
    if best_photo:
        best_photo.save(track_id, snapshot_dir)

    # Direction inference
    if lines:
        alert = dsm.on_track_lost(
            trajectory_full, lines, roi_centroid=roi_centroid,
        )
        if alert:
            alert["track_id"] = track_id
            alert["frame"] = frame_count
            alert["label"] = label
            alert["channel"] = channel_id
            alert["first_seen_frame"] = stitcher_state.get("first_frame", 0)
            alert["last_seen_frame"] = frame_count
            alert["trajectory"] = trajectory_full
            alert["per_frame_data"] = stitcher_state.get("per_frame_data", [])
            logger.info(
                "TRANSIT: Track #%d: %s -> %s (%s)",
                track_id,
                alert["entry_label"],
                alert["exit_label"],
                alert["method"],
            )
            alert_callback(alert)

    lost = {
        "track_id": track_id,
        "label": label,
        "lifetime": lifetime,
        "trajectory": trajectory_full,
        "roi_active": True,
        "direction_state": dsm.state.value,
    }
    track_ended_callback(lost)


# -- Cleanup --


def cleanup_channel_snapshots(channel_id: int) -> None:
    """Remove snapshot directory for a channel."""
    snapshot_dir = Path("snapshots") / str(channel_id)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        logger.info("Cleaned up snapshots for channel %d", channel_id)


# -- Thread-safe callbacks --


def safe_callback(
    callback,
    *args,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Dispatch callback from a background thread to the asyncio event loop.

    Falls back to direct invocation if no loop is running.
    """
    if callback is None:
        return
    try:
        if loop and loop.is_running():
            loop.call_soon_threadsafe(callback, *args)
        else:
            callback(*args)
    except Exception as e:
        logger.error("Callback error: %s", e)
