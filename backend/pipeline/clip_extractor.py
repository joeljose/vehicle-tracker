"""Background clip extraction for transit alert replay.

Extracts short video clips per transit alert using ffmpeg (NVENC when available,
software fallback otherwise). Stagnant alerts get a single-frame JPEG with bbox overlay.
"""

import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

CLIPS_DIR = Path("/tmp/vt_clips")
PADDING_BEFORE_S = 1.0  # seconds before first detection
PADDING_AFTER_S = 0.5   # seconds after last detection


class ClipExtractor:
    """Manages clip extraction for replay.

    Thread-safe: extraction runs in a background thread pool.
    Status per alert_id: "pending", "ready", "failed".
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps
        self._lock = threading.Lock()
        self._status: dict[str, str] = {}  # alert_id -> status
        self._clip_paths: dict[str, Path] = {}  # alert_id -> file path
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clip")

    def extract_clips(
        self,
        channel_id: int,
        source: str,
        alerts: list[dict],
    ) -> None:
        """Start background extraction for all alerts in a channel.

        Args:
            channel_id: Channel ID (used for temp directory).
            source: Video source file path.
            alerts: List of full alert dicts from AlertStore.
        """
        clip_dir = CLIPS_DIR / str(channel_id)
        clip_dir.mkdir(parents=True, exist_ok=True)

        for alert in alerts:
            alert_id = alert["alert_id"]
            alert_type = alert["type"]

            if alert_type == "transit_alert":
                with self._lock:
                    self._status[alert_id] = "pending"
                self._pool.submit(
                    self._extract_transit_clip,
                    alert_id,
                    source,
                    alert,
                    clip_dir,
                )
            elif alert_type == "stagnant_alert":
                with self._lock:
                    self._status[alert_id] = "pending"
                self._pool.submit(
                    self._extract_stagnant_frame,
                    alert_id,
                    source,
                    alert,
                    clip_dir,
                )

    def extract_single(self, channel_id: int, source: str, alert: dict) -> None:
        """Extract clip for a single alert (for late arrivals during phase transition).

        Args:
            channel_id: Channel ID (used for temp directory).
            source: Video source file path.
            alert: Full alert dict from AlertStore.
        """
        clip_dir = CLIPS_DIR / str(channel_id)
        clip_dir.mkdir(parents=True, exist_ok=True)

        alert_id = alert["alert_id"]
        alert_type = alert["type"]

        if alert_type == "transit_alert":
            with self._lock:
                self._status[alert_id] = "pending"
            self._pool.submit(
                self._extract_transit_clip, alert_id, source, alert, clip_dir
            )
        elif alert_type == "stagnant_alert":
            with self._lock:
                self._status[alert_id] = "pending"
            self._pool.submit(
                self._extract_stagnant_frame, alert_id, source, alert, clip_dir
            )

    def _extract_transit_clip(
        self,
        alert_id: str,
        source: str,
        alert: dict,
        clip_dir: Path,
    ) -> None:
        """Extract a short clip for a transit alert using ffmpeg."""
        # Use per_frame_data timestamps (ms) — fps-agnostic
        pfd = alert.get("per_frame_data", [])
        if pfd:
            first_ms = pfd[0].get("timestamp_ms", 0)
            last_ms = pfd[-1].get("timestamp_ms", 0)
        else:
            first_ms = alert.get("first_seen_frame", 0) / self._fps * 1000
            last_ms = alert.get("last_seen_frame", 0) / self._fps * 1000

        start_s = max(0, first_ms / 1000 - PADDING_BEFORE_S)
        end_s = last_ms / 1000 + PADDING_AFTER_S

        output = clip_dir / f"clip_{alert_id}.mp4"

        # Try NVENC first, fall back to libx264
        for codec in ("h264_nvenc", "libx264"):
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_s:.3f}",
                "-to",
                f"{end_s:.3f}",
                "-i",
                source,
                "-c:v",
                codec,
                "-an",  # no audio
                "-loglevel",
                "error",
                str(output),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=30)
                # Reject corrupt files (e.g. 261-byte ftyp-only MP4)
                if output.exists() and output.stat().st_size > 1024:
                    with self._lock:
                        self._clip_paths[alert_id] = output
                        self._status[alert_id] = "ready"
                    logger.info("Clip ready: %s (%s)", alert_id, codec)
                    return
                else:
                    logger.warning(
                        "Clip too small for %s (%d bytes), trying next codec",
                        alert_id,
                        output.stat().st_size if output.exists() else 0,
                    )
                    continue
            except (subprocess.CalledProcessError, FileNotFoundError):
                if codec == "h264_nvenc":
                    logger.debug("NVENC unavailable, falling back to libx264")
                    continue
                logger.warning("Clip extraction failed for %s", alert_id)
            except subprocess.TimeoutExpired:
                logger.warning("Clip extraction timed out for %s", alert_id)

        with self._lock:
            self._status[alert_id] = "failed"

    def _extract_stagnant_frame(
        self,
        alert_id: str,
        source: str,
        alert: dict,
        clip_dir: Path,
    ) -> None:
        """Extract a single frame for a stagnant alert using ffmpeg."""
        pfd = alert.get("per_frame_data", [])
        if pfd:
            first_ms = pfd[0].get("timestamp_ms", 0)
            last_ms = pfd[-1].get("timestamp_ms", 0)
            timestamp_s = (first_ms + last_ms) / 2 / 1000
        else:
            first_frame = alert.get("first_seen_frame", 0)
            last_frame = alert.get("last_seen_frame", first_frame)
            timestamp_s = (first_frame + last_frame) / 2 / self._fps

        output = clip_dir / f"frame_{alert_id}.jpg"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_s:.3f}",
            "-i",
            source,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-loglevel",
            "error",
            str(output),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=10)
            if output.exists() and output.stat().st_size > 100:
                with self._lock:
                    self._clip_paths[alert_id] = output
                    self._status[alert_id] = "ready"
                logger.info("Stagnant frame ready: %s", alert_id)
            else:
                with self._lock:
                    self._status[alert_id] = "failed"
                logger.warning("Stagnant frame too small for %s", alert_id)
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.warning("Stagnant frame extraction failed for %s", alert_id)
            with self._lock:
                self._status[alert_id] = "failed"

    def get_status(self, alert_id: str) -> str | None:
        """Get extraction status: 'pending', 'ready', 'failed', or None."""
        with self._lock:
            return self._status.get(alert_id)

    def get_clip_path(self, alert_id: str) -> Path | None:
        """Get path to extracted clip/frame, or None if not ready."""
        with self._lock:
            if self._status.get(alert_id) != "ready":
                return None
            return self._clip_paths.get(alert_id)

    def cleanup_channel(self, channel_id: int) -> None:
        """Remove all clips for a channel and clear status tracking."""
        clip_dir = CLIPS_DIR / str(channel_id)
        if clip_dir.exists():
            shutil.rmtree(clip_dir, ignore_errors=True)
            logger.info("Cleaned up clips for channel %d", channel_id)

        # Clear status entries for this channel's alerts
        # (caller should clear alert store separately)
        with self._lock:
            to_remove = [
                aid
                for aid, path in self._clip_paths.items()
                if path.parent.name == str(channel_id)
            ]
            for aid in to_remove:
                self._status.pop(aid, None)
                self._clip_paths.pop(aid, None)

    def cleanup_all(self) -> None:
        """Remove all clips and reset state."""
        if CLIPS_DIR.exists():
            shutil.rmtree(CLIPS_DIR, ignore_errors=True)
        with self._lock:
            self._status.clear()
            self._clip_paths.clear()

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=False)
