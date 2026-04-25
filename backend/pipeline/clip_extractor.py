"""Background clip extraction for transit alert replay.

Extracts short video clips per transit alert using ffmpeg (NVENC when available,
software fallback otherwise). Stagnant alerts get a single-frame JPEG with bbox overlay.

Cleanup is race-free: futures are tracked per channel so `cleanup_channel` can
drain in-flight ffmpeg jobs before `rmtree`-ing the output directory. Without
this, jobs writing to `/tmp/vt_clips/<id>/` would fail with ENOENT mid-write
when `/channel/remove` (or `/pipeline/stop`) deletes the directory.
"""

import logging
import shutil
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

logger = logging.getLogger(__name__)

CLIPS_DIR = Path("/tmp/vt_clips")
PADDING_BEFORE_S = 1.0  # seconds before first detection
PADDING_AFTER_S = 0.5   # seconds after last detection
# Bounded wait when draining in-flight ffmpeg on cleanup. Each ffmpeg call has
# its own 30s subprocess timeout, so the worst case is a single slow job; we
# don't want to block the API thread for that long.
CLEANUP_DRAIN_TIMEOUT_S = 5.0


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
        # Track in-flight futures per channel so cleanup_channel can drain.
        self._futures: dict[int, set[Future]] = defaultdict(set)

    def _submit(self, channel_id: int, fn, *args) -> None:
        """Submit a job to the pool and track its future for cleanup.

        Wraps every ffmpeg job in a try/finally that removes the future
        from `_futures` regardless of outcome.
        """
        with self._lock:
            futures = self._futures[channel_id]

        def runner():
            try:
                fn(*args)
            except Exception:
                logger.exception("Clip job crashed")

        fut: Future = self._pool.submit(runner)
        futures.add(fut)

        def _drop(_f, fset=futures):
            with self._lock:
                fset.discard(_f)

        fut.add_done_callback(_drop)

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
                self._submit(
                    channel_id,
                    self._extract_transit_clip,
                    alert_id,
                    source,
                    alert,
                    clip_dir,
                )
            elif alert_type == "stagnant_alert":
                with self._lock:
                    self._status[alert_id] = "pending"
                self._submit(
                    channel_id,
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
            self._submit(
                channel_id,
                self._extract_transit_clip,
                alert_id,
                source,
                alert,
                clip_dir,
            )
        elif alert_type == "stagnant_alert":
            with self._lock:
                self._status[alert_id] = "pending"
            self._submit(
                channel_id,
                self._extract_stagnant_frame,
                alert_id,
                source,
                alert,
                clip_dir,
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

        # Defensive mkdir: if cleanup_channel raced our submission and rmtree'd
        # the dir, this recreates it so ffmpeg has somewhere to write. Cleanup
        # of an orphaned clip in the recreated dir happens on the next
        # cleanup_all (e.g. /pipeline/stop or backend restart).
        clip_dir.mkdir(parents=True, exist_ok=True)

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
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    stderr_tail = (result.stderr or "").strip()[-300:]
                    if codec == "h264_nvenc":
                        logger.debug(
                            "NVENC unavailable, falling back to libx264: %s",
                            stderr_tail,
                        )
                        continue
                    logger.warning(
                        "Clip extraction failed for %s: %s", alert_id, stderr_tail,
                    )
                    continue
                # Reject corrupt files (e.g. 261-byte ftyp-only MP4)
                if output.exists() and output.stat().st_size > 1024:
                    with self._lock:
                        self._clip_paths[alert_id] = output
                        self._status[alert_id] = "ready"
                    logger.info("Clip ready: %s (%s)", alert_id, codec)
                    return
                logger.warning(
                    "Clip too small for %s (%d bytes), trying next codec",
                    alert_id,
                    output.stat().st_size if output.exists() else 0,
                )
                continue
            except FileNotFoundError:
                logger.warning(
                    "ffmpeg binary not found — clip extraction unavailable",
                )
                break
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

        # Defensive mkdir — see _extract_transit_clip for rationale.
        clip_dir.mkdir(parents=True, exist_ok=True)

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
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                stderr_tail = (result.stderr or "").strip()[-300:]
                logger.warning(
                    "Stagnant frame extraction failed for %s: %s",
                    alert_id,
                    stderr_tail,
                )
                with self._lock:
                    self._status[alert_id] = "failed"
                return
            if output.exists() and output.stat().st_size > 100:
                with self._lock:
                    self._clip_paths[alert_id] = output
                    self._status[alert_id] = "ready"
                logger.info("Stagnant frame ready: %s", alert_id)
            else:
                with self._lock:
                    self._status[alert_id] = "failed"
                logger.warning("Stagnant frame too small for %s", alert_id)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "Stagnant frame extraction failed for %s: %s", alert_id, exc,
            )
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
        """Drain in-flight ffmpeg jobs for this channel, then remove clips.

        The drain step is what prevents the "No such file or directory" race:
        without it, `shutil.rmtree` would delete `/tmp/vt_clips/<id>/` while
        ffmpeg is still writing to it.
        """
        with self._lock:
            futures = list(self._futures.get(channel_id, set()))
            self._futures.pop(channel_id, None)

        # Cancel queued jobs first (no-op for already-running ones), then
        # wait for in-flight ones to finish. Bounded so a stuck ffmpeg
        # doesn't hold the API thread forever.
        for f in futures:
            f.cancel()
        if futures:
            done, not_done = wait(futures, timeout=CLEANUP_DRAIN_TIMEOUT_S)
            if not_done:
                logger.warning(
                    "cleanup_channel(%d): %d clip job(s) still running after %.1fs; "
                    "rmtree may race them",
                    channel_id,
                    len(not_done),
                    CLEANUP_DRAIN_TIMEOUT_S,
                )

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
        """Drain all in-flight ffmpeg jobs across all channels, then wipe."""
        with self._lock:
            all_futures: list[Future] = []
            for fset in self._futures.values():
                all_futures.extend(fset)
            self._futures.clear()

        for f in all_futures:
            f.cancel()
        if all_futures:
            done, not_done = wait(all_futures, timeout=CLEANUP_DRAIN_TIMEOUT_S)
            if not_done:
                logger.warning(
                    "cleanup_all: %d clip job(s) still running after %.1fs; "
                    "rmtree may race them",
                    len(not_done),
                    CLEANUP_DRAIN_TIMEOUT_S,
                )

        if CLIPS_DIR.exists():
            shutil.rmtree(CLIPS_DIR, ignore_errors=True)
        with self._lock:
            self._status.clear()
            self._clip_paths.clear()

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=False)
