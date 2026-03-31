"""Source resolver — YouTube Live URL resolution via yt-dlp.

Resolves user-provided source strings to pipeline-consumable URLs.
File paths pass through as-is. YouTube URLs are resolved to HLS stream
URLs via yt-dlp subprocess calls.

All yt-dlp calls are serialized through a single asyncio.Lock to avoid
hitting YouTube rate limits when multiple channels resolve concurrently.
"""

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

# YouTube URL patterns
_YOUTUBE_PATTERNS = [
    re.compile(r"https?://(www\.)?youtube\.com/watch\?v="),
    re.compile(r"https?://(www\.)?youtube\.com/live/"),
    re.compile(r"https?://youtu\.be/"),
]

# Single lock for all yt-dlp subprocess calls
_ytdlp_lock = asyncio.Lock()

_YTDLP_TIMEOUT = 30  # seconds


@dataclass
class ResolvedSource:
    """Result of resolving a user-provided source string."""

    source_type: Literal["file", "youtube_live"]
    stream_url: str  # file path or resolved HLS URL
    original_url: str  # what the user provided


def is_youtube_url(source: str) -> bool:
    """Check if a source string is a YouTube URL."""
    return any(p.match(source) for p in _YOUTUBE_PATTERNS)


def _run_ytdlp(
    *args: str, timeout: int = _YTDLP_TIMEOUT
) -> subprocess.CompletedProcess:
    """Run a yt-dlp subprocess. Must be called while holding _ytdlp_lock."""
    cmd = ["yt-dlp", *args]
    log.debug("yt-dlp: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


async def resolve_source(source: str) -> ResolvedSource:
    """Resolve a user-provided source to a pipeline-consumable URL.

    For file paths: returns as-is (no validation here — caller validates).
    For YouTube URLs: extracts best-quality HLS URL via yt-dlp.

    Raises:
        ValueError: If yt-dlp fails to resolve the URL.
        TimeoutError: If yt-dlp takes too long.
    """
    if not is_youtube_url(source):
        return ResolvedSource(
            source_type="file",
            stream_url=source,
            original_url=source,
        )

    async with _ytdlp_lock:
        try:
            result = await asyncio.to_thread(
                _run_ytdlp,
                "-f",
                "bv*+ba/b",
                "--get-url",
                source,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"yt-dlp timed out resolving {source}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ValueError(f"yt-dlp failed: {stderr}")

    hls_url = result.stdout.strip()
    if not hls_url:
        raise ValueError("yt-dlp returned empty URL")

    log.info("Resolved YouTube URL to HLS: %s -> %s...", source, hls_url[:80])
    return ResolvedSource(
        source_type="youtube_live",
        stream_url=hls_url,
        original_url=source,
    )


async def check_stream_liveness(original_url: str) -> bool:
    """Check if a YouTube Live stream is still broadcasting.

    Used during stream recovery to distinguish 'stream ended' from
    'URL expired / network issue'.
    """
    async with _ytdlp_lock:
        try:
            result = await asyncio.to_thread(
                _run_ytdlp,
                "--dump-json",
                "--no-download",
                original_url,
            )
        except subprocess.TimeoutExpired:
            log.warning("yt-dlp timed out checking liveness for %s", original_url)
            return False

    if result.returncode != 0:
        log.warning("yt-dlp failed liveness check: %s", result.stderr.strip())
        return False

    try:
        info = json.loads(result.stdout)
        return bool(info.get("is_live"))
    except (json.JSONDecodeError, KeyError):
        return False


async def re_extract_url(original_url: str) -> str:
    """Re-extract a fresh HLS URL from a YouTube Live URL.

    Used during stream recovery when the existing HLS URL has expired.

    Raises:
        ValueError: If re-extraction fails.
    """
    resolved = await resolve_source(original_url)
    return resolved.stream_url
