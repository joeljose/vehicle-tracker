"""Tests for source_resolver — YouTube URL detection and resolution."""

import asyncio
import subprocess
from unittest.mock import patch

import pytest

from backend.pipeline.source_resolver import (
    ResolvedSource,
    is_youtube_url,
    resolve_source,
    check_stream_liveness,
    re_extract_url,
)


class TestIsYoutubeUrl:
    def test_youtube_watch(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=abc123")

    def test_youtube_watch_no_www(self):
        assert is_youtube_url("https://youtube.com/watch?v=abc123")

    def test_youtube_live(self):
        assert is_youtube_url("https://www.youtube.com/live/abc123")

    def test_youtu_be(self):
        assert is_youtube_url("https://youtu.be/abc123")

    def test_file_path(self):
        assert not is_youtube_url("/data/video.mp4")

    def test_other_url(self):
        assert not is_youtube_url("https://example.com/video.mp4")

    def test_rtsp(self):
        assert not is_youtube_url("rtsp://camera.local/stream")

    def test_http_youtube(self):
        assert is_youtube_url("http://youtube.com/watch?v=abc")


class TestResolveSource:
    def test_file_source_passthrough(self):
        result = asyncio.run(resolve_source("/data/video.mp4"))
        assert result == ResolvedSource(
            source_type="file",
            stream_url="/data/video.mp4",
            original_url="/data/video.mp4",
        )

    def test_youtube_url_resolved(self):
        fake_hls = "https://manifest.googlevideo.com/api/manifest/hls_playlist/test"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_hls + "\n", stderr=""
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            result = asyncio.run(
                resolve_source("https://www.youtube.com/watch?v=abc123")
            )
        assert result.source_type == "youtube_live"
        assert result.stream_url == fake_hls
        assert result.original_url == "https://www.youtube.com/watch?v=abc123"

    def test_youtube_url_ytdlp_fails(self):
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ERROR: Video unavailable"
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            with pytest.raises(ValueError, match="yt-dlp failed"):
                asyncio.run(resolve_source("https://www.youtube.com/watch?v=bad"))

    def test_youtube_url_ytdlp_timeout(self):
        def timeout_side_effect(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30)

        with patch(
            "backend.pipeline.source_resolver._run_ytdlp",
            side_effect=timeout_side_effect,
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                asyncio.run(resolve_source("https://www.youtube.com/watch?v=slow"))

    def test_youtube_url_empty_result(self):
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n", stderr=""
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            with pytest.raises(ValueError, match="empty URL"):
                asyncio.run(resolve_source("https://www.youtube.com/watch?v=empty"))


class TestCheckStreamLiveness:
    def test_stream_is_live(self):
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"is_live": true, "title": "test"}',
            stderr="",
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            result = asyncio.run(
                check_stream_liveness("https://www.youtube.com/watch?v=live")
            )
        assert result is True

    def test_stream_ended(self):
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"is_live": false, "title": "ended"}',
            stderr="",
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            result = asyncio.run(
                check_stream_liveness("https://www.youtube.com/watch?v=ended")
            )
        assert result is False

    def test_ytdlp_fails_returns_false(self):
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ERROR"
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            result = asyncio.run(
                check_stream_liveness("https://www.youtube.com/watch?v=bad")
            )
        assert result is False

    def test_timeout_returns_false(self):
        def timeout_side_effect(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30)

        with patch(
            "backend.pipeline.source_resolver._run_ytdlp",
            side_effect=timeout_side_effect,
        ):
            result = asyncio.run(
                check_stream_liveness("https://www.youtube.com/watch?v=slow")
            )
        assert result is False


class TestReExtractUrl:
    def test_returns_fresh_url(self):
        fresh_hls = "https://manifest.googlevideo.com/fresh"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fresh_hls + "\n", stderr=""
        )
        with patch(
            "backend.pipeline.source_resolver._run_ytdlp", return_value=mock_result
        ):
            result = asyncio.run(
                re_extract_url("https://www.youtube.com/watch?v=abc123")
            )
        assert result == fresh_hls


class TestYtdlpSerialization:
    def test_concurrent_calls_are_serialized(self):
        """Verify that concurrent resolve calls don't run yt-dlp in parallel."""
        import time

        call_times = []

        def slow_ytdlp(*args, **kwargs):
            call_times.append(("start", time.monotonic()))
            time.sleep(0.1)
            call_times.append(("end", time.monotonic()))
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="https://manifest.googlevideo.com/test\n",
                stderr="",
            )

        async def run_two():
            with patch(
                "backend.pipeline.source_resolver._run_ytdlp", side_effect=slow_ytdlp
            ):
                await asyncio.gather(
                    resolve_source("https://www.youtube.com/watch?v=a"),
                    resolve_source("https://www.youtube.com/watch?v=b"),
                )

        asyncio.run(run_two())
        # With serialization: call1 ends before call2 starts
        assert len(call_times) == 4
        # First end should be before second start
        first_end = call_times[1][1]
        second_start = call_times[2][1]
        assert second_start >= first_end, "yt-dlp calls should be serialized"
