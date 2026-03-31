"""Integration tests for YouTube Live channel add (async resolution flow)."""

from unittest.mock import patch, AsyncMock


class TestYoutubeChannelAdd:
    """Test the async YouTube URL resolution flow via POST /channel/add."""

    def test_youtube_url_returns_202(self, client):
        """YouTube URLs should return 202 with a request_id."""
        client.post("/pipeline/start")
        resp = client.post(
            "/channel/add",
            json={"source": "https://www.youtube.com/watch?v=abc123"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "resolving"
        assert "request_id" in data

    def test_file_source_still_returns_200(self, client):
        """File sources should continue to work synchronously (200)."""
        client.post("/pipeline/start")
        resp = client.post(
            "/channel/add",
            json={"source": "/data/video.mp4"},
        )
        assert resp.status_code == 200
        assert "channel_id" in resp.json()

    def test_youtube_url_variants_return_202(self, client):
        """All YouTube URL patterns should trigger async resolution."""
        client.post("/pipeline/start")
        urls = [
            "https://www.youtube.com/watch?v=abc",
            "https://youtube.com/live/abc",
            "https://youtu.be/abc",
        ]
        for url in urls:
            resp = client.post("/channel/add", json={"source": url})
            assert resp.status_code == 202, f"Expected 202 for {url}"

    def test_pipeline_not_started_rejects_youtube(self, client):
        """YouTube URLs should be rejected if pipeline not started."""
        resp = client.post(
            "/channel/add",
            json={"source": "https://www.youtube.com/watch?v=abc"},
        )
        assert resp.status_code == 409


class TestYoutubeResolutionBackground:
    """Test the background resolution task and WS notifications."""

    def test_successful_resolution_adds_channel(self, app, client):
        """After successful resolution, channel should exist in backend."""
        from backend.pipeline.source_resolver import ResolvedSource

        fake_resolved = ResolvedSource(
            source_type="youtube_live",
            stream_url="https://manifest.googlevideo.com/hls",
            original_url="https://www.youtube.com/watch?v=abc123",
        )

        client.post("/pipeline/start")

        with patch(
            "backend.api.routes.resolve_source",
            new_callable=AsyncMock,
            return_value=fake_resolved,
        ):
            resp = client.post(
                "/channel/add",
                json={"source": "https://www.youtube.com/watch?v=abc123"},
            )
            assert resp.status_code == 202

        # Background task runs synchronously in TestClient
        backend = app.state.backend
        assert 0 in backend.channels
        assert backend.channels[0] == "https://manifest.googlevideo.com/hls"

    def test_failed_resolution_no_channel(self, app, client):
        """Failed resolution should not create a channel."""
        client.post("/pipeline/start")

        with patch(
            "backend.api.routes.resolve_source",
            new_callable=AsyncMock,
            side_effect=ValueError("yt-dlp failed: Video unavailable"),
        ):
            resp = client.post(
                "/channel/add",
                json={"source": "https://www.youtube.com/watch?v=bad"},
            )
            assert resp.status_code == 202

        backend = app.state.backend
        assert len(backend.channels) == 0

    def test_ws_notification_on_success(self, app, client):
        """Successful resolution should enqueue channel_added WS event."""
        from backend.pipeline.source_resolver import ResolvedSource

        fake_resolved = ResolvedSource(
            source_type="youtube_live",
            stream_url="https://manifest.googlevideo.com/hls",
            original_url="https://www.youtube.com/watch?v=abc123",
        )

        client.post("/pipeline/start")
        ws = app.state.ws

        with patch(
            "backend.api.routes.resolve_source",
            new_callable=AsyncMock,
            return_value=fake_resolved,
        ):
            resp = client.post(
                "/channel/add",
                json={"source": "https://www.youtube.com/watch?v=abc123"},
            )
            request_id = resp.json()["request_id"]

        # Check the WS queue for channel_added event
        messages = []
        while not ws._queue.empty():
            messages.append(ws._queue.get_nowait())

        channel_added_msgs = [m for m in messages if m["type"] == "channel_added"]
        assert len(channel_added_msgs) == 1
        msg = channel_added_msgs[0]
        assert msg["channel"] == 0
        assert msg["source_type"] == "youtube_live"
        assert msg["request_id"] == request_id

    def test_ws_notification_on_failure(self, app, client):
        """Failed resolution should enqueue resolution_failed WS event."""
        client.post("/pipeline/start")
        ws = app.state.ws

        with patch(
            "backend.api.routes.resolve_source",
            new_callable=AsyncMock,
            side_effect=ValueError("yt-dlp failed: Private video"),
        ):
            resp = client.post(
                "/channel/add",
                json={"source": "https://www.youtube.com/watch?v=private"},
            )
            request_id = resp.json()["request_id"]

        messages = []
        while not ws._queue.empty():
            messages.append(ws._queue.get_nowait())

        failed_msgs = [m for m in messages if m["type"] == "resolution_failed"]
        assert len(failed_msgs) == 1
        msg = failed_msgs[0]
        assert msg["request_id"] == request_id
        assert "Private video" in msg["error"]
