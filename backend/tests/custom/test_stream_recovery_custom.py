"""Stream recovery tests for the custom pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from backend.pipeline.protocol import ChannelPhase


@pytest.fixture
def pipeline():
    """Create a CustomPipeline with mocked GPU components."""
    with patch("backend.pipeline.custom.adapter.ensure_engine", return_value="/fake/engine"), \
         patch("backend.pipeline.custom.adapter.TRTDetector"), \
         patch("backend.pipeline.custom.adapter.GpuPreprocessor"), \
         patch("backend.pipeline.custom.adapter.FrameRenderer"):
        from backend.pipeline.custom.adapter import CustomPipeline

        p = CustomPipeline()
        yield p


class TestYouTubeEosTriggersRecovery:
    """When a YouTube Live stream hits EOS, recovery should trigger instead of Review."""

    def test_file_eos_goes_to_review(self, pipeline):
        """File sources should transition directly to Review on EOS."""
        pipeline._states[0] = MagicMock(
            source_type="file",
            phase=ChannelPhase.ANALYTICS,
            original_url=None,
        )
        pipeline.channels[0] = "/data/video.mp4"

        with patch.object(pipeline, "_do_phase_transition") as mock_transition:
            pipeline._handle_eos(0)
            mock_transition.assert_called_once_with(0, ChannelPhase.REVIEW)

    def test_youtube_eos_triggers_recovery(self, pipeline):
        """YouTube Live should trigger recovery, not immediate Review."""
        pipeline._states[0] = MagicMock(
            source_type="youtube_live",
            phase=ChannelPhase.ANALYTICS,
            original_url="https://youtube.com/watch?v=test",
        )
        pipeline.channels[0] = "https://hls.url/stream.m3u8"

        with patch.object(pipeline, "_trigger_recovery") as mock_recovery:
            pipeline._handle_eos(0)
            mock_recovery.assert_called_once_with(0, "decode returned None (HLS)")

    def test_youtube_eos_skips_if_already_recovering(self, pipeline):
        """Don't trigger recovery if already recovering this channel."""
        pipeline._states[0] = MagicMock(
            source_type="youtube_live",
            phase=ChannelPhase.ANALYTICS,
            original_url="https://youtube.com/watch?v=test",
        )
        pipeline._recovering.add(0)

        with patch.object(pipeline, "_trigger_recovery") as mock_recovery:
            pipeline._handle_eos(0)
            mock_recovery.assert_not_called()


class TestCircuitBreakerEjects:
    """Circuit breaker should eject channel after too many reconnects."""

    def test_circuit_breaker_trips(self, pipeline):
        """After threshold reconnects, channel should be ejected to Review."""
        from backend.pipeline.stream_recovery import CIRCUIT_BREAKER_THRESHOLD

        # Trip the breaker
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            pipeline._circuit_breaker.record_rebuild(0)

        assert pipeline._circuit_breaker.is_tripped(0)


class TestReconnectTransition:
    """The reconnect transition should recreate the decoder."""

    def test_reconnect_updates_source(self, pipeline):
        """Reconnect should update the channel source URL."""
        mock_decoder = MagicMock()
        state = MagicMock(decoder=mock_decoder, source="old_url")
        pipeline._states[0] = state

        pipeline._do_reconnect_channel(0, "new_hls_url")

        mock_decoder.reconnect.assert_called_once_with("new_hls_url")
        assert state.source == "new_hls_url"
        assert state.frame_count == 0


class TestLastFramePersists:
    """last_frame_jpeg should not be cleared during recovery."""

    def test_reconnect_preserves_last_frame(self, pipeline):
        """Reconnect reuses existing state — last_frame_jpeg is untouched."""
        mock_decoder = MagicMock()
        state = MagicMock(
            decoder=mock_decoder,
            source="old_url",
            last_frame_jpeg=b"jpeg_data_here",
        )
        pipeline._states[0] = state

        pipeline._do_reconnect_channel(0, "new_hls_url")

        # last_frame_jpeg should still be the original value
        assert state.last_frame_jpeg == b"jpeg_data_here"
