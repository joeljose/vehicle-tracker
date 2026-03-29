"""Fake pipeline backend for testing — no GPU, no GStreamer."""

from backend.pipeline.protocol import ChannelPhase, FrameResult

from typing import Callable


class FakeBackend:
    """In-memory PipelineBackend implementation for testing.

    Satisfies the PipelineBackend Protocol with deterministic behavior.
    Raises the same exceptions that routes should handle.
    """

    def __init__(self):
        self.started: bool = False
        self.channels: dict[int, str] = {}  # channel_id -> source
        self.channel_phases: dict[int, ChannelPhase] = {}
        self.confidence_threshold: float = 0.5
        self.inference_interval: int = 0
        self._frame_callback: Callable[[FrameResult], None] | None = None
        self._alert_callback: Callable[[dict], None] | None = None
        self._track_ended_callback: Callable[[dict], None] | None = None
        self._snapshots: dict[int, bytes] = {}  # track_id -> jpeg bytes

    def start(self) -> None:
        if self.started:
            raise RuntimeError("Pipeline already started")
        self.started = True

    def stop(self) -> None:
        if not self.started:
            raise RuntimeError("Pipeline not started")
        self.started = False
        self.channels.clear()
        self.channel_phases.clear()

    def add_channel(self, channel_id: int, source: str) -> None:
        if not self.started:
            raise RuntimeError("Pipeline not started")
        if channel_id in self.channels:
            raise ValueError(f"Channel {channel_id} already exists")
        self.channels[channel_id] = source
        self.channel_phases[channel_id] = ChannelPhase.SETUP

    def remove_channel(self, channel_id: int) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        del self.channels[channel_id]
        del self.channel_phases[channel_id]

    def set_channel_phase(self, channel_id: int, phase: ChannelPhase) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        self.channel_phases[channel_id] = phase

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold

    def set_inference_interval(self, interval: int) -> None:
        self.inference_interval = interval

    def register_frame_callback(
        self, callback: Callable[[FrameResult], None]
    ) -> None:
        self._frame_callback = callback

    def register_alert_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        self._alert_callback = callback

    def register_track_ended_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        self._track_ended_callback = callback

    def get_snapshot(self, track_id: int) -> bytes | None:
        return self._snapshots.get(track_id)

    def get_channel_phase(self, channel_id: int) -> ChannelPhase:
        if channel_id not in self.channel_phases:
            raise KeyError(f"Channel {channel_id} not found")
        return self.channel_phases[channel_id]

    # -- Test helpers (not part of PipelineBackend Protocol) --

    def emit_frame(
        self, channel_id: int, frame_number: int = 0, jpeg: bytes = b"\xff\xd8fake"
    ) -> None:
        """Simulate pipeline producing a frame. Calls registered frame callback."""
        if self._frame_callback is None:
            return
        result = FrameResult(
            channel_id=channel_id,
            frame_number=frame_number,
            timestamp_ms=frame_number * 33,
            detections=[],
            annotated_jpeg=jpeg,
        )
        self._frame_callback(result)

    def emit_alert(self, alert: dict) -> None:
        """Simulate pipeline producing an alert. Calls registered alert callback."""
        if self._alert_callback is not None:
            self._alert_callback(alert)

    def emit_track_ended(self, track: dict) -> None:
        """Simulate pipeline producing a track_ended event."""
        if self._track_ended_callback is not None:
            self._track_ended_callback(track)
