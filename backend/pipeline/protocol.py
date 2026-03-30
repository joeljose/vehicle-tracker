"""Pipeline-agnostic interface contract.

Both DeepStream and custom pipelines implement this Protocol.
The API layer (FastAPI routes, WebSocket broadcaster) imports ONLY
this module — never the concrete pipeline classes.

Swapping backends is a one-line change in main.py:
    pipeline = DeepStreamPipeline(...)  # or CustomPipeline(...)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class ChannelPhase(Enum):
    SETUP = "setup"
    ANALYTICS = "analytics"
    REVIEW = "review"


@dataclass
class Detection:
    """Single detection in a frame."""

    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    centroid: tuple[int, int]


@dataclass
class FrameResult:
    """Per-frame detection results for a channel."""

    channel_id: int
    frame_number: int
    timestamp_ms: int
    detections: list[Detection]
    annotated_jpeg: bytes | None  # MJPEG frame with overlays baked in
    inference_ms: float = 0.0
    phase: str = "setup"
    idle_mode: bool = False


class PipelineBackend(Protocol):
    """Interface that both DeepStream and custom pipelines must implement.

    The API layer uses this Protocol exclusively. It never imports
    concrete pipeline classes (DeepStreamPipeline, CustomPipeline).
    """

    channels: dict[int, str]  # channel_id -> source

    def start(self) -> None:
        """Boot the inference pipeline (load model, allocate GPU resources)."""
        ...

    def stop(self) -> None:
        """Stop pipeline, release all GPU resources cleanly."""
        ...

    def add_channel(self, channel_id: int, source: str) -> None:
        """Add a video source (file path or resolved HLS URL) to the pipeline."""
        ...

    def remove_channel(self, channel_id: int) -> None:
        """Remove a channel, release its decoder and tracker resources."""
        ...

    def configure_channel(
        self,
        channel_id: int,
        roi_polygon: list[tuple[float, float]],
        entry_exit_lines: dict,
    ) -> None:
        """Set ROI and entry/exit lines for a channel.

        Called before transitioning to analytics. The pipeline uses these
        coordinates for direction detection and ROI filtering.
        """
        ...

    def set_channel_phase(self, channel_id: int, phase: ChannelPhase) -> None:
        """Transition a channel's phase. Queued and applied between frames."""
        ...

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update detection confidence threshold at runtime."""
        ...

    def set_inference_interval(self, interval: int) -> None:
        """Set nvinfer interval (0=every frame, 15=idle mode)."""
        ...

    def register_frame_callback(
        self, callback: Callable[[FrameResult], None]
    ) -> None:
        """Register callback invoked per-channel per-frame with detection results.

        The FastAPI layer uses this to feed WebSocket broadcaster and MJPEG endpoints.
        """
        ...

    def register_alert_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        """Register callback invoked when a transit or stagnant alert is generated.

        The dict matches the AlertStore input format (pipeline-side alert).
        """
        ...

    def register_track_ended_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        """Register callback invoked when a track ends.

        The dict matches the track_ended WebSocket schema.
        """
        ...

    def register_phase_callback(
        self, callback: Callable[[int, ChannelPhase, ChannelPhase], None]
    ) -> None:
        """Register callback invoked when a channel auto-transitions phase.

        Called with (channel_id, new_phase, previous_phase). Used by the API
        layer to push WS notifications and trigger clip extraction on EOS.
        """
        ...

    def get_snapshot(self, track_id: int) -> bytes | None:
        """Return the best-photo JPEG crop for a given track, or None."""
        ...

    def get_channel_phase(self, channel_id: int) -> ChannelPhase:
        """Return the current phase of a channel."""
        ...
