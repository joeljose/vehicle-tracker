"""BatchMetadataRouter — routes batched frame metadata to per-channel reporters.

In the shared pipeline, nvtracker produces batched metadata containing frames
from all active channels. This router reads source_id from each frame and
dispatches it to the correct channel's TrackingReporter.

Also provides per-source EOS detection via frame-gap watchdog: if a source
stops producing frames for >2 seconds, it's considered EOS'd.
"""

import time

from pyservicemaker import BatchMetadataOperator


class _SingleFrameBatch:
    """Wraps a single frame_meta to look like a batch_meta with one item.

    This lets us call TrackingReporter.handle_metadata() unchanged —
    it iterates batch_meta.frame_items and processes each frame.
    """

    __slots__ = ("frame_items",)

    def __init__(self, frame_meta):
        self.frame_items = [frame_meta]


class BatchMetadataRouter(BatchMetadataOperator):
    """Probe that routes batched frame metadata to per-channel reporters."""

    EOS_TIMEOUT = 2.0  # seconds without frames to consider a source EOS'd

    def __init__(self, reporters: dict):
        super().__init__()
        self._reporters = dict(reporters)  # source_id -> TrackingReporter
        self._last_frame_time: dict[int, float] = {}  # source_id -> monotonic time
        self._eos_callbacks: dict[int, object] = {}  # source_id -> callable

    def add_reporter(self, source_id: int, reporter) -> None:
        self._reporters[source_id] = reporter

    def remove_reporter(self, source_id: int) -> None:
        self._reporters.pop(source_id, None)

    def watch_for_eos(self, source_id: int, callback) -> None:
        """Register a callback for when this source stops producing frames."""
        self._eos_callbacks[source_id] = callback
        self._last_frame_time[source_id] = time.monotonic()

    def unwatch_eos(self, source_id: int) -> None:
        """Cancel EOS watching for a source."""
        self._eos_callbacks.pop(source_id, None)
        self._last_frame_time.pop(source_id, None)

    def fire_all_pending_eos(self) -> None:
        """Fire all pending EOS callbacks (pipeline-level EOS fallback)."""
        for source_id, callback in list(self._eos_callbacks.items()):
            del self._eos_callbacks[source_id]
            self._last_frame_time.pop(source_id, None)
            callback(source_id)

    def handle_metadata(self, batch_meta):
        now = time.monotonic()

        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            # Update frame time for EOS tracking
            if source_id in self._eos_callbacks:
                self._last_frame_time[source_id] = now
            # Route to reporter
            reporter = self._reporters.get(source_id)
            if reporter is not None:
                reporter.handle_metadata(_SingleFrameBatch(frame_meta))

        # Check for stale sources (per-source EOS detection)
        for source_id in list(self._eos_callbacks):
            last = self._last_frame_time.get(source_id, 0)
            if now - last > self.EOS_TIMEOUT:
                callback = self._eos_callbacks.pop(source_id)
                self._last_frame_time.pop(source_id, None)
                callback(source_id)
