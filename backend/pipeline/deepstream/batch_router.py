"""BatchMetadataRouter — routes batched frame metadata to per-channel reporters.

In the shared pipeline, nvtracker produces batched metadata containing frames
from all active channels. This router reads source_id from each frame and
dispatches it to the correct channel's TrackingReporter.
"""

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

    def __init__(self, reporters: dict):
        super().__init__()
        self._reporters = dict(reporters)  # source_id -> TrackingReporter

    def add_reporter(self, source_id: int, reporter) -> None:
        self._reporters[source_id] = reporter

    def remove_reporter(self, source_id: int) -> None:
        self._reporters.pop(source_id, None)

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            reporter = self._reporters.get(source_id)
            if reporter is not None:
                reporter.handle_metadata(_SingleFrameBatch(frame_meta))
