"""Tests for BatchMetadataRouter — routes batched frame metadata to per-channel reporters."""

from unittest.mock import MagicMock

import pytest

from backend.pipeline.deepstream.batch_router import BatchMetadataRouter


class TestBatchMetadataRouter:
    def test_routes_frame_to_correct_channel_reporter(self):
        """Each frame in a batch is dispatched to the reporter matching its source_id."""
        reporter_0 = MagicMock()
        reporter_1 = MagicMock()
        router = BatchMetadataRouter({0: reporter_0, 1: reporter_1})

        frame_0 = MagicMock()
        frame_0.source_id = 0
        frame_1 = MagicMock()
        frame_1.source_id = 1

        batch_meta = MagicMock()
        batch_meta.frame_items = [frame_0, frame_1]

        router.handle_metadata(batch_meta)

        reporter_0.handle_metadata.assert_called_once()
        reporter_1.handle_metadata.assert_called_once()

        call_args_0 = reporter_0.handle_metadata.call_args[0][0]
        assert list(call_args_0.frame_items) == [frame_0]

        call_args_1 = reporter_1.handle_metadata.call_args[0][0]
        assert list(call_args_1.frame_items) == [frame_1]

    def test_ignores_frames_from_unknown_source_id(self):
        """Frames from removed channels are silently ignored."""
        reporter_0 = MagicMock()
        router = BatchMetadataRouter({0: reporter_0})

        frame_0 = MagicMock()
        frame_0.source_id = 0
        frame_unknown = MagicMock()
        frame_unknown.source_id = 99

        batch_meta = MagicMock()
        batch_meta.frame_items = [frame_0, frame_unknown]

        router.handle_metadata(batch_meta)

        assert reporter_0.handle_metadata.call_count == 1

    def test_add_reporter_dynamically(self):
        """Reporters can be added after construction for new channels."""
        router = BatchMetadataRouter({})

        reporter = MagicMock()
        router.add_reporter(0, reporter)

        frame = MagicMock()
        frame.source_id = 0
        batch_meta = MagicMock()
        batch_meta.frame_items = [frame]

        router.handle_metadata(batch_meta)
        reporter.handle_metadata.assert_called_once()

    def test_remove_reporter_dynamically(self):
        """Removed reporters stop receiving frames."""
        reporter = MagicMock()
        router = BatchMetadataRouter({0: reporter})

        router.remove_reporter(0)

        frame = MagicMock()
        frame.source_id = 0
        batch_meta = MagicMock()
        batch_meta.frame_items = [frame]

        router.handle_metadata(batch_meta)
        reporter.handle_metadata.assert_not_called()
