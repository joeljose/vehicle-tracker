"""Tests for BatchMetadataRouter — routes batched frame metadata to per-channel reporters."""

import time
from unittest.mock import MagicMock, patch

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


class TestEosDetection:
    """P2: per-source EOS detection via frame-gap watchdog."""

    def _make_batch(self, source_ids):
        """Create a mock batch_meta with frames from given source_ids."""
        frames = []
        for sid in source_ids:
            f = MagicMock()
            f.source_id = sid
            frames.append(f)
        batch = MagicMock()
        batch.frame_items = frames
        return batch

    def test_eos_fires_on_frame_gap(self):
        """EOS callback fires when a watched source stops producing frames."""
        router = BatchMetadataRouter({})
        reporter = MagicMock()
        router.add_reporter(0, reporter)
        eos_calls = []
        router.watch_for_eos(0, lambda sid: eos_calls.append(sid))

        # Simulate frames flowing
        router.handle_metadata(self._make_batch([0]))
        assert len(eos_calls) == 0

        # Simulate time passing beyond EOS_TIMEOUT
        with patch("backend.pipeline.deepstream.batch_router.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 10.0
            # Source 0 is absent from this batch, but batch still arrives
            # (other sources still flowing)
            router.handle_metadata(self._make_batch([1]))

        assert len(eos_calls) == 1
        assert eos_calls[0] == 0

    def test_eos_does_not_fire_while_frames_flow(self):
        """No EOS callback while frames keep coming."""
        router = BatchMetadataRouter({})
        reporter = MagicMock()
        router.add_reporter(0, reporter)
        eos_calls = []
        router.watch_for_eos(0, lambda sid: eos_calls.append(sid))

        # Simulate continuous frames
        for _ in range(10):
            router.handle_metadata(self._make_batch([0]))

        assert len(eos_calls) == 0

    def test_unwatch_prevents_callback(self):
        """Unwatching before timeout prevents the EOS callback."""
        router = BatchMetadataRouter({})
        reporter = MagicMock()
        router.add_reporter(0, reporter)
        eos_calls = []
        router.watch_for_eos(0, lambda sid: eos_calls.append(sid))

        router.handle_metadata(self._make_batch([0]))
        router.unwatch_eos(0)

        with patch("backend.pipeline.deepstream.batch_router.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 10.0
            router.handle_metadata(self._make_batch([1]))

        assert len(eos_calls) == 0

    def test_fire_all_pending_eos(self):
        """fire_all_pending_eos fires all watched callbacks."""
        router = BatchMetadataRouter({})
        eos_calls = []
        router.watch_for_eos(0, lambda sid: eos_calls.append(sid))
        router.watch_for_eos(1, lambda sid: eos_calls.append(sid))

        router.fire_all_pending_eos()

        assert sorted(eos_calls) == [0, 1]
        # Callbacks should not fire again
        router.fire_all_pending_eos()
        assert sorted(eos_calls) == [0, 1]

    def test_eos_callback_fires_only_once(self):
        """EOS callback should fire exactly once per watch."""
        router = BatchMetadataRouter({})
        reporter = MagicMock()
        router.add_reporter(0, reporter)
        eos_calls = []
        router.watch_for_eos(0, lambda sid: eos_calls.append(sid))

        with patch("backend.pipeline.deepstream.batch_router.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 10.0
            router.handle_metadata(self._make_batch([1]))
            router.handle_metadata(self._make_batch([1]))

        assert len(eos_calls) == 1
