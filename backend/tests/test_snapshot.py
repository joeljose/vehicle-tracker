"""Tests for BestPhotoTracker — best-photo capture scoring and cropping."""

import numpy as np

from backend.pipeline.snapshot import BestPhotoTracker


class TestScoring:
    """Test per-frame scoring logic: bbox_area * confidence."""

    def test_first_score_always_stored(self):
        bpt = BestPhotoTracker()
        bbox = (100, 200, 50, 80)  # left, top, width, height
        bpt.score(track_id=1, area=4000, confidence=0.9, bbox=bbox)
        assert bpt.best_scores[1] == 4000 * 0.9
        assert bpt.best_bboxes[1] == bbox

    def test_higher_score_replaces(self):
        bpt = BestPhotoTracker()
        bbox_low = (10, 10, 20, 20)
        bbox_high = (100, 100, 100, 100)
        bpt.score(track_id=1, area=400, confidence=0.5, bbox=bbox_low)
        bpt.score(track_id=1, area=10000, confidence=0.95, bbox=bbox_high)
        assert bpt.best_scores[1] == 10000 * 0.95
        assert bpt.best_bboxes[1] == bbox_high

    def test_lower_score_ignored(self):
        bpt = BestPhotoTracker()
        bbox_high = (100, 100, 100, 100)
        bbox_low = (10, 10, 10, 10)
        bpt.score(track_id=1, area=10000, confidence=0.9, bbox=bbox_high)
        bpt.score(track_id=1, area=100, confidence=0.1, bbox=bbox_low)
        assert bpt.best_scores[1] == 10000 * 0.9
        assert bpt.best_bboxes[1] == bbox_high

    def test_pending_set_on_new_best(self):
        bpt = BestPhotoTracker()
        bbox = (50, 50, 60, 80)
        bpt.score(track_id=1, area=4800, confidence=0.8, bbox=bbox)
        assert 1 in bpt.pending_crops

    def test_pending_not_set_when_score_lower(self):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=10000, confidence=0.9, bbox=(0, 0, 100, 100))
        bpt.extract_crops(_make_frame(1080, 1920))
        bpt.score(track_id=1, area=100, confidence=0.1, bbox=(0, 0, 10, 10))
        assert 1 not in bpt.pending_crops

    def test_independent_tracks(self):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.8, bbox=(10, 10, 50, 50))
        bpt.score(track_id=2, area=2000, confidence=0.7, bbox=(200, 200, 80, 80))
        assert bpt.best_scores[1] == 1000 * 0.8
        assert bpt.best_scores[2] == 2000 * 0.7


class TestCropFromFrame:
    """Test frame cropping."""

    def test_crop_basic(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(10, 20, 30, 40))
        assert crop.shape == (40, 30, 3)

    def test_crop_contains_correct_pixels(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(10, 20, 30, 40))
        # Verify crop matches the source region
        expected = frame[20:60, 10:40, :]
        np.testing.assert_array_equal(crop, expected)

    def test_crop_is_a_copy(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(10, 20, 30, 40))
        crop[:] = 0
        # Original frame should be unmodified
        assert frame[20, 10, 0] > 0

    def test_crop_clamps_to_frame_bounds(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        # Bbox goes 20px past right edge and 10px past bottom
        crop = bpt.crop_from_frame(frame, bbox=(180, 90, 40, 20))
        assert crop.shape[1] == 20  # clamped: 200 - 180
        assert crop.shape[0] == 10  # clamped: 100 - 90

    def test_crop_zero_size_returns_none(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(0, 0, 0, 0))
        assert crop is None

    def test_crop_fully_outside_returns_none(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(300, 300, 50, 50))
        assert crop is None


class TestExtractCrops:
    """Test batch crop extraction from frame buffer."""

    def test_extract_clears_pending(self):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.9, bbox=(10, 10, 50, 50))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))
        assert 1 not in bpt.pending_crops
        assert 1 in bpt.best_crops

    def test_extract_stores_crop_array(self):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.9, bbox=(10, 10, 50, 50))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))
        crop = bpt.best_crops[1]
        assert isinstance(crop, np.ndarray)
        assert crop.shape == (50, 50, 3)

    def test_extract_replaces_previous_crop(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        bpt.score(track_id=1, area=100, confidence=0.5, bbox=(10, 10, 20, 20))
        bpt.extract_crops(frame)
        old_crop = bpt.best_crops[1].copy()

        bpt.score(track_id=1, area=10000, confidence=0.99, bbox=(100, 100, 80, 80))
        bpt.extract_crops(frame)
        new_crop = bpt.best_crops[1]
        assert new_crop.shape != old_crop.shape

    def test_no_pending_no_op(self):
        bpt = BestPhotoTracker()
        bpt.extract_crops(_make_gradient_frame(100, 200))
        assert len(bpt.best_crops) == 0


class TestSave:
    """Test JPEG writing on track end."""

    def test_save_writes_jpeg(self, tmp_path):
        bpt = BestPhotoTracker()
        bpt.score(track_id=42, area=5000, confidence=0.85, bbox=(10, 10, 50, 60))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))

        result = bpt.save(track_id=42, output_dir=str(tmp_path))
        jpeg_path = tmp_path / "42.jpg"
        assert jpeg_path.exists()
        assert jpeg_path.stat().st_size > 0
        assert result == str(jpeg_path)

    def test_save_cleans_up_state(self, tmp_path):
        bpt = BestPhotoTracker()
        bpt.score(track_id=7, area=1000, confidence=0.9, bbox=(0, 0, 30, 30))
        bpt.extract_crops(_make_gradient_frame(100, 200))
        bpt.save(track_id=7, output_dir=str(tmp_path))
        assert 7 not in bpt.best_scores
        assert 7 not in bpt.best_bboxes
        assert 7 not in bpt.best_crops

    def test_save_no_crop_returns_none(self, tmp_path):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.9, bbox=(0, 0, 30, 30))
        # Never called extract_crops, so no crop stored
        result = bpt.save(track_id=1, output_dir=str(tmp_path))
        assert result is None

    def test_save_unknown_track_returns_none(self, tmp_path):
        bpt = BestPhotoTracker()
        result = bpt.save(track_id=999, output_dir=str(tmp_path))
        assert result is None

    def test_save_creates_output_dir(self, tmp_path):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.9, bbox=(0, 0, 30, 30))
        bpt.extract_crops(_make_gradient_frame(100, 200))
        out = tmp_path / "subdir" / "nested"
        bpt.save(track_id=1, output_dir=str(out))
        assert (out / "1.jpg").exists()

    def test_save_logs_message(self, tmp_path, caplog):
        bpt = BestPhotoTracker()
        bpt.score(track_id=5, area=2000, confidence=0.75, bbox=(10, 10, 40, 50))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))
        with caplog.at_level("INFO"):
            bpt.save(track_id=5, output_dir=str(tmp_path))
        assert "Track #5" in caplog.text
        assert "best photo saved" in caplog.text
        assert "40x50" in caplog.text


# --- Helpers ---

def _make_frame(height: int, width: int) -> np.ndarray:
    """Create a blank RGB frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_gradient_frame(height: int, width: int) -> np.ndarray:
    """Create an RGB frame with a gradient pattern (non-zero pixels)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    row = (np.arange(width) % 255 + 1).astype(np.uint8)
    for c in range(3):
        frame[:, :, c] = row[np.newaxis, :]
    return frame
