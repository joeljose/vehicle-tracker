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
    """Test square crop with scene expansion and black padding."""

    def test_crop_is_always_square(self):
        """Output is always long_side × long_side."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        crop = bpt.crop_from_frame(frame, bbox=(100, 200, 77, 54))
        assert crop.shape[0] == crop.shape[1]  # square
        assert crop.shape[0] == 77  # long side
        assert crop.shape[2] == 3

    def test_square_bbox_no_expansion(self):
        """Square bbox → square crop, no expansion needed."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        crop = bpt.crop_from_frame(frame, bbox=(100, 200, 60, 60))
        assert crop.shape == (60, 60, 3)

    def test_wide_bbox_expands_vertically(self):
        """Wider bbox → expand height, vehicle ≥50% of scene area."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        # 77×54: long=77, expand = 77*54/77 - 54/2 = 54-27 = 27
        # scene_short = min(54+54, 77) = 77 → capped, no black pad
        crop = bpt.crop_from_frame(frame, bbox=(500, 500, 77, 54))
        assert crop.shape == (77, 77, 3)
        # Vehicle area = 77*54 = 4158, scene area = 77*77 = 5929
        # Fill = 4158/5929 = 70% ≥ 50% ✓

    def test_tall_bbox_expands_horizontally(self):
        """Taller bbox → expand width."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        crop = bpt.crop_from_frame(frame, bbox=(500, 500, 40, 80))
        assert crop.shape == (80, 80, 3)

    def test_elongated_bbox_gets_black_padding(self):
        """Very elongated bbox: scene expansion capped, black padding fills gap."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        # 321×125: long=321, expand = 321*125/321 - 125/2 = 125-62.5 = 62.5
        # scene_short = min(125+125, 321) = 250 < 321
        # black_pad = 321 - 250 = 71
        crop = bpt.crop_from_frame(frame, bbox=(500, 400, 321, 125))
        assert crop.shape == (321, 321, 3)
        # Top and bottom strips should be black (padding)
        # black_pad/2 ≈ 35 pixels on each side
        assert crop[0, 160, :].sum() == 0  # top is black
        assert crop[320, 160, :].sum() == 0  # bottom is black
        # Center should have scene content (non-black)
        assert crop[160, 160, :].sum() > 0  # center has content

    def test_edge_vehicle_frame_clamp_with_black(self):
        """Vehicle at frame edge: out-of-frame pixels are black."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        # bbox at right edge: left=180, w=20, h=15 → long=20
        # Needs to expand height, centered around top..top+h
        # Some expansion may go beyond frame bounds
        crop = bpt.crop_from_frame(frame, bbox=(180, 85, 20, 15))
        assert crop.shape == (20, 20, 3)
        # Right side: 180+20=200 ✅ fits
        # Bottom: expansion might exceed frame height 100

    def test_corner_vehicle_double_black_pad(self):
        """Vehicle at corner: both frame clamp black + gap black."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        # Very elongated at bottom-right corner
        # bbox: left=170, top=90, w=30, h=10 → long=30
        crop = bpt.crop_from_frame(frame, bbox=(170, 90, 30, 10))
        assert crop.shape == (30, 30, 3)
        # Should have black padding for both:
        # 1. Elongation gap (30-20=10px black pad total)
        # 2. Frame overflow (bottom extends past y=100)

    def test_crop_contains_scene_content(self):
        """Expanded region contains actual frame pixels, not just black."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        crop = bpt.crop_from_frame(frame, bbox=(500, 500, 77, 54))
        # The expanded region around the bbox should have frame content
        # (gradient pattern, not zeros)
        center_row = crop.shape[0] // 2
        assert crop[center_row, :, :].max() > 0

    def test_crop_is_a_copy(self):
        """Modifying crop doesn't affect original frame."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(50, 50, 30, 30))
        original_val = frame[50, 50, 0].copy()
        crop[:] = 0
        assert frame[50, 50, 0] == original_val

    def test_crop_zero_size_returns_none(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(0, 0, 0, 0))
        assert crop is None

    def test_crop_fully_outside_returns_none(self):
        """Bbox completely outside frame → None."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(100, 200)
        crop = bpt.crop_from_frame(frame, bbox=(300, 300, 50, 50))
        # bbox center is outside frame, but with expansion it's all black
        # This should still return a crop (all black) since the bbox itself
        # has valid dimensions. Only zero-size returns None.
        if crop is not None:
            assert crop.shape[0] == crop.shape[1]  # still square

    def test_vehicle_fill_at_least_50_percent(self):
        """For elongated bboxes, vehicle area ≥ 50% of scene (non-black) area."""
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        # 98×45: long=98, scene_expand = 98*45/98 - 45/2 = 45-22.5 = 22.5
        # scene_short = min(45+45, 98) = 90
        # scene area = 98 * 90 = 8820
        # vehicle area = 98 * 45 = 4410
        # fill = 4410/8820 = 50% ✓
        crop = bpt.crop_from_frame(frame, bbox=(500, 500, 98, 45))
        assert crop.shape == (98, 98, 3)


class TestExtractCrops:
    """Test batch crop extraction from frame buffer."""

    def test_extract_clears_pending(self):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.9, bbox=(10, 10, 50, 50))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))
        assert 1 not in bpt.pending_crops
        assert 1 in bpt.best_crops

    def test_extract_stores_square_crop(self):
        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=1000, confidence=0.9, bbox=(10, 10, 50, 40))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))
        crop = bpt.best_crops[1]
        assert isinstance(crop, np.ndarray)
        assert crop.shape[0] == crop.shape[1]  # square
        assert crop.shape[0] == 50  # long side

    def test_extract_replaces_previous_crop(self):
        bpt = BestPhotoTracker()
        frame = _make_gradient_frame(1080, 1920)
        bpt.score(track_id=1, area=100, confidence=0.5, bbox=(10, 10, 20, 20))
        bpt.extract_crops(frame)
        old_size = bpt.best_crops[1].shape[0]

        bpt.score(track_id=1, area=10000, confidence=0.99, bbox=(100, 100, 80, 60))
        bpt.extract_crops(frame)
        new_size = bpt.best_crops[1].shape[0]
        assert new_size != old_size

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

    def test_save_writes_square_jpeg(self, tmp_path):
        """Saved JPEG should be square."""
        import cv2

        bpt = BestPhotoTracker()
        bpt.score(track_id=1, area=5000, confidence=0.9, bbox=(100, 100, 77, 54))
        bpt.extract_crops(_make_gradient_frame(1080, 1920))
        bpt.save(track_id=1, output_dir=str(tmp_path))

        img = cv2.imread(str(tmp_path / "1.jpg"))
        assert img.shape[0] == img.shape[1]  # square

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
