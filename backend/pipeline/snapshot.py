"""Best-photo capture — score, crop, and save the best frame per tracked vehicle."""

import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BestPhotoTracker:
    """Tracks the highest-quality frame crop for each vehicle track.

    Scoring: bbox_area * confidence.  Only the best crop is kept in memory
    per track (~50 KB JPEG equivalent).  On track end, the crop is written
    to disk as a JPEG.

    Crops are always square (long_side × long_side) with the vehicle
    occupying ≥50% of the scene-expanded area.  Scene expansion adds
    real frame content around the vehicle for context.  Black padding
    fills any gap from frame boundaries or elongation.

    Usage (called from pipeline probes each frame):
        1. score()          — called per detection (metadata probe)
        2. extract_crops()  — called once per frame  (buffer probe)
        3. save()           — called on track loss    (metadata probe)
    """

    def __init__(self):
        self.best_scores: dict[int, float] = {}
        self.best_bboxes: dict[int, tuple] = {}
        self.best_crops: dict[int, np.ndarray] = {}
        self.pending_crops: dict[int, tuple] = {}  # track_id → (left, top, w, h)

    def score(
        self,
        track_id: int,
        area: float,
        confidence: float,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Update best score for a track.  Queues a crop if this is a new best.

        Args:
            track_id: Unique track identifier.
            area: Bounding box area (width * height).
            confidence: Detection confidence [0, 1].
            bbox: (left, top, width, height) in pixel coordinates.
        """
        score = area * confidence
        if score > self.best_scores.get(track_id, -1):
            self.best_scores[track_id] = score
            self.best_bboxes[track_id] = bbox
            self.pending_crops[track_id] = bbox

    def crop_from_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        """Crop a square region centered on the bbox from an RGB frame.

        Algorithm:
            1. long = max(w, h), short = min(w, h)
            2. Expand shorter side until vehicle area ≥ 50% of expanded area
            3. Cap expansion at long side
            4. Extract from frame, clamp to bounds (black for out-of-frame)
            5. Pad remaining gap with black to reach long × long square

        Args:
            frame: RGB uint8 array, shape (H, W, 3).
            bbox: (left, top, width, height).

        Returns:
            Square RGB array (long_side × long_side × 3) or None if zero area.
        """
        left, top, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if w <= 0 or h <= 0:
            return None

        frame_h, frame_w = frame.shape[:2]
        long = max(w, h)
        short = min(w, h)

        # Scene expansion: ensure vehicle ≥ 50% of scene area
        scene_expand = max(0, math.ceil(w * h / long - short / 2))
        scene_short = min(short + 2 * scene_expand, long)
        black_pad = long - scene_short

        # Determine expansion direction and compute crop region
        if w >= h:
            # Wider bbox → expand vertically
            expand_each = (scene_short - h) / 2
            crop_x1 = left
            crop_x2 = left + w
            crop_y1 = top - math.floor(expand_each)
            crop_y2 = top + h + math.ceil(expand_each)
        else:
            # Taller bbox → expand horizontally
            expand_each = (scene_short - w) / 2
            crop_x1 = left - math.floor(expand_each)
            crop_x2 = left + w + math.ceil(expand_each)
            crop_y1 = top
            crop_y2 = top + h

        # Create output square (black-filled)
        result = np.zeros((long, long, 3), dtype=np.uint8)

        # Determine where scene content goes in the result
        # (centered, with black_pad split evenly on the expansion axis)
        pad_before = black_pad // 2
        pad_after = black_pad - pad_before

        if w >= h:
            # Black padding is on top/bottom of the vertical axis
            dst_y1 = pad_before
            dst_y2 = long - pad_after
            dst_x1 = 0
            dst_x2 = long  # long == w for wider bbox
        else:
            # Black padding is on left/right of the horizontal axis
            dst_x1 = pad_before
            dst_x2 = long - pad_after
            dst_y1 = 0
            dst_y2 = long  # long == h for taller bbox

        # Clamp crop region to frame bounds and adjust destination
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(frame_w, crop_x2)
        src_y2 = min(frame_h, crop_y2)

        # Offset into destination for frame-boundary clipping
        clip_left = src_x1 - crop_x1
        clip_top = src_y1 - crop_y1
        clip_right = crop_x2 - src_x2
        clip_bottom = crop_y2 - src_y2

        # Actual destination region (after clipping adjustments)
        fill_x1 = dst_x1 + clip_left
        fill_y1 = dst_y1 + clip_top
        fill_x2 = dst_x2 - clip_right
        fill_y2 = dst_y2 - clip_bottom

        # Copy frame content into result
        if (
            fill_x2 > fill_x1
            and fill_y2 > fill_y1
            and src_x2 > src_x1
            and src_y2 > src_y1
        ):
            result[fill_y1:fill_y2, fill_x1:fill_x2, :] = frame[
                src_y1:src_y2, src_x1:src_x2, :
            ]

        return result

    def extract_crops(self, frame: np.ndarray) -> None:
        """Extract pending crops from the current frame buffer.

        Called once per frame after all score() calls for that frame.

        Args:
            frame: RGB uint8 array from BufferOperator (after CuPy asnumpy).
        """
        # Snapshot and remove only processed entries — .clear() would
        # discard entries added by the metadata probe for later frames
        # (which runs ahead due to GStreamer queue decoupling).
        to_process = dict(self.pending_crops)
        for track_id, bbox in to_process.items():
            crop = self.crop_from_frame(frame, bbox)
            if crop is not None:
                self.best_crops[track_id] = crop
            self.pending_crops.pop(track_id, None)

    def save(self, track_id: int, output_dir: str) -> str | None:
        """Write the best crop for a track as JPEG and clean up state.

        Args:
            track_id: Track to save.
            output_dir: Directory for output JPEGs.

        Returns:
            Path to saved JPEG, or None if no crop available.
        """
        crop = self.best_crops.get(track_id)
        if crop is None:
            # Clean up score/bbox even if no crop
            self.best_scores.pop(track_id, None)
            self.best_bboxes.pop(track_id, None)
            return None

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        jpeg_path = out / f"{track_id}.jpg"

        score = self.best_scores.get(track_id, 0)
        h, w = crop.shape[:2]

        # cv2.imwrite expects BGR
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(jpeg_path), bgr)

        logger.info(
            "Track #%d: best photo saved (score=%d, %dx%d crop)",
            track_id,
            score,
            w,
            h,
        )

        # Clean up
        del self.best_scores[track_id]
        del self.best_bboxes[track_id]
        del self.best_crops[track_id]
        self.pending_crops.pop(track_id, None)

        return str(jpeg_path)
