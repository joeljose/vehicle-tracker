"""Best-photo capture — score, crop, and save the best frame per tracked vehicle."""

from pathlib import Path

import cv2
import numpy as np


class BestPhotoTracker:
    """Tracks the highest-quality frame crop for each vehicle track.

    Scoring: bbox_area * confidence.  Only the best crop is kept in memory
    per track (~50 KB JPEG equivalent).  On track end, the crop is written
    to disk as a JPEG.

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
        """Crop a bounding box region from an RGB frame.

        CuPy's asnumpy() handles GPU row-stride padding transparently,
        so the frame shape reflects actual pixel dimensions.

        Args:
            frame: RGB uint8 array, shape (H, W, 3).
            bbox: (left, top, width, height).

        Returns:
            Cropped RGB array or None if the crop has zero area.
        """
        left, top, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if w <= 0 or h <= 0:
            return None

        frame_h, frame_w = frame.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(frame_w, left + w)
        y2 = min(frame_h, top + h)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2, :].copy()

    def extract_crops(self, frame: np.ndarray) -> None:
        """Extract pending crops from the current frame buffer.

        Called once per frame after all score() calls for that frame.

        Args:
            frame: RGB uint8 array from BufferOperator (after CuPy asnumpy).
        """
        for track_id, bbox in list(self.pending_crops.items()):
            crop = self.crop_from_frame(frame, bbox)
            if crop is not None:
                self.best_crops[track_id] = crop
        self.pending_crops.clear()

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

        print(
            f"Track #{track_id}: best photo saved "
            f"(score={score:.0f}, {w}x{h} crop)",
            flush=True,
        )

        # Clean up
        del self.best_scores[track_id]
        del self.best_bboxes[track_id]
        del self.best_crops[track_id]
        self.pending_crops.pop(track_id, None)

        return str(jpeg_path)
