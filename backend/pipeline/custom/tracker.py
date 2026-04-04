"""Standalone BoT-SORT tracker wrapper.

Wraps vendored BOTSORT (from ultralytics v8.4, no torch/ultralytics dependency) with:
- A DetectionResults adapter (BoT-SORT expects .conf/.xyxy/.xywh/.cls)
- Sequential ID mapping (tracker IDs → 1, 2, 3...)
- Reset on phase transition
"""

import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)


class DetectionResults:
    """Minimal Results-like object for BoT-SORT.

    Wraps (N, 6) numpy array: [x1, y1, x2, y2, conf, cls].
    """

    def __init__(self, dets: np.ndarray):
        self._dets = dets if len(dets) > 0 else np.empty((0, 6), dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        return self._dets[:, 4]

    @property
    def xyxy(self) -> np.ndarray:
        return self._dets[:, :4]

    @property
    def xywh(self) -> np.ndarray:
        xyxy = self._dets[:, :4]
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return xywh

    @property
    def cls(self) -> np.ndarray:
        return self._dets[:, 5]

    def __len__(self) -> int:
        return len(self._dets)

    def __getitem__(self, idx):
        return DetectionResults(self._dets[idx])


class TrackerWrapper:
    """Per-channel BoT-SORT tracker with sequential ID mapping."""

    def __init__(self):
        self._tracker = self._create_tracker()
        self._to_seq: dict[int, int] = {}
        self._next_seq_id = 1

    def update(self, detections: np.ndarray) -> list[dict]:
        """Feed detections, return tracked objects.

        Args:
            detections: (N, 6) array [x1, y1, x2, y2, conf, cls].

        Returns:
            List of dicts with keys: track_id, bbox (x,y,w,h), confidence,
            class_name, class_id, centroid (cx, cy).
        """
        results = DetectionResults(detections)
        if len(results) == 0:
            self._tracker.multi_predict(self._tracker.tracked_stracks)
            return []

        raw_tracks = self._tracker.update(results)
        if len(raw_tracks) == 0:
            return []

        tracks = []
        for t in raw_tracks:
            # raw_tracks shape: (M, N) — columns vary by ultralytics version
            # Typically: x1, y1, x2, y2, track_id, conf, cls
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            raw_tid = int(t[4])
            conf = float(t[5]) if len(t) > 5 else 0.0
            cls_id = int(t[6]) if len(t) > 6 else 0

            seq_tid = self._seq_id(raw_tid)
            w = x2 - x1
            h = y2 - y1
            cx = int(x1 + w / 2)
            cy = int(y1 + h / 2)

            from backend.pipeline.custom.detector import COCO_VEHICLE_CLASSES

            tracks.append({
                "track_id": seq_tid,
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "confidence": conf,
                "class_name": COCO_VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}"),
                "class_id": cls_id,
                "centroid": (cx, cy),
                "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
            })

        return tracks

    def reset(self) -> None:
        """Reset tracker state. Called on Setup→Analytics transition."""
        self._tracker = self._create_tracker()
        self._to_seq.clear()
        self._next_seq_id = 1
        logger.debug("TrackerWrapper: reset")

    def _seq_id(self, raw_id: int) -> int:
        """Map tracker ID to sequential ID."""
        if raw_id not in self._to_seq:
            self._to_seq[raw_id] = self._next_seq_id
            self._next_seq_id += 1
        return self._to_seq[raw_id]

    @staticmethod
    def _create_tracker():
        from backend.pipeline.custom.botsort import BOTSORT

        args = argparse.Namespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="none",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
        )
        return BOTSORT(args)
