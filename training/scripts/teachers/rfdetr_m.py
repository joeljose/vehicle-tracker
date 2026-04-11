"""RF-DETR-Medium teacher (Roboflow, ICLR 2026, transformer).

IMPORTANT: RF-DETR uses 1-indexed COCO 91-class IDs (the original Microsoft
COCO API style), NOT the 0-indexed 80-class IDs that ultralytics uses. Verify
with `from rfdetr.assets.coco_classes import COCO_CLASSES; print(COCO_CLASSES)`
— it returns a dict like `{1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
6: 'bus', 8: 'truck', ...}`. So our class remap needs to be different from the
ultralytics one, but under the single-class collapse the destination is the
same (project class 0 = vehicle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ._base import FrameDetections

# Single-class collapse: every RF-DETR vehicle id -> project class 0.
RFDETR_TO_PROJECT: dict[int, int] = {
    3: 0,  # car        -> vehicle
    4: 0,  # motorcycle -> vehicle
    6: 0,  # bus        -> vehicle
    8: 0,  # truck      -> vehicle
}


class RFDETRMediumTeacher:
    name = "rfdetr_m"

    def __init__(self) -> None:
        from rfdetr import RFDETRMedium

        # RFDETRMedium auto-downloads weights to a cache dir on first use.
        # The container's $HOME points at /home/training so the cache lands
        # in our persistent .training-cache mount.
        self._model = RFDETRMedium()

    def predict(
        self,
        image_paths: Iterable[Path],
        conf_threshold: float = 0.3,
    ) -> list[FrameDetections]:
        import cv2

        results: list[FrameDetections] = []
        for path in (Path(p) for p in image_paths):
            # RFDETRMedium.predict accepts an image path or PIL/np image and
            # returns a supervision Detections object: .xyxy (pixel x1y1x2y2),
            # .confidence, .class_id (COCO 80-class IDs).
            det = self._model.predict(str(path), threshold=conf_threshold)

            # We need image H,W for normalizing the bbox. Read it once.
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameDetections(image_path=path, image_hw=(0, 0)))
                continue
            h, w = img.shape[:2]

            if len(det) == 0:
                results.append(FrameDetections(image_path=path, image_hw=(h, w)))
                continue

            xyxy = np.asarray(det.xyxy, dtype=np.float32)
            confs = np.asarray(det.confidence, dtype=np.float32)
            coco_ids = np.asarray(det.class_id, dtype=int)

            # Filter to vehicle classes and remap (RF-DETR uses 1-indexed COCO)
            keep_idx = []
            project_ids = []
            for i, c in enumerate(coco_ids):
                pid = RFDETR_TO_PROJECT.get(int(c))
                if pid is None:
                    continue
                keep_idx.append(i)
                project_ids.append(pid)

            if not keep_idx:
                results.append(FrameDetections(image_path=path, image_hw=(h, w)))
                continue

            keep_idx = np.array(keep_idx, dtype=np.int32)
            xyxy = xyxy[keep_idx]
            # Convert xyxy (pixels) -> xywhn (normalized)
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            xywhn = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)

            results.append(
                FrameDetections(
                    image_path=path,
                    image_hw=(int(h), int(w)),
                    boxes_xywhn=xywhn,
                    confs=confs[keep_idx],
                    project_class_ids=np.array(project_ids, dtype=np.int32),
                )
            )

        return results
