"""YOLOv8x teacher (the current baseline)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ._base import (
    COCO_TO_PROJECT,
    COCO_VEHICLE_CLASS_IDS,
    FrameDetections,
)

BATCH = 4
IMGSZ = 640
IOU = 0.45


class YOLOv8xTeacher:
    name = "yolov8x"
    _weights = "yolov8x.pt"

    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self._weights)

    def predict(
        self,
        image_paths: Iterable[Path],
        conf_threshold: float = 0.3,
    ) -> list[FrameDetections]:
        paths = [Path(p) for p in image_paths]
        results: list[FrameDetections] = []

        # Chunk to dodge ultralytics' batch=ignored-on-list-source quirk
        for start in range(0, len(paths), BATCH):
            chunk = paths[start : start + BATCH]
            for path, raw in zip(
                chunk,
                self._model.predict(
                    source=[str(p) for p in chunk],
                    conf=conf_threshold,
                    iou=IOU,
                    imgsz=IMGSZ,
                    classes=COCO_VEHICLE_CLASS_IDS,
                    device=0,
                    half=True,
                    stream=False,
                    verbose=False,
                ),
            ):
                results.append(_ultralytics_result_to_frame(path, raw))

        return results


def _ultralytics_result_to_frame(path: Path, raw) -> FrameDetections:
    """Convert one ultralytics Result into a FrameDetections."""
    h, w = raw.orig_shape  # (H, W)
    boxes = raw.boxes
    if boxes is None or len(boxes) == 0:
        return FrameDetections(image_path=path, image_hw=(h, w))

    xywhn = boxes.xywhn.cpu().numpy().astype(np.float32)
    confs = boxes.conf.cpu().numpy().astype(np.float32)
    coco_ids = boxes.cls.cpu().numpy().astype(int)

    # Filter to vehicle classes and remap to project IDs
    keep_idx = []
    project_ids = []
    for i, c in enumerate(coco_ids):
        pid = COCO_TO_PROJECT.get(int(c))
        if pid is None:
            continue
        keep_idx.append(i)
        project_ids.append(pid)

    if not keep_idx:
        return FrameDetections(image_path=path, image_hw=(h, w))

    keep_idx = np.array(keep_idx, dtype=np.int32)
    return FrameDetections(
        image_path=path,
        image_hw=(int(h), int(w)),
        boxes_xywhn=xywhn[keep_idx],
        confs=confs[keep_idx],
        project_class_ids=np.array(project_ids, dtype=np.int32),
    )
