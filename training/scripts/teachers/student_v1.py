"""Fine-tuned YOLOv8s student from M8-P1.5 v2.

The "student" is the model we actually deploy. It follows the same Teacher
protocol as the bake-off candidates so run_comparison.py can evaluate it
against the comparison set using the same code path.

IMPORTANT: this wrapper differs from the other YOLO wrappers because the
student was fine-tuned as a 1-class model. Its predictions come out with
class id 0 (vehicle) directly — there's no COCO class filter to apply and
no COCO-to-project remap. If you inherited from YOLOv8xTeacher as-is, the
parent's COCO_TO_PROJECT filter would drop every single prediction.

Subclasses (e.g. StudentV1ContTeacher) override `weights` to point at a
different checkpoint. Everything else stays the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ._base import FrameDetections

BATCH = 8
IMGSZ = 640
IOU = 0.45


class StudentV1Teacher:
    name = "student_v1"
    weights = "/app/runs/detect/yolov8s_rfdetr_v1/weights/best.pt"

    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.weights)

    def predict(
        self,
        image_paths: Iterable[Path],
        conf_threshold: float = 0.3,
    ) -> list[FrameDetections]:
        paths = [Path(p) for p in image_paths]
        results: list[FrameDetections] = []

        # Same chunking trick as the other YOLO wrappers — ultralytics'
        # batch= parameter is silently ignored on list sources, so we
        # call predict() per chunk manually.
        for start in range(0, len(paths), BATCH):
            chunk = paths[start : start + BATCH]
            for path, raw in zip(
                chunk,
                self._model.predict(
                    source=[str(p) for p in chunk],
                    conf=conf_threshold,
                    iou=IOU,
                    imgsz=IMGSZ,
                    # No classes= filter: the student only knows one class
                    # and it's already project class 0 (vehicle).
                    device=0,
                    half=True,
                    stream=False,
                    verbose=False,
                ),
            ):
                results.append(_student_result_to_frame(path, raw))

        return results


def _student_result_to_frame(path: Path, raw) -> FrameDetections:
    """Convert an ultralytics Result from the 1-class student into FrameDetections.

    The student's class id is always 0 post-collapse, so the class-id array
    goes straight through to project_class_ids without any remap.
    """
    h, w = raw.orig_shape  # (H, W)
    boxes = raw.boxes
    if boxes is None or len(boxes) == 0:
        return FrameDetections(image_path=path, image_hw=(h, w))

    xywhn = boxes.xywhn.cpu().numpy().astype(np.float32)
    confs = boxes.conf.cpu().numpy().astype(np.float32)
    class_ids = boxes.cls.cpu().numpy().astype(np.int32)
    # Sanity: the student only ever emits class 0. If some pathological
    # checkpoint emits other ids, force them to 0 — all our downstream
    # code assumes project class 0 = vehicle.
    class_ids[:] = 0

    return FrameDetections(
        image_path=path,
        image_hw=(int(h), int(w)),
        boxes_xywhn=xywhn,
        confs=confs,
        project_class_ids=class_ids,
    )
