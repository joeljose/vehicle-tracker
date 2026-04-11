"""Grounding DINO Tiny teacher (HuggingFace, text-prompted, zero-shot)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ._base import FrameDetections

# Per HF docs we use the period-separated phrase format. We still prompt with
# multiple phrases (vehicle + car + truck + bus + motorcycle) because Grounding
# DINO's text grounding benefits from multiple semantic anchors, but every
# matched label routes to a single project class under the collapse.
TEXT_LABELS_PER_IMAGE = [["a vehicle", "a car", "a truck", "a bus", "a motorcycle"]]
VEHICLE_LABEL_KEYWORDS = ("vehicle", "car", "truck", "bus", "motorcycle")

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_THRESHOLD = 0.25  # how strongly the matched phrase must align


class GroundingDinoTinyTeacher:
    name = "grounding_dino_t"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID
        ).to(self._device)
        self._model.eval()

    def predict(
        self,
        image_paths: Iterable[Path],
        conf_threshold: float = 0.3,
    ) -> list[FrameDetections]:
        import torch
        from PIL import Image

        results: list[FrameDetections] = []
        for path in (Path(p) for p in image_paths):
            image = Image.open(str(path)).convert("RGB")
            w, h = image.size

            inputs = self._processor(
                images=image,
                text=TEXT_LABELS_PER_IMAGE,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            post = self._processor.post_process_grounded_object_detection(
                outputs,
                threshold=conf_threshold,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w)],
            )[0]

            boxes = post["boxes"].cpu().numpy().astype(np.float32)  # xyxy pixels
            scores = post["scores"].cpu().numpy().astype(np.float32)
            text_labels = post.get("text_labels") or post.get("labels") or []

            if len(boxes) == 0:
                results.append(FrameDetections(image_path=path, image_hw=(h, w)))
                continue

            keep_idx: list[int] = []
            project_ids: list[int] = []
            for i, label in enumerate(text_labels):
                pid = _resolve_project_id(str(label))
                if pid is None:
                    continue
                keep_idx.append(i)
                project_ids.append(pid)

            if not keep_idx:
                results.append(FrameDetections(image_path=path, image_hw=(h, w)))
                continue

            keep_idx = np.array(keep_idx, dtype=np.int32)
            boxes = boxes[keep_idx]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
                    confs=scores[keep_idx],
                    project_class_ids=np.array(project_ids, dtype=np.int32),
                )
            )

        return results


def _resolve_project_id(label: str) -> int | None:
    """Map a Grounding DINO text label to a project class id.

    Under the single-class collapse, any of our vehicle-keyword labels
    map to project class 0 (vehicle). Return None for anything that
    doesn't mention a vehicle keyword (e.g. spurious model output).
    """
    lo = label.strip().lower()
    if not lo:
        return None
    for keyword in VEHICLE_LABEL_KEYWORDS:
        if keyword in lo:
            return 0
    return None
