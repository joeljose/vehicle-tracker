"""Teacher protocol — what every teacher backend must implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


@dataclass
class FrameDetections:
    """Detections for a single frame, in NORMALIZED xywh format.

    Coordinates are 0..1 in the image's own (W, H). Project class IDs:
        0 = vehicle (single class, see training/docs/prd.md §6.2 for rationale)
    """

    image_path: Path
    image_hw: tuple[int, int]  # (height, width) in pixels
    boxes_xywhn: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32)
    )  # (N, 4)
    confs: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )  # (N,)
    project_class_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # (N,) — all zero under the single-class collapse

    def __post_init__(self) -> None:
        n = len(self.boxes_xywhn)
        assert self.confs.shape == (n,), (
            f"confs shape {self.confs.shape} != ({n},)"
        )
        assert self.project_class_ids.shape == (n,), (
            f"class_ids shape {self.project_class_ids.shape} != ({n},)"
        )


# Single-class collapse (M8-P1.5 v2): all COCO vehicle classes map to
# project class 0 ("vehicle"). The application doesn't use per-type
# distinctions; collapsing removes the rare-class problem for bus/motorcycle
# and the SUV-vs-truck ambiguity that hurt YOLOv8x.
COCO_TO_PROJECT: dict[int, int] = {
    2: 0,  # car       -> vehicle
    3: 0,  # motorcycle -> vehicle
    5: 0,  # bus        -> vehicle
    7: 0,  # truck      -> vehicle
}
PROJECT_NAMES: dict[int, str] = {0: "vehicle"}
COCO_VEHICLE_CLASS_IDS: list[int] = sorted(COCO_TO_PROJECT.keys())  # [2, 3, 5, 7]


class Teacher(Protocol):
    """Run inference on a list of images and return detections per frame."""

    name: str

    def predict(
        self,
        image_paths: Iterable[Path],
        conf_threshold: float = 0.3,
    ) -> list[FrameDetections]:
        """Returns one FrameDetections per input path, in the same order."""
        ...
