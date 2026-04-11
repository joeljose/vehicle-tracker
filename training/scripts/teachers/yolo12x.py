"""YOLO12-X teacher (attention-centric, 2025)."""

from __future__ import annotations

from .yolov8x import YOLOv8xTeacher


class YOLO12xTeacher(YOLOv8xTeacher):
    name = "yolo12x"
    _weights = "yolo12x.pt"
