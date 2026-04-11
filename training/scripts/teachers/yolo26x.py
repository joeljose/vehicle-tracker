"""YOLO26-X teacher (NMS-free, January 2026)."""

from __future__ import annotations

from .yolov8x import YOLOv8xTeacher


class YOLO26xTeacher(YOLOv8xTeacher):
    name = "yolo26x"
    _weights = "yolo26x.pt"
