"""Teacher model dispatch.

Each teacher implements the same `Teacher` protocol from `_base.py`. This
package exposes a single `get_teacher(name)` factory so callers don't import
specific implementations and don't need to know whether the underlying model
is ultralytics, rfdetr, or HuggingFace.
"""

from __future__ import annotations

from ._base import FrameDetections, Teacher

REGISTRY: dict[str, str] = {
    # name -> "module:ClassName"
    "yolov8x": "yolov8x:YOLOv8xTeacher",
    "yolo26x": "yolo26x:YOLO26xTeacher",
    "yolo12x": "yolo12x:YOLO12xTeacher",
    "rfdetr_m": "rfdetr_m:RFDETRMediumTeacher",
    "grounding_dino_t": "grounding_dino_t:GroundingDinoTinyTeacher",
    # The fine-tuned student we just trained — wrapped as a "teacher" so
    # run_comparison.py can evaluate it against the same 1000-frame set.
    "student_v1": "student_v1:StudentV1Teacher",
    "student_v1_cont": "student_v1_cont:StudentV1ContTeacher",
}


def get_teacher(name: str) -> Teacher:
    """Lazy-import a teacher by name and return an instance."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown teacher {name!r}. Available: {sorted(REGISTRY)}"
        )
    module_name, class_name = REGISTRY[name].split(":")
    # Lazy import — heavy ML deps shouldn't load until you actually need them
    import importlib

    module = importlib.import_module(f"teachers.{module_name}")
    cls = getattr(module, class_name)
    return cls()


__all__ = ["FrameDetections", "Teacher", "get_teacher", "REGISTRY"]
