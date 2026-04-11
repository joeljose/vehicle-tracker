"""Fine-tuned YOLOv8s student — continuation training checkpoint.

Same architecture and training data as student_v1, but trained for an
additional ~18 effective epochs at a 10x lower learning rate. Final val
metrics: mAP50-95 = 0.7602 (+0.0025 over v1), precision = 0.9321.

Keeping both StudentV1Teacher and StudentV1ContTeacher around lets the
1000-frame pairwise-agreement comparison score them separately against
RF-DETR so we can verify that the small val improvement survives at
real-data scale before committing to which one ships.
"""

from __future__ import annotations

from .student_v1 import StudentV1Teacher


class StudentV1ContTeacher(StudentV1Teacher):
    name = "student_v1_cont"
    weights = "/app/runs/detect/yolov8s_rfdetr_v1_cont/weights/best.pt"
