# M8 Implementation Plan: Custom Model Training

**Milestone:** M8 — Custom model training
**Version target:** 0.8.0
**Status:** Phase A–F complete (v0.8.0-dev); Phase G (training-set review) deferred.

**Authoritative design docs:**
- `training/docs/prd.md` — product requirements, data inventory, class taxonomy, success criteria
- `training/docs/design.md` — data pipeline, auto-labeling, fine-tuning strategy, evaluation plan

This file is a lightweight pointer-plan that tracks phase status. For
rationale and technical detail, read `training/docs/`.

---

## Scope change: single-class collapse (P1.5 v2)

The original M8 plan had 4 classes (car / truck / bus / motorcycle). That
was collapsed to a single `vehicle` class in P1.5 v2 because:
- The backend tracker counts crossings, not types — no code path in
  `backend/pipeline/` branches on class.
- Bus (106 instances) and motorcycle (168 instances) are well under the
  ~1,500-instance fine-tuning comfort floor.
- YOLOv8x systematically mislabels SUVs/vans as "truck" at distance, and
  ~40% of annotation time went to taxonomic decisions with no operational
  value.

Reversible: if per-type analytics are ever needed later, a lightweight
second-stage classifier on cropped vehicle thumbnails can be added
without retraining the detector. Full rationale in
`training/docs/prd.md` §6.2.

---

## Phases

### Phase A — Single-class collapse (code + docs) ✅
Collapse all auto-labeling / training / export code to `nc=1`. Updates
to `training/scripts/teachers/*`, `training/scripts/auto_label.py`,
`training/scripts/ls_setup.py`, `training/scripts/ls_export.py`, and
`training/docs/`. `backend/deepstream_parsers/nvds_parse_yolov8.cpp`
rewritten for the (1,5,8400) output tensor and `config/deepstream/`
updated (single-line `labels.txt`, `num-detected-classes: 1`,
`pre-cluster-threshold: 0.25`).

### Phase B — Build 1000-frame comparison set ✅
`training/scripts/build_comparison_set.py` samples 1000 frames from
`data/frames_dedup/` stratified across the 3 sites: all 734 frames
from 741_73 plus 150 from Lytle and 116 from Drugmart.

### Phase C — Run all 5 teachers on the comparison set ✅
`training/scripts/run_comparison.py --teacher <name>` per teacher,
then `training/scripts/compute_comparison_stats.py` aggregates
detection counts, per-site density, and pairwise IoU agreement vs
YOLOv8x baseline into `data/comparison/stats.json`.

### Phase D — Label Studio multi-teacher predictions ✅
`training/scripts/ls_setup.py --mode comparison` creates a project
with 1000 tasks, 5 predictions per task (one per teacher,
distinct `model_version`). User flips between teachers in the LS UI.

### Phase E — Pick winning teacher ✅
**Winner: RF-DETR-M.** Chosen for best handling of the 741_73
gantry fragmentation case and highest overall detection quality.
See `training/experiments/teacher_comparison.md`.

### Phase F — Full auto-label + YOLOv8s fine-tune + export ✅
- `make train-label TEACHER=rfdetr_m` regenerates all 6,434 frames.
- `make train-student` fine-tunes YOLOv8s on the full RF-DETR-M
  auto-labels (fresh run `yolov8s_rfdetr_v1`, continue run
  `yolov8s_rfdetr_v1_cont`, final mAP50-95 = 0.7602).
- `make train-export-onnx RUN=yolov8s_rfdetr_v1_cont` exports to
  `/models/yolov8s_rfdetr_v1_cont.onnx`. Both backends load this
  ONNX at startup and build their own TRT engines.

### Phase G — Training-set review (deferred) ⏸️
Scope of human review over the full 6,434-frame auto-label output
is deferred until we've seen the current model's inference quality
on real footage. Could be full review, stratified-subset review,
or zero review depending on auto-label quality. Separate plan
after Phase F ships.

### Phase H — Retrain + final release (out of scope) 📅
Retrain with the reviewed labels (if Phase G produces corrections),
re-evaluate on the P0 test split, tag `v0.8.0`. Separate plan when
Phase G completes.

---

## Key references

- Teacher API protocol: `training/scripts/teachers/_base.py`
- Ultralytics-style teachers: `training/scripts/teachers/yolov8x.py`
- RF-DETR wrapper (1-indexed COCO): `training/scripts/teachers/rfdetr_m.py`
- Fine-tune entry point: `training/scripts/train_student.py`
- ONNX export entry point: `training/scripts/export_onnx.py`
- Custom backend ONNX consumer: `backend/pipeline/custom/engine_builder.py`
- DeepStream ONNX consumer: `config/deepstream/pgie_config.yml`
- DS YOLOv8 C++ parser: `backend/deepstream_parsers/nvds_parse_yolov8.cpp`
