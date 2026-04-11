# Product Requirements Document
## Custom Model Training — Vehicle Detector Fine-Tuning

**Version:** 0.1
**Status:** Draft
**Last updated:** April 2026
**Parent project:** Vehicle Tracker (v0.7.0)

---

## 1. Purpose

This document defines the requirements for fine-tuning the YOLOv8s vehicle detection model used in the Vehicle Tracker system. The fine-tuned model will be trained specifically on our three target traffic junctions, improving detection accuracy (especially for overhead camera angles) compared to the generic COCO-pretrained weights currently used by both backends.

This is a standalone subproject within the Vehicle Tracker repository. Its deliverable is a TensorRT engine file that drops into the existing pipeline.

---

## 2. Problem Statement

Both backends currently use YOLOv8s with COCO-pretrained weights, detecting 4 vehicle classes (car, truck, bus, motorcycle). While this works, the COCO-pretrained model has gaps on our junction footage:
- Overhead/top-down camera angles are underrepresented in COCO training data — missed detections
- Junction-specific vehicle appearances (stopped vehicles, partial occlusions) not well covered
- Fine-tuning on site-specific data should improve recall significantly

We have 10 hours of unlabeled video from our 3 target junctions (balanced: 200 min per site). A fine-tuned model trained on this site-specific data should outperform the generic COCO-pretrained weights on our footage.

---

## 3. Goals

- Train a YOLOv8s model that distinguishes 4 vehicle classes: `car`, `truck`, `bus`, `motorcycle`
- Use auto-labeling with a strong teacher model (YOLOv8x) to avoid manual annotation of thousands of frames
- Achieve higher detection accuracy on our junction footage than pretrained YOLOv8s COCO
- Export the model to TensorRT FP16 for deployment in the existing pipeline
- Build a reproducible training pipeline (scripts, configs, documentation) so the model can be retrained as new video data is collected
- Keep all training inside Docker containers — nothing installed on host

---

## 4. Non-Goals

- Training from scratch (we fine-tune from COCO-pretrained weights)
- Real-time or online learning during pipeline operation
- Multi-GPU or distributed training
- Night-time footage (v1 targets daytime only, matching the parent project)
- License plate recognition or vehicle re-identification
- INT8 quantization (start with FP16; INT8 calibration is a future optimization)
- Deploying CVAT or any annotation tool in production — it's a development-time dependency only
- Modifying the Vehicle Tracker pipeline code (model is a drop-in replacement via config change)

---

## 5. Users

### Primary user — developer / ML engineer
The person running the training pipeline. Comfortable with Python, Docker, and ML tooling. Will extract frames, review auto-labels, run training, and evaluate results.

### Secondary user — operator (downstream)
Uses the Vehicle Tracker system with the improved model. Does not interact with the training pipeline directly. Benefits from better detection accuracy and vehicle type classification.

---

## 6. Data Inventory

### 6.1 Available Video

Inventory captured 2026-04-10. Each junction has 7 videos (1×20 min + 6×30 min) for a balanced 200 min / site.

| Site | Files | Duration | Size |
|------|-------|----------|------|
| 741 & 73 | 7 (1×20 min + 6×30 min) | 200 min | ~2.6 GB |
| 741 & Lytle South | 7 (1×20 min + 6×30 min) | 200 min | ~3.4 GB |
| Drugmart & 73 | 7 (1×20 min + 6×30 min) | 200 min | ~2.4 GB |
| **Total** | **21 files** | **600 min (10 hrs)** | **~8.4 GB** |

All video is stored in `data_collection/site_videos/` (gitignored). Filename → site mapping is recorded in `training/configs/video_sites.yaml`.

### 6.2 Target Classes

**M8-P1.5 v2 scope change (2026-04-11):** the detection taxonomy has been collapsed from 4 classes (car/truck/bus/motorcycle) to a single class. The Vehicle Tracker backend counts crossings, not vehicle types — no code path in `backend/pipeline/` branches on class, and no stakeholder has requested per-type analytics. The 4-class split cost us marginal bus (106) and motorcycle (168) instance counts well under the 1,500-instance fine-tuning comfort floor, plus significant annotation time on SUV-vs-truck taxonomic calls that the application doesn't use. If per-type analytics are ever needed later, they can be added as a lightweight second-stage classifier on cropped vehicle thumbnails without retraining the detector. See `/home/joel/.claude/plans/whimsical-doodling-aurora.md` for the full rationale.

| ID | Class | Includes | Source COCO IDs (all collapse here) |
|----|-------|----------|--------------------------------------|
| 0 | vehicle | sedans, SUVs, hatchbacks, pickups, box trucks, semis, trailers, transit/school buses, motorcycles, scooters | 2 (car), 3 (motorcycle), 5 (bus), 7 (truck) |

Dropped from consideration:
- **person** — system tracks vehicles through junctions, not pedestrians
- **bicycle** — not tracked through entry/exit lines at these junctions

---

## 7. Requirements

### 7.1 Data Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| D-01 | Extract frames from all 21 site videos at 1 FPS | Must |
| D-02 | Perceptual deduplication to remove near-identical frames (static camera, stopped traffic). Target: 5,000–10,000 unique frames from ~36,000 raw. | Must |
| D-03 | Auto-label all deduplicated frames using YOLOv8x (COCO pretrained) as teacher model | Must |
| D-04 | Filter auto-labels to target 4 vehicle classes only, remap COCO IDs to project class IDs | Must |
| D-05 | Split auto-labels by confidence: ≥0.5 accepted directly, 0.3–0.5 flagged for human review | Must |
| D-06 | Export labels in YOLO format (one .txt per image, normalized xywh) | Must |
| D-07 | All frame extraction and auto-labeling runs inside Docker | Must |

### 7.2 Label Review

| ID | Requirement | Priority |
|----|-------------|----------|
| L-01 | Provide a way to review and correct auto-labels (CVAT or equivalent, running in Docker) | Must |
| L-02 | Review prioritization: low-confidence frames first, then rare classes (truck, bus, motorcycle), then random sample of high-confidence frames | Should |
| L-03 | Reviewer can add missed detections, correct class labels, and adjust bounding boxes | Must |
| L-04 | Full skim pass across all frames to catch vehicles the teacher model missed — especially top-down/overhead views of cars which are poorly represented in COCO training data | Must |
| L-05 | Export corrected labels in YOLO format | Must |

### 7.3 Evaluation Baseline

| ID | Requirement | Priority |
|----|-------------|----------|
| E-01 | Create a manually-annotated held-out test set of 200 frames (balanced across 3 sites) — never used for training | Must |
| E-02 | Run TrafficCamNet on the test set and compute mAP50 as baseline 1 | Must |
| E-03 | Run pretrained YOLOv8s (COCO, no fine-tuning) on the test set as baseline 2 | Must |
| E-04 | All evaluation metrics computed on the same held-out test set for fair comparison | Must |

### 7.4 Training

| ID | Requirement | Priority |
|----|-------------|----------|
| T-01 | Fine-tune YOLOv8s from COCO pretrained weights | Must |
| T-02 | Training must fit in 6 GB VRAM (RTX 4050 Mobile) | Must |
| T-03 | Dataset split: 70% train / 15% val / 15% test, stratified by site | Must |
| T-04 | Training uses frozen backbone (first 10 layers) initially, then full fine-tuning | Should |
| T-05 | Data augmentation: mosaic, HSV jitter. No horizontal/vertical flip (fixed camera angle) | Must |
| T-06 | Early stopping with patience of 20 epochs | Should |
| T-07 | All training runs inside Docker with GPU access | Must |

### 7.5 Evaluation

| ID | Requirement | Priority |
|----|-------------|----------|
| V-01 | Compute mAP50 and mAP50-95 on held-out test set | Must |
| V-02 | Compute per-class precision and recall | Must |
| V-03 | Compare against both baselines (TrafficCamNet and pretrained YOLOv8s COCO) | Must |
| V-04 | Benchmark TensorRT FP16 inference speed on RTX 4050 | Must |
| V-05 | Qualitative evaluation: run inference on 1-min clips from each site, visual inspection | Must |

### 7.6 Export & Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| X-01 | Export best model to TensorRT FP16 engine | Must |
| X-02 | Engine file drops into `models/` directory for use by the Vehicle Tracker pipeline | Must |
| X-03 | Update DeepStream nvinfer config to use new model (class count, label file, bbox parser) | Should |
| X-04 | Document config changes needed for both DeepStream and custom pipeline integration | Must |

---

## 8. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Overall mAP50 on junction test set | ≥ pretrained YOLOv8s COCO baseline (vehicle-class equivalent) | Fine-tuning on site data should not degrade overall performance |
| Vehicle recall (including top-down views) | > pretrained YOLOv8s COCO | The key gap — overhead camera angles are underrepresented in COCO |
| Fragmentation index in the 741_73 gantry zone | Lower than YOLOv8x baseline | The known failure mode the teacher-bake-off is solving |
| TensorRT FP16 inference | ≥ 100 fps on RTX 4050 | Must not bottleneck the real-time pipeline (runs at 30 fps) |

---

## 9. Constraints

| Constraint | Details |
|------------|---------|
| Hardware | Single RTX 4050 Mobile (6 GB VRAM, Ada Lovelace) |
| Docker only | All processing in containers — no host installs |
| Data privacy | Video is from public traffic cameras — no PII concerns |
| Labeling budget | Minimal manual labeling — auto-label with teacher model, human review only for flagged frames |
| Timeline | M8 milestone — both backends already use YOLOv8s COCO; fine-tuned model is a drop-in upgrade |

---

## 10. Milestones

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| P0 — Evaluation baseline | 200 manually-annotated test frames, TrafficCamNet + YOLOv8s COCO baseline metrics | None |
| P1 — Auto-labeling pipeline | Frame extraction, deduplication, auto-labeled dataset (~3,000–5,000 frames) | P0 |
| P2 — Label review | Reviewed and corrected labels, exported in YOLO format | P1 |
| P3 — Dataset preparation | Train/val/test splits, dataset YAML, augmentation config | P2 |
| P4 — Training | Fine-tuned YOLOv8s model, training logs and metrics | P3 |
| P5 — Evaluation | Evaluation report comparing fine-tuned model vs both baselines | P4 |
| P6 — Export & integration | TensorRT FP16 engine, integration documentation | P5 |

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Teacher model misses top-down/overhead car views (COCO bias toward street-level angles) | High | High | Manual review pass catches missed vehicles. These manually-added annotations are the most valuable training examples — they teach the model exactly the cases COCO-pretrained models fail on. |
| Auto-labels have systematic errors (SUVs labeled as trucks) | Medium | Medium | Review all rare-class labels. YOLOv8x is strong on COCO vehicle classes. |
| 6 GB VRAM insufficient for training | Low | High | YOLOv8s at batch 8, 640x640 fits. Reduce batch to 4 if needed. |
| Fine-tuned model overfits to 3 junctions | Medium | Low | Acceptable — the model only needs to work at these junctions. Augmentation mitigates. |
| Trucks/buses too rare in dataset for reliable training | Medium | Medium | Oversample rare classes during training. If insufficient, collect more video from these junctions during heavy truck hours. |
| Custom nvinfer bbox parser for YOLO output is complex | Medium | Medium | Well-documented by NVIDIA (DeepStream sample apps include YOLO parsers). Can defer DeepStream integration and use model only with custom pipeline (M7). |

---

## 12. Out of Scope for v1

- INT8 quantization and calibration (future optimization after FP16 is validated)
- Active learning loop (automatically flagging low-confidence production detections for retraining)
- Model versioning system (DVC, MLflow, etc.) — git tags and manual tracking sufficient at this scale
- Training on night-time or adverse weather footage
- Multi-class tracking (current tracker is class-agnostic; adding class-aware tracking is a pipeline change, not a model change)
