# Product Requirements Document
## Custom Model Training — Vehicle Detector Fine-Tuning

**Version:** 0.1
**Status:** Draft
**Last updated:** April 2026
**Parent project:** Vehicle Tracker (v0.6.0)

---

## 1. Purpose

This document defines the requirements for training a custom vehicle detection model to replace the pre-trained TrafficCamNet detector used in the Vehicle Tracker system. The trained model will provide finer vehicle class granularity (distinguishing cars, trucks, buses, and motorcycles) and be tuned specifically for the three target traffic junctions.

This is a standalone subproject within the Vehicle Tracker repository. Its deliverable is a TensorRT engine file that drops into the existing pipeline.

---

## 2. Problem Statement

The current detector (TrafficCamNet) has a fundamental limitation: it classifies all vehicles as a single "car" class. There is no distinction between cars, trucks, buses, or motorcycles. This limits the system's value for traffic analysis — an analyst cannot determine vehicle type composition at a junction.

TrafficCamNet's 4 classes are: `car`, `bicycle`, `person`, `road_sign`. For a traffic monitoring system, the vehicle breakdown matters:
- Trucks and buses have different traffic impact than cars (size, speed, turning radius)
- Motorcycle counts are relevant for safety analysis
- Vehicle type distribution changes by time of day and is a key metric for traffic engineers

We have 5.2 hours of unlabeled video from our 3 target junctions. A fine-tuned model trained on this site-specific data should outperform both TrafficCamNet and a generic COCO-pretrained detector on our footage.

---

## 3. Goals

- Train a YOLOv8s model that distinguishes 4 vehicle classes: `car`, `truck`, `bus`, `motorcycle`
- Use auto-labeling with a strong teacher model (YOLOv8x) to avoid manual annotation of thousands of frames
- Achieve higher detection accuracy on our junction footage than both TrafficCamNet and pretrained YOLOv8s COCO
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

| Site | Files | Duration | Size |
|------|-------|----------|------|
| 741 & 73 | `Live Traffic @ 741 & 73.mp4` (3.7 hrs), `...23_30.mp4` (10 min), `...00_03...183336.mp4` (20 min) | ~4.2 hrs | ~1.7 GB |
| 741 & Lytle South | `...23_17.mp4` (10 min), `...00_03...183334.mp4` (20 min) | ~30 min | ~0.5 GB |
| Drugmart & 73 | `...23_19.mp4` (10 min), `...00_03...183337.mp4` (20 min) | ~30 min | ~0.4 GB |
| **Total** | **7 files** | **~5.2 hrs** | **~2.5 GB** |

All video is stored in `data_collection/site_videos/` (gitignored).

### 6.2 Target Classes

| ID | Class | COCO ID | Notes |
|----|-------|---------|-------|
| 0 | car | 2 | Sedans, SUVs, hatchbacks, pickups |
| 1 | truck | 7 | Box trucks, semis, flatbeds, trailers, long vehicles |
| 2 | bus | 5 | Transit buses, school buses |
| 3 | motorcycle | 3 | Motorcycles, scooters |

Dropped from consideration:
- **person** — system tracks vehicles through junctions, not pedestrians
- **bicycle** — not tracked through entry/exit lines at these junctions

---

## 7. Requirements

### 7.1 Data Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| D-01 | Extract frames from all 7 site videos at 1 FPS | Must |
| D-02 | Perceptual deduplication to remove near-identical frames (static camera, stopped traffic). Target: 3,000–5,000 unique frames from ~18,700 raw. | Must |
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
| mAP50 on junction test set | ≥ pretrained YOLOv8s COCO baseline | Fine-tuning on site data should not degrade overall performance |
| Car recall (including top-down views) | > pretrained YOLOv8s COCO | The key gap — overhead camera angles are underrepresented in COCO |
| Truck/bus/motorcycle recall | ≥ 70% | These are the new classes that TrafficCamNet cannot detect at all |
| TensorRT FP16 inference | ≥ 100 fps on RTX 4050 | Must not bottleneck the real-time pipeline (runs at 30 fps) |

---

## 9. Constraints

| Constraint | Details |
|------------|---------|
| Hardware | Single RTX 4050 Mobile (6 GB VRAM, Ada Lovelace) |
| Docker only | All processing in containers — no host installs |
| Data privacy | Video is from public traffic cameras — no PII concerns |
| Labeling budget | Minimal manual labeling — auto-label with teacher model, human review only for flagged frames |
| Timeline | Complete before M7 (custom pipeline) — M7 uses this model |

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
