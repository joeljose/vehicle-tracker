# Custom Model Training — Design Document

## 1. Overview

This document describes the technical design for fine-tuning a YOLOv8s vehicle detector on site-specific traffic junction footage. The fine-tuned model improves upon the COCO-pretrained YOLOv8s currently used by both backends, especially for overhead camera angles and junction-specific vehicle appearances.

**Input:** 5.2 hours of unlabeled video from 3 junctions (7 files, 2.5 GB)
**Output:** TensorRT FP16 engine file, drop-in replacement for `models/yolov8s.onnx` (both backends)
**Classes:** 4 — `car`, `truck` (includes trailers), `bus`, `motorcycle`

---

## 2. Hardware Constraints

| Property | Value | Impact on Training |
|----------|-------|--------------------|
| GPU | RTX 4050 Mobile (Ada Lovelace) | 6 GB VRAM limits batch size |
| VRAM | 6 GB GDDR6 | YOLOv8s batch 8 @ 640x640 ≈ 4.5 GB — fits with margin |
| NVDEC | Gen 5 | Not used for training (CPU decode via ffmpeg) |
| TensorRT | 10.x | FP16 export target |
| Storage | ~50 GB free needed | Extracted frames (~10 GB), training runs (~5 GB per run) |

---

## 3. Model Selection

### Why YOLOv8s

| Consideration | Decision | Rationale |
|---------------|----------|-----------|
| Architecture | YOLOv8s (11.2M params) | Best accuracy/VRAM trade-off for 6 GB. Already specified in parent project PRD for custom pipeline (M7). |
| Not YOLOv8n | Too small (3.2M) | 7.6 mAP50 gap vs YOLOv8s. Not worth the speed gain — pipeline runs at 30 fps, both models exceed 100 fps. |
| Not YOLOv8m | Too large (25.9M) | Requires batch ≤4 on 6 GB, slowing convergence. Only 5.3 mAP50 gain over YOLOv8s. |
| Not RT-DETR | Transformer overhead | 60 fps TRT estimate is marginal for multi-channel. YOLO's CNN backbone is more predictable. |
| Framework | Ultralytics | Standard YOLOv8 training framework. Handles augmentation, scheduling, export. Single `pip install ultralytics`. |

### Pretrained Weights

Start from `yolov8s.pt` (COCO pretrained, 80 classes, 44.9 mAP50-95). Fine-tune on our 6-class subset. The COCO pretraining already includes all our target classes — we're specializing, not teaching new concepts.

---

## 4. Auto-Labeling Architecture

### Why Auto-Label

Manual annotation of 5,000 frames × ~15 objects per frame = ~75,000 bounding boxes. At ~2 seconds per box, that's ~42 hours of manual work. Auto-labeling with a strong teacher model reduces this to reviewing only problematic frames (~5-8 hours).

### Teacher Model: YOLOv8x

| Property | Value |
|----------|-------|
| Model | YOLOv8x (COCO pretrained) |
| Params | 68.2M |
| mAP50-95 (COCO) | 53.9 |
| Role | Generate pseudo-labels on unlabeled frames |
| Inference mode | Batch 1 (inference only, fits 6 GB) |

YOLOv8x is 9 mAP points stronger than YOLOv8s on COCO. It serves as a "teacher" — its predictions become training labels for the smaller "student" (YOLOv8s). This is a standard knowledge distillation approach.

**Known limitation:** The teacher model shares COCO's bias toward street-level camera angles. Top-down/overhead views of cars (common in our traffic junction cameras) will have lower detection rates. The human review pass (Section 6) compensates for this by manually annotating missed vehicles — these are the most valuable training examples.

### Auto-Label Pipeline

```
site_videos/*.mp4
    │
    ▼ [extract_frames.py]
    │  ffmpeg -vf "fps=1" → ~18,700 raw frames
    │
    ▼ [dedup_frames.py]
    │  CNN perceptual hashing → remove near-duplicates
    │  Target: 3,000–5,000 unique frames
    │
    ▼ [auto_label.py]
    │  YOLOv8x inference @ conf≥0.3
    │  Filter to classes: car(2), truck(7), bus(5), motorcycle(3)
    │  Remap COCO IDs → project IDs (0-3)
    │  Output: YOLO format .txt per image
    │
    ▼ Confidence split
       ├── ≥0.5: accepted (training set)
       └── 0.3–0.5: flagged for human review
```

### COCO-to-Project Class Mapping

| COCO Class | COCO ID | Project Class | Project ID |
|------------|---------|---------------|------------|
| car | 2 | car | 0 |
| truck | 7 | truck | 1 |
| bus | 5 | bus | 2 |
| motorcycle | 3 | motorcycle | 3 |

All other COCO classes (76 remaining, including person and bicycle) are discarded. The truck class includes trailers and long vehicles — COCO's "truck" label covers these.

---

## 5. Frame Extraction Strategy

### Extraction Rate: 1 FPS

Traffic cameras are fixed-angle with slow scene changes. At 30 fps source rate, consecutive frames are nearly identical. 1 FPS captures all meaningful scene changes (vehicle arrivals/departures take 2-10 seconds to cross the junction).

```
5.2 hours × 3600 sec/hr × 1 frame/sec = 18,720 raw frames
```

### Deduplication

Fixed traffic cameras produce long stretches of near-identical frames (red lights, empty roads, stalled traffic). Perceptual deduplication using CNN feature hashing removes these.

**Method:** `imagededup` library with MobileNetV3 features. Cosine similarity threshold of 0.95 — frames above this are considered duplicates.

**Expected yield:** 3,000–5,000 unique frames (70-85% reduction). The exact number depends on traffic density and signal timing.

### Site Balance

The 741 & 73 junction has ~4.2 hours vs ~30 min each for the other two. To prevent the model from overfitting to one junction:
- Extract all frames from all videos at 1 FPS
- During dataset splitting, stratify by site (each split contains proportional representation)
- If the imbalance is extreme (>10:1), subsample the dominant site to a 5:1 ratio

---

## 6. Label Review Strategy

### What Gets Reviewed

Not all auto-labels need human review. Prioritize by error likelihood:

| Category | Estimated Count | Review Priority |
|----------|----------------|-----------------|
| Low confidence (0.3–0.5) | ~500–800 frames | High — most likely errors |
| Rare classes (truck, bus, motorcycle) | ~200–400 frames | High — small sample, errors are costly |
| Full skim for missed vehicles | All frames (quick pass) | **Critical** — catches top-down cars the teacher missed. These are the most valuable training examples. |
| Random high-confidence sample | 200 frames | Medium — detailed quality check |

**Total review effort:** The full skim is fast (~2-3 seconds per frame to spot missing boxes) but covers all frames. Detailed corrections focus on the ~700–1,200 flagged/rare frames.

### The Top-Down Detection Gap

The teacher model (YOLOv8x) was trained on COCO — mostly street-level photos with side views of vehicles. Our traffic cameras look down from above, showing roofs, hoods, and windshields. The teacher will:
- **Detect well:** vehicles approaching/leaving (angled view shows side profile)
- **Detect poorly:** vehicles directly below the camera (pure top-down, roof only)
- **Miss entirely:** some overhead cars, especially small/dark vehicles

The manual pass to add these missed detections is what makes the fine-tuned model better than any COCO-pretrained model. Without it, the training data has the same blind spots as the teacher.

### Review Tool: CVAT

CVAT (Computer Vision Annotation Tool) runs in Docker, supports YOLO format import/export, and provides a browser-based UI for bounding box editing.

```bash
# CVAT runs as a separate docker-compose stack (not part of vehicle tracker)
# Import auto-labels, review, export corrected labels
```

### Review Actions

For each flagged frame, the reviewer can:
1. **Accept** — auto-label is correct, no changes
2. **Correct class** — change label (e.g., SUV mislabeled as truck → car)
3. **Adjust box** — resize/reposition bounding box
4. **Add missed** — draw new box for vehicles the teacher missed
5. **Delete false positive** — remove incorrect detection (e.g., road sign detected as vehicle)

---

## 7. Dataset Structure

### Directory Layout

```
training/data/
  frames/                    # all deduplicated frames (JPG)
  labels/                    # YOLO format labels (.txt per frame)
  splits/
    train/
      images/                # symlinks to frames/
      labels/                # symlinks to labels/
    val/
      images/
      labels/
    test/
      images/                # 200 manually-annotated frames
      labels/                # ground truth annotations
  dataset.yaml               # Ultralytics dataset config
```

### Split Ratios

| Split | Percentage | Source | Purpose |
|-------|-----------|--------|---------|
| Train | 70% | Auto-labeled + reviewed | Model training |
| Val | 15% | Auto-labeled + reviewed | Epoch-level evaluation, early stopping |
| Test | 15% | Manually annotated (P0) | Final evaluation, never seen during training |

Splits are stratified by site — each split contains frames from all 3 junctions in proportion.

### Dataset YAML

```yaml
path: /data/splits
train: train/images
val: val/images
test: test/images

names:
  0: car
  1: truck
  2: bus
  3: motorcycle
```

---

## 8. Training Configuration

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | yolov8s.pt | COCO pretrained, 11.2M params |
| Image size | 640×640 | Standard YOLO input. Source video is 1920x1080, downscaled. |
| Batch size | 8 | Fits 6 GB VRAM with ~1.5 GB headroom |
| Epochs | 100 (max) | Early stopping at patience=20 |
| Optimizer | SGD (Ultralytics default) | Momentum=0.937, weight_decay=0.0005 |
| Learning rate | 0.01 → 0.0001 (cosine decay) | Ultralytics default schedule |
| Warmup | 3 epochs | Standard YOLO warmup |

### Two-Phase Fine-Tuning

**Phase 1 (epochs 1–50):** Freeze backbone (first 10 layers). Only train the detection head and neck. This preserves the low-level feature extraction learned from COCO while adapting the high-level detection to our junction scenes.

**Phase 2 (epochs 51–100):** Unfreeze all layers, reduce learning rate to 0.001. Full fine-tuning allows the backbone to adapt to junction-specific visual patterns (camera angle, lighting, vehicle appearance).

If Phase 1 converges early (val loss plateaus before epoch 50), skip directly to Phase 2.

### Data Augmentation

| Augmentation | Value | Rationale |
|-------------|-------|-----------|
| Mosaic | 1.0 | Combines 4 images — effective for small objects and varying object counts |
| HSV hue | 0.015 | Handles minor lighting variation |
| HSV saturation | 0.7 | Day-to-day color variation |
| HSV value | 0.4 | Brightness variation (clouds, time of day) |
| Horizontal flip | 0.0 (disabled) | Fixed camera angle — flipping changes directional semantics |
| Vertical flip | 0.0 (disabled) | Same reason |
| Scale | 0.5 | Simulates vehicles at different distances |
| Translation | 0.1 | Minor position variation |

### What NOT to Augment

- **No rotation** — traffic cameras are fixed-mount, rotation creates unrealistic views
- **No cutout/erasing** — vehicles are the only objects of interest, erasing them defeats the purpose
- **No mixup** — can confuse class boundaries for similar classes (car vs truck)

---

## 9. Evaluation Methodology

### Baselines

| Baseline | Model | Classes | How to Evaluate |
|----------|-------|---------|-----------------|
| B1 — TrafficCamNet | DetectNet_v2 (ResNet18) | 4 (car, bicycle, person, road_sign) | Map "car" predictions to all vehicle types. Compute mAP50 on the 200-frame test set. TrafficCamNet cannot distinguish vehicle types — its per-class recall for truck/bus/motorcycle is definitionally 0%. |
| B2 — YOLOv8s COCO | YOLOv8s pretrained | 80 (filter to 4) | Run inference on test set, filter to our 4 vehicle classes, remap IDs. This is the "no fine-tuning" baseline — shows the value of site-specific training, especially for overhead camera angles. |

### Metrics

| Metric | What it Measures |
|--------|-----------------|
| mAP50 | Overall detection quality at IoU≥0.5 |
| mAP50-95 | Stricter localization quality |
| Per-class recall | Detection rate for each vehicle type — critical for rare classes |
| Per-class precision | False positive rate per class |
| FPS (TensorRT FP16) | Inference speed — must exceed 100 fps |

### Evaluation Protocol

1. Run inference on the 200-frame manually-annotated test set
2. Compute all metrics using Ultralytics' built-in `model.val()` with the test split
3. Generate confusion matrix to identify systematic misclassifications
4. Qualitative check: run on 1-min clips from each junction, visually verify detections

---

## 10. TensorRT Export

### Export Pipeline

```python
from ultralytics import YOLO

model = YOLO("runs/vehicle_detect/best.pt")
model.export(
    format="engine",
    half=True,          # FP16
    imgsz=640,
    device=0,
    workspace=4,        # GB workspace for TensorRT optimization
)
# Output: best.engine (~15-20 MB)
```

### Integration with Vehicle Tracker

The exported engine is a drop-in replacement. Two integration paths:

**Path A — DeepStream pipeline (existing):**
- Requires a custom `nvinfer` bounding box parser for YOLO output format
- NVIDIA provides example YOLO parsers in DeepStream sample apps
- Update `config/deepstream/pgie_config.yml`:
  - `onnx-file` → path to YOLOv8s ONNX
  - `model-engine-file` → path to new TensorRT engine
  - `num-detected-classes` → 4
  - `custom-lib-path` → path to custom bbox parser .so
  - `parse-bbox-func-name` → parser function name
- Update `config/deepstream/labels.txt` → 4 class names

**Path B — Custom pipeline (M7, planned):**
- Simpler — Ultralytics' built-in TensorRT inference or direct TensorRT Python API
- No custom parser needed
- Just load the engine and run inference

**Recommendation:** Target Path B first (simpler, aligns with M7 timeline). Path A can be done later as an enhancement to the DeepStream pipeline.

---

## 11. Docker Configuration

### Training Container

A dedicated Dockerfile for the training subproject. Separate from the backend container because:
- Different base image (no need for DeepStream/GStreamer)
- Different Python dependencies (ultralytics, imagededup, not fastapi)
- Training workloads have different resource profiles

```dockerfile
# training/Dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Python + pip
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

# ML dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
```

### Requirements

```
ultralytics>=8.3
imagededup>=0.3
opencv-python-headless>=4.10
numpy>=1.26
```

### Docker Compose Service

Added to `docker-compose.dev.yml`:

```yaml
training:
  build:
    context: ./training
    dockerfile: Dockerfile
  runtime: nvidia
  volumes:
    - ./training:/app
    - ./data_collection:/data
    - ./models:/models
  user: "${DS_UID}:${DS_GID}"
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
  stdin_open: true
  tty: true
```

### Makefile Targets

```makefile
train-shell:     ## Shell into training container
train-extract:   ## Extract frames from site videos
train-dedup:     ## Deduplicate extracted frames
train-label:     ## Auto-label with YOLOv8x teacher
train-run:       ## Run training
train-eval:      ## Evaluate model on test set
train-export:    ## Export to TensorRT FP16
```

---

## 12. Script Specifications

### extract_frames.py

**Input:** Video files in `/data/site_videos/`
**Output:** JPEG frames in `/app/data/frames/{site_name}/frame_{N:06d}.jpg`
**Method:** `ffmpeg -i input.mp4 -vf "fps=1" -q:v 2 output/frame_%06d.jpg`
**Metadata:** Writes `data/frames/manifest.json` with frame count per site and total.

### dedup_frames.py

**Input:** `data/frames/` directory
**Output:** `data/frames_dedup/` (deduplicated frames, flat directory)
**Method:** MobileNetV3 CNN features → cosine similarity → threshold 0.95
**Metadata:** Writes `data/frames_dedup/manifest.json` with counts (before/after per site, duplicates removed).

### auto_label.py

**Input:** `data/frames_dedup/` directory
**Output:** `data/labels/` (YOLO format .txt per image), `data/labels/manifest.json`
**Method:**
1. Load YOLOv8x
2. Inference on all frames at conf≥0.3
3. Filter to 4 vehicle classes (car, truck, bus, motorcycle), remap COCO IDs
4. Write YOLO format labels
5. Split by confidence: `data/labels/accepted/` (≥0.5), `data/labels/flagged/` (0.3–0.5)
**Metadata:** Per-class detection counts, confidence distribution, flagged frame count.

### evaluate.py

**Input:** Model path, test set path
**Output:** Metrics JSON + confusion matrix PNG
**Method:** Ultralytics `model.val()` on test split, plus custom per-class reporting.

### export_engine.py

**Input:** Best model checkpoint path
**Output:** TensorRT FP16 engine in `models/`
**Method:** Ultralytics `model.export(format="engine", half=True)`

---

## 13. File Inventory

All files that will be created by this subproject:

```
training/
  docs/
    prd.md                    # this PRD
    design.md                 # this design doc
  Dockerfile
  requirements.txt
  scripts/
    extract_frames.py
    dedup_frames.py
    auto_label.py
    evaluate.py
    export_engine.py
  configs/
    vehicle_detect.yaml       # dataset config
  data/                       # gitignored
    frames/                   # raw extracted frames
    frames_dedup/             # after deduplication
    labels/                   # auto-generated labels
      accepted/               # conf ≥ 0.5
      flagged/                # conf 0.3–0.5, needs review
    splits/                   # train/val/test splits
  runs/                       # gitignored — training runs
  README.md
```

Parent project changes:
- `.gitignore` — add training/data/*, training/runs/*
- `docker-compose.dev.yml` — add training service
- `Makefile` — add train-* targets
- `models/` — new engine file after training
- `config/deepstream/pgie_config.yml` — updated model reference (Path A only)
- `config/deepstream/labels.txt` — 4 class names (Path A only)
