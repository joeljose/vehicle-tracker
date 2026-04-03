# Experiment 10: ByteTrack vs BoT-SORT Tracker Comparison

## Date
2026-04-03

## Setup
- Container: DeepStream 8.0 backend
- GPU: RTX 4050 Mobile (6 GB VRAM)
- Model: YOLOv8s (COCO pretrained, PyTorch — not TensorRT optimized)
- Source: 741_73_1min.mp4 (1920x1080, 60fps, processing every 2nd frame = 30fps inference)
- Frames: 450 inferences (30s of video)
- Classes filtered: car(2), motorcycle(3), bus(5), truck(7)

## Results

| Metric | ByteTrack | BoT-SORT |
|--------|-----------|----------|
| Unique IDs | 50 | 48 |
| ID jumps (>200px) | 0 | 0 |
| Total detections | 6,019 | 6,028 |
| Inference FPS | 51.7 | 37.7 |
| Total time | 8.7s | 11.9s |

## Findings

1. **BoT-SORT produces 2 fewer unique IDs** — meaning 2 vehicles that ByteTrack assigned new IDs to were correctly re-identified by BoT-SORT's ReID feature. This is a modest improvement on this clip.

2. **BoT-SORT is 27% slower** (37.7 vs 51.7 fps). The ReID feature extraction adds ~2-3ms per frame. With TensorRT optimization (not used here — this was PyTorch), both trackers will be faster.

3. **Neither tracker had large ID jumps** on this clip. However, this is a 30-second sample. At intersections, vehicles stop at red lights, get occluded by cross-traffic, and resume — scenarios where ReID's appearance matching prevents ID switches that position-only matching (ByteTrack) would miss.

4. **FPS headroom for 2-channel:** ByteTrack at 51.7 fps has clear headroom (need 60 fps for 2×30fps). BoT-SORT at 37.7 fps is tight for 2-channel — batch inference (batch=2) should improve throughput vs sequential, and TensorRT will add significant speedup.

## TensorRT FP16 Benchmark (follow-up)

Exported YOLOv8s to TensorRT FP16 (`yolov8s.engine`, 23.9 MB) and re-ran all configurations.
Also tested BoT-SORT with ReID enabled (`with_reid: True`, model: `yolo26n-cls.pt`).

| Config | FPS (PyTorch) | FPS (TensorRT) | Unique IDs |
|--------|--------------|----------------|------------|
| ByteTrack | 70.2 | **102.4** | 50 |
| BoT-SORT (no ReID) | 39.0 | **42.3** | 48 |
| BoT-SORT (ReID) | 37.5 | **16.3** | 53 |

### Key Findings

1. **TensorRT speeds up ByteTrack dramatically** (70→102 fps, +46%) because the bottleneck was detection.

2. **TensorRT barely helps BoT-SORT** (39→42 fps, +8%) because the bottleneck is the GMC motion model (sparseOptFlow), not detection.

3. **ReID with TensorRT is SLOWER than PyTorch** (37.5→16.3 fps). This is because:
   - With PyTorch detector, `model: auto` reuses backbone features for free via hook
   - With TensorRT detector, `auto` falls back to a separate PyTorch model (`yolo26n-cls.pt`)
   - The ReID model runs **per-detection crop** — 15-20 separate PyTorch inference calls per frame
   - This dominates total latency

4. **ReID produced MORE unique IDs (53) than without (48)** — the opposite of expected. The appearance model may be confusing vehicles at this camera angle, causing incorrect identity splits.

## Decision
**BoT-SORT without ReID** selected for the custom pipeline. Rationale:
- 42.3 fps TensorRT — above 30fps target with headroom
- 48 unique IDs — 2 fewer than ByteTrack (better continuity)
- GMC (global motion compensation) helps with camera vibration
- ReID is counterproductive at this camera angle and destroys FPS
- Track stitcher handles remaining ID switches
- `track_buffer` can be tuned up to 150 frames (5s) for longer occlusion recovery

## GMC Impact Test (follow-up)

Tested `gmc_method: sparseOptFlow` vs `gmc_method: none` on fixed-mount traffic cameras.

| Config | FPS (TRT) | Unique IDs |
|--------|-----------|------------|
| ByteTrack | 90.3 | 50 |
| BoT-SORT + sparseOptFlow | 43.0 | 48 |
| **BoT-SORT + none (no GMC)** | **103.6** | **48** |

**GMC costs 58.5% speed for zero tracking benefit on fixed cameras.** Optical flow camera motion compensation is designed for moving/handheld cameras — fixed traffic pole mounts don't need it.

## Final Decision
**BoT-SORT with `gmc_method: none` and `with_reid: False`** — 103.6 fps TensorRT, 48 unique IDs, 3.4x headroom over 30fps target. Best tracking quality at fastest speed.
