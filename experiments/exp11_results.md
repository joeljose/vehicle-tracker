# Experiment 11: End-to-End Custom Pipeline Loop

## Date
2026-04-03

## Setup
- Container: DeepStream 8.0 backend
- GPU: RTX 4050 Mobile (6 GB VRAM)
- Detection: YOLOv8s TensorRT FP16 (`yolov8s.engine`, 23.9 MB)
- Tracking: BoT-SORT (no ReID, no GMC, `gmc_method: none`)
- Decode: OpenCV VideoCapture
- Sources: 741_73_1min.mp4 (1920x1080, 60fps), lytle_south_1min.mp4 (1920x1080, 60fps)
- Processing: every 2nd frame (60fps source → 30fps inference)

## Pipeline Stages
```
decode (cv2.VideoCapture) → infer+track (TRT + BoT-SORT) → draw bboxes (cv2) → JPEG encode (cv2.imencode)
```

## Results

### Single Channel (300 inferences)

| Stage | Avg (ms) | P50 (ms) | P99 (ms) |
|-------|----------|----------|----------|
| Decode | 1.4 | 1.3 | 3.1 |
| Infer + Track | 9.4 | 6.7 | 10.3 |
| Draw bboxes | 1.1 | 1.0 | 2.3 |
| JPEG encode | 4.1 | 3.8 | 6.9 |
| **Total** | **15.9** | **12.9** | **17.4** |
| **FPS** | **62.9** | | |

### Two Channels (150 inferences each, sequential)

| Stage | Avg (ms) | P50 (ms) | P99 (ms) |
|-------|----------|----------|----------|
| Decode (×2) | 3.3 | 2.9 | 5.7 |
| Infer + Track (×2) | 13.1 | 12.5 | 16.0 |
| Draw (×2) | 2.0 | 1.9 | 3.5 |
| JPEG encode (×2) | 8.7 | 8.2 | 12.4 |
| **Total** | **26.6** | **25.9** | **32.8** |
| **FPS** | **37.5** | | |

### GPU Memory

| Config | VRAM |
|--------|------|
| Single TRT engine | 38 MB |
| Two TRT engines (2-channel) | 152 MB |

## Findings

1. **Both single and 2-channel hit the 30fps target.** Single at 62.9 fps (2.1x headroom), dual at 37.5 fps (1.25x headroom).

2. **Latency budget at 30fps (33ms per frame):**
   - Single channel uses 15.9ms → 17.1ms free for post-processing (ROI, direction FSM, alerts, snapshots)
   - Two channels use 26.6ms → 6.4ms free for post-processing

3. **JPEG encoding is the second biggest cost** (4.1ms single, 8.7ms dual). Since MJPEG streaming only needs ~15fps, encoding every other inference frame saves ~4ms per cycle.

4. **Infer + Track dominates** at 9.4ms single / 13.1ms dual. Sequential inference (not batched) — batching could improve this.

5. **GPU memory is minimal:** 152 MB for 2-channel TRT — well within 6 GB VRAM. DeepStream's shared pipeline used ~459 MB for comparison.

6. **Detection quality looks good** on sample frame — vehicles detected at various distances and angles, track IDs assigned.

## Optimization Opportunities

- **Batch inference**: Stack 2 frames → single TRT call instead of 2 sequential calls. Requires exporting engine with batch=2.
- **JPEG on alternate frames**: Encode JPEG at 15fps instead of 30fps for MJPEG stream.
- **Skip draw on non-MJPEG frames**: Only draw bboxes when the frame will be JPEG-encoded.
- **Threaded decode**: Read frames on a separate thread to overlap with inference.

## Verdict
**P0 spike: PASS.** The full pipeline loop works end-to-end at real-time speeds for both single and dual channel operation. Tech stack validated: OpenCV decode + TensorRT YOLOv8s + BoT-SORT (Kalman only).
