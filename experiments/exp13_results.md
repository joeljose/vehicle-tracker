# Experiment 13: JPEG Encoding — GPU download + cv2.imencode

## Date
2026-04-03

## Setup
- Container: DeepStream 8.0 backend container
- GPU: RTX 4050 Mobile (6 GB VRAM)
- CuPy: 13.4.1 (GPU→CPU transfer)
- OpenCV: 4.13.0 (cv2.imencode)
- Frames decoded via NVDEC (PyNvVideoCodec 2.1.0), converted NV12→BGR on GPU
- Test clip: 741_73_1min.mp4 (1920x1080, 60fps)

## Results

### JPEG encoding at 1920x1080 (quality=80)

| Method | avg (ms) | p50 (ms) | FPS | Size (KB) |
|--------|----------|----------|-----|-----------|
| cv2.imencode (CPU numpy, baseline) | 4.15 | 3.84 | 241 | 356 |
| GPU→CPU download + cv2.imencode | 5.36 | 5.20 | 186 | 356 |

### JPEG encoding at 720p (quality=80)

| Method | avg (ms) | p50 (ms) | FPS | Size (KB) |
|--------|----------|----------|-----|-----------|
| cv2.imencode 720p (CPU numpy) | 1.84 | 1.75 | 543 | 179 |
| GPU→CPU download + cv2.imencode 720p | 2.45 | 2.30 | 409 | 179 |

### Quality vs speed at 1080p (CPU numpy)

| Quality | avg (ms) | FPS | Size (KB) |
|---------|----------|-----|-----------|
| 50 | 3.42 | 293 | 215 |
| 70 | 3.69 | 271 | 288 |
| 80 | 3.84 | 260 | 356 |
| 90 | 4.29 | 233 | 513 |

### Isolated GPU→CPU download

| Metric | Value |
|--------|-------|
| Download time (1080p RGB, 6.2 MB) | 1.29 ms |

## Findings

1. **cv2.imencode is fast enough.** At 1080p q=80: 4.15ms on CPU, 5.36ms with GPU download. MJPEG runs at ~15fps (66ms budget per frame). Even with GPU download overhead, encoding uses <10% of the frame budget.

2. **GPU→CPU download adds ~1.3ms** for a 1080p RGB frame (6.2 MB PCIe transfer). This is the cost of not having GPU JPEG encoding.

3. **nvJPEG is not needed.** The CPU encoding path (GPU→CPU download + cv2.imencode) at 186 fps is 12x faster than the 15fps MJPEG requirement. Adding nvJPEG complexity is not justified.

4. **Quality 70 is a good tradeoff** — saves 19% bandwidth vs q=80 with minimal visual impact (3.69ms vs 3.84ms, negligible speed difference).

5. **720p encoding is 2x faster** (1.84ms vs 4.15ms). If bandwidth becomes a concern, downscaling before encoding is an option.

## Decision

**Use CPU encoding: GPU→CPU download + cv2.imencode at quality 80.** The 5.36ms total cost per MJPEG frame is well within the 66ms budget at 15fps. nvJPEG adds complexity for no meaningful gain at this output rate.

For the MJPEG path in the custom pipeline:
1. Every other inference frame (15fps), download the GPU frame to CPU (~1.3ms)
2. Draw bboxes with OpenCV on CPU (~1.1ms, from exp11)
3. Encode JPEG with cv2.imencode at q=80 (~4.1ms)
4. Total: ~6.5ms per MJPEG frame — 10% of the 66ms budget

The remaining 90% of the frame budget is available for inference + tracking + analytics.
