# Experiment 12: GPU-Resident Pipeline (NVDEC + Direct TensorRT)

## Date
2026-04-03

## Setup
- Container: DeepStream 8.0 backend container
- GPU: RTX 4050 Mobile (6 GB VRAM)
- TensorRT: 10.16.0.72 (pip), engine built with pip TRT Python API
- PyNvVideoCodec: 2.1.0 (NVDEC hardware decode)
- CuPy: 13.4.1 (GPU preprocess: NV12→RGB, resize, normalize)
- Tracker: standalone BoT-SORT via ultralytics.trackers (no ReID, no GMC)
- NMS: CPU-side (numpy), postprocess from (1, 84, 8400) TRT output
- Test clips: 741_73_1min.mp4 (1920x1080, 60fps), lytle_south_1min.mp4

## Architecture
```
GPU:  NVDEC decode → CuPy NV12→RGB → CuPy resize+normalize → TRT inference → bbox download (1KB)
CPU:  NMS → BoT-SORT tracking
```
Frame never leaves GPU from decode through inference. Only GPU→CPU transfer is the raw output tensor (1, 84, 8400) = ~2.7 MB float32 for NMS.

## Results

### NVDEC decode only (300 frames, 1920x1080)

| Metric | Value |
|--------|-------|
| FPS | 902 |
| ms/frame | 1.11 |

### Single channel (300 inferences)

| Stage | avg (ms) | p50 (ms) | p99 (ms) |
|-------|----------|----------|----------|
| decode | 0.55 | 0.68 | 0.96 |
| preprocess | 1.13 | 1.01 | 1.53 |
| infer | 1.63 | 1.54 | 2.93 |
| nms | 1.77 | 1.56 | 4.26 |
| track | 0.89 | 0.83 | 2.75 |
| **total** | **6.27** | **5.86** | **8.96** |

**FPS: 159.4** — 5.3x headroom over 30fps target.
Unique IDs: 30.

### Two channels (150 inferences each, sequential inference)

| Stage | avg (ms) | p50 (ms) | p99 (ms) |
|-------|----------|----------|----------|
| decode (both) | 1.12 | 0.96 | 1.89 |
| preprocess (both) | 2.03 | 2.01 | 2.57 |
| infer (both, sequential) | 4.85 | 4.31 | 6.90 |
| nms (both) | 2.04 | 1.89 | 3.88 |
| track (both) | 1.56 | 1.45 | 3.49 |
| **total** | **12.11** | **11.40** | **14.76** |

**FPS: 82.6 total (41.3 per channel)** — 1.38x headroom over 30fps per channel target.
Unique IDs: ch0=22, ch1=27.
GPU memory: 210 MB for 2-channel.

## Comparison vs exp11 (Ultralytics wrapper, CPU↔GPU bouncing)

| Metric | exp11 (Ultralytics) | exp12 (GPU-resident) | Speedup |
|--------|-------------------|---------------------|---------|
| Single channel FPS | 62.9 | **159.4** | **2.5x** |
| Single channel ms | 15.9 | **6.3** | 2.5x |
| Dual channel FPS | 37.5 | **82.6** | **2.2x** |
| Dual channel ms | 26.6 | **12.1** | 2.2x |
| Per-channel (dual) | 18.75 fps (**FAIL**) | 41.3 fps (**PASS**) | Meets target |
| GPU memory (dual) | 152 MB | 210 MB | +58 MB |

## Findings

1. **GPU-resident pipeline is 2.2-2.5x faster than Ultralytics wrapper.** Eliminating CPU↔GPU frame transfers and Ultralytics overhead is the primary win.

2. **NVDEC decode (902 fps) is faster than OpenCV VideoCapture (780 fps).** Frames stay on GPU as NV12 CUDA surfaces — no PCIe transfer for decode.

3. **Direct TRT inference is 1.63ms per frame** vs ~5-8ms through Ultralytics. The wrapper adds significant overhead beyond just the PCIe transfer.

4. **CuPy GPU preprocess works** — NV12→RGB conversion via custom CUDA kernel, resize via cupyx.scipy.ndimage.zoom, normalize to float32. Total: 1.13ms.

5. **CPU NMS is acceptable at 1.77ms** for ~10-20 detections per frame. GPU NMS (EfficientNMS plugin) could save ~1ms but adds export complexity.

6. **Standalone BoT-SORT works** with a simple DetectionResults wrapper. No need to reimplement — just adapt the interface.

7. **Dual channel meets 30fps target** with 1.38x headroom. Not as much headroom as single channel (5.3x), but sufficient for production.

8. **GPU memory is modest at 210 MB** for 2-channel (3.5% of 6 GB VRAM). Plenty of room.

## TRT version note

The DeepStream container has two TensorRT installations:
- System: TRT 10.9 (`/usr/lib/x86_64-linux-gnu/libnvinfer.so.10.9.0`, `trtexec v100900`)
- Pip: TRT 10.16 (`tensorrt_cu13` package, Python bindings)

Engines built with one version cannot be loaded by the other. The experiment uses pip TRT 10.16 for both building and loading the engine. The production pipeline should standardize on one version.

## Decision

**GPU-resident architecture validated.** NVDEC + direct TensorRT + CuPy preprocess delivers 2.2x speedup over Ultralytics and meets the 30fps per channel target for 2 channels.

Next: exp13 (nvJPEG GPU encoding for MJPEG output), then design doc.
