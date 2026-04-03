# Experiment 9: Custom Pipeline Decode Options

## Date
2026-04-03

## Setup
- Container: DeepStream 8.0 backend container
- GPU: RTX 4050 Mobile (6 GB VRAM)
- ffmpeg: 7.0.2-static (johnvansickle build — **no NVDEC/h264_cuvid**)
- OpenCV: built with FFmpeg backend (CAP_FFMPEG)
- Test clips: 741_73_1min.mp4 (1920x1080, 60fps), lytle_south_1min.mp4

## Results

### Single-channel decode (300 frames, 1920x1080)

| Method | FPS | ms/frame | Notes |
|--------|-----|----------|-------|
| ffmpeg subprocess pipe (CPU) | 244 | 4.1 | `ffmpeg -i src -f rawvideo -pix_fmt bgr24 pipe:1` |
| OpenCV VideoCapture | 780 | 1.3 | `cv2.VideoCapture(src, cv2.CAP_FFMPEG)` |

### Two-channel decode (150 frames each, sequential reads)

| Method | Ch0 FPS | Ch1 FPS | Combined FPS |
|--------|---------|---------|--------------|
| ffmpeg subprocess × 2 | 225 | 214 | 219 |

### Target: 2 channels × 30fps = 60 fps total

| Method | Headroom over 60 fps |
|--------|---------------------|
| ffmpeg subprocess | 3.6x |
| OpenCV VideoCapture | 13x |

## Findings

1. **CPU decode is more than sufficient.** Both methods far exceed the 60fps requirement. OpenCV is 3x faster than ffmpeg pipe due to avoiding the pipe I/O overhead.

2. **NVDEC is not available** in the current static ffmpeg build. The h264_cuvid decoder is absent. This is not a problem — CPU decode has ample headroom.

3. **OpenCV VideoCapture is the better choice** for the custom pipeline:
   - 3x faster than ffmpeg subprocess pipe (780 vs 244 fps)
   - Handles both files and URLs (HLS, RTSP) via its ffmpeg backend
   - Returns numpy arrays directly (no pipe parsing)
   - Already a project dependency (no new deps)
   - Simpler code (no subprocess management, pipe buffering, cleanup)

4. **ffmpeg subprocess is viable as fallback** if OpenCV has issues with specific HLS streams, but the primary path should be OpenCV.

5. **Source is 60fps** — the pipeline normalizes to 30fps inference by processing every 2nd frame (same as DeepStream's interval=1 approach).

## Key decision
**Use OpenCV VideoCapture as the decoder for the custom pipeline.** It's faster, simpler, already installed, and handles both files and URLs. No new dependencies needed.

## Not tested (needs live stream)
- HLS URL decode via OpenCV (requires active YouTube Live stream)
- Stream recovery behavior on connection drops
- Long-running stability (hours of continuous decode)

These should be validated in M7 P0 spike with a real YouTube Live stream.
