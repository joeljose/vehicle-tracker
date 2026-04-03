# exp14: HLS Decode for Custom Pipeline

**Date:** 2026-04-03
**Goal:** Find a GPU-resident HLS decode path for the custom pipeline.

## Problem

PyNvVideoCodec's `CreateDemuxer` uses a bundled `libavformat.so.61` compiled
WITHOUT TLS support. It cannot open HLS URLs (`"Protocol not found"`).

## Approaches Tested

### exp14a: Direct decode methods

| Method | Local file | HLS stream | Notes |
|--------|-----------|------------|-------|
| PyNvVideoCodec (NVDEC) | 621 fps, 1.0ms | **FAIL** (no TLS) | Bundled ffmpeg lacks openssl |
| OpenCV CPU + GPU upload | 320 fps, 3.1ms | 163 fps, 6.0ms | Works but breaks GPU-resident flow |
| OpenCV cudacodec | N/A | N/A | Not available in container |
| GStreamer appsink | — | 27.4 fps, 0.1ms | Rate-limited to real-time, NVMM pointers inaccessible |

### exp14b: ffmpeg FIFO → PyNvVideoCodec

- ffmpeg subprocess pipes raw H.264 to a named FIFO
- PyNvVideoCodec reads from FIFO → NVDEC decode
- **Local file: 150 fps** (works)
- **HLS: FAIL** (ffmpeg static binary segfaults on HLS URLs)

### exp14c: GStreamer FIFO → PyNvVideoCodec (WINNER)

Pipeline: `gst-launch-1.0 souphttpsrc ! hlsdemux ! tsdemux ! h264parse ! mpegtsmux ! filesink`
→ named FIFO → `CreateDemuxer(fifo)` → NVDEC → NV12 CuPy on GPU

| Source | FPS | Latency | Startup | Output |
|--------|-----|---------|---------|--------|
| Local file (741_73) | 908 fps | 1.07ms | 245ms | NV12 on GPU |
| YouTube HLS (eAdWk0Vg64A) | 161 fps | 6.17ms | 2859ms | NV12 on GPU |

**Key findings:**
- Fully GPU-resident: same NV12 CuPy output as local files
- Zero PCIe transfers for decode
- GStreamer handles HTTPS via libsoup (not ffmpeg)
- Only ~80 MB of GStreamer apt packages needed (not full DeepStream)
- `mpegtsmux` required: raw H.264 byte-stream lacks container metadata (0x0 resolution)
- FIFO provides natural synchronization between writer (GStreamer) and reader (PyNvVideoCodec)

## Decision

Use GStreamer FIFO → NVDEC for HLS in the custom pipeline. Install minimal
GStreamer packages (tools, plugins-base, plugins-good, plugins-bad) in
`Dockerfile.custom`. No OpenCV fallback needed.

## Test stream

YouTube Live traffic camera: `https://www.youtube.com/watch?v=eAdWk0Vg64A`
