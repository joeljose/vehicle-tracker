"""Measure per-stage latency of the custom pipeline hot path.

Confirms (or refutes) the hypothesis that the slow MP4 playback in
Setup/Analytics is bottlenecked by render-thread CPU work, not by
the GPU detect/track stages. Also probes:

- TRT inference cold-vs-warm gap
- cupyx.scipy.ndimage.zoom vs cv2 host-side resize (cv2.cuda is not
  available in the custom image so we compare CPU resize as the
  realistic alternative)
- nv12_to_bgr + asnumpy cost
- JPEG encode at 1920x1080
- PyNvVideoCodec PTS units on a real MP4 (sanity-check the 90kHz divisor)

Run inside the custom backend container via:
    docker compose ... exec backend python experiments/exp_perf_audit.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cupy as cp
import cv2
import numpy as np
import tensorrt as trt
from cupyx.scipy.ndimage import zoom

from PyNvVideoCodec import CreateDecoder, CreateDemuxer

from backend.pipeline.custom.preprocess import nv12_to_bgr_gpu, GpuPreprocessor


VIDEO_DIR = Path("/data/site_videos") if Path("/data/site_videos").exists() else Path("data_collection/site_videos")
ENGINE = Path("/app/models/yolov8s_rfdetr_v1_cont_custom.engine")


def time_block(fn, n=30):
    fn()  # untimed warmup of caller (not the system, just to populate args)
    cp.cuda.Device(0).synchronize()
    samples = []
    for _ in range(n):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device(0).synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def stats(samples):
    return {
        "min_ms": round(min(samples), 2),
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)], 2),
        "max_ms": round(max(samples), 2),
        "n": len(samples),
    }


def find_mp4(prefer_60fps=True):
    candidates = sorted(VIDEO_DIR.glob("*.mp4"))
    if not candidates:
        raise RuntimeError(f"No mp4 in {VIDEO_DIR}")
    if prefer_60fps:
        for c in candidates:
            if "Lytle" not in c.name:
                return c
    return candidates[0]


def probe_pts(mp4: Path):
    print(f"\n## PTS probe on {mp4.name}")
    demux = CreateDemuxer(str(mp4))
    dec = CreateDecoder(gpuid=0, codec=demux.GetNvCodecId(), usedevicememory=True)
    print(f"  declared: {demux.Width()}x{demux.Height()} @ {demux.FrameRate()}fps")
    pts_raw = []
    for _ in range(50):
        pkt = demux.Demux()
        if pkt.bsl == 0:
            break
        for f in dec.Decode(pkt):
            if hasattr(f, "getPTS"):
                pts_raw.append(int(f.getPTS()))
            if len(pts_raw) >= 30:
                break
        if len(pts_raw) >= 30:
            break
    if len(pts_raw) < 2:
        print("  (could not collect pts)")
        return
    deltas = [pts_raw[i + 1] - pts_raw[i] for i in range(len(pts_raw) - 1)]
    deltas = [d for d in deltas if d > 0]
    median_delta = statistics.median(deltas) if deltas else 0
    print(f"  first 5 raw PTS: {pts_raw[:5]}")
    print(f"  median PTS delta: {median_delta}")
    # Interpret: if 90kHz ticks, 60fps → 1500 ticks/frame; 30fps → 3000 ticks/frame.
    # If microseconds, 60fps → 16667 us/frame.
    # If milliseconds, 60fps → 16-17 ms/frame.
    if median_delta:
        for label, divisor, target_ms in [("90kHz ticks", 90, None), ("microseconds", 1000, None), ("milliseconds", 1, None)]:
            implied_ms = median_delta / divisor if divisor else median_delta
            print(f"    if {label}: implied frame interval = {implied_ms:.2f} ms (= {1000/implied_ms:.1f} fps)" if implied_ms else "")


def make_dummy_nv12(h=1080, w=1920):
    return cp.random.randint(0, 255, (h * 3 // 2, w), dtype=cp.uint8)


def main():
    print("=" * 70)
    print("CUSTOM-PIPELINE HOT-PATH AUDIT")
    print("=" * 70)

    # 1. PTS units sanity check
    mp4_60 = find_mp4(prefer_60fps=True)
    probe_pts(mp4_60)
    for c in VIDEO_DIR.glob("*Lytle*.mp4"):
        probe_pts(c)
        break

    # 2. NV12 → BGR (GPU kernel) + asnumpy
    print("\n## NV12→BGR GPU kernel + cp.asnumpy (1920x1080)")
    nv12 = make_dummy_nv12()

    def f_kernel():
        return nv12_to_bgr_gpu(nv12, 1080, 1920)

    print("  kernel only (no asnumpy):", stats(time_block(f_kernel)))

    def f_kernel_dl():
        return cp.asnumpy(nv12_to_bgr_gpu(nv12, 1080, 1920))

    print("  kernel + asnumpy        :", stats(time_block(f_kernel_dl)))

    # 3. Preprocess (zoom-based letterbox)
    print("\n## GpuPreprocessor (NV12→RGB→letterbox 640) — current implementation")
    pre = GpuPreprocessor(640)

    def f_pre():
        return pre(nv12, 1080, 1920)

    print("  full preprocess         :", stats(time_block(f_pre)))

    # 3b. Just the cupyx zoom step in isolation
    rgb = cp.random.rand(1080, 1920, 3).astype(cp.float32)
    scale_h = 360 / 1080
    scale_w = 640 / 1920

    def f_zoom():
        return zoom(rgb, (scale_h, scale_w, 1.0), order=1)

    print("  cupyx ndimage.zoom only :", stats(time_block(f_zoom)))

    # 3c. CPU bilinear via cv2 as a baseline (the easy alternative if cv2.cuda is missing)
    rgb_cpu = np.random.rand(1080, 1920, 3).astype(np.float32)

    def f_cv2_resize():
        return cv2.resize(rgb_cpu, (640, 360), interpolation=cv2.INTER_LINEAR)

    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        f_cv2_resize()
        samples.append((time.perf_counter() - t0) * 1000)
    print("  cv2.resize CPU baseline :", stats(samples))

    # 4. JPEG encode (1920x1080 BGR uint8)
    print("\n## JPEG encode (cv2.imencode quality 80)")
    bgr = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)

    def f_jpeg():
        return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        f_jpeg()
        samples.append((time.perf_counter() - t0) * 1000)
    print("  per encode              :", stats(samples))
    print("  → 2 encodes per frame   :", round(2 * statistics.median(samples), 2), "ms")

    # 5. TRT inference cold vs warm
    if ENGINE.exists():
        print("\n## TRT inference cold vs warm")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(ENGINE, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        in_name = engine.get_tensor_name(0)
        out_name = engine.get_tensor_name(1)
        in_shape = (1, 3, 640, 640)
        out_shape = tuple(engine.get_tensor_shape(out_name))
        in_buf = cp.ones(in_shape, dtype=cp.float32)
        out_buf = cp.empty(out_shape, dtype=cp.float32)
        ctx.set_tensor_address(in_name, in_buf.data.ptr)
        ctx.set_tensor_address(out_name, out_buf.data.ptr)
        stream = cp.cuda.Stream()

        # Cold (first run)
        t0 = time.perf_counter()
        ctx.execute_async_v3(stream.ptr)
        stream.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000
        print(f"  cold (1st call)         : {cold_ms:.2f} ms")

        # Warm
        samples = []
        for _ in range(30):
            t0 = time.perf_counter()
            ctx.execute_async_v3(stream.ptr)
            stream.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        print("  warm                    :", stats(samples))
    else:
        print(f"\n## TRT engine not found at {ENGINE} — skipping inference timing")

    # 6. Theoretical minimum frame budget
    print("\n## Source FPS budgets")
    for fps in (30, 60):
        print(f"  {fps} FPS → {1000/fps:.2f} ms/frame budget")

    # 7. End-to-end serial estimate based on measured pieces
    print("\n## Predicted custom-pipeline serial latency per frame")
    print("  preprocess + (TRT warm) + 2×asnumpy(BGR) + 2×JPEG ≈ sum of medians")


if __name__ == "__main__":
    main()
