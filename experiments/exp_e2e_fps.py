"""End-to-end achieved-FPS measurement on a real MP4 through the full
custom pipeline path. Mirrors what _pipeline_loop does per frame, so
the measured rate is what the display pacer would see.

Reports rolling FPS so we can see the warmup transient vs steady state.
"""

from __future__ import annotations

import time
from pathlib import Path

import cupy as cp
import cv2
import numpy as np

from backend.pipeline.custom.decoder import NvDecoder
from backend.pipeline.custom.detector import TRTDetector
from backend.pipeline.custom.preprocess import GpuPreprocessor, nv12_to_bgr_gpu
from backend.pipeline.custom.tracker import TrackerWrapper

VIDEO_DIR = Path("/data/site_videos") if Path("/data/site_videos").exists() else Path("data_collection/site_videos")
ENGINE = "/app/models/yolov8s_rfdetr_v1_cont_custom.engine"

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 80]


def time_path(label: str, mp4: Path, n_frames: int = 240, with_extra_clean_jpeg: bool = True):
    print(f"\n## {label}: {mp4.name}")
    decoder = NvDecoder(str(mp4), loop=False, channel_id=99)
    detector = TRTDetector(ENGINE)
    preprocess = GpuPreprocessor()
    tracker = TrackerWrapper()

    stage = {"decode": [], "preprocess": [], "infer": [], "track": [], "render": [], "total": []}
    fps_window = []
    t_start = time.perf_counter()

    for i in range(n_frames):
        t0 = time.perf_counter()

        # --- decode ---
        td0 = time.perf_counter()
        nv12, pts = decoder.read()
        if nv12 is None:
            print(f"  EOS at frame {i}")
            break
        stage["decode"].append((time.perf_counter() - td0) * 1000)

        # --- preprocess ---
        tp0 = time.perf_counter()
        cp.cuda.Device(0).synchronize()
        input_tensor, scale_info = preprocess(nv12, decoder.height, decoder.width)
        cp.cuda.Device(0).synchronize()
        stage["preprocess"].append((time.perf_counter() - tp0) * 1000)

        # --- infer + NMS ---
        ti0 = time.perf_counter()
        detections = detector.detect(input_tensor, scale_info)
        stage["infer"].append((time.perf_counter() - ti0) * 1000)

        # --- BGR for ReID + track ---
        tt0 = time.perf_counter()
        bgr_for_reid = None
        if len(detections) > 0 and tracker._with_reid:
            bgr_gpu = nv12_to_bgr_gpu(nv12, decoder.height, decoder.width)
            bgr_for_reid = cp.asnumpy(bgr_gpu)
        tracks = tracker.update(detections, frame=bgr_for_reid)
        stage["track"].append((time.perf_counter() - tt0) * 1000)

        # --- render: same pattern as _store_display_frame ---
        tr0 = time.perf_counter()
        bgr_gpu_render = nv12_to_bgr_gpu(nv12, decoder.height, decoder.width)
        bgr = cp.asnumpy(bgr_gpu_render)
        if with_extra_clean_jpeg:
            _, _ = cv2.imencode(".jpg", bgr, JPEG_PARAMS)  # encode_clean (last_frame)
        for t in tracks:
            x, y, w, h = t["bbox"]
            tid = t["track_id"]
            conf = t["confidence"]
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(bgr, f"#{tid} {conf:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _, _ = cv2.imencode(".jpg", bgr, JPEG_PARAMS)  # annotated
        stage["render"].append((time.perf_counter() - tr0) * 1000)

        total_ms = (time.perf_counter() - t0) * 1000
        stage["total"].append(total_ms)

        # rolling 1-second FPS
        now = time.perf_counter()
        fps_window.append(now)
        fps_window = [t for t in fps_window if now - t <= 1.0]
        if (i + 1) % 30 == 0 or i < 5:
            print(f"  frame {i+1:>3}: total={total_ms:6.2f} ms, "
                  f"rolling FPS≈{len(fps_window)}, dets={len(detections)}, tracks={len(tracks)}")

    decoder.release()
    elapsed = time.perf_counter() - t_start
    n = len(stage["total"])
    print(f"\n  achieved FPS over {n} frames: {n/elapsed:.1f}")

    import statistics

    print("  per-stage medians (ms):", {k: round(statistics.median(v), 2) for k, v in stage.items() if v})
    # warmup vs steady (skip first 30)
    if n >= 60:
        warm = stage["total"][:30]
        steady = stage["total"][30:]
        print(f"  first  30 frames mean total: {sum(warm)/30:.2f} ms")
        print(f"  later     frames mean total: {sum(steady)/len(steady):.2f} ms")


def main():
    print("=" * 70)
    print("E2E ACHIEVED FPS — full custom pipeline replication")
    print("=" * 70)

    # 60 fps file
    mp4_60 = next((p for p in sorted(VIDEO_DIR.glob("*.mp4")) if "Lytle" not in p.name), None)
    # 30 fps file
    mp4_30 = next(VIDEO_DIR.glob("*Lytle*.mp4"), None)

    if mp4_60:
        time_path("60 FPS source — current path (2 JPEG encodes)", mp4_60, n_frames=180)
    if mp4_30:
        time_path("30 FPS source — current path (2 JPEG encodes)", mp4_30, n_frames=180)

    # what if we drop the redundant clean-JPEG?
    if mp4_60:
        time_path("60 FPS source — proposed path (1 JPEG encode)", mp4_60, n_frames=180, with_extra_clean_jpeg=False)


if __name__ == "__main__":
    main()
