"""Experiment 09: Custom pipeline decode options

Tests decode approaches for M7 custom pipeline on RTX 4050.
Validates: ffmpeg subprocess pipe, PyNvVideoCodec, OpenCV VideoCapture.
Measures: FPS, latency, CPU/GPU usage, memory.

Usage (inside backend container):
    python3 experiments/exp09_custom_decode.py /data/test_clips/741_73_1min.mp4
    python3 experiments/exp09_custom_decode.py <HLS_URL>
"""

import os
import subprocess
import sys
import time

import numpy as np

# Frame config — 1080p source
WIDTH = 1920
HEIGHT = 1080
FRAME_SIZE = WIDTH * HEIGHT * 3  # BGR24
MAX_FRAMES = 300  # 10s at 30fps


def test_ffmpeg_cpu(source, max_frames=MAX_FRAMES):
    """ffmpeg subprocess with software decode."""
    cmd = [
        "ffmpeg",
        "-i", source,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an", "-sn",
        "-v", "quiet",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE * 4)

    frames = 0
    latencies = []
    try:
        while frames < max_frames:
            t0 = time.monotonic()
            raw = proc.stdout.read(FRAME_SIZE)
            if len(raw) != FRAME_SIZE:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
            dt = time.monotonic() - t0
            latencies.append(dt)
            frames += 1
    finally:
        proc.terminate()
        proc.wait()

    return frames, latencies


def test_ffmpeg_nvdec(source, max_frames=MAX_FRAMES):
    """ffmpeg subprocess with NVDEC hardware decode."""
    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-c:v", "h264_cuvid",
        "-i", source,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an", "-sn",
        "-v", "quiet",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE * 4)

    frames = 0
    latencies = []
    try:
        while frames < max_frames:
            t0 = time.monotonic()
            raw = proc.stdout.read(FRAME_SIZE)
            if len(raw) != FRAME_SIZE:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
            dt = time.monotonic() - t0
            latencies.append(dt)
            frames += 1
    finally:
        proc.terminate()
        proc.wait()

    return frames, latencies


def test_opencv(source, max_frames=MAX_FRAMES):
    """OpenCV VideoCapture (ffmpeg backend, CPU decode)."""
    import cv2

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"  ERROR: cv2.VideoCapture failed to open {source}")
        return 0, []

    frames = 0
    latencies = []
    try:
        while frames < max_frames:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                break
            dt = time.monotonic() - t0
            latencies.append(dt)
            frames += 1
    finally:
        cap.release()

    return frames, latencies


def test_pynvvideocodec(source, max_frames=MAX_FRAMES):
    """PyNvVideoCodec decode (GPU-resident frames)."""
    try:
        import pynvvideocodec as pynvc  # noqa: F811
    except ImportError:
        print("  SKIP: pynvvideocodec not installed")
        return 0, []

    try:
        decoder = pynvc.SimpleDecoder(source)
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, []

    frames = 0
    latencies = []
    try:
        for packet in decoder:
            t0 = time.monotonic()
            frame = packet  # GPU surface
            dt = time.monotonic() - t0
            latencies.append(dt)
            frames += 1
            if frames >= max_frames:
                break
    except Exception as e:
        print(f"  ERROR during decode: {e}")

    return frames, latencies


def report(name, frames, latencies):
    """Print results for one test."""
    if frames == 0:
        print(f"  {name}: NO FRAMES DECODED")
        return

    total = sum(latencies)
    fps = frames / total if total > 0 else 0
    avg_ms = (total / frames) * 1000
    p50 = sorted(latencies)[len(latencies) // 2] * 1000
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] * 1000

    print(f"  {name}:")
    print(f"    Frames: {frames}")
    print(f"    FPS:    {fps:.1f}")
    print(f"    Avg:    {avg_ms:.2f} ms/frame")
    print(f"    P50:    {p50:.2f} ms")
    print(f"    P99:    {p99:.2f} ms")
    print(f"    Total:  {total:.2f}s")


def main(source):
    print(f"=== Experiment 09: Custom Pipeline Decode ===")
    print(f"Source: {source}")
    print(f"Max frames: {MAX_FRAMES}")
    print()

    # Detect source type
    is_hls = source.startswith("http") or source.endswith(".m3u8")
    print(f"Source type: {'HLS/URL' if is_hls else 'File'}")
    print()

    # Test 1: ffmpeg CPU
    print("[TEST 1] ffmpeg subprocess (CPU decode)")
    frames, latencies = test_ffmpeg_cpu(source)
    report("ffmpeg-cpu", frames, latencies)
    print()

    # Test 2: ffmpeg NVDEC
    print("[TEST 2] ffmpeg subprocess (NVDEC hardware decode)")
    frames, latencies = test_ffmpeg_nvdec(source)
    report("ffmpeg-nvdec", frames, latencies)
    print()

    # Test 3: OpenCV VideoCapture
    print("[TEST 3] OpenCV VideoCapture")
    frames, latencies = test_opencv(source)
    report("opencv", frames, latencies)
    print()

    # Test 4: PyNvVideoCodec (file only)
    if not is_hls:
        print("[TEST 4] PyNvVideoCodec (GPU decode)")
        frames, latencies = test_pynvvideocodec(source)
        report("pynvvideocodec", frames, latencies)
        print()
    else:
        print("[TEST 4] PyNvVideoCodec — SKIPPED (HLS not supported)")
        print()

    print("=== DONE ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4 or HLS_URL>")
        sys.exit(1)
    main(sys.argv[1])
