"""Experiment 06: Test pipeline.start() without .wait()

Verifies that:
1. pipeline.start() returns immediately (non-blocking)
2. Probe callbacks fire on a background thread
3. We can read callback results from the main thread
4. pipeline.stop() gracefully shuts down

Usage:
    python3 experiments/exp06_async_pipeline.py /data/test_clips/741_73_10s.mp4
"""

import os
import sys
import threading
import time

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator, BufferOperator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.pipeline.deepstream.config import PGIE_CONFIG, TRACKER_CONFIG, TRACKER_LIB, load_labels


class SimpleReporter(BatchMetadataOperator):
    """Minimal probe that counts frames and detections."""

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.frame_count = 0
        self.detection_count = 0
        self.thread_name = None
        self.last_frame_time = None

    def handle_metadata(self, batch_meta):
        self.thread_name = threading.current_thread().name
        self.last_frame_time = time.monotonic()
        for frame_meta in batch_meta.frame_items:
            self.frame_count += 1
            for obj_meta in frame_meta.object_items:
                self.detection_count += 1


def build_test_pipeline(video_path):
    """Build a minimal detection + tracking pipeline."""
    file_uri = f"file://{os.path.abspath(video_path)}"
    labels = load_labels()
    reporter = SimpleReporter(labels)

    pipeline = Pipeline("async-test")

    pipeline.add("nvurisrcbin", "src", {"uri": file_uri})
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1,
        "batched-push-timeout": 33000,
        "width": 1920,
        "height": 1080,
    })
    pipeline.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
        "ll-lib-file": TRACKER_LIB,
        "ll-config-file": TRACKER_CONFIG,
    })
    pipeline.add("fakesink", "sink", {"sync": False})

    pipeline.link(("src", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "infer", "tracker", "sink")
    pipeline.attach("tracker", Probe("reporter", reporter))

    return pipeline, reporter


def main(video_path):
    print(f"=== Experiment 06: Async Pipeline Test ===")
    print(f"Video: {video_path}")
    print(f"Main thread: {threading.current_thread().name}")
    print()

    pipeline, reporter = build_test_pipeline(video_path)

    # TEST 1: start() returns immediately
    print("[TEST 1] pipeline.start() returns immediately?")
    t0 = time.monotonic()
    pipeline.start()
    dt = time.monotonic() - t0
    print(f"  start() returned in {dt:.3f}s")
    assert dt < 5.0, f"start() blocked for {dt:.1f}s — not async!"
    print("  PASS\n")

    # TEST 2: Probe callbacks fire in background
    print("[TEST 2] Probe callbacks fire in background?")
    for i in range(10):
        time.sleep(0.5)
        if reporter.frame_count > 0:
            break
    print(f"  Frames after {(i+1)*0.5:.1f}s: {reporter.frame_count}")
    print(f"  Detections: {reporter.detection_count}")
    print(f"  Callback thread: {reporter.thread_name}")
    print(f"  Main thread: {threading.current_thread().name}")
    assert reporter.frame_count > 0, "No frames received!"
    assert reporter.thread_name != threading.current_thread().name, "Callback on main thread!"
    print("  PASS\n")

    # TEST 3: Can read results from main thread while pipeline runs
    print("[TEST 3] Main thread reads while pipeline runs?")
    prev_count = reporter.frame_count
    time.sleep(1.0)
    new_count = reporter.frame_count
    delta = new_count - prev_count
    print(f"  Frames in 1s: {delta} (prev={prev_count}, now={new_count})")
    assert delta > 0, "Pipeline stopped producing frames!"
    fps = delta  # frames per second approximation
    print(f"  ~{fps} fps")
    print("  PASS\n")

    # TEST 4: pipeline.stop() gracefully shuts down
    print("[TEST 4] pipeline.stop() graceful shutdown?")
    total_before_stop = reporter.frame_count
    t0 = time.monotonic()
    pipeline.stop()
    dt = time.monotonic() - t0
    total_after_stop = reporter.frame_count
    print(f"  stop() returned in {dt:.3f}s")
    print(f"  Frames before stop: {total_before_stop}")
    print(f"  Frames after stop: {total_after_stop}")

    # Verify no more frames arrive
    time.sleep(0.5)
    final = reporter.frame_count
    print(f"  Frames 0.5s after stop: {final}")
    assert final == total_after_stop or final - total_after_stop < 5, "Pipeline still running after stop!"
    print("  PASS\n")

    # TEST 5: Can we start a NEW pipeline after stopping? (reusability)
    print("[TEST 5] Can start a new pipeline after stopping?")
    pipeline2, reporter2 = build_test_pipeline(video_path)
    pipeline2.start()
    time.sleep(2.0)
    print(f"  Pipeline 2 frames: {reporter2.frame_count}")
    assert reporter2.frame_count > 0, "Second pipeline didn't produce frames!"
    pipeline2.stop()
    print("  PASS\n")

    print("=== ALL TESTS PASSED ===")
    print(f"Total frames processed: {reporter.frame_count} + {reporter2.frame_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
