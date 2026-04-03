"""Experiment 11: End-to-end custom pipeline loop

Full loop: decode → resize → TensorRT infer → BoT-SORT track → draw bboxes → encode JPEG
Tests sustained throughput, per-stage latency breakdown, and 2-channel batching.

Usage (inside backend container):
    python3 experiments/exp11_e2e_pipeline.py /data/test_clips/741_73_1min.mp4
    python3 experiments/exp11_e2e_pipeline.py /data/test_clips/741_73_1min.mp4 /data/test_clips/lytle_south_1min.mp4
"""

import os
import sys
import time

import cv2
import numpy as np
import yaml


def create_tracker_config():
    """Create BoT-SORT config: no ReID, no GMC."""
    config = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "none",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
    }
    path = "/tmp/botsort_e2e.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def draw_boxes(frame, results):
    """Draw bounding boxes and track IDs on frame."""
    if results[0].boxes.id is None:
        return frame
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    for tid, box, conf in zip(ids, boxes, confs):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{tid} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def test_single_channel(model, sources, tracker_cfg, max_inferences=300):
    """Single channel: decode → infer+track → draw → JPEG encode."""
    src = sources[0]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {src}")
        return

    timings = {"decode": [], "infer_track": [], "draw": [], "jpeg": [], "total": []}
    frame_count = 0
    infer_count = 0
    all_ids = set()

    while infer_count < max_inferences:
        # Decode
        t_total = time.monotonic()
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            break
        timings["decode"].append(time.monotonic() - t0)

        # Skip every other frame (60fps → 30fps)
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # Infer + Track
        t0 = time.monotonic()
        results = model.track(frame, persist=True, tracker=tracker_cfg,
                              verbose=False, classes=[2, 3, 5, 7])
        timings["infer_track"].append(time.monotonic() - t0)

        if results[0].boxes.id is not None:
            for tid in results[0].boxes.id.cpu().numpy().astype(int):
                all_ids.add(tid)

        # Draw
        t0 = time.monotonic()
        annotated = draw_boxes(frame.copy(), results)
        timings["draw"].append(time.monotonic() - t0)

        # JPEG encode
        t0 = time.monotonic()
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        timings["jpeg"].append(time.monotonic() - t0)

        timings["total"].append(time.monotonic() - t_total)
        infer_count += 1

    cap.release()
    return infer_count, len(all_ids), timings


def test_two_channel(sources, tracker_cfg, max_inferences=150):
    """Two channels: decode both → batch infer+track → draw → JPEG encode."""
    from ultralytics import YOLO
    ENGINE = "/app/models/yolov8s.engine"

    # Separate model per channel (each has its own tracker state)
    model_a = YOLO(ENGINE, task="detect")
    model_b = YOLO(ENGINE, task="detect")

    cap_a = cv2.VideoCapture(sources[0])
    cap_b = cv2.VideoCapture(sources[1] if len(sources) > 1 else sources[0])

    timings = {"decode": [], "infer_track": [], "draw": [], "jpeg": [], "total": []}
    frame_count = 0
    infer_count = 0
    ids_a = set()
    ids_b = set()

    while infer_count < max_inferences:
        t_total = time.monotonic()

        # Decode both
        t0 = time.monotonic()
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break
        timings["decode"].append(time.monotonic() - t0)

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # Infer + Track (sequential — each model has its own tracker)
        t0 = time.monotonic()
        res_a = model_a.track(frame_a, persist=True, tracker=tracker_cfg,
                              verbose=False, classes=[2, 3, 5, 7])
        res_b = model_b.track(frame_b, persist=True, tracker=tracker_cfg,
                              verbose=False, classes=[2, 3, 5, 7])
        timings["infer_track"].append(time.monotonic() - t0)

        if res_a[0].boxes.id is not None:
            for tid in res_a[0].boxes.id.cpu().numpy().astype(int):
                ids_a.add(tid)
        if res_b[0].boxes.id is not None:
            for tid in res_b[0].boxes.id.cpu().numpy().astype(int):
                ids_b.add(tid)

        # Draw both
        t0 = time.monotonic()
        ann_a = draw_boxes(frame_a.copy(), res_a)
        ann_b = draw_boxes(frame_b.copy(), res_b)
        timings["draw"].append(time.monotonic() - t0)

        # JPEG encode both
        t0 = time.monotonic()
        cv2.imencode(".jpg", ann_a, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imencode(".jpg", ann_b, [cv2.IMWRITE_JPEG_QUALITY, 80])
        timings["jpeg"].append(time.monotonic() - t0)

        timings["total"].append(time.monotonic() - t_total)
        infer_count += 1

    cap_a.release()
    cap_b.release()
    return infer_count, (len(ids_a), len(ids_b)), timings


def print_timings(timings, infer_count):
    """Print per-stage latency breakdown."""
    for stage in ["decode", "infer_track", "draw", "jpeg", "total"]:
        vals = timings[stage]
        if not vals:
            continue
        avg = sum(vals) / len(vals) * 1000
        p50 = sorted(vals)[len(vals) // 2] * 1000
        p99 = sorted(vals)[int(len(vals) * 0.99)] * 1000
        print(f"  {stage:12s}: avg {avg:6.2f} ms  p50 {p50:6.2f} ms  p99 {p99:6.2f} ms")

    total_time = sum(timings["total"])
    fps = infer_count / total_time if total_time > 0 else 0
    print(f"  {'FPS':12s}: {fps:.1f}")
    print(f"  {'Budget':12s}: {'OK' if fps >= 30 else 'FAIL'} (need 30 fps)")


def main():
    from ultralytics import YOLO

    sources = sys.argv[1:]
    if not sources:
        print(f"Usage: {sys.argv[0]} <video1.mp4> [video2.mp4]")
        sys.exit(1)

    ENGINE = "/app/models/yolov8s.engine"
    tracker_cfg = create_tracker_config()

    print("=== Experiment 11: End-to-End Custom Pipeline ===")
    print(f"Engine: {ENGINE}")
    print(f"Tracker: BoT-SORT (no ReID, no GMC)")
    print(f"Sources: {sources}")
    print()

    # Test 1: Single channel E2E
    print("[TEST 1] Single channel — full loop (300 inferences)")
    model = YOLO(ENGINE, task="detect")
    n, ids, timings = test_single_channel(model, sources, tracker_cfg, 300)
    print(f"  Inferences: {n}, Unique IDs: {ids}")
    print_timings(timings, n)
    print()

    # Save a sample annotated frame
    cap = cv2.VideoCapture(sources[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    if ret:
        model2 = YOLO(ENGINE, task="detect")
        res = model2.track(frame, persist=True, tracker=tracker_cfg,
                           verbose=False, classes=[2, 3, 5, 7])
        annotated = draw_boxes(frame, res)
        cv2.imwrite("/tmp/exp11_sample.jpg", annotated)
        print("  Sample frame saved: /tmp/exp11_sample.jpg")
    cap.release()
    print()
    sys.stdout.flush()

    # Test 2: Two channel E2E
    if len(sources) >= 2:
        src_label = f"{sources[0]} + {sources[1]}"
    else:
        src_label = f"{sources[0]} × 2"
    print(f"[TEST 2] Two channels — sequential infer (150 inferences each)")
    print(f"  Sources: {src_label}")
    n, ids_tuple, timings = test_two_channel(sources, tracker_cfg, 150)
    print(f"  Inferences: {n} per channel, IDs: ch0={ids_tuple[0]}, ch1={ids_tuple[1]}")
    print_timings(timings, n)
    print()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
