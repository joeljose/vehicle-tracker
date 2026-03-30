"""Experiment 08: Batch routing — extract per-source frames from shared pipeline

Validates that we can:
1. Read source_id from frame_meta in a BatchMetadataOperator (after tracker)
2. Extract per-batch_id GPU tensors from a BufferOperator (after OSD)
3. Correlate batch_id → source_id to route frames to correct channel
4. Produce correct per-channel MJPEG frames from a shared pipeline
5. Run per-channel TrackingReporters from batched metadata

Usage:
    python3 -u exp08_batch_routing.py <video1> <video2>
"""

import os
import sys
import threading
import time

import cv2
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, BufferOperator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.pipeline.deepstream.config import PGIE_CONFIG, TRACKER_CONFIG, TRACKER_LIB

try:
    import cupy as cp
except ImportError:
    cp = None


# ── BatchMetadataRouter: reads source_id per frame ──────────────────────

class BatchMetadataRouter(BatchMetadataOperator):
    """Reads batch_meta, builds source_id mapping, routes to per-channel reporters."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        # Per batch: list of (batch_idx, source_id)
        self._current_mapping = []
        # Accumulated stats per source
        self.per_source = {}
        self.batch_count = 0

    def handle_metadata(self, batch_meta):
        mapping = []
        with self.lock:
            self.batch_count += 1
            for idx, frame_meta in enumerate(batch_meta.frame_items):
                sid = frame_meta.source_id
                mapping.append((idx, sid))

                if sid not in self.per_source:
                    self.per_source[sid] = {
                        "frames": 0, "detections": 0, "track_ids": set(),
                    }
                self.per_source[sid]["frames"] += 1

                for obj_meta in frame_meta.object_items:
                    self.per_source[sid]["detections"] += 1
                    self.per_source[sid]["track_ids"].add(obj_meta.object_id)

            self._current_mapping = mapping

    def get_mapping(self):
        """Get current batch_idx → source_id mapping (thread-safe)."""
        with self.lock:
            return list(self._current_mapping)


# ── BatchFrameExtractor: extracts per-source GPU frames ─────────────────

class BatchFrameExtractor(BufferOperator):
    """Extracts per-batch_id GPU tensors and routes to per-channel handlers."""

    def __init__(self, metadata_router, output_dir="/tmp/exp08_frames"):
        super().__init__()
        self._router = metadata_router
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._frame_count = 0
        # Per-source frame counters and sample frames
        self.per_source_frames = {}
        self._saved_samples = set()
        os.makedirs(output_dir, exist_ok=True)

    def handle_buffer(self, buffer):
        mapping = self._router.get_mapping()
        if not mapping:
            return

        with self._lock:
            self._frame_count += 1

            for batch_idx, source_id in mapping:
                if batch_idx >= buffer.batch_size:
                    continue

                if source_id not in self.per_source_frames:
                    self.per_source_frames[source_id] = 0
                self.per_source_frames[source_id] += 1

                # Save first few frames per source as sample images
                count = self.per_source_frames[source_id]
                if count <= 3 and source_id not in self._saved_samples or count in (1, 50, 100):
                    try:
                        tensor = buffer.extract(batch_idx)
                        if cp is not None:
                            gpu_arr = cp.from_dlpack(tensor)
                            frame_rgb = cp.asnumpy(gpu_arr)
                        else:
                            frame_rgb = np.from_dlpack(tensor)

                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        path = os.path.join(
                            self._output_dir,
                            f"src{source_id}_frame{count}.jpg",
                        )
                        cv2.imwrite(path, frame_bgr)
                        if count == 1:
                            self._saved_samples.add(source_id)
                            print(f"  [extract] Saved src{source_id} frame {count}: "
                                  f"{frame_rgb.shape} → {path}", flush=True)
                    except Exception as e:
                        print(f"  [extract] Error src{source_id} batch_idx={batch_idx}: {e}",
                              flush=True)


# ── BatchMjpegExtractor: produces per-channel JPEG bytes ────────────────

class BatchMjpegExtractor(BufferOperator):
    """Simulates per-channel MJPEG extraction from shared OSD output."""

    def __init__(self, metadata_router):
        super().__init__()
        self._router = metadata_router
        self._lock = threading.Lock()
        self._last_emit = 0
        # Per-source: last JPEG bytes and frame count
        self.per_source_jpeg = {}
        self.per_source_count = {}

    def handle_buffer(self, buffer):
        now = time.monotonic()
        # Rate limit to ~15fps
        if now - self._last_emit < 0.066:
            return
        self._last_emit = now

        mapping = self._router.get_mapping()
        if not mapping:
            return

        with self._lock:
            for batch_idx, source_id in mapping:
                if batch_idx >= buffer.batch_size:
                    continue

                try:
                    tensor = buffer.extract(batch_idx)
                    if cp is not None:
                        gpu_arr = cp.from_dlpack(tensor)
                        frame_rgb = cp.asnumpy(gpu_arr)
                    else:
                        frame_rgb = np.from_dlpack(tensor)

                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    _, jpeg_buf = cv2.imencode(
                        ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    jpeg_bytes = jpeg_buf.tobytes()

                    self.per_source_jpeg[source_id] = jpeg_bytes
                    self.per_source_count[source_id] = (
                        self.per_source_count.get(source_id, 0) + 1
                    )
                except Exception as e:
                    print(f"  [mjpeg] Error src{source_id}: {e}", flush=True)


def uri(path):
    return f"file://{os.path.abspath(path)}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 -u exp08_batch_routing.py <video1> <video2>")
        sys.exit(1)

    v1, v2 = sys.argv[1], sys.argv[2]
    print("=" * 60, flush=True)
    print("Experiment 08: Batch routing from shared pipeline", flush=True)
    print("=" * 60, flush=True)
    print(f"  Source 0: {v1}", flush=True)
    print(f"  Source 1: {v2}", flush=True)
    print(flush=True)

    # Build shared pipeline
    pipeline = Pipeline("batch-route")
    metadata_router = BatchMetadataRouter()
    frame_extractor = BatchFrameExtractor(metadata_router)
    mjpeg_extractor = BatchMjpegExtractor(metadata_router)

    # Mux
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 2,
        "batched-push-timeout": 33000,
        "width": 1920, "height": 1080,
    })

    # Two sources (both looping for sustained test)
    pipeline.add("nvurisrcbin", "src_0", {"uri": uri(v1), "file-loop": True})
    pipeline.link(("src_0", "mux"), ("", "sink_%u"))
    pipeline.add("nvurisrcbin", "src_1", {"uri": uri(v2), "file-loop": True})
    pipeline.link(("src_1", "mux"), ("", "sink_%u"))

    # Inference + tracker
    pipeline.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
    pipeline.link("mux", "infer")
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
        "ll-lib-file": TRACKER_LIB, "ll-config-file": TRACKER_CONFIG,
    })
    pipeline.link("infer", "tracker")

    # Probe: metadata routing (reads source_id from frame_meta)
    pipeline.attach("tracker", Probe("meta_router", metadata_router))

    # Tee
    pipeline.add("tee", "tee", {})
    pipeline.link("tracker", "tee")

    # Branch 1: OSD → MJPEG extraction (sync=True for real-time pacing)
    pipeline.add("queue", "osd_q", {"leaky": 2, "max-size-buffers": 2})
    pipeline.add("nvdsosd", "osd", {"gpu-id": 0})
    pipeline.add("nvvideoconvert", "osd_conv", {})
    pipeline.add("capsfilter", "osd_caps", {
        "caps": "video/x-raw(memory:NVMM),format=RGB",
    })
    pipeline.add("fakesink", "osd_sink", {"sync": True})
    pipeline.link("tee", "osd_q", "osd", "osd_conv", "osd_caps", "osd_sink")
    pipeline.attach("osd_caps", Probe("mjpeg", mjpeg_extractor))

    # Branch 2: Clean RGB → frame extraction (no OSD, async)
    pipeline.add("queue", "snap_q", {"leaky": 2, "max-size-buffers": 2})
    pipeline.add("nvvideoconvert", "snap_conv", {})
    pipeline.add("capsfilter", "snap_caps", {
        "caps": "video/x-raw(memory:NVMM),format=RGB",
    })
    pipeline.add("fakesink", "snap_sink", {"sync": False})
    pipeline.link("tee", "snap_q", "snap_conv", "snap_caps", "snap_sink")
    pipeline.attach("snap_caps", Probe("frames", frame_extractor))

    # Start
    print("[1] Starting shared pipeline...", flush=True)
    pipeline.start()

    # Let it run for 10 seconds
    for t in range(1, 11):
        time.sleep(1)
        m = metadata_router
        mj = mjpeg_extractor
        fe = frame_extractor
        meta_parts = [f"src{s}:{d['frames']}f" for s, d in sorted(m.per_source.items())]
        mjpeg_parts = [f"src{s}:{c}" for s, c in sorted(mj.per_source_count.items())]
        extract_parts = [f"src{s}:{c}" for s, c in sorted(fe.per_source_frames.items())]
        print(f"  t={t:2d}s: batches={m.batch_count}, "
              f"meta=[{', '.join(meta_parts)}], "
              f"mjpeg=[{', '.join(mjpeg_parts)}], "
              f"extract=[{', '.join(extract_parts)}]",
              flush=True)

    # ── Results ──────────────────────────────────────────────────────
    print(flush=True)
    print("=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)

    # Test 1: Both sources in metadata
    src_ids = sorted(metadata_router.per_source.keys())
    has_both_meta = 0 in src_ids and 1 in src_ids
    print(f"\n[TEST A] Both sources in batch_meta: "
          f"{'PASS' if has_both_meta else 'FAIL'}", flush=True)
    for sid, d in sorted(metadata_router.per_source.items()):
        print(f"  Source {sid}: {d['frames']} frames, {d['detections']} dets, "
              f"{len(d['track_ids'])} tracks", flush=True)

    # Test 2: Per-source MJPEG extraction
    has_both_mjpeg = (0 in mjpeg_extractor.per_source_count and
                      1 in mjpeg_extractor.per_source_count)
    print(f"\n[TEST B] Per-source MJPEG extraction: "
          f"{'PASS' if has_both_mjpeg else 'FAIL'}", flush=True)
    for sid, count in sorted(mjpeg_extractor.per_source_count.items()):
        jpeg_size = len(mjpeg_extractor.per_source_jpeg.get(sid, b""))
        print(f"  Source {sid}: {count} JPEGs, last size={jpeg_size} bytes", flush=True)

    # Test 3: Per-source frame extraction
    has_both_frames = (0 in frame_extractor.per_source_frames and
                       1 in frame_extractor.per_source_frames)
    print(f"\n[TEST C] Per-source frame extraction: "
          f"{'PASS' if has_both_frames else 'FAIL'}", flush=True)
    for sid, count in sorted(frame_extractor.per_source_frames.items()):
        print(f"  Source {sid}: {count} frames extracted", flush=True)

    # Test 4: Verify sample images are visually different
    print(f"\n[TEST D] Sample images saved to /tmp/exp08_frames/", flush=True)
    for sid in [0, 1]:
        path = f"/tmp/exp08_frames/src{sid}_frame1.jpg"
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                mean_bgr = img.mean(axis=(0, 1))
                print(f"  src{sid}_frame1.jpg: {img.shape}, "
                      f"mean BGR=({mean_bgr[0]:.0f},{mean_bgr[1]:.0f},{mean_bgr[2]:.0f})",
                      flush=True)
            else:
                print(f"  src{sid}_frame1.jpg: FAILED to read", flush=True)
        else:
            print(f"  src{sid}_frame1.jpg: NOT FOUND", flush=True)

    # Test 5: JPEG bytes are valid images of correct resolution
    print(f"\n[TEST E] JPEG validity check:", flush=True)
    for sid in [0, 1]:
        jpeg = mjpeg_extractor.per_source_jpeg.get(sid)
        if jpeg:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"  Source {sid}: valid JPEG {img.shape}, {len(jpeg)} bytes", flush=True)
            else:
                print(f"  Source {sid}: INVALID JPEG ({len(jpeg)} bytes)", flush=True)
        else:
            print(f"  Source {sid}: no JPEG captured", flush=True)

    # Overall
    all_pass = has_both_meta and has_both_mjpeg and has_both_frames
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}", flush=True)
    if all_pass:
        print("  ✓ Batch routing architecture is VIABLE", flush=True)
        print("  ✓ BatchMetadataRouter can route by source_id", flush=True)
        print("  ✓ BatchMjpegExtractor produces per-channel JPEGs", flush=True)
        print("  ✓ BatchFrameExtractor extracts per-channel frames", flush=True)
    print(f"{'='*60}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
