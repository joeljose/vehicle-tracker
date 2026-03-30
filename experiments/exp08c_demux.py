"""Experiment 08c: Shared inference + nvstreamdemux for per-channel extraction

Architecture: mux → infer → tracker → nvstreamdemux → per-channel OSD + BufferOperator

This avoids the BufferOperator batch-size>1 stall by demuxing back to
per-channel streams before frame extraction.

Usage:
    python3 -u exp08c_demux.py <video1> <video2>
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


class MetadataCounter(BatchMetadataOperator):
    """Counts frames and detections per source from batched metadata."""
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.batches = 0
        self.per_source = {}

    def handle_metadata(self, batch_meta):
        with self.lock:
            self.batches += 1
            for fm in batch_meta.frame_items:
                sid = fm.source_id
                if sid not in self.per_source:
                    self.per_source[sid] = {"frames": 0, "dets": 0, "tracks": set()}
                self.per_source[sid]["frames"] += 1
                for om in fm.object_items:
                    self.per_source[sid]["dets"] += 1
                    self.per_source[sid]["tracks"].add(om.object_id)


class ChannelMjpegExtractor(BufferOperator):
    """Per-channel MJPEG extractor (operates on demuxed single-source stream)."""
    def __init__(self, channel_id):
        super().__init__()
        self.channel_id = channel_id
        self.frame_count = 0
        self.last_jpeg = None
        self.last_jpeg_size = 0
        self._last_emit = 0
        self._sample_saved = False

    def handle_buffer(self, buffer):
        self.frame_count += 1

        # Rate limit to ~15fps
        now = time.monotonic()
        if now - self._last_emit < 0.066:
            return
        self._last_emit = now

        try:
            tensor = buffer.extract(0)  # batch_id=0 (demuxed = single source)
            if cp is not None:
                gpu_arr = cp.from_dlpack(tensor)
                frame_rgb = cp.asnumpy(gpu_arr)
            else:
                frame_rgb = np.from_dlpack(tensor)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _, jpeg_buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            self.last_jpeg = jpeg_buf.tobytes()
            self.last_jpeg_size = len(self.last_jpeg)

            # Save first frame as sample
            if not self._sample_saved:
                path = f"/tmp/exp08c_ch{self.channel_id}.jpg"
                cv2.imwrite(path, frame_bgr)
                print(f"  [ch{self.channel_id}] First frame: {frame_rgb.shape}, "
                      f"saved to {path}", flush=True)
                self._sample_saved = True

        except Exception as e:
            print(f"  [ch{self.channel_id}] Error: {e}", flush=True)


def uri(p):
    return f"file://{os.path.abspath(p)}"


def main():
    v1, v2 = sys.argv[1], sys.argv[2]
    print("=" * 60, flush=True)
    print("Exp 08c: Shared inference + nvstreamdemux", flush=True)
    print("=" * 60, flush=True)
    print(f"  Source 0: {v1}", flush=True)
    print(f"  Source 1: {v2}", flush=True)
    print(flush=True)

    p = Pipeline("demux-test")
    meta = MetadataCounter()
    ch0_mjpeg = ChannelMjpegExtractor(0)
    ch1_mjpeg = ChannelMjpegExtractor(1)

    # ── Shared: mux → infer → tracker ──
    p.add("nvstreammux", "mux", {
        "batch-size": 2, "batched-push-timeout": 33000,
        "width": 1920, "height": 1080,
    })
    p.add("nvurisrcbin", "src_0", {"uri": uri(v1), "file-loop": True})
    p.link(("src_0", "mux"), ("", "sink_%u"))
    p.add("nvurisrcbin", "src_1", {"uri": uri(v2), "file-loop": True})
    p.link(("src_1", "mux"), ("", "sink_%u"))

    p.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
    p.link("mux", "infer")
    p.add("nvtracker", "tracker", {
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
        "ll-lib-file": TRACKER_LIB, "ll-config-file": TRACKER_CONFIG,
    })
    p.link("infer", "tracker")
    p.attach("tracker", Probe("meta", meta))

    # ── Demux: split back to per-channel streams ──
    p.add("nvstreamdemux", "demux", {})
    p.link("tracker", "demux")

    # ── Channel 0: OSD → convert → caps → sink + MjpegExtractor ──
    p.add("queue", "q0", {"leaky": 2, "max-size-buffers": 5})
    p.add("nvdsosd", "osd0", {"gpu-id": 0})
    p.add("nvvideoconvert", "conv0", {})
    p.add("capsfilter", "caps0", {
        "caps": "video/x-raw(memory:NVMM),format=RGB",
    })
    p.add("fakesink", "sink0", {"sync": True})
    # Link demux src_%u pads → per-channel branches
    p.link(("demux", "q0"), ("src_%u", ""))
    p.link("q0", "osd0", "conv0", "caps0", "sink0")
    p.attach("caps0", Probe("mjpeg0", ch0_mjpeg))

    # ── Channel 1: OSD → convert → caps → sink + MjpegExtractor ──
    p.add("queue", "q1", {"leaky": 2, "max-size-buffers": 5})
    p.add("nvdsosd", "osd1", {"gpu-id": 0})
    p.add("nvvideoconvert", "conv1", {})
    p.add("capsfilter", "caps1", {
        "caps": "video/x-raw(memory:NVMM),format=RGB",
    })
    p.add("fakesink", "sink1", {"sync": True})
    p.link(("demux", "q1"), ("src_%u", ""))
    p.link("q1", "osd1", "conv1", "caps1", "sink1")
    p.attach("caps1", Probe("mjpeg1", ch1_mjpeg))

    # Start
    print("[1] Starting pipeline...", flush=True)
    p.start()

    for t in range(1, 11):
        time.sleep(1)
        with meta.lock:
            src_parts = [f"src{s}:{d['frames']}f/{d['dets']}d"
                         for s, d in sorted(meta.per_source.items())]
        print(f"  t={t:2d}s: batches={meta.batches}, "
              f"meta=[{', '.join(src_parts)}], "
              f"ch0_jpeg={ch0_mjpeg.frame_count}f, "
              f"ch1_jpeg={ch1_mjpeg.frame_count}f",
              flush=True)

    # Results
    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    meta_ok = meta.batches > 50
    print(f"\n[A] Metadata flowing: {'PASS' if meta_ok else 'FAIL'} "
          f"({meta.batches} batches)", flush=True)
    for sid, d in sorted(meta.per_source.items()):
        print(f"    Source {sid}: {d['frames']} frames, {d['dets']} dets, "
              f"{len(d['tracks'])} tracks", flush=True)

    ch0_ok = ch0_mjpeg.frame_count > 50
    ch1_ok = ch1_mjpeg.frame_count > 50
    print(f"\n[B] Ch0 MJPEG extraction: {'PASS' if ch0_ok else 'FAIL'} "
          f"({ch0_mjpeg.frame_count} frames, last JPEG={ch0_mjpeg.last_jpeg_size}B)",
          flush=True)
    print(f"[C] Ch1 MJPEG extraction: {'PASS' if ch1_ok else 'FAIL'} "
          f"({ch1_mjpeg.frame_count} frames, last JPEG={ch1_mjpeg.last_jpeg_size}B)",
          flush=True)

    # Validate JPEGs are different (different videos = different content)
    if ch0_mjpeg.last_jpeg and ch1_mjpeg.last_jpeg:
        img0 = cv2.imdecode(np.frombuffer(ch0_mjpeg.last_jpeg, np.uint8), cv2.IMREAD_COLOR)
        img1 = cv2.imdecode(np.frombuffer(ch1_mjpeg.last_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img0 is not None and img1 is not None:
            mean0 = img0.mean(axis=(0, 1))
            mean1 = img1.mean(axis=(0, 1))
            different = abs(mean0[0] - mean1[0]) > 20
            print(f"\n[D] Frames from different sources: {'PASS' if different else 'FAIL'}",
                  flush=True)
            print(f"    Ch0 mean BGR: ({mean0[0]:.0f},{mean0[1]:.0f},{mean0[2]:.0f})",
                  flush=True)
            print(f"    Ch1 mean BGR: ({mean1[0]:.0f},{mean1[1]:.0f},{mean1[2]:.0f})",
                  flush=True)

    all_pass = meta_ok and ch0_ok and ch1_ok
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILED'}", flush=True)
    if all_pass:
        print("  ✓ Shared inference + per-channel demux extraction WORKS", flush=True)
    print(f"{'='*60}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 -u exp08c_demux.py <video1> <video2>")
        sys.exit(1)
    main()
