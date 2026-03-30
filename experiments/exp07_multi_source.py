"""Experiment 07: Multi-source shared inference via nvstreammux

Run individual tests:
    python3 -u exp07_multi_source.py test1 <video>
    python3 -u exp07_multi_source.py test2 <video1> <video2>
    python3 -u exp07_multi_source.py test3 <short_video> <long_video>
    python3 -u exp07_multi_source.py test4 <video1> <video2>
"""

import os
import sys
import subprocess
import threading
import time

os.environ["PYTHONUNBUFFERED"] = "1"

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.pipeline.deepstream.config import PGIE_CONFIG, TRACKER_CONFIG, TRACKER_LIB


def vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


class Reporter(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.per_source = {}
        self.lock = threading.Lock()
        self.batches = 0

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


def build(name, uris, loops=None, batch_size=None):
    if loops is None:
        loops = [False] * len(uris)
    pipeline = Pipeline(name)
    reporter = Reporter()

    pipeline.add("nvstreammux", "mux", {
        "batch-size": batch_size or len(uris),
        "batched-push-timeout": 33000,
        "width": 1920, "height": 1080,
    })
    for i, uri in enumerate(uris):
        props = {"uri": uri}
        if loops[i]:
            props["file-loop"] = True
        pipeline.add("nvurisrcbin", f"src_{i}", props)
        pipeline.link((f"src_{i}", "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
    pipeline.link("mux", "infer")
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
        "ll-lib-file": TRACKER_LIB, "ll-config-file": TRACKER_CONFIG,
    })
    pipeline.link("infer", "tracker")
    pipeline.attach("tracker", Probe("r", reporter))
    pipeline.add("fakesink", "sink", {"sync": False})
    pipeline.link("tracker", "sink")
    return pipeline, reporter


def uri(path):
    return f"file://{os.path.abspath(path)}"


# ── TEST 1: Single source VRAM ──────────────────────────────────────────

def test1(video):
    print("TEST 1: Single source VRAM baseline", flush=True)
    p, r = build("t1", [uri(video)], loops=[True])
    v0 = vram_mb()
    p.start()
    time.sleep(8)
    v1 = vram_mb()
    s = r.per_source.get(0, {})
    print(f"  Source 0: {s.get('frames', 0)} frames, {s.get('dets', 0)} dets, "
          f"{len(s.get('tracks', set()))} tracks", flush=True)
    print(f"  VRAM: {v0} → {v1} MB (+{v1-v0} MB)", flush=True)
    print(f"  RESULT: +{v1-v0} MB for 1 source", flush=True)
    os._exit(0)


# ── TEST 2: Two sources shared inference + source_id + VRAM ─────────────

def test2(video1, video2):
    print("TEST 2: Two sources, shared inference", flush=True)
    p, r = build("t2", [uri(video1), uri(video2)], loops=[True, True])
    v0 = vram_mb()
    p.start()
    time.sleep(8)
    v1 = vram_mb()

    for sid in sorted(r.per_source):
        s = r.per_source[sid]
        print(f"  Source {sid}: {s['frames']} frames, {s['dets']} dets, "
              f"{len(s['tracks'])} tracks", flush=True)

    print(f"  VRAM: {v0} → {v1} MB (+{v1-v0} MB)", flush=True)

    has0 = 0 in r.per_source and r.per_source[0]["frames"] > 0
    has1 = 1 in r.per_source and r.per_source[1]["frames"] > 0
    print(f"  source_id=0: {'YES' if has0 else 'NO'}", flush=True)
    print(f"  source_id=1: {'YES' if has1 else 'NO'}", flush=True)

    if has0 and has1:
        t0 = r.per_source[0]["tracks"]
        t1 = r.per_source[1]["tracks"]
        overlap = t0 & t1
        print(f"  Track ID overlap: {len(overlap)} (src0={len(t0)}, src1={len(t1)})", flush=True)

    print(f"  RESULT: {'PASS' if has0 and has1 else 'FAIL'} — +{v1-v0} MB for 2 sources", flush=True)
    os._exit(0)


# ── TEST 3: Per-source EOS ──────────────────────────────────────────────

def test3(short_video, long_video):
    print("TEST 3: Per-source EOS (10s + 1min)", flush=True)
    p, r = build("t3", [uri(short_video), uri(long_video)], loops=[False, False])

    eos_events = []
    def on_msg(msg):
        if "eos" in str(msg).lower():
            eos_events.append(time.monotonic())

    p._instance.start_with_callback(on_msg)
    t0 = time.monotonic()

    for _ in range(6):
        time.sleep(5)
        f0 = r.per_source.get(0, {}).get("frames", 0)
        f1 = r.per_source.get(1, {}).get("frames", 0)
        print(f"  t={time.monotonic()-t0:4.0f}s: src0={f0:5d}  src1={f1:5d}  "
              f"eos_count={len(eos_events)}", flush=True)

    f0 = r.per_source.get(0, {}).get("frames", 0)
    f1 = r.per_source.get(1, {}).get("frames", 0)
    ok = f1 > f0 * 2
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} — src0={f0}, src1={f1}, "
          f"per-source EOS={'works' if ok else 'broken'}", flush=True)
    os._exit(0)


# ── TEST 4: Mixed loop (src0 loops, src1 plays once) ────────────────────

def test4(video1, video2):
    print("TEST 4: Mixed loop (src0=loop, src1=once)", flush=True)
    p, r = build("t4", [uri(video1), uri(video2)], loops=[True, False])

    eos_events = []
    def on_msg(msg):
        if "eos" in str(msg).lower():
            eos_events.append(time.monotonic())

    p._instance.start_with_callback(on_msg)
    t0 = time.monotonic()

    for _ in range(4):
        time.sleep(5)
        f0 = r.per_source.get(0, {}).get("frames", 0)
        f1 = r.per_source.get(1, {}).get("frames", 0)
        print(f"  t={time.monotonic()-t0:4.0f}s: src0={f0:5d} (loop)  src1={f1:5d} (once)  "
              f"eos={len(eos_events)}", flush=True)

    f0 = r.per_source.get(0, {}).get("frames", 0)
    f1 = r.per_source.get(1, {}).get("frames", 0)
    src0_looped = f0 > 400  # 10s video at ~30fps = ~300; looped should be more
    src1_capped = 100 < f1 < 500
    ok = src0_looped and src1_capped
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} — src0={f0} (looped={src0_looped}), "
          f"src1={f1} (capped={src1_capped})", flush=True)
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 -u exp07_multi_source.py <test1|test2|test3|test4> <args...>")
        print("  test1 <video>              — Single source VRAM baseline")
        print("  test2 <video1> <video2>    — Two sources shared inference + VRAM")
        print("  test3 <short> <long>       — Per-source EOS")
        print("  test4 <video1> <video2>    — Mixed loop settings")
        sys.exit(1)

    test = sys.argv[1]
    if test == "test1":
        test1(sys.argv[2])
    elif test == "test2":
        test2(sys.argv[2], sys.argv[3])
    elif test == "test3":
        test3(sys.argv[2], sys.argv[3])
    elif test == "test4":
        test4(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown test: {test}")
        sys.exit(1)
