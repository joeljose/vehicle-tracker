"""Debug: isolate what causes the 2-source pipeline to stall with tee branches.

Test A: 2 sources → mux → infer → tracker → fakesink (no tee) — baseline
Test B: 2 sources → mux → infer → tracker → tee → 2 fakesinks (no probes)
Test C: 2 sources → mux → infer → tracker → tee → OSD → fakesink + fakesink
Test D: Like current adapter: full OSD+convert+caps+probe branches
"""

import os
import sys
import threading
import time

os.environ["PYTHONUNBUFFERED"] = "1"

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, BufferOperator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.pipeline.deepstream.config import PGIE_CONFIG, TRACKER_CONFIG, TRACKER_LIB

try:
    import cupy as cp
except ImportError:
    cp = None


class Counter(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.sources = {}
    def handle_metadata(self, batch_meta):
        self.count += 1
        for fm in batch_meta.frame_items:
            sid = fm.source_id
            self.sources[sid] = self.sources.get(sid, 0) + 1


class DummyBuffer(BufferOperator):
    def __init__(self):
        super().__init__()
        self.count = 0
    def handle_buffer(self, buffer):
        self.count += 1


def uri(p):
    return f"file://{os.path.abspath(p)}"


def run_test(name, v1, v2, use_tee=False, use_osd=False, use_buffer_probe=False):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    p = Pipeline(name.replace(" ", "-"))
    counter = Counter()
    buf_probe = DummyBuffer() if use_buffer_probe else None

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
    p.attach("tracker", Probe("counter", counter))

    if not use_tee:
        p.add("fakesink", "sink", {"sync": True})
        p.link("tracker", "sink")
    else:
        p.add("tee", "tee", {})
        p.link("tracker", "tee")

        if use_osd:
            p.add("queue", "q1", {"leaky": 2, "max-size-buffers": 5})
            p.add("nvdsosd", "osd", {"gpu-id": 0})
            p.add("nvvideoconvert", "conv1", {})
            p.add("capsfilter", "caps1", {
                "caps": "video/x-raw(memory:NVMM),format=RGB",
            })
            p.add("fakesink", "sink1", {"sync": True})
            p.link("tee", "q1", "osd", "conv1", "caps1", "sink1")
            if use_buffer_probe:
                p.attach("caps1", Probe("buf", buf_probe))
        else:
            p.add("queue", "q1", {})
            p.add("fakesink", "sink1", {"sync": True})
            p.link("tee", "q1", "sink1")

        # Second branch: simple sink
        p.add("queue", "q2", {"leaky": 2, "max-size-buffers": 5})
        p.add("fakesink", "sink2", {"sync": False})
        p.link("tee", "q2", "sink2")

    p.start()
    time.sleep(8)

    src_str = ", ".join(f"src{s}:{c}" for s, c in sorted(counter.sources.items()))
    buf_str = f", buf_probes={buf_probe.count}" if buf_probe else ""
    print(f"  batches={counter.count}, [{src_str}]{buf_str}", flush=True)

    if counter.count > 50:
        print(f"  ✓ PASS — flowing", flush=True)
    else:
        print(f"  ✗ STALLED — only {counter.count} batches in 8s", flush=True)

    os._exit(0)


if __name__ == "__main__":
    test = sys.argv[1]
    v1, v2 = sys.argv[2], sys.argv[3]

    if test == "A":
        run_test("Test A: no tee (baseline)", v1, v2,
                 use_tee=False, use_osd=False, use_buffer_probe=False)
    elif test == "B":
        run_test("Test B: tee + 2 fakesinks (no OSD/probes)", v1, v2,
                 use_tee=True, use_osd=False, use_buffer_probe=False)
    elif test == "C":
        run_test("Test C: tee + OSD branch (no buffer probe)", v1, v2,
                 use_tee=True, use_osd=True, use_buffer_probe=False)
    elif test == "D":
        run_test("Test D: tee + OSD + buffer probe", v1, v2,
                 use_tee=True, use_osd=True, use_buffer_probe=True)
