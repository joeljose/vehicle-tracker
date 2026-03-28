"""
Experiment 4: Runtime interval change.

Tests whether nvinfer's interval property can be changed while the
pipeline is in PLAYING state without crashing.

Usage (inside container):
    python3 experiments/exp04_interval.py /data/site_videos/<video.mp4>
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator
from pyservicemaker.flow import RenderMode
import sys
import os
import time

PGIE_CONFIG = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"


class IntervalTester(BatchMetadataOperator):
    """Test interval changes at runtime."""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.frame_count = 0
        self.start_time = None
        self.phase = 0  # 0: initial (interval=0), 1: interval=5, 2: interval=15, 3: back to 0
        self.phase_frames = []
        self.phase_start_time = None
        self.phase_frame_start = 0

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()
                self.phase_start_time = time.time()
                print("Probe active, processing frames...", flush=True)
                print(f"Phase 0: interval=0 (every frame)", flush=True)

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                n_obj = sum(1 for _ in frame_meta.object_items)

                # Change interval at specific frame counts
                if self.frame_count == 200 and self.phase == 0:
                    self._record_phase()
                    self.phase = 1
                    self._set_interval(5)

                elif self.frame_count == 400 and self.phase == 1:
                    self._record_phase()
                    self.phase = 2
                    self._set_interval(15)

                elif self.frame_count == 600 and self.phase == 2:
                    self._record_phase()
                    self.phase = 3
                    self._set_interval(0)

                elif self.frame_count == 800:
                    self._record_phase()
                    self._print_summary()
                    os._exit(0)

                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[Frame {self.frame_count}] phase={self.phase}, "
                          f"objects={n_obj}, fps={fps:.1f}", flush=True)

        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)

    def _set_interval(self, interval):
        try:
            infer_elem = self.pipeline["infer-infer-0"]
            infer_elem.set({"interval": interval})
            actual = infer_elem.get("interval")
            print(f"Phase {self.phase}: set interval={interval}, "
                  f"readback={actual}", flush=True)
            self.phase_start_time = time.time()
            self.phase_frame_start = self.frame_count
        except Exception as e:
            print(f"Failed to set interval: {e}", flush=True)
            # Document alternative
            print("Alternative: would need to rebuild pipeline", flush=True)

    def _record_phase(self):
        elapsed = time.time() - self.phase_start_time
        frames = self.frame_count - self.phase_frame_start
        fps = frames / elapsed if elapsed > 0 else 0
        self.phase_frames.append({
            "phase": self.phase,
            "frames": frames,
            "elapsed": elapsed,
            "fps": fps,
        })

    def _print_summary(self):
        lines = [
            "",
            "=" * 60,
            "EXPERIMENT 4: RUNTIME INTERVAL CHANGE",
            "=" * 60,
        ]
        for pf in self.phase_frames:
            interval = {0: 0, 1: 5, 2: 15, 3: 0}[pf["phase"]]
            lines.append(
                f"Phase {pf['phase']} (interval={interval:2d}): "
                f"{pf['frames']:4d} frames in {pf['elapsed']:.2f}s = {pf['fps']:.1f} fps"
            )
        lines.append("=" * 60)
        print("\n".join(lines), flush=True)


def main(file_path):
    file_uri = f"file://{os.path.abspath(file_path)}"

    pipeline = Pipeline("exp04-interval")
    flow = Flow(pipeline, [file_uri])
    tester = IntervalTester(pipeline)
    flow.batch_capture([file_uri]) \
        .infer(PGIE_CONFIG) \
        .attach(Probe("tester", tester)) \
        .render(mode=RenderMode.DISCARD, enable_osd=False)
    flow()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4>")
        sys.exit(1)

    main(sys.argv[1])
