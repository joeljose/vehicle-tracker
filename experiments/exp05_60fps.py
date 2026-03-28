"""
Experiment 5: 60fps source handling.

Verifies that a 60fps source decodes at full rate while inference
runs at 30fps using interval=1 (infer every 2nd frame).

Usage (inside container):
    python3 experiments/exp05_60fps.py /data/site_videos/<60fps_video.mp4> [max_frames]
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator
from pyservicemaker.flow import RenderMode
import sys
import os
import time

PGIE_CONFIG = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"


class FPSAnalyzer(BatchMetadataOperator):
    """Analyze frame rate and inference behavior with interval=1."""

    def __init__(self, max_frames=600):
        super().__init__()
        self.max_frames = max_frames
        self.frame_count = 0
        self.start_time = None
        self.prev_pts = None
        self.pts_deltas = []
        self.frames_with_detections = 0
        self.frames_without_detections = 0
        self.total_detections = 0

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()
                print("Probe active, processing 60fps source with interval=1...", flush=True)

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                pts = frame_meta.buffer_pts

                if self.prev_pts is not None:
                    delta_ns = pts - self.prev_pts
                    self.pts_deltas.append(delta_ns)
                self.prev_pts = pts

                n_obj = sum(1 for _ in frame_meta.object_items)
                self.total_detections += n_obj
                if n_obj > 0:
                    self.frames_with_detections += 1
                else:
                    self.frames_without_detections += 1

                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    avg_delta = sum(self.pts_deltas[-50:]) / min(50, len(self.pts_deltas)) if self.pts_deltas else 0
                    source_fps = 1e9 / avg_delta if avg_delta > 0 else 0
                    print(f"[Frame {self.frame_count}] pipeline_fps={fps:.1f}, "
                          f"source_fps={source_fps:.1f}, objects={n_obj}", flush=True)

                if self.frame_count >= self.max_frames:
                    self._print_summary()
                    os._exit(0)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)

    def _print_summary(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        pipeline_fps = self.frame_count / elapsed if elapsed > 0 else 0

        avg_delta = sum(self.pts_deltas) / len(self.pts_deltas) if self.pts_deltas else 0
        source_fps = 1e9 / avg_delta if avg_delta > 0 else 0

        # With interval=1, every other frame should have detections
        # Frames with detections should be ~50% of total
        det_ratio = self.frames_with_detections / self.frame_count if self.frame_count else 0

        lines = [
            "",
            "=" * 60,
            "EXPERIMENT 5: 60FPS SOURCE HANDLING",
            "=" * 60,
            f"Frames processed:       {self.frame_count}",
            f"Pipeline FPS:           {pipeline_fps:.1f}",
            f"Source FPS (from PTS):  {source_fps:.1f}",
            f"Elapsed:                {elapsed:.1f}s",
            f"",
            f"Frames WITH detections: {self.frames_with_detections} ({det_ratio*100:.1f}%)",
            f"Frames w/o detections:  {self.frames_without_detections} ({(1-det_ratio)*100:.1f}%)",
            f"Total detections:       {self.total_detections}",
            f"",
            f"Expected: ~50% frames have detections (interval=1 = every 2nd frame)",
            f"Result:   {'PASS' if 0.3 < det_ratio < 0.7 else 'UNEXPECTED RATIO'}",
            "=" * 60,
        ]
        print("\n".join(lines), flush=True)


def main(file_path, max_frames):
    analyzer = FPSAnalyzer(max_frames=max_frames)
    file_uri = f"file://{os.path.abspath(file_path)}"

    pipeline = Pipeline("exp05-60fps")
    flow = Flow(pipeline, [file_uri])
    flow.batch_capture([file_uri]) \
        .infer(PGIE_CONFIG, interval=1) \
        .attach(Probe("analyzer", analyzer)) \
        .render(mode=RenderMode.DISCARD, enable_osd=False)
    flow()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <60fps_video.mp4> [max_frames=600]")
        sys.exit(1)

    file_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    main(file_path, max_frames)
