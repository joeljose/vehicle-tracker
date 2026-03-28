"""
Experiment 1: TrafficCamNet detection quality on junction footage.

Runs DeepStream pipeline with TrafficCamNet on a video file, counts
detections per frame, and reports detection stats.

Usage (inside container):
    python3 experiments/exp01_detection.py /data/site_videos/<video.mp4> [max_frames]
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator
from pyservicemaker.flow import RenderMode
import sys
import os
import time

PGIE_CONFIG = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"

CLASS_LABELS = {0: "car", 1: "bicycle", 2: "person", 3: "roadsign"}


class DetectionCounter(BatchMetadataOperator):
    """Count detections per frame and collect stats."""

    def __init__(self, max_frames=300):
        super().__init__()
        self.frame_count = 0
        self.total_detections = 0
        self.max_frames = max_frames
        self.class_counts = {}
        self.confidence_sum = 0.0
        self.frames_with_zero = 0
        self.start_time = None

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()
                print("Probe active, processing frames...", flush=True)

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                frame_detections = 0

                for obj_meta in frame_meta.object_items:
                    frame_detections += 1
                    class_id = obj_meta.class_id
                    confidence = obj_meta.confidence
                    label = CLASS_LABELS.get(class_id, f"unknown_{class_id}")
                    self.class_counts[label] = self.class_counts.get(label, 0) + 1
                    self.confidence_sum += confidence

                self.total_detections += frame_detections
                if frame_detections == 0:
                    self.frames_with_zero += 1

                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[Frame {self.frame_count}] detections={frame_detections}, "
                          f"total={self.total_detections}, fps={fps:.1f}", flush=True)

                if self.frame_count >= self.max_frames:
                    self.print_summary()
                    os._exit(0)
        except Exception as e:
            import traceback
            print(f"ERROR in handle_metadata: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)

    def print_summary(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_det = self.total_detections / self.frame_count if self.frame_count > 0 else 0
        avg_conf = self.confidence_sum / self.total_detections if self.total_detections > 0 else 0

        lines = [
            "",
            "=" * 60,
            "EXPERIMENT 1: DETECTION RESULTS",
            "=" * 60,
            f"Frames processed:       {self.frame_count}",
            f"Processing FPS:         {fps:.1f}",
            f"Total detections:       {self.total_detections}",
            f"Avg detections/frame:   {avg_det:.1f}",
            f"Frames with 0 dets:     {self.frames_with_zero} ({100*self.frames_with_zero/self.frame_count:.1f}%)",
            f"Avg confidence:         {avg_conf:.3f}",
            f"Class distribution:     {self.class_counts}",
            f"Elapsed:                {elapsed:.1f}s",
            "=" * 60,
        ]
        print("\n".join(lines), flush=True)


def main(file_path, max_frames):
    counter = DetectionCounter(max_frames=max_frames)
    file_uri = f"file://{os.path.abspath(file_path)}"

    pipeline = Pipeline("exp01-detection")
    flow = Flow(pipeline, [file_uri])
    flow.batch_capture([file_uri]) \
        .infer(PGIE_CONFIG) \
        .attach(Probe("counter", counter)) \
        .render(mode=RenderMode.DISCARD, enable_osd=False)
    flow()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4> [max_frames=300]")
        sys.exit(1)

    file_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(file_path, max_frames)
