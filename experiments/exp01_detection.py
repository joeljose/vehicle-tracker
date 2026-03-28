"""
Experiment 1: TrafficCamNet detection quality on junction footage.

Runs DeepStream pipeline with TrafficCamNet on a video file, counts
detections per frame, and reports detection stats.

Usage (inside container):
    python3 experiments/exp01_detection.py /data/site_videos/<video.mp4> [max_frames]
"""

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator
from multiprocessing import Process
import sys
import os
import time

PGIE_CONFIG = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"


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
        if self.start_time is None:
            self.start_time = time.time()

        for frame_meta in batch_meta.frame_items:
            self.frame_count += 1
            frame_detections = 0

            for obj_meta in frame_meta.object_items:
                frame_detections += 1
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1
                self.confidence_sum += confidence

            self.total_detections += frame_detections
            if frame_detections == 0:
                self.frames_with_zero += 1

            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"[Frame {self.frame_count}] detections={frame_detections}, "
                      f"total={self.total_detections}, fps={fps:.1f}")

            if self.frame_count >= self.max_frames:
                self.print_summary()
                os._exit(0)

    def print_summary(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_det = self.total_detections / self.frame_count if self.frame_count > 0 else 0
        avg_conf = self.confidence_sum / self.total_detections if self.total_detections > 0 else 0

        print("\n" + "=" * 60)
        print("EXPERIMENT 1: DETECTION RESULTS")
        print("=" * 60)
        print(f"Frames processed:       {self.frame_count}")
        print(f"Processing FPS:         {fps:.1f}")
        print(f"Total detections:       {self.total_detections}")
        print(f"Avg detections/frame:   {avg_det:.1f}")
        print(f"Frames with 0 dets:     {self.frames_with_zero} ({100*self.frames_with_zero/self.frame_count:.1f}%)")
        print(f"Avg confidence:         {avg_conf:.3f}")
        print(f"Class distribution:     {self.class_counts}")
        print(f"Elapsed:                {elapsed:.1f}s")
        print("=" * 60)


def main(file_path, max_frames):
    counter = DetectionCounter(max_frames=max_frames)

    file_uri = f"file://{os.path.abspath(file_path)}"

    pipeline = Pipeline("exp01-detection")
    pipeline.add("uridecodebin", "src", {"uri": file_uri})
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1,
        "batched-push-timeout": 33000,
        "width": 1920,
        "height": 1080,
    })
    pipeline.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
    pipeline.add("fakesink", "sink")
    pipeline.link(("src", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "infer", "sink")
    pipeline.attach("infer", Probe("counter", counter))
    pipeline.start().wait()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4> [max_frames=300]")
        sys.exit(1)

    file_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    process = Process(target=main, args=(file_path, max_frames))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nTerminating...")
        process.terminate()
