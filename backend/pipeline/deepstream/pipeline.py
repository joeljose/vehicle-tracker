"""DeepStream detection pipeline — Phase 1: decode + detect + annotated output.

Builds a GStreamer pipeline via pyservicemaker that:
  1. Decodes a video file (nvurisrcbin via Flow API)
  2. Runs TrafficCamNet inference (nvinfer)
  3. Draws bounding boxes (nvdsosd)
  4. Optionally encodes annotated output to MP4 (nvv4l2h264enc)
  5. Reports per-frame detection stats via a probe callback
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator
from pyservicemaker.flow import RenderMode
import os
import time

from backend.pipeline.deepstream.config import PGIE_CONFIG, load_labels


class DetectionReporter(BatchMetadataOperator):
    """Probe callback that logs per-frame detection stats."""

    def __init__(self, labels: dict[int, str], report_interval: int = 100):
        super().__init__()
        self.labels = labels
        self.report_interval = report_interval
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts: dict[str, int] = {}
        self.start_time: float | None = None

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                frame_detections = 0

                for obj_meta in frame_meta.object_items:
                    frame_detections += 1
                    label = self.labels.get(obj_meta.class_id, f"class_{obj_meta.class_id}")
                    self.class_counts[label] = self.class_counts.get(label, 0) + 1

                self.total_detections += frame_detections

                if self.frame_count % self.report_interval == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    avg_det = self.total_detections / self.frame_count
                    print(
                        f"[Frame {self.frame_count}] "
                        f"detections={frame_detections} "
                        f"avg={avg_det:.1f}/frame "
                        f"fps={fps:.1f}",
                        flush=True,
                    )
        except Exception as e:
            print(f"ERROR in DetectionReporter: {e}", flush=True)

    def summary(self) -> dict:
        """Return summary stats."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "frames": self.frame_count,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": (
                self.total_detections / self.frame_count if self.frame_count else 0
            ),
            "class_counts": dict(self.class_counts),
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
        }


def build_pipeline(
    video_path: str,
    output_path: str | None = None,
    interval: int = 0,
) -> tuple[Pipeline, Flow, DetectionReporter]:
    """Build a DeepStream detection pipeline.

    Args:
        video_path: Path to the input video file.
        output_path: Path for annotated output MP4. None = discard output.
        interval: nvinfer interval (0 = every frame, 1 = every 2nd frame).

    Returns:
        (pipeline, flow, reporter) tuple. Call flow() to run.
    """
    file_uri = f"file://{os.path.abspath(video_path)}"
    labels = load_labels()
    reporter = DetectionReporter(labels)

    pipeline = Pipeline("vehicle-tracker")
    flow = Flow(pipeline, [file_uri])

    # Source → mux → infer → probe (stats)
    infer_flow = flow.batch_capture([file_uri]).infer(PGIE_CONFIG, interval=interval)
    # The infer element name is the current stream in the flow
    infer_element_name = infer_flow._streams[0]
    probed = infer_flow.attach(Probe("reporter", reporter))

    if output_path:
        # infer → OSD (draws bboxes) → encode to MP4
        pipeline.add("nvdsosd", "osd", {"gpu-id": 0})
        pipeline.link(infer_element_name, "osd")
        osd_flow = Flow(pipeline, ["osd"])
        osd_flow.encode(f"file://{os.path.abspath(output_path)}")
    else:
        # No output — discard frames after probe
        probed.render(mode=RenderMode.DISCARD, enable_osd=False)

    return pipeline, flow, reporter


def run_pipeline(
    video_path: str,
    output_path: str | None = None,
    interval: int = 0,
) -> dict:
    """Run the detection pipeline to completion and return stats.

    Args:
        video_path: Path to the input video file.
        output_path: Path for annotated output MP4. None = discard.
        interval: nvinfer interval.

    Returns:
        Detection summary dict.
    """
    pipeline, flow, reporter = build_pipeline(video_path, output_path, interval)

    print(f"Starting pipeline: {video_path}", flush=True)
    print(f"  interval={interval}", flush=True)
    if output_path:
        print(f"  output={output_path}", flush=True)

    flow()

    summary = reporter.summary()
    print(
        f"\nPipeline complete: {summary['frames']} frames, "
        f"{summary['total_detections']} detections, "
        f"{summary['fps']:.1f} fps",
        flush=True,
    )
    return summary
