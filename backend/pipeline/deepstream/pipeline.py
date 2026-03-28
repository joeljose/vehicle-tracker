"""DeepStream detection + tracking pipeline — Phase 2: decode + detect + track + display.

Builds a GStreamer pipeline via pyservicemaker that:
  1. Decodes a video file (nvurisrcbin via Flow API)
  2. Runs TrafficCamNet inference (nvinfer)
  3. Tracks vehicles with NvDCF (nvtracker) — persistent IDs
  4. Draws bounding boxes with track IDs (nvdsosd)
  5. Optionally encodes annotated output to MP4 (nvv4l2h264enc)
  6. Reports per-frame detection/tracking stats via a probe callback
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator
from pyservicemaker.flow import RenderMode
import os
import time

from backend.pipeline.deepstream.config import (
    PGIE_CONFIG,
    TRACKER_CONFIG,
    TRACKER_LIB,
    load_labels,
)
from backend.pipeline.trajectory import TrajectoryBuffer


class TrackingReporter(BatchMetadataOperator):
    """Probe callback that logs per-frame detection and tracking stats."""

    def __init__(self, labels: dict[int, str], report_interval: int = 100):
        super().__init__()
        self.labels = labels
        self.report_interval = report_interval
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts: dict[str, int] = {}
        self.start_time: float | None = None
        # Track lifecycle state
        self.active_tracks: dict[int, dict] = {}  # track_id -> {label, first_frame, last_frame, trajectory}
        self.lost_tracks: list[dict] = []

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                frame_detections = 0
                seen_track_ids = set()

                for obj_meta in frame_meta.object_items:
                    frame_detections += 1
                    label = self.labels.get(obj_meta.class_id, f"class_{obj_meta.class_id}")
                    self.class_counts[label] = self.class_counts.get(label, 0) + 1

                    track_id = obj_meta.object_id
                    seen_track_ids.add(track_id)

                    # Centroid from bbox
                    rect = obj_meta.rect_params
                    cx = int(rect.left + rect.width / 2)
                    cy = int(rect.top + rect.height / 2)

                    if track_id not in self.active_tracks:
                        traj = TrajectoryBuffer()
                        traj.append(cx, cy, self.frame_count)
                        self.active_tracks[track_id] = {
                            "label": label,
                            "first_frame": self.frame_count,
                            "last_frame": self.frame_count,
                            "trajectory": traj,
                        }
                        print(
                            f"New track #{track_id} ({label})",
                            flush=True,
                        )
                    else:
                        self.active_tracks[track_id]["last_frame"] = self.frame_count
                        self.active_tracks[track_id]["trajectory"].append(
                            cx, cy, self.frame_count
                        )

                self.total_detections += frame_detections

                # Detect lost tracks (not seen this frame)
                lost_ids = [
                    tid for tid in list(self.active_tracks)
                    if tid not in seen_track_ids
                ]
                for tid in lost_ids:
                    track = self.active_tracks.pop(tid)
                    lifetime = track["last_frame"] - track["first_frame"] + 1
                    trajectory = track["trajectory"].get_full()
                    print(
                        f"Track #{tid} lost after {lifetime} frames "
                        f"trajectory={trajectory}",
                        flush=True,
                    )
                    self.lost_tracks.append({
                        "track_id": tid,
                        "label": track["label"],
                        "lifetime": lifetime,
                        "trajectory": trajectory,
                    })

                if self.frame_count % self.report_interval == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    avg_det = self.total_detections / self.frame_count
                    print(
                        f"[Frame {self.frame_count}] "
                        f"detections={frame_detections} "
                        f"tracks={len(self.active_tracks)} "
                        f"avg={avg_det:.1f}/frame "
                        f"fps={fps:.1f}",
                        flush=True,
                    )
        except Exception as e:
            print(f"ERROR in TrackingReporter: {e}", flush=True)

    def summary(self) -> dict:
        """Return summary stats including tracking info."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        # Tracks still active at end count as completed too
        all_tracks = list(self.lost_tracks)
        for tid, track in self.active_tracks.items():
            all_tracks.append({
                "track_id": tid,
                "label": track["label"],
                "lifetime": track["last_frame"] - track["first_frame"] + 1,
                "trajectory": track["trajectory"].get_full(),
            })
        return {
            "frames": self.frame_count,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": (
                self.total_detections / self.frame_count if self.frame_count else 0
            ),
            "class_counts": dict(self.class_counts),
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
            "unique_tracks": len(all_tracks),
            "tracks": all_tracks,
        }


def build_pipeline(
    video_path: str,
    output_path: str | None = None,
    interval: int = 0,
) -> tuple[Pipeline, Flow, TrackingReporter]:
    """Build a DeepStream detection + tracking pipeline.

    Args:
        video_path: Path to the input video file.
        output_path: Path for annotated output MP4. None = discard output.
        interval: nvinfer interval (0 = every frame, 1 = every 2nd frame).

    Returns:
        (pipeline, flow, reporter) tuple. Call flow() to run.
    """
    file_uri = f"file://{os.path.abspath(video_path)}"
    labels = load_labels()
    reporter = TrackingReporter(labels)

    pipeline = Pipeline("vehicle-tracker")
    flow = Flow(pipeline, [file_uri])

    # Source → mux → infer
    infer_flow = flow.batch_capture([file_uri]).infer(PGIE_CONFIG, interval=interval)
    infer_element_name = infer_flow._streams[0]

    # infer → nvtracker (NvDCF)
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
        "ll-lib-file": TRACKER_LIB,
        "ll-config-file": TRACKER_CONFIG,
        "enable-past-frame": 1,
        "enable-batch-process": 1,
    })
    pipeline.link(infer_element_name, "tracker")

    # Attach probe after tracker
    tracker_flow = Flow(pipeline, ["tracker"])
    probed = tracker_flow.attach(Probe("reporter", reporter))

    if output_path:
        # tracker → OSD (draws bboxes + track IDs) → encode to MP4
        pipeline.add("nvdsosd", "osd", {"gpu-id": 0})
        pipeline.link("tracker", "osd")
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
    """Run the detection + tracking pipeline to completion and return stats.

    Args:
        video_path: Path to the input video file.
        output_path: Path for annotated output MP4. None = discard.
        interval: nvinfer interval.

    Returns:
        Detection and tracking summary dict.
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
        f"{summary['unique_tracks']} unique tracks, "
        f"{summary['fps']:.1f} fps",
        flush=True,
    )
    return summary
