"""DeepStream detection + tracking pipeline.

Builds a GStreamer pipeline via pyservicemaker that:
  1. Decodes a video file (nvurisrcbin via Flow API)
  2. Runs TrafficCamNet inference (nvinfer)
  3. Tracks vehicles with NvDCF (nvtracker) — persistent IDs
  4. Draws bounding boxes with track IDs (nvdsosd)
  5. Optionally encodes annotated output to MP4 (nvv4l2h264enc)
  6. Reports per-frame detection/tracking stats via a probe callback
  7. Captures best-photo crops per tracked vehicle (Phase 7)
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator, BufferOperator
import os
import time

from backend.pipeline.deepstream.config import (
    PGIE_CONFIG,
    TRACKER_CONFIG,
    TRACKER_LIB,
    load_labels,
)
from backend.pipeline.direction import (
    LineSeg,
    LineCalibration,
    DirectionStateMachine,
    check_line_crossing,
)
from backend.pipeline.idle import IdleOptimizer
from backend.pipeline.roi import point_in_polygon
from backend.pipeline.snapshot import BestPhotoTracker
from backend.pipeline.stitch import TrackStitcher
from backend.pipeline.trajectory import TrajectoryBuffer


class TrackingReporter(BatchMetadataOperator):
    """Probe callback that logs per-frame detection and tracking stats."""

    def __init__(
        self,
        labels: dict[int, str],
        roi_polygon: list[tuple[float, float]] | None = None,
        lines: dict[str, LineSeg] | None = None,
        report_interval: int = 100,
        best_photo: BestPhotoTracker | None = None,
        snapshot_dir: str = "snapshots",
    ):
        super().__init__()
        self.labels = labels
        self.roi_polygon = roi_polygon
        self.report_interval = report_interval
        # Line crossing state
        self.line_calibrations: dict[str, LineCalibration] = {}
        if lines:
            for arm_id, line in lines.items():
                self.line_calibrations[arm_id] = LineCalibration(line)
        self.lines = lines or {}
        self.stagnant_threshold_sec = 150.0
        self.crossings: list[dict] = []  # all crossing events
        self.transit_alerts: list[dict] = []  # completed transit alerts
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts: dict[str, int] = {}
        self.start_time: float | None = None
        # Track lifecycle state
        self.active_tracks: dict[int, dict] = {}  # track_id -> {label, first_frame, last_frame, trajectory, dsm}
        self.lost_tracks: list[dict] = []
        # Best-photo capture
        self.best_photo = best_photo
        self.snapshot_dir = snapshot_dir
        # Track stitching
        self.stitcher = TrackStitcher()
        self.merge_count = 0
        # Idle optimization
        self.idle_optimizer = IdleOptimizer()
        self.infer_node = None  # set by build_pipeline after construction

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

                    # ROI check
                    in_roi = (
                        point_in_polygon((cx, cy), self.roi_polygon)
                        if self.roi_polygon
                        else True
                    )

                    if track_id not in self.active_tracks:
                        # Check stitcher for a matching lost track
                        match = self.stitcher.find_match(
                            position=(cx, cy), frame_number=self.frame_count,
                        )
                        if match:
                            # Merge: inherit trajectory, DSM from lost track
                            traj = match["trajectory"]
                            traj.append(cx, cy, self.frame_count)
                            dsm = match["dsm"]
                            lost_tid = match["lost_track_id"]
                            self.merge_count += 1
                            gap_sec = match["gap_frames"] / 30.0
                            print(
                                f"Track #{track_id} merged with lost track "
                                f"#{lost_tid} (dist={match['distance']:.0f}px, "
                                f"gap={gap_sec:.1f}s)",
                                flush=True,
                            )
                        else:
                            traj = TrajectoryBuffer()
                            traj.append(cx, cy, self.frame_count)
                            dsm = DirectionStateMachine()

                        self.active_tracks[track_id] = {
                            "label": label,
                            "first_frame": self.frame_count,
                            "last_frame": self.frame_count,
                            "trajectory": traj,
                            "roi_active": in_roi,
                            "dsm": dsm,
                        }
                        if not match:
                            roi_tag = "IN ROI" if in_roi else "OUT OF ROI"
                            print(
                                f"New track #{track_id} ({label}): {roi_tag}",
                                flush=True,
                            )
                    else:
                        track_state = self.active_tracks[track_id]
                        # Get previous centroid before appending new one
                        prev_traj = track_state["trajectory"].get_full()
                        prev_cx, prev_cy = prev_traj[-1][0], prev_traj[-1][1]

                        track_state["last_frame"] = self.frame_count
                        track_state["trajectory"].append(cx, cy, self.frame_count)
                        # Update ROI status (track can move in/out)
                        was_in_roi = track_state["roi_active"]
                        track_state["roi_active"] = in_roi
                        if in_roi != was_in_roi:
                            roi_tag = "IN ROI" if in_roi else "OUT OF ROI"
                            print(
                                f"Track #{track_id}: {roi_tag}",
                                flush=True,
                            )

                        # Line crossing check (only for ROI-active tracks)
                        if in_roi and self.line_calibrations:
                            self._check_crossings(
                                track_id, (prev_cx, prev_cy), (cx, cy)
                            )

                        # Stagnant check (any ROI-active track)
                        if in_roi:
                            dsm = track_state["dsm"]
                            traj = track_state["trajectory"].get_full()
                            if dsm.check_stagnant(traj):
                                print(
                                    f"STAGNANT: Track #{track_id} after "
                                    f"{self.stagnant_threshold_sec}s",
                                    flush=True,
                                )

                    # Best-photo scoring (ROI-active tracks only)
                    if in_roi and self.best_photo:
                        area = rect.width * rect.height
                        self.best_photo.score(
                            track_id=track_id,
                            area=area,
                            confidence=obj_meta.confidence,
                            bbox=(rect.left, rect.top, rect.width, rect.height),
                        )

                self.total_detections += frame_detections

                # Expire old stitcher tracks — do deferred processing
                for exp_tid, exp_state in self.stitcher.expire(self.frame_count):
                    self._finalize_lost_track(exp_tid, exp_state)

                # Detect lost tracks (not seen this frame)
                lost_ids = [
                    tid for tid in list(self.active_tracks)
                    if tid not in seen_track_ids
                ]
                for tid in lost_ids:
                    track = self.active_tracks.pop(tid)
                    lifetime = track["last_frame"] - track["first_frame"] + 1
                    print(
                        f"Track #{tid} lost after {lifetime} frames",
                        flush=True,
                    )

                    if track["roi_active"]:
                        # Buffer for stitching — defer direction inference
                        self.stitcher.on_track_lost(
                            track_id=tid,
                            trajectory=track["trajectory"],
                            dsm=track["dsm"],
                            frame_number=self.frame_count,
                        )
                        # Store label for finalization
                        self.stitcher.lost_tracks[tid]["label"] = track["label"]
                        self.stitcher.lost_tracks[tid]["lifetime"] = lifetime
                    else:
                        # Non-ROI track — finalize immediately
                        if self.best_photo:
                            self.best_photo.save(tid, self.snapshot_dir)
                        self.lost_tracks.append({
                            "track_id": tid,
                            "label": track["label"],
                            "lifetime": lifetime,
                            "trajectory": track["trajectory"].get_full(),
                            "roi_active": False,
                            "direction_state": track["dsm"].state.value,
                        })

                # Idle optimization
                transition = self.idle_optimizer.update(
                    num_detections=frame_detections,
                    num_active_tracks=len(self.active_tracks),
                )
                if transition and self.infer_node is not None:
                    new_interval = self.idle_optimizer.recommended_interval
                    self.infer_node.set({"interval": new_interval})
                    mode = "IDLE" if transition == "idle" else "ACTIVE"
                    print(
                        f"{mode} MODE: interval={new_interval}",
                        flush=True,
                    )

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

    def _finalize_lost_track(self, tid: int, stitcher_state: dict) -> None:
        """Handle a track that expired from the stitcher (no merge found).

        Performs deferred direction inference, best-photo save, and
        adds the track to the lost_tracks summary.
        """
        dsm = stitcher_state["dsm"]
        trajectory = stitcher_state["trajectory"]
        trajectory_full = trajectory.get_full()
        label = stitcher_state.get("label", "unknown")
        lifetime = stitcher_state.get("lifetime", 0)

        # Save best photo
        if self.best_photo:
            self.best_photo.save(tid, self.snapshot_dir)

        # Direction inference
        if self.lines:
            alert = dsm.on_track_lost(trajectory_full, self.lines)
            if alert:
                alert["track_id"] = tid
                alert["frame"] = self.frame_count
                self.transit_alerts.append(alert)
                print(
                    f"TRANSIT: Track #{tid}: "
                    f"{alert['entry_label']} -> {alert['exit_label']} "
                    f"({alert['method']})",
                    flush=True,
                )

        self.lost_tracks.append({
            "track_id": tid,
            "label": label,
            "lifetime": lifetime,
            "trajectory": trajectory_full,
            "roi_active": True,
            "direction_state": dsm.state.value,
        })

    def _check_crossings(
        self,
        track_id: int,
        prev: tuple[float, float],
        curr: tuple[float, float],
    ) -> None:
        """Check if a track crossed any entry/exit line."""
        for arm_id, cal in self.line_calibrations.items():
            direction = check_line_crossing(cal.line, prev, curr)
            if direction is None:
                continue

            # Feed to calibration
            cal.observe(direction)
            if not cal.calibrated:
                continue

            crossing_type = cal.classify(direction)
            print(
                f"Track #{track_id} crossed {cal.line.label} line ({crossing_type})",
                flush=True,
            )
            self.crossings.append({
                "track_id": track_id,
                "arm": arm_id,
                "label": cal.line.label,
                "type": crossing_type,
                "frame": self.frame_count,
            })

            # Feed to direction state machine
            track_state = self.active_tracks.get(track_id)
            if track_state:
                dsm = track_state["dsm"]
                alert = dsm.on_crossing(
                    arm_id, cal.line.label, crossing_type,
                    trajectory=track_state["trajectory"].get_full(),
                    arms=self.lines,
                )
                if alert:
                    alert["track_id"] = track_id
                    alert["frame"] = self.frame_count
                    self.transit_alerts.append(alert)
                    print(
                        f"TRANSIT: Track #{track_id}: "
                        f"{alert['entry_label']} -> {alert['exit_label']} "
                        f"({alert['method']})",
                        flush=True,
                    )

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
                "roi_active": track["roi_active"],
                "direction_state": track["dsm"].state.value,
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
            "crossings": list(self.crossings),
            "transit_alerts": list(self.transit_alerts),
            "merges": self.merge_count,
            "calibrations": {
                arm_id: {
                    "label": cal.line.label,
                    "calibrated": cal.calibrated,
                    "entry_sign": cal.entry_sign,
                    "observations": len(cal.observations),
                }
                for arm_id, cal in self.line_calibrations.items()
            },
        }


class FrameExtractor(BufferOperator):
    """BufferOperator probe that extracts frame data for best-photo cropping.

    Runs on the RGB-converted branch of the pipeline.  Each frame, it
    converts the GPU buffer to a numpy array via CuPy and calls
    BestPhotoTracker.extract_crops() for any pending best-score updates.
    """

    def __init__(self, best_photo: BestPhotoTracker):
        super().__init__()
        self.best_photo = best_photo

    def handle_buffer(self, buffer):
        try:
            if not self.best_photo.pending_crops:
                return True

            import cupy as cp

            for batch_id in range(buffer.batch_size):
                tensor = buffer.extract(batch_id)
                gpu_arr = cp.from_dlpack(tensor)
                frame = cp.asnumpy(gpu_arr)
                self.best_photo.extract_crops(frame)

            return True
        except Exception as e:
            print(f"ERROR in FrameExtractor: {e}", flush=True)
            return True


def build_pipeline(
    video_path: str,
    output_path: str | None = None,
    interval: int = 0,
    roi_polygon: list[tuple[float, float]] | None = None,
    lines: dict[str, LineSeg] | None = None,
    snapshot_dir: str = "snapshots",
) -> tuple[Pipeline, Flow, TrackingReporter]:
    """Build a DeepStream detection + tracking pipeline.

    Args:
        video_path: Path to the input video file.
        output_path: Path for annotated output MP4. None = discard output.
        interval: nvinfer interval (0 = every frame, 1 = every 2nd frame).
        roi_polygon: ROI polygon vertices. Tracks outside are tagged roi_active=False.
        lines: Entry/exit line segments keyed by arm ID.
        snapshot_dir: Directory for best-photo JPEG output.

    Returns:
        (pipeline, flow, reporter) tuple. Call flow() to run.
    """
    file_uri = f"file://{os.path.abspath(video_path)}"
    labels = load_labels()
    best_photo = BestPhotoTracker()
    reporter = TrackingReporter(
        labels,
        roi_polygon=roi_polygon,
        lines=lines,
        best_photo=best_photo,
        snapshot_dir=snapshot_dir,
    )
    extractor = FrameExtractor(best_photo)

    pipeline = Pipeline("vehicle-tracker")
    flow = Flow(pipeline, [file_uri])

    # Source → mux → infer
    infer_flow = flow.batch_capture([file_uri]).infer(PGIE_CONFIG, interval=interval)
    infer_element_name = infer_flow._streams[0]

    # Give reporter a reference to the infer element for idle optimization
    reporter.infer_node = pipeline[infer_element_name]

    # infer → nvtracker (NvDCF)
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
        "ll-lib-file": TRACKER_LIB,
        "ll-config-file": TRACKER_CONFIG,
    })
    pipeline.link(infer_element_name, "tracker")

    # Attach metadata probe on tracker (scores detections before buffer flows downstream)
    pipeline.attach("tracker", Probe("reporter", reporter))

    if output_path:
        # With output: tracker → tee → [RGB snapshot branch] + [OSD → encode branch]
        pipeline.add("tee", "split", {})
        pipeline.link("tracker", "split")

        # Branch 1: RGB for best-photo
        pipeline.add("queue", "snap_queue", {})
        pipeline.add("nvvideoconvert", "snap_conv", {})
        pipeline.add("capsfilter", "snap_caps", {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", "snap_sink", {"sync": False})
        pipeline.link("split", "snap_queue", "snap_conv", "snap_caps", "snap_sink")
        pipeline.attach("snap_caps", Probe("extractor", extractor))

        # Branch 2: OSD → encode
        pipeline.add("queue", "out_queue", {})
        pipeline.add("nvdsosd", "osd", {"gpu-id": 0})
        pipeline.link("split", "out_queue", "osd")
        osd_flow = Flow(pipeline, ["osd"])
        osd_flow.encode(f"file://{os.path.abspath(output_path)}")
    else:
        # No output: linear — tracker → RGB conversion → [buffer probe] → fakesink
        pipeline.add("nvvideoconvert", "snap_conv", {})
        pipeline.add("capsfilter", "snap_caps", {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", "snap_sink", {"sync": False})
        pipeline.link("tracker", "snap_conv", "snap_caps", "snap_sink")
        pipeline.attach("snap_caps", Probe("extractor", extractor))

    return pipeline, flow, reporter


def run_pipeline(
    video_path: str,
    output_path: str | None = None,
    interval: int = 0,
    roi_polygon: list[tuple[float, float]] | None = None,
    lines: dict[str, LineSeg] | None = None,
    snapshot_dir: str = "snapshots",
) -> dict:
    """Run the detection + tracking pipeline to completion and return stats.

    Args:
        video_path: Path to the input video file.
        output_path: Path for annotated output MP4. None = discard.
        interval: nvinfer interval.
        roi_polygon: ROI polygon vertices for filtering.
        lines: Entry/exit line segments keyed by arm ID.
        snapshot_dir: Directory for best-photo JPEG output.

    Returns:
        Detection and tracking summary dict.
    """
    pipeline, flow, reporter = build_pipeline(
        video_path, output_path, interval,
        roi_polygon=roi_polygon, lines=lines,
        snapshot_dir=snapshot_dir,
    )

    print(f"Starting pipeline: {video_path}", flush=True)
    print(f"  interval={interval}", flush=True)
    if output_path:
        print(f"  output={output_path}", flush=True)

    flow()

    # Finalize remaining stitcher tracks (no merge will happen)
    for tid, state in list(reporter.stitcher.lost_tracks.items()):
        reporter._finalize_lost_track(tid, state)
    reporter.stitcher.lost_tracks.clear()

    # Save any remaining active tracks' photos
    if reporter.best_photo:
        for tid in list(reporter.active_tracks):
            reporter.best_photo.save(tid, reporter.snapshot_dir)

    summary = reporter.summary()
    print(
        f"\nPipeline complete: {summary['frames']} frames, "
        f"{summary['total_detections']} detections, "
        f"{summary['unique_tracks']} unique tracks, "
        f"{summary['merges']} merges, "
        f"{summary['fps']:.1f} fps",
        flush=True,
    )
    return summary
