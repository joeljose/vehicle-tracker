"""DeepStream detection + tracking pipeline.

Builds a GStreamer pipeline via pyservicemaker that:
  1. Decodes a video file (nvurisrcbin via Flow API)
  2. Runs YOLOv8s inference (nvinfer with EfficientNMS)
  3. Tracks vehicles with NvDCF (nvtracker) — persistent IDs
  4. Draws bounding boxes with track IDs (nvdsosd)
  5. Optionally encodes annotated output to MP4 (nvv4l2h264enc)
  6. Reports per-frame detection/tracking stats via a probe callback
  7. Captures best-photo crops per tracked vehicle (Phase 7)
"""

from collections import deque

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator, BufferOperator
import logging
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
    DirectionStateMachine,
)
from backend.pipeline.idle import IdleOptimizer
from backend.pipeline.roi import point_in_polygon
from backend.pipeline.shared import check_crossings, finalize_lost_track
from backend.pipeline.snapshot import BestPhotoTracker
from backend.pipeline.stitch import TrackStitcher
from backend.pipeline.trajectory import TrajectoryBuffer

# Max per-frame data entries per track (matches TrajectoryBuffer default)
_MAX_PER_FRAME_DATA = 300

# Project class IDs that count as a "vehicle" at runtime.
# Under the M8-P1.5 v2 single-class fine-tuned student, the only class id
# produced by the custom YOLO parser is 0 (vehicle), and all of them are
# vehicles by construction. Kept as a set for easy future extension.
VEHICLE_CLASS_IDS = {0}

logger = logging.getLogger(__name__)


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
        channel_id: int = 0,
        alert_callback=None,
        track_ended_callback=None,
    ):
        super().__init__()
        self.labels = labels
        self.roi_polygon = roi_polygon
        self.report_interval = report_interval
        self.channel_id = channel_id
        self.alert_callback = alert_callback
        self.track_ended_callback = track_ended_callback
        # Line crossing state
        self.lines = lines or {}
        self.stagnant_threshold_sec = 150.0
        self.crossings: list[dict] = []  # all crossing events
        self.transit_alerts: list[dict] = []  # completed transit alerts
        self.frame_count = 0
        self._current_pts_ms = 0  # PTS of current frame in ms
        self.total_detections = 0
        self.class_counts: dict[str, int] = {}
        self.start_time: float | None = None
        # Track lifecycle state
        self.active_tracks: dict[
            int, dict
        ] = {}  # track_id -> {label, first_frame, last_frame, trajectory, dsm}
        self.lost_tracks: list[dict] = []
        # Best-photo capture
        self.best_photo = best_photo
        self.snapshot_dir = snapshot_dir
        # Track stitching
        self.stitcher = TrackStitcher()
        self.merge_count = 0
        # ROI centroid for direction inference
        if roi_polygon and len(roi_polygon) >= 3:
            n = len(roi_polygon)
            self.roi_centroid = (
                sum(p[0] for p in roi_polygon) / n,
                sum(p[1] for p in roi_polygon) / n,
            )
        else:
            self.roi_centroid = None
        # Sequential track ID mapping — DeepStream IDs are huge integers
        # that overflow JavaScript's Number.MAX_SAFE_INTEGER.  We assign
        # small sequential IDs for display and API use.
        self._ds_to_seq: dict[int, int] = {}  # deepstream_id -> sequential_id
        self._next_seq_id = 1
        # Idle optimization
        self.idle_optimizer = IdleOptimizer()
        self.infer_node = None  # set by build_pipeline after construction

    def _seq_id(self, ds_id: int) -> int:
        """Map a DeepStream track ID to a small sequential ID."""
        if ds_id not in self._ds_to_seq:
            self._ds_to_seq[ds_id] = self._next_seq_id
            self._next_seq_id += 1
        return self._ds_to_seq[ds_id]

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                # Use PTS for accurate timestamps (ns → ms)
                self._current_pts_ms = int(frame_meta.buffer_pts / 1_000_000)
                frame_detections = 0
                seen_track_ids = set()

                for obj_meta in frame_meta.object_items:
                    # Single-class model: class 0 = vehicle. The filter is a
                    # belt-and-braces guard in case the parser ever emits
                    # anything else; under normal operation it's a no-op.
                    if obj_meta.class_id not in VEHICLE_CLASS_IDS:
                        continue

                    frame_detections += 1
                    label = self.labels.get(
                        obj_meta.class_id, f"class_{obj_meta.class_id}"
                    )
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
                            position=(cx, cy),
                            frame_number=self.frame_count,
                        )
                        if match:
                            # Merge: inherit trajectory, DSM, per_frame_data
                            traj = match["trajectory"]
                            traj.append(cx, cy, self.frame_count)
                            dsm = match["dsm"]
                            old_pfd = match.get("per_frame_data", [])
                            pfd = deque(old_pfd, maxlen=_MAX_PER_FRAME_DATA)
                            first_frame = match.get("first_frame", self.frame_count)
                            lost_tid = match["lost_track_id"]
                            self.merge_count += 1
                            gap_sec = match["gap_frames"] / 30.0
                            logger.info(
                                "Track #%d merged with lost track "
                                "#%d (dist=%dpx, gap=%.1fs, %d pfd)",
                                track_id,
                                lost_tid,
                                match["distance"],
                                gap_sec,
                                len(old_pfd),
                            )
                        else:
                            traj = TrajectoryBuffer()
                            traj.append(cx, cy, self.frame_count)
                            dsm = DirectionStateMachine()
                            pfd = deque(maxlen=_MAX_PER_FRAME_DATA)
                            first_frame = self.frame_count

                        pfd.append(
                            {
                                "frame": self.frame_count,
                                "bbox": [
                                    int(rect.left),
                                    int(rect.top),
                                    int(rect.width),
                                    int(rect.height),
                                ],
                                "centroid": [cx, cy],
                                "confidence": round(obj_meta.confidence, 3),
                                "timestamp_ms": self._current_pts_ms,
                            }
                        )
                        self.active_tracks[track_id] = {
                            "label": label,
                            "first_frame": first_frame,
                            "last_frame": self.frame_count,
                            "trajectory": traj,
                            "roi_active": in_roi,
                            "dsm": dsm,
                            "per_frame_data": pfd,
                        }
                        if not match:
                            roi_tag = "IN ROI" if in_roi else "OUT OF ROI"
                            logger.info(
                                "New track #%d (%s): %s",
                                track_id,
                                label,
                                roi_tag,
                            )
                    else:
                        track_state = self.active_tracks[track_id]
                        # Get previous centroid before appending new one
                        prev_traj = track_state["trajectory"].get_full()
                        prev_cx, prev_cy = prev_traj[-1][0], prev_traj[-1][1]

                        track_state["last_frame"] = self.frame_count
                        track_state["trajectory"].append(cx, cy, self.frame_count)
                        track_state["per_frame_data"].append(
                            {
                                "frame": self.frame_count,
                                "bbox": [
                                    int(rect.left),
                                    int(rect.top),
                                    int(rect.width),
                                    int(rect.height),
                                ],
                                "centroid": [cx, cy],
                                "confidence": round(obj_meta.confidence, 3),
                                "timestamp_ms": self._current_pts_ms,
                            }
                        )
                        # Update ROI status (track can move in/out)
                        was_in_roi = track_state["roi_active"]
                        track_state["roi_active"] = in_roi
                        if in_roi != was_in_roi:
                            roi_tag = "IN ROI" if in_roi else "OUT OF ROI"
                            logger.debug(
                                "Track #%d: %s",
                                track_id,
                                roi_tag,
                            )

                        # Line crossing check (only for ROI-active tracks)
                        if in_roi and self.lines:
                            self._check_crossings(
                                track_id, (prev_cx, prev_cy), (cx, cy)
                            )

                        # Stagnant check (any ROI-active track)
                        if in_roi:
                            dsm = track_state["dsm"]
                            traj = track_state["trajectory"].get_full()
                            if dsm.check_stagnant(traj):
                                logger.warning(
                                    "STAGNANT: Track #%d after %ds",
                                    track_id,
                                    self.stagnant_threshold_sec,
                                )
                                if self.alert_callback:
                                    self.alert_callback(
                                        {
                                            "type": "stagnant_alert",
                                            "track_id": self._seq_id(track_id),
                                            "label": track_state["label"],
                                            "position": (cx, cy),
                                            "stationary_duration_frames": (
                                                self.frame_count
                                                - dsm.stopped_since_frame
                                            ),
                                            "first_seen_frame": track_state[
                                                "first_frame"
                                            ],
                                            "last_seen_frame": self.frame_count,
                                            "channel": self.channel_id,
                                        }
                                    )

                    # Best-photo scoring (ROI-active tracks only)
                    if in_roi and self.best_photo:
                        area = rect.width * rect.height
                        self.best_photo.score(
                            track_id=self._seq_id(track_id),
                            area=area,
                            confidence=obj_meta.confidence,
                            bbox=(rect.left, rect.top, rect.width, rect.height),
                        )

                self.total_detections += frame_detections

                # Expire old stitcher tracks — do deferred processing
                for exp_tid, exp_state in self.stitcher.expire(self.frame_count):
                    self.finalize_lost_track(exp_tid, exp_state)

                # Detect lost tracks (not seen this frame)
                lost_ids = [
                    tid for tid in list(self.active_tracks) if tid not in seen_track_ids
                ]
                for tid in lost_ids:
                    track = self.active_tracks.pop(tid)
                    lifetime = track["last_frame"] - track["first_frame"] + 1
                    logger.info("Track #%d lost after %d frames", tid, lifetime)

                    if track["roi_active"]:
                        # Buffer for stitching — defer direction inference
                        self.stitcher.on_track_lost(
                            track_id=tid,
                            trajectory=track["trajectory"],
                            dsm=track["dsm"],
                            frame_number=self.frame_count,
                        )
                        # Store label and frame info for finalization
                        self.stitcher.lost_tracks[tid]["label"] = track["label"]
                        self.stitcher.lost_tracks[tid]["lifetime"] = lifetime
                        self.stitcher.lost_tracks[tid]["first_frame"] = track[
                            "first_frame"
                        ]
                        self.stitcher.lost_tracks[tid]["per_frame_data"] = track.get(
                            "per_frame_data", []
                        )
                    else:
                        # Non-ROI track — finalize immediately
                        if self.best_photo:
                            self.best_photo.save(self._seq_id(tid), self.snapshot_dir)
                        lost_track = {
                            "track_id": self._seq_id(tid),
                            "label": track["label"],
                            "lifetime": lifetime,
                            "trajectory": track["trajectory"].get_full(),
                            "roi_active": False,
                            "direction_state": track["dsm"].state.value,
                        }
                        self.lost_tracks.append(lost_track)
                        if self.track_ended_callback:
                            self.track_ended_callback(lost_track)

                # Idle optimization
                transition = self.idle_optimizer.update(
                    num_detections=frame_detections,
                    num_active_tracks=len(self.active_tracks),
                )
                if transition and self.infer_node is not None:
                    new_interval = self.idle_optimizer.recommended_interval
                    self.infer_node.set({"interval": new_interval})
                    mode = "IDLE" if transition == "idle" else "ACTIVE"
                    logger.info("%s MODE: interval=%d", mode, new_interval)

                if self.frame_count % self.report_interval == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    avg_det = self.total_detections / self.frame_count
                    logger.debug(
                        "[Frame %d] detections=%d tracks=%d avg=%.1f/frame fps=%.1f",
                        self.frame_count,
                        frame_detections,
                        len(self.active_tracks),
                        avg_det,
                        fps,
                    )
        except Exception as e:
            logger.error("TrackingReporter: %s", e)

    def finalize_lost_track(self, tid: int, stitcher_state: dict) -> None:
        """Handle a track that expired from the stitcher (no merge found).

        Public so the adapter / standalone run_pipeline path can drain
        remaining stitcher state on shutdown without reaching into a private.
        """
        seq_tid = self._seq_id(tid)

        def on_alert(alert):
            self.transit_alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        def on_track_ended(lost):
            self.lost_tracks.append(lost)
            if self.track_ended_callback:
                self.track_ended_callback(lost)

        finalize_lost_track(
            track_id=seq_tid,
            stitcher_state=stitcher_state,
            lines=self.lines,
            roi_centroid=self.roi_centroid,
            frame_count=self.frame_count,
            channel_id=self.channel_id,
            best_photo=self.best_photo,
            snapshot_dir=self.snapshot_dir,
            alert_callback=on_alert,
            track_ended_callback=on_track_ended,
        )

    def _check_crossings(
        self,
        track_id: int,
        prev: tuple[float, float],
        curr: tuple[float, float],
    ) -> None:
        """Check if a track crossed any entry/exit line."""
        seq_tid = self._seq_id(track_id)

        def on_alert(alert):
            # Remap to sequential ID for external consumers
            alert["track_id"] = seq_tid
            self.transit_alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        check_crossings(
            track_id=track_id,
            prev=prev,
            curr=curr,
            lines=self.lines,
            active_tracks=self.active_tracks,
            roi_centroid=self.roi_centroid,
            frame_count=self.frame_count,
            channel_id=self.channel_id,
            alert_callback=on_alert,
        )

    def summary(self) -> dict:
        """Return summary stats including tracking info."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        # Tracks still active at end count as completed too
        all_tracks = list(self.lost_tracks)
        for tid, track in self.active_tracks.items():
            all_tracks.append(
                {
                    "track_id": self._seq_id(tid),
                    "label": track["label"],
                    "lifetime": track["last_frame"] - track["first_frame"] + 1,
                    "trajectory": track["trajectory"].get_full(),
                    "roi_active": track["roi_active"],
                    "direction_state": track["dsm"].state.value,
                }
            )
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
            "lines": {
                arm_id: {
                    "label": line.label,
                    "junction_side": line.junction_side,
                }
                for arm_id, line in self.lines.items()
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
            logger.error("FrameExtractor: %s", e)
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
    pipeline.add(
        "nvtracker",
        "tracker",
        {
            "tracker-width": 640,
            "tracker-height": 384,
            "gpu-id": 0,
            "ll-lib-file": TRACKER_LIB,
            "ll-config-file": TRACKER_CONFIG,
        },
    )
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
        pipeline.add(
            "capsfilter",
            "snap_caps",
            {
                "caps": "video/x-raw(memory:NVMM),format=RGB",
            },
        )
        pipeline.add("fakesink", "snap_sink", {"sync": False})
        pipeline.link("split", "snap_queue", "snap_conv", "snap_caps", "snap_sink")
        pipeline.attach("snap_caps", Probe("extractor", extractor))

        # Branch 2: OSD → encode
        pipeline.add("queue", "out_queue", {})
        pipeline.add("nvdsosd", "osd", {"gpu-id": 0, "display-text": False})
        pipeline.link("split", "out_queue", "osd")
        osd_flow = Flow(pipeline, ["osd"])
        osd_flow.encode(f"file://{os.path.abspath(output_path)}")
    else:
        # No output: linear — tracker → RGB conversion → [buffer probe] → fakesink
        pipeline.add("nvvideoconvert", "snap_conv", {})
        pipeline.add(
            "capsfilter",
            "snap_caps",
            {
                "caps": "video/x-raw(memory:NVMM),format=RGB",
            },
        )
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
        video_path,
        output_path,
        interval,
        roi_polygon=roi_polygon,
        lines=lines,
        snapshot_dir=snapshot_dir,
    )

    logger.info("Starting pipeline: %s", video_path)
    logger.info("  interval=%d", interval)
    if output_path:
        logger.info("  output=%s", output_path)

    flow()

    # Finalize remaining stitcher tracks (no merge will happen)
    for tid, state in list(reporter.stitcher.lost_tracks.items()):
        reporter.finalize_lost_track(tid, state)
    reporter.stitcher.lost_tracks.clear()

    # Save any remaining active tracks' photos
    if reporter.best_photo:
        for tid in list(reporter.active_tracks):
            reporter.best_photo.save(tid, reporter.snapshot_dir)

    summary = reporter.summary()
    logger.info(
        "Pipeline complete: %d frames, %d detections, %d unique tracks, "
        "%d merges, %.1f fps",
        summary["frames"],
        summary["total_detections"],
        summary["unique_tracks"],
        summary["merges"],
        summary["fps"],
    )
    return summary
