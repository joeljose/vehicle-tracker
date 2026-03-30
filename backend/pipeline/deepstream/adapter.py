"""DeepStreamPipeline adapter — implements PipelineBackend Protocol.

Wraps existing DeepStream components (TrackingReporter, FrameExtractor,
build_pipeline) into a multi-channel streaming model with per-channel
pipeline lifecycle and asyncio thread bridging.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from pyservicemaker import Pipeline, Flow, Probe, BufferOperator

from backend.pipeline.deepstream.config import (
    PGIE_CONFIG,
    TRACKER_CONFIG,
    TRACKER_LIB,
    load_labels,
)
from backend.pipeline.deepstream.pipeline import TrackingReporter, FrameExtractor
from backend.pipeline.protocol import ChannelPhase, Detection, FrameResult
from backend.pipeline.snapshot import BestPhotoTracker

logger = logging.getLogger(__name__)


class MjpegExtractor(BufferOperator):
    """BufferOperator that extracts annotated JPEG frames from post-OSD buffers.

    Runs on the OSD branch of the pipeline. Converts GPU buffer to numpy
    via CuPy, encodes to JPEG, and invokes the frame callback at ~15fps.
    """

    def __init__(self, channel_id: int, callback, reporter: TrackingReporter):
        super().__init__()
        self._channel_id = channel_id
        self._callback = callback
        self._reporter = reporter
        self._frame_count = 0
        self._last_emit_time = 0.0
        self._min_interval = 1.0 / 15.0  # ~15fps cap

    def handle_buffer(self, buffer):
        try:
            self._frame_count += 1

            # Rate limit to ~15fps
            now = time.monotonic()
            if now - self._last_emit_time < self._min_interval:
                return True

            if self._callback is None:
                return True

            import cupy as cp
            import cv2

            for batch_id in range(buffer.batch_size):
                tensor = buffer.extract(batch_id)
                gpu_arr = cp.from_dlpack(tensor)
                frame_rgb = cp.asnumpy(gpu_arr)

                # Capsfilter outputs RGB; cv2.imencode expects BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Encode to JPEG
                _, jpeg_buf = cv2.imencode(
                    ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                jpeg_bytes = jpeg_buf.tobytes()

                # Build detections from reporter's active tracks
                detections = []
                for tid, track in self._reporter.active_tracks.items():
                    traj = track["trajectory"].get_full()
                    if traj:
                        cx, cy = traj[-1][0], traj[-1][1]
                        detections.append(Detection(
                            track_id=self._reporter._seq_id(tid),
                            class_name=track["label"],
                            bbox=(0, 0, 0, 0),  # OSD already drew boxes
                            confidence=0.0,
                            centroid=(cx, cy),
                        ))

                result = FrameResult(
                    channel_id=self._channel_id,
                    frame_number=self._reporter.frame_count,
                    timestamp_ms=int(self._reporter.frame_count * (1000 / 30)),
                    detections=detections,
                    annotated_jpeg=jpeg_bytes,
                    phase=(
                        "analytics"
                        if self._reporter.roi_polygon
                        else "setup"
                    ),
                    idle_mode=self._reporter.idle_optimizer.is_idle,
                )
                self._callback(result)

            self._last_emit_time = now
            return True
        except Exception as e:
            logger.error("MjpegExtractor: %s", e)
            return True


@dataclass
class _ChannelState:
    """Per-channel pipeline state."""

    source: str
    phase: ChannelPhase = ChannelPhase.SETUP
    pipeline: object | None = None  # pyservicemaker Pipeline
    reporter: TrackingReporter | None = None
    best_photo: BestPhotoTracker | None = None
    mjpeg_extractor: MjpegExtractor | None = None
    roi_polygon: list[tuple[float, float]] | None = None
    entry_exit_lines: dict = field(default_factory=dict)


class DeepStreamPipeline:
    """PipelineBackend implementation wrapping DeepStream via pyservicemaker.

    Uses a single shared GStreamer pipeline for all channels:
    nvstreammux → nvinfer → nvtracker → nvstreamdemux → per-channel OSD/MJPEG.
    """

    def __init__(self):
        self.channels: dict[int, str] = {}
        self._states: dict[int, _ChannelState] = {}
        self._frame_callback: Callable[[FrameResult], None] | None = None
        self._alert_callback: Callable[[dict], None] | None = None
        self._track_ended_callback: Callable[[dict], None] | None = None
        self._phase_callback: Callable[[int, object, object], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._confidence_threshold: float = 0.5
        self._inference_interval: int = 0
        self._labels: dict[int, str] | None = None
        self._started = False
        # Shared pipeline state (M5)
        self._shared_pipeline: Pipeline | None = None
        self._batch_router = None
        self._source_index: int = 0  # monotonic source counter for mux pads

    # -- Lifecycle --

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._labels = load_labels()
        self._started = True
        logger.info("DeepStreamPipeline started")

    def stop(self) -> None:
        for channel_id in list(self._states):
            self._stop_channel_pipeline(channel_id)
        self.channels.clear()
        self._states.clear()
        if self._shared_pipeline is not None:
            self._destroy_shared_pipeline()
        self._started = False
        logger.info("DeepStreamPipeline stopped")

    # -- Channel management --

    def add_channel(self, channel_id: int, source: str) -> None:
        if channel_id in self.channels:
            raise ValueError(f"Channel {channel_id} already exists")
        self.channels[channel_id] = source
        state = _ChannelState(source=source)
        self._states[channel_id] = state
        self._ensure_shared_pipeline()
        self._add_channel_to_pipeline(channel_id)
        logger.info("Channel %d added: %s", channel_id, source)

    def remove_channel(self, channel_id: int) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        self._stop_channel_pipeline(channel_id)
        del self.channels[channel_id]
        del self._states[channel_id]
        # Destroy shared pipeline if no channels remain
        if not self._states and self._shared_pipeline is not None:
            self._destroy_shared_pipeline()
        logger.info("Channel %d removed", channel_id)

    def configure_channel(
        self,
        channel_id: int,
        roi_polygon: list[tuple[float, float]],
        entry_exit_lines: dict,
    ) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        state = self._states[channel_id]
        state.roi_polygon = roi_polygon
        state.entry_exit_lines = entry_exit_lines
        logger.info("Channel %d configured with ROI + %d lines",
                     channel_id, len(entry_exit_lines))

    def set_channel_phase(self, channel_id: int, phase: ChannelPhase) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        state = self._states[channel_id]
        old_phase = state.phase

        if phase == ChannelPhase.ANALYTICS:
            self._stop_channel_pipeline(channel_id)
            self._start_analytics_pipeline(channel_id)
            state.phase = ChannelPhase.ANALYTICS
        elif phase == ChannelPhase.REVIEW:
            self._stop_channel_pipeline(channel_id)
            state.phase = ChannelPhase.REVIEW
        else:
            state.phase = phase

        logger.info("Channel %d: %s -> %s",
                     channel_id, old_phase.value, phase.value)

    def get_channel_phase(self, channel_id: int) -> ChannelPhase:
        if channel_id not in self._states:
            raise KeyError(f"Channel {channel_id} not found")
        return self._states[channel_id].phase

    def get_channel_config(self, channel_id: int) -> dict:
        """Return the stored ROI/lines config for a channel."""
        if channel_id not in self._states:
            raise KeyError(f"Channel {channel_id} not found")
        state = self._states[channel_id]
        return {
            "entry_exit_lines": state.entry_exit_lines or {},
        }

    # -- Configuration --

    def set_confidence_threshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold

    def set_inference_interval(self, interval: int) -> None:
        self._inference_interval = interval

    # -- Callbacks --

    def register_frame_callback(
        self, callback: Callable[[FrameResult], None]
    ) -> None:
        self._frame_callback = callback

    def register_alert_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        self._alert_callback = callback

    def register_track_ended_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        self._track_ended_callback = callback

    def register_phase_callback(self, callback) -> None:
        self._phase_callback = callback

    # -- Snapshots --

    def get_snapshot(self, track_id: int) -> bytes | None:
        import cv2
        from pathlib import Path

        # Try in-memory first (tracks still being processed)
        for state in self._states.values():
            if state.best_photo:
                crop = state.best_photo.best_crops.get(track_id)
                if crop is not None:
                    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode(".jpg", bgr)
                    return buf.tobytes()

        # Fall back to disk (finalized tracks)
        snapshot_path = Path("snapshots") / f"{track_id}.jpg"
        if snapshot_path.exists():
            return snapshot_path.read_bytes()

        return None

    # -- Thread bridging --

    def _safe_callback(self, fn, *args):
        """Dispatch callback from GStreamer thread to asyncio event loop."""
        if fn is None:
            return
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(fn, *args)
        else:
            fn(*args)

    # -- Shared pipeline (M5) --

    def _ensure_shared_pipeline(self) -> None:
        """Create the shared pipeline if it doesn't exist yet."""
        if self._shared_pipeline is not None:
            return

        from backend.pipeline.deepstream.batch_router import BatchMetadataRouter

        pipeline = Pipeline("shared")

        # Mux — sources added dynamically per channel
        pipeline.add("nvstreammux", "mux", {
            "batch-size": 8,  # max channels
            "batched-push-timeout": 33000,
            "width": 1920,
            "height": 1080,
        })

        # Shared inference
        pipeline.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
        pipeline.link("mux", "infer")

        # Shared tracker
        pipeline.add("nvtracker", "tracker", {
            "tracker-width": 640,
            "tracker-height": 384,
            "gpu-id": 0,
            "ll-lib-file": TRACKER_LIB,
            "ll-config-file": TRACKER_CONFIG,
        })
        pipeline.link("infer", "tracker")

        # Batch metadata router — dispatches per-source frames to channel reporters
        self._batch_router = BatchMetadataRouter({})
        pipeline.attach("tracker", Probe("batch_router", self._batch_router))

        # Demux — per-channel output branches added dynamically
        pipeline.add("nvstreamdemux", "demux", {})
        pipeline.link("tracker", "demux")

        self._shared_pipeline = pipeline
        pipeline.start()
        logger.info("Shared pipeline created and started")

    def _add_channel_to_pipeline(self, channel_id: int) -> None:
        """Add a source and per-channel output branch to the shared pipeline."""
        state = self._states[channel_id]
        source = state.source
        file_uri = f"file://{os.path.abspath(source)}"
        pipeline = self._shared_pipeline

        # Per-channel reporter (preview mode — no ROI/lines)
        best_photo = BestPhotoTracker()
        state.best_photo = best_photo
        reporter = TrackingReporter(
            self._labels,
            roi_polygon=None,
            lines=None,
            best_photo=best_photo,
            channel_id=channel_id,
        )
        state.reporter = reporter
        self._batch_router.add_reporter(channel_id, reporter)

        # Frame callback bridged to asyncio
        def on_frame(result):
            self._safe_callback(self._frame_callback, result)

        mjpeg_extractor = MjpegExtractor(channel_id, on_frame, reporter)
        state.mjpeg_extractor = mjpeg_extractor
        frame_extractor = FrameExtractor(best_photo)

        # Add source to mux
        src_idx = self._source_index
        self._source_index += 1
        src_name = f"src_{channel_id}"
        pipeline.add("nvurisrcbin", src_name, {
            "uri": file_uri,
            "file-loop": True,  # preview loops
        })
        pipeline.link((src_name, "mux"), ("", "sink_%u"))
        state._src_name = src_name

        # Per-channel demux output → OSD → MJPEG
        q_name = f"q_{channel_id}"
        osd_name = f"osd_{channel_id}"
        conv_name = f"conv_{channel_id}"
        caps_name = f"caps_{channel_id}"
        sink_name = f"sink_{channel_id}"
        pipeline.add("queue", q_name, {"leaky": 2, "max-size-buffers": 5})
        pipeline.add("nvdsosd", osd_name, {"gpu-id": 0})
        pipeline.add("nvvideoconvert", conv_name, {})
        pipeline.add("capsfilter", caps_name, {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", sink_name, {"sync": True})
        pipeline.link(("demux", q_name), ("src_%u", ""))
        pipeline.link(q_name, osd_name, conv_name, caps_name, sink_name)
        pipeline.attach(caps_name, Probe(f"mjpeg_{channel_id}", mjpeg_extractor))

        state.pipeline = pipeline
        logger.info("Channel %d added to shared pipeline", channel_id)

    def _destroy_shared_pipeline(self) -> None:
        """Tear down the shared pipeline."""
        if self._shared_pipeline is None:
            return
        import threading as _threading

        pipeline = self._shared_pipeline
        self._shared_pipeline = None
        self._batch_router = None

        def _deferred_stop():
            try:
                pipeline.stop()
            except Exception:
                pass

        _threading.Thread(target=_deferred_stop, daemon=True).start()
        logger.info("Shared pipeline destroyed")

    # -- Legacy pipeline builders (kept for reference, will be removed) --

    def _start_preview_pipeline(self, channel_id: int) -> None:
        """Build and start a looping preview pipeline (Setup phase)."""
        state = self._states[channel_id]
        source = state.source
        file_uri = f"file://{os.path.abspath(source)}"

        best_photo = BestPhotoTracker()
        state.best_photo = best_photo

        # Reporter without ROI/lines (preview mode)
        reporter = TrackingReporter(
            self._labels,
            roi_polygon=None,
            lines=None,
            best_photo=best_photo,
            channel_id=channel_id,
        )
        state.reporter = reporter

        # Frame callback bridged to asyncio
        def on_frame(result):
            self._safe_callback(self._frame_callback, result)

        mjpeg_extractor = MjpegExtractor(channel_id, on_frame, reporter)
        state.mjpeg_extractor = mjpeg_extractor
        frame_extractor = FrameExtractor(best_photo)

        pipeline = Pipeline(f"preview-ch{channel_id}")
        flow = Flow(pipeline, [file_uri])

        # Source → mux → infer
        infer_flow = flow.batch_capture(
            [file_uri]
        ).infer(PGIE_CONFIG, interval=self._inference_interval)
        infer_element_name = infer_flow._streams[0]

        # Enable file looping on the source element for preview.
        # nvurisrcbin supports file-loop natively.
        try:
            src = pipeline["batch_capture-source-0_0"]
            src.set({"file-loop": True})
        except Exception as e:
            logger.warning("Could not set file-loop: %s", e)

        reporter.infer_node = pipeline[infer_element_name]

        # infer → nvtracker
        tracker_name = f"tracker_ch{channel_id}"
        pipeline.add("nvtracker", tracker_name, {
            "tracker-width": 640,
            "tracker-height": 384,
            "gpu-id": 0,
            "ll-lib-file": TRACKER_LIB,
            "ll-config-file": TRACKER_CONFIG,
        })
        pipeline.link(infer_element_name, tracker_name)
        pipeline.attach(tracker_name, Probe(f"reporter_ch{channel_id}", reporter))

        # tracker → tee → [OSD + MJPEG branch] + [RGB + snapshot branch]
        tee_name = f"tee_ch{channel_id}"
        pipeline.add("tee", tee_name, {})
        pipeline.link(tracker_name, tee_name)

        # Branch 1: OSD → MJPEG
        osd_queue = f"osd_queue_ch{channel_id}"
        osd_name = f"osd_ch{channel_id}"
        osd_conv = f"osd_conv_ch{channel_id}"
        osd_caps = f"osd_caps_ch{channel_id}"
        osd_sink = f"osd_sink_ch{channel_id}"
        pipeline.add("queue", osd_queue, {})
        pipeline.add("nvdsosd", osd_name, {"gpu-id": 0})
        pipeline.add("nvvideoconvert", osd_conv, {})
        pipeline.add("capsfilter", osd_caps, {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", osd_sink, {"sync": True})
        pipeline.link(tee_name, osd_queue, osd_name, osd_conv, osd_caps, osd_sink)
        pipeline.attach(osd_caps, Probe(f"mjpeg_ch{channel_id}", mjpeg_extractor))

        # Branch 2: RGB for best-photo
        snap_queue = f"snap_queue_ch{channel_id}"
        snap_conv = f"snap_conv_ch{channel_id}"
        snap_caps = f"snap_caps_ch{channel_id}"
        snap_sink = f"snap_sink_ch{channel_id}"
        pipeline.add("queue", snap_queue, {})
        pipeline.add("nvvideoconvert", snap_conv, {})
        pipeline.add("capsfilter", snap_caps, {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", snap_sink, {"sync": False})
        pipeline.link(tee_name, snap_queue, snap_conv, snap_caps, snap_sink)
        pipeline.attach(snap_caps, Probe(f"extractor_ch{channel_id}", frame_extractor))

        state.pipeline = pipeline
        pipeline.start()
        logger.info("Preview pipeline started for channel %d", channel_id)

    def _start_analytics_pipeline(self, channel_id: int) -> None:
        """Build and start an analytics pipeline (plays once, with ROI/lines)."""
        state = self._states[channel_id]
        source = state.source
        file_uri = f"file://{os.path.abspath(source)}"

        best_photo = BestPhotoTracker()
        state.best_photo = best_photo

        # Convert entry_exit_lines dict format to LineSeg objects
        from backend.pipeline.direction import LineSeg
        lines = {}
        if state.entry_exit_lines:
            for arm_id, line_data in state.entry_exit_lines.items():
                lines[arm_id] = LineSeg(
                    label=line_data["label"],
                    start=tuple(line_data["start"]),
                    end=tuple(line_data["end"]),
                )

        # Alert callback bridged to asyncio
        def on_alert(alert):
            self._safe_callback(self._alert_callback, alert)

        def on_track_ended(track):
            self._safe_callback(self._track_ended_callback, track)

        reporter = TrackingReporter(
            self._labels,
            roi_polygon=state.roi_polygon,
            lines=lines if lines else None,
            best_photo=best_photo,
            channel_id=channel_id,
            alert_callback=on_alert,
            track_ended_callback=on_track_ended,
        )
        state.reporter = reporter

        # Frame callback bridged to asyncio
        def on_frame(result):
            self._safe_callback(self._frame_callback, result)

        mjpeg_extractor = MjpegExtractor(channel_id, on_frame, reporter)
        state.mjpeg_extractor = mjpeg_extractor
        frame_extractor = FrameExtractor(best_photo)

        pipeline = Pipeline(f"analytics-ch{channel_id}")
        flow = Flow(pipeline, [file_uri])

        # Source → mux → infer (single play, no loop)
        infer_flow = flow.batch_capture([file_uri]).infer(
            PGIE_CONFIG, interval=self._inference_interval
        )
        infer_element_name = infer_flow._streams[0]

        reporter.infer_node = pipeline[infer_element_name]

        # infer → nvtracker
        tracker_name = f"tracker_ch{channel_id}"
        pipeline.add("nvtracker", tracker_name, {
            "tracker-width": 640,
            "tracker-height": 384,
            "gpu-id": 0,
            "ll-lib-file": TRACKER_LIB,
            "ll-config-file": TRACKER_CONFIG,
        })
        pipeline.link(infer_element_name, tracker_name)
        pipeline.attach(tracker_name, Probe(f"reporter_ch{channel_id}", reporter))

        # tracker → tee → [OSD + MJPEG] + [RGB + snapshot]
        tee_name = f"tee_ch{channel_id}"
        pipeline.add("tee", tee_name, {})
        pipeline.link(tracker_name, tee_name)

        # Branch 1: OSD → MJPEG
        osd_queue = f"osd_queue_ch{channel_id}"
        osd_name = f"osd_ch{channel_id}"
        osd_conv = f"osd_conv_ch{channel_id}"
        osd_caps = f"osd_caps_ch{channel_id}"
        osd_sink = f"osd_sink_ch{channel_id}"
        pipeline.add("queue", osd_queue, {})
        pipeline.add("nvdsosd", osd_name, {"gpu-id": 0})
        pipeline.add("nvvideoconvert", osd_conv, {})
        pipeline.add("capsfilter", osd_caps, {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", osd_sink, {"sync": True})
        pipeline.link(tee_name, osd_queue, osd_name, osd_conv, osd_caps, osd_sink)
        pipeline.attach(osd_caps, Probe(f"mjpeg_ch{channel_id}", mjpeg_extractor))

        # Branch 2: RGB for best-photo
        snap_queue = f"snap_queue_ch{channel_id}"
        snap_conv = f"snap_conv_ch{channel_id}"
        snap_caps = f"snap_caps_ch{channel_id}"
        snap_sink = f"snap_sink_ch{channel_id}"
        pipeline.add("queue", snap_queue, {})
        pipeline.add("nvvideoconvert", snap_conv, {})
        pipeline.add("capsfilter", snap_caps, {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
        pipeline.add("fakesink", snap_sink, {"sync": False})
        pipeline.link(tee_name, snap_queue, snap_conv, snap_caps, snap_sink)
        pipeline.attach(snap_caps, Probe(f"extractor_ch{channel_id}", frame_extractor))

        state.pipeline = pipeline

        # EOS handler: auto-transition to Review via start_with_callback
        def on_message(message):
            if "EOS" in str(message) or "eos" in str(message):
                logger.info("Analytics EOS on channel %d", channel_id)
                self._safe_callback(self._on_eos, channel_id)

        pipeline._instance.start_with_callback(on_message)
        logger.info("Analytics pipeline started for channel %d", channel_id)

    def _on_eos(self, channel_id: int) -> None:
        """Handle end-of-stream — auto-transition to Review."""
        if channel_id not in self._states:
            return
        state = self._states[channel_id]
        if state.phase == ChannelPhase.ANALYTICS:
            previous = state.phase
            self._stop_channel_pipeline(channel_id)
            state.phase = ChannelPhase.REVIEW
            logger.info("Channel %d auto-transitioned to REVIEW (EOS)", channel_id)
            self._safe_callback(
                self._phase_callback, channel_id, ChannelPhase.REVIEW, previous
            )

    def _stop_channel_pipeline(self, channel_id: int) -> None:
        """Stop the running pipeline for a channel."""
        state = self._states.get(channel_id)
        if state is None or state.pipeline is None:
            return

        # Finalize remaining stitcher tracks
        if state.reporter:
            for tid, sstate in list(state.reporter.stitcher.lost_tracks.items()):
                state.reporter._finalize_lost_track(tid, sstate)
            state.reporter.stitcher.lost_tracks.clear()

            # Save remaining active track photos (using sequential IDs)
            if state.best_photo:
                for tid in list(state.reporter.active_tracks):
                    seq_tid = state.reporter._seq_id(tid)
                    state.best_photo.save(seq_tid, state.reporter.snapshot_dir)

        # Disable MJPEG callback before deferred stop to prevent
        # the old pipeline from interleaving frames with the new one
        if state.mjpeg_extractor:
            state.mjpeg_extractor._callback = None
            state.mjpeg_extractor = None

        pipeline = state.pipeline
        state.pipeline = None
        state.reporter = None

        # pyservicemaker's pipeline.stop() triggers C++ terminate() when
        # called while an asyncio event loop is running (GLib main loop
        # conflict). Workaround: disable file-loop so the pipeline will
        # naturally reach EOS, then call stop() from a background thread
        # after EOS arrives. The pipeline object is released immediately
        # so the adapter can build a new pipeline for the next phase.
        import threading as _threading

        try:
            src = pipeline["batch_capture-source-0_0"]
            src.set({"file-loop": False})
        except Exception:
            pass

        def _deferred_stop():
            try:
                pipeline.wait()  # blocks until EOS
                pipeline.stop()
            except Exception:
                pass

        _threading.Thread(target=_deferred_stop, daemon=True).start()
        logger.debug("Pipeline stop initiated for channel %d", channel_id)
