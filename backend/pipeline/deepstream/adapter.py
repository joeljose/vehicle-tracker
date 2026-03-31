"""DeepStreamPipeline adapter — implements PipelineBackend Protocol.

Uses a single shared GStreamer pipeline for all channels:
nvstreammux → nvinfer → nvtracker → nvstreamdemux → per-channel OSD/MJPEG.
Channels are added/removed dynamically via soft-removal (pyservicemaker
has no element remove API, so sources drain via file-loop=False).
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from pyservicemaker import Pipeline, Probe, BufferOperator

from backend.pipeline.deepstream.config import (
    PGIE_CONFIG,
    TRACKER_CONFIG,
    TRACKER_LIB,
    load_labels,
)
from backend.pipeline.deepstream.pipeline import TrackingReporter
from backend.pipeline.protocol import ChannelPhase, Detection, FrameResult
from backend.pipeline.snapshot import BestPhotoTracker
from backend.pipeline.stream_recovery import CircuitBreaker

logger = logging.getLogger(__name__)


class MjpegExtractor(BufferOperator):
    """BufferOperator that extracts annotated JPEG frames from post-OSD buffers.

    Runs on the OSD branch of the pipeline. Converts GPU buffer to numpy
    via CuPy, encodes to JPEG, and invokes the frame callback at ~15fps.

    Also keeps a reference to the most recent JPEG as `last_frame` for
    Phase 3 frozen-frame replay (YouTube Live sources have no recorded video).
    """

    def __init__(self, channel_id: int, callback, reporter: TrackingReporter):
        super().__init__()
        self._channel_id = channel_id
        self._callback = callback
        self._reporter = reporter
        self._frame_count = 0
        self._last_emit_time = 0.0
        self._min_interval = 1.0 / 15.0  # ~15fps cap
        self.last_frame: bytes | None = None  # most recent JPEG for Phase 3 replay

    def handle_buffer(self, buffer):
        try:
            self._frame_count += 1

            import cupy as cp
            import cv2

            # Extract best-photo crops on every frame (not rate-limited).
            # Runs post-OSD so crops include annotations — acceptable until
            # pyservicemaker supports tee without stalling BufferOperator.
            best_photo = self._reporter.best_photo
            if best_photo and best_photo.pending_crops:
                for batch_id in range(buffer.batch_size):
                    tensor = buffer.extract(batch_id)
                    gpu_arr = cp.from_dlpack(tensor)
                    frame = cp.asnumpy(gpu_arr)
                    best_photo.extract_crops(frame)

            # Rate limit MJPEG emission to ~15fps
            now = time.monotonic()
            if now - self._last_emit_time < self._min_interval:
                return True

            if self._callback is None:
                return True

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
                        detections.append(
                            Detection(
                                track_id=self._reporter._seq_id(tid),
                                class_name=track["label"],
                                bbox=(0, 0, 0, 0),  # OSD already drew boxes
                                confidence=0.0,
                                centroid=(cx, cy),
                            )
                        )

                # Keep reference to most recent JPEG for Phase 3 replay
                self.last_frame = jpeg_bytes

                result = FrameResult(
                    channel_id=self._channel_id,
                    frame_number=self._reporter.frame_count,
                    timestamp_ms=int(self._reporter.frame_count * (1000 / 30)),
                    detections=detections,
                    annotated_jpeg=jpeg_bytes,
                    phase=("analytics" if self._reporter.roi_polygon else "setup"),
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
    _source_idx: int | None = None  # mux pad index (matches frame_meta.source_id)
    _src_name: str | None = None  # GStreamer element name (e.g. "src_0")
    source_type: str = "file"  # "file" or "youtube_live"
    original_url: str | None = None  # YouTube URL for recovery (None for files)


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
        # Stream recovery (M6)
        self._circuit_breaker = CircuitBreaker()
        self._recovering: set[int] = set()  # channel_ids currently in recovery

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
            self._finalize_channel_state(channel_id)
        self.channels.clear()
        self._states.clear()
        if self._shared_pipeline is not None:
            self._destroy_shared_pipeline()
        self._started = False
        logger.info("DeepStreamPipeline stopped")

    # -- Channel management --

    def add_channel(
        self,
        channel_id: int,
        source: str,
        *,
        source_type: str = "file",
        original_url: str | None = None,
    ) -> None:
        if channel_id in self.channels:
            raise ValueError(f"Channel {channel_id} already exists")
        self.channels[channel_id] = source
        state = _ChannelState(
            source=source, source_type=source_type, original_url=original_url
        )
        self._states[channel_id] = state
        self._circuit_breaker.reset(channel_id)
        self._rebuild_shared_pipeline()
        logger.info("Channel %d added: %s (type=%s)", channel_id, source, source_type)

    def remove_channel(self, channel_id: int) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        self._finalize_channel_state(channel_id)
        self._cleanup_channel_snapshots(channel_id)
        del self.channels[channel_id]
        del self._states[channel_id]
        self._rebuild_shared_pipeline()
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
        logger.info(
            "Channel %d configured with ROI + %d lines",
            channel_id,
            len(entry_exit_lines),
        )

    def set_channel_phase(self, channel_id: int, phase: ChannelPhase) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        state = self._states[channel_id]
        old_phase = state.phase

        if phase == ChannelPhase.ANALYTICS:
            self._finalize_channel_state(channel_id)
            state.phase = ChannelPhase.ANALYTICS
            self._rebuild_shared_pipeline()
        elif phase == ChannelPhase.REVIEW:
            # Persist last frame before finalizing (for Phase 3 replay)
            self._persist_last_frame(channel_id)
            self._finalize_channel_state(channel_id)
            state.phase = ChannelPhase.REVIEW
            self._rebuild_shared_pipeline()
        else:
            state.phase = phase

        logger.info("Channel %d: %s -> %s", channel_id, old_phase.value, phase.value)

        if old_phase != phase:
            self._safe_callback(self._phase_callback, channel_id, phase, old_phase)

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

    def register_frame_callback(self, callback: Callable[[FrameResult], None]) -> None:
        self._frame_callback = callback

    def register_alert_callback(self, callback: Callable[[dict], None]) -> None:
        self._alert_callback = callback

    def register_track_ended_callback(self, callback: Callable[[dict], None]) -> None:
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

        # Fall back to disk (finalized tracks) — check per-channel dirs
        snapshots_root = Path("snapshots")
        for channel_dir in snapshots_root.iterdir() if snapshots_root.exists() else []:
            if channel_dir.is_dir():
                snapshot_path = channel_dir / f"{track_id}.jpg"
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

    def _create_shared_pipeline(self) -> None:
        """Create the shared pipeline infrastructure (does NOT start it).

        Sources must be added via _add_channel_to_pipeline() before calling
        _start_shared_pipeline(). nvurisrcbin elements added after start()
        won't transition to PLAYING state.
        """
        from backend.pipeline.deepstream.batch_router import BatchMetadataRouter

        pipeline = Pipeline("shared")

        pipeline.add(
            "nvstreammux",
            "mux",
            {
                "batch-size": 8,
                "batched-push-timeout": 33000,
                "width": 1920,
                "height": 1080,
            },
        )

        pipeline.add("nvinfer", "infer", {"config-file-path": PGIE_CONFIG})
        pipeline.link("mux", "infer")

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
        pipeline.link("infer", "tracker")

        self._batch_router = BatchMetadataRouter({})
        pipeline.attach("tracker", Probe("batch_router", self._batch_router))

        pipeline.add("nvstreamdemux", "demux", {})
        pipeline.link("tracker", "demux")

        self._shared_pipeline = pipeline
        self._source_index = 0
        logger.info("Shared pipeline created (not yet started)")

    def _start_shared_pipeline(self) -> None:
        """Start the shared pipeline (all sources must be added first)."""
        if self._shared_pipeline is None:
            return
        self._shared_pipeline.start(on_message=self._on_pipeline_message)
        logger.info("Shared pipeline started")

    def _rebuild_shared_pipeline(self) -> None:
        """Tear down and rebuild the shared pipeline with all current channels.

        nvurisrcbin can't be added to a running pipeline (won't sync to
        PLAYING state), so we rebuild from scratch whenever the channel
        set changes. Old pipeline is stopped in a background thread.
        """
        # Tear down old pipeline — set sources to non-loop so they drain,
        # then defer stop in background thread after EOS.
        # Direct stop() causes C++ terminate when asyncio loop is running.
        if self._shared_pipeline is not None:
            old_pipeline = self._shared_pipeline
            self._shared_pipeline = None
            self._batch_router = None

            # Drain all sources on old pipeline
            for state in self._states.values():
                if state._src_name:
                    try:
                        old_pipeline[state._src_name].set({"file-loop": False})
                    except Exception:
                        pass

            import threading as _threading

            def _deferred_stop():
                try:
                    old_pipeline.wait()
                    old_pipeline.stop()
                except Exception:
                    pass

            _threading.Thread(target=_deferred_stop, daemon=True).start()

        # Disable MJPEG callbacks on all channels (prevent stale frames)
        for state in self._states.values():
            if state.mjpeg_extractor:
                state.mjpeg_extractor._callback = None
                state.mjpeg_extractor = None
            state.pipeline = None
            state.reporter = None
            state.best_photo = None
            state._source_idx = None
            state._src_name = None

        # Rebuild with all active channels
        active = [
            cid
            for cid, s in self._states.items()
            if s.phase in (ChannelPhase.SETUP, ChannelPhase.ANALYTICS)
        ]
        if not active:
            return

        self._create_shared_pipeline()
        for channel_id in active:
            state = self._states[channel_id]
            if state.phase == ChannelPhase.ANALYTICS:
                self._add_analytics_source(channel_id)
            else:
                self._add_channel_to_pipeline(channel_id)
        self._start_shared_pipeline()
        logger.info("Shared pipeline rebuilt with %d channels", len(active))

    def _add_channel_to_pipeline(
        self,
        channel_id: int,
        *,
        file_loop: bool = True,
        roi_polygon=None,
        lines=None,
        alert_callback=None,
        track_ended_callback=None,
        eos_callback=None,
    ) -> None:
        """Add a source and per-channel output branch to the shared pipeline.

        Args:
            file_loop: True for setup (loops), False for analytics (single play).
            roi_polygon: ROI polygon for analytics reporter.
            lines: LineSeg dict for analytics reporter.
            alert_callback: Transit alert callback (analytics only).
            track_ended_callback: Track ended callback (analytics only).
            eos_callback: Called when this source stops producing frames.
        """
        state = self._states[channel_id]
        source = state.source
        # YouTube HLS URLs are passed directly; file paths need file:// prefix
        if source.startswith(("http://", "https://")):
            source_uri = source
        else:
            source_uri = f"file://{os.path.abspath(source)}"
        pipeline = self._shared_pipeline

        # Assign monotonic source index (matches mux pad / frame_meta.source_id)
        src_idx = self._source_index
        self._source_index += 1
        state._source_idx = src_idx

        # Per-channel reporter — only track best photos during analytics
        best_photo = BestPhotoTracker() if roi_polygon is not None else None
        state.best_photo = best_photo
        snapshot_dir = os.path.join("snapshots", str(channel_id))
        reporter = TrackingReporter(
            self._labels,
            roi_polygon=roi_polygon,
            lines=lines,
            best_photo=best_photo,
            channel_id=channel_id,
            alert_callback=alert_callback,
            track_ended_callback=track_ended_callback,
            snapshot_dir=snapshot_dir,
        )
        state.reporter = reporter
        self._batch_router.add_reporter(src_idx, reporter)

        if eos_callback is not None:
            self._batch_router.watch_for_eos(src_idx, eos_callback)

        # Frame callback bridged to asyncio
        def on_frame(result):
            self._safe_callback(self._frame_callback, result)

        mjpeg_extractor = MjpegExtractor(channel_id, on_frame, reporter)
        state.mjpeg_extractor = mjpeg_extractor

        # Create demux output branch BEFORE source — nvstreamdemux can't
        # allocate request pads while data is already flowing through it.
        #
        # Single chain (tee stalls with pyservicemaker's BufferOperator):
        # demux → queue → conv_pre(RGBA) → caps_pre(RGBA) [FrameExtractor: clean crops]
        #       → OSD → conv_post(RGB) → caps_post(RGB) [MjpegExtractor: annotated MJPEG]
        #       → fakesink
        q_name = f"q_{src_idx}"
        osd_name = f"osd_{src_idx}"
        conv_name = f"conv_{src_idx}"
        caps_name = f"caps_{src_idx}"
        sink_name = f"sink_{src_idx}"
        # Add all elements first, then link (pyservicemaker requires
        # elements to exist before linking)
        pipeline.add("queue", q_name, {"leaky": 2, "max-size-buffers": 5})
        pipeline.add("nvdsosd", osd_name, {"gpu-id": 0, "display-text": False})
        pipeline.add("nvvideoconvert", conv_name, {})
        pipeline.add(
            "capsfilter",
            caps_name,
            {
                "caps": "video/x-raw(memory:NVMM),format=RGB",
            },
        )
        pipeline.add("fakesink", sink_name, {"sync": True})

        pipeline.link(("demux", q_name), ("src_%u", ""))
        pipeline.link(q_name, osd_name)

        # Post-OSD → RGB for annotated MJPEG
        pipeline.link(osd_name, conv_name, caps_name, sink_name)
        pipeline.attach(caps_name, Probe(f"mjpeg_{src_idx}", mjpeg_extractor))

        # Add source to mux AFTER demux branch is ready
        src_name = f"src_{src_idx}"
        state._src_name = src_name
        src_props = {"uri": source_uri}
        # file-loop only applies to file sources; YouTube HLS streams don't loop
        if not source.startswith(("http://", "https://")):
            src_props["file-loop"] = file_loop
        pipeline.add("nvurisrcbin", src_name, src_props)
        pipeline.link((src_name, "mux"), ("", "sink_%u"))

        state.pipeline = pipeline
        logger.info(
            "Channel %d added to shared pipeline (src_idx=%d, loop=%s)",
            channel_id,
            src_idx,
            file_loop,
        )

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

    def _finalize_channel_state(self, channel_id: int) -> None:
        """Finalize reporter state: save tracks, photos, disable MJPEG."""
        state = self._states.get(channel_id)
        if state is None:
            return

        if state.reporter:
            for tid, sstate in list(state.reporter.stitcher.lost_tracks.items()):
                state.reporter._finalize_lost_track(tid, sstate)
            state.reporter.stitcher.lost_tracks.clear()

            if state.best_photo:
                for tid in list(state.reporter.active_tracks):
                    seq_tid = state.reporter._seq_id(tid)
                    state.best_photo.save(seq_tid, state.reporter.snapshot_dir)

        if state.mjpeg_extractor:
            state.mjpeg_extractor._callback = None
            state.mjpeg_extractor = None

    def _cleanup_channel_snapshots(self, channel_id: int) -> None:
        """Remove snapshot directory for a channel."""
        import shutil
        from pathlib import Path

        snapshot_dir = Path("snapshots") / str(channel_id)
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            logger.info("Cleaned up snapshots for channel %d", channel_id)

    def _soft_remove_source(self, channel_id: int) -> None:
        """Soft-remove a channel's source from the shared pipeline.

        Since pyservicemaker has no remove/unlink API, we:
        1. Finalize reporter state (save tracks/photos)
        2. Remove reporter from batch router (metadata ignored)
        3. Set file-loop=False on nvurisrcbin (source drains to EOS)
        4. Clear state references

        GStreamer elements stay in the pipeline but become dormant after EOS.
        """
        state = self._states.get(channel_id)
        if state is None or state.pipeline is None:
            return

        self._finalize_channel_state(channel_id)

        # Remove from batch router
        if self._batch_router is not None and state._source_idx is not None:
            self._batch_router.remove_reporter(state._source_idx)
            self._batch_router.unwatch_eos(state._source_idx)

        # Drain source (set file-loop=False so it reaches EOS)
        if state._src_name and self._shared_pipeline is not None:
            try:
                self._shared_pipeline[state._src_name].set({"file-loop": False})
            except Exception:
                pass

        state.pipeline = None
        state.reporter = None
        state.best_photo = None
        state._source_idx = None
        state._src_name = None
        logger.debug("Soft-removed source for channel %d", channel_id)

    def _add_analytics_source(self, channel_id: int) -> None:
        """Add a non-looping analytics source to the shared pipeline."""
        state = self._states[channel_id]

        from backend.pipeline.direction import LineSeg

        lines = {}
        if state.entry_exit_lines:
            for arm_id, line_data in state.entry_exit_lines.items():
                lines[arm_id] = LineSeg(
                    label=line_data["label"],
                    start=tuple(line_data["start"]),
                    end=tuple(line_data["end"]),
                )

        def on_alert(alert):
            self._safe_callback(self._alert_callback, alert)

        def on_track_ended(track):
            self._safe_callback(self._track_ended_callback, track)

        def on_eos(source_id):
            self._safe_callback(self._on_source_eos, channel_id)

        self._add_channel_to_pipeline(
            channel_id,
            file_loop=False,
            roi_polygon=state.roi_polygon,
            lines=lines if lines else None,
            alert_callback=on_alert,
            track_ended_callback=on_track_ended,
            eos_callback=on_eos,
        )

    def _on_source_eos(self, channel_id: int) -> None:
        """Handle per-source EOS — auto-transition to Review."""
        if channel_id not in self._states:
            return
        state = self._states[channel_id]
        if state.phase != ChannelPhase.ANALYTICS:
            return
        previous = state.phase
        # Persist last frame before finalizing (for Phase 3 replay)
        self._persist_last_frame(channel_id)
        self._finalize_channel_state(channel_id)
        state.phase = ChannelPhase.REVIEW
        self._rebuild_shared_pipeline()
        logger.info("Channel %d auto-transitioned to REVIEW (source EOS)", channel_id)
        self._safe_callback(
            self._phase_callback, channel_id, ChannelPhase.REVIEW, previous
        )

    def _on_pipeline_message(self, message) -> None:
        """Handle pipeline-level messages (EOS fallback + error detection)."""
        msg_str = str(message).lower()
        if "eos" in msg_str:
            logger.info("Pipeline-level EOS received")
            if self._batch_router is not None:
                self._safe_callback(lambda: self._batch_router.fire_all_pending_eos())
        elif "error" in msg_str:
            # GStreamer error — check if it's from a YouTube source
            logger.warning("Pipeline error: %s", msg_str[:200])
            self._safe_callback(self._on_pipeline_error, msg_str)

    def _on_pipeline_error(self, error_msg: str) -> None:
        """Handle GStreamer errors — trigger recovery for YouTube sources."""
        # Find YouTube channels in ANALYTICS that might need recovery
        for channel_id, state in self._states.items():
            if (
                state.source_type == "youtube_live"
                and state.phase == ChannelPhase.ANALYTICS
                and channel_id not in self._recovering
            ):
                self._trigger_recovery(channel_id, error_msg)
                break  # Handle one channel at a time

    def _trigger_recovery(self, channel_id: int, reason: str) -> None:
        """Start async recovery for a YouTube Live channel."""
        if channel_id in self._recovering:
            return
        self._recovering.add(channel_id)
        logger.info(
            "Channel %d: starting stream recovery (reason: %s)",
            channel_id,
            reason[:100],
        )
        # Notify frontend
        self._safe_callback(
            self._ws_enqueue,
            {
                "type": "pipeline_event",
                "event": "stream_recovering",
                "channel": channel_id,
                "detail": "Stream interrupted, attempting recovery",
            },
        )
        # Run recovery in background
        import threading

        threading.Thread(
            target=self._run_recovery, args=(channel_id,), daemon=True
        ).start()

    def _run_recovery(self, channel_id: int) -> None:
        """Recovery flow: retry → liveness → re-extract → give up.

        Runs in a background thread. On success, triggers pipeline rebuild
        on the asyncio event loop.
        """
        import asyncio as _asyncio

        from backend.pipeline.source_resolver import (
            check_stream_liveness,
            re_extract_url,
        )
        from backend.pipeline.stream_recovery import RETRY_DELAYS

        state = self._states.get(channel_id)
        if state is None or state.original_url is None:
            self._recovering.discard(channel_id)
            return

        original_url = state.original_url

        # Step 1: Retry existing URL (backoff)
        for i, delay in enumerate(RETRY_DELAYS):
            logger.info(
                "Channel %d: retry %d/%d in %.0fs",
                channel_id,
                i + 1,
                len(RETRY_DELAYS),
                delay,
            )
            time.sleep(delay)
            # If channel was removed during recovery, bail
            if channel_id not in self._states:
                self._recovering.discard(channel_id)
                return

        # Step 2: Check liveness
        loop = _asyncio.new_event_loop()
        try:
            is_live = loop.run_until_complete(check_stream_liveness(original_url))
        finally:
            loop.close()

        if not is_live:
            # Stream ended — auto-transition to Review
            logger.info("Channel %d: stream ended, transitioning to Review", channel_id)
            self._recovering.discard(channel_id)
            self._safe_callback(self._on_source_eos, channel_id)
            return

        # Step 3: Re-extract fresh HLS URL
        logger.info("Channel %d: stream still live, re-extracting URL", channel_id)
        loop = _asyncio.new_event_loop()
        try:
            new_url = loop.run_until_complete(re_extract_url(original_url))
        except (ValueError, TimeoutError) as exc:
            logger.warning("Channel %d: re-extraction failed: %s", channel_id, exc)
            self._recovering.discard(channel_id)
            self._safe_callback(self._on_recovery_failed, channel_id, str(exc))
            return
        finally:
            loop.close()

        # Step 4: Check circuit breaker before rebuilding
        self._circuit_breaker.record_rebuild(channel_id)
        if self._circuit_breaker.is_tripped(channel_id):
            logger.warning("Channel %d: circuit breaker tripped, ejecting", channel_id)
            self._recovering.discard(channel_id)
            self._safe_callback(self._eject_channel, channel_id)
            return

        # Step 5: Swap source URL and rebuild
        logger.info("Channel %d: reconnecting with fresh URL", channel_id)
        state = self._states.get(channel_id)
        if state is not None:
            state.source = new_url
            self.channels[channel_id] = new_url
        self._recovering.discard(channel_id)
        self._safe_callback(self._rebuild_shared_pipeline)
        self._safe_callback(
            self._ws_enqueue,
            {
                "type": "pipeline_event",
                "event": "stream_recovered",
                "channel": channel_id,
                "detail": "Stream reconnected",
            },
        )

    def _on_recovery_failed(self, channel_id: int, error: str) -> None:
        """Notify operator that stream recovery failed."""
        self._safe_callback(
            self._ws_enqueue,
            {
                "type": "pipeline_event",
                "event": "stream_recovery_failed",
                "channel": channel_id,
                "detail": f"Recovery failed: {error}",
            },
        )

    def _eject_channel(self, channel_id: int) -> None:
        """Remove a channel due to circuit breaker trip."""
        state = self._states.get(channel_id)
        if state is None:
            return
        previous = state.phase
        self._finalize_channel_state(channel_id)
        state.phase = ChannelPhase.REVIEW
        self._rebuild_shared_pipeline()
        logger.info("Channel %d ejected (circuit breaker)", channel_id)
        self._safe_callback(
            self._phase_callback, channel_id, ChannelPhase.REVIEW, previous
        )
        self._safe_callback(
            self._ws_enqueue,
            {
                "type": "pipeline_event",
                "event": "channel_ejected",
                "channel": channel_id,
                "detail": "Channel ejected: too many recovery attempts",
            },
        )

    def _ws_enqueue(self, message: dict) -> None:
        """Enqueue a WS message (requires _ws_broadcaster set by API layer)."""
        if hasattr(self, "_ws_broadcaster") and self._ws_broadcaster is not None:
            self._ws_broadcaster.enqueue(message)

    def register_ws_broadcaster(self, broadcaster) -> None:
        """Register WS broadcaster for recovery notifications."""
        self._ws_broadcaster = broadcaster

    def _persist_last_frame(self, channel_id: int) -> None:
        """Save the last MJPEG frame to disk for Phase 3 replay."""
        state = self._states.get(channel_id)
        if state is None:
            return
        jpeg = None
        if state.mjpeg_extractor and state.mjpeg_extractor.last_frame:
            jpeg = state.mjpeg_extractor.last_frame
        if not isinstance(jpeg, bytes):
            return

        from pathlib import Path

        frame_dir = Path("snapshots") / str(channel_id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "last_frame.jpg"
        frame_path.write_bytes(jpeg)
        logger.info(
            "Channel %d: persisted last frame (%d bytes)", channel_id, len(jpeg)
        )
