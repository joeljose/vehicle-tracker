"""CustomPipeline adapter — implements PipelineBackend Protocol.

GPU-resident pipeline: NVDEC decode → CuPy preprocess → direct TRT
inference → CPU NMS → BoT-SORT tracking → phase-routed analytics.
"""

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable


from backend.pipeline.custom.decoder import NvDecoder
from backend.pipeline.custom.detector import TRTDetector
from backend.pipeline.stream_recovery import CircuitBreaker
from backend.pipeline.custom.engine_builder import ensure_engine
from backend.pipeline.custom.preprocess import GpuPreprocessor, nv12_to_bgr_gpu, nv12_to_rgb_gpu
from backend.pipeline.custom.renderer import FrameRenderer
from backend.pipeline.custom.tracker import TrackerWrapper
from backend.pipeline.direction import DirectionStateMachine, LineSeg
from backend.pipeline.idle import IdleOptimizer
from backend.pipeline.protocol import ChannelPhase, Detection, FrameResult
from backend.pipeline.roi import point_in_polygon
from backend.pipeline.shared import (
    check_crossings,
    cleanup_channel_snapshots,
    finalize_lost_track,
    get_snapshot,
    parse_lines,
    safe_callback,
)
from backend.pipeline.snapshot import BestPhotoTracker
from backend.pipeline.stitch import TrackStitcher
from backend.pipeline.trajectory import TrajectoryBuffer

logger = logging.getLogger(__name__)

_MAX_PER_FRAME_DATA = 300  # ring buffer size per track

# Uniform processing rate for all sources. Tracker / stitcher / idle-optimizer
# all assume 30 FPS internally (see TrackerWrapper.frame_rate, track_buffer=150,
# IdleOptimizer thresholds), so capping decode→infer→track to 30 FPS makes their
# implicit time math actually true on 60 FPS sources, halves GPU/CPU work, and
# leaves the source clip untouched for replay (ffmpeg slices the original file).
PROCESS_FPS = 30.0
PROCESS_INTERVAL_MS = 1000.0 / PROCESS_FPS

# Drop-late-frame threshold: if the display thread is more than this far behind
# real-time, drop the head-of-queue frame and re-anchor. Set generously so brief
# stalls don't cause visible drops, but small enough that we recover within a
# second or two from a long stall.
DISPLAY_LATE_DROP_MS = 100.0


@dataclass
class _ChannelState:
    """Per-channel mutable state."""

    source: str
    decoder: NvDecoder | None = None
    tracker: TrackerWrapper | None = None
    phase: ChannelPhase = ChannelPhase.SETUP
    source_type: str = "file"
    original_url: str | None = None
    roi_polygon: list[tuple[float, float]] | None = None
    entry_exit_lines: dict = field(default_factory=dict)
    # Analytics state (created on Setup→Analytics)
    active_tracks: dict[int, dict] = field(default_factory=dict)
    stitcher: TrackStitcher | None = None
    best_photo: BestPhotoTracker | None = None
    idle_optimizer: IdleOptimizer | None = None
    roi_centroid: tuple[float, float] | None = None
    # Counters
    frame_count: int = 0
    infer_count: int = 0
    inference_interval: int = 1  # 1 = every frame, 15 = idle
    # Current PTS from decoder (ms) — used for per_frame_data timestamps
    current_pts_ms: int = 0
    # 30 FPS cap: PTS of last frame that passed the rate cap (-1 = none yet).
    last_processed_pts_ms: int = -1
    # Last computed tracks list (reused across idle-skipped frames so the
    # display still updates at full source FPS while inference is sparse).
    last_tracks: list = field(default_factory=list)
    # Display FIFO — processing pushes rendered frames, display pops at PTS rate
    display_queue: Queue = field(default_factory=lambda: Queue(maxsize=5))
    display_wall_start: float = 0.0  # wall-clock when display started
    display_pts_start: int = -1      # PTS of first displayed frame (ms)
    # Seek epoch: bumped by NvDecoder._restart on Setup-loop seek; the display
    # thread compares against its own `display_seek_epoch` and re-anchors.
    seek_epoch: int = 0
    display_seek_epoch: int = 0
    # Last clean BGR frame, kept for Phase-3 last-frame persistence. Encoded to
    # JPEG only on phase transition, not every frame (B3).
    last_frame_bgr: object | None = None


class CustomPipeline:
    """GPU-resident custom pipeline implementing PipelineBackend Protocol."""

    def __init__(self):
        self.channels: dict[int, str] = {}
        self._states: dict[int, _ChannelState] = {}
        self._detector: TRTDetector | None = None
        self._preprocess = GpuPreprocessor()
        self._renderer = FrameRenderer()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Callbacks
        self._frame_callback: Callable | None = None
        self._alert_callback: Callable | None = None
        self._track_ended_callback: Callable | None = None
        self._phase_callback: Callable | None = None
        # Stats tracking (per-channel, for stats_update broadcasts)
        self._stats_last_time: dict[int, float] = {}
        self._stats_frame_count: dict[int, int] = {}
        self._stats_infer_sum: dict[int, float] = {}
        # Thread-safe transition queue
        self._transition_queue: Queue = Queue()
        # Stream recovery (YouTube Live)
        self._circuit_breaker = CircuitBreaker()
        self._recovering: set[int] = set()
        self._ws_broadcaster = None

    # -- Lifecycle --

    def start(self) -> None:
        """Boot pipeline: load TRT engine, start pipeline thread."""
        if self._running:
            return

        # Capture asyncio loop for callbacks
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Ensure TRT engine exists
        engine_path = ensure_engine()
        self._detector = TRTDetector(engine_path)

        # Warm up the GPU paths so the first real frame doesn't pay
        # one-time costs (TRT lazy module loads, RawKernel NVRTC compile,
        # cupy memory pool init). Without this the first frame takes
        # ~150 ms vs ~17 ms steady-state — visible as a startup hitch.
        self._warmup()

        self._running = True
        self._thread = threading.Thread(
            target=self._pipeline_loop, name="custom-pipeline", daemon=True,
        )
        self._thread.start()
        self._display_thread = threading.Thread(
            target=self._display_loop, name="custom-display", daemon=True,
        )
        self._display_thread.start()
        logger.info("CustomPipeline started")

    def _warmup(self) -> None:
        """Pay one-time GPU-init costs at startup (B7).

        Runs one dummy preprocess + inference + NV12 colour conversion to
        force CuPy RawKernel JIT, TensorRT lazy-init, and CUDA module loads.
        """
        try:
            import cupy as cp

            t0 = time.monotonic()
            dummy_h, dummy_w = 1080, 1920
            dummy_nv12 = cp.zeros(
                (dummy_h * 3 // 2, dummy_w), dtype=cp.uint8,
            )
            # Trigger NVRTC compile of both colour-conversion kernels.
            nv12_to_rgb_gpu(dummy_nv12, dummy_h, dummy_w)
            nv12_to_bgr_gpu(dummy_nv12, dummy_h, dummy_w)
            # Trigger preprocess + TRT context first-call cost.
            input_tensor, scale_info = self._preprocess(
                dummy_nv12, dummy_h, dummy_w,
            )
            self._detector.detect(input_tensor, scale_info)
            cp.cuda.Device(0).synchronize()
            logger.info(
                "CustomPipeline warmup completed in %.0f ms",
                (time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            # Warmup failures are non-fatal — just means the first real
            # frame will pay the startup cost as before.
            logger.warning("Warmup failed (non-fatal): %s", exc)

    def stop(self) -> None:
        """Stop pipeline, release all resources."""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Finalize all channels
        for ch_id in list(self._states):
            self._finalize_channel_state(ch_id)
            state = self._states[ch_id]
            if state.decoder:
                state.decoder.release()
                state.decoder = None

        self._states.clear()
        self.channels.clear()
        self._detector = None
        logger.info("CustomPipeline stopped")

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
            source=source,
            source_type=source_type,
            original_url=original_url,
        )
        self._states[channel_id] = state

        # Queue decoder creation for pipeline thread
        self._transition_queue.put(("add", channel_id))
        logger.info("Channel %d queued for add: %s", channel_id, source)

    def remove_channel(self, channel_id: int) -> None:
        if channel_id not in self.channels:
            return
        self._transition_queue.put(("remove", channel_id))

    def configure_channel(
        self,
        channel_id: int,
        roi_polygon: list[tuple[float, float]],
        entry_exit_lines: dict,
    ) -> None:
        if channel_id not in self._states:
            raise KeyError(f"Channel {channel_id} not found")
        state = self._states[channel_id]
        state.roi_polygon = roi_polygon
        state.entry_exit_lines = entry_exit_lines

    def set_channel_phase(self, channel_id: int, phase: ChannelPhase) -> None:
        if channel_id not in self.channels:
            raise KeyError(f"Channel {channel_id} not found")
        self._transition_queue.put(("phase", channel_id, phase))

    def get_channel_phase(self, channel_id: int) -> ChannelPhase:
        if channel_id not in self._states:
            raise KeyError(f"Channel {channel_id} not found")
        return self._states[channel_id].phase

    def get_channel_config(self, channel_id: int) -> dict:
        if channel_id not in self._states:
            raise KeyError(f"Channel {channel_id} not found")
        state = self._states[channel_id]
        return {
            "roi_polygon": [list(p) for p in state.roi_polygon] if state.roi_polygon else [],
            "entry_exit_lines": state.entry_exit_lines or {},
        }

    # -- Configuration --

    def set_confidence_threshold(self, threshold: float) -> None:
        if self._detector:
            self._detector.set_confidence_threshold(threshold)

    # -- Callbacks --

    def register_frame_callback(self, callback: Callable[[FrameResult], None]) -> None:
        self._frame_callback = callback

    def register_alert_callback(self, callback: Callable[[dict], None]) -> None:
        self._alert_callback = callback

    def register_track_ended_callback(self, callback: Callable[[dict], None]) -> None:
        self._track_ended_callback = callback

    def register_phase_callback(self, callback: Callable) -> None:
        self._phase_callback = callback

    def register_ws_broadcaster(self, broadcaster) -> None:
        """Register WS broadcaster for stream recovery notifications."""
        self._ws_broadcaster = broadcaster

    def _ws_enqueue(self, message: dict) -> None:
        if self._ws_broadcaster is not None:
            self._ws_broadcaster.enqueue(message)

    # -- Data access --

    def get_snapshot(self, track_id: int) -> bytes | None:
        return get_snapshot(self._states, track_id)

    # -- Pipeline loop (runs on background thread) --

    def _pipeline_loop(self):
        """Main pipeline loop — decode, (rate-cap), infer, track, analytics, render.

        Per iteration: read one frame from each non-Review channel, gate it
        through the 30 FPS rate cap, then process. The pipeline thread blocks
        on `display_queue.put` (B17 backpressure) so the decoder is naturally
        rate-limited to display rate when the source is faster than 30 FPS or
        when display is gated by the PTS-pacer.
        """
        import cupy as cp

        while self._running:
            self._drain_transitions()

            # Find channels with a frame ready to process this iteration
            active = {}
            for ch_id, state in list(self._states.items()):
                if state.decoder is None:
                    continue
                if state.phase == ChannelPhase.REVIEW:
                    continue

                # Re-sync rate-cap baseline if the decoder has seeked. We pick
                # this up here (in addition to the explicit reset on phase
                # transition) so that Setup-loop seeks don't double-skip the
                # first post-loop frame.
                dec_epoch = state.decoder.seek_epoch
                if dec_epoch != state.seek_epoch:
                    state.seek_epoch = dec_epoch
                    state.last_processed_pts_ms = -1

                nv12, pts = state.decoder.read()
                if nv12 is None:
                    self._handle_eos(ch_id)
                    continue

                state.current_pts_ms = pts
                state.frame_count += 1

                # 30 FPS rate cap (post-decode). NVDEC still decodes every
                # frame because H.264 is inter-coded, but we drop alternate
                # frames before the expensive preprocess/infer/render path.
                last_pts = state.last_processed_pts_ms
                if last_pts >= 0 and (pts - last_pts) < PROCESS_INTERVAL_MS - 1:
                    continue
                state.last_processed_pts_ms = pts

                # Idle inference skipping: every Nth processed frame runs the
                # detector; in between, we reuse `state.last_tracks` so the
                # display still gets a frame at the (capped) source rate.
                run_inference = True
                if state.inference_interval > 1:
                    if state.infer_count % state.inference_interval != 0:
                        run_inference = False

                active[ch_id] = (nv12, pts, state, run_inference)

            if not active:
                time.sleep(0.001)
                continue

            # Process each channel that passed the rate cap.
            for ch_id, (nv12, pts, state, run_inference) in active.items():
                t0 = time.monotonic()

                if run_inference:
                    # GPU preprocess
                    input_tensor, scale_info = self._preprocess(
                        nv12, state.decoder.height, state.decoder.width,
                    )

                    # TRT inference + NMS
                    detections = self._detector.detect(input_tensor, scale_info)

                    # Single NV12→BGR conversion shared between ReID extractor
                    # and renderer (was two separate conversions; B6).
                    bgr_frame = None
                    if state.tracker._with_reid and len(detections) > 0:
                        bgr_gpu = nv12_to_bgr_gpu(
                            nv12, state.decoder.height, state.decoder.width,
                        )
                        bgr_frame = cp.asnumpy(bgr_gpu)

                    tracks = state.tracker.update(detections, frame=bgr_frame)
                    state.last_tracks = tracks

                    # Phase routing: always track, conditionally analyze.
                    if state.phase == ChannelPhase.ANALYTICS:
                        self._process_analytics(ch_id, state, tracks)

                    # Idle optimization update — only when we actually inferred.
                    if state.idle_optimizer:
                        transition = state.idle_optimizer.update(
                            num_detections=len(detections),
                            num_active_tracks=len(state.active_tracks),
                        )
                        if transition:
                            state.inference_interval = (
                                state.idle_optimizer.recommended_interval
                                if transition == "idle" else 1
                            )
                            mode = "IDLE" if transition == "idle" else "ACTIVE"
                            logger.info(
                                "Channel %d: %s mode (interval=%d)",
                                ch_id, mode, state.inference_interval,
                            )
                else:
                    # Idle skip: reuse last computed tracks; bgr_frame stays
                    # None (renderer derives BGR from NV12 itself).
                    tracks = state.last_tracks
                    bgr_frame = None

                infer_ms = (time.monotonic() - t0) * 1000
                state.infer_count += 1

                # Render + display. Blocking put provides backpressure (B17):
                # if the display queue is full, the pipeline thread waits
                # rather than draining the decoder ahead of display.
                if self._frame_callback:
                    self._store_display_frame(
                        ch_id, state, nv12, bgr_frame, tracks, infer_ms, pts,
                    )

                # Best-photo crop extraction (uses RGB; only when there's
                # actually pending work and we ran inference this frame).
                if (
                    run_inference
                    and state.best_photo
                    and state.best_photo.pending_crops
                ):
                    rgb_gpu = nv12_to_rgb_gpu(
                        nv12, state.decoder.height, state.decoder.width,
                    )
                    frame_cpu = cp.asnumpy(rgb_gpu)
                    state.best_photo.extract_crops(frame_cpu)

    # -- Transition handling --

    def _drain_transitions(self):
        """Drain all pending transitions from the queue."""
        while True:
            try:
                item = self._transition_queue.get_nowait()
            except Empty:
                break

            action = item[0]
            if action == "add":
                self._do_add_channel(item[1])
            elif action == "remove":
                self._do_remove_channel(item[1])
            elif action == "phase":
                self._do_phase_transition(item[1], item[2])
            elif action == "reconnect":
                self._do_reconnect_channel(item[1], item[2])

    def _do_add_channel(self, channel_id: int):
        state = self._states.get(channel_id)
        if state is None:
            return
        is_live = state.source_type == "youtube_live"
        state.decoder = NvDecoder(
            state.source, loop=not is_live, channel_id=channel_id,
        )
        state.tracker = TrackerWrapper()
        state.stitcher = TrackStitcher()
        state.idle_optimizer = IdleOptimizer()
        logger.info("Channel %d: decoder + tracker created", channel_id)

    def _do_remove_channel(self, channel_id: int):
        state = self._states.get(channel_id)
        if state is None:
            return
        self._finalize_channel_state(channel_id)
        if state.decoder:
            state.decoder.release()
        self._states.pop(channel_id, None)
        self.channels.pop(channel_id, None)
        self._cleanup_channel_snapshots(channel_id)
        logger.info("Channel %d: removed", channel_id)

    def _do_phase_transition(self, channel_id: int, new_phase: ChannelPhase):
        state = self._states.get(channel_id)
        if state is None:
            return
        old_phase = state.phase

        if new_phase == ChannelPhase.ANALYTICS:
            # Finalize setup state
            self._finalize_channel_state(channel_id)

            # Recreate decoder without looping
            if state.decoder:
                state.decoder.release()
            state.decoder = NvDecoder(
                state.source, loop=False, channel_id=channel_id,
            )

            # Fresh tracker
            state.tracker = TrackerWrapper()
            state.active_tracks.clear()
            state.stitcher = TrackStitcher()
            state.idle_optimizer = IdleOptimizer()
            state.frame_count = 0
            state.infer_count = 0
            state.inference_interval = 1
            state.last_processed_pts_ms = -1  # reset 30 FPS rate-cap baseline
            state.last_tracks = []
            state.seek_epoch = 0
            state.display_seek_epoch = 0
            state.display_pts_start = -1  # reset display timeline

            # Best-photo tracker
            state.best_photo = BestPhotoTracker()

            # ROI centroid for direction inference
            if state.roi_polygon and len(state.roi_polygon) >= 3:
                n = len(state.roi_polygon)
                state.roi_centroid = (
                    sum(p[0] for p in state.roi_polygon) / n,
                    sum(p[1] for p in state.roi_polygon) / n,
                )
            else:
                state.roi_centroid = None

            state.phase = ChannelPhase.ANALYTICS
            logger.info("Channel %d: Setup → Analytics", channel_id)

        elif new_phase == ChannelPhase.REVIEW:
            self._persist_last_frame(channel_id)
            self._finalize_channel_state(channel_id)
            if state.decoder:
                state.decoder.release()
                state.decoder = None
            state.phase = ChannelPhase.REVIEW
            logger.info("Channel %d: Analytics → Review", channel_id)

        else:
            state.phase = new_phase

        if old_phase != new_phase:
            self._safe_callback(
                self._phase_callback, channel_id, new_phase, old_phase,
            )

    def _handle_eos(self, channel_id: int):
        """Handle end-of-stream for a channel.

        For YouTube Live in Analytics: trigger async recovery on the first EOS
        and then *swallow* subsequent EOS calls while recovery is in flight.
        Without this short-circuit, the pipeline thread loops back into
        `decoder.read()` (still returning None for an EOS HLS decoder), calls
        `_handle_eos` again, and the fall-through transitions the channel to
        Review while recovery is mid-retry — leaving us decoderless when the
        new HLS URL eventually arrives. (B1.)
        """
        state = self._states.get(channel_id)
        if state is None:
            return

        # Recovery already in flight for this channel — don't act on this EOS.
        if channel_id in self._recovering:
            return

        if (
            state.source_type == "youtube_live"
            and state.phase == ChannelPhase.ANALYTICS
        ):
            # YouTube: try recovery before giving up
            self._trigger_recovery(channel_id, "decode returned None (HLS)")
            return

        if state.phase == ChannelPhase.ANALYTICS:
            logger.info("Channel %d: EOS — transitioning to Review", channel_id)
            self._do_phase_transition(channel_id, ChannelPhase.REVIEW)

    # -- Stream recovery (YouTube Live) --

    def _trigger_recovery(self, channel_id: int, reason: str) -> None:
        """Start async recovery for a YouTube Live channel."""
        if channel_id in self._recovering:
            return
        self._recovering.add(channel_id)
        logger.info(
            "Channel %d: starting stream recovery (reason: %s)",
            channel_id, reason[:100],
        )
        self._ws_enqueue({
            "type": "pipeline_event",
            "event": "stream_recovering",
            "channel": channel_id,
            "detail": "Stream interrupted, attempting recovery",
        })
        threading.Thread(
            target=self._run_recovery, args=(channel_id,), daemon=True,
        ).start()

    def _run_recovery(self, channel_id: int) -> None:
        """Recovery flow: retry → liveness → re-extract → reconnect or eject.

        Runs in a background thread.
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

        # Step 1: Retry with backoff
        for i, delay in enumerate(RETRY_DELAYS):
            logger.info(
                "Channel %d: retry %d/%d in %.0fs",
                channel_id, i + 1, len(RETRY_DELAYS), delay,
            )
            time.sleep(delay)
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
            logger.info(
                "Channel %d: stream ended, transitioning to Review", channel_id,
            )
            self._recovering.discard(channel_id)
            self._safe_callback(
                self._do_phase_transition, channel_id, ChannelPhase.REVIEW,
            )
            return

        # Step 3: Re-extract fresh HLS URL
        logger.info("Channel %d: stream still live, re-extracting URL", channel_id)
        loop = _asyncio.new_event_loop()
        try:
            new_url = loop.run_until_complete(re_extract_url(original_url))
        except (ValueError, TimeoutError) as exc:
            logger.warning("Channel %d: re-extraction failed: %s", channel_id, exc)
            self._recovering.discard(channel_id)
            self._on_recovery_failed(channel_id, str(exc))
            return
        finally:
            loop.close()

        # Step 4: Circuit breaker
        self._circuit_breaker.record_rebuild(channel_id)
        if self._circuit_breaker.is_tripped(channel_id):
            logger.warning(
                "Channel %d: circuit breaker tripped, ejecting", channel_id,
            )
            self._recovering.discard(channel_id)
            self._safe_callback(self._eject_channel, channel_id)
            return

        # Step 5: Reconnect with fresh URL
        logger.info("Channel %d: reconnecting with fresh URL", channel_id)
        state = self._states.get(channel_id)
        if state is not None:
            state.source = new_url
            self.channels[channel_id] = new_url
        self._recovering.discard(channel_id)
        self._transition_queue.put(("reconnect", channel_id, new_url))
        self._ws_enqueue({
            "type": "pipeline_event",
            "event": "stream_recovered",
            "channel": channel_id,
            "detail": "Stream reconnected",
        })

    def _do_reconnect_channel(self, channel_id: int, new_url: str):
        """Reconnect a channel's decoder with a fresh HLS URL."""
        state = self._states.get(channel_id)
        if state is None:
            return
        if state.decoder:
            state.decoder.reconnect(new_url)
        state.source = new_url
        state.frame_count = 0
        logger.info("Channel %d: decoder reconnected", channel_id)

    def _on_recovery_failed(self, channel_id: int, error: str) -> None:
        """Notify that stream recovery failed."""
        self._ws_enqueue({
            "type": "pipeline_event",
            "event": "stream_recovery_failed",
            "channel": channel_id,
            "detail": f"Recovery failed: {error}",
        })

    def _eject_channel(self, channel_id: int) -> None:
        """Eject a channel after circuit breaker trips — transition to Review."""
        logger.warning("Channel %d: ejecting (circuit breaker)", channel_id)
        self._do_phase_transition(channel_id, ChannelPhase.REVIEW)
        self._ws_enqueue({
            "type": "pipeline_event",
            "event": "stream_recovery_failed",
            "channel": channel_id,
            "detail": "Circuit breaker tripped — too many reconnects",
        })

    # -- Analytics processing --

    def _process_analytics(
        self, channel_id: int, state: _ChannelState, tracks: list[dict],
    ):
        """Process tracks through direction/ROI/stitch/alert pipeline."""
        lines = parse_lines(state.entry_exit_lines)
        seen_track_ids = set()

        for track in tracks:
            tid = track["track_id"]
            cx, cy = track["centroid"]
            seen_track_ids.add(tid)

            # ROI check
            in_roi = (
                point_in_polygon((cx, cy), state.roi_polygon)
                if state.roi_polygon
                else True
            )

            if tid not in state.active_tracks:
                # New track — check stitcher for merge
                match = state.stitcher.find_match(
                    position=(cx, cy), frame_number=state.infer_count,
                )
                if match:
                    traj = match["trajectory"]
                    traj.append(cx, cy, state.infer_count)
                    dsm = match["dsm"]
                    # Inherit per_frame_data from the lost track
                    old_pfd = match.get("per_frame_data", [])
                    pfd = deque(old_pfd, maxlen=_MAX_PER_FRAME_DATA)
                    first_frame = match.get("first_frame", state.infer_count)
                    logger.info(
                        "Track #%d merged with lost #%d (%d pfd inherited)",
                        tid, match["lost_track_id"], len(old_pfd),
                    )
                else:
                    traj = TrajectoryBuffer()
                    traj.append(cx, cy, state.infer_count)
                    dsm = DirectionStateMachine()
                    pfd = deque(maxlen=_MAX_PER_FRAME_DATA)
                    first_frame = state.infer_count

                bbox = track["bbox"]
                pfd.append({
                    "frame": state.infer_count,
                    "bbox": list(bbox),
                    "centroid": [cx, cy],
                    "confidence": round(track["confidence"], 3),
                    "timestamp_ms": state.current_pts_ms,
                })
                state.active_tracks[tid] = {
                    "label": track["class_name"],
                    "first_frame": first_frame,
                    "last_frame": state.infer_count,
                    "trajectory": traj,
                    "roi_active": in_roi,
                    "dsm": dsm,
                    "per_frame_data": pfd,
                }
            else:
                ts = state.active_tracks[tid]
                prev_traj = ts["trajectory"].get_full()
                prev_cx, prev_cy = prev_traj[-1][0], prev_traj[-1][1]

                ts["last_frame"] = state.infer_count
                ts["trajectory"].append(cx, cy, state.infer_count)
                bbox = track["bbox"]
                ts["per_frame_data"].append({
                    "frame": state.infer_count,
                    "bbox": list(bbox),
                    "centroid": [cx, cy],
                    "confidence": round(track["confidence"], 3),
                    "timestamp_ms": state.current_pts_ms,
                })
                ts["roi_active"] = in_roi

                # Line crossing check
                if in_roi and lines:
                    self._check_crossings(
                        channel_id, state, tid, (prev_cx, prev_cy), (cx, cy), lines,
                    )

                # Stagnant check
                if in_roi:
                    traj_full = ts["trajectory"].get_full()
                    if ts["dsm"].check_stagnant(traj_full):
                        logger.warning("STAGNANT: Track #%d", tid)
                        if self._alert_callback:
                            self._safe_callback(
                                self._alert_callback,
                                {
                                    "type": "stagnant_alert",
                                    "track_id": tid,
                                    "label": ts["label"],
                                    "position": (cx, cy),
                                    "stationary_duration_frames": (
                                        state.infer_count - ts["dsm"].stopped_since_frame
                                    ),
                                    "first_seen_frame": ts["first_frame"],
                                    "last_seen_frame": state.infer_count,
                                    "channel": channel_id,
                                },
                            )

            # Best-photo scoring (ROI-active only)
            if in_roi and state.best_photo:
                bbox = track["bbox"]
                area = bbox[2] * bbox[3]
                state.best_photo.score(
                    track_id=tid,
                    area=area,
                    confidence=track["confidence"],
                    bbox=bbox,
                )

        # Expire stitcher tracks
        for exp_tid, exp_state in state.stitcher.expire(state.infer_count):
            self._finalize_lost_track(channel_id, state, exp_tid, exp_state, lines)

        # Detect lost tracks
        lost_ids = [
            tid for tid in list(state.active_tracks) if tid not in seen_track_ids
        ]
        for tid in lost_ids:
            ts = state.active_tracks.pop(tid)
            lifetime = ts["last_frame"] - ts["first_frame"] + 1

            if ts["roi_active"]:
                state.stitcher.on_track_lost(
                    track_id=tid,
                    trajectory=ts["trajectory"],
                    dsm=ts["dsm"],
                    frame_number=state.infer_count,
                )
                state.stitcher.lost_tracks[tid]["label"] = ts["label"]
                state.stitcher.lost_tracks[tid]["lifetime"] = lifetime
                state.stitcher.lost_tracks[tid]["first_frame"] = ts["first_frame"]
                state.stitcher.lost_tracks[tid]["per_frame_data"] = ts.get(
                    "per_frame_data", [],
                )
            else:
                if state.best_photo:
                    snapshot_dir = os.path.join("snapshots", str(channel_id))
                    state.best_photo.save(tid, snapshot_dir)
                lost = {
                    "track_id": tid,
                    "label": ts["label"],
                    "lifetime": lifetime,
                    "trajectory": ts["trajectory"].get_full(),
                    "roi_active": False,
                    "direction_state": ts["dsm"].state.value,
                }
                self._safe_callback(self._track_ended_callback, lost)

    def _check_crossings(
        self,
        channel_id: int,
        state: _ChannelState,
        track_id: int,
        prev: tuple[float, float],
        curr: tuple[float, float],
        lines: dict[str, LineSeg],
    ):
        check_crossings(
            track_id=track_id,
            prev=prev,
            curr=curr,
            lines=lines,
            active_tracks=state.active_tracks,
            roi_centroid=state.roi_centroid,
            frame_count=state.infer_count,
            channel_id=channel_id,
            alert_callback=lambda a: self._safe_callback(self._alert_callback, a),
        )

    def _finalize_lost_track(
        self,
        channel_id: int,
        state: _ChannelState,
        tid: int,
        stitcher_state: dict,
        lines: dict[str, LineSeg],
    ):
        finalize_lost_track(
            track_id=tid,
            stitcher_state=stitcher_state,
            lines=lines,
            roi_centroid=state.roi_centroid,
            frame_count=state.infer_count,
            channel_id=channel_id,
            best_photo=state.best_photo,
            snapshot_dir=os.path.join("snapshots", str(channel_id)),
            alert_callback=lambda a: self._safe_callback(self._alert_callback, a),
            track_ended_callback=lambda t: self._safe_callback(self._track_ended_callback, t),
        )

    # -- Frame rendering + display --

    def _store_display_frame(
        self,
        channel_id: int,
        state: _ChannelState,
        nv12,
        bgr_frame,
        tracks: list[dict],
        infer_ms: float,
        pts: int,
    ):
        """Render annotated frame and push to display FIFO with backpressure.

        Reuses the BGR ndarray produced for ReID when available (single
        NV12→BGR conversion per frame, B6). Caches the clean BGR for Phase-3
        replay so we don't encode a clean JPEG every frame (B3). Uses a
        blocking put so the pipeline thread is naturally rate-limited to the
        display rate when source > 30 FPS or display gates by PTS (B17).
        """
        if bgr_frame is None:
            bgr_frame = self._renderer.decode_nv12(
                nv12, state.decoder.height, state.decoder.width,
            )

        # Cache the latest clean BGR frame for Phase-3 last-frame persistence.
        # Encoding a JPEG happens lazily inside _persist_last_frame on phase
        # transition — not on every frame.
        state.last_frame_bgr = bgr_frame

        # Render annotated JPEG (bboxes only — ROI/lines are frontend overlays).
        annotated_jpeg = self._renderer.annotate_and_encode(bgr_frame, tracks)

        det_list = [
            Detection(
                track_id=t["track_id"],
                class_name=t["class_name"],
                bbox=t["bbox"],
                confidence=t["confidence"],
                centroid=t["centroid"],
            )
            for t in tracks
        ]

        result = FrameResult(
            channel_id=channel_id,
            frame_number=state.infer_count,
            timestamp_ms=pts,
            detections=det_list,
            annotated_jpeg=annotated_jpeg,
            inference_ms=infer_ms,
            phase=state.phase.value,
            idle_mode=state.idle_optimizer.is_idle if state.idle_optimizer else False,
        )
        # Bounded blocking put: backpressure for the pipeline thread. The
        # short timeout ensures we drop on permanent display stalls (e.g.
        # display thread wedged) rather than blocking forever.
        try:
            state.display_queue.put(result, block=True, timeout=1.0)
        except Exception:
            pass

    def _display_loop(self):
        """Emit frames to MJPEG clients at real-time PTS rate.

        Pops rendered frames from each channel's display FIFO and emits
        them when wall-clock time matches the frame's PTS. Two correctness
        properties beyond simple PTS gating:

        - **Seek-epoch resync (B4)**: when the decoder seeks (Setup loop),
          the pipeline-side `state.seek_epoch` increments. We compare to our
          own `state.display_seek_epoch`, drain the now-stale queue, and
          reset the wall/PTS baseline so the post-loop pass paces correctly.

        - **Drop-late-frame (B2)**: if wall-clock has fallen more than
          `DISPLAY_LATE_DROP_MS` behind the queued frame's PTS, drop the
          head and re-anchor. Without this, a transient stall (cold start,
          GC, etc.) would leave the pacer permanently behind real-time.
        """
        while self._running:
            emitted = False
            now = time.monotonic()

            for ch_id, state in list(self._states.items()):
                # On seek (decoder loop), drain stale frames and reset baseline.
                if state.display_seek_epoch != state.seek_epoch:
                    while True:
                        try:
                            state.display_queue.get_nowait()
                        except Empty:
                            break
                    state.display_pts_start = -1
                    state.display_seek_epoch = state.seek_epoch
                    continue  # re-evaluate next iteration

                if state.display_queue.empty():
                    continue

                # Peek at next frame's PTS
                try:
                    result = state.display_queue.queue[0]
                except (IndexError, AttributeError):
                    continue

                # Initialize PTS baseline on first frame after start/seek.
                if state.display_pts_start < 0:
                    state.display_pts_start = result.timestamp_ms
                    state.display_wall_start = now

                video_elapsed_ms = result.timestamp_ms - state.display_pts_start
                wall_elapsed_ms = (now - state.display_wall_start) * 1000

                # Drop-late-frame: if we're significantly behind real-time,
                # skip this frame and re-anchor on the next one. This keeps
                # cumulative drift bounded regardless of pipeline jitter.
                if wall_elapsed_ms - video_elapsed_ms > DISPLAY_LATE_DROP_MS:
                    try:
                        state.display_queue.get_nowait()
                    except Empty:
                        continue
                    state.display_pts_start = -1  # re-anchor on next frame
                    emitted = True
                    continue

                if wall_elapsed_ms >= video_elapsed_ms:
                    # Time to emit — pop from queue
                    try:
                        result = state.display_queue.get_nowait()
                    except Empty:
                        continue
                    self._safe_callback(self._frame_callback, result)
                    emitted = True

            if not emitted:
                time.sleep(0.005)  # 5ms — sub-frame precision

    # -- Helpers --

    def _finalize_channel_state(self, channel_id: int):
        """Finalize all pending tracks, save photos, disable callbacks."""
        state = self._states.get(channel_id)
        if state is None:
            return

        lines = parse_lines(state.entry_exit_lines)

        # Expire all stitcher tracks
        if state.stitcher:
            for tid, sstate in list(state.stitcher.lost_tracks.items()):
                self._finalize_lost_track(channel_id, state, tid, sstate, lines)
            state.stitcher.lost_tracks.clear()

        # Save photos for remaining active tracks
        if state.best_photo:
            snapshot_dir = os.path.join("snapshots", str(channel_id))
            for tid in list(state.active_tracks):
                state.best_photo.save(tid, snapshot_dir)

        state.active_tracks.clear()

    def _cleanup_channel_snapshots(self, channel_id: int):
        cleanup_channel_snapshots(channel_id)

    def _persist_last_frame(self, channel_id: int) -> None:
        """Save the last rendered frame to disk for Phase 3 frozen-frame replay.

        Encodes JPEG from the cached BGR ndarray. This is the single
        encode_clean call per channel lifetime — it used to fire on every
        frame just so the most recent one would be on disk (B3).
        """
        from pathlib import Path

        state = self._states.get(channel_id)
        if state is None or state.last_frame_bgr is None:
            return

        jpeg = self._renderer.encode_clean(state.last_frame_bgr)

        frame_dir = Path("snapshots") / str(channel_id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "last_frame.jpg"
        frame_path.write_bytes(jpeg)
        logger.info(
            "Channel %d: persisted last frame (%d bytes)", channel_id, len(jpeg),
        )

    def _safe_callback(self, callback, *args):
        safe_callback(callback, *args, loop=self._loop)
