import { useCallback, useEffect, useRef, useState } from "react";
import { getAlert, replayUrl, getReplayStatus, lastFrameUrl } from "../api/rest";
import LinesOverlay from "./LinesOverlay";

const TRANSIT_COLOR = "#60a5fa";
const STAGNANT_COLOR = "#fbbf24";
const ENTRY_COLOR = "#4ade80";
const LABEL_BG = "rgba(0,0,0,0.7)";

/**
 * ReplayView — replaces MJPEG video panel in Review phase.
 *
 * For file sources: plays extracted video clip with bbox overlay.
 * For YouTube Live sources: shows frozen last frame with animated bbox overlay.
 *
 * Props:
 *   alert - selected alert summary (from AlertFeed click)
 *   lines - entry/exit lines to overlay
 */
export default function ReplayView({ alert, lines }) {
  const [fullAlert, setFullAlert] = useState(null);
  const [replayStatus, setReplayStatus] = useState(null);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Fetch full alert data + replay status
  useEffect(() => {
    if (!alert) return;
    let cancelled = false;

    async function load() {
      try {
        const [full, status] = await Promise.all([
          getAlert(alert.alert_id),
          getReplayStatus(alert.alert_id).catch(() => ({ status: "not_started" })),
        ]);
        if (cancelled) return;
        setFullAlert(full);
        setReplayStatus(status.status);
        setError(null);
      } catch {
        if (!cancelled) setError("Failed to load replay");
      }
    }
    load();

    return () => { cancelled = true; };
  }, [alert?.alert_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Poll for clip extraction (file sources only)
  useEffect(() => {
    if (!alert || !fullAlert) return;
    // Don't poll for YouTube Live — there's no clip to extract
    if (replayStatus !== "extracting") return;
    let cancelled = false;
    const pollTimer = setInterval(async () => {
      try {
        const s = await getReplayStatus(alert.alert_id);
        if (!cancelled) setReplayStatus(s.status);
      } catch { /* ignore */ }
    }, 2000);
    return () => { cancelled = true; clearInterval(pollTimer); };
  }, [alert?.alert_id, replayStatus, fullAlert]);

  // Determine if this is a frozen-frame replay (YouTube Live)
  const isFrozenFrame = replayStatus === "not_started" && fullAlert != null;

  // Canvas overlay for bbox + trajectory (video clip replay)
  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !fullAlert) return;

    const ctx = canvas.getContext("2d");
    const container = containerRef.current;
    if (!container) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const perFrame = fullAlert.per_frame_data || [];
    if (perFrame.length === 0) return;

    // Map video playback time to original-video time.
    const currentMs = (video.currentTime || 0) * 1000;
    const fps = 30;
    const paddingMs = 1000;
    const clipStartMs = Math.max(0, (fullAlert.first_seen_frame / fps) * 1000 - paddingMs);
    const originalMs = currentMs + clipStartMs;

    const firstTs = perFrame[0].timestamp_ms;
    const lastTs = perFrame[perFrame.length - 1].timestamp_ms;
    if (originalMs < firstTs - 500 || originalMs > lastTs + 500) return;

    // Binary search for closest frame
    let lo = 0, hi = perFrame.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (perFrame[mid].timestamp_ms < originalMs) lo = mid + 1;
      else hi = mid;
    }
    const frame = perFrame[lo];
    if (!frame) return;

    const natW = video.videoWidth || 1920;
    const natH = video.videoHeight || 1080;
    drawBboxAndTrajectory(ctx, canvas, natW, natH, frame, fullAlert);
  }, [fullAlert]);

  // Frozen-frame overlay — animated bbox replay on static image
  const drawFrozenOverlay = useCallback((progressMs) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !fullAlert) return;

    const ctx = canvas.getContext("2d");
    const container = containerRef.current;
    if (!container) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const perFrame = fullAlert.per_frame_data || [];
    if (perFrame.length === 0) return;

    const firstTs = perFrame[0].timestamp_ms;
    const lastTs = perFrame[perFrame.length - 1].timestamp_ms;
    const duration = lastTs - firstTs;
    if (duration <= 0) return;

    // Loop the animation
    const elapsed = progressMs % (duration + 1000); // 1s pause between loops
    const currentMs = firstTs + Math.min(elapsed, duration);

    // Binary search for closest frame
    let lo = 0, hi = perFrame.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (perFrame[mid].timestamp_ms < currentMs) lo = mid + 1;
      else hi = mid;
    }
    const frame = perFrame[lo];
    if (!frame) return;

    const natW = img.naturalWidth || 1920;
    const natH = img.naturalHeight || 1080;
    drawBboxAndTrajectory(ctx, canvas, natW, natH, frame, fullAlert);
  }, [fullAlert]);

  // Video frame callback sync (file source replay)
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !fullAlert || fullAlert.type !== "transit_alert") return;

    let rafId;
    function onFrame() {
      drawOverlay();
      if ("requestVideoFrameCallback" in video) {
        rafId = video.requestVideoFrameCallback(onFrame);
      }
    }

    if ("requestVideoFrameCallback" in video) {
      rafId = video.requestVideoFrameCallback(onFrame);
    } else {
      function fallback() {
        drawOverlay();
        rafId = requestAnimationFrame(fallback);
      }
      rafId = requestAnimationFrame(fallback);
      return () => cancelAnimationFrame(rafId);
    }

    return () => {
      if ("cancelVideoFrameCallback" in video) {
        video.cancelVideoFrameCallback(rafId);
      }
    };
  }, [fullAlert, drawOverlay]);

  // Frozen-frame animation loop (YouTube Live transit replay)
  useEffect(() => {
    if (!isFrozenFrame || !fullAlert || fullAlert.type !== "transit_alert") return;

    let rafId;
    const startTime = performance.now();

    function animate() {
      const elapsed = performance.now() - startTime;
      drawFrozenOverlay(elapsed);
      rafId = requestAnimationFrame(animate);
    }
    rafId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(rafId);
  }, [isFrozenFrame, fullAlert, drawFrozenOverlay]);

  // Frozen-frame static bbox (YouTube Live stagnant replay)
  useEffect(() => {
    if (!isFrozenFrame || !fullAlert || fullAlert.type !== "stagnant_alert") return;

    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    function draw() {
      const container = containerRef.current;
      if (!container) return;

      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const perFrame = fullAlert.per_frame_data || [];
      if (perFrame.length === 0) return;

      // Use the last frame's bbox for stagnant
      const frame = perFrame[perFrame.length - 1];
      const natW = img.naturalWidth || 1920;
      const natH = img.naturalHeight || 1080;

      const scaleX = canvas.width / natW;
      const scaleY = canvas.height / natH;
      const scale = Math.min(scaleX, scaleY);
      const offsetX = (canvas.width - natW * scale) / 2;
      const offsetY = (canvas.height - natH * scale) / 2;
      const tx = (x) => x * scale + offsetX;
      const ty = (y) => y * scale + offsetY;

      const [bx, by, bw, bh] = frame.bbox;
      ctx.strokeStyle = STAGNANT_COLOR;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(tx(bx), ty(by), bw * scale, bh * scale);
    }

    // Draw once the image loads
    if (img.complete) {
      draw();
    } else {
      img.addEventListener("load", draw);
      return () => img.removeEventListener("load", draw);
    }
  }, [isFrozenFrame, fullAlert]);

  if (!alert) {
    return (
      <div className="flex-1 bg-black flex items-center justify-center">
        <div className="text-center">
          <svg className="w-12 h-12 text-text-muted mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-text-muted text-sm">Select an alert to view replay</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 bg-black flex items-center justify-center">
        <p className="text-error-red text-sm">{error}</p>
      </div>
    );
  }

  // Still loading alert data
  if (!fullAlert) {
    return (
      <div className="flex-1 bg-black flex items-center justify-center">
        <div className="text-center">
          <svg className="w-6 h-6 text-text-muted animate-spin mx-auto mb-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="text-text-muted text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  // Frozen-frame replay (YouTube Live sources)
  if (isFrozenFrame) {
    const frameUrl = lastFrameUrl(fullAlert.channel);
    return (
      <div className="flex-1 bg-black flex items-center justify-center relative" ref={containerRef}>
        <img
          ref={imgRef}
          src={frameUrl}
          alt="Last captured frame"
          className="max-w-full max-h-full object-contain"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        <LinesOverlay lines={lines} targetRef={imgRef} />
        {/* Frozen frame badge */}
        <div className="absolute top-3 left-3 px-2 py-1 bg-black/70 rounded text-xs text-text-muted">
          Frozen frame replay
        </div>
      </div>
    );
  }

  // Extracting clip (file sources)
  if (replayStatus === "extracting") {
    return (
      <div className="flex-1 bg-black flex items-center justify-center">
        <div className="text-center">
          <svg className="w-6 h-6 text-text-muted animate-spin mx-auto mb-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="text-text-muted text-sm">Extracting replay clip...</p>
        </div>
      </div>
    );
  }

  const isTransit = alert.type === "transit_alert";
  const clipUrl = replayUrl(alert.alert_id);

  // Stagnant alert — static image
  if (!isTransit) {
    return (
      <div className="flex-1 bg-black flex items-center justify-center relative" ref={containerRef}>
        <img
          src={clipUrl}
          alt="Stagnant alert frame"
          className="max-w-full max-h-full object-contain"
        />
      </div>
    );
  }

  // Transit alert — looping video with canvas overlay
  return (
    <div className="flex-1 bg-black flex items-center justify-center relative" ref={containerRef}>
      <video
        ref={videoRef}
        src={clipUrl}
        loop
        autoPlay
        muted
        playsInline
        className="max-w-full max-h-full object-contain"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
      <LinesOverlay lines={lines} targetRef={videoRef} />
    </div>
  );
}

/**
 * Draw bbox and trajectory trail on a canvas overlay.
 * Shared between video replay and frozen-frame replay.
 */
function drawBboxAndTrajectory(ctx, canvas, natW, natH, frame, fullAlert) {
  const scaleX = canvas.width / natW;
  const scaleY = canvas.height / natH;
  const scale = Math.min(scaleX, scaleY);
  const offsetX = (canvas.width - natW * scale) / 2;
  const offsetY = (canvas.height - natH * scale) / 2;
  const tx = (x) => x * scale + offsetX;
  const ty = (y) => y * scale + offsetY;

  const color = TRANSIT_COLOR;

  // Draw trajectory trail
  const trajectory = fullAlert.full_trajectory || [];
  if (trajectory.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(tx(trajectory[0][0]), ty(trajectory[0][1]));
    for (let i = 1; i < trajectory.length; i++) {
      ctx.lineTo(tx(trajectory[i][0]), ty(trajectory[i][1]));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.globalAlpha = 0.5;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Entry point
    const entry = trajectory[0];
    ctx.beginPath();
    ctx.arc(tx(entry[0]), ty(entry[1]), 4, 0, Math.PI * 2);
    ctx.fillStyle = ENTRY_COLOR;
    ctx.globalAlpha = 0.7;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Draw bbox
  const [bx, by, bw, bh] = frame.bbox;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.strokeRect(tx(bx), ty(by), bw * scale, bh * scale);
}
