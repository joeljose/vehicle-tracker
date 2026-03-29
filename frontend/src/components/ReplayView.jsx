import { useCallback, useEffect, useRef, useState } from "react";
import { getAlert, replayUrl, getReplayStatus } from "../api/rest";

const TRANSIT_COLOR = "#60a5fa";
const STAGNANT_COLOR = "#fbbf24";
const ENTRY_COLOR = "#4ade80";
const LABEL_BG = "rgba(0,0,0,0.7)";

/**
 * ReplayView — replaces MJPEG video panel in Review phase.
 *
 * Props:
 *   alert - selected alert summary (from AlertFeed click)
 */
export default function ReplayView({ alert }) {
  const [fullAlert, setFullAlert] = useState(null);
  const [replayStatus, setReplayStatus] = useState(null);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
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
          getReplayStatus(alert.alert_id),
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

  // Separate polling effect — re-runs when replayStatus changes
  useEffect(() => {
    if (!alert || (replayStatus !== "extracting" && replayStatus !== "not_started")) return;
    let cancelled = false;
    const pollTimer = setInterval(async () => {
      try {
        const s = await getReplayStatus(alert.alert_id);
        if (!cancelled) setReplayStatus(s.status);
      } catch { /* ignore */ }
    }, 2000);
    return () => { cancelled = true; clearInterval(pollTimer); };
  }, [alert?.alert_id, replayStatus]);

  // Canvas overlay for bbox + trajectory (transit clips)
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

    // Get current video time and find matching frame
    const currentMs = (video.currentTime || 0) * 1000;
    const firstTs = perFrame[0].timestamp_ms;
    const adjustedMs = currentMs + firstTs; // clip starts at first_seen - padding

    // Binary search for closest frame
    let lo = 0, hi = perFrame.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (perFrame[mid].timestamp_ms < adjustedMs) lo = mid + 1;
      else hi = mid;
    }
    const frame = perFrame[lo];
    if (!frame) return;

    // Scale factor accounting for object-fit:contain letterboxing
    const natW = video.videoWidth || 1920;
    const natH = video.videoHeight || 1080;
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
  }, [fullAlert]);

  // Video frame callback for sync
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
      // Fallback: use requestAnimationFrame
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

  // Extracting / not ready
  if (replayStatus === "extracting" || replayStatus === "not_started") {
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
    </div>
  );
}
