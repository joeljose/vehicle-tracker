import { useEffect, useRef } from "react";
import { imageToDisplayCoords } from "../lib/coords";

const LINE_COLORS = ["#4ade80", "#fbbf24", "#60a5fa", "#f87171"];
const LABEL_BG = "rgba(0,0,0,0.7)";

/**
 * LinesOverlay — draws entry/exit lines on a transparent canvas over
 * the MJPEG stream (Analytics) or replay video (Review).
 *
 * Props:
 *   lines     - array of {label, start: {x,y}, end: {x,y}} in image coords
 *   targetRef - ref to the <img> or <video> element for geometry
 */
export default function LinesOverlay({ lines, targetRef }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!lines || lines.length === 0) return;

    let rafId;
    function draw() {
      const canvas = canvasRef.current;
      const target = targetRef?.current;
      if (!canvas || !target) {
        rafId = requestAnimationFrame(draw);
        return;
      }

      const parent = canvas.parentElement;
      if (!parent) return;

      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;

      // Works for both <img> (naturalWidth) and <video> (videoWidth)
      const natW = target.naturalWidth || target.videoWidth;
      const natH = target.naturalHeight || target.videoHeight;
      if (!natW || !natH) {
        rafId = requestAnimationFrame(draw);
        return;
      }
      const scaleX = target.clientWidth / natW;
      const scaleY = target.clientHeight / natH;
      const scale = Math.min(scaleX, scaleY);
      const displayW = natW * scale;
      const displayH = natH * scale;
      const offsetX = (target.clientWidth - displayW) / 2;
      const offsetY = (target.clientHeight - displayH) / 2;
      let containerOffsetX = 0, containerOffsetY = 0;
      const targetParent = target.parentElement;
      if (targetParent) {
        const tRect = target.getBoundingClientRect();
        const pRect = targetParent.getBoundingClientRect();
        containerOffsetX = tRect.left - pRect.left;
        containerOffsetY = tRect.top - pRect.top;
      }
      const geo = { scale, offsetX, offsetY, containerOffsetX, containerOffsetY };
      if (geo.scale === 0) {
        rafId = requestAnimationFrame(draw);
        return;
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const color = LINE_COLORS[i % LINE_COLORS.length];
        const s = imageToDisplayCoords(line.start.x, line.start.y, geo);
        const e = imageToDisplayCoords(line.end.x, line.end.y, geo);

        // Line
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(e.x, e.y);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.7;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Label at midpoint
        const midX = (s.x + e.x) / 2;
        const midY = (s.y + e.y) / 2;
        const label = line.label || `Line ${i + 1}`;
        ctx.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
        const textWidth = ctx.measureText(label).width;
        const padX = 10;
        const rectW = textWidth + padX * 2;
        const rectH = 20;

        ctx.fillStyle = LABEL_BG;
        ctx.beginPath();
        ctx.roundRect(midX - rectW / 2, midY - rectH - 4, rectW, rectH, 4);
        ctx.fill();

        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, midX, midY - rectH / 2 - 4 + 4);
      }
    }

    // Draw once, then redraw on resize
    draw();
    const observer = new ResizeObserver(draw);
    const parent = canvasRef.current?.parentElement;
    if (parent) observer.observe(parent);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      observer.disconnect();
    };
  }, [lines, targetRef]);

  if (!lines || lines.length === 0) return null;

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 5 }}
    />
  );
}
