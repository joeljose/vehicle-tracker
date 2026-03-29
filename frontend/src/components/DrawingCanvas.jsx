import { useCallback, useEffect, useRef } from "react";
import {
  getContainedImageGeometry,
  clientToImageCoords,
  imageToDisplayCoords,
} from "../lib/coords";

// Colors matching mockup
const ROI_STROKE = "#6366f1";
const ROI_FILL = "rgba(99,102,241,0.12)";
const VERTEX_FILL = "#fafafa";
const VERTEX_STROKE = "#09090b";
const FIRST_VERTEX_FILL = "#4ade80";
const RUBBER_BAND_ALPHA = 0.6;
const CLOSE_PREVIEW_ALPHA = 0.4;
const LINE_COLORS = ["#4ade80", "#fbbf24", "#60a5fa", "#f87171"];
const LABEL_BG = "rgba(0,0,0,0.7)";
const CLOSE_THRESHOLD_PX = 12; // distance to first vertex to close polygon

/**
 * DrawingCanvas — transparent canvas overlay for ROI polygon + entry/exit lines.
 *
 * Props:
 *   imgRef        - ref to the MJPEG <img> element
 *   tool          - "roi" | "line" | null (current drawing tool)
 *   roi           - array of {x,y} points in image space (completed polygon)
 *   lines         - array of { label, start: {x,y}, end: {x,y} } in image space
 *   onRoiChange   - callback(newRoi)
 *   onLineAdd     - callback({ label, start, end })
 *   pendingPoints - array of {x,y} for in-progress drawing (managed by parent)
 *   onPendingChange - callback(newPending)
 *   onLineComplete  - callback({ start, end }) — line placed, parent shows label input
 */
export default function DrawingCanvas({
  imgRef,
  tool,
  roi,
  lines,
  onRoiChange,
  pendingPoints,
  onPendingChange,
  onLineComplete,
}) {
  const canvasRef = useRef(null);
  const mousePos = useRef(null);
  const geoRef = useRef(null);

  // Recompute geometry and redraw
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    const geo = getContainedImageGeometry(img);
    geoRef.current = geo;

    // Size canvas to container
    const parent = canvas.parentElement;
    if (canvas.width !== parent.clientWidth || canvas.height !== parent.clientHeight) {
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw completed ROI polygon
    if (roi.length >= 3) {
      ctx.beginPath();
      const first = imageToDisplayCoords(roi[0].x, roi[0].y, geo);
      ctx.moveTo(first.x, first.y);
      for (let i = 1; i < roi.length; i++) {
        const p = imageToDisplayCoords(roi[i].x, roi[i].y, geo);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.fillStyle = ROI_FILL;
      ctx.fill();
      ctx.strokeStyle = ROI_STROKE;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Vertices
      for (let i = 0; i < roi.length; i++) {
        const p = imageToDisplayCoords(roi[i].x, roi[i].y, geo);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = VERTEX_FILL;
        ctx.fill();
        ctx.strokeStyle = VERTEX_STROKE;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Draw in-progress ROI polygon
    if (tool === "roi" && pendingPoints.length > 0 && roi.length < 3) {
      ctx.beginPath();
      const first = imageToDisplayCoords(pendingPoints[0].x, pendingPoints[0].y, geo);
      ctx.moveTo(first.x, first.y);
      for (let i = 1; i < pendingPoints.length; i++) {
        const p = imageToDisplayCoords(pendingPoints[i].x, pendingPoints[i].y, geo);
        ctx.lineTo(p.x, p.y);
      }

      // Rubber band to cursor
      if (mousePos.current) {
        const mp = imageToDisplayCoords(mousePos.current.x, mousePos.current.y, geo);
        ctx.lineTo(mp.x, mp.y);
      }

      ctx.strokeStyle = ROI_STROKE;
      ctx.lineWidth = 2;
      ctx.globalAlpha = RUBBER_BAND_ALPHA;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Closing preview (dashed line from cursor back to first vertex)
      if (pendingPoints.length >= 2 && mousePos.current) {
        const mp = imageToDisplayCoords(mousePos.current.x, mousePos.current.y, geo);
        ctx.beginPath();
        ctx.moveTo(mp.x, mp.y);
        ctx.lineTo(first.x, first.y);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = ROI_STROKE;
        ctx.globalAlpha = CLOSE_PREVIEW_ALPHA;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // Vertices
      for (let i = 0; i < pendingPoints.length; i++) {
        const p = imageToDisplayCoords(pendingPoints[i].x, pendingPoints[i].y, geo);
        ctx.beginPath();
        ctx.arc(p.x, p.y, i === 0 ? 6 : 5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? FIRST_VERTEX_FILL : VERTEX_FILL;
        ctx.fill();
        ctx.strokeStyle = VERTEX_STROKE;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Draw completed entry/exit lines
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const color = LINE_COLORS[i % LINE_COLORS.length];
      const s = imageToDisplayCoords(line.start.x, line.start.y, geo);
      const e = imageToDisplayCoords(line.end.x, line.end.y, geo);

      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(e.x, e.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Endpoints
      for (const p of [s, e]) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = VERTEX_FILL;
        ctx.fill();
        ctx.strokeStyle = VERTEX_STROKE;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label at midpoint
      const midX = (s.x + e.x) / 2;
      const midY = (s.y + e.y) / 2;
      const label = line.label || `Line ${i + 1}`;
      ctx.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const padX = 10;
      const padY = 4;
      const rectW = textWidth + padX * 2;
      const rectH = 20;

      ctx.fillStyle = LABEL_BG;
      ctx.beginPath();
      ctx.roundRect(midX - rectW / 2, midY - rectH - 4, rectW, rectH, 4);
      ctx.fill();

      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, midX, midY - rectH / 2 - 4 + padY);
    }

    // Draw in-progress line (single point placed)
    if (tool === "line" && pendingPoints.length === 1) {
      const color = LINE_COLORS[lines.length % LINE_COLORS.length];
      const s = imageToDisplayCoords(pendingPoints[0].x, pendingPoints[0].y, geo);

      // Endpoint
      ctx.beginPath();
      ctx.arc(s.x, s.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = VERTEX_FILL;
      ctx.fill();
      ctx.strokeStyle = VERTEX_STROKE;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Rubber band to cursor
      if (mousePos.current) {
        const mp = imageToDisplayCoords(mousePos.current.x, mousePos.current.y, geo);
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(mp.x, mp.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = RUBBER_BAND_ALPHA;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Draw tooltip when in drawing mode
    if (tool && pendingPoints.length > 0) {
      let tip = "";
      if (tool === "roi") {
        tip = pendingPoints.length >= 3
          ? "Click first point or double-click to close \u00b7 Ctrl+Z to undo \u00b7 Esc to cancel"
          : "Click to add points \u00b7 Ctrl+Z to undo \u00b7 Esc to cancel";
      } else if (tool === "line") {
        tip = "Click to place second endpoint \u00b7 Esc to cancel";
      }
      if (tip) {
        ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
        const tw = ctx.measureText(tip).width;
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.beginPath();
        ctx.roundRect(12, 12, tw + 20, 28, 6);
        ctx.fill();
        ctx.fillStyle = "#a1a1aa";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(tip, 22, 26);
      }
    }
  }, [roi, lines, tool, pendingPoints, imgRef]);

  // Animation loop for smooth rubber band
  useEffect(() => {
    let raf;
    function loop() {
      draw();
      raf = requestAnimationFrame(loop);
    }
    // Only run animation loop when actively drawing
    if (tool && pendingPoints.length > 0) {
      raf = requestAnimationFrame(loop);
    } else {
      draw();
    }
    return () => cancelAnimationFrame(raf);
  }, [draw, tool, pendingPoints.length]);

  // ResizeObserver
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas.parentElement);
    return () => observer.disconnect();
  }, [draw]);

  // Mouse move tracking
  const handleMouseMove = useCallback(
    (e) => {
      const img = imgRef.current;
      if (!img || !tool) return;
      const coords = clientToImageCoords(e.clientX, e.clientY, img);
      mousePos.current = coords;
    },
    [imgRef, tool]
  );

  // Click handler
  const handleClick = useCallback(
    (e) => {
      const img = imgRef.current;
      if (!img || !tool) return;

      const coords = clientToImageCoords(e.clientX, e.clientY, img);
      if (!coords) return; // letterbox click

      if (tool === "roi") {
        // Check if clicking near first vertex to close polygon
        if (pendingPoints.length >= 3) {
          const first = pendingPoints[0];
          const geo = geoRef.current;
          if (geo) {
            const fp = imageToDisplayCoords(first.x, first.y, geo);
            const cp = imageToDisplayCoords(coords.x, coords.y, geo);
            const dist = Math.hypot(fp.x - cp.x, fp.y - cp.y);
            if (dist < CLOSE_THRESHOLD_PX) {
              // Close polygon
              onRoiChange([...pendingPoints]);
              onPendingChange([]);
              return;
            }
          }
        }
        onPendingChange([...pendingPoints, coords]);
      } else if (tool === "line") {
        if (pendingPoints.length === 0) {
          onPendingChange([coords]);
        } else {
          // Line complete — 2 points placed
          onLineComplete({
            start: pendingPoints[0],
            end: coords,
          });
          onPendingChange([]);
        }
      }
    },
    [imgRef, tool, pendingPoints, onRoiChange, onPendingChange, onLineComplete]
  );

  // Double-click to close ROI polygon
  const handleDblClick = useCallback(
    (e) => {
      if (tool !== "roi" || pendingPoints.length < 3) return;
      e.preventDefault();
      onRoiChange([...pendingPoints]);
      onPendingChange([]);
    },
    [tool, pendingPoints, onRoiChange, onPendingChange]
  );

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e) {
      if (!tool) return;

      // Ctrl+Z — undo last point
      if (e.ctrlKey && e.key === "z") {
        e.preventDefault();
        if (pendingPoints.length > 0) {
          onPendingChange(pendingPoints.slice(0, -1));
        }
      }

      // Escape — cancel current drawing
      if (e.key === "Escape") {
        e.preventDefault();
        onPendingChange([]);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [tool, pendingPoints, onPendingChange]);

  const cursor = tool ? "crosshair" : "default";

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ cursor, pointerEvents: tool ? "auto" : "none" }}
      onClick={handleClick}
      onDoubleClick={handleDblClick}
      onMouseMove={handleMouseMove}
    />
  );
}
