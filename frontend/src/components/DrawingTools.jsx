import { useCallback, useEffect, useState } from "react";
import { saveSiteConfig, loadSiteConfig, listSiteConfigs } from "../api/rest";
import { useToast } from "./Toast";

const LINE_COLORS = ["bg-active-green", "bg-stagnant-amber", "bg-transit-blue", "bg-error-red"];
const DEFAULT_LABELS = ["North", "South", "East", "West"];

/**
 * DrawingTools sidebar — tool selector, lines list, site config, Start Analytics.
 *
 * Props:
 *   tool            - "roi" | "line" | null
 *   onToolChange    - callback(tool)
 *   roi             - completed ROI polygon points
 *   lines           - completed entry/exit lines
 *   pendingPoints   - in-progress points
 *   onClearRoi      - callback()
 *   onClearLine     - callback(index)
 *   onClearAll      - callback()
 *   onUndo          - callback()
 *   onStartAnalytics - callback()
 *   labelInput      - { start, end } | null — line awaiting label
 *   onLabelConfirm  - callback(label) — confirm label for pending line
 *   onLabelCancel   - callback() — cancel pending line
 *   onLoadConfig    - callback({ roi, lines }) — loaded config to render
 */
export default function DrawingTools({
  tool,
  onToolChange,
  roi,
  lines,
  pendingPoints,
  onClearRoi,
  onClearLine,
  onFlipJunctionSide,
  onClearAll,
  onUndo,
  onStartAnalytics,
  labelInput,
  onLabelConfirm,
  onLabelCancel,
  onLoadConfig,
  loading,
}) {
  const toast = useToast();
  const [siteId, setSiteId] = useState("");
  const [savedSites, setSavedSites] = useState([]);
  const [selectedSite, setSelectedSite] = useState("");
  const [labelValue, setLabelValue] = useState("");

  // Fetch saved site configs on mount
  useEffect(() => {
    listSiteConfigs()
      .then((data) => setSavedSites(data.sites || []))
      .catch(() => {});
  }, []);

  // Reset label input when new line is placed
  useEffect(() => {
    if (labelInput) {
      setLabelValue(DEFAULT_LABELS[lines.length] || "");
    }
  }, [labelInput, lines.length]);

  const handleSave = useCallback(async () => {
    if (!siteId.trim()) {
      toast("Enter a site ID", "error");
      return;
    }
    if (roi.length < 3) {
      toast("Draw ROI polygon first", "error");
      return;
    }
    const entryExitLines = {};
    for (const line of lines) {
      entryExitLines[line.label] = {
        label: line.label,
        start: [line.start.x, line.start.y],
        end: [line.end.x, line.end.y],
        junction_side: line.junction_side || "left",
      };
    }
    try {
      await saveSiteConfig({
        site_id: siteId.trim(),
        roi_polygon: roi.map((p) => [p.x, p.y]),
        entry_exit_lines: entryExitLines,
      });
      toast("Config saved");
      // Refresh list
      const data = await listSiteConfigs();
      setSavedSites(data.sites || []);
    } catch (e) {
      toast(e.message, "error");
    }
  }, [siteId, roi, lines, toast]);

  const handleLoad = useCallback(async () => {
    if (!selectedSite) return;
    try {
      const data = await loadSiteConfig(selectedSite);
      const loadedRoi = (data.roi_polygon || []).map(([x, y]) => ({ x, y }));
      const loadedLines = Object.entries(data.entry_exit_lines || {}).map(
        ([key, val]) => ({
          label: val.label || key,
          start: { x: val.start[0], y: val.start[1] },
          end: { x: val.end[0], y: val.end[1] },
          junction_side: val.junction_side || "left",
        })
      );
      onLoadConfig({ roi: loadedRoi, lines: loadedLines });
      setSiteId(selectedSite);
      toast("Config loaded");
    } catch (e) {
      toast(e.message, "error");
    }
  }, [selectedSite, onLoadConfig, toast]);

  const handleLabelSubmit = useCallback(() => {
    const label = labelValue.trim() || DEFAULT_LABELS[lines.length] || "Line";
    onLabelConfirm(label);
    setLabelValue("");
  }, [labelValue, lines.length, onLabelConfirm]);

  const hasRoi = roi.length >= 3;
  const isDrawing = pendingPoints.length > 0;

  return (
    <div className="w-[300px] bg-surface border-l border-border flex flex-col shrink-0">
      {/* Drawing Tools */}
      <div className="p-4 border-b border-border">
        <h3 className="text-sm font-semibold mb-3">Drawing Tools</h3>
        <div className="space-y-2">
          {/* ROI tool */}
          <button
            onClick={() => onToolChange(tool === "roi" ? null : "roi")}
            disabled={hasRoi}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              tool === "roi"
                ? "bg-accent/10 border border-accent/30 text-accent"
                : hasRoi
                  ? "bg-background border border-border text-text-muted cursor-not-allowed"
                  : "bg-background border border-border text-text-secondary hover:text-text-primary hover:border-border-strong cursor-pointer"
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z" />
            </svg>
            Draw ROI Polygon
            {tool === "roi" && <span className="ml-auto text-xs text-accent/60">Active</span>}
            {hasRoi && <span className="ml-auto text-xs text-active-green/60">Done</span>}
          </button>

          {/* Line tool */}
          <button
            onClick={() => onToolChange(tool === "line" ? null : "line")}
            disabled={lines.length >= 4 || !!labelInput}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              tool === "line"
                ? "bg-accent/10 border border-accent/30 text-accent"
                : lines.length >= 4
                  ? "bg-background border border-border text-text-muted cursor-not-allowed"
                  : "bg-background border border-border text-text-secondary hover:text-text-primary hover:border-border-strong cursor-pointer"
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 12h16" />
            </svg>
            Add Entry/Exit Line
            {lines.length > 0 && <span className="ml-auto text-xs text-text-muted">{lines.length}/4</span>}
          </button>

          {/* Undo */}
          <button
            onClick={onUndo}
            disabled={!isDrawing}
            className="w-full flex items-center gap-3 px-3 py-2.5 bg-background border border-border rounded-lg text-sm text-text-secondary hover:text-text-primary hover:border-border-strong transition-colors disabled:text-text-muted disabled:cursor-not-allowed disabled:hover:border-border"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h10a5 5 0 015 5v2M3 10l4-4M3 10l4 4" />
            </svg>
            Undo Last Point
            <span className="ml-auto text-xs text-text-muted">Ctrl+Z</span>
          </button>

          {/* Clear All */}
          <button
            onClick={onClearAll}
            disabled={!hasRoi && lines.length === 0 && !isDrawing}
            className="w-full flex items-center gap-3 px-3 py-2.5 bg-background border border-border rounded-lg text-sm text-text-secondary hover:text-error-red hover:border-error-red/30 transition-colors disabled:text-text-muted disabled:cursor-not-allowed disabled:hover:text-text-muted disabled:hover:border-border"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear All
          </button>
        </div>
      </div>

      {/* Label Input (shown when line just placed) */}
      {labelInput && (
        <div className="p-4 border-b border-border">
          <h3 className="text-sm font-semibold mb-3">Label This Line</h3>
          <input
            type="text"
            value={labelValue}
            onChange={(e) => setLabelValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLabelSubmit()}
            placeholder="Label..."
            autoFocus
            className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm text-text-primary text-center focus:outline-none focus:border-accent/50 mb-2"
          />
          <div className="flex gap-1 mb-2">
            {DEFAULT_LABELS.map((l) => (
              <button
                key={l}
                onClick={() => setLabelValue(l)}
                className={`flex-1 px-1 py-1.5 text-xs rounded transition-colors ${
                  labelValue === l
                    ? "text-accent bg-accent/10 font-medium"
                    : "text-text-muted hover:text-text-primary bg-background"
                }`}
              >
                {l.charAt(0)}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleLabelSubmit}
              className="flex-1 px-3 py-1.5 bg-accent text-white rounded-lg text-xs font-medium hover:bg-accent-hover transition-colors"
            >
              Confirm
            </button>
            <button
              onClick={onLabelCancel}
              className="flex-1 px-3 py-1.5 bg-background border border-border rounded-lg text-xs text-text-secondary hover:text-text-primary transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Entry/Exit Lines List */}
      {lines.length > 0 && (
        <div className="p-4 border-b border-border">
          <h3 className="text-sm font-semibold mb-3">Entry/Exit Lines</h3>
          <div className="space-y-2">
            {lines.map((line, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-2 bg-background rounded-lg">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${LINE_COLORS[i % LINE_COLORS.length]}`} />
                  <span className="text-sm">{line.label}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => onFlipJunctionSide(i)}
                    title={`Junction side: ${line.junction_side || "left"} (click to flip)`}
                    className="flex items-center gap-1 px-1.5 py-0.5 bg-surface border border-border rounded text-[10px] text-text-muted hover:text-text-primary hover:border-border-strong transition-colors"
                  >
                    <svg className={`w-3 h-3 ${line.junction_side === "right" ? "scale-x-[-1]" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                    </svg>
                    {line.junction_side === "right" ? "R" : "L"}
                  </button>
                  <button
                    onClick={() => onClearLine(i)}
                    className="text-text-muted hover:text-error-red transition-colors"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Site Configuration */}
      <div className="p-4 flex-1">
        <h3 className="text-sm font-semibold mb-3">Site Configuration</h3>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-text-muted block mb-1">Site ID</label>
            <input
              type="text"
              value={siteId}
              onChange={(e) => setSiteId(e.target.value)}
              placeholder="e.g. 741_73"
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent/50"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!hasRoi || !siteId.trim()}
              className="flex-1 px-3 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
            <button
              onClick={handleLoad}
              disabled={!selectedSite}
              className="flex-1 px-3 py-2 bg-background border border-border rounded-lg text-sm text-text-secondary hover:text-text-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Load
            </button>
          </div>
          {savedSites.length > 0 && (
            <div>
              <label className="text-xs text-text-muted block mb-1">Saved Configs</label>
              <select
                value={selectedSite}
                onChange={(e) => setSelectedSite(e.target.value)}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent/50"
              >
                <option value="">Select a config...</option>
                {savedSites.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Start Analytics */}
      <div className="p-4 border-t border-border">
        <button
          onClick={onStartAnalytics}
          disabled={!hasRoi || loading}
          className={`w-full px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
            hasRoi
              ? "bg-active-green/10 text-active-green border border-active-green/20 hover:bg-active-green/20"
              : "bg-elevated border border-border text-text-muted cursor-not-allowed"
          }`}
        >
          {loading ? "Starting..." : hasRoi ? "Start Analytics" : "Draw ROI to continue"}
        </button>
      </div>
    </div>
  );
}
