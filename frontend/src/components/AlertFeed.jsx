import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useAlerts } from "../contexts/AlertContext";
import { snapshotUrl } from "../api/rest";

const CARD_HEIGHT = 76; // px per alert card
const EXPANDED_HEIGHT = 340; // px for expanded card with detail
const FILTERS = [
  { key: "all", label: "All" },
  { key: "transit_alert", label: "Transit" },
  { key: "stagnant_alert", label: "Stagnant" },
];

function formatRelativeTime(timestamp) {
  if (!timestamp) return "";
  const now = Date.now();
  const ts = typeof timestamp === "number" ? timestamp : new Date(timestamp).getTime();
  const diff = Math.max(0, Math.floor((now - ts) / 1000));
  if (diff < 5) return "just now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function AlertCard({ alert, isReview, isSelected, onSelect }) {
  const isTransit = alert.type === "transit_alert";
  const dotColor = isTransit ? "bg-transit-blue" : "bg-stagnant-amber";
  const textColor = isTransit ? "text-transit-blue" : "text-stagnant-amber";
  const borderColor = isTransit ? "border-l-transit-blue" : "border-l-stagnant-amber";
  const bgColor = isTransit ? "bg-transit-blue/5" : "bg-stagnant-amber/5";

  let directionLabel = "Stagnant";
  if (isTransit) {
    const entry = alert.entry_arm || alert.entry_label || "?";
    const exit = alert.exit_arm || alert.exit_label || "?";
    const e = entry.charAt(0).toUpperCase();
    const x = exit.charAt(0).toUpperCase();
    directionLabel = `${e} \u21A3 ${x}`;
  }

  const label = alert.label || alert.class || "vehicle";
  const trackId = alert.track_id ?? "?";
  const duration = alert.duration_frames
    ? `${(alert.duration_frames / 30).toFixed(1)}s`
    : isTransit
      ? alert.duration_ms ? `${(alert.duration_ms / 1000).toFixed(1)}s` : ""
      : alert.stationary_duration_frames
        ? `${(alert.stationary_duration_frames / 30).toFixed(0)}s`
        : alert.stationary_duration_ms ? `${(alert.stationary_duration_ms / 1000).toFixed(0)}s` : "";
  const method = alert.method || "";

  return (
    <div
      className={`px-3 py-2.5 border-b border-border/50 ${
        isSelected ? `${bgColor} border-l-2 ${borderColor}` : ""
      } ${isReview ? "cursor-pointer hover:bg-background/50 transition-colors" : "cursor-default"}`}
      onClick={() => isReview && onSelect?.(alert)}
    >
      <div className="flex gap-3">
        {/* Thumbnail */}
        <div className="w-16 h-16 rounded-lg bg-background flex-shrink-0 overflow-hidden">
          {alert.track_id != null ? (
            <img
              src={snapshotUrl(alert.track_id)}
              alt={`Track ${trackId}`}
              className="w-full h-full object-cover"
              onError={(e) => {
                const img = e.target;
                if (!img.dataset.retried) {
                  img.dataset.retried = "1";
                  setTimeout(() => { img.src = img.src; }, 1000);
                } else {
                  img.style.display = "none";
                }
              }}
            />
          ) : (
            <div className="w-full h-full bg-gradient-to-br from-zinc-700 to-zinc-800" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
            <span className={`text-xs font-medium ${textColor}`}>{directionLabel}</span>
            <span className="text-[10px] text-text-muted ml-auto">
              {formatRelativeTime(alert.timestamp || alert.timestamp_ms)}
            </span>
          </div>
          <div className="text-xs text-text-secondary truncate">
            {label} #{trackId}
            {duration && ` \u00b7 ${duration}`}
          </div>
          {method && (
            <div className="text-[10px] text-text-muted mt-0.5">{method}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function AlertDetail({ alert }) {
  const isTransit = alert.type === "transit_alert";

  return (
    <div className="px-3 pb-3 space-y-3">
      {/* Best photo */}
      <div className="w-full aspect-video rounded-lg bg-background overflow-hidden">
        {alert.track_id != null ? (
          <img
            src={snapshotUrl(alert.track_id)}
            alt="Best photo"
            className="w-full h-full object-cover"
            onError={(e) => {
                const img = e.target;
                if (!img.dataset.retried) {
                  img.dataset.retried = "1";
                  setTimeout(() => { img.src = img.src; }, 1000);
                } else {
                  img.style.display = "none";
                }
              }}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center">
            <span className="text-text-muted text-xs">No photo</span>
          </div>
        )}
      </div>

      {/* Metadata grid */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        <div>
          <span className="text-text-muted">Track ID</span>
          <p className="text-text-primary">#{alert.track_id ?? "?"}</p>
        </div>
        <div>
          <span className="text-text-muted">Class</span>
          <p className="text-text-primary">{alert.class || alert.label || "unknown"}</p>
        </div>
        {isTransit ? (
          <>
            <div>
              <span className="text-text-muted">Direction</span>
              <p className="text-text-primary">
                {alert.entry_direction || alert.entry_arm || "?"} → {alert.exit_direction || alert.exit_arm || "?"}
              </p>
            </div>
            <div>
              <span className="text-text-muted">Duration</span>
              <p className="text-text-primary">
                {alert.duration_ms ? `${(alert.duration_ms / 1000).toFixed(1)}s` : "—"}
              </p>
            </div>
            <div>
              <span className="text-text-muted">Method</span>
              <p className="text-text-primary">{alert.method || "—"}</p>
            </div>
            <div>
              <span className="text-text-muted">Confidence</span>
              <p className="text-text-primary">
                {alert.avg_confidence != null ? alert.avg_confidence.toFixed(2) : "—"}
              </p>
            </div>
          </>
        ) : (
          <>
            <div>
              <span className="text-text-muted">Duration</span>
              <p className="text-text-primary">
                {alert.stationary_duration_ms
                  ? `${Math.floor(alert.stationary_duration_ms / 1000)}s`
                  : "—"}
              </p>
            </div>
            <div>
              <span className="text-text-muted">Position</span>
              <p className="text-text-primary">
                {alert.position ? `(${alert.position[0]}, ${alert.position[1]})` : "—"}
              </p>
            </div>
          </>
        )}
      </div>

      {/* Download button */}
      {alert.track_id != null && (
        <a
          href={snapshotUrl(alert.track_id)}
          download={`track_${alert.track_id}.jpg`}
          className="block w-full px-3 py-2 bg-background border border-border rounded-lg text-xs text-text-secondary hover:text-text-primary hover:border-border-strong transition-colors text-center"
        >
          Download Photo
        </a>
      )}
    </div>
  );
}

export default function AlertFeed({ phase, selectedAlertId, onSelectAlert, onStopAnalytics, stopLoading }) {
  const { state, dispatch } = useAlerts();
  const parentRef = useRef(null);
  const [newCount, setNewCount] = useState(0);
  const [atTop, setAtTop] = useState(true);
  const prevLenRef = useRef(state.alerts.length);
  const isReview = phase === "review";

  // Filtered alerts
  const filtered = useMemo(() => {
    if (state.filter === "all") return state.alerts;
    return state.alerts.filter((a) => a.type === state.filter);
  }, [state.alerts, state.filter]);

  // Virtualizer
  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => {
      const alert = filtered[index];
      if (isReview && alert && alert.alert_id === selectedAlertId) {
        return CARD_HEIGHT + EXPANDED_HEIGHT;
      }
      return CARD_HEIGHT;
    },
    overscan: 10,
  });

  // Re-measure when selection changes
  useEffect(() => {
    virtualizer.measure();
  }, [selectedAlertId, virtualizer]);

  // Track if user is at top of scroll
  const handleScroll = useCallback(() => {
    const el = parentRef.current;
    if (!el) return;
    const isAtTop = el.scrollTop < 20;
    setAtTop(isAtTop);
    if (isAtTop) setNewCount(0);
  }, []);

  // Auto-scroll when new alerts arrive and user is at top
  useEffect(() => {
    const len = filtered.length;
    const prevLen = prevLenRef.current;
    prevLenRef.current = len;

    if (len > prevLen) {
      if (atTop) {
        setNewCount(0);
      } else {
        setNewCount((c) => c + (len - prevLen));
      }
    }
  }, [filtered.length, atTop]);

  const jumpToTop = useCallback(() => {
    parentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    setNewCount(0);
    setAtTop(true);
  }, []);

  return (
    <div className="w-[300px] bg-surface border-l border-border flex flex-col shrink-0">
      {/* Stop Analytics */}
      {onStopAnalytics && (
        <div className="px-3 py-2.5 border-b border-border shrink-0">
          <button
            onClick={onStopAnalytics}
            disabled={stopLoading}
            className="w-full px-3 py-2 bg-error-red/10 text-error-red hover:bg-error-red/20 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {stopLoading ? "Stopping..." : "Stop Analytics"}
          </button>
        </div>
      )}
      {/* Filter Bar */}
      <div className="px-3 py-2.5 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-1 bg-background rounded-lg p-0.5">
          {FILTERS.map((f) => {
            const count = f.key === "all"
              ? state.alerts.length
              : state.alerts.filter((a) => a.type === f.key).length;
            return (
              <button
                key={f.key}
                onClick={() => dispatch({ type: "SET_FILTER", filter: f.key })}
                className={`px-2 py-1 text-xs font-medium rounded-md transition-colors ${
                  state.filter === f.key
                    ? "bg-text-primary/10 text-text-primary"
                    : "text-text-muted hover:text-text-primary"
                }`}
              >
                {f.label} ({count})
              </button>
            );
          })}
        </div>

        {/* New alerts badge */}
        {newCount > 0 && (
          <button
            onClick={jumpToTop}
            className="px-2 py-0.5 bg-transit-blue/10 text-transit-blue text-[10px] rounded-full font-medium animate-pulse"
          >
            {newCount} new ↓
          </button>
        )}
      </div>

      {/* Scrollable Alert List */}
      <div
        ref={parentRef}
        className="flex-1 overflow-y-auto"
        onScroll={handleScroll}
      >
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-text-muted text-xs">
              {isReview ? "No alerts to review" : "Waiting for alerts..."}
            </p>
          </div>
        ) : (
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: "100%",
              position: "relative",
            }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const alert = filtered[virtualRow.index];
              const isSelected = isReview && alert.alert_id === selectedAlertId;
              return (
                <div
                  key={alert.alert_id || virtualRow.index}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: `${virtualRow.size}px`,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <AlertCard
                    alert={alert}
                    isReview={isReview}
                    isSelected={isSelected}
                    onSelect={onSelectAlert}
                  />
                  {isSelected && <AlertDetail alert={alert} />}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
