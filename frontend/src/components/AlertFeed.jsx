import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useAlerts } from "../contexts/AlertContext";
import { snapshotUrl } from "../api/rest";

const CARD_HEIGHT = 76; // px per alert card
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

function AlertCard({ alert }) {
  const isTransit = alert.type === "transit_alert";
  const dotColor = isTransit ? "bg-transit-blue" : "bg-stagnant-amber";
  const textColor = isTransit ? "text-transit-blue" : "text-stagnant-amber";

  let directionLabel = "Stagnant";
  if (isTransit) {
    const entry = alert.entry_arm || alert.entry_label || "?";
    const exit = alert.exit_arm || alert.exit_label || "?";
    // Use short arm names (first letter uppercase)
    const e = entry.charAt(0).toUpperCase();
    const x = exit.charAt(0).toUpperCase();
    directionLabel = `${e} \u21A3 ${x}`;
  }

  const label = alert.label || "vehicle";
  const trackId = alert.track_id ?? "?";
  const duration = alert.duration_frames
    ? `${(alert.duration_frames / 30).toFixed(1)}s`
    : isTransit
      ? ""
      : alert.stationary_duration_frames
        ? `${(alert.stationary_duration_frames / 30).toFixed(0)}s`
        : "";
  const method = alert.method || "";

  return (
    <div className="px-3 py-2.5 border-b border-border/50 cursor-default">
      <div className="flex gap-3">
        {/* Thumbnail */}
        <div className="w-16 h-16 rounded-lg bg-background flex-shrink-0 overflow-hidden">
          {alert.track_id != null ? (
            <img
              src={snapshotUrl(alert.track_id)}
              alt={`Track ${trackId}`}
              className="w-full h-full object-cover"
              onError={(e) => { e.target.style.display = "none"; }}
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
              {formatRelativeTime(alert.timestamp)}
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

export default function AlertFeed({ alertCount }) {
  const { state, dispatch } = useAlerts();
  const parentRef = useRef(null);
  const [newCount, setNewCount] = useState(0);
  const [atTop, setAtTop] = useState(true);
  const prevLenRef = useRef(state.alerts.length);

  // Filtered alerts
  const filtered = useMemo(() => {
    if (state.filter === "all") return state.alerts;
    return state.alerts.filter((a) => a.type === state.filter);
  }, [state.alerts, state.filter]);

  // Virtualizer
  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => CARD_HEIGHT,
    overscan: 10,
  });

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
        // Already at top — scroll stays at top (prepend)
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
            <p className="text-text-muted text-xs">Waiting for alerts...</p>
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
                  <AlertCard alert={alert} />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
