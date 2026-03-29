import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { ChannelProvider, useChannel } from "../contexts/ChannelContext";
import { AlertProvider, useAlerts } from "../contexts/AlertContext";
import { useToast } from "../components/Toast";
import { getChannel, getAlerts, setChannelPhase, updateConfig } from "../api/rest";
import { createWs } from "../api/ws";

const PHASES = ["setup", "analytics", "review"];

const PHASE_LABELS = { setup: "Setup", analytics: "Analytics", review: "Review" };

const PHASE_COLORS = {
  setup: "bg-stagnant-amber/10 text-stagnant-amber",
  analytics: "bg-accent/10 text-accent",
  review: "bg-active-green/10 text-active-green",
};

function PhaseIndicator({ current }) {
  return (
    <div className="flex items-center gap-1 bg-background rounded-lg p-0.5">
      {PHASES.map((p) => (
        <span
          key={p}
          className={`px-3 py-1.5 text-xs font-medium rounded-md ${
            p === current ? PHASE_COLORS[p] : "text-text-muted"
          }`}
        >
          {PHASE_LABELS[p]}
        </span>
      ))}
    </div>
  );
}

function VideoPanel({ channelId, pipelineStarted }) {
  const [loaded, setLoaded] = useState(false);
  const [errored, setErrored] = useState(false);
  const imgRef = useRef(null);

  // Reset state when pipeline status changes
  useEffect(() => {
    if (pipelineStarted) {
      setErrored(false);
      setLoaded(false);
    }
  }, [pipelineStarted]);

  const streamUrl = `/api/stream/${channelId}`;

  if (!pipelineStarted || errored) {
    return (
      <div className="flex-1 bg-black flex items-center justify-center">
        <div className="text-center">
          <svg className="w-12 h-12 text-text-muted mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <p className="text-text-muted text-sm">
            {!pipelineStarted ? "Pipeline not started" : "Stream unavailable"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-black flex items-center justify-center relative">
      {!loaded && (
        <div className="absolute inset-0 flex items-center justify-center">
          <svg className="w-6 h-6 text-text-muted animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        </div>
      )}
      <img
        ref={imgRef}
        src={streamUrl}
        alt="Live feed"
        className={`max-w-full max-h-full object-contain ${loaded ? "" : "invisible"}`}
        onLoad={() => setLoaded(true)}
        onError={() => setErrored(true)}
      />
    </div>
  );
}

function ControlPanel({ phase }) {
  const { state, dispatch } = useChannel();
  const toast = useToast();
  const [confidence, setConfidence] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef(null);

  const handleConfidenceChange = useCallback(
    (e) => {
      const val = parseFloat(e.target.value);
      setConfidence(val);
      clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        updateConfig({ confidence_threshold: val }).catch(() => {});
      }, 300);
    },
    []
  );

  const handleStartAnalytics = useCallback(async () => {
    setLoading(true);
    try {
      await setChannelPhase(state.channelId, "analytics");
      dispatch({ type: "SET_PHASE", phase: "analytics" });
      toast("Analytics started");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setLoading(false);
    }
  }, [state.channelId, dispatch, toast]);

  if (phase !== "setup") return null;

  return (
    <div className="w-[300px] bg-surface border-l border-border flex flex-col shrink-0">
      {/* Drawing Tools Placeholder */}
      <div className="p-4 border-b border-border">
        <h3 className="text-sm font-semibold mb-3">Drawing Tools</h3>
        <div className="space-y-2">
          <button disabled className="w-full flex items-center gap-3 px-3 py-2.5 bg-background border border-border rounded-lg text-sm text-text-muted cursor-not-allowed">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z"/></svg>
            Draw ROI Polygon
          </button>
          <button disabled className="w-full flex items-center gap-3 px-3 py-2.5 bg-background border border-border rounded-lg text-sm text-text-muted cursor-not-allowed">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 12h16"/></svg>
            Add Entry/Exit Line
          </button>
        </div>
        <p className="text-text-muted text-xs mt-2">Drawing tools available in Phase 4</p>
      </div>

      {/* Confidence Threshold */}
      <div className="p-4 flex-1">
        <h3 className="text-sm font-semibold mb-3">Configuration</h3>
        <div>
          <label className="text-xs text-text-muted block mb-1">Confidence Threshold</label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={confidence}
              onChange={handleConfidenceChange}
              className="flex-1 h-1.5 bg-elevated rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
            />
            <span className="text-xs text-text-secondary w-8 text-right">
              {confidence.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Start Analytics */}
      <div className="p-4 border-t border-border">
        <button
          onClick={handleStartAnalytics}
          disabled={loading}
          className="w-full px-4 py-2.5 bg-active-green/10 text-active-green border border-active-green/20 rounded-lg text-sm font-medium hover:bg-active-green/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Starting..." : "Start Analytics"}
        </button>
      </div>
    </div>
  );
}

function AnalyticsPlaceholder() {
  return (
    <div className="w-[300px] bg-surface border-l border-border flex flex-col shrink-0">
      <div className="p-4 border-b border-border">
        <h3 className="text-sm font-semibold mb-2">Alert Feed</h3>
        <p className="text-text-muted text-xs">Alerts will appear here during analytics.</p>
      </div>
      <div className="flex-1 flex items-center justify-center">
        <p className="text-text-muted text-xs">Waiting for alerts...</p>
      </div>
    </div>
  );
}

function StatsBar({ phase }) {
  if (phase !== "analytics") return null;

  return (
    <div className="bg-surface border-t border-border px-4 py-2 flex items-center gap-6 shrink-0">
      <span className="text-xs text-text-muted">Stats bar — Phase 5</span>
    </div>
  );
}

function ChannelContent() {
  const { id } = useParams();
  const channelId = Number(id);
  const { state, dispatch } = useChannel();
  const { dispatch: alertDispatch } = useAlerts();
  const toast = useToast();
  const wsRef = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load channel state on mount
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await getChannel(channelId);
        if (cancelled) return;
        dispatch({
          type: "SET_CHANNEL",
          channelId: data.channel_id,
          phase: data.phase,
          source: data.source,
        });
        dispatch({ type: "SET_PIPELINE_STARTED", started: data.pipeline_started });

        // Backfill alerts
        const alerts = await getAlerts({ channel: channelId });
        if (cancelled) return;
        alertDispatch({ type: "SET_ALERTS", alerts });
        setError(null);
      } catch {
        if (!cancelled) setError("load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [channelId, dispatch, alertDispatch]);

  // WebSocket subscription
  useEffect(() => {
    const ws = createWs({
      channels: [channelId],
      types: [
        "transit_alert",
        "stagnant_alert",
        "stats_update",
        "pipeline_event",
        "phase_changed",
      ],
    });

    ws.on("phase_changed", (msg) => {
      if (msg.channel === channelId) {
        dispatch({ type: "SET_PHASE", phase: msg.phase });
      }
    });

    ws.on("pipeline_event", (msg) => {
      if (msg.event === "started") {
        dispatch({ type: "SET_PIPELINE_STARTED", started: true });
      } else if (msg.event === "stopped") {
        dispatch({ type: "SET_PIPELINE_STARTED", started: false });
        toast("Pipeline stopped", "error");
      }
    });

    ws.on("transit_alert", (msg) => {
      alertDispatch({ type: "ADD_ALERT", alert: msg });
    });

    ws.on("stagnant_alert", (msg) => {
      alertDispatch({ type: "ADD_ALERT", alert: msg });
    });

    ws.on("stats_update", (msg) => {
      if (msg.channel === channelId) {
        dispatch({ type: "SET_STATS", stats: msg });
      }
    });

    ws.connect();
    wsRef.current = ws;

    return () => ws.close();
  }, [channelId, dispatch, alertDispatch, toast]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <svg className="w-6 h-6 text-text-muted animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="bg-error-red/5 border border-error-red/20 rounded-xl p-6 text-center max-w-md">
          <h3 className="text-error-red font-medium mb-1">Channel Not Found</h3>
          <p className="text-text-secondary text-sm">
            Could not load channel {channelId}. The pipeline may not be running.
          </p>
          <a
            href="/"
            className="mt-4 inline-block px-4 py-2 bg-surface border border-border rounded-lg text-sm text-text-primary hover:border-border-strong transition-colors"
          >
            Back to Home
          </a>
        </div>
      </div>
    );
  }

  const { phase, source, pipelineStarted } = state;

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Control Bar */}
      <div className="bg-surface border-b border-border px-4 py-2.5 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <a
            href="/"
            className="text-text-muted hover:text-text-primary transition-colors text-sm flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
            </svg>
            Home
          </a>
          <div className="h-4 w-px bg-border" />
          <span className="text-sm font-semibold">Channel {channelId}</span>
          <span className="text-text-muted text-xs truncate max-w-[200px]">{source}</span>
        </div>

        <PhaseIndicator current={phase} />

        <div className="flex items-center gap-3">
          {pipelineStarted ? (
            <span className="flex items-center gap-1.5 text-xs text-active-green">
              <span className="w-1.5 h-1.5 rounded-full bg-active-green" />
              Connected
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs text-error-red">
              <span className="w-1.5 h-1.5 rounded-full bg-error-red" />
              Disconnected
            </span>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 min-h-0">
        {/* Video Panel */}
        <div className="flex-1 flex flex-col min-w-0">
          <VideoPanel channelId={channelId} pipelineStarted={pipelineStarted} />
          <StatsBar phase={phase} />
        </div>

        {/* Right Sidebar */}
        {phase === "setup" ? (
          <ControlPanel phase={phase} />
        ) : (
          <AnalyticsPlaceholder />
        )}
      </div>
    </div>
  );
}

export default function Channel() {
  const { id } = useParams();

  return (
    <ChannelProvider channelId={Number(id)}>
      <AlertProvider>
        <ChannelContent />
      </AlertProvider>
    </ChannelProvider>
  );
}
