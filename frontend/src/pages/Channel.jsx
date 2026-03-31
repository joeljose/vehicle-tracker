import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { ChannelProvider, useChannel } from "../contexts/ChannelContext";
import { AlertProvider, useAlerts } from "../contexts/AlertContext";
import { useToast } from "../components/Toast";
import AlertFeed from "../components/AlertFeed";
import DrawingCanvas from "../components/DrawingCanvas";
import DrawingTools from "../components/DrawingTools";
import ReplayView from "../components/ReplayView";
import LinesOverlay from "../components/LinesOverlay";
import StatsBar from "../components/StatsBar";
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

function VideoPanel({ channelId, pipelineStarted, phase, imgRef, drawingProps, lines }) {
  const [loaded, setLoaded] = useState(false);
  const [errored, setErrored] = useState(false);

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
        <div className="absolute inset-0 flex items-center justify-center z-10">
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
      {phase === "setup" && loaded && (
        <DrawingCanvas imgRef={imgRef} {...drawingProps} />
      )}
      {phase === "analytics" && loaded && (
        <LinesOverlay lines={lines} targetRef={imgRef} />
      )}
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
  const imgRef = useRef(null);
  const confTimerRef = useRef(null);
  const [error, setError] = useState(null);
  const [pageLoading, setPageLoading] = useState(true);
  const [phaseLoading, setPhaseLoading] = useState(false);

  // Refs to avoid stale closures in WS handlers
  const dispatchRef = useRef(dispatch);
  const alertDispatchRef = useRef(alertDispatch);
  const toastRef = useRef(toast);
  useEffect(() => { dispatchRef.current = dispatch; }, [dispatch]);
  useEffect(() => { alertDispatchRef.current = alertDispatch; }, [alertDispatch]);
  useEffect(() => { toastRef.current = toast; }, [toast]);

  // Review state
  const [selectedAlert, setSelectedAlert] = useState(null);

  // Drawing state
  const [tool, setTool] = useState(null);
  const [roi, setRoi] = useState([]);
  const [lines, setLines] = useState([]);
  const [pendingPoints, setPendingPoints] = useState([]);
  const [labelInput, setLabelInput] = useState(null);

  // Load channel state on mount — no cancellation guards, always set loading false
  useEffect(() => {
    async function load() {
      try {
        const [data, alerts] = await Promise.all([
          getChannel(channelId),
          getAlerts({ channel: channelId }),
        ]);
        dispatch({
          type: "SET_CHANNEL",
          channelId: data.channel_id,
          phase: data.phase,
          source: data.source,
        });
        dispatch({ type: "SET_PIPELINE_STARTED", started: data.pipeline_started });
        alertDispatch({ type: "SET_ALERTS", alerts });
        // Restore entry/exit lines from backend config (survives page reload)
        if (data.entry_exit_lines && Object.keys(data.entry_exit_lines).length > 0) {
          const restored = Object.values(data.entry_exit_lines).map((l) => ({
            label: l.label,
            start: { x: l.start[0], y: l.start[1] },
            end: { x: l.end[0], y: l.end[1] },
          }));
          setLines(restored);
        }
        setError(null);
      } catch {
        setError("load");
      } finally {
        setPageLoading(false);
      }
    }
    load();
  }, [channelId, dispatch, alertDispatch]);

  // WebSocket subscription — uses refs to avoid stale closures
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
        dispatchRef.current({ type: "SET_PHASE", phase: msg.phase });
      }
    });

    ws.on("pipeline_event", (msg) => {
      if (msg.event === "started") {
        dispatchRef.current({ type: "SET_PIPELINE_STARTED", started: true });
      } else if (msg.event === "stopped") {
        dispatchRef.current({ type: "SET_PIPELINE_STARTED", started: false });
        toastRef.current("Pipeline stopped", "error");
      }
    });

    ws.on("transit_alert", (msg) => {
      alertDispatchRef.current({ type: "ADD_ALERT", alert: msg });
    });

    ws.on("stagnant_alert", (msg) => {
      alertDispatchRef.current({ type: "ADD_ALERT", alert: msg });
    });

    ws.on("stats_update", (msg) => {
      if (msg.channel === channelId) {
        dispatchRef.current({ type: "SET_STATS", stats: msg });
      }
    });

    ws.connect();
    wsRef.current = ws;

    return () => ws.close();
  }, [channelId]);

  // Close tab when channel is removed from Home
  useEffect(() => {
    try {
      const bc = new BroadcastChannel("vehicle_tracker");
      bc.onmessage = (e) => {
        if (e.data?.type === "channel_removed" && e.data.channelId === channelId) {
          window.close();
        }
      };
      return () => bc.close();
    } catch (_) { /* BroadcastChannel not supported */ }
  }, [channelId]);

  // Clean up debounce timer on unmount
  useEffect(() => () => clearTimeout(confTimerRef.current), []);

  // Drawing tool callbacks
  const handleToolChange = useCallback((newTool) => {
    setTool(newTool);
    setPendingPoints([]);
    setLabelInput(null);
  }, []);

  const handleRoiChange = useCallback((newRoi) => {
    setRoi(newRoi);
    setTool(null);
  }, []);

  const handleLineComplete = useCallback((line) => {
    setLabelInput(line);
    setTool(null);
  }, []);

  const handleLabelConfirm = useCallback((label) => {
    if (!labelInput) return;
    setLines((prev) => [...prev, { label, start: labelInput.start, end: labelInput.end }]);
    setLabelInput(null);
  }, [labelInput]);

  const handleLabelCancel = useCallback(() => {
    setLabelInput(null);
  }, []);

  const handleClearRoi = useCallback(() => {
    setRoi([]);
  }, []);

  const handleClearLine = useCallback((index) => {
    setLines((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleClearAll = useCallback(() => {
    setRoi([]);
    setLines([]);
    setPendingPoints([]);
    setTool(null);
    setLabelInput(null);
  }, []);

  const handleUndo = useCallback(() => {
    if (pendingPoints.length > 0) {
      setPendingPoints((prev) => prev.slice(0, -1));
    }
  }, [pendingPoints.length]);

  const handleLoadConfig = useCallback(({ roi: loadedRoi, lines: loadedLines }) => {
    setRoi(loadedRoi);
    setLines(loadedLines);
    setPendingPoints([]);
    setTool(null);
    setLabelInput(null);
  }, []);

  const handleStartAnalytics = useCallback(async () => {
    setPhaseLoading(true);
    try {
      // Send ROI + lines with the phase transition
      const entryExitLines = {};
      for (const line of lines) {
        entryExitLines[line.label] = {
          label: line.label,
          start: [line.start.x, line.start.y],
          end: [line.end.x, line.end.y],
        };
      }
      await setChannelPhase(channelId, "analytics", {
        roiPolygon: roi.map((p) => [p.x, p.y]),
        entryExitLines,
      });
      dispatch({ type: "SET_PHASE", phase: "analytics" });
      toast("Analytics started");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setPhaseLoading(false);
    }
  }, [channelId, roi, lines, dispatch, toast]);

  const handleStopAnalytics = useCallback(async () => {
    setPhaseLoading(true);
    try {
      await setChannelPhase(channelId, "review");
      dispatch({ type: "SET_PHASE", phase: "review" });
      toast("Analytics stopped — moved to Review");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setPhaseLoading(false);
    }
  }, [channelId, dispatch, toast]);

  const handleConfidenceChange = useCallback((e) => {
    const val = parseFloat(e.target.value);
    clearTimeout(confTimerRef.current);
    confTimerRef.current = setTimeout(() => {
      updateConfig({ confidence_threshold: val }).catch(() => {});
    }, 300);
  }, []);

  if (pageLoading) {
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

  const drawingProps = {
    tool,
    roi,
    lines,
    onRoiChange: handleRoiChange,
    pendingPoints,
    onPendingChange: setPendingPoints,
    onLineComplete: handleLineComplete,
  };

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
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
        {/* Video / Replay Panel */}
        <div className="flex-1 flex flex-col min-w-0">
          {phase === "review" ? (
            <ReplayView alert={selectedAlert} lines={lines} />
          ) : (
            <VideoPanel
              channelId={channelId}
              pipelineStarted={pipelineStarted}
              phase={phase}
              imgRef={imgRef}
              drawingProps={drawingProps}
              lines={lines}
            />
          )}
          <StatsBar />
        </div>

        {/* Right Sidebar */}
        {phase === "setup" ? (
          <DrawingTools
            tool={tool}
            onToolChange={handleToolChange}
            roi={roi}
            lines={lines}
            pendingPoints={pendingPoints}
            onClearRoi={handleClearRoi}
            onClearLine={handleClearLine}
            onClearAll={handleClearAll}
            onUndo={handleUndo}
            onStartAnalytics={handleStartAnalytics}
            labelInput={labelInput}
            onLabelConfirm={handleLabelConfirm}
            onLabelCancel={handleLabelCancel}
            onLoadConfig={handleLoadConfig}
            loading={phaseLoading}
          />
        ) : (
          <AlertFeed
            phase={phase}
            selectedAlertId={selectedAlert?.alert_id}
            onSelectAlert={setSelectedAlert}
            onStopAnalytics={phase === "analytics" ? handleStopAnalytics : undefined}
            stopLoading={phaseLoading}
          />
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
