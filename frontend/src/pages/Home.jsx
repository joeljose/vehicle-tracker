import { useCallback, useEffect, useRef, useState } from "react";
import { useHome } from "../contexts/HomeContext";
import { useToast } from "../components/Toast";
import ChannelCard from "../components/ChannelCard";
import ConfirmModal from "../components/ConfirmModal";
import {
  getChannels,
  startPipeline,
  stopPipeline,
  addChannel,
  removeChannel,
} from "../api/rest";
import { createWs } from "../api/ws";

export default function Home() {
  const { state, dispatch } = useHome();
  const toast = useToast();
  const toastRef = useRef(toast);
  toastRef.current = toast;
  const [source, setSource] = useState("");
  const [confirmStop, setConfirmStop] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(null);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  // Fetch initial state
  useEffect(() => {
    getChannels()
      .then((data) => {
        dispatch({
          type: "SET_CHANNELS",
          channels: data.channels,
          pipelineStarted: data.pipeline_started,
        });
        setError(null);
      })
      .catch(() => setError("backend"));
  }, [dispatch]);

  // WebSocket connection
  useEffect(() => {
    const ws = createWs({
      types: [
        "pipeline_event",
        "phase_changed",
        "transit_alert",
        "stagnant_alert",
        "channel_added",
        "resolution_failed",
      ],
    });

    ws.on("pipeline_event", (msg) => {
      if (msg.event === "started") {
        dispatch({ type: "SET_PIPELINE_STARTED", started: true });
      } else if (msg.event === "stopped") {
        dispatch({ type: "PIPELINE_STOPPED" });
      }
    });

    ws.on("channel_added", (msg) => {
      dispatch({
        type: "ADD_CHANNEL",
        channel: {
          channel_id: msg.channel,
          source: msg.source,
          phase: "setup",
          alert_count: 0,
          source_type: msg.source_type,
        },
      });
      toastRef.current("YouTube stream connected");
      window.open(`/channel/${msg.channel}`, `channel-${msg.channel}`);
    });

    ws.on("resolution_failed", (msg) => {
      toastRef.current(msg.error || "Stream resolution failed", "error");
    });

    ws.on("phase_changed", (msg) => {
      dispatch({
        type: "PHASE_CHANGED",
        channel: msg.channel,
        phase: msg.phase,
      });
    });

    ws.on("transit_alert", (msg) => {
      dispatch({ type: "INCREMENT_ALERT", channel: msg.channel });
    });

    ws.on("stagnant_alert", (msg) => {
      dispatch({ type: "INCREMENT_ALERT", channel: msg.channel });
    });

    ws.connect();
    wsRef.current = ws;

    return () => ws.close();
  }, [dispatch]);

  const handleStart = useCallback(async () => {
    dispatch({ type: "SET_LOADING", loading: true });
    try {
      await startPipeline();
      dispatch({ type: "SET_PIPELINE_STARTED", started: true });
      toast("Pipeline started");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      dispatch({ type: "SET_LOADING", loading: false });
    }
  }, [dispatch, toast]);

  const handleStop = useCallback(async () => {
    setConfirmStop(false);
    dispatch({ type: "SET_LOADING", loading: true });
    try {
      await stopPipeline();
      dispatch({ type: "PIPELINE_STOPPED" });
      toast("Pipeline stopped");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      dispatch({ type: "SET_LOADING", loading: false });
    }
  }, [dispatch, toast]);

  const handleAddChannel = useCallback(async () => {
    if (!source.trim()) return;
    try {
      const data = await addChannel(source.trim());
      if (data.resolving) {
        // YouTube URL — resolving asynchronously, WS will notify when done
        setSource("");
        toast("Resolving stream...");
      } else {
        // File source — added synchronously
        dispatch({
          type: "ADD_CHANNEL",
          channel: {
            channel_id: data.channel_id,
            source: source.trim(),
            phase: "setup",
            alert_count: 0,
          },
        });
        window.open(`/channel/${data.channel_id}`, `channel-${data.channel_id}`);
        setSource("");
        toast("Channel added");
      }
    } catch (e) {
      toast(e.message, "error");
    }
  }, [source, dispatch, toast]);

  const handleRemoveChannel = useCallback(
    async (channelId) => {
      setConfirmRemove(null);
      try {
        await removeChannel(channelId);
        dispatch({ type: "REMOVE_CHANNEL", channelId });
        // Notify open channel tabs to close
        try {
          const bc = new BroadcastChannel("vehicle_tracker");
          bc.postMessage({ type: "channel_removed", channelId });
          bc.close();
        } catch (_) { /* BroadcastChannel not supported */ }
        toast("Channel removed");
      } catch (e) {
        toast(e.message, "error");
      }
    },
    [dispatch, toast]
  );

  // Backend unreachable
  if (error === "backend") {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="bg-error-red/5 border border-error-red/20 rounded-xl p-6 text-center max-w-md">
          <div className="w-12 h-12 bg-error-red/10 rounded-xl flex items-center justify-center mx-auto mb-3">
            <svg className="w-6 h-6 text-error-red" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h3 className="text-error-red font-medium mb-1">Backend Unreachable</h3>
          <p className="text-text-secondary text-sm">
            Cannot connect to{" "}
            <code className="text-text-primary bg-elevated px-1.5 py-0.5 rounded text-xs">
              localhost:8000
            </code>
          </p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-surface border border-border rounded-lg text-sm text-text-primary hover:border-border-strong transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  // Pipeline stopped — centered CTA
  if (!state.pipelineStarted && !state.loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-6">
        <div className="w-16 h-16 bg-surface rounded-2xl flex items-center justify-center mb-4">
          <svg className="w-8 h-8 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium mb-1">No active channels</h3>
        <p className="text-text-secondary text-sm max-w-md text-center mb-6">
          Start the pipeline, then add a video source to begin monitoring a
          junction.
        </p>
        <button
          onClick={handleStart}
          className="px-6 py-3 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-hover transition-colors"
        >
          Start Pipeline
        </button>
      </div>
    );
  }

  // Pipeline running (or loading)
  return (
    <div className="max-w-5xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Vehicle Tracker</h1>
          <p className="text-text-secondary text-sm mt-1">
            Junction monitoring dashboard
          </p>
        </div>
        <div className="flex items-center gap-3">
          {state.loading ? (
            <>
              <span className="flex items-center gap-2 text-sm text-stagnant-amber">
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Starting...
              </span>
              <button disabled className="px-4 py-2 bg-surface border border-border rounded-lg text-sm font-medium text-text-muted cursor-not-allowed">
                Starting...
              </button>
            </>
          ) : (
            <>
              <span className="flex items-center gap-2 text-sm text-active-green">
                <span className="w-2 h-2 rounded-full bg-active-green animate-pulse" />
                Pipeline Running
              </span>
              <button
                onClick={() => setConfirmStop(true)}
                className="px-4 py-2 bg-error-red/10 text-error-red border border-error-red/20 rounded-lg text-sm font-medium hover:bg-error-red/20 transition-colors"
              >
                Stop Pipeline
              </button>
            </>
          )}
        </div>
      </div>

      {/* Add Channel */}
      <div className="flex gap-3 mb-6">
        <input
          type="text"
          placeholder="Video source path or URL..."
          value={source}
          onChange={(e) => setSource(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAddChannel()}
          className="flex-1 px-4 py-2.5 bg-surface border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/20"
        />
        <button
          onClick={handleAddChannel}
          disabled={!source.trim() || state.loading}
          className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-hover transition-colors whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
        >
          + Add Channel
        </button>
      </div>

      {/* Channel Cards */}
      {state.channels.length === 0 ? (
        <div className="text-center py-12 text-text-muted text-sm">
          No channels yet. Add a video source above to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {state.channels.map((ch) => (
            <ChannelCard
              key={ch.channel_id}
              channel={ch}
              onRemove={(id) => setConfirmRemove(id)}
            />
          ))}
        </div>
      )}

      {/* Confirmation modals */}
      <ConfirmModal
        open={confirmStop}
        title="Stop Pipeline?"
        message="All in-memory alerts, snapshots, and replay data will be cleared. This cannot be undone."
        confirmLabel="Stop Pipeline"
        confirmVariant="danger"
        onConfirm={handleStop}
        onCancel={() => setConfirmStop(false)}
      />
      <ConfirmModal
        open={confirmRemove !== null}
        title="Remove Channel?"
        message={`Channel ${confirmRemove} and all its alerts will be removed.`}
        confirmLabel="Remove"
        confirmVariant="danger"
        onConfirm={() => handleRemoveChannel(confirmRemove)}
        onCancel={() => setConfirmRemove(null)}
      />
    </div>
  );
}
