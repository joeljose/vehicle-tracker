import { useChannel } from "../contexts/ChannelContext";
import { useAlerts } from "../contexts/AlertContext";

function fpsColor(fps) {
  if (fps == null) return "text-text-muted";
  if (fps < 5) return "text-error-red";
  if (fps < 15) return "text-stagnant-amber";
  return "text-active-green";
}

function inferenceColor(ms) {
  if (ms == null) return "text-text-muted";
  if (ms > 40) return "text-error-red";
  if (ms > 20) return "text-stagnant-amber";
  return "text-text-primary";
}

const PHASE_COLORS = {
  setup: "text-stagnant-amber",
  analytics: "text-transit-blue",
  review: "text-active-green",
};

export default function StatsBar() {
  const { state } = useChannel();
  const { state: alertState } = useAlerts();
  const { stats, phase } = state;

  if (phase !== "analytics") return null;

  const fps = stats?.fps;
  const tracks = stats?.active_tracks;
  const inferenceMs = stats?.inference_ms;
  const alertCount = alertState.alerts.length;

  return (
    <div className="bg-surface border-t border-border px-4 py-2 flex items-center gap-6 text-xs shrink-0">
      <span className="text-text-muted">
        FPS:{" "}
        <span className={`font-medium ${fpsColor(fps)}`}>
          {fps != null ? fps.toFixed(1) : "—"}
        </span>
      </span>
      <span className="text-text-muted">
        Tracks:{" "}
        <span className="text-text-primary font-medium">
          {tracks ?? "—"}
        </span>
      </span>
      <span className="text-text-muted">
        Inference:{" "}
        <span className={`font-medium ${inferenceColor(inferenceMs)}`}>
          {inferenceMs != null ? `${inferenceMs.toFixed(1)}ms` : "—"}
        </span>
      </span>
      <span className="text-text-muted">
        Phase:{" "}
        <span className={`font-medium ${PHASE_COLORS[phase] || "text-text-primary"}`}>
          {phase ? phase.charAt(0).toUpperCase() + phase.slice(1) : "—"}
        </span>
      </span>
      <div className="ml-auto text-text-secondary">
        {alertCount} alert{alertCount !== 1 ? "s" : ""}
      </div>
    </div>
  );
}
