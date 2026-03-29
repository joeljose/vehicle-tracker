const phaseBadge = {
  setup: {
    bg: "bg-stagnant-amber/10",
    text: "text-stagnant-amber",
    pulse: true,
  },
  analytics: {
    bg: "bg-transit-blue/10",
    text: "text-transit-blue",
    pulse: false,
  },
  review: {
    bg: "bg-active-green/10",
    text: "text-active-green",
    pulse: false,
  },
};

export default function ChannelCard({ channel, onRemove }) {
  const badge = phaseBadge[channel.phase] || phaseBadge.setup;

  function handleOpen(e) {
    // Don't open if clicking the remove button
    if (e.target.closest("[data-remove]")) return;
    window.open(`/channel/${channel.channel_id}`, `channel-${channel.channel_id}`);
  }

  function handleRemove(e) {
    e.preventDefault();
    e.stopPropagation();
    onRemove(channel.channel_id);
  }

  return (
    <div
      onClick={handleOpen}
      className="bg-surface border border-border rounded-xl p-4 hover:border-accent/30 transition-colors cursor-pointer group"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">
            Channel {channel.channel_id}
          </span>
          <span
            className={`px-2 py-0.5 ${badge.bg} ${badge.text} text-xs rounded-full font-medium flex items-center gap-1`}
          >
            {badge.pulse && (
              <span className="w-1.5 h-1.5 rounded-full bg-stagnant-amber animate-pulse" />
            )}
            {channel.phase.charAt(0).toUpperCase() + channel.phase.slice(1)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            data-remove
            onClick={handleRemove}
            className="text-text-muted hover:text-error-red transition-colors opacity-0 group-hover:opacity-100"
            title="Remove channel"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <svg className="w-4 h-4 text-text-muted group-hover:text-text-primary transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </div>
      </div>
      <p className="text-text-muted text-xs truncate mb-3">{channel.source}</p>
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-secondary">
          {channel.alert_count > 0
            ? `${channel.alert_count} alerts`
            : "No alerts yet"}
        </span>
      </div>
    </div>
  );
}
