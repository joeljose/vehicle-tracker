export default function ConfirmModal({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  confirmVariant = "danger",
  onConfirm,
  onCancel,
}) {
  if (!open) return null;

  const confirmClass =
    confirmVariant === "danger"
      ? "bg-error-red text-white hover:bg-error-red/90"
      : "bg-accent text-white hover:bg-accent-hover";

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-40">
      <div className="bg-surface border border-border rounded-xl p-6 w-[420px] shadow-2xl">
        <h3 className="text-base font-semibold mb-2">{title}</h3>
        <p className="text-text-secondary text-sm mb-5">{message}</p>
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-elevated border border-border-strong rounded-lg text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${confirmClass}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
