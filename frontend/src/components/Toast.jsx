import { createContext, useCallback, useContext, useRef, useState } from "react";

const ToastContext = createContext(null);

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const idRef = useRef(0);

  const addToast = useCallback((message, variant = "success") => {
    const id = ++idRef.current;
    setToasts((prev) => [...prev, { id, message, variant, exiting: false }]);
    setTimeout(() => {
      setToasts((prev) =>
        prev.map((t) => (t.id === id ? { ...t, exiting: true } : t))
      );
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 150);
    }, 3000);
  }, []);

  return (
    <ToastContext.Provider value={addToast}>
      {children}
      <div className="fixed bottom-4 right-4 flex flex-col items-end gap-2 z-50">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`bg-surface border border-border rounded-lg px-4 py-3 shadow-lg flex items-center gap-3 min-w-[280px] ${
              t.exiting ? "animate-toast-out" : "animate-toast-in"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                t.variant === "error" ? "bg-error-red" : "bg-active-green"
              }`}
            />
            <span
              className={`text-sm ${
                t.variant === "error" ? "text-error-red" : "text-text-primary"
              }`}
            >
              {t.message}
            </span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}
