/**
 * WebSocket client with auto-reconnect and event emitter pattern.
 *
 * Usage:
 *   const ws = createWs({ channels: [0], types: ["transit_alert", "stats_update"] });
 *   ws.on("transit_alert", (msg) => { ... });
 *   ws.on("stats_update", (msg) => { ... });
 *   ws.connect();
 *   // later:
 *   ws.close();
 */

const MAX_BACKOFF_MS = 5000;

export function createWs({ channels, types } = {}) {
  let socket = null;
  let reconnectTimer = null;
  let backoff = 500;
  let intentionallyClosed = false;
  const listeners = {};

  function buildUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const params = new URLSearchParams();
    if (channels?.length) params.set("channels", channels.join(","));
    if (types?.length) params.set("types", types.join(","));
    const qs = params.toString();
    return `${proto}//${window.location.host}/ws${qs ? `?${qs}` : ""}`;
  }

  function emit(type, data) {
    const handlers = listeners[type] || [];
    for (const fn of handlers) {
      try {
        fn(data);
      } catch (err) {
        console.error(`WS handler error [${type}]:`, err);
      }
    }
  }

  function connect() {
    intentionallyClosed = false;
    const url = buildUrl();
    socket = new WebSocket(url);

    socket.onopen = () => {
      backoff = 500;
      emit("_open", null);
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type) {
          emit(msg.type, msg);
        }
      } catch {
        // ignore malformed messages
      }
    };

    socket.onclose = () => {
      emit("_close", null);
      if (!intentionallyClosed) {
        reconnectTimer = setTimeout(() => {
          backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
          connect();
        }, backoff);
      }
    };

    socket.onerror = () => {
      // onclose will fire after this, triggering reconnect
    };
  }

  function close() {
    intentionallyClosed = true;
    clearTimeout(reconnectTimer);
    if (socket) {
      socket.close();
      socket = null;
    }
  }

  function on(type, handler) {
    if (!listeners[type]) listeners[type] = [];
    listeners[type].push(handler);
  }

  function off(type, handler) {
    if (!listeners[type]) return;
    listeners[type] = listeners[type].filter((fn) => fn !== handler);
  }

  return { connect, close, on, off };
}
