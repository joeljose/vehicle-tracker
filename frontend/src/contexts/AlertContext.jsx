import { createContext, useContext, useReducer } from "react";

const AlertContext = createContext(null);

const initialState = {
  alerts: [],
  filter: "all", // "all" | "transit_alert" | "stagnant_alert"
};

function alertReducer(state, action) {
  switch (action.type) {
    case "SET_ALERTS": {
      // Backfill: merge with existing alerts, dedup by alert_id
      const existingIds = new Set(state.alerts.map((a) => a.alert_id));
      const newAlerts = action.alerts.filter((a) => !existingIds.has(a.alert_id));
      // Existing (from WS) go first (newest), then backfilled
      return { ...state, alerts: [...state.alerts, ...newAlerts] };
    }
    case "ADD_ALERT": {
      // Skip duplicate (WS might re-deliver)
      if (state.alerts.some((a) => a.alert_id === action.alert.alert_id)) {
        return state;
      }
      return { ...state, alerts: [action.alert, ...state.alerts] };
    }
    case "SET_FILTER":
      return { ...state, filter: action.filter };
    default:
      return state;
  }
}

export function AlertProvider({ children }) {
  const [state, dispatch] = useReducer(alertReducer, initialState);
  return (
    <AlertContext.Provider value={{ state, dispatch }}>
      {children}
    </AlertContext.Provider>
  );
}

export function useAlerts() {
  const ctx = useContext(AlertContext);
  if (!ctx) throw new Error("useAlerts must be used within AlertProvider");
  return ctx;
}
