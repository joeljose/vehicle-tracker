import { createContext, useContext, useReducer } from "react";

const AlertContext = createContext(null);

const initialState = {
  alerts: [],
  filter: "all", // "all" | "transit_alert" | "stagnant_alert"
};

function alertReducer(state, action) {
  switch (action.type) {
    case "SET_ALERTS":
      return { ...state, alerts: action.alerts };
    case "ADD_ALERT":
      return { ...state, alerts: [action.alert, ...state.alerts] };
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
