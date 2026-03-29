import { createContext, useContext, useReducer } from "react";

const HomeContext = createContext(null);

const initialState = {
  channels: [],
  pipelineStarted: false,
};

function homeReducer(state, action) {
  switch (action.type) {
    case "SET_CHANNELS":
      return { ...state, channels: action.channels };
    case "SET_PIPELINE_STARTED":
      return { ...state, pipelineStarted: action.started };
    case "UPDATE_CHANNEL": {
      const updated = state.channels.map((ch) =>
        ch.channel_id === action.channel.channel_id ? action.channel : ch
      );
      return { ...state, channels: updated };
    }
    case "PHASE_CHANGED": {
      const updated = state.channels.map((ch) =>
        ch.channel_id === action.channel
          ? { ...ch, phase: action.phase }
          : ch
      );
      return { ...state, channels: updated };
    }
    default:
      return state;
  }
}

export function HomeProvider({ children }) {
  const [state, dispatch] = useReducer(homeReducer, initialState);
  return (
    <HomeContext.Provider value={{ state, dispatch }}>
      {children}
    </HomeContext.Provider>
  );
}

export function useHome() {
  const ctx = useContext(HomeContext);
  if (!ctx) throw new Error("useHome must be used within HomeProvider");
  return ctx;
}
