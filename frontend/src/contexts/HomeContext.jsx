import { createContext, useContext, useReducer } from "react";

const HomeContext = createContext(null);

const initialState = {
  channels: [],
  pipelineStarted: false,
  loading: false,
};

function homeReducer(state, action) {
  switch (action.type) {
    case "SET_CHANNELS":
      return {
        ...state,
        channels: action.channels,
        pipelineStarted: action.pipelineStarted,
      };
    case "SET_PIPELINE_STARTED":
      return { ...state, pipelineStarted: action.started };
    case "SET_LOADING":
      return { ...state, loading: action.loading };
    case "ADD_CHANNEL": {
      const exists = state.channels.some(
        (ch) => ch.channel_id === action.channel.channel_id
      );
      if (exists) return state;
      return { ...state, channels: [...state.channels, action.channel] };
    }
    case "REMOVE_CHANNEL":
      return {
        ...state,
        channels: state.channels.filter(
          (ch) => ch.channel_id !== action.channelId
        ),
      };
    case "PHASE_CHANGED": {
      const updated = state.channels.map((ch) =>
        ch.channel_id === action.channel
          ? { ...ch, phase: action.phase }
          : ch
      );
      return { ...state, channels: updated };
    }
    case "INCREMENT_ALERT": {
      const updated = state.channels.map((ch) =>
        ch.channel_id === action.channel
          ? { ...ch, alert_count: (ch.alert_count || 0) + 1 }
          : ch
      );
      return { ...state, channels: updated };
    }
    case "PIPELINE_STOPPED":
      return { ...state, pipelineStarted: false, channels: [] };
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
