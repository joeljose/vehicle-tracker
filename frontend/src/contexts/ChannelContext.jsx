import { createContext, useContext, useReducer } from "react";

const ChannelContext = createContext(null);

const initialState = {
  channelId: null,
  phase: "setup",
  source: "",
  pipelineStarted: false,
  stats: null,
};

function channelReducer(state, action) {
  switch (action.type) {
    case "SET_CHANNEL":
      return {
        ...state,
        channelId: action.channelId,
        phase: action.phase ?? "setup",
        source: action.source ?? "",
      };
    case "SET_PHASE":
      return { ...state, phase: action.phase };
    case "SET_PIPELINE_STARTED":
      return { ...state, pipelineStarted: action.started };
    case "SET_STATS":
      return { ...state, stats: action.stats };
    default:
      return state;
  }
}

export function ChannelProvider({ channelId, children }) {
  const [state, dispatch] = useReducer(channelReducer, {
    ...initialState,
    channelId,
  });
  return (
    <ChannelContext.Provider value={{ state, dispatch }}>
      {children}
    </ChannelContext.Provider>
  );
}

export function useChannel() {
  const ctx = useContext(ChannelContext);
  if (!ctx) throw new Error("useChannel must be used within ChannelProvider");
  return ctx;
}
