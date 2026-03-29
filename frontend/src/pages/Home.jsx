import { useHome } from "../contexts/HomeContext";

export default function Home() {
  const { state } = useHome();

  return (
    <div className="min-h-screen p-6">
      <h1 className="text-2xl font-bold mb-4">Vehicle Tracker</h1>
      <p className="text-text-secondary">
        Pipeline: {state.pipelineStarted ? "Running" : "Stopped"}
      </p>
      <p className="text-text-secondary mt-2">
        Channels: {state.channels.length}
      </p>
    </div>
  );
}
