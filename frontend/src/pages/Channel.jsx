import { useParams } from "react-router-dom";
import { ChannelProvider } from "../contexts/ChannelContext";
import { AlertProvider } from "../contexts/AlertContext";

function ChannelContent() {
  const { id } = useParams();

  return (
    <div className="min-h-screen p-6">
      <h1 className="text-2xl font-bold mb-4">Channel {id}</h1>
      <p className="text-text-secondary">Phase-driven layout will go here.</p>
    </div>
  );
}

export default function Channel() {
  const { id } = useParams();

  return (
    <ChannelProvider channelId={Number(id)}>
      <AlertProvider>
        <ChannelContent />
      </AlertProvider>
    </ChannelProvider>
  );
}
