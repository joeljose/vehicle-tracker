import { Routes, Route } from "react-router-dom";
import { HomeProvider } from "./contexts/HomeContext";
import Home from "./pages/Home";
import Channel from "./pages/Channel";

export default function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <HomeProvider>
            <Home />
          </HomeProvider>
        }
      />
      <Route path="/channel/:id" element={<Channel />} />
    </Routes>
  );
}
