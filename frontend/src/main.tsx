import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
// Self-hosted variable fonts (no third-party CDN): Inter for UI with optical
// sizing, JetBrains Mono for tabular data.
import "@fontsource-variable/inter/opsz.css";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";
import { ApiError } from "./api/client";
import { Layout, RouteError } from "./components/Layout";
import { SlateDashboard } from "./views/SlateDashboard";
import { PlayerDetail } from "./views/PlayerDetail";
import { EdgeScanner } from "./views/EdgeScanner";
import { CrossBook } from "./views/CrossBook";
import { TeamCharts } from "./views/TeamCharts";
import { Parlay } from "./views/Parlay";
import { PaperTrades } from "./views/PaperTrades";
import { Watchlist } from "./views/Watchlist";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Matches the API's Cache-Control max-age=60 on data endpoints.
      staleTime: 60_000,
      // 4xx is deterministic (bad input / not found) — retrying only delays
      // the error. Retry once on network errors and 5xx.
      retry: (failureCount, error) =>
        !(error instanceof ApiError && error.status < 500) && failureCount < 1,
      refetchOnWindowFocus: false,
    },
  },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <SlateDashboard /> },
      { path: "player", element: <PlayerDetail /> },
      { path: "player/:playerId", element: <PlayerDetail /> },
      { path: "edges", element: <EdgeScanner /> },
      { path: "cross-book", element: <CrossBook /> },
      { path: "teams", element: <TeamCharts /> },
      { path: "parlay", element: <Parlay /> },
      { path: "paper-trades", element: <PaperTrades /> },
      { path: "watchlist", element: <Watchlist /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
