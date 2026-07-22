import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import { Layout } from "./components/Layout";
import { SlateDashboard } from "./views/SlateDashboard";
import { PlayerDetail } from "./views/PlayerDetail";
import { EdgeScanner } from "./views/EdgeScanner";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <SlateDashboard /> },
      { path: "player", element: <PlayerDetail /> },
      { path: "player/:playerId", element: <PlayerDetail /> },
      { path: "edges", element: <EdgeScanner /> },
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
