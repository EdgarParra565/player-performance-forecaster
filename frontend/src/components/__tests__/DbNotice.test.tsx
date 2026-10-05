import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, apiGet } from "../../api/client";
import { DbNotice } from "../DbNotice";
import { errorMessage, isDbNotMounted } from "../Loading";

afterEach(() => vi.restoreAllMocks());

describe("database-state errors", () => {
  it("client keeps the API's machine-readable code", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "database file not mounted: …", code: "db_not_mounted" }), { status: 503 }),
    );
    const err = (await apiGet("/meta").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("db_not_mounted");
    expect(isDbNotMounted(err)).toBe(true);
    expect(errorMessage(err, "Edge scan failed.")).toBe("Database file not mounted — see the notice at the top.");
  });

  it("not-mounted notice gives the exact mount commands", () => {
    render(<DbNotice error={new ApiError(503, "x", "db_not_mounted")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Database file not mounted");
    expect(screen.getByText(/docker compose up flagship/)).toBeInTheDocument();
    expect(screen.getByText(/-v "\$\(pwd\)\/data\/database:\/data:ro"/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("not an offseason slate");
  });

  it("invalid DB gets its own copy; other errors show nothing", () => {
    const { rerender } = render(<DbNotice error={new ApiError(503, "x", "db_invalid")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Database unavailable");
    rerender(<DbNotice error={new ApiError(400, "bad stat")} />);
    expect(screen.queryByRole("alert")).toBeNull();
    rerender(<DbNotice error={null} />);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

const hookState: { kpisError: unknown; edges: unknown } = { kpisError: null, edges: undefined };
vi.mock("../../api/hooks", () => ({
  useSlateKpis: () => ({ data: undefined, isLoading: false, isError: !!hookState.kpisError, error: hookState.kpisError }),
  useEdges: () => ({ data: hookState.edges, isLoading: false, isError: false, error: null }),
  useRecentGames: () => ({ data: { rows: [] }, isLoading: false, isError: false, error: null }),
  useHealth: () => ({ data: undefined, error: new ApiError(503, "x", "db_not_mounted") }),
}));

import { SlateDashboard } from "../../views/SlateDashboard";
import { Layout } from "../Layout";

describe("not-mounted vs offseason copy", () => {
  it("offseason (DB present, no lines) shows the offseason empty state, not the mount notice", () => {
    hookState.kpisError = null;
    hookState.edges = { rows: [] };
    render(<MemoryRouter><SlateDashboard /></MemoryRouter>);
    expect(screen.getByText("No scored edges right now.")).toBeInTheDocument();
    expect(screen.getByText(/Outside the season/)).toBeInTheDocument();
    expect(screen.queryByText(/not mounted/i)).toBeNull();
  });

  it("missing DB shows the mount notice app-wide and a 'No database' pill, never 'Offseason'", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter><Layout /></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(screen.getByText("Database file not mounted")).toBeInTheDocument();
    expect(screen.getByText("No database")).toBeInTheDocument();
    expect(screen.queryByText("Offseason")).toBeNull();
  });
});
