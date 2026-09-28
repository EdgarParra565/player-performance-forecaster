import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StarButton } from "../StarButton";
import { DataTable } from "../DataTable";
import { _resetWatchlistCacheForTests, getWatchlist } from "../../lib/watchlist";

beforeEach(() => {
  window.localStorage.clear();
  _resetWatchlistCacheForTests();
});

describe("StarButton", () => {
  it("toggles pressed state and the store", () => {
    render(<StarButton item={{ kind: "prop", player_name: "LeBron James", stat: "points" }} />);
    const btn = screen.getByRole("button", { name: "Add LeBron James PTS to watchlist" });
    expect(btn).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(btn);
    expect(screen.getByRole("button", { name: "Remove LeBron James PTS from watchlist" })).toHaveAttribute("aria-pressed", "true");
    expect(getWatchlist()).toHaveLength(1);
  });

  it("does not trigger the row's click / keyboard navigation", () => {
    const onRow = vi.fn();
    render(
      <DataTable
        columns={[{ key: "p", header: "Player", render: () => <StarButton item={{ kind: "player", player_name: "A" }} /> }]}
        rows={[{ id: 1 }]}
        rowKey={() => "1"}
        onRowClick={onRow}
      />,
    );
    const star = screen.getByRole("button", { name: /Add A/ });
    fireEvent.click(star);
    fireEvent.keyDown(star, { key: "Enter" });
    expect(onRow).not.toHaveBeenCalled();
    expect(getWatchlist()).toHaveLength(1);
  });
});

vi.mock("../../api/hooks", () => ({
  useHealth: () => ({ data: { db_exists: true, version: "0.1.0", access_code_required: false, freshest_scrape_utc: null, last_game_date: null } }),
  useEdges: () => ({ data: { rows: [] }, isLoading: false, isError: false }),
}));

import { Layout } from "../Layout";
import { Watchlist } from "../../views/Watchlist";

describe("Layout mobile navigation", () => {
  it("opens and closes the drawer; bottom bar has primary views + More", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter><Layout /></MemoryRouter>
      </QueryClientProvider>,
    );
    const quick = screen.getByRole("navigation", { name: "Quick navigation" });
    expect(quick).toHaveTextContent("Slate");
    expect(quick).toHaveTextContent("Watch");
    fireEvent.click(screen.getByRole("button", { name: "More" }));
    const dialog = screen.getByRole("dialog", { name: "Navigation" });
    expect(dialog).toHaveTextContent("Paper Trades");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Navigation" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open menu" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Close menu" })[0]);
    expect(screen.queryByRole("dialog", { name: "Navigation" })).toBeNull();
  });
});

describe("Watchlist view", () => {
  it("empty state explains how to star", () => {
    render(<MemoryRouter><Watchlist /></MemoryRouter>);
    expect(screen.getByText("Nothing starred yet.")).toBeInTheDocument();
  });

  it("shared link offers add / replace and shows 'not on the board'", () => {
    render(
      <MemoryRouter initialEntries={["/watchlist?w=q~2544~points~LeBron%20James|p~-~-~Nikola%20Joki%C4%87"]}>
        <Watchlist />
      </MemoryRouter>,
    );
    expect(screen.getByRole("region", { name: "Shared watchlist" })).toHaveTextContent("2 items");
    fireEvent.click(screen.getByRole("button", { name: "Add to mine" }));
    expect(getWatchlist()).toHaveLength(2);
    expect(screen.queryByRole("region", { name: "Shared watchlist" })).toBeNull();
    expect(screen.getAllByText("Not on the board")).toHaveLength(2);
    expect(screen.getByRole("link", { name: "LeBron James" })).toHaveAttribute("href", "/player/2544?stat=points");
  });
});
