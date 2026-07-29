import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CrossBookTable } from "../CrossBookTable";
import type { CrossBookRow } from "../../api/types";

function row(overrides: Partial<CrossBookRow>): CrossBookRow {
  return {
    player_name: "LeBron James",
    stat_type: "points",
    n_books: 2,
    line_min: 17.5,
    line_max: 20.5,
    line_gap: 3.0,
    best_over_book: "Underdog",
    best_under_book: "PrizePicks",
    consensus_mean: 19.0,
    p_over_at_line_min: 0.66,
    p_over_at_line_max: 0.4,
    middle_size: 3.0,
    opportunity_type: "middle_candidate",
    model_mu: 19.0,
    model_sigma: 6.3,
    ...overrides,
  };
}

describe("CrossBookTable", () => {
  it("renders a row with the player and best books", () => {
    render(<CrossBookTable rows={[row({})]} />);
    expect(screen.getByText("LeBron James")).toBeInTheDocument();
    expect(screen.getByText("Underdog")).toBeInTheDocument();
    expect(screen.getByText("PrizePicks")).toBeInTheDocument();
  });

  it("labels middles as candidates ('MIDDLE?'), never guaranteed", () => {
    render(<CrossBookTable rows={[row({ opportunity_type: "middle_candidate" })]} />);
    expect(screen.getByText("MIDDLE?")).toBeInTheDocument();
    expect(screen.queryByText(/guaranteed/i)).toBeNull();
  });

  it("shows SHOP for a plain line-gap opportunity", () => {
    render(<CrossBookTable rows={[row({ opportunity_type: "line_gap" })]} />);
    expect(screen.getByText("SHOP")).toBeInTheDocument();
  });

  it("sorts by gap and fires onRowClick", () => {
    const onRowClick = vi.fn();
    const rows = [
      row({ player_name: "Small Gap", line_gap: 1.0 }),
      row({ player_name: "Big Gap", line_gap: 5.0 }),
    ];
    render(<CrossBookTable rows={rows} onRowClick={onRowClick} />);
    // initialSort is line_gap desc -> "Big Gap" first data row.
    const dataRows = screen.getAllByRole("row").slice(1);
    expect(dataRows[0].textContent).toContain("Big Gap");

    fireEvent.click(screen.getByText("Big Gap"));
    expect(onRowClick).toHaveBeenCalledTimes(1);
    expect(onRowClick.mock.calls[0][0].player_name).toBe("Big Gap");
  });

  it("renders nothing but headers for an empty row set", () => {
    render(<CrossBookTable rows={[]} />);
    // Header row only.
    expect(screen.getAllByRole("row")).toHaveLength(1);
  });
});
