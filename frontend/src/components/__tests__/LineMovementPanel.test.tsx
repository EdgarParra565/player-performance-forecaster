import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { LineMovementResponse } from "../../api/types";
import { alignSeries, clampIndex, LineMovementPanel } from "../LineMovementPanel";

// Canvas charts don't run in jsdom; the panel's state logic is what's tested.
vi.mock("../EChart", () => ({ EChart: () => <div data-testid="chart" /> }));

const hookState: { data: LineMovementResponse | undefined } = { data: undefined };
vi.mock("../../api/hooks", () => ({
  useLineMovement: () => ({ data: hookState.data, isLoading: false, isError: false, error: null }),
}));

function response(n: number): LineMovementResponse {
  const timestamps = Array.from({ length: n }, (_, i) => `2026-09-2${i}T12:00:00+00:00`);
  return {
    player_id: 1,
    stat_type: "points",
    timestamps,
    series: [
      {
        book: "FanDuel",
        points: timestamps.map((ts, i) => ({ ts, line: 20 + i * 0.5, over_odds: -110, under_odds: -110 })),
        open_line: 20,
        close_line: 20 + (n - 1) * 0.5,
        line_delta: (n - 1) * 0.5,
      },
    ],
    n_books: 1,
    n_snapshots: n,
    last_snapshot_utc: timestamps[n - 1],
  };
}

describe("line movement helpers", () => {
  it("clamps a stale index into range", () => {
    expect(clampIndex(4, 2)).toBe(1);
    expect(clampIndex(-3, 2)).toBe(0);
    expect(clampIndex(5, 0)).toBe(0);
  });

  it("forward-fills and reveals up to the scrubber", () => {
    const data = response(3);
    data.series[0].points.splice(1, 1); // gap at t1 -> forward-filled
    expect(alignSeries(data, 2)[0]).toEqual([20, 20, 21]);
    expect(alignSeries(data, 0)[0]).toEqual([20, null, null]);
  });
});

describe("LineMovementPanel", () => {
  beforeEach(() => {
    hookState.data = undefined;
  });

  it("does not crash when the series shrinks under a scrubbed index", () => {
    hookState.data = response(5);
    const { rerender } = render(<LineMovementPanel playerId={1} stat="points" />);
    const slider = screen.getByRole("slider", { name: "Snapshot" });
    fireEvent.change(slider, { target: { value: "4" } });

    // New player/stat with a single snapshot: previously timestamps[4] was
    // undefined for one render and the page crashed.
    hookState.data = response(1);
    rerender(<LineMovementPanel playerId={2} stat="points" />);
    expect(screen.getByRole("slider", { name: "Snapshot" })).toHaveValue("0");
    expect(screen.getByText(/One snapshot so far/)).toBeInTheDocument();
  });

  it("renders the empty state without data", () => {
    render(<LineMovementPanel playerId={1} stat="points" />);
    expect(screen.getByText(/No stored line movement/)).toBeInTheDocument();
  });
});
