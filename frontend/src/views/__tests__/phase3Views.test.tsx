import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { CalibrationResponse, PaperTradesResponse } from "../../api/types";

vi.mock("../../components/EChart", () => ({
  EChart: ({ ariaLabel }: { ariaLabel?: string }) => <div data-testid="chart">{ariaLabel}</div>,
}));

const state: { trades?: PaperTradesResponse; cal?: CalibrationResponse } = {};
vi.mock("../../api/hooks", () => ({
  usePaperTrades: () => ({ data: state.trades, isLoading: false, isError: false }),
  useCalibration: () => ({ data: state.cal, isLoading: false, isError: false, isPlaceholderData: false }),
  useMeta: () => ({ data: { stats: ["points", "rebounds"] } }),
  useParlayPrice: () => ({ data: undefined, isLoading: false, isError: false, isFetching: false }),
  usePlayerDetail: () => ({ data: undefined }),
  usePlayerSearch: () => ({ data: undefined, isFetching: false, isPlaceholderData: false }),
}));

import { PaperTrades } from "../PaperTrades";
import { Parlay } from "../Parlay";

const emptySummary = {
  total: 0, pending: 0, won: 0, lost: 0, push: 0, void: 0, win_rate: null,
  n_clv: 0, mean_clv: null, positive_clv_rate: null, est_units: null,
};
const emptyCal: CalibrationResponse = {
  source: "bet_log", stat: "all", n_buckets: 10, n_settled: 0,
  stats_available: [], reliability: [], brier: null, brier_by_stat: [],
};

function row(id: number, status: string, clv: number | null) {
  return {
    log_id: id, created_at_utc: "2026-10-20 18:00:00", game_date: "2026-10-21", player_id: 1,
    player_name: `Player ${id}`, stat_type: "points", book: "fanduel", line: 22.5, side: "over",
    model_prob: 0.6, implied_prob: 0.524, edge: 0.076, model_mode: "full", stake_units: 1,
    status, settled_at_utc: null, actual_value: 25, clv_delta: clv,
    est_profit_units: status === "won" ? 0.91 : -1,
  };
}

describe("PaperTrades view", () => {
  beforeEach(() => {
    state.trades = undefined;
    state.cal = emptyCal;
  });

  it("empty bet_log points at the bet_slip exporter", () => {
    state.trades = { status_filter: "all", rows: [], summary: emptySummary };
    render(<MemoryRouter><PaperTrades /></MemoryRouter>);
    expect(screen.getByText("No paper trades yet.")).toBeInTheDocument();
    expect(screen.getByText(/python -m nba_model.evaluation.bet_slip/)).toBeInTheDocument();
    expect(screen.getByText(/not validated for real money/)).toBeInTheDocument();
    expect(screen.getByText("No settled paper trades to calibrate yet.")).toBeInTheDocument();
  });

  it("splits settled vs pending and shows CLV", () => {
    state.trades = {
      status_filter: "all",
      rows: [row(1, "won", 0.5), row(2, "lost", -1), row(3, "pending", null)],
      summary: { ...emptySummary, total: 3, pending: 1, won: 1, lost: 1, win_rate: 0.5, n_clv: 2, mean_clv: -0.25 },
    };
    render(<MemoryRouter><PaperTrades /></MemoryRouter>);
    expect(screen.getByRole("heading", { name: "Settled" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Pending" })).toBeInTheDocument();
    expect(screen.getByText("+0.5")).toBeInTheDocument(); // CLV column
    expect(screen.getByText("won")).toBeInTheDocument();
    expect(screen.getByText("Player 3")).toBeInTheDocument();
    expect(screen.getByText("1–1")).toBeInTheDocument();
  });

  it("renders the reliability chart when settled rows exist", () => {
    state.trades = { status_filter: "all", rows: [], summary: emptySummary };
    state.cal = {
      ...emptyCal, source: "predictions", n_settled: 10, stats_available: ["points"],
      reliability: [{ stat_type: "all", bucket: 5, bucket_low: 0.5, bucket_high: 0.6, n: 10,
        mean_pred: 0.55, realized_rate: 0.6, calibration_gap: 0.05 }],
      brier: { stat_type: "all", n: 10, brier_score: 0.21, mean_pred: 0.55, realized_rate: 0.6 },
      brier_by_stat: [],
    };
    render(<MemoryRouter><PaperTrades /></MemoryRouter>);
    expect(screen.getByTestId("chart")).toHaveTextContent("Reliability curve for 10 settled predictions rows");
    expect(screen.getByText("0.2100")).toBeInTheDocument();
  });
});

describe("Parlay view", () => {
  it("starts empty with the honest-labeling banner", () => {
    render(<MemoryRouter initialEntries={["/parlay"]}><Parlay /></MemoryRouter>);
    expect(screen.getByText(/not validated for real money/)).toBeInTheDocument();
    expect(screen.getByText("Add at least two legs to price a parlay.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Add leg/ })).toBeDisabled();
  });

  it("restores legs from the URL", () => {
    render(
      <MemoryRouter
        initialEntries={["/parlay?legs=1~points~22.5~over~-110~Alpha%20Guard|2~rebounds~7.5~under~-115~Beta%20Forward"]}
      >
        <Parlay />
      </MemoryRouter>,
    );
    expect(screen.getByText("Alpha Guard")).toBeInTheDocument();
    expect(screen.getByText("Beta Forward")).toBeInTheDocument();
    expect(screen.getByText("2 / 6 legs")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove leg 2" })).toBeInTheDocument();
  });
});
