import { describe, it, expect } from "vitest";
import type { ParlayResponse, ReliabilityBucket } from "../../api/types";
import { COLOR } from "../../lib/tokens";
import { CORR_NEG, CORR_POS, correlationHeatmapOption, legLabel, probabilityComparisonOption } from "../parlayCharts";
import { reliabilityOption } from "../calibrationCharts";

function parlay(overrides: Partial<ParlayResponse> = {}): ParlayResponse {
  const leg = { player_id: 1, player_name: "Nikola Jokić", stat: "points", line: 25.5, side: "over", odds: -110,
    n_games: 25, mu: 25, sigma: 7, p_over: 0.47, p_hit: 0.47, implied_prob: 0.524, ev: -0.1 };
  return {
    legs: [leg, { ...leg, stat: "rebounds", line: 11.5, side: "under", p_hit: 0.3 }],
    n_games: 25, n_sims: 20000, n_joint_games: 25, correlation_fallback: false,
    joint_prob: 0.15, joint_prob_se: 0.002, independent_prob: 0.141, correlation_lift: 1.06,
    combined_decimal: 3.64, combined_american: 264, offered_american: 264, offered_is_custom: false,
    implied_prob: 0.275, fair_american: 567, ev_joint: -0.45, ev_independent: -0.49,
    correlation: { labels: ["a", "b"], matrix: [[1, 0.31], [0.31, 1]] },
    disclaimer: "Model estimate — not validated for real money.",
    ...overrides,
  };
}

describe("parlay charts", () => {
  it("labels legs compactly with side and line", () => {
    expect(legLabel(parlay().legs[1], 1)).toBe("2. Jokić REB u11.5");
  });

  it("heatmap uses a non-polarity diverging scale and all cells", () => {
    const opt = correlationHeatmapOption(parlay());
    const colors = (opt.visualMap as { inRange: { color: string[] } }).inRange.color;
    expect(colors).toEqual([CORR_NEG, COLOR.surface3, CORR_POS]);
    for (const c of colors) expect([COLOR.pos, COLOR.neg]).not.toContain(c);
    const series = (opt.series as Array<{ data: number[][] }>)[0];
    expect(series.data).toHaveLength(4);
    expect(series.data).toContainEqual([1, 0, 0.31]);
  });

  it("comparison shows joint, independent and implied", () => {
    const opt = probabilityComparisonOption(parlay());
    expect((opt.yAxis as { data: string[] }).data).toEqual(["Correlated (model)", "Independent product", "Price implies"]);
    const noPrice = probabilityComparisonOption(parlay({ implied_prob: null }));
    expect((noPrice.yAxis as { data: string[] }).data).toHaveLength(2);
  });
});

describe("reliability chart", () => {
  const b = (i: number, n: number, pred: number, real: number): ReliabilityBucket => ({
    stat_type: "all", bucket: i, bucket_low: i / 10, bucket_high: (i + 1) / 10, n,
    mean_pred: pred, realized_rate: real, calibration_gap: real - pred,
  });

  it("plots realized vs predicted with a perfect-calibration diagonal", () => {
    const opt = reliabilityOption([b(2, 10, 0.25, 0.3), b(7, 40, 0.72, 0.65)]);
    const series = opt.series as Array<{ name: string; data: unknown[] }>;
    expect(series.map((s) => s.name)).toEqual(["Perfect calibration", "Realized hit rate"]);
    const pts = series[1].data as Array<{ value: number[]; symbolSize: number }>;
    expect(pts[0].value).toEqual([0.25, 0.3, 10]);
    expect(pts[1].symbolSize).toBeGreaterThan(pts[0].symbolSize); // sized by n
  });

  it("tooltip survives an out-of-range index", () => {
    const opt = reliabilityOption([b(2, 10, 0.25, 0.3)]);
    const fmt = (opt.tooltip as { formatter: (p: unknown) => string }).formatter;
    expect(fmt({ dataIndex: 9, seriesName: "Realized hit rate" })).toBe("");
    expect(fmt({ dataIndex: 0, seriesName: "Realized hit rate" })).toContain("30.0%");
  });
});
