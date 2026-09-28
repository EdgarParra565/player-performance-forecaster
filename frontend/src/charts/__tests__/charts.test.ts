import { describe, it, expect } from "vitest";
import type { PlayerDetail } from "../../api/types";
import { escapeHtml, terminalTheme, tooltipHtml } from "../../components/chartTheme";
import { SERIES, COLOR } from "../../lib/tokens";
import { distributionOption, hitRateOption, performanceOption } from "../playerCharts";

function detail(overrides: Partial<PlayerDetail> = {}): PlayerDetail {
  return {
    player_id: 1,
    player_name: "Test Player",
    stat_type: "points",
    n_games: 3,
    rolling_window: 2,
    kpis: { n_games: 3, mu: 20, sigma: 4, market_consensus_line: 20.5, n_books: 1, positive_ev_sides: 0 },
    series: [
      { game_date: "2026-03-01", value: 18, rolling_mean: null, opponent: "<img src=x onerror=alert(1)>", home_away: "home", result: "W" },
      { game_date: "2026-03-03", value: 22, rolling_mean: 20, opponent: "DEN", home_away: "away", result: "L" },
      { game_date: "2026-03-05", value: 21, rolling_mean: 21.5, opponent: null, home_away: "home", result: "W" },
    ],
    histogram: [{ x0: 15, x1: 20, count: 1 }, { x0: 20, x1: 25, count: 2 }],
    fitted: [{ x: 15, y: 0.5 }, { x: 20, y: 1.2 }],
    distribution: "normal",
    book_lines: [
      { book: "FanDuel", line: 20.5, over_odds: -110, under_odds: -110, p_over: 0.52, p_under: 0.48, best_side: "over", model_edge: 0.0, ev_over: 0.0, ev_under: -0.1, hit_rate: 0.66, breakeven: 0.524, is_dfs: false },
      { book: "Pick6", line: 21.5, over_odds: null, under_odds: null, p_over: 0.4, p_under: 0.6, best_side: "under", model_edge: 0.08, ev_over: null, ev_under: 0.1, hit_rate: null, breakeven: null, is_dfs: true },
    ],
    notes: [],
    last_line_scraped_utc: null,
    ...overrides,
  };
}

type Formatter = (p: unknown) => string;

describe("chart theme", () => {
  it("uses the token series palette and themed tooltip chrome", () => {
    expect(terminalTheme.color).toEqual([...SERIES]);
    expect(terminalTheme.tooltip.backgroundColor).toBe(COLOR.surface3);
  });

  it("escapes DB strings in tooltip HTML", () => {
    expect(escapeHtml(`<b>"x"&'y'</b>`)).toBe("&lt;b&gt;&quot;x&quot;&amp;&#39;y&#39;&lt;/b&gt;");
    const html = tooltipHtml("<script>", [{ label: "<i>", value: "1" }]);
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<i>");
  });
});

describe("player chart builders", () => {
  it("colors bars by hit / miss vs the consensus line", () => {
    const opt = performanceOption(detail());
    const bars = (opt.series as Array<{ data: Array<{ itemStyle: { color: string } }> }>)[0].data;
    expect(bars.map((b) => b.itemStyle.color)).toEqual([COLOR.neg, COLOR.pos, COLOR.pos]);
  });

  it("tooltip escapes the opponent and survives a bad index", () => {
    const fmt = (performanceOption(detail()).tooltip as { formatter: Formatter }).formatter;
    const html = fmt([{ dataIndex: 0 }]);
    expect(html).toContain("&lt;img");
    expect(html).not.toContain("<img");
    expect(fmt([{ dataIndex: 99 }])).toBe("");
    expect(fmt([])).toBe("");
  });

  it("marks each book line and the consensus on the distribution", () => {
    const opt = distributionOption(detail());
    const marks = (opt.series as Array<{ markLine?: { data: unknown[] } }>)[0].markLine!.data;
    expect(marks).toHaveLength(3); // 2 books + consensus
  });

  it("hit-rate chart only includes books with a hit rate", () => {
    const opt = hitRateOption(detail().book_lines);
    expect((opt.yAxis as { data: string[] }).data).toEqual(["FanDuel"]);
  });
});
