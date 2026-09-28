// ECharts option builders for Player Detail. Pure functions of API data so
// they're memoisable and unit-testable; all styling comes from the shared
// "terminal" theme + tokens.
import type { EChartsOption, MarkLineComponentOption } from "echarts";
import type { BookLineRow, PlayerDetail } from "../api/types";
import { CHART, grid, refLabel, tooltipHtml, type TooltipRow } from "../components/chartTheme";
import { fmtAxisDate, fmtDateShort, fmtLine, fmtNum, fmtPct, statLabel } from "../lib/format";
import { FONT } from "../lib/tokens";

type MarkLineData = NonNullable<MarkLineComponentOption["data"]>;

// Recent-N bars, colored by hit (>= consensus line) / miss, rolling mean line.
export function performanceOption(d: PlayerDetail): EChartsOption {
  const consensus = d.kpis.market_consensus_line;
  const label = statLabel(d.stat_type);
  return {
    grid: grid({ legend: true }),
    legend: {
      top: 0,
      left: 0,
      data: [
        { name: consensus != null ? `${label} (hit / miss vs book mean)` : label },
        { name: "Rolling mean" },
      ],
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const i = (params as Array<{ dataIndex: number }>)[0]?.dataIndex;
        const p = i === undefined ? undefined : d.series[i];
        if (!p) return "";
        const rows: TooltipRow[] = [
          { label, value: fmtNum(p.value, 0), strong: true, color: CHART.accent },
          { label: "Rolling mean", value: fmtNum(p.rolling_mean), color: CHART.warn },
        ];
        if (consensus != null) rows.push({ label: "Book mean", value: fmtLine(consensus), color: CHART.muted });
        return tooltipHtml(fmtDateShort(p.game_date?.slice(0, 10)), rows, p.opponent ? `vs ${p.opponent}` : undefined);
      },
    },
    xAxis: { type: "category", data: d.series.map((p) => fmtAxisDate(p.game_date)), boundaryGap: true },
    yAxis: { type: "value", scale: true },
    series: [
      {
        name: consensus != null ? `${label} (hit / miss vs book mean)` : label,
        type: "bar",
        barMaxWidth: 26,
        itemStyle: { color: CHART.accent },
        data: d.series.map((p) => ({
          value: p.value,
          itemStyle: {
            color:
              consensus == null ? CHART.accent : p.value >= consensus ? CHART.pos : CHART.neg,
            opacity: 0.72,
          },
        })),
        markLine:
          consensus != null
            ? {
                symbol: "none",
                silent: true,
                lineStyle: { color: CHART.muted, type: "dashed", width: 1 },
                label: refLabel(`book mean ${fmtLine(consensus)}`, CHART.muted, "insideEndTop"),
                data: [{ yAxis: consensus }],
              }
            : undefined,
      },
      {
        name: "Rolling mean",
        type: "line",
        data: d.series.map((p) => p.rolling_mean),
        smooth: true,
        showSymbol: false,
        lineStyle: { color: CHART.warn, width: 2 },
        itemStyle: { color: CHART.warn },
        z: 3,
      },
    ],
  };
}

// Histogram + fitted normal, with the consensus (mean) and each book's line.
export function distributionOption(d: PlayerDetail): EChartsOption {
  const consensus = d.kpis.market_consensus_line;
  const marks: MarkLineData = d.book_lines
    .filter((b) => b.line != null)
    .map((b) => ({
      xAxis: b.line as number,
      name: b.book,
      lineStyle: { color: CHART.info, type: "dotted" as const, width: 1, opacity: 0.8 },
      label: { show: false },
    }));
  if (consensus != null) {
    marks.push({
      xAxis: consensus,
      lineStyle: { color: CHART.warn, type: "dashed" as const, width: 1.5 },
      label: refLabel(`book mean ${fmtLine(consensus)}`, CHART.warn, "end"),
    });
  }
  const label = statLabel(d.stat_type);
  return {
    grid: grid({ legend: true }),
    legend: { top: 0, left: 0, data: ["Games", "Fitted normal"] },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const arr = params as Array<{ seriesName: string; value: [number, number] }>;
        const bar = arr.find((p) => p.seriesName === "Games");
        if (!bar) return "";
        const bin = d.histogram.find((b) => (b.x0 + b.x1) / 2 === bar.value[0]);
        const title = bin ? `${label} ${fmtNum(bin.x0)}–${fmtNum(bin.x1)}` : label;
        return tooltipHtml(title, [{ label: "Games", value: fmtNum(bar.value[1], 0), color: CHART.lineStrong, strong: true }]);
      },
    },
    xAxis: {
      type: "value",
      scale: true,
      axisLabel: { formatter: (v: number) => fmtNum(v, 0) },
      splitLine: { show: false },
    },
    yAxis: { type: "value", minInterval: 1 },
    series: [
      {
        name: "Games",
        type: "bar",
        data: d.histogram.map((b) => [(b.x0 + b.x1) / 2, b.count]),
        itemStyle: { color: CHART.surface4, borderColor: CHART.lineStrong, borderWidth: 1, borderRadius: [3, 3, 0, 0] },
        barWidth: "92%",
        markLine: marks.length ? { symbol: "none", silent: true, data: marks } : undefined,
      },
      {
        name: "Fitted normal",
        type: "line",
        data: d.fitted.map((p) => [p.x, p.y]),
        smooth: true,
        showSymbol: false,
        lineStyle: { color: CHART.accent, width: 2 },
        itemStyle: { color: CHART.accent },
        areaStyle: { color: CHART.accent, opacity: 0.08 },
        z: 3,
      },
    ],
  };
}

// Historical over-rate vs each book's line; 50% reference.
export function hitRateOption(books: BookLineRow[]): EChartsOption {
  const rows = books.filter((b) => b.hit_rate != null);
  return {
    grid: { ...grid({ right: 48 }), top: 28, bottom: 8 },
    tooltip: {
      trigger: "item",
      formatter: (p: unknown) => {
        const item = p as { dataIndex: number };
        const b = rows[item.dataIndex];
        if (!b) return "";
        return tooltipHtml(b.book, [
          { label: "Over rate", value: fmtPct(b.hit_rate), strong: true },
          { label: "Line", value: fmtLine(b.line) },
        ]);
      },
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { formatter: (v: number) => fmtPct(v, 0) },
    },
    yAxis: {
      type: "category",
      data: rows.map((b) => b.book),
      axisLabel: { fontFamily: FONT.sans, color: CHART.muted },
      axisLine: { show: false },
    },
    series: [
      {
        type: "bar",
        barMaxWidth: 16,
        data: rows.map((b) => ({
          value: b.hit_rate as number,
          itemStyle: {
            color: (b.hit_rate as number) >= 0.5 ? CHART.pos : CHART.neg,
            opacity: 0.72,
            borderRadius: [0, 3, 3, 0],
          },
        })),
        label: {
          show: true,
          position: "right",
          color: CHART.muted,
          fontSize: 11,
          formatter: (p: unknown) => fmtPct((p as { value: number }).value),
        },
        markLine: {
          symbol: "none",
          silent: true,
          data: [{ xAxis: 0.5 }],
          lineStyle: { color: CHART.muted, type: "dashed", width: 1 },
          label: refLabel("50%", CHART.muted, "end"),
        },
      },
    ],
  };
}
