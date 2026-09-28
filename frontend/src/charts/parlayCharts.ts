// Option builders for the parlay view (shared "terminal" theme).
import type { EChartsOption } from "echarts";
import type { ParlayLegPriced, ParlayResponse } from "../api/types";
import { CHART, grid, tooltipHtml } from "../components/chartTheme";
import { fmtLine, fmtNum, fmtPct, statLabel } from "../lib/format";
import { FONT, SERIES, TYPE } from "../lib/tokens";

// Correlation is a relationship, not a value judgement: a diverging scale
// from cyan (negative) through the surface to violet (positive) — never the
// +EV/-EV green/red.
export const CORR_NEG = SERIES[1];
export const CORR_POS = SERIES[3];

// "1. Jokić PTS o25.5" — surname keeps the axis readable at 6 legs.
export function legLabel(l: Pick<ParlayLegPriced, "player_name" | "stat" | "side" | "line">, i: number): string {
  const parts = l.player_name.trim().split(/\s+/);
  const surname = parts.length > 1 ? parts.slice(1).join(" ") : parts[0];
  return `${i + 1}. ${surname} ${statLabel(l.stat)} ${l.side === "over" ? "o" : "u"}${fmtLine(l.line)}`;
}

export function correlationHeatmapOption(d: ParlayResponse): EChartsOption {
  const labels = d.legs.length === d.correlation.matrix.length
    ? d.legs.map((l, i) => legLabel(l, i))
    : d.correlation.labels.map((l, i) => `${i + 1}. ${l}`);
  const data: [number, number, number][] = [];
  d.correlation.matrix.forEach((row, i) =>
    row.forEach((v, j) => data.push([j, i, Number(v.toFixed(2))])),
  );
  return {
    grid: { left: 8, right: 16, top: 8, bottom: 56, containLabel: true },
    tooltip: {
      trigger: "item",
      formatter: (p: unknown) => {
        const v = (p as { value: [number, number, number] }).value;
        if (!v) return "";
        return tooltipHtml("Correlation", [
          { label: labels[v[1]] ?? "", value: "" },
          { label: labels[v[0]] ?? "", value: "" },
          { label: "ρ", value: fmtNum(v[2], 2), strong: true },
        ]);
      },
    },
    xAxis: {
      type: "category",
      data: labels.map((_, i) => `${i + 1}`),
      splitArea: { show: false },
      axisLine: { show: false },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLabel: { fontFamily: FONT.sans, color: CHART.muted },
      axisLine: { show: false },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      itemWidth: 10,
      itemHeight: 160,
      text: ["+1", "−1"],
      textStyle: { color: CHART.faint, fontFamily: FONT.mono, fontSize: TYPE.caption },
      inRange: { color: [CORR_NEG, CHART.surface3, CORR_POS] },
    },
    series: [
      {
        type: "heatmap",
        data,
        label: {
          show: true,
          color: CHART.fg,
          fontFamily: FONT.mono,
          fontSize: TYPE.label,
          formatter: (p: unknown) => fmtNum((p as { value: number[] }).value[2], 2),
        },
        itemStyle: { borderColor: CHART.surface1, borderWidth: 2, borderRadius: 4 },
        emphasis: { itemStyle: { borderColor: CHART.accent } },
      },
    ],
  };
}

// Correlated joint vs naive independent product vs the price's implied prob.
export function probabilityComparisonOption(d: ParlayResponse): EChartsOption {
  const rows = [
    { name: "Correlated (model)", value: d.joint_prob, color: CHART.accent },
    { name: "Independent product", value: d.independent_prob, color: CHART.muted },
    ...(d.implied_prob != null
      ? [{ name: "Price implies", value: d.implied_prob, color: CHART.warn }]
      : []),
  ];
  const max = Math.max(...rows.map((r) => r.value)) * 1.25;
  return {
    grid: { ...grid({ right: 64 }), top: 8, bottom: 8 },
    tooltip: {
      trigger: "item",
      formatter: (p: unknown) => {
        const r = rows[(p as { dataIndex: number }).dataIndex];
        return r ? tooltipHtml(r.name, [{ label: "P(all legs hit)", value: fmtPct(r.value), strong: true, color: r.color }]) : "";
      },
    },
    xAxis: { type: "value", min: 0, max, axisLabel: { formatter: (v: number) => fmtPct(v, 0) } },
    yAxis: {
      type: "category",
      data: rows.map((r) => r.name),
      inverse: true,
      axisLabel: { fontFamily: FONT.sans, color: CHART.muted },
      axisLine: { show: false },
    },
    series: [
      {
        type: "bar",
        barMaxWidth: 18,
        data: rows.map((r) => ({ value: r.value, itemStyle: { color: r.color, borderRadius: [0, 3, 3, 0] } })),
        label: {
          show: true,
          position: "right",
          color: CHART.fg,
          fontFamily: FONT.mono,
          fontSize: TYPE.label,
          formatter: (p: unknown) => fmtPct((p as { value: number }).value),
        },
      },
    ],
  };
}
