// ONE ECharts theme for every chart in the terminal. Registered once (in the
// lazy EChartImpl chunk) as "terminal"; views only describe data + series and
// inherit axes, gridlines, tooltip chrome, legend and palette from here.
import { COLOR, FONT, SERIES, TYPE } from "../lib/tokens";

export const THEME_NAME = "terminal";

// Chart-facing color aliases (canvas can't read CSS vars).
export const CHART = {
  ...COLOR,
  gridline: "rgba(255,255,255,0.05)",
  series: SERIES,
} as const;

const axisCommon = {
  axisLine: { show: true, lineStyle: { color: COLOR.lineStrong } },
  axisTick: { show: false },
  axisLabel: {
    color: COLOR.faint,
    fontFamily: FONT.mono,
    fontSize: TYPE.caption,
    margin: 10,
    // Narrow screens: drop labels that would overlap instead of colliding.
    hideOverlap: true,
  },
  splitLine: { show: true, lineStyle: { color: "rgba(255,255,255,0.05)", type: "solid" } },
  splitArea: { show: false },
  nameTextStyle: {
    color: COLOR.faint,
    fontFamily: FONT.sans,
    fontSize: TYPE.caption,
    fontWeight: 500,
  },
  nameGap: 12,
};

export const terminalTheme = {
  color: [...SERIES],
  backgroundColor: "transparent",
  textStyle: { fontFamily: FONT.sans, color: COLOR.muted, fontSize: TYPE.label },
  title: { textStyle: { color: COLOR.fg, fontFamily: FONT.sans } },
  legend: {
    icon: "roundRect",
    itemWidth: 10,
    itemHeight: 4,
    itemGap: 16,
    textStyle: { color: COLOR.muted, fontFamily: FONT.sans, fontSize: TYPE.caption },
    inactiveColor: COLOR.faint,
    pageTextStyle: { color: COLOR.muted },
    pageIconColor: COLOR.muted,
    pageIconInactiveColor: COLOR.line,
  },
  tooltip: {
    backgroundColor: COLOR.surface3,
    borderColor: COLOR.lineStrong,
    borderWidth: 1,
    padding: [8, 12],
    textStyle: { color: COLOR.fg, fontFamily: FONT.sans, fontSize: TYPE.label },
    extraCssText:
      "border-radius:8px;box-shadow:0 12px 32px -8px rgba(0,0,0,.7);backdrop-filter:blur(6px);",
    axisPointer: {
      lineStyle: { color: COLOR.lineStrong, width: 1 },
      crossStyle: { color: COLOR.lineStrong },
      shadowStyle: { color: "rgba(255,255,255,0.03)" },
    },
  },
  categoryAxis: { ...axisCommon, splitLine: { show: false } },
  valueAxis: { ...axisCommon, axisLine: { show: false } },
  timeAxis: axisCommon,
  line: { symbolSize: 5, smooth: false, lineStyle: { width: 2 } },
  bar: { itemStyle: { borderRadius: [3, 3, 0, 0] } },
  markLine: { symbol: "none", silent: true },
};

// Default grid: containLabel keeps axis labels from colliding with the plot
// edge; top leaves room for a legend when a chart has one.
export function grid(opts: { legend?: boolean; left?: number; right?: number } = {}) {
  return {
    left: opts.left ?? 8,
    right: opts.right ?? 16,
    top: opts.legend ? 40 : 20,
    bottom: 8,
    containLabel: true,
  };
}

export type RefLabelPosition =
  | "start"
  | "middle"
  | "end"
  | "insideStartTop"
  | "insideStartBottom"
  | "insideMiddleTop"
  | "insideMiddleBottom"
  | "insideEndTop"
  | "insideEndBottom";

// Reference-line label (book mean, implied total, ...).
export function refLabel(text: string, color: string, position: RefLabelPosition = "insideEndTop") {
  return {
    show: true,
    position,
    formatter: text,
    color,
    fontFamily: FONT.mono,
    fontSize: TYPE.caption,
    backgroundColor: COLOR.base,
    padding: [2, 6],
    borderRadius: 4,
  };
}

// --- Tooltip HTML -------------------------------------------------------------

// ECharts renders tooltip formatters as innerHTML, and labels here come from
// scraped/DB strings (player, book, opponent) — always escape.
export function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export interface TooltipRow {
  label: string;
  value: string;
  color?: string;
  strong?: boolean;
}

export function tooltipHtml(title: string, rows: TooltipRow[], subtitle?: string): string {
  const head =
    `<div style="font-weight:600;color:${COLOR.fg};margin-bottom:${subtitle ? 0 : 6}px">` +
    `${escapeHtml(title)}</div>` +
    (subtitle
      ? `<div style="color:${COLOR.faint};font-size:${TYPE.caption}px;margin-bottom:6px">${escapeHtml(subtitle)}</div>`
      : "");
  const body = rows
    .map((r) => {
      const dot = r.color
        ? `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${r.color};margin-right:8px"></span>`
        : "";
      return (
        `<div style="display:flex;align-items:center;justify-content:space-between;gap:24px;line-height:20px">` +
        `<span style="color:${COLOR.muted}">${dot}${escapeHtml(r.label)}</span>` +
        `<span style="font-family:${FONT.mono};font-variant-numeric:tabular-nums;color:${COLOR.fg};font-weight:${r.strong ? 600 : 400}">${escapeHtml(r.value)}</span>` +
        `</div>`
      );
    })
    .join("");
  return head + body;
}
