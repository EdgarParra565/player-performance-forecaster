// Shared ECharts styling so every chart reads as one system. Colors mirror the
// CSS tokens in index.css (kept in sync by hand — they are the design source).
export const CHART = {
  base: "#08090b",
  panel: "#0e1014",
  line: "#20242c",
  lineStrong: "#2c313b",
  fg: "#e7eaf0",
  muted: "#8a92a1",
  faint: "#59606d",
  pos: "#12e29a",
  neg: "#ff495f",
  warn: "#f4b740",
  info: "#58a6ff",
  fontMono: "'JetBrains Mono', ui-monospace, Menlo, monospace",
};

// Axis config shared across the terminal's line/bar charts.
export function axis(opts: { name?: string } = {}) {
  return {
    name: opts.name,
    nameTextStyle: { color: CHART.faint, fontSize: 10 },
    axisLine: { lineStyle: { color: CHART.lineStrong } },
    axisTick: { show: false },
    axisLabel: {
      color: CHART.muted,
      fontFamily: CHART.fontMono,
      fontSize: 10,
    },
    splitLine: { lineStyle: { color: CHART.line, type: "dashed" as const } },
  };
}

export const tooltipStyle = {
  backgroundColor: "#14171d",
  borderColor: CHART.lineStrong,
  borderWidth: 1,
  padding: [6, 10] as [number, number],
  textStyle: { color: CHART.fg, fontFamily: CHART.fontMono, fontSize: 11 },
  extraCssText: "border-radius:4px;box-shadow:0 6px 24px rgba(0,0,0,0.5);",
};

export const gridTight = {
  left: 40,
  right: 16,
  top: 28,
  bottom: 28,
  containLabel: false,
};
