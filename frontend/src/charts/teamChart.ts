import type { EChartsOption, MarkLineComponentOption } from "echarts";
import type { TeamChartResponse } from "../api/types";
import { CHART, grid, refLabel, tooltipHtml } from "../components/chartTheme";
import { fmtAxisDate, fmtDateShort, fmtInt, fmtNum, statLabel } from "../lib/format";

type MarkLineData = NonNullable<MarkLineComponentOption["data"]>;

export function teamChartOption(d: TeamChartResponse): EChartsOption {
  const consensus = d.market_consensus_line;
  const derived = d.derived_reference_line;

  const refMarks: MarkLineData = [];
  if (consensus != null) {
    refMarks.push({
      yAxis: consensus,
      lineStyle: { color: CHART.warn, type: "dashed", width: 1.25 },
      label: refLabel(`implied total ${fmtNum(consensus)}`, CHART.warn, "insideStartTop"),
    });
  }
  if (derived != null) {
    refMarks.push({
      yAxis: derived,
      lineStyle: { color: CHART.info, type: "dotted", width: 1.25 },
      label: refLabel(`props-derived ${fmtNum(derived)}`, CHART.info),
    });
  }

  return {
    grid: grid(),
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const i = (params as Array<{ dataIndex: number }>)[0]?.dataIndex;
        const p = i === undefined ? undefined : d.series[i];
        if (!p) return "";
        return tooltipHtml(
          fmtDateShort(p.game_date?.slice(0, 10)),
          [{ label: `Team ${statLabel(d.stat_type)}`, value: fmtNum(p.value, 0), color: CHART.accent, strong: true }],
          p.opponent ? `vs ${p.opponent}` : undefined,
        );
      },
    },
    xAxis: { type: "category", data: d.series.map((p) => fmtAxisDate(p.game_date)), boundaryGap: true },
    yAxis: { type: "value", scale: true, axisLabel: { formatter: (v: number) => fmtInt(v) } },
    series: [
      {
        type: "bar",
        data: d.series.map((p) => p.value),
        barMaxWidth: 28,
        itemStyle: { color: CHART.accent, opacity: 0.75 },
        emphasis: { itemStyle: { opacity: 1 } },
        markLine: refMarks.length ? { symbol: "none", silent: true, data: refMarks } : undefined,
      },
    ],
  };
}
