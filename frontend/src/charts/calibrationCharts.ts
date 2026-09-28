// Reliability ("calibration curve") option builder: predicted vs realized.
import type { EChartsOption } from "echarts";
import type { ReliabilityBucket } from "../api/types";
import { CHART, grid, tooltipHtml } from "../components/chartTheme";
import { fmtInt, fmtPct, fmtSignedPct } from "../lib/format";

export function reliabilityOption(buckets: ReliabilityBucket[]): EChartsOption {
  const maxN = Math.max(1, ...buckets.map((b) => b.n));
  const points = buckets.map((b) => ({
    value: [b.mean_pred, b.realized_rate, b.n],
    symbolSize: 8 + 14 * Math.sqrt(b.n / maxN),
  }));
  return {
    grid: { ...grid({ legend: true }), bottom: 24 },
    legend: { top: 0, left: 0, data: ["Realized hit rate", "Perfect calibration"] },
    tooltip: {
      trigger: "item",
      formatter: (p: unknown) => {
        const idx = (p as { dataIndex: number; seriesName: string }).dataIndex;
        const name = (p as { seriesName: string }).seriesName;
        const b = name === "Realized hit rate" ? buckets[idx] : undefined;
        if (!b) return "";
        return tooltipHtml(`Predicted ${fmtPct(b.bucket_low, 0)}–${fmtPct(b.bucket_high, 0)}`, [
          { label: "Picks", value: fmtInt(b.n) },
          { label: "Mean predicted", value: fmtPct(b.mean_pred) },
          { label: "Realized", value: fmtPct(b.realized_rate), strong: true, color: CHART.accent },
          { label: "Gap (realized − predicted)", value: fmtSignedPct(b.calibration_gap) },
        ]);
      },
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      name: "predicted",
      nameLocation: "middle",
      nameGap: 28,
      axisLabel: { formatter: (v: number) => fmtPct(v, 0) },
    },
    yAxis: { type: "value", min: 0, max: 1, axisLabel: { formatter: (v: number) => fmtPct(v, 0) } },
    series: [
      {
        name: "Perfect calibration",
        type: "line",
        data: [
          [0, 0],
          [1, 1],
        ],
        showSymbol: false,
        silent: true,
        lineStyle: { color: CHART.faint, type: "dashed", width: 1 },
        itemStyle: { color: CHART.faint },
      },
      {
        name: "Realized hit rate",
        type: "line",
        data: points,
        smooth: false,
        lineStyle: { color: CHART.accent, width: 2 },
        itemStyle: { color: CHART.accent, borderColor: CHART.base, borderWidth: 1 },
        z: 3,
      },
    ],
  };
}
