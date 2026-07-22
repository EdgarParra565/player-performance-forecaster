import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { CHART } from "./chartBase";

interface LineSparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}

// Minimal inline trend line — no axes, no grid, just the shape of a series.
// Used in table cells and the recent-games strip.
export function LineSparkline({
  data,
  width = 96,
  height = 24,
  color = CHART.info,
}: LineSparklineProps) {
  if (!data.length) return <span className="text-faint">—</span>;
  const option: EChartsOption = {
    backgroundColor: "transparent",
    grid: { left: 1, right: 1, top: 2, bottom: 2 },
    xAxis: { type: "category", show: false, data: data.map((_, i) => i) },
    yAxis: { type: "value", show: false, scale: true },
    series: [
      {
        type: "line",
        data,
        showSymbol: false,
        smooth: true,
        lineStyle: { width: 1.5, color },
        areaStyle: { color, opacity: 0.08 },
      },
    ],
  };
  return (
    <ReactECharts
      option={option}
      style={{ width, height }}
      opts={{ renderer: "canvas" }}
    />
  );
}
