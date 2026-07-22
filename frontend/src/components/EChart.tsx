import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

interface EChartProps {
  option: EChartsOption;
  height?: number | string;
}

// Thin wrapper: transparent background (the panel behind it provides the
// surface), no merge so option changes fully replace, and a consistent height.
export function EChart({ option, height = 260 }: EChartProps) {
  return (
    <ReactECharts
      option={{ backgroundColor: "transparent", ...option }}
      notMerge
      lazyUpdate
      style={{ height, width: "100%" }}
      opts={{ renderer: "canvas" }}
    />
  );
}
