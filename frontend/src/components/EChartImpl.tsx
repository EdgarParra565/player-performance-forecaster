import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

export interface EChartProps {
  option: EChartsOption;
  height?: number | string;
}

// The real ECharts wrapper. Imported only through the lazy boundary in
// `EChart.tsx` so the (heavy) echarts payload is code-split out of the initial
// route bundle.
export default function EChartImpl({ option, height = 260 }: EChartProps) {
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
