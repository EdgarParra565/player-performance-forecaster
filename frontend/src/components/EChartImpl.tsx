import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { THEME_NAME, terminalTheme } from "./chartTheme";

// Registered once per page load, inside the lazy chunk (echarts only loads
// behind the chart boundary).
echarts.registerTheme(THEME_NAME, terminalTheme);

export interface EChartProps {
  option: EChartsOption;
  height?: number | string;
  // Replace the whole option on update (default). Pass false for charts that
  // update incrementally (replay) so series extend instead of re-animating.
  notMerge?: boolean;
  // Accessible summary for screen readers (canvas has no text).
  ariaLabel?: string;
}

// The real ECharts wrapper. Imported only through the lazy boundary in
// `EChart.tsx` so the (heavy) echarts payload is code-split out of the initial
// route bundle. echarts-for-react disposes the instance on unmount and
// handles container resizes.
export default function EChartImpl({
  option,
  height = 260,
  notMerge = true,
  ariaLabel,
}: EChartProps) {
  return (
    <div role="img" aria-label={ariaLabel}>
      <ReactECharts
        option={option}
        theme={THEME_NAME}
        notMerge={notMerge}
        lazyUpdate
        style={{ height, width: "100%" }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
