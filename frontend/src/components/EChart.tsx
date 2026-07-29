import { lazy, Suspense } from "react";
import type { EChartProps } from "./EChartImpl";

const EChartImpl = lazy(() => import("./EChartImpl"));

function ChartSkeleton({ height }: { height?: number | string }) {
  return (
    <div
      className="animate-pulse rounded bg-panel-2"
      style={{ height: height ?? 260, width: "100%" }}
    />
  );
}

// Public chart entry point. Lazy-loads the echarts bundle on first render so
// non-chart routes (and the initial paint) don't pay the ~1 MB cost.
export function EChart({ option, height = 260 }: EChartProps) {
  return (
    <Suspense fallback={<ChartSkeleton height={height} />}>
      <EChartImpl option={option} height={height} />
    </Suspense>
  );
}
