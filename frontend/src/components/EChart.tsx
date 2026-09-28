import { Component, lazy, Suspense, type ReactNode } from "react";
import type { EChartProps } from "./EChartImpl";
import { Skeleton } from "./Skeleton";

// Wait for the self-hosted fonts before the first canvas paint; canvas text
// is rasterised once, so a late web font would leave fallback-font labels.
function fontsReady(): Promise<unknown> {
  const fonts = typeof document !== "undefined" ? document.fonts : undefined;
  return fonts?.ready ?? Promise.resolve();
}

const EChartImpl = lazy(() =>
  Promise.all([import("./EChartImpl"), fontsReady()]).then(([m]) => m),
);

// A failed chunk load or chart render degrades to an inline notice instead of
// blanking the whole route.
class ChartBoundary extends Component<
  { height: number | string; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return (
        <div
          role="alert"
          className="flex items-center justify-center rounded-md border border-dashed border-line-strong text-label text-faint"
          style={{ height: this.props.height }}
        >
          Chart failed to load — refresh to retry.
        </div>
      );
    }
    return this.props.children;
  }
}

// Public chart entry point. Lazy-loads the echarts bundle on first render so
// non-chart routes (and the initial paint) don't pay the ~1 MB cost.
export function EChart({ height = 260, ...rest }: EChartProps) {
  return (
    <ChartBoundary height={height}>
      <Suspense fallback={<Skeleton style={{ height, width: "100%" }} />}>
        <EChartImpl height={height} {...rest} />
      </Suspense>
    </ChartBoundary>
  );
}
