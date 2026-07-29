import { lazy, Suspense } from "react";
import type { LineSparklineProps } from "./LineSparklineImpl";

const Impl = lazy(() => import("./LineSparklineImpl"));

// Lazy wrapper so sparklines share the code-split echarts chunk instead of
// dragging it into the initial bundle.
export function LineSparkline(props: LineSparklineProps) {
  return (
    <Suspense
      fallback={
        <span
          className="inline-block animate-pulse rounded bg-panel-2 align-middle"
          style={{ width: props.width ?? 96, height: props.height ?? 24 }}
        />
      }
    >
      <Impl {...props} />
    </Suspense>
  );
}
