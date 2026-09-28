import type { CSSProperties } from "react";

// Skeleton primitives: shimmer blocks shaped like the content they stand in
// for, so the layout doesn't jump when data lands.

export function Skeleton({
  className = "",
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return <div aria-hidden className={`skeleton ${className}`} style={style} />;
}

export function KpiSkeleton() {
  return (
    <div className="panel px-4 py-4" aria-hidden>
      <Skeleton className="h-3 w-20" />
      <Skeleton className="mt-3 h-7 w-24" />
      <Skeleton className="mt-3 h-3 w-16" />
    </div>
  );
}

export function KpiRowSkeleton({ count = 4 }: { count?: number }) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <KpiSkeleton key={i} />
      ))}
    </>
  );
}

export function TableSkeleton({ rows = 6, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div role="status" aria-label="Loading" className="px-4 py-3">
      <div className="flex gap-4 border-b border-line pb-3">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} className="h-3 flex-1" />
        ))}
      </div>
      <div className="divide-y divide-line/60">
        {Array.from({ length: rows }).map((_, r) => (
          <div key={r} className="flex gap-4 py-3" style={{ opacity: 1 - r * 0.1 }}>
            {Array.from({ length: cols }).map((_, c) => (
              <Skeleton key={c} className={`h-4 flex-1 ${c === 0 ? "max-w-40" : ""}`} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ChartSkeleton({ height = 260 }: { height?: number }) {
  return (
    <div role="status" aria-label="Loading chart" className="p-4">
      <div className="flex h-full items-end gap-2" style={{ height }}>
        {Array.from({ length: 18 }).map((_, i) => (
          <Skeleton
            key={i}
            className="flex-1"
            style={{ height: `${30 + ((i * 37) % 60)}%` }}
          />
        ))}
      </div>
    </div>
  );
}
