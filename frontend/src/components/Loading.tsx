interface LoadingProps {
  label?: string;
  rows?: number;
}

// Skeleton shimmer rows — keeps the dense-table rhythm while data loads.
export function Loading({ label, rows = 5 }: LoadingProps) {
  return (
    <div className="py-4">
      {label && <div className="eyebrow mb-3 px-3">{label}</div>}
      <div className="space-y-1.5 px-3">
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={i}
            className="h-7 animate-pulse rounded bg-panel-2"
            style={{ opacity: 1 - i * 0.12 }}
          />
        ))}
      </div>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 rounded border border-neg-dim/40 bg-neg-soft px-3 py-2.5 text-xs text-neg">
      <span className="h-1.5 w-1.5 rounded-full bg-neg" />
      {message}
    </div>
  );
}
