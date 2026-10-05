interface EmptyStateProps {
  title: string;
  hint?: string;
  lastData?: string | null;
  compact?: boolean;
}

// Deliberate empty-state for a mounted DB with no current lines (offseason,
// or books haven't posted yet). We show WHY it is empty and how fresh the DB
// is, rather than a blank pane. A MISSING database is a different state:
// see DbNotice.
export function EmptyState({ title, hint, lastData, compact }: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center px-6 text-center ${
        compact ? "py-8" : "py-16"
      }`}
    >
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg border border-line-strong bg-surface-2">
        <svg viewBox="0 0 20 20" className="h-4 w-4 text-faint" aria-hidden>
          <path
            d="M3 14l4-4 3 3 7-7"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <div className="text-title font-medium text-fg">{title}</div>
      {hint && <div className="mt-2 max-w-md text-body text-muted">{hint}</div>}
      {lastData && (
        <div className="tnum mt-4 rounded-full border border-line px-3 py-1 text-caption text-faint">
          last data · {lastData}
        </div>
      )}
    </div>
  );
}
