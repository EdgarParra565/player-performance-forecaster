interface EmptyStateProps {
  title: string;
  hint?: string;
  lastData?: string | null;
  compact?: boolean;
}

// Deliberate empty-state — it is the NBA offseason, so most live-line surfaces
// are empty until October. We show WHY it is empty and how fresh the DB is,
// rather than a blank pane.
export function EmptyState({ title, hint, lastData, compact }: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center ${
        compact ? "py-8" : "py-16"
      }`}
    >
      <div className="mb-3 flex h-9 w-9 items-center justify-center rounded border border-line-strong bg-panel-2">
        <span className="h-1.5 w-1.5 rounded-full bg-faint" />
      </div>
      <div className="text-sm font-medium text-muted">{title}</div>
      {hint && (
        <div className="mt-1.5 max-w-sm text-xs leading-relaxed text-faint">
          {hint}
        </div>
      )}
      {lastData && (
        <div className="tnum mt-3 text-[11px] text-faint">
          last data · {lastData}
        </div>
      )}
    </div>
  );
}
