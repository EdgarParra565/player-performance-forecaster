import { fmtPct } from "../lib/format";

interface ProbabilityBarProps {
  value: number | null | undefined; // 0..1
  // Optional reference tick (e.g. the break-even implied probability).
  marker?: number | null;
  // Color the fill by polarity relative to the marker (over the break-even
  // reads green). When false the bar stays neutral.
  polarity?: boolean;
  label?: boolean;
}

// Horizontal probability meter with an optional break-even tick. The fill is
// deliberately flat (no gradient) to keep the terminal look.
export function ProbabilityBar({
  value,
  marker,
  polarity = true,
  label = true,
}: ProbabilityBarProps) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return <span className="tnum text-xs text-faint">—</span>;
  }
  const pct = Math.max(0, Math.min(1, value));
  const above = marker != null ? value >= marker : value >= 0.5;
  const fill = polarity
    ? above
      ? "var(--color-pos)"
      : "var(--color-neg)"
    : "var(--color-info)";

  return (
    <div className="flex items-center gap-2">
      <div className="relative h-1.5 w-full min-w-14 overflow-hidden rounded-full bg-panel-3">
        <div
          className="h-full rounded-full"
          style={{ width: `${pct * 100}%`, backgroundColor: fill }}
        />
        {marker != null && (
          <div
            className="absolute top-0 h-full w-px bg-fg/60"
            style={{ left: `${Math.max(0, Math.min(1, marker)) * 100}%` }}
            title={`break-even ${fmtPct(marker)}`}
          />
        )}
      </div>
      {label && (
        <span className="tnum w-11 shrink-0 text-right text-xs text-fg">
          {fmtPct(value)}
        </span>
      )}
    </div>
  );
}
