import { fmtPct } from "../lib/format";

interface ProbabilityBarProps {
  value: number | null | undefined; // 0..1
  // Optional reference tick (e.g. the break-even implied probability).
  marker?: number | null;
  // Color the fill by polarity relative to the marker (clearing break-even
  // reads +EV green). When false the bar stays neutral.
  polarity?: boolean;
  label?: boolean;
}

// Horizontal probability meter with an optional break-even tick. Flat fill
// (no gradient) to keep the terminal look.
export function ProbabilityBar({
  value,
  marker,
  polarity = true,
  label = true,
}: ProbabilityBarProps) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className="tnum text-label text-faint">—</span>;
  }
  const pct = Math.max(0, Math.min(1, value));
  const above = marker != null ? value >= marker : value >= 0.5;
  const fill = polarity
    ? above
      ? "var(--color-pos)"
      : "var(--color-neg)"
    : "var(--color-accent)";

  return (
    <div className="flex items-center gap-2">
      <div
        className="relative h-1.5 w-full min-w-14 overflow-hidden rounded-full bg-surface-3"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={pct}
        aria-label={marker != null ? `probability vs break-even ${fmtPct(marker)}` : "probability"}
      >
        <div
          className="h-full rounded-full opacity-90"
          style={{ width: `${pct * 100}%`, backgroundColor: fill }}
        />
        {marker != null && (
          <div
            className="absolute top-0 h-full w-0.5 bg-fg/70"
            style={{ left: `${Math.max(0, Math.min(1, marker)) * 100}%` }}
            title={`break-even ${fmtPct(marker)}`}
          />
        )}
      </div>
      {label && (
        <span className="tnum w-12 shrink-0 text-right text-label text-fg">{fmtPct(value)}</span>
      )}
    </div>
  );
}
