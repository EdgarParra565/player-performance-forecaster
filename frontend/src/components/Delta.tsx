interface DeltaProps {
  value: number | null | undefined;
  format: (v: number | null | undefined) => string;
  // Zero-reference for polarity coloring (default 0).
  zero?: number;
  strong?: boolean;
  // Digits the formatter rounds to; a value that rounds to the reference is
  // neutral, so "-0.0%" never shows up red.
  digits?: number;
}

// A signed metric that colors green above the reference and red below it —
// the +EV / -EV language used everywhere numeric edge is shown.
export function Delta({ value, format, zero = 0, strong, digits = 4 }: DeltaProps) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className="tnum text-faint">—</span>;
  }
  const d = Number((value - zero).toFixed(digits));
  const tone = d > 0 ? "text-pos" : d < 0 ? "text-neg" : "text-muted";
  return <span className={`tnum ${tone} ${strong ? "font-semibold" : ""}`}>{format(value)}</span>;
}

// Over / Under side label. Side is a direction, not a value judgement, so it
// is neutral (arrow + text) — green/red stay reserved for EV.
export function SideTag({ side }: { side: string | null | undefined }) {
  if (!side) return <span className="tnum text-faint">—</span>;
  const over = side === "over";
  return (
    <span className="tnum inline-flex items-center gap-1 text-caption font-medium text-fg uppercase">
      <span aria-hidden className="text-faint">
        {over ? "▲" : "▼"}
      </span>
      {side}
    </span>
  );
}
