interface DeltaProps {
  value: number | null | undefined;
  format: (v: number | null | undefined) => string;
  // Zero-reference for polarity coloring (default 0).
  zero?: number;
  strong?: boolean;
}

// A signed metric that colors green above the reference and red below it —
// the +EV / -EV language used everywhere numeric edge is shown.
export function Delta({ value, format, zero = 0, strong }: DeltaProps) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return <span className="tnum text-faint">—</span>;
  }
  const tone =
    value > zero ? "text-pos" : value < zero ? "text-neg" : "text-muted";
  return (
    <span className={`tnum ${tone} ${strong ? "font-semibold" : ""}`}>
      {format(value)}
    </span>
  );
}
