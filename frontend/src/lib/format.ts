// Formatting for the numeric surfaces. All money/odds/probabilities render in
// tabular mono elsewhere; these just produce the strings.

export function fmtNum(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.round(value).toLocaleString("en-US");
}

export function fmtPct(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

// American odds: DFS books post none -> render an em dash.
export function fmtOdds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const v = Math.round(value);
  return v > 0 ? `+${v}` : `${v}`;
}

// Signed edge/EV, e.g. "+0.084" / "-0.021".
export function fmtSigned(
  value: number | null | undefined,
  digits = 3,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const s = value.toFixed(digits);
  return value > 0 ? `+${s}` : s;
}

export function fmtSignedPct(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const s = (value * 100).toFixed(digits);
  return value > 0 ? `+${s}%` : `${s}%`;
}

// "2026-06-13T00:00:00" -> "Jun 13".
export function fmtDateShort(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// Relative freshness: "3m ago" / "2.4h ago" / "5d ago".
export function fmtAgo(value: string | null | undefined): string {
  if (!value) return "no data";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "no data";
  const hours = (Date.now() - then) / 3_600_000;
  if (hours < 1 / 60) return "just now";
  if (hours < 1) return `${Math.round(hours * 60)}m ago`;
  if (hours < 48) return `${hours.toFixed(1)}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function titleCase(value: string | null | undefined): string {
  if (!value) return "—";
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// Canonical stat key -> compact display label.
const STAT_LABELS: Record<string, string> = {
  points: "PTS",
  assists: "AST",
  rebounds: "REB",
  pra: "PRA",
  ra: "R+A",
  three_pointers_made: "3PM",
  field_goals_made: "FGM",
  minutes: "MIN",
};

export function statLabel(stat: string | null | undefined): string {
  if (!stat) return "—";
  return STAT_LABELS[stat] ?? stat.toUpperCase();
}
