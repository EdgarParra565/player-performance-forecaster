// Shared number/date formatting for every numeric surface. Views never call
// toFixed / toLocaleString directly — they go through these so odds are always
// signed, percentages are 1dp, and null / NaN render as an em dash.

export const DASH = "—";

function isMissing(value: number | null | undefined): value is null | undefined {
  return value === null || value === undefined || !Number.isFinite(value);
}

// toFixed that never prints "-0.0": values that round to zero lose the sign.
function fixed(value: number, digits: number): string {
  const s = value.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s;
}

// Sign of a value AFTER rounding to `digits` (so -0.0004 at 3dp is 0).
export function roundedSign(value: number, digits: number): -1 | 0 | 1 {
  const n = Number(value.toFixed(digits));
  return n > 0 ? 1 : n < 0 ? -1 : 0;
}

export function fmtNum(value: number | null | undefined, digits = 1): string {
  if (isMissing(value)) return DASH;
  return fixed(value, digits);
}

// Prop lines always carry one decimal (27.5, 6.0).
export function fmtLine(value: number | null | undefined): string {
  return fmtNum(value, 1);
}

export function fmtInt(value: number | null | undefined): string {
  if (isMissing(value)) return DASH;
  return Math.round(value).toLocaleString("en-US");
}

// Probability 0..1 -> "53.2%".
export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (isMissing(value)) return DASH;
  return `${fixed(value * 100, digits)}%`;
}

// American odds, always signed: +110 / -110. DFS books post none -> dash.
export function fmtOdds(value: number | null | undefined): string {
  if (isMissing(value)) return DASH;
  const v = Math.round(value);
  return v > 0 ? `+${v}` : `${v}`;
}

// Signed number, e.g. "+0.084" / "-0.021" / "0.000".
export function fmtSigned(value: number | null | undefined, digits = 3): string {
  if (isMissing(value)) return DASH;
  const s = fixed(value, digits);
  return roundedSign(value, digits) > 0 ? `+${s}` : s;
}

// Signed percentage from a 0..1 fraction: "+4.2%".
export function fmtSignedPct(value: number | null | undefined, digits = 1): string {
  if (isMissing(value)) return DASH;
  const s = fixed(value * 100, digits);
  return roundedSign(value * 100, digits) > 0 ? `+${s}%` : `${s}%`;
}

// Expected value per 1-unit stake: "+0.76u".
export function fmtEv(value: number | null | undefined): string {
  if (isMissing(value)) return DASH;
  return `${fmtSigned(value, 2)}u`;
}

// --- Dates ------------------------------------------------------------------

const DATE_ONLY = /^(\d{4})-(\d{2})-(\d{2})$/;
const HAS_OFFSET = /(Z|[+-]\d{2}:?\d{2})$/i;

// Parse API timestamps with the right zone semantics:
//  - "2026-06-13"            -> a calendar date, in LOCAL time (no UTC shift,
//                              which would print Jun 12 west of Greenwich)
//  - "2026-07-22 01:25:56"   -> naive DB timestamps are UTC
//  - "...+00:00" / "...Z"    -> as given
export function parseApiDate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const v = String(value).trim();
  const m = DATE_ONLY.exec(v);
  if (m) return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  let iso = v.replace(" ", "T");
  if (/T\d{2}:\d{2}/.test(iso) && !HAS_OFFSET.test(iso)) iso += "Z";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

// "2026-06-13" -> "Jun 13".
export function fmtDateShort(value: string | null | undefined): string {
  if (!value) return DASH;
  const d = parseApiDate(value);
  if (!d) return String(value).slice(0, 10);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// Game-date axis label: "2026-03-05T00:00:00" -> "Mar 5" (calendar date).
export function fmtAxisDate(value: string | null | undefined): string {
  return fmtDateShort(value ? String(value).slice(0, 10) : value);
}

// "Sep 28, 09:22 AM".
export function fmtDateTime(value: string | null | undefined): string {
  const d = parseApiDate(value);
  if (!d) return value ? String(value).slice(0, 16) : DASH;
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Relative freshness from a timestamp: "3m ago" / "2.4h ago" / "5d ago".
export function fmtAgo(value: string | null | undefined, now = Date.now()): string {
  const d = parseApiDate(value);
  if (!d) return "no data";
  return fmtHoursAgo((now - d.getTime()) / 3_600_000);
}

// Relative freshness from an age in hours.
export function fmtHoursAgo(hours: number | null | undefined): string {
  if (isMissing(hours)) return DASH;
  if (hours < 1 / 60) return "just now";
  if (hours < 1) return `${Math.round(hours * 60)}m ago`;
  if (hours < 48) return `${hours.toFixed(1)}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

// --- Labels -----------------------------------------------------------------

export function titleCase(value: string | null | undefined): string {
  if (!value) return DASH;
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
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
  if (!stat) return DASH;
  return STAT_LABELS[stat] ?? stat.toUpperCase();
}
