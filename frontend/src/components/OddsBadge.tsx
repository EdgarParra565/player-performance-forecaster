import { fmtOdds } from "../lib/format";

interface OddsBadgeProps {
  odds: number | null | undefined;
  // Show a muted "DFS" chip when a DFS book posts no American price.
  dfs?: boolean;
}

// American odds pill, always signed (+110 / -110). Plus-money gets a brighter
// face to catch the eye (as a sportsbook does) but stays neutral — price is
// not EV, so it never borrows the +EV green.
export function OddsBadge({ odds, dfs }: OddsBadgeProps) {
  const base = "tnum inline-flex h-6 min-w-12 items-center justify-center rounded-md border px-2 text-caption";
  if (odds === null || odds === undefined || !Number.isFinite(odds)) {
    return <span className={`${base} border-line text-faint`}>{dfs ? "DFS" : "—"}</span>;
  }
  const plus = odds > 0;
  return (
    <span
      className={`${base} ${
        plus ? "border-line-strong bg-surface-3 text-fg" : "border-line bg-surface-2 text-muted"
      }`}
    >
      {fmtOdds(odds)}
    </span>
  );
}
