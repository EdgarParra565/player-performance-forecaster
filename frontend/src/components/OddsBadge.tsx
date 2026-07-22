import { fmtOdds } from "../lib/format";

interface OddsBadgeProps {
  odds: number | null | undefined;
  // Show a muted "DFS" chip when a DFS book posts no American price.
  dfs?: boolean;
}

// American odds pill. Favorites (negative) read neutral; plus-money leans faint
// green to catch the eye, matching a sportsbook's own emphasis.
export function OddsBadge({ odds, dfs }: OddsBadgeProps) {
  if (odds === null || odds === undefined || Number.isNaN(odds)) {
    return (
      <span className="tnum rounded border border-line px-1.5 py-0.5 text-[11px] text-faint">
        {dfs ? "DFS" : "—"}
      </span>
    );
  }
  const plus = odds > 0;
  return (
    <span
      className={`tnum rounded border px-1.5 py-0.5 text-[11px] ${
        plus
          ? "border-pos-dim/50 text-pos"
          : "border-line-strong text-muted"
      }`}
    >
      {fmtOdds(odds)}
    </span>
  );
}
