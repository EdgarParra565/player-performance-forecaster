import type { MouseEvent } from "react";
import { useWatchlist, type WatchItem } from "../lib/watchlist";
import { statLabel } from "../lib/format";

const STAR = "M10 2.5l2.3 4.7 5.2.8-3.8 3.7.9 5.1L10 14.4l-4.6 2.4.9-5.1L2.5 8l5.2-.8L10 2.5z";

// Toggle a player / prop in the local watchlist. Stops propagation so a star
// inside a clickable table row doesn't also navigate. Accent (not EV green):
// starring is a preference, not a value judgement.
export function StarButton({ item, size = "sm" }: { item: WatchItem; size?: "sm" | "md" }) {
  const { has, toggle } = useWatchlist();
  const on = has(item);
  const what = item.kind === "prop" ? `${item.player_name} ${statLabel(item.stat)}` : item.player_name;
  const box = size === "md" ? "h-9 w-9" : "h-7 w-7";
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? `Remove ${what} from watchlist` : `Add ${what} to watchlist`}
      title={on ? "Remove from watchlist" : "Add to watchlist"}
      onClick={(e: MouseEvent) => {
        e.stopPropagation();
        toggle(item);
      }}
      onKeyDown={(e) => e.stopPropagation()}
      className={`inline-flex ${box} shrink-0 items-center justify-center rounded-md transition-colors hover:bg-surface-3 pointer-coarse:h-11 pointer-coarse:w-11 ${
        on ? "text-accent" : "text-faint hover:text-muted"
      }`}
    >
      <svg viewBox="0 0 20 20" className={size === "md" ? "h-5 w-5" : "h-4 w-4"} aria-hidden>
        <path d={STAR} fill={on ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
    </button>
  );
}

// Player cell for prop tables: star + name (sticky first column).
export function PropPlayerCell({ name, stat }: { name: string; stat: string }) {
  return (
    <span className="-ml-2 inline-flex items-center gap-1">
      <StarButton item={{ kind: "prop", player_name: name, stat }} />
      <span className="font-medium text-fg">{name}</span>
    </span>
  );
}
