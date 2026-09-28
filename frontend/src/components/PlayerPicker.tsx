import { useId, useState } from "react";
import { usePlayerSearch } from "../api/hooks";
import type { PlayerSearchRow } from "../api/types";
import { useDebounced } from "./controls";

// Accessible player search combobox (debounced server search, arrow/Enter/
// Escape keys, closes on blur). Shared by Player Detail and the parlay builder.
export function PlayerPicker({
  onSelect,
  placeholder = "Search players…",
  label = "Search players",
}: {
  onSelect: (r: PlayerSearchRow) => void;
  placeholder?: string;
  label?: string;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const q = useDebounced(query.trim(), 200);
  const { data, isFetching, isPlaceholderData } = usePlayerSearch(q, { enabled: q.length > 0 });
  const rows = q ? (data?.rows ?? []) : [];
  const searching = isFetching || isPlaceholderData || q !== query.trim();

  function choose(r: PlayerSearchRow) {
    onSelect(r);
    setQuery("");
    setOpen(false);
  }

  return (
    <div className="relative w-full sm:w-80">
      <svg viewBox="0 0 20 20" aria-hidden className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-faint">
        <circle cx="9" cy="9" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M13.5 13.5L17 17" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <input
        role="combobox"
        aria-label={label}
        aria-expanded={open && query.length > 0}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open && rows[active] ? `${listId}-${rows[active].player_id}` : undefined}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((i) => Math.min(i + 1, Math.max(0, rows.length - 1)));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((i) => Math.max(i - 1, 0));
          } else if (e.key === "Enter" && rows[active]) {
            e.preventDefault();
            choose(rows[active]);
          } else if (e.key === "Escape") {
            setOpen(false);
          }
        }}
        placeholder={placeholder}
        className="h-10 w-full rounded-lg border border-line bg-surface-2 pr-3 pl-9 text-body text-fg placeholder:text-faint transition-colors hover:border-line-strong focus:border-accent/70 focus:outline-none pointer-coarse:h-11"
      />
      {open && query.length > 0 && (
        <ul
          id={listId}
          role="listbox"
          className="absolute z-30 mt-2 max-h-80 w-full overflow-auto rounded-lg border border-line-strong bg-surface-2 p-1 shadow-[var(--shadow-pop)]"
        >
          {searching && !rows.length && <li className="px-3 py-2 text-label text-faint">Searching…</li>}
          {!searching && !rows.length && <li className="px-3 py-2 text-label text-faint">No matches.</li>}
          {rows.map((r, i) => (
            <li
              key={r.player_id}
              id={`${listId}-${r.player_id}`}
              role="option"
              aria-selected={i === active}
              // mousedown (not click) so the input's blur doesn't close first.
              onMouseDown={(e) => {
                e.preventDefault();
                choose(r);
              }}
              onMouseEnter={() => setActive(i)}
              className={`flex cursor-pointer items-center justify-between rounded-md px-3 py-2 text-body pointer-coarse:min-h-11 ${
                i === active ? "bg-surface-3 text-fg" : "text-muted"
              }`}
            >
              <span>
                {r.player_name}
                {r.team && <span className="tnum ml-2 text-caption text-faint">{r.team}</span>}
              </span>
              {r.n_books > 0 && (
                <span className="tnum text-caption text-faint">
                  {r.n_books} book{r.n_books === 1 ? "" : "s"}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
