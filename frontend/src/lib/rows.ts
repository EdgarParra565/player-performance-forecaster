import type { EdgeRow } from "../api/types";

// Stable row identity for scored edges. book_line is part of the key: a book
// can post alt lines for the same player + stat.
export function edgeRowKey(r: EdgeRow): string {
  return `${r.book}-${r.player_name}-${r.stat_type}-${r.book_line}`;
}

// Deep link into Player Detail by name (resolved to an id there).
export function playerHref(name: string, stat: string): string {
  return `/player?name=${encodeURIComponent(name)}&stat=${encodeURIComponent(stat)}`;
}

// Multi-select chip toggle.
export function toggleItem(list: string[], item: string): string[] {
  return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
}

// Accent/case-insensitive name key: "Nikola Jokić" === "nikola jokic".
export function nameKey(name: string | null | undefined): string {
  return (name ?? "")
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

// URL int param with bounds + default (hand-edited URLs can't break a view).
export function clampInt(raw: string | null, min: number, max: number, fallback: number): number {
  if (raw == null || !/^-?\d+$/.test(raw)) return fallback;
  return Math.min(max, Math.max(min, Number(raw)));
}
