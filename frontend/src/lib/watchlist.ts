import { useCallback, useSyncExternalStore } from "react";
import type { EdgeRow } from "../api/types";
import { nameKey } from "./rows";

// Client-side watchlist: starred players and props, kept in localStorage
// (no server writes, consistent with the no-auth posture). Survives refresh,
// syncs across tabs via the `storage` event, and round-trips through a URL
// param for sharing (`/watchlist?w=`).

export type WatchKind = "player" | "prop";

export interface WatchItem {
  kind: WatchKind;
  player_name: string;
  player_id?: number | null;
  stat?: string | null; // props only
  added_at?: string;
}

const KEY = "flagship.watchlist";
const EVENT = "flagship:watchlist";
export const MAX_ITEMS = 50;

export function itemKey(i: Pick<WatchItem, "kind" | "player_name" | "stat">): string {
  return i.kind === "prop" ? `prop:${nameKey(i.player_name)}:${i.stat ?? ""}` : `player:${nameKey(i.player_name)}`;
}

function sanitize(raw: unknown): WatchItem[] {
  if (!Array.isArray(raw)) return [];
  const out: WatchItem[] = [];
  const seen = new Set<string>();
  for (const r of raw) {
    if (!r || typeof r !== "object") continue;
    const o = r as Record<string, unknown>;
    const kind = o.kind === "prop" ? "prop" : o.kind === "player" ? "player" : null;
    const name = typeof o.player_name === "string" ? o.player_name.trim().slice(0, 80) : "";
    const stat = typeof o.stat === "string" && /^[a-z_]{2,32}$/.test(o.stat) ? o.stat : null;
    if (!kind || !name || (kind === "prop" && !stat)) continue;
    const id = typeof o.player_id === "number" && Number.isInteger(o.player_id) && o.player_id > 0 ? o.player_id : null;
    const item: WatchItem = {
      kind,
      player_name: name,
      player_id: id,
      stat: kind === "prop" ? stat : null,
      added_at: typeof o.added_at === "string" ? o.added_at.slice(0, 32) : undefined,
    };
    const k = itemKey(item);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(item);
    if (out.length >= MAX_ITEMS) break;
  }
  return out;
}

// --- store (module-level so every component shares one snapshot) ------------

let cache: WatchItem[] | null = null;

function read(): WatchItem[] {
  if (cache) return cache;
  try {
    cache = sanitize(JSON.parse(window.localStorage.getItem(KEY) ?? "[]"));
  } catch {
    cache = []; // storage disabled or corrupt JSON -> empty, never throw
  }
  return cache;
}

function write(items: WatchItem[]): void {
  cache = sanitize(items);
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    /* storage full / disabled: keep the in-memory list for this session */
  }
  window.dispatchEvent(new Event(EVENT));
}

function subscribe(cb: () => void): () => void {
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY || e.key === null) {
      cache = null; // another tab changed it
      cb();
    }
  };
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", onStorage);
  };
}

export function getWatchlist(): WatchItem[] {
  return read();
}

export function setWatchlist(items: WatchItem[]): void {
  write(items);
}

export function toggleWatch(item: WatchItem): boolean {
  const items = read();
  const k = itemKey(item);
  if (items.some((i) => itemKey(i) === k)) {
    write(items.filter((i) => itemKey(i) !== k));
    return false;
  }
  write([{ ...item, added_at: new Date().toISOString() }, ...items].slice(0, MAX_ITEMS));
  return true;
}

export function removeWatch(key: string): void {
  write(read().filter((i) => itemKey(i) !== key));
}

// Merge a shared list into ours (ours first, no duplicates, capped).
export function mergeWatchlist(incoming: WatchItem[]): void {
  write([...read(), ...incoming]);
}

export function useWatchlist() {
  const items = useSyncExternalStore(subscribe, read, () => []);
  const has = useCallback((i: Pick<WatchItem, "kind" | "player_name" | "stat">) => {
    const k = itemKey(i);
    return items.some((x) => itemKey(x) === k);
  }, [items]);
  return { items, has, toggle: toggleWatch, remove: removeWatch, replace: setWatchlist, merge: mergeWatchlist };
}

// --- URL sharing ------------------------------------------------------------------

// `<kind p|q>~<player_id or ->~<stat or ->~<name>` joined by `|`.
export function encodeWatchlist(items: WatchItem[]): string {
  return items
    .map((i) => [i.kind === "prop" ? "q" : "p", i.player_id ?? "-", i.stat ?? "-", encodeURIComponent(i.player_name)].join("~"))
    .join("|");
}

export function decodeWatchlist(raw: string | null): WatchItem[] {
  if (!raw) return [];
  const parsed = raw.split("|").slice(0, MAX_ITEMS).map((part) => {
    const [k, id, stat, name] = part.split("~");
    let player_name = "";
    try {
      player_name = decodeURIComponent(name ?? "");
    } catch {
      player_name = "";
    }
    return {
      kind: k === "q" ? "prop" : k === "p" ? "player" : "",
      player_id: /^\d+$/.test(id ?? "") ? Number(id) : null,
      stat: stat && stat !== "-" ? stat : null,
      player_name,
    };
  });
  return sanitize(parsed);
}

export function _resetWatchlistCacheForTests(): void {
  cache = null;
}

export interface WatchRow {
  item: WatchItem;
  best: EdgeRow | null; // highest-edge line on the board for this item
  nLines: number;
}

// Match each starred item against the scored board: props by player + stat,
// players by name across all their stats. Pure for testing.
export function matchWatchlist(items: WatchItem[], board: EdgeRow[]): WatchRow[] {
  const byPlayer = new Map<string, EdgeRow[]>();
  for (const r of board) {
    const k = nameKey(r.player_name);
    const list = byPlayer.get(k);
    if (list) list.push(r);
    else byPlayer.set(k, [r]);
  }
  return items.map((item) => {
    const rows = (byPlayer.get(nameKey(item.player_name)) ?? []).filter(
      (r) => item.kind === "player" || r.stat_type === item.stat,
    );
    const best = rows.reduce<EdgeRow | null>(
      (acc, r) => (acc == null || (r.model_edge ?? -Infinity) > (acc.model_edge ?? -Infinity) ? r : acc),
      null,
    );
    return { item, best, nLines: rows.length };
  });
}
