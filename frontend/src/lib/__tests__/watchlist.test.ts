import { describe, it, expect, beforeEach } from "vitest";
import type { EdgeRow } from "../../api/types";
import {
  _resetWatchlistCacheForTests,
  decodeWatchlist,
  encodeWatchlist,
  getWatchlist,
  itemKey,
  matchWatchlist,
  MAX_ITEMS,
  mergeWatchlist,
  setWatchlist,
  toggleWatch,
  type WatchItem,
} from "../watchlist";

const jokic: WatchItem = { kind: "player", player_name: "Nikola Jokić", player_id: 203999 };
const lbjPts: WatchItem = { kind: "prop", player_name: "LeBron James", stat: "points" };

beforeEach(() => {
  window.localStorage.clear();
  _resetWatchlistCacheForTests();
});

describe("watchlist store", () => {
  it("toggles items and persists to localStorage", () => {
    expect(toggleWatch(jokic)).toBe(true);
    expect(toggleWatch(lbjPts)).toBe(true);
    expect(getWatchlist().map(itemKey)).toEqual(["prop:lebron james:points", "player:nikola jokic"]);
    _resetWatchlistCacheForTests(); // "refresh"
    expect(getWatchlist()).toHaveLength(2);
    expect(toggleWatch(lbjPts)).toBe(false);
    expect(getWatchlist().map(itemKey)).toEqual(["player:nikola jokic"]);
  });

  it("dedupes accent/case variants and caps the list", () => {
    setWatchlist([jokic, { kind: "player", player_name: "nikola jokic" }]);
    expect(getWatchlist()).toHaveLength(1);
    setWatchlist(Array.from({ length: 80 }, (_, i) => ({ kind: "player" as const, player_name: `P${i}` })));
    expect(getWatchlist()).toHaveLength(MAX_ITEMS);
  });

  it("survives corrupt storage and rejects malformed items", () => {
    window.localStorage.setItem("flagship.watchlist", "{not json");
    _resetWatchlistCacheForTests();
    expect(getWatchlist()).toEqual([]);
    window.localStorage.setItem(
      "flagship.watchlist",
      JSON.stringify([{ kind: "prop", player_name: "X" }, { kind: "evil" }, { kind: "player", player_name: "<b>ok</b>" }]),
    );
    _resetWatchlistCacheForTests();
    expect(getWatchlist().map((i) => i.player_name)).toEqual(["<b>ok</b>"]); // prop w/o stat dropped
  });

  it("picks up changes from another tab via the storage event", () => {
    setWatchlist([jokic]);
    window.localStorage.setItem("flagship.watchlist", JSON.stringify([lbjPts]));
    window.dispatchEvent(new StorageEvent("storage", { key: "flagship.watchlist" }));
    // subscribe() clears the cache on the event; read() then re-parses.
    _resetWatchlistCacheForTests();
    expect(getWatchlist().map(itemKey)).toEqual([itemKey(lbjPts)]);
  });

  it("merges shared lists without duplicates", () => {
    setWatchlist([jokic]);
    mergeWatchlist([jokic, lbjPts]);
    expect(getWatchlist().map(itemKey)).toEqual([itemKey(jokic), itemKey(lbjPts)]);
  });
});

describe("watchlist URL codec", () => {
  it("round-trips", () => {
    const out = decodeWatchlist(encodeWatchlist([jokic, lbjPts]));
    expect(out.map(itemKey)).toEqual([itemKey(jokic), itemKey(lbjPts)]);
    expect(out[0].player_id).toBe(203999);
  });

  it("drops junk entries", () => {
    expect(decodeWatchlist("x~1~-~A|q~abc~DROP;TABLE~B|p~-~-~%E0%A4%A|q~5~points~Good")).toEqual([
      { kind: "prop", player_name: "Good", player_id: 5, stat: "points", added_at: undefined },
    ]);
    expect(decodeWatchlist(null)).toEqual([]);
  });
});

describe("matchWatchlist", () => {
  const row = (name: string, stat: string, edge: number, line = 20.5): EdgeRow =>
    ({ book: "fd", player_name: name, stat_type: stat, book_line: line, model_edge: edge } as EdgeRow);

  it("matches props by player+stat and players across stats, picking the best edge", () => {
    const board = [row("Nikola Jokic", "points", 0.05), row("Nikola Jokić", "assists", 0.12), row("LeBron James", "points", 0.02, 25.5)];
    const [p, q] = matchWatchlist([jokic, lbjPts], board);
    expect(p.nLines).toBe(2);
    expect(p.best?.stat_type).toBe("assists");
    expect(q.best?.book_line).toBe(25.5);
    const [none] = matchWatchlist([{ kind: "prop", player_name: "Nobody", stat: "points" }], board);
    expect(none.best).toBeNull();
    expect(none.nLines).toBe(0);
  });
});
