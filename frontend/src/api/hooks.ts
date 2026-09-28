import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { apiGet, apiPost } from "./client";
import type {
  CalibrationResponse,
  CrossBookResponse,
  EdgeScanResponse,
  Health,
  LineMovementResponse,
  Meta,
  PaperTradesResponse,
  ParlayLegInput,
  ParlayResponse,
  PlayerDetail,
  PlayerSearchResponse,
  RecentGamesResponse,
  SlateKpis,
  TeamChartResponse,
} from "./types";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiGet<Health>("/health"),
    refetchInterval: 60_000,
  });
}

export function useMeta() {
  return useQuery({
    queryKey: ["meta"],
    queryFn: () => apiGet<Meta>("/meta"),
    staleTime: 5 * 60_000,
  });
}

export function useSlateKpis() {
  return useQuery({
    queryKey: ["slate", "kpis"],
    queryFn: () => apiGet<SlateKpis>("/slate/kpis"),
  });
}

export function useRecentGames(n = 12) {
  return useQuery({
    queryKey: ["slate", "recent-games", n],
    queryFn: () => apiGet<RecentGamesResponse>("/slate/recent-games", { n }),
  });
}

export interface EdgeParams {
  books?: string[];
  stats?: string[];
  model_mode?: string;
  min_edge?: number;
  min_p_over?: number;
  only_positive_ev?: boolean;
  limit?: number;
  since_hours?: number;
}

export function useEdges(params: EdgeParams, { enabled = true }: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ["slate", "edges", params],
    enabled,
    queryFn: () =>
      apiGet<EdgeScanResponse>(
        "/slate/edges",
        params as Record<string, unknown>,
      ),
    placeholderData: keepPreviousData,
  });
}

export function usePlayerSearch(
  q: string,
  { team, onlyWithLines = false, enabled = true }: { team?: string; onlyWithLines?: boolean; enabled?: boolean } = {},
) {
  return useQuery({
    queryKey: ["players", "search", q, team, onlyWithLines],
    enabled,
    queryFn: () =>
      apiGet<PlayerSearchResponse>("/players/search", {
        q,
        team,
        only_with_lines: onlyWithLines,
        limit: 40,
      }),
    placeholderData: keepPreviousData,
  });
}

export interface CrossBookParams {
  books?: string[];
  stats?: string[];
  model_mode?: string;
  min_gap?: number;
  min_books?: number;
  since_hours?: number;
}

export function useCrossBook(params: CrossBookParams) {
  return useQuery({
    queryKey: ["cross-book", params],
    queryFn: () =>
      apiGet<CrossBookResponse>(
        "/cross-book",
        params as Record<string, unknown>,
      ),
    placeholderData: keepPreviousData,
  });
}

export function useLineMovement(
  playerId: number | null,
  stat: string,
  lookbackHours = 168,
) {
  return useQuery({
    queryKey: ["line-movement", playerId, stat, lookbackHours],
    enabled: playerId !== null,
    // Never show another player's / stat's drift while the new one loads.
    placeholderData: (prev) =>
      prev && prev.player_id === playerId && prev.stat_type === stat ? prev : undefined,
    queryFn: () =>
      apiGet<LineMovementResponse>(`/players/${playerId}/line-movement`, {
        stat,
        lookback_hours: lookbackHours,
      }),
  });
}

export function useTeamChart(team: string | null, stat: string, nGames = 25) {
  return useQuery({
    queryKey: ["team-chart", team, stat, nGames],
    enabled: !!team,
    // Keep the old chart only while tweaking the same team (n_games / stat).
    placeholderData: (prev) => (prev && prev.team === team ? prev : undefined),
    queryFn: () =>
      apiGet<TeamChartResponse>(`/teams/${team}/chart`, {
        stat,
        n_games: nGames,
      }),
  });
}

export interface PlayerDetailParams {
  playerId: number;
  name?: string;
  stat: string;
  n_games: number;
  rolling_window: number;
}

export function usePlayerDetail(params: PlayerDetailParams | null) {
  return useQuery({
    queryKey: ["players", "detail", params],
    enabled: params !== null,
    // Keep previous data only while tweaking the SAME player (stat / window);
    // switching players must not leave player A's EV table under B's name.
    placeholderData: (prev) =>
      prev && params && prev.player_id === params.playerId ? prev : undefined,
    queryFn: () => {
      const p = params!;
      return apiGet<PlayerDetail>(`/players/${p.playerId}`, {
        name: p.name,
        stat: p.stat,
        n_games: p.n_games,
        rolling_window: p.rolling_window,
      });
    },
  });
}

// Parlay pricing is compute-only (POST); cached by the exact leg set so
// re-renders and toggling back to a previous parlay don't re-simulate.
export function useParlayPrice(legs: ParlayLegInput[], nGames: number, parlayOdds: number | null) {
  const body = {
    legs: legs.map(({ player_id, stat, line, side, odds }) => ({ player_id, stat, line, side, odds })),
    n_games: nGames,
    parlay_odds: parlayOdds,
  };
  return useQuery({
    queryKey: ["parlay", body],
    enabled: legs.length >= 2,
    placeholderData: keepPreviousData,
    queryFn: () => apiPost<ParlayResponse>("/parlay/price", body),
  });
}

export function usePaperTrades(status: "all" | "pending" | "settled" = "all") {
  return useQuery({
    queryKey: ["paper-trades", status],
    queryFn: () => apiGet<PaperTradesResponse>("/paper-trades", { status, limit: 500 }),
  });
}

export function useCalibration(source: string, stat: string | null, nBuckets = 10) {
  return useQuery({
    queryKey: ["calibration", source, stat, nBuckets],
    placeholderData: keepPreviousData,
    queryFn: () =>
      apiGet<CalibrationResponse>("/paper-trades/calibration", {
        source,
        stat: stat ?? undefined,
        n_buckets: nBuckets,
      }),
  });
}
