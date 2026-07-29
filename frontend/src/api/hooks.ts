import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { apiGet } from "./client";
import type {
  CrossBookResponse,
  EdgeScanResponse,
  Health,
  LineMovementResponse,
  Meta,
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

export function useEdges(params: EdgeParams) {
  return useQuery({
    queryKey: ["slate", "edges", params],
    queryFn: () =>
      apiGet<EdgeScanResponse>(
        "/slate/edges",
        params as Record<string, unknown>,
      ),
    placeholderData: keepPreviousData,
  });
}

export function usePlayerSearch(q: string, team?: string, onlyWithLines = false) {
  return useQuery({
    queryKey: ["players", "search", q, team, onlyWithLines],
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
    placeholderData: keepPreviousData,
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
    placeholderData: keepPreviousData,
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
    placeholderData: keepPreviousData,
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
