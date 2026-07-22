import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { apiGet } from "./client";
import type {
  EdgeScanResponse,
  Health,
  Meta,
  PlayerDetail,
  PlayerSearchResponse,
  RecentGamesResponse,
  SlateKpis,
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
