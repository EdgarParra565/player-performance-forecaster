import { useNavigate } from "react-router-dom";
import { useEdges, useRecentGames, useSlateKpis } from "../api/hooks";
import { StatCard } from "../components/StatCard";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { KpiRowSkeleton, Skeleton, TableSkeleton } from "../components/Skeleton";
import { Card, KpiGrid, Page } from "../components/Page";
import { Delta, SideTag } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { PropPlayerCell } from "../components/StarButton";
import type { EdgeRow, RecentGame } from "../api/types";
import {
  DASH,
  fmtAgo,
  fmtDateShort,
  fmtEv,
  fmtInt,
  fmtLine,
  fmtSignedPct,
  statLabel,
} from "../lib/format";
import { edgeRowKey, playerHref } from "../lib/rows";

function RecentGamesStrip() {
  const { data, isLoading, isError, error } = useRecentGames(10);
  if (isLoading) {
    return (
      <div className="flex gap-4 overflow-hidden">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-24 w-40 shrink-0" />
        ))}
      </div>
    );
  }
  if (isError) return <ErrorState message={errorMessage(error, "Failed to load recent games.")} />;
  const rows = data?.rows ?? [];
  if (!rows.length) return <EmptyState compact title="No games in the window." />;
  return (
    <div className="-mx-1 flex gap-3 overflow-x-auto px-1 pb-2">
      {rows.map((g: RecentGame) => (
        <div key={g.game_id} className="panel w-40 shrink-0 px-4 py-3">
          <div className="tnum mb-2 text-caption text-faint">{fmtDateShort(g.game_date)}</div>
          <ScoreLine team={g.away_abbrev} pts={g.away_pts} won={g.winner === g.away_abbrev} />
          <ScoreLine team={g.home_abbrev} pts={g.home_pts} won={g.winner === g.home_abbrev} />
        </div>
      ))}
    </div>
  );
}

function ScoreLine({ team, pts, won }: { team: string | null; pts: number | null; won: boolean }) {
  return (
    <div className="flex h-6 items-center justify-between">
      <span className={`text-body ${won ? "font-semibold text-fg" : "text-muted"}`}>
        {team ?? DASH}
      </span>
      <span className={`tnum text-body ${won ? "font-semibold text-fg" : "text-faint"}`}>
        {fmtInt(pts)}
      </span>
    </div>
  );
}

const EDGE_COLUMNS: Column<EdgeRow>[] = [
  {
    key: "player",
    header: "Player",
    render: (r) => <PropPlayerCell name={r.player_name} stat={r.stat_type} />,
    sortable: true,
    sortValue: (r) => r.player_name,
  },
  {
    key: "stat",
    header: "Stat",
    render: (r) => <span className="tnum text-muted">{statLabel(r.stat_type)}</span>,
  },
  { key: "book", header: "Book", render: (r) => <span className="text-muted">{r.book}</span> },
  {
    key: "line",
    header: "Line",
    align: "right",
    render: (r) => <span className="tnum text-fg">{fmtLine(r.book_line)}</span>,
    sortable: true,
    sortValue: (r) => r.book_line,
  },
  { key: "side", header: "Side", render: (r) => <SideTag side={r.best_side} /> },
  {
    key: "p",
    header: "P(best)",
    align: "right",
    width: "160px",
    render: (r) => <ProbabilityBar value={r.best_side === "under" ? r.p_under : r.p_over} />,
  },
  {
    key: "edge",
    header: "Edge",
    align: "right",
    render: (r) => <Delta value={r.model_edge} format={fmtSignedPct} strong />,
    sortable: true,
    sortValue: (r) => r.model_edge,
  },
  {
    key: "ev",
    header: "EV / unit",
    align: "right",
    render: (r) => <Delta value={r.ev_best} format={fmtEv} />,
    sortable: true,
    sortValue: (r) => r.ev_best,
  },
];

function TopEdges({ lastData }: { lastData: string | null }) {
  const navigate = useNavigate();
  // Full model mode, as the CLI/Streamlit "edge scanner full mode" does.
  const { data, isLoading, isError, error } = useEdges({ model_mode: "full", limit: 10 });

  if (isLoading) return <TableSkeleton rows={8} cols={7} />;
  if (isError) {
    return (
      <div className="p-4">
        <ErrorState message={errorMessage(error, "Edge scan failed.")} />
      </div>
    );
  }
  const rows = data?.rows ?? [];
  if (!rows.length) {
    return (
      <EmptyState
        title="No scored edges right now."
        hint="The edge scanner ranks scraped prop lines against the model. Outside the season, or before books post the day's player props, the slate is empty."
        lastData={lastData ? `scrape ${fmtAgo(lastData)}` : null}
      />
    );
  }
  return (
    <DataTable
      columns={EDGE_COLUMNS}
      rows={rows}
      rowKey={edgeRowKey}
      initialSort={{ key: "edge", dir: "desc" }}
      onRowClick={(r) => navigate(playerHref(r.player_name, r.stat_type))}
      rowActionLabel={(r) => `Open ${r.player_name} ${statLabel(r.stat_type)}`}
    />
  );
}

export function SlateDashboard() {
  const { data, isLoading, isError, error } = useSlateKpis();
  const live = (data?.prop_lines_recent ?? 0) > 0;

  return (
    <Page title="Slate Dashboard" description="Model coverage, the day's top model edges, and recent results.">
      {isError && <ErrorState message={errorMessage(error, "Failed to load slate KPIs.")} />}

      <KpiGrid cols={4}>
        {isLoading ? (
          <KpiRowSkeleton count={4} />
        ) : (
          <>
            <StatCard
              label="Games in DB"
              value={fmtInt(data?.games_in_db)}
              sub={`last game · ${fmtDateShort(data?.last_game_date)}`}
            />
            <StatCard label="Players tracked" value={fmtInt(data?.players_tracked)} sub="with game logs" />
            <StatCard
              label="Books producing"
              value={fmtInt(data?.books_producing)}
              sub={`${fmtInt(data?.prop_lines_recent)} prop lines · 48h`}
            />
            <StatCard
              label="Freshest scrape"
              value={data ? fmtAgo(data.freshest_scrape_utc) : DASH}
              sub="live-line sources"
              accent={live}
            />
          </>
        )}
      </KpiGrid>

      <Card title="Top model edges" meta="Full model · P(model) vs implied" flush>
        <TopEdges lastData={data?.freshest_scrape_utc ?? null} />
      </Card>

      <section className="space-y-3">
        <h2 className="text-title font-semibold text-fg">Recent games</h2>
        <RecentGamesStrip />
      </section>
    </Page>
  );
}
