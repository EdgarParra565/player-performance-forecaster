import { useNavigate } from "react-router-dom";
import { useEdges, useRecentGames, useSlateKpis } from "../api/hooks";
import { StatCard } from "../components/StatCard";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { Loading, ErrorState } from "../components/Loading";
import { Delta } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import type { EdgeRow, RecentGame } from "../api/types";
import {
  fmtAgo,
  fmtDateShort,
  fmtInt,
  fmtNum,
  fmtSigned,
  fmtSignedPct,
  statLabel,
} from "../lib/format";

function RecentGamesStrip() {
  const { data, isLoading, isError } = useRecentGames(10);
  if (isLoading) return <Loading rows={2} />;
  if (isError) return <ErrorState message="Failed to load recent games." />;
  const rows = data?.rows ?? [];
  if (!rows.length) {
    return <EmptyState compact title="No games in the window." />;
  }
  return (
    <div className="flex gap-2 overflow-x-auto pb-1">
      {rows.map((g: RecentGame) => {
        const homeWon = g.winner === g.home_abbrev;
        return (
          <div
            key={g.game_id}
            className="panel min-w-[150px] shrink-0 px-3 py-2.5"
          >
            <div className="eyebrow mb-1.5">{fmtDateShort(g.game_date)}</div>
            <ScoreLine
              team={g.away_abbrev}
              pts={g.away_pts}
              won={g.winner === g.away_abbrev}
            />
            <ScoreLine team={g.home_abbrev} pts={g.home_pts} won={homeWon} />
          </div>
        );
      })}
    </div>
  );
}

function ScoreLine({
  team,
  pts,
  won,
}: {
  team: string | null;
  pts: number | null;
  won: boolean;
}) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span
        className={`text-xs ${won ? "font-semibold text-fg" : "text-muted"}`}
      >
        {team ?? "—"}
      </span>
      <span
        className={`tnum text-xs ${won ? "text-pos" : "text-muted"}`}
      >
        {pts != null ? Math.round(pts) : "—"}
      </span>
    </div>
  );
}

function TopEdges({ lastData }: { lastData: string | null }) {
  const navigate = useNavigate();
  // Full model mode, as the CLI/Streamlit "edge scanner full mode" does.
  const { data, isLoading, isError } = useEdges({
    model_mode: "full",
    limit: 10,
  });

  if (isLoading) return <Loading label="Top model edges" rows={6} />;
  if (isError) return <ErrorState message="Edge scan failed." />;

  const rows = data?.rows ?? [];
  if (!rows.length) {
    return (
      <EmptyState
        title="No scored edges right now."
        hint="The edge scanner ranks scraped prop lines against the model. During the NBA offseason the books post no player props, so the slate is empty until October."
        lastData={lastData ? `scrape ${fmtAgo(lastData)}` : null}
      />
    );
  }

  const columns: Column<EdgeRow>[] = [
    {
      key: "player",
      header: "Player",
      render: (r) => <span className="text-fg">{r.player_name}</span>,
      sortable: true,
      sortValue: (r) => r.player_name,
    },
    {
      key: "stat",
      header: "Stat",
      render: (r) => (
        <span className="tnum text-muted">{statLabel(r.stat_type)}</span>
      ),
    },
    {
      key: "book",
      header: "Book",
      render: (r) => <span className="text-muted">{r.book}</span>,
    },
    {
      key: "line",
      header: "Line",
      align: "right",
      render: (r) => <span className="tnum">{fmtNum(r.book_line)}</span>,
      sortable: true,
      sortValue: (r) => r.book_line,
    },
    {
      key: "side",
      header: "Side",
      render: (r) => (
        <span
          className={`tnum uppercase ${
            r.best_side === "over" ? "text-pos" : "text-neg"
          }`}
        >
          {r.best_side ?? "—"}
        </span>
      ),
    },
    {
      key: "p",
      header: "P(best)",
      align: "right",
      render: (r) => (
        <div className="w-24">
          <ProbabilityBar
            value={r.best_side === "under" ? r.p_under : r.p_over}
          />
        </div>
      ),
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
      header: "EV/u",
      align: "right",
      render: (r) => <Delta value={r.ev_best} format={fmtSigned} />,
      sortable: true,
      sortValue: (r) => r.ev_best,
    },
  ];

  return (
    <DataTable
      columns={columns}
      rows={rows}
      rowKey={(r) => `${r.book}-${r.player_name}-${r.stat_type}`}
      initialSort={{ key: "edge", dir: "desc" }}
      onRowClick={(r) =>
        navigate(
          `/player?name=${encodeURIComponent(r.player_name)}&stat=${r.stat_type}`,
        )
      }
    />
  );
}

export function SlateDashboard() {
  const { data, isLoading, isError } = useSlateKpis();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-fg">Slate Dashboard</h1>
        <p className="mt-0.5 text-xs text-faint">
          Model coverage, top edges, and recent results.
        </p>
      </div>

      {isError && <ErrorState message="Failed to load slate KPIs." />}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {isLoading || !data ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="panel h-[86px] animate-pulse" />
          ))
        ) : (
          <>
            <StatCard
              label="Games in DB"
              value={fmtInt(data.games_in_db)}
              sub={`last game · ${fmtDateShort(data.last_game_date)}`}
            />
            <StatCard
              label="Players tracked"
              value={fmtInt(data.players_tracked)}
              sub="with game logs"
            />
            <StatCard
              label="Books producing"
              value={fmtInt(data.books_producing)}
              sub={`${fmtInt(data.prop_lines_recent)} prop lines · 48h`}
            />
            <StatCard
              label="Freshest scrape"
              value={fmtAgo(data.freshest_scrape_utc)}
              sub="live-line sources"
              tone={data.prop_lines_recent > 0 ? "pos" : "default"}
              accent={data.prop_lines_recent > 0}
            />
          </>
        )}
      </div>

      <section className="panel">
        <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
          <h2 className="eyebrow">Top model edges · full mode</h2>
          <span className="text-[11px] text-faint">P(model) vs implied</span>
        </div>
        <TopEdges lastData={data?.freshest_scrape_utc ?? null} />
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="eyebrow">Recent games</h2>
        </div>
        <RecentGamesStrip />
      </section>
    </div>
  );
}
