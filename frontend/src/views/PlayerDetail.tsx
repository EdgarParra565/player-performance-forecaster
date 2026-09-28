import { useEffect, useMemo } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useMeta, usePlayerDetail, usePlayerSearch } from "../api/hooks";
import type { BookLineRow } from "../api/types";
import { StatCard } from "../components/StatCard";
import { EChart } from "../components/EChart";
import { LineMovementPanel } from "../components/LineMovementPanel";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { ChartSkeleton, KpiRowSkeleton } from "../components/Skeleton";
import { Card, KpiGrid, Page } from "../components/Page";
import { Delta, SideTag } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { OddsBadge } from "../components/OddsBadge";
import { NumberField, Segmented } from "../components/controls";
import { PlayerPicker } from "../components/PlayerPicker";
import { StarButton } from "../components/StarButton";
import { distributionOption, hitRateOption, performanceOption } from "../charts/playerCharts";
import { DASH, fmtAgo, fmtEv, fmtInt, fmtLine, fmtNum, fmtPct, fmtSignedPct, statLabel } from "../lib/format";
import { clampInt, nameKey } from "../lib/rows";

// --- Per-book line table ----------------------------------------------------

const BOOK_COLUMNS: Column<BookLineRow>[] = [
  {
    key: "book",
    header: "Book",
    render: (r) => <span className="font-medium text-fg">{r.book}</span>,
    sortable: true,
    sortValue: (r) => r.book,
  },
  {
    key: "line",
    header: "Line",
    align: "right",
    render: (r) => <span className="tnum text-fg">{fmtLine(r.line)}</span>,
    sortable: true,
    sortValue: (r) => r.line,
  },
  {
    key: "odds",
    header: "Over / Under",
    align: "right",
    render: (r) => (
      <span className="inline-flex gap-1">
        <OddsBadge odds={r.over_odds} dfs={r.is_dfs} />
        <OddsBadge odds={r.under_odds} dfs={r.is_dfs} />
      </span>
    ),
  },
  {
    key: "p_over",
    header: "P(over)",
    align: "right",
    width: "150px",
    render: (r) => <ProbabilityBar value={r.p_over} marker={r.breakeven} />,
    sortable: true,
    sortValue: (r) => r.p_over,
  },
  {
    key: "hit",
    header: "Hit rate",
    align: "right",
    render: (r) => <span className="tnum text-muted">{fmtPct(r.hit_rate)}</span>,
    sortable: true,
    sortValue: (r) => r.hit_rate,
  },
  { key: "side", header: "Best side", render: (r) => <SideTag side={r.best_side} /> },
  {
    key: "edge",
    header: "Edge",
    align: "right",
    render: (r) => <Delta value={r.model_edge} format={fmtSignedPct} strong />,
    sortable: true,
    sortValue: (r) => r.model_edge,
  },
  {
    // EV of the SAME side the edge refers to (EV-over next to a best-side
    // edge read as contradictory green/red).
    key: "ev",
    header: "EV / unit",
    align: "right",
    render: (r) => <Delta value={bestSideEv(r)} format={fmtEv} />,
    sortable: true,
    sortValue: (r) => bestSideEv(r),
  },
];

function bestSideEv(r: BookLineRow): number | null {
  return r.best_side === "under" ? r.ev_under : r.ev_over;
}

// --- Name deep-link resolution ---------------------------------------------

// `/player?name=X` (from edge / cross-book rows) resolves ONLY on an exact
// (accent/case-insensitive) name match — a fuzzy rows[0] fallback silently
// loaded a different player's EV.
function NameResolver({ name, stat }: { name: string; stat: string }) {
  const navigate = useNavigate();
  const { data, isLoading, isError, error } = usePlayerSearch(name);
  const rows = useMemo(() => data?.rows ?? [], [data]);
  const exact = rows.find((r) => nameKey(r.player_name) === nameKey(name));

  useEffect(() => {
    if (exact) navigate(`/player/${exact.player_id}?stat=${encodeURIComponent(stat)}`, { replace: true });
  }, [exact, navigate, stat]);

  if (isLoading || exact) return <KpiGrid cols={6}><KpiRowSkeleton count={6} /></KpiGrid>;
  if (isError) return <ErrorState message={errorMessage(error, "Player lookup failed.")} />;
  return (
    <Card>
      <EmptyState
        title={`No exact match for “${name}”.`}
        hint={rows.length ? "Did you mean one of these?" : "Search for the player above."}
      />
      {rows.length > 0 && (
        <div className="flex flex-wrap justify-center gap-2 pb-6">
          {rows.slice(0, 8).map((r) => (
            <button
              type="button"
              key={r.player_id}
              onClick={() => navigate(`/player/${r.player_id}?stat=${encodeURIComponent(stat)}`)}
              className="h-8 rounded-full border border-line bg-surface-2 px-3 text-label text-muted hover:border-line-strong hover:text-fg pointer-coarse:h-11"
            >
              {r.player_name}
              {r.team && <span className="tnum ml-2 text-faint">{r.team}</span>}
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}

// --- View -------------------------------------------------------------------

function parsePlayerId(raw: string | undefined): number | null {
  if (!raw || !/^\d+$/.test(raw)) return null;
  const n = Number(raw);
  return n >= 1 && n <= 2 ** 31 - 1 ? n : null;
}

export function PlayerDetail() {
  const navigate = useNavigate();
  const params = useParams();
  const [search, setSearch] = useSearchParams();
  const meta = useMeta();

  // URL is the source of truth: back/forward and shared links just work.
  const playerId = parsePlayerId(params.playerId);
  const stat = search.get("stat") ?? "points";
  const nameParam = search.get("name");
  const nGames = clampInt(search.get("n"), 3, 200, 25);
  const rollingWindow = clampInt(search.get("roll"), 1, 60, 5);

  function setParam(key: string, value: string) {
    setSearch(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set(key, value);
        return next;
      },
      { replace: true },
    );
  }
  const setStat = (s: string) => setParam("stat", s);
  const setNGames = (n: number) => setParam("n", String(n));
  const setRollingWindow = (n: number) => setParam("roll", String(n));

  const detail = usePlayerDetail(
    playerId !== null ? { playerId, stat, n_games: nGames, rolling_window: rollingWindow } : null,
  );
  const d = detail.data;

  const stats = useMemo(() => meta.data?.stats ?? ["points"], [meta.data]);
  const statOptions = stats.map((s) => ({ key: s, label: statLabel(s) }));

  const perf = useMemo(() => (d && d.n_games ? performanceOption(d) : null), [d]);
  const dist = useMemo(() => (d && d.n_games ? distributionOption(d) : null), [d]);
  const hit = useMemo(
    () => (d && d.book_lines.some((b) => b.hit_rate != null) ? hitRateOption(d.book_lines) : null),
    [d],
  );
  const stale = detail.isPlaceholderData;

  return (
    <Page
      eyebrow={d ? `Player · ${statLabel(d.stat_type)}` : "Player"}
      title={
        d ? (
          <span className="inline-flex items-center gap-2">
            {d.player_name}
            <StarButton size="md" item={{ kind: "player", player_name: d.player_name, player_id: d.player_id }} />
          </span>
        ) : (
          nameParam || "Player Detail"
        )
      }
      description="Recent form, distribution, and per-book pricing."
      actions={
        <PlayerPicker
          onSelect={(r) =>
            navigate(`/player/${r.player_id}?stat=${encodeURIComponent(stat)}&n=${nGames}&roll=${rollingWindow}`)
          }
        />
      }
    >
      {playerId === null && nameParam && <NameResolver name={nameParam} stat={stat} />}

      {playerId === null && !nameParam && (
        <Card>
          <EmptyState
            title="Search for a player to begin."
            hint="Charts recent-N performance with a rolling mean and book-line overlay, a fitted distribution, and every book's P(over) / EV."
          />
        </Card>
      )}

      {playerId !== null && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Segmented label="Stat" options={statOptions} value={stat} onChange={setStat} />
              {d && (
                <StarButton
                  size="md"
                  item={{ kind: "prop", player_name: d.player_name, player_id: d.player_id, stat: d.stat_type }}
                />
              )}
            </div>
            <div className="flex items-center gap-4">
              <NumberField label="Games" value={nGames} onChange={setNGames} min={3} max={200} />
              <NumberField label="Rolling" value={rollingWindow} onChange={setRollingWindow} min={1} max={60} />
            </div>
          </div>

          {detail.isError && <ErrorState message={errorMessage(detail.error, "Failed to load player detail.")} />}

          {detail.isLoading && (
            <>
              <KpiGrid cols={6}>
                <KpiRowSkeleton count={6} />
              </KpiGrid>
              <div className="panel">
                <ChartSkeleton height={280} />
              </div>
            </>
          )}

          {d && d.n_games === 0 && (
            <Card>
              <EmptyState
                title="No game logs for this player + stat."
                hint="Pick another stat or player; this combination has no history in the DB yet."
              />
            </Card>
          )}

          {d && d.n_games > 0 && (
            <div className={`space-y-6 transition-opacity ${stale ? "opacity-60" : ""}`} aria-busy={stale}>
              <KpiGrid cols={6}>
                <StatCard label={`${statLabel(d.stat_type)} mean`} value={fmtNum(d.kpis.mu)} sub={`last ${d.n_games} games`} />
                <StatCard label="Std dev" value={fmtNum(d.kpis.sigma)} sub="σ" />
                <StatCard label="Book mean" value={fmtLine(d.kpis.market_consensus_line)} sub="consensus line" />
                <StatCard label="Books" value={fmtInt(d.kpis.n_books)} sub="posting a line" />
                <StatCard
                  label="+EV sides"
                  value={fmtInt(d.kpis.positive_ev_sides)}
                  tone={d.kpis.positive_ev_sides > 0 ? "pos" : "default"}
                  accent={d.kpis.positive_ev_sides > 0}
                  sub="model vs price"
                />
                <StatCard
                  label="Lines age"
                  value={d.last_line_scraped_utc ? fmtAgo(d.last_line_scraped_utc) : DASH}
                  sub="freshest book"
                />
              </KpiGrid>

              <Card title={`Last ${d.n_games} games`} meta={`rolling mean · ${d.rolling_window}-game window`}>
                {perf && (
                  <EChart
                    option={perf}
                    height={300}
                    ariaLabel={`${d.player_name} ${statLabel(d.stat_type)} over the last ${d.n_games} games`}
                  />
                )}
              </Card>

              <Card title="Per-book lines" meta="P(over) vs break-even · edge & EV on the best side" flush>
                {d.book_lines.length ? (
                  <DataTable
                    columns={BOOK_COLUMNS}
                    rows={d.book_lines}
                    rowKey={(r) => `${r.book}-${r.line}`}
                    initialSort={{ key: "edge", dir: "desc" }}
                  />
                ) : (
                  <EmptyState
                    compact
                    title="No book lines for this stat."
                    hint="Scraped prop lines appear here once the books post this player + stat."
                  />
                )}
              </Card>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <Card title="Distribution" meta="fitted normal · dotted = book lines">
                  {dist && <EChart option={dist} height={300} ariaLabel="Distribution with fitted normal" />}
                </Card>
                <Card title="Over-rate vs each book line" meta="dashed = 50%">
                  {hit ? (
                    <EChart option={hit} height={300} ariaLabel="Over rate per book" />
                  ) : (
                    <EmptyState compact title="No book lines to compare." />
                  )}
                </Card>
              </div>

              <Card title="Line movement" meta="snapshot replay · per-book drift" flush>
                <LineMovementPanel key={`${playerId}-${stat}`} playerId={playerId} stat={stat} />
              </Card>

              {d.notes.length > 0 && (
                <ul className="space-y-1 text-label text-faint">
                  {d.notes.map((n, i) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </Page>
  );
}
