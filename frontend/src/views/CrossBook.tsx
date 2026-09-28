import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCrossBook, useMeta, type CrossBookParams } from "../api/hooks";
import type { ArbRow, CrossBookRow } from "../api/types";
import { StatCard } from "../components/StatCard";
import { DataTable, type Column } from "../components/DataTable";
import { CrossBookTable } from "../components/CrossBookTable";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { TableSkeleton } from "../components/Skeleton";
import { Card, KpiGrid, Page } from "../components/Page";
import { OddsBadge } from "../components/OddsBadge";
import { Delta } from "../components/Delta";
import { Button, Chip, FilterRow, MODEL_MODES, NumberField, Segmented } from "../components/controls";
import { downloadCsv } from "../lib/csv";
import { DASH, fmtHoursAgo, fmtInt, fmtLine, fmtNum, fmtSignedPct, statLabel } from "../lib/format";
import { playerHref, toggleItem } from "../lib/rows";

function ArbSection({ arbs }: { arbs: ArbRow[] }) {
  if (!arbs.length) {
    return (
      <EmptyState
        compact
        title="No true two-way arbitrage."
        hint="A true arb needs REAL posted odds on both legs whose raw implied probabilities sum below 1.0. DFS -110 defaults can never qualify — this section only fires from betting_lines odds."
      />
    );
  }
  const columns: Column<ArbRow>[] = [
    {
      key: "player_name",
      header: "Player",
      render: (r) => <span className="font-medium text-fg">{r.player_name}</span>,
      sortable: true,
      sortValue: (r) => r.player_name,
    },
    {
      key: "stat_type",
      header: "Stat",
      render: (r) => <span className="tnum text-muted">{statLabel(r.stat_type)}</span>,
    },
    {
      key: "over",
      header: "OVER leg",
      render: (r) => (
        <span className="inline-flex items-center gap-2 text-muted">
          {r.over_book} <span className="tnum text-fg">o{fmtLine(r.over_line)}</span>
          <OddsBadge odds={r.over_odds} />
        </span>
      ),
    },
    {
      key: "under",
      header: "UNDER leg",
      render: (r) => (
        <span className="inline-flex items-center gap-2 text-muted">
          {r.under_book} <span className="tnum text-fg">u{fmtLine(r.under_line)}</span>
          <OddsBadge odds={r.under_odds} />
        </span>
      ),
    },
    {
      key: "combined",
      header: "Σ implied",
      align: "right",
      render: (r) => <span className="tnum text-fg">{fmtNum(r.combined_implied, 4)}</span>,
      sortable: true,
      sortValue: (r) => r.combined_implied,
    },
    {
      key: "margin",
      header: "Locked",
      align: "right",
      render: (r) => <Delta value={r.guaranteed_margin} format={fmtSignedPct} strong />,
      sortable: true,
      sortValue: (r) => r.guaranteed_margin,
    },
  ];
  return (
    <DataTable
      columns={columns}
      rows={arbs}
      rowKey={(r, i) => `${r.player_name}-${r.stat_type}-${i}`}
      initialSort={{ key: "margin", dir: "desc" }}
    />
  );
}

export function CrossBook() {
  const navigate = useNavigate();
  const meta = useMeta();

  const [modelMode, setModelMode] = useState("chart_mean");
  const [minGap, setMinGap] = useState(0.5);
  const [minBooks, setMinBooks] = useState(2);
  const [books, setBooks] = useState<string[]>([]);
  const [stats, setStats] = useState<string[]>([]);

  const params: CrossBookParams = useMemo(
    () => ({
      model_mode: modelMode,
      min_gap: minGap,
      min_books: minBooks,
      books: books.length ? books : undefined,
      stats: stats.length ? stats : undefined,
    }),
    [modelMode, minGap, minBooks, books, stats],
  );

  const { data, isLoading, isError, error, isFetching } = useCrossBook(params);
  const rows = data?.rows ?? [];
  const arbs = data?.arbs ?? [];
  const allBooks = data?.books_available ?? meta.data?.books ?? [];
  const allStats = data?.stats_available ?? meta.data?.stats ?? [];
  const arbCount = data?.kpis.arb_count ?? 0;

  function exportCsv() {
    downloadCsv<CrossBookRow>("cross_book.csv", rows, [
      { key: "player_name", header: "player", value: (r) => r.player_name },
      { key: "stat_type", header: "stat", value: (r) => r.stat_type },
      { key: "n_books", header: "n_books", value: (r) => r.n_books },
      { key: "line_min", header: "line_min", value: (r) => r.line_min },
      { key: "line_max", header: "line_max", value: (r) => r.line_max },
      { key: "line_gap", header: "line_gap", value: (r) => r.line_gap },
      { key: "best_over_book", header: "best_over_book", value: (r) => r.best_over_book },
      { key: "best_under_book", header: "best_under_book", value: (r) => r.best_under_book },
      { key: "consensus_mean", header: "consensus_mean", value: (r) => r.consensus_mean },
      { key: "opportunity_type", header: "opportunity_type", value: (r) => r.opportunity_type },
    ]);
  }

  return (
    <Page
      title="Cross-book"
      description="Line shopping and middle candidates from the DFS board, plus TRUE two-way arbitrage from real posted odds. Middles are candidates, never guaranteed."
      actions={
        <>
          {isFetching && (
            <span className="text-label text-faint" role="status">
              Scanning…
            </span>
          )}
          <Button onClick={exportCsv} disabled={!rows.length} title="Download the line-shopping table">
            <span aria-hidden>↓</span> Export CSV
          </Button>
        </>
      }
    >
      <KpiGrid cols={5}>
        <StatCard label="Pairs (2+ books)" value={data ? fmtInt(data.kpis.pairs) : DASH} />
        <StatCard label="Max gap" value={data ? fmtLine(data.kpis.max_gap) : DASH} />
        <StatCard label={`Gap ≥ ${fmtNum(minGap)}`} value={data ? fmtInt(data.kpis.over_threshold) : DASH} />
        <StatCard
          label="True arbs"
          value={data ? fmtInt(arbCount) : DASH}
          tone={arbCount > 0 ? "pos" : "default"}
          accent={arbCount > 0}
          sub="real odds only"
        />
        <StatCard label="Freshest line" value={fmtHoursAgo(data?.kpis.freshest_hours)} sub="scraped" />
      </KpiGrid>

      <Card title="Filters">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-6">
            <Segmented label="Model mode" options={MODEL_MODES} value={modelMode} onChange={setModelMode} />
            <NumberField label="Min gap" value={minGap} onChange={setMinGap} step={0.5} min={0} max={100} />
            <NumberField label="Min books" value={minBooks} onChange={setMinBooks} min={2} max={12} />
          </div>
          <FilterRow label="Books">
            {allBooks.map((b) => (
              <Chip key={b} active={books.includes(b)} onClick={() => setBooks(toggleItem(books, b))}>
                {b}
              </Chip>
            ))}
          </FilterRow>
          <FilterRow label="Stats">
            {allStats.map((s) => (
              <Chip key={s} active={stats.includes(s)} onClick={() => setStats(toggleItem(stats, s))}>
                {statLabel(s)}
              </Chip>
            ))}
          </FilterRow>
        </div>
      </Card>

      <Card
        title="Line shopping & middle candidates"
        meta={data ? `${fmtInt(rows.length)} shown · ${fmtInt(data.n_lines)} lines scanned` : undefined}
        flush
      >
        {isError ? (
          <div className="p-4">
            <ErrorState message={errorMessage(error, "Cross-book scan failed.")} />
          </div>
        ) : isLoading ? (
          <TableSkeleton rows={8} cols={10} />
        ) : rows.length ? (
          <CrossBookTable rows={rows} onRowClick={(r) => navigate(playerHref(r.player_name, r.stat_type))} />
        ) : (
          <EmptyState
            title="No cross-book opportunities."
            hint="Needs 2+ books quoting the same player + stat within the lookback window. During the NBA offseason the DFS board is empty, so this fills in once lines return in October."
            lastData={
              data?.kpis.freshest_hours != null ? `freshest line ${fmtHoursAgo(data.kpis.freshest_hours)}` : null
            }
          />
        )}
      </Card>

      {/* TRUE arb — the one surface whose whole meaning is locked +EV. */}
      <Card
        tone="pos"
        title="True two-way arbitrage · real odds only"
        meta={<span className="tnum text-pos">{fmtInt(arbs.length)} locked</span>}
        flush
      >
        <ArbSection arbs={arbs} />
      </Card>

      <p className="max-w-4xl text-label text-faint">
        Line shopping / middle rows come from the DFS board (assumed -110) and are NEVER guaranteed
        profit. Only the true-arb section — sourced from real posted odds on both legs with raw
        implied sum &lt; 1.0 in an executable direction — is locked profit. P(over) shown at each end
        uses the row's fitted model, for reference.
      </p>
    </Page>
  );
}
