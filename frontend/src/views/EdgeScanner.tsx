import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useEdges, useMeta, type EdgeParams } from "../api/hooks";
import type { EdgeRow } from "../api/types";
import { StatCard } from "../components/StatCard";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { TableSkeleton } from "../components/Skeleton";
import { Card, KpiGrid, Page } from "../components/Page";
import { Delta, SideTag } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { PropPlayerCell } from "../components/StarButton";
import { Chip, FilterRow, MODEL_MODES, NumberField, Segmented, Toggle } from "../components/controls";
import { DASH, fmtEv, fmtInt, fmtLine, fmtNum, fmtSignedPct, statLabel, titleCase } from "../lib/format";
import { edgeRowKey, playerHref, toggleItem } from "../lib/rows";

const COLUMNS: Column<EdgeRow>[] = [
  {
    key: "player_name",
    header: "Player",
    render: (r) => <PropPlayerCell name={r.player_name} stat={r.stat_type} />,
    sortable: true,
    sortValue: (r) => r.player_name,
  },
  {
    key: "stat_type",
    header: "Stat",
    render: (r) => <span className="tnum text-muted">{statLabel(r.stat_type)}</span>,
    sortable: true,
    sortValue: (r) => r.stat_type,
  },
  {
    key: "book",
    header: "Book",
    render: (r) => <span className="text-muted">{r.book}</span>,
    sortable: true,
    sortValue: (r) => r.book,
  },
  {
    key: "book_line",
    header: "Line",
    align: "right",
    render: (r) => <span className="tnum text-fg">{fmtLine(r.book_line)}</span>,
    sortable: true,
    sortValue: (r) => r.book_line,
  },
  {
    key: "model_mu",
    header: "Model mean",
    align: "right",
    render: (r) => <span className="tnum text-muted">{fmtNum(r.model_mu)}</span>,
    sortable: true,
    sortValue: (r) => r.model_mu,
  },
  {
    key: "line_vs_mu",
    header: "Line − mean",
    align: "right",
    render: (r) => <span className="tnum text-faint">{fmtNum(r.line_vs_mu)}</span>,
    sortable: true,
    sortValue: (r) => r.line_vs_mu,
  },
  { key: "best_side", header: "Side", render: (r) => <SideTag side={r.best_side} /> },
  {
    key: "p_over",
    header: "P(best)",
    align: "right",
    width: "160px",
    render: (r) => <ProbabilityBar value={r.best_side === "under" ? r.p_under : r.p_over} />,
    sortable: true,
    sortValue: (r) => Math.max(r.p_over ?? 0, r.p_under ?? 0),
  },
  {
    key: "model_edge",
    header: "Edge",
    align: "right",
    render: (r) => <Delta value={r.model_edge} format={fmtSignedPct} strong />,
    sortable: true,
    sortValue: (r) => r.model_edge,
  },
  {
    key: "ev_best",
    header: "EV / unit",
    align: "right",
    render: (r) => <Delta value={r.ev_best} format={fmtEv} />,
    sortable: true,
    sortValue: (r) => r.ev_best,
  },
  {
    key: "distribution",
    header: "Dist",
    render: (r) => <span className="text-caption text-faint">{r.distribution ?? DASH}</span>,
  },
];

export function EdgeScanner() {
  const navigate = useNavigate();
  const meta = useMeta();

  const [modelMode, setModelMode] = useState("full");
  const [books, setBooks] = useState<string[]>([]);
  const [stats, setStats] = useState<string[]>([]);
  const [minEdge, setMinEdge] = useState(0);
  const [minPOver, setMinPOver] = useState(0);
  const [onlyPositiveEv, setOnlyPositiveEv] = useState(false);

  const params: EdgeParams = useMemo(
    () => ({
      model_mode: modelMode,
      books: books.length ? books : undefined,
      stats: stats.length ? stats : undefined,
      min_edge: minEdge > 0 ? minEdge : undefined,
      min_p_over: minPOver > 0 ? minPOver : undefined,
      only_positive_ev: onlyPositiveEv,
      limit: 300,
    }),
    [modelMode, books, stats, minEdge, minPOver, onlyPositiveEv],
  );

  const { data, isLoading, isError, error, isFetching } = useEdges(params);

  const allBooks = data?.books_available ?? meta.data?.books ?? [];
  const allStats = data?.stats_available ?? meta.data?.stats ?? [];
  const rows = data?.rows ?? [];

  return (
    <Page
      title="Edge Scanner"
      description="The scored slate ranked by model-vs-line edge. Model edge only — not arbitrage."
      actions={
        isFetching ? (
          <span className="text-label text-faint" role="status">
            Scanning…
          </span>
        ) : null
      }
    >
      <KpiGrid cols={4}>
        <StatCard label="Lines scanned" value={data ? fmtInt(data.n_lines) : DASH} />
        <StatCard label="Scored" value={data ? fmtInt(data.n_scored) : DASH} />
        <StatCard label="Shown" value={data ? fmtInt(data.n_returned) : DASH} />
        <StatCard label="Model mode" value={titleCase(data?.model_mode ?? modelMode)} />
      </KpiGrid>

      <Card title="Filters">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-6">
            <Segmented label="Model mode" options={MODEL_MODES} value={modelMode} onChange={setModelMode} />
            <NumberField label="Min edge" value={minEdge} onChange={setMinEdge} step={0.01} min={0} max={1} />
            <NumberField label="Min P(over)" value={minPOver} onChange={setMinPOver} step={0.01} min={0} max={1} />
            <Toggle label="Only +EV" checked={onlyPositiveEv} onChange={setOnlyPositiveEv} />
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

      <Card title="Scored props" meta={data ? `${fmtInt(rows.length)} rows` : undefined} flush>
        {isError ? (
          <div className="p-4">
            <ErrorState message={errorMessage(error, "Edge scan failed.")} />
          </div>
        ) : isLoading ? (
          <TableSkeleton rows={10} cols={9} />
        ) : rows.length ? (
          <DataTable
            columns={COLUMNS}
            rows={rows}
            rowKey={edgeRowKey}
            initialSort={{ key: "model_edge", dir: "desc" }}
            maxHeight="calc(100vh - 220px)"
            onRowClick={(r) => navigate(playerHref(r.player_name, r.stat_type))}
            rowActionLabel={(r) => `Open ${r.player_name} ${statLabel(r.stat_type)}`}
          />
        ) : (
          <EmptyState
            title="No props cleared the current filters."
            hint="The scanner scores scraped prop lines against the model. Outside the season (and before books post a game's props) the slate is empty — loosen filters once lines return."
          />
        )}
      </Card>
    </Page>
  );
}
