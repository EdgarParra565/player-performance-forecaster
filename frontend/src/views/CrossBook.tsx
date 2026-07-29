import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCrossBook, useMeta, type CrossBookParams } from "../api/hooks";
import type { ArbRow, CrossBookRow } from "../api/types";
import { StatCard } from "../components/StatCard";
import { DataTable, type Column } from "../components/DataTable";
import { CrossBookTable } from "../components/CrossBookTable";
import { EmptyState } from "../components/EmptyState";
import { Loading, ErrorState } from "../components/Loading";
import { OddsBadge } from "../components/OddsBadge";
import { Delta } from "../components/Delta";
import { Chip, Segmented, NumberField } from "../components/controls";
import { downloadCsv } from "../lib/csv";
import { fmtInt, fmtNum, fmtSignedPct, statLabel } from "../lib/format";

const MODEL_MODES = [
  { key: "chart_mean", label: "Chart mean" },
  { key: "rolling", label: "Rolling" },
  { key: "full", label: "Full (beta)" },
];

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
      render: (r) => <span className="text-fg">{r.player_name}</span>,
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
        <span className="text-muted">
          {r.over_book} <span className="tnum">{fmtNum(r.over_line)}</span>{" "}
          <OddsBadge odds={r.over_odds} />
        </span>
      ),
    },
    {
      key: "under",
      header: "UNDER leg",
      render: (r) => (
        <span className="text-muted">
          {r.under_book} <span className="tnum">{fmtNum(r.under_line)}</span>{" "}
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

  const { data, isLoading, isError, isFetching } = useCrossBook(params);
  const rows = data?.rows ?? [];
  const arbs = data?.arbs ?? [];
  const allBooks = data?.books_available ?? meta.data?.books ?? [];
  const allStats = data?.stats_available ?? meta.data?.stats ?? [];

  function toggle(list: string[], setter: (v: string[]) => void, item: string) {
    setter(list.includes(item) ? list.filter((x) => x !== item) : [...list, item]);
  }

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
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-lg font-semibold text-fg">Cross-book</h1>
          <p className="mt-0.5 text-xs text-faint">
            Line shopping &amp; middle candidates from the DFS board, plus TRUE
            two-way arbitrage from real posted odds. Middles are candidates, not
            guaranteed.
          </p>
        </div>
        {isFetching && <span className="eyebrow text-faint">scanning…</span>}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <StatCard label="Pairs (2+ books)" value={fmtInt(data?.kpis.pairs ?? 0)} />
        <StatCard label="Max gap" value={fmtNum(data?.kpis.max_gap ?? 0)} />
        <StatCard label={`≥ ${minGap} gap`} value={fmtInt(data?.kpis.over_threshold ?? 0)} />
        <StatCard
          label="True arbs"
          value={fmtInt(data?.kpis.arb_count ?? 0)}
          tone={(data?.kpis.arb_count ?? 0) > 0 ? "pos" : "default"}
          accent={(data?.kpis.arb_count ?? 0) > 0}
          sub="real odds only"
        />
        <StatCard
          label="Freshest line"
          value={
            data?.kpis.freshest_hours != null
              ? `${fmtNum(data.kpis.freshest_hours)}h`
              : "—"
          }
          sub="scraped ago"
        />
      </div>

      {/* Filters */}
      <div className="panel space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="eyebrow">Model</span>
            <Segmented options={MODEL_MODES} value={modelMode} onChange={setModelMode} />
          </div>
          <NumberField label="min gap" value={minGap} onChange={setMinGap} step={0.5} min={0} />
          <NumberField label="min books" value={minBooks} onChange={setMinBooks} min={2} max={12} />
          <button
            onClick={exportCsv}
            disabled={!rows.length}
            className="tnum rounded border border-line px-2.5 py-1 text-[11px] text-muted enabled:hover:border-line-strong enabled:hover:text-fg disabled:opacity-40"
          >
            ↓ CSV
          </button>
        </div>
        <div className="flex flex-wrap items-start gap-2">
          <span className="eyebrow mt-1 w-12">Books</span>
          <div className="flex flex-1 flex-wrap gap-1">
            {allBooks.map((b) => (
              <Chip key={b} active={books.includes(b)} onClick={() => toggle(books, setBooks, b)}>
                {b}
              </Chip>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-start gap-2">
          <span className="eyebrow mt-1 w-12">Stats</span>
          <div className="flex flex-1 flex-wrap gap-1">
            {allStats.map((s) => (
              <Chip key={s} active={stats.includes(s)} onClick={() => toggle(stats, setStats, s)}>
                {statLabel(s)}
              </Chip>
            ))}
          </div>
        </div>
      </div>

      {/* Line shopping / middles */}
      <section className="panel">
        <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
          <h2 className="eyebrow">Line shopping &amp; middle candidates</h2>
          <span className="text-[11px] text-faint">
            {fmtInt(rows.length)} shown · {fmtInt(data?.n_lines ?? 0)} lines scanned
          </span>
        </div>
        {isError ? (
          <div className="p-4">
            <ErrorState message="Cross-book scan failed." />
          </div>
        ) : isLoading ? (
          <Loading rows={8} />
        ) : rows.length ? (
          <CrossBookTable
            rows={rows}
            onRowClick={(r) =>
              navigate(
                `/player?name=${encodeURIComponent(r.player_name)}&stat=${r.stat_type}`,
              )
            }
          />
        ) : (
          <EmptyState
            title="No cross-book opportunities."
            hint="Needs 2+ books quoting the same player+stat within the lookback window. During the NBA offseason the DFS board is empty, so this fills in once lines return in October."
            lastData={
              data?.kpis.freshest_hours != null
                ? `freshest line ${fmtNum(data.kpis.freshest_hours)}h ago`
                : null
            }
          />
        )}
      </section>

      {/* TRUE arb — visually distinct */}
      <section className="rounded-md border border-pos-dim/40 bg-pos-soft/30">
        <div className="flex items-center justify-between border-b border-pos-dim/40 px-4 py-2.5">
          <h2 className="eyebrow text-pos">True two-way arbitrage · real odds only</h2>
          <span className="tnum text-[11px] text-pos">
            {fmtInt(arbs.length)} locked
          </span>
        </div>
        <ArbSection arbs={arbs} />
      </section>

      <p className="text-[11px] leading-relaxed text-faint">
        Line shopping / middle rows come from the DFS board (assumed -110) and are
        NEVER guaranteed profit. Only the true-arb section — sourced from real
        posted odds on both legs with raw implied sum &lt; 1.0 in an executable
        direction — is locked profit. P(over) shown at each end uses the row's
        fitted model, for reference.
      </p>
    </div>
  );
}
