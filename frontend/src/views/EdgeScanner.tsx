import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useEdges, useMeta, type EdgeParams } from "../api/hooks";
import type { EdgeRow } from "../api/types";
import { StatCard } from "../components/StatCard";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { Loading, ErrorState } from "../components/Loading";
import { Delta } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import {
  fmtInt,
  fmtNum,
  fmtSigned,
  fmtSignedPct,
  statLabel,
  titleCase,
} from "../lib/format";

const MODEL_MODES = [
  { key: "chart_mean", label: "Chart mean" },
  { key: "rolling", label: "Rolling" },
  { key: "full", label: "Full (beta)" },
];

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`tnum rounded border px-2 py-0.5 text-[11px] transition-colors ${
        active
          ? "border-pos-dim bg-pos-soft text-pos"
          : "border-line bg-panel text-muted hover:border-line-strong hover:text-fg"
      }`}
    >
      {children}
    </button>
  );
}

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

  const { data, isLoading, isError, isFetching } = useEdges(params);

  const allBooks = data?.books_available ?? meta.data?.books ?? [];
  const allStats = data?.stats_available ?? meta.data?.stats ?? [];
  const rows = data?.rows ?? [];

  function toggle(list: string[], setter: (v: string[]) => void, item: string) {
    setter(
      list.includes(item) ? list.filter((x) => x !== item) : [...list, item],
    );
  }

  const columns: Column<EdgeRow>[] = [
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
      render: (r) => (
        <span className="tnum text-muted">{statLabel(r.stat_type)}</span>
      ),
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
      render: (r) => <span className="tnum">{fmtNum(r.book_line)}</span>,
      sortable: true,
      sortValue: (r) => r.book_line,
    },
    {
      key: "model_mu",
      header: "μ",
      align: "right",
      render: (r) => <span className="tnum text-muted">{fmtNum(r.model_mu)}</span>,
      sortable: true,
      sortValue: (r) => r.model_mu,
    },
    {
      key: "line_vs_mu",
      header: "Line−μ",
      align: "right",
      render: (r) => <span className="tnum text-faint">{fmtNum(r.line_vs_mu)}</span>,
      sortable: true,
      sortValue: (r) => r.line_vs_mu,
    },
    {
      key: "best_side",
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
      key: "p_over",
      header: "P(best)",
      align: "right",
      render: (r) => (
        <div className="w-24">
          <ProbabilityBar
            value={r.best_side === "under" ? r.p_under : r.p_over}
          />
        </div>
      ),
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
      header: "EV/u",
      align: "right",
      render: (r) => <Delta value={r.ev_best} format={fmtSigned} />,
      sortable: true,
      sortValue: (r) => r.ev_best,
    },
    {
      key: "distribution",
      header: "Dist",
      render: (r) => (
        <span className="text-[11px] text-faint">{r.distribution ?? "—"}</span>
      ),
    },
  ];

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-lg font-semibold text-fg">Edge Scanner</h1>
          <p className="mt-0.5 text-xs text-faint">
            The scored slate ranked by model-vs-line edge. Not arbitrage — model
            edge only.
          </p>
        </div>
        {isFetching && (
          <span className="eyebrow text-faint">scanning…</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Lines scanned" value={fmtInt(data?.n_lines ?? 0)} />
        <StatCard label="Scored" value={fmtInt(data?.n_scored ?? 0)} />
        <StatCard label="Shown" value={fmtInt(data?.n_returned ?? 0)} />
        <StatCard label="Model mode" value={titleCase(data?.model_mode ?? modelMode)} />
      </div>

      {/* Filters */}
      <div className="panel space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="eyebrow">Model</span>
            <div className="flex gap-1 rounded border border-line bg-panel p-1">
              {MODEL_MODES.map((m) => (
                <button
                  key={m.key}
                  onClick={() => setModelMode(m.key)}
                  className={`rounded px-2.5 py-1 text-[11px] transition-colors ${
                    modelMode === m.key
                      ? "bg-panel-3 text-fg"
                      : "text-muted hover:text-fg"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <label className="flex items-center gap-2 text-[11px] text-faint">
            min edge
            <input
              type="number"
              step="0.01"
              value={minEdge}
              onChange={(e) => setMinEdge(Number(e.target.value) || 0)}
              className="tnum w-16 rounded border border-line bg-panel-2 px-2 py-1 text-fg focus:border-line-strong focus:outline-none"
            />
          </label>
          <label className="flex items-center gap-2 text-[11px] text-faint">
            min P(over)
            <input
              type="number"
              step="0.01"
              value={minPOver}
              onChange={(e) => setMinPOver(Number(e.target.value) || 0)}
              className="tnum w-16 rounded border border-line bg-panel-2 px-2 py-1 text-fg focus:border-line-strong focus:outline-none"
            />
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-[11px] text-muted">
            <input
              type="checkbox"
              checked={onlyPositiveEv}
              onChange={(e) => setOnlyPositiveEv(e.target.checked)}
              className="accent-pos"
            />
            only +EV
          </label>
        </div>

        <div className="flex flex-wrap items-start gap-2">
          <span className="eyebrow mt-1 w-12">Books</span>
          <div className="flex flex-1 flex-wrap gap-1">
            {allBooks.map((b) => (
              <Chip
                key={b}
                active={books.includes(b)}
                onClick={() => toggle(books, setBooks, b)}
              >
                {b}
              </Chip>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-start gap-2">
          <span className="eyebrow mt-1 w-12">Stats</span>
          <div className="flex flex-1 flex-wrap gap-1">
            {allStats.map((s) => (
              <Chip
                key={s}
                active={stats.includes(s)}
                onClick={() => toggle(stats, setStats, s)}
              >
                {statLabel(s)}
              </Chip>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      <section className="panel">
        {isError ? (
          <div className="p-4">
            <ErrorState message="Edge scan failed." />
          </div>
        ) : isLoading ? (
          <Loading rows={10} />
        ) : rows.length ? (
          <DataTable
            columns={columns}
            rows={rows}
            rowKey={(r) => `${r.book}-${r.player_name}-${r.stat_type}`}
            initialSort={{ key: "model_edge", dir: "desc" }}
            maxHeight="calc(100vh - 420px)"
            onRowClick={(r) =>
              navigate(
                `/player?name=${encodeURIComponent(
                  r.player_name,
                )}&stat=${r.stat_type}`,
              )
            }
          />
        ) : (
          <EmptyState
            title="No props cleared the current filters."
            hint="The scanner scores scraped prop lines against the model. During the NBA offseason no books post player props, so the slate is empty until October — loosen filters once lines return."
          />
        )}
      </section>
    </div>
  );
}
