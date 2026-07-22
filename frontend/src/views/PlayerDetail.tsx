import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import type { EChartsOption } from "echarts";
import {
  useMeta,
  usePlayerDetail,
  usePlayerSearch,
} from "../api/hooks";
import type { BookLineRow, PlayerDetail as PlayerDetailData } from "../api/types";
import { StatCard } from "../components/StatCard";
import { EChart } from "../components/EChart";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { Loading, ErrorState } from "../components/Loading";
import { Delta } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { OddsBadge } from "../components/OddsBadge";
import { CHART, axis, gridTight, tooltipStyle } from "../components/chartBase";
import {
  fmtAgo,
  fmtNum,
  fmtPct,
  fmtSigned,
  fmtSignedPct,
  statLabel,
} from "../lib/format";

interface Selected {
  id: number;
  name: string;
}

// --- Player search picker -------------------------------------------------

function PlayerPicker({
  selected,
  onSelect,
}: {
  selected: Selected | null;
  onSelect: (s: Selected) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const { data, isFetching } = usePlayerSearch(query);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const rows = data?.rows ?? [];

  return (
    <div ref={boxRef} className="relative w-72">
      <input
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        placeholder={selected ? selected.name : "Search players…"}
        className="tnum w-full rounded border border-line bg-panel-2 px-3 py-1.5 text-sm text-fg placeholder:text-faint focus:border-line-strong focus:outline-none"
      />
      {open && (query.length > 0 || rows.length > 0) && (
        <div className="panel absolute z-30 mt-1 max-h-80 w-full overflow-auto p-1">
          {isFetching && !rows.length && (
            <div className="px-3 py-2 text-xs text-faint">Searching…</div>
          )}
          {!isFetching && !rows.length && (
            <div className="px-3 py-2 text-xs text-faint">No matches.</div>
          )}
          {rows.map((r) => (
            <button
              key={r.player_id}
              onClick={() => {
                onSelect({ id: r.player_id, name: r.player_name });
                setQuery("");
                setOpen(false);
              }}
              className="flex w-full items-center justify-between rounded px-3 py-1.5 text-left text-sm text-muted hover:bg-panel-3 hover:text-fg"
            >
              <span>
                {r.player_name}
                {r.team && (
                  <span className="tnum ml-2 text-[11px] text-faint">
                    {r.team}
                  </span>
                )}
              </span>
              {r.n_books > 0 && (
                <span className="tnum text-[11px] text-pos">
                  {r.n_books}b
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Charts ---------------------------------------------------------------

function PerformanceChart({ d }: { d: PlayerDetailData }) {
  const consensus = d.kpis.market_consensus_line;
  const dates = d.series.map((p) =>
    p.game_date ? p.game_date.slice(5, 10) : "",
  );
  const values = d.series.map((p) => p.value);
  const rolling = d.series.map((p) => p.rolling_mean);

  const option: EChartsOption = {
    grid: { ...gridTight, top: 20 },
    tooltip: {
      trigger: "axis",
      ...tooltipStyle,
      formatter: (params: unknown) => {
        const arr = params as Array<{ dataIndex: number }>;
        const i = arr[0]?.dataIndex ?? 0;
        const p = d.series[i];
        const opp = p.opponent ? ` vs ${p.opponent}` : "";
        return `${p.game_date?.slice(0, 10) ?? ""}${opp}<br/>${statLabel(
          d.stat_type,
        )} <b>${fmtNum(p.value, 0)}</b> · roll ${fmtNum(p.rolling_mean)}`;
      },
    },
    xAxis: { type: "category", data: dates, boundaryGap: true, ...axis() },
    yAxis: { type: "value", scale: true, ...axis() },
    series: [
      {
        name: "value",
        type: "bar",
        data: values.map((v) => ({
          value: v,
          itemStyle: {
            color:
              consensus != null
                ? v >= consensus
                  ? CHART.pos
                  : CHART.neg
                : CHART.info,
            opacity: 0.55,
          },
        })),
        barWidth: "62%",
      },
      {
        name: "rolling",
        type: "line",
        data: rolling,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: CHART.warn, width: 1.75 },
        z: 3,
        ...(consensus != null
          ? {
              markLine: {
                symbol: "none",
                label: {
                  position: "insideStartTop",
                  color: CHART.muted,
                  fontFamily: CHART.fontMono,
                  fontSize: 10,
                  formatter: `book mean ${fmtNum(consensus)}`,
                },
                lineStyle: { color: CHART.muted, type: "dashed", width: 1 },
                data: [{ yAxis: consensus }],
              },
            }
          : {}),
      },
    ],
  };
  return <EChart option={option} height={280} />;
}

function DistributionChart({ d }: { d: PlayerDetailData }) {
  const bars = d.histogram.map((b) => [(b.x0 + b.x1) / 2, b.count]);
  const fitted = d.fitted.map((p) => [p.x, p.y]);
  const consensus = d.kpis.market_consensus_line;

  const bookMarks = d.book_lines
    .filter((b) => b.line != null)
    .map((b) => ({
      xAxis: b.line as number,
      lineStyle: { color: CHART.info, type: "dotted" as const, width: 1 },
      label: {
        show: false as const,
      },
    }));

  const option: EChartsOption = {
    grid: { ...gridTight, top: 20 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      ...tooltipStyle,
    },
    xAxis: { type: "value", scale: true, ...axis({ name: statLabel(d.stat_type) }) },
    yAxis: { type: "value", ...axis({ name: "games" }) },
    series: [
      {
        name: "count",
        type: "bar",
        data: bars,
        itemStyle: { color: CHART.lineStrong },
        barWidth: "96%",
        markLine: bookMarks.length
          ? {
              symbol: "none",
              data: [
                ...bookMarks,
                ...(consensus != null
                  ? [
                      {
                        xAxis: consensus,
                        lineStyle: {
                          color: CHART.warn,
                          type: "dashed" as const,
                          width: 1.25,
                        },
                        label: {
                          show: true,
                          color: CHART.warn,
                          fontFamily: CHART.fontMono,
                          fontSize: 10,
                          formatter: "mean",
                        },
                      },
                    ]
                  : []),
              ],
            }
          : undefined,
      },
      {
        name: "fitted normal",
        type: "line",
        data: fitted,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: CHART.pos, width: 2 },
        areaStyle: { color: CHART.pos, opacity: 0.06 },
        z: 3,
      },
    ],
  };
  return <EChart option={option} height={260} />;
}

function HitRateChart({ books }: { books: BookLineRow[] }) {
  const withLines = books.filter((b) => b.hit_rate != null);
  if (!withLines.length) {
    return <EmptyState compact title="No book lines to compare." />;
  }
  const names = withLines.map((b) => b.book);
  const rates = withLines.map((b) => b.hit_rate as number);
  const option: EChartsOption = {
    grid: { left: 78, right: 40, top: 8, bottom: 24 },
    tooltip: {
      ...tooltipStyle,
      formatter: (p: unknown) => {
        const item = p as { name: string; value: number };
        return `${item.name}<br/>over rate <b>${fmtPct(item.value)}</b>`;
      },
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      ...axis(),
      axisLabel: {
        color: CHART.muted,
        fontFamily: CHART.fontMono,
        fontSize: 10,
        formatter: (v: number) => `${Math.round(v * 100)}%`,
      },
    },
    yAxis: {
      type: "category",
      data: names,
      ...axis(),
      axisLabel: { color: CHART.muted, fontSize: 10 },
    },
    series: [
      {
        type: "bar",
        data: rates.map((v) => ({
          value: v,
          itemStyle: { color: v >= 0.5 ? CHART.pos : CHART.neg, opacity: 0.7 },
        })),
        barWidth: "58%",
        markLine: {
          symbol: "none",
          data: [{ xAxis: 0.5 }],
          lineStyle: { color: CHART.muted, type: "dashed", width: 1 },
          label: { show: false },
        },
      },
    ],
  };
  return <EChart option={option} height={Math.max(120, names.length * 34)} />;
}

// --- Per-book line table --------------------------------------------------

function BookLinesTable({ books }: { books: BookLineRow[] }) {
  if (!books.length) {
    return (
      <EmptyState
        compact
        title="No book lines for this stat."
        hint="Scraped prop lines appear here once the books post this player+stat."
      />
    );
  }
  const columns: Column<BookLineRow>[] = [
    {
      key: "book",
      header: "Book",
      render: (r) => <span className="text-fg">{r.book}</span>,
      sortable: true,
      sortValue: (r) => r.book,
    },
    {
      key: "line",
      header: "Line",
      align: "right",
      render: (r) => <span className="tnum">{fmtNum(r.line)}</span>,
      sortable: true,
      sortValue: (r) => r.line,
    },
    {
      key: "odds",
      header: "O / U",
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
      render: (r) => (
        <div className="w-28">
          <ProbabilityBar value={r.p_over} marker={r.breakeven} />
        </div>
      ),
      sortable: true,
      sortValue: (r) => r.p_over,
    },
    {
      key: "hit",
      header: "Hit%",
      align: "right",
      render: (r) => <span className="tnum text-muted">{fmtPct(r.hit_rate, 0)}</span>,
      sortable: true,
      sortValue: (r) => r.hit_rate,
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
      header: "EV·o",
      align: "right",
      render: (r) => <Delta value={r.ev_over} format={fmtSigned} />,
      sortable: true,
      sortValue: (r) => r.ev_over,
    },
  ];
  return (
    <DataTable
      columns={columns}
      rows={books}
      rowKey={(r) => r.book}
      initialSort={{ key: "edge", dir: "desc" }}
    />
  );
}

// --- Controls -------------------------------------------------------------

function StatSegmented({
  stats,
  value,
  onChange,
}: {
  stats: string[];
  value: string;
  onChange: (s: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1 rounded border border-line bg-panel p-1">
      {stats.map((s) => (
        <button
          key={s}
          onClick={() => onChange(s)}
          className={`tnum rounded px-2.5 py-1 text-[11px] transition-colors ${
            s === value
              ? "bg-panel-3 text-fg"
              : "text-muted hover:text-fg"
          }`}
        >
          {statLabel(s)}
        </button>
      ))}
    </div>
  );
}

function NumberControl({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min: number;
  max: number;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px] text-faint">
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (!Number.isNaN(n)) onChange(Math.min(max, Math.max(min, n)));
        }}
        className="tnum w-16 rounded border border-line bg-panel-2 px-2 py-1 text-fg focus:border-line-strong focus:outline-none"
      />
    </label>
  );
}

// --- View -----------------------------------------------------------------

export function PlayerDetail() {
  const routeParams = useParams();
  const [search] = useSearchParams();
  const meta = useMeta();

  const [selected, setSelected] = useState<Selected | null>(() => {
    const id = routeParams.playerId ? Number(routeParams.playerId) : NaN;
    if (!Number.isNaN(id)) return { id, name: search.get("name") ?? "" };
    return null;
  });
  const [stat, setStat] = useState(search.get("stat") ?? "points");
  const [nGames, setNGames] = useState(25);
  const [rollingWindow, setRollingWindow] = useState(5);

  // Deep-link by name (from the dashboard's edge table): resolve to an id.
  const nameParam = search.get("name");
  const nameSearch = usePlayerSearch(nameParam ?? "");
  useEffect(() => {
    if (selected || !nameParam) return;
    const rows = nameSearch.data?.rows ?? [];
    const match =
      rows.find(
        (r) => r.player_name.toLowerCase() === nameParam.toLowerCase(),
      ) ?? rows[0];
    if (match) setSelected({ id: match.player_id, name: match.player_name });
  }, [nameParam, nameSearch.data, selected]);

  const detail = usePlayerDetail(
    selected
      ? {
          playerId: selected.id,
          name: selected.name || undefined,
          stat,
          n_games: nGames,
          rolling_window: rollingWindow,
        }
      : null,
  );

  const stats = useMemo(() => meta.data?.stats ?? ["points"], [meta.data]);
  const d = detail.data;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-fg">
            {d ? d.player_name : selected?.name || "Player Detail"}
          </h1>
          <p className="mt-0.5 text-xs text-faint">
            Recent form, distribution, and per-book pricing.
          </p>
        </div>
        <PlayerPicker selected={selected} onSelect={setSelected} />
      </div>

      {!selected && (
        <div className="panel">
          <EmptyState
            title="Search for a player to begin."
            hint="Type a name above. The flagship view charts recent-N performance with a rolling mean and book-line overlay, a fitted distribution, and every book's P(over) / EV."
          />
        </div>
      )}

      {selected && detail.isError && (
        <ErrorState message="Failed to load player detail." />
      )}

      {selected && detail.isLoading && !d && <Loading rows={8} />}

      {selected && d && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <StatSegmented stats={stats} value={stat} onChange={setStat} />
            <div className="flex items-center gap-4">
              <NumberControl
                label="games"
                value={nGames}
                onChange={setNGames}
                min={1}
                max={200}
              />
              <NumberControl
                label="roll"
                value={rollingWindow}
                onChange={setRollingWindow}
                min={1}
                max={60}
              />
            </div>
          </div>

          {d.n_games === 0 ? (
            <div className="panel">
              <EmptyState
                title="No game logs for this player+stat."
                hint="Pick another stat or player; this combination has no history in the DB yet."
              />
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
                <StatCard
                  label={`${statLabel(stat)} mean`}
                  value={fmtNum(d.kpis.mu)}
                  sub={`last ${d.n_games} games`}
                />
                <StatCard label="Std dev" value={fmtNum(d.kpis.sigma)} sub="σ" />
                <StatCard
                  label="Book mean"
                  value={fmtNum(d.kpis.market_consensus_line)}
                  sub="consensus line"
                />
                <StatCard
                  label="Books"
                  value={d.kpis.n_books}
                  sub="posting a line"
                />
                <StatCard
                  label="+EV sides"
                  value={d.kpis.positive_ev_sides}
                  tone={d.kpis.positive_ev_sides > 0 ? "pos" : "default"}
                  accent={d.kpis.positive_ev_sides > 0}
                  sub="model vs price"
                />
                <StatCard
                  label="Lines age"
                  value={
                    d.last_line_scraped_utc
                      ? fmtAgo(d.last_line_scraped_utc)
                      : "—"
                  }
                  sub="freshest book"
                />
              </div>

              <section className="panel">
                <div className="border-b border-line px-4 py-2.5">
                  <h2 className="eyebrow">
                    Recent {d.n_games} · {statLabel(stat)} with rolling mean
                  </h2>
                </div>
                <div className="px-2 py-2">
                  <PerformanceChart d={d} />
                </div>
              </section>

              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                <section className="panel">
                  <div className="border-b border-line px-4 py-2.5">
                    <h2 className="eyebrow">
                      Distribution · fitted normal + book lines
                    </h2>
                  </div>
                  <div className="px-2 py-2">
                    <DistributionChart d={d} />
                  </div>
                </section>

                <section className="panel">
                  <div className="border-b border-line px-4 py-2.5">
                    <h2 className="eyebrow">Per-book lines · P(over) / EV</h2>
                  </div>
                  <BookLinesTable books={d.book_lines} />
                </section>
              </div>

              <section className="panel">
                <div className="border-b border-line px-4 py-2.5">
                  <h2 className="eyebrow">
                    Historical over-rate vs each book line
                  </h2>
                </div>
                <div className="px-2 py-3">
                  <HitRateChart books={d.book_lines} />
                </div>
              </section>

              {d.notes.length > 0 && (
                <div className="text-[11px] text-faint">
                  {d.notes.map((n, i) => (
                    <div key={i}>· {n}</div>
                  ))}
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
