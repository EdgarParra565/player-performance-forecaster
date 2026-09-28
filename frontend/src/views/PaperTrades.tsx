import { useMemo, useState } from "react";
import { useCalibration, usePaperTrades } from "../api/hooks";
import type { BrierRow, PaperTradeRow } from "../api/types";
import { Card, KpiGrid, Page } from "../components/Page";
import { StatCard, signedTone } from "../components/StatCard";
import { EChart } from "../components/EChart";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { ChartSkeleton, KpiRowSkeleton, TableSkeleton } from "../components/Skeleton";
import { DataTable, type Column } from "../components/DataTable";
import { Delta, SideTag } from "../components/Delta";
import { ModelDisclaimer } from "../components/ModelDisclaimer";
import { Segmented, Select } from "../components/controls";
import { reliabilityOption } from "../charts/calibrationCharts";
import {
  DASH,
  fmtDateShort,
  fmtInt,
  fmtLine,
  fmtNum,
  fmtPct,
  fmtSigned,
  fmtSignedPct,
  statLabel,
} from "../lib/format";

const BET_SLIP_CMD =
  "python -m nba_model.evaluation.bet_slip --db data/database/nba_data.db --min-edge 0.03 --min-p 0.55 --max-picks 10";

// Result is an outcome, so won/lost may carry polarity; push/void stay neutral.
function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "won"
      ? "border-pos-dim/60 bg-pos-soft text-pos"
      : status === "lost"
        ? "border-neg-dim/60 bg-neg-soft text-neg"
        : "border-line bg-surface-2 text-muted";
  return (
    <span className={`tnum inline-flex h-6 items-center rounded-md border px-2 text-caption font-medium uppercase ${tone}`}>
      {status}
    </span>
  );
}

const BASE: Column<PaperTradeRow>[] = [
  {
    key: "date",
    header: "Game",
    render: (r) => <span className="tnum text-muted">{fmtDateShort(r.game_date)}</span>,
    sortable: true,
    sortValue: (r) => r.game_date,
  },
  {
    key: "player",
    header: "Player",
    render: (r) => <span className="font-medium text-fg">{r.player_name}</span>,
    sortable: true,
    sortValue: (r) => r.player_name,
  },
  { key: "stat", header: "Stat", render: (r) => <span className="tnum text-muted">{statLabel(r.stat_type)}</span> },
  { key: "book", header: "Book", render: (r) => <span className="text-muted">{r.book ?? DASH}</span> },
  { key: "side", header: "Side", render: (r) => <SideTag side={r.side} /> },
  { key: "line", header: "Line", align: "right", render: (r) => <span className="tnum text-fg">{fmtLine(r.line)}</span> },
  {
    key: "p",
    header: "Model P",
    align: "right",
    render: (r) => <span className="tnum text-fg">{fmtPct(r.model_prob)}</span>,
    sortable: true,
    sortValue: (r) => r.model_prob,
  },
  {
    key: "edge",
    header: "Edge",
    align: "right",
    render: (r) => <Delta value={r.edge} format={fmtSignedPct} />,
    sortable: true,
    sortValue: (r) => r.edge,
  },
  {
    key: "stake",
    header: "Stake",
    align: "right",
    render: (r) => <span className="tnum text-muted">{r.stake_units != null ? `${fmtNum(r.stake_units, 2)}u` : DASH}</span>,
  },
];

const SETTLED_COLUMNS: Column<PaperTradeRow>[] = [
  ...BASE,
  { key: "status", header: "Result", render: (r) => <StatusBadge status={r.status} /> },
  {
    key: "actual",
    header: "Actual",
    align: "right",
    render: (r) => <span className="tnum text-muted">{fmtNum(r.actual_value)}</span>,
  },
  {
    key: "clv",
    header: "CLV",
    align: "right",
    render: (r) => <Delta value={r.clv_delta} format={(v) => fmtSigned(v, 1)} strong />,
    sortable: true,
    sortValue: (r) => r.clv_delta,
  },
  {
    key: "pl",
    header: "Est. P/L",
    align: "right",
    render: (r) => <Delta value={r.est_profit_units} format={(v) => `${fmtSigned(v, 2)}u`} />,
    sortable: true,
    sortValue: (r) => r.est_profit_units,
  },
];

const PENDING_COLUMNS: Column<PaperTradeRow>[] = [
  ...BASE,
  {
    key: "logged",
    header: "Logged",
    align: "right",
    render: (r) => <span className="tnum text-faint">{fmtDateShort(r.created_at_utc?.slice(0, 10))}</span>,
  },
];

const SOURCES = [
  { key: "bet_log", label: "Paper trades" },
  { key: "predictions", label: "Model predictions" },
];

function Calibration() {
  const [source, setSource] = useState("bet_log");
  const [stat, setStat] = useState("all");
  const cal = useCalibration(source, stat === "all" ? null : stat);
  const d = cal.data;
  const option = useMemo(() => (d && d.reliability.length ? reliabilityOption(d.reliability) : null), [d]);
  const statOptions = ["all", ...(d?.stats_available ?? [])];

  const brierCols: Column<BrierRow>[] = [
    { key: "stat", header: "Stat", render: (r) => <span className="tnum text-muted">{statLabel(r.stat_type)}</span> },
    { key: "n", header: "N", align: "right", render: (r) => <span className="tnum text-muted">{fmtInt(r.n)}</span> },
    { key: "brier", header: "Brier", align: "right", render: (r) => <span className="tnum text-fg">{fmtNum(r.brier_score, 4)}</span> },
    { key: "real", header: "Realized", align: "right", render: (r) => <span className="tnum text-fg">{fmtPct(r.realized_rate)}</span> },
  ];

  return (
    <Card
      title="Calibration"
      meta="reliability: predicted vs realized · Brier (lower is better; 0.25 = coin flip)"
      actions={
        <div className="flex flex-wrap items-center gap-3">
          <Segmented label="Calibration source" options={SOURCES} value={source} onChange={(k) => { setSource(k); setStat("all"); }} />
          <Select
            label="Stat"
            hideLabel
            value={stat}
            onChange={setStat}
            options={statOptions}
            format={(v) => (v === "all" ? "All stats" : statLabel(v))}
          />
        </div>
      }
    >
      {cal.isError ? (
        <ErrorState message={errorMessage(cal.error, "Couldn't load calibration.")} />
      ) : cal.isLoading ? (
        <ChartSkeleton height={300} />
      ) : !d || d.n_settled === 0 ? (
        <EmptyState
          compact
          title={source === "bet_log" ? "No settled paper trades to calibrate yet." : "No settled predictions yet."}
          hint={
            source === "bet_log"
              ? "Calibration needs won/lost picks in bet_log. Switch to “Model predictions” to see the hourly model's historical calibration meanwhile."
              : "The hourly predictions table has no settled over/under outcomes in this database."
          }
        />
      ) : (
        <div className={`grid grid-cols-1 gap-6 xl:grid-cols-3 ${cal.isPlaceholderData ? "opacity-60" : ""}`}>
          <div className="xl:col-span-2">
            {option && (
              <EChart option={option} height={320} ariaLabel={`Reliability curve for ${d.n_settled} settled ${d.source} rows`} />
            )}
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <StatCard label="Settled" value={fmtInt(d.n_settled)} sub={SOURCES.find((x) => x.key === d.source)?.label} />
              <StatCard label="Brier" value={fmtNum(d.brier?.brier_score, 4)} sub={d.stat === "all" ? "all stats" : statLabel(d.stat)} />
            </div>
            {d.brier_by_stat.length > 1 && (
              <div className="overflow-hidden rounded-lg border border-line">
                <DataTable columns={brierCols} rows={d.brier_by_stat} rowKey={(r) => r.stat_type} />
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}

export function PaperTrades() {
  const trades = usePaperTrades("all");
  const s = trades.data?.summary;
  const rows = trades.data?.rows ?? [];
  const settled = rows.filter((r) => r.status !== "pending");
  const pending = rows.filter((r) => r.status === "pending");

  return (
    <Page
      title="Paper trades"
      description="Picks recorded by the bet_slip exporter (WS10 Phase 2 — measurement, not execution), how they settled, closing-line value, and how well the model's probabilities are calibrated."
    >
      <ModelDisclaimer>These are paper trades for measuring the model; no bets are placed.</ModelDisclaimer>

      {trades.isError && <ErrorState message={errorMessage(trades.error, "Couldn't load paper trades.")} />}

      <KpiGrid cols={6}>
        {trades.isLoading ? (
          <KpiRowSkeleton count={6} />
        ) : (
          <>
            <StatCard label="Pending" value={fmtInt(s?.pending)} sub="awaiting settlement" />
            <StatCard
              label="Settled"
              value={s ? `${s.won}–${s.lost}${s.push ? `–${s.push}` : ""}` : DASH}
              sub="W–L(–P)"
            />
            <StatCard label="Win rate" value={fmtPct(s?.win_rate)} sub="won / (won + lost)" />
            <StatCard
              label="Mean CLV"
              value={s?.mean_clv != null ? fmtSigned(s.mean_clv, 2) : DASH}
              tone={signedTone(s?.mean_clv)}
              sub={s?.n_clv ? `${fmtInt(s.n_clv)} picks with a close` : "no closing lines yet"}
            />
            <StatCard label="+CLV rate" value={fmtPct(s?.positive_clv_rate)} sub="beat the close" />
            <StatCard
              label="Est. units"
              value={s?.est_units != null ? `${fmtSigned(s.est_units, 2)}u` : DASH}
              tone={signedTone(s?.est_units)}
              sub="at logged implied price"
            />
          </>
        )}
      </KpiGrid>

      {trades.isLoading ? (
        <div className="panel">
          <TableSkeleton rows={5} cols={10} />
        </div>
      ) : s && s.total === 0 ? (
        <Card>
          <EmptyState
            title="No paper trades yet."
            hint="bet_log fills when the bet_slip exporter records the day's picks (it's empty until the season starts). Run it on a game day — add --dry-run to preview without writing:"
          />
          <pre className="tnum mx-auto -mt-8 mb-8 max-w-3xl overflow-x-auto rounded-lg border border-line bg-surface-2 px-4 py-3 text-caption text-muted">
            {BET_SLIP_CMD}
          </pre>
        </Card>
      ) : (
        <>
          <Card title="Settled" meta={`${fmtInt(settled.length)} picks · CLV = line moved in your favour by close`} flush>
            {settled.length ? (
              <DataTable columns={SETTLED_COLUMNS} rows={settled} rowKey={(r) => String(r.log_id)} />
            ) : (
              <EmptyState compact title="Nothing settled yet." hint="Picks settle after their game's results are ingested." />
            )}
          </Card>
          <Card title="Pending" meta={`${fmtInt(pending.length)} picks`} flush>
            {pending.length ? (
              <DataTable columns={PENDING_COLUMNS} rows={pending} rowKey={(r) => String(r.log_id)} />
            ) : (
              <EmptyState compact title="No pending picks." />
            )}
          </Card>
        </>
      )}

      <Calibration />
    </Page>
  );
}
