import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { useMeta, useTeamChart } from "../api/hooks";
import type { TeamChartResponse } from "../api/types";
import { StatCard } from "../components/StatCard";
import { EChart } from "../components/EChart";
import { EmptyState } from "../components/EmptyState";
import { Loading, ErrorState } from "../components/Loading";
import { Segmented, NumberField } from "../components/controls";
import { CHART, axis, gridTight, tooltipStyle } from "../components/chartBase";
import { fmtNum, statLabel } from "../lib/format";

function TeamPerformanceChart({ d }: { d: TeamChartResponse }) {
  const dates = d.series.map((p) => (p.game_date ? p.game_date.slice(5, 10) : ""));
  const values = d.series.map((p) => p.value);
  const consensus = d.market_consensus_line;
  const derived = d.derived_reference_line;

  const refMarks: Array<Record<string, unknown>> = [];
  if (consensus != null) {
    refMarks.push({
      yAxis: consensus,
      lineStyle: { color: CHART.warn, type: "dashed", width: 1.25 },
      label: {
        position: "insideStartTop",
        color: CHART.warn,
        fontFamily: CHART.fontMono,
        fontSize: 10,
        formatter: `implied total ${fmtNum(consensus)}`,
      },
    });
  }
  if (derived != null) {
    refMarks.push({
      yAxis: derived,
      lineStyle: { color: CHART.info, type: "dotted", width: 1.25 },
      label: {
        position: "insideEndTop",
        color: CHART.info,
        fontFamily: CHART.fontMono,
        fontSize: 10,
        formatter: "props-derived",
      },
    });
  }

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
        return `${p.game_date?.slice(0, 10) ?? ""}${opp}<br/>team ${statLabel(
          d.stat_type,
        )} <b>${fmtNum(p.value, 0)}</b>`;
      },
    },
    xAxis: { type: "category", data: dates, boundaryGap: true, ...axis() },
    yAxis: { type: "value", scale: true, ...axis() },
    series: [
      {
        type: "bar",
        data: values,
        barWidth: "62%",
        itemStyle: { color: CHART.info, opacity: 0.5 },
        markLine: refMarks.length
          ? { symbol: "none", data: refMarks as never }
          : undefined,
      },
    ],
  };
  return <EChart option={option} height={300} />;
}

export function TeamCharts() {
  const meta = useMeta();
  const teams = useMemo(() => meta.data?.teams ?? [], [meta.data]);
  const stats = useMemo(() => meta.data?.stats ?? ["points"], [meta.data]);

  const [team, setTeam] = useState<string>("");
  const [stat, setStat] = useState("points");
  const [nGames, setNGames] = useState(25);

  // Default to the first available team once meta loads.
  useEffect(() => {
    if (!team && teams.length) setTeam(teams[0]);
  }, [teams, team]);

  const { data, isLoading, isError } = useTeamChart(team || null, stat, nGames);

  const statOptions = stats.map((s) => ({ key: s, label: statLabel(s) }));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-fg">Team Charts</h1>
          <p className="mt-0.5 text-xs text-faint">
            Per-game team aggregates. Points carries the implied team total;
            other stats show a props-derived reference (Σ players' consensus
            lines) — a derived signal, never a posted book line.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            className="tnum rounded border border-line bg-panel-2 px-3 py-1.5 text-sm text-fg focus:border-line-strong focus:outline-none"
          >
            {teams.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <NumberField label="games" value={nGames} onChange={setNGames} min={1} max={200} />
        </div>
      </div>

      <Segmented options={statOptions} value={stat} onChange={setStat} />

      {isError && <ErrorState message="Failed to load team chart." />}
      {isLoading && !data && <Loading rows={8} />}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard
              label={`Team ${statLabel(stat)} mean`}
              value={fmtNum(data.kpis.mu)}
              sub={`last ${data.n_games} games`}
            />
            <StatCard label="Std dev" value={fmtNum(data.kpis.sigma)} sub="σ" />
            <StatCard
              label={stat === "points" ? "Implied total" : "Props-derived"}
              value={fmtNum(
                stat === "points"
                  ? data.market_consensus_line
                  : data.derived_reference_line,
              )}
              sub={stat === "points" ? "book mean" : data.derived_reference_label ?? "Σ players"}
            />
            <StatCard label="Games" value={data.n_games} sub="in window" />
          </div>

          {data.n_games === 0 ? (
            <div className="panel">
              <EmptyState
                title={`No game logs for ${data.team}.`}
                hint="Pick another team or stat; this team has no rows in the DB for the selected window."
              />
            </div>
          ) : (
            <section className="panel">
              <div className="border-b border-line px-4 py-2.5">
                <h2 className="eyebrow">
                  {data.team} · team {statLabel(stat)} per game
                </h2>
              </div>
              <div className="px-2 py-2">
                <TeamPerformanceChart d={data} />
              </div>
            </section>
          )}

          {data.notes.length > 0 && (
            <div className="text-[11px] leading-relaxed text-faint">
              {data.notes.map((n, i) => (
                <div key={i}>· {n}</div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
