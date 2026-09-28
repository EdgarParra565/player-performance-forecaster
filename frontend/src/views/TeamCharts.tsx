import { useEffect, useMemo, useState } from "react";
import { useMeta, useTeamChart } from "../api/hooks";
import { StatCard } from "../components/StatCard";
import { EChart } from "../components/EChart";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { ChartSkeleton, KpiRowSkeleton } from "../components/Skeleton";
import { Card, KpiGrid, Page } from "../components/Page";
import { NumberField, Segmented, Select } from "../components/controls";
import { teamChartOption } from "../charts/teamChart";
import { fmtInt, fmtNum, statLabel } from "../lib/format";

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

  const { data, isLoading, isError, error } = useTeamChart(team || null, stat, nGames);
  const option = useMemo(() => (data ? teamChartOption(data) : null), [data]);
  const statOptions = stats.map((s) => ({ key: s, label: statLabel(s) }));
  const isPoints = stat === "points";

  return (
    <Page
      title="Team Charts"
      description="Per-game team aggregates. Points carries the implied team total; other stats show a props-derived reference (Σ players' consensus lines) — a derived signal, never a posted book line."
      actions={
        <>
          <Select label="Team" value={team} onChange={setTeam} options={teams} />
          <NumberField label="Games" value={nGames} onChange={setNGames} min={3} max={200} />
        </>
      }
    >
      {meta.isError && <ErrorState message={errorMessage(meta.error, "Failed to load teams.")} />}

      <Segmented label="Stat" options={statOptions} value={stat} onChange={setStat} />

      {isError && <ErrorState message={errorMessage(error, "Failed to load team chart.")} />}

      {(isLoading || (!data && !isError && !meta.isError)) && (
        <>
          <KpiGrid cols={4}>
            <KpiRowSkeleton count={4} />
          </KpiGrid>
          <div className="panel">
            <ChartSkeleton height={300} />
          </div>
        </>
      )}

      {data && (
        <>
          <KpiGrid cols={4}>
            <StatCard label={`Team ${statLabel(stat)} mean`} value={fmtNum(data.kpis.mu)} sub={`last ${data.n_games} games`} />
            <StatCard label="Std dev" value={fmtNum(data.kpis.sigma)} sub="σ" />
            <StatCard
              label={isPoints ? "Implied total" : "Props-derived"}
              value={fmtNum(isPoints ? data.market_consensus_line : data.derived_reference_line)}
              sub={isPoints ? "book mean" : (data.derived_reference_label ?? "Σ players")}
            />
            <StatCard label="Games" value={fmtInt(data.n_games)} sub="in window" />
          </KpiGrid>

          {data.n_games === 0 ? (
            <Card>
              <EmptyState
                title={`No game logs for ${data.team}.`}
                hint="Pick another team or stat; this team has no rows in the DB for the selected window."
              />
            </Card>
          ) : (
            <Card
              title={`${data.team} · team ${statLabel(stat)} per game`}
              meta={
                data.market_consensus_line == null && data.derived_reference_line == null
                  ? "no reference line in window"
                  : undefined
              }
            >
              {option && (
                <EChart
                  option={option}
                  height={320}
                  ariaLabel={`${data.team} team ${statLabel(stat)} over the last ${data.n_games} games`}
                />
              )}
            </Card>
          )}

          {data.notes.length > 0 && (
            <ul className="space-y-1 text-label text-faint">
              {data.notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </Page>
  );
}
