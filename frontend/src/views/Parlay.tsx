import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMeta, useParlayPrice, usePlayerDetail } from "../api/hooks";
import type { LegSide, ParlayLegInput, ParlayLegPriced, PlayerSearchRow } from "../api/types";
import { Card, KpiGrid, Page } from "../components/Page";
import { StatCard, signedTone } from "../components/StatCard";
import { EChart } from "../components/EChart";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { ChartSkeleton, KpiRowSkeleton } from "../components/Skeleton";
import { DataTable, type Column } from "../components/DataTable";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { Delta, SideTag } from "../components/Delta";
import { OddsBadge } from "../components/OddsBadge";
import { ModelDisclaimer } from "../components/ModelDisclaimer";
import { PlayerPicker } from "../components/PlayerPicker";
import { Button, NumberField, Segmented, Select } from "../components/controls";
import { correlationHeatmapOption, probabilityComparisonOption } from "../charts/parlayCharts";
import { DASH, fmtEv, fmtInt, fmtLine, fmtNum, fmtOdds, fmtPct, statLabel } from "../lib/format";
import { MAX_LEGS, decodeLegs, encodeLegs, legKey, propLine } from "../lib/parlayLegs";

const SIDES = [
  { key: "over", label: "Over" },
  { key: "under", label: "Under" },
];

// --- Add-leg form -------------------------------------------------------------

function AddLegForm({
  stats,
  disabled,
  existing,
  onAdd,
}: {
  stats: string[];
  disabled: boolean;
  existing: Set<string>;
  onAdd: (leg: ParlayLegInput) => void;
}) {
  const [player, setPlayer] = useState<PlayerSearchRow | null>(null);
  const [stat, setStat] = useState("points");
  const [line, setLine] = useState(20.5);
  const [side, setSide] = useState<LegSide>("over");
  const [odds, setOdds] = useState(-110);

  // Prefill the line with the book consensus (else the player's mean), so a
  // leg starts at a realistic number instead of a placeholder.
  const detail = usePlayerDetail(
    player ? { playerId: player.player_id, stat, n_games: 25, rolling_window: 5 } : null,
  );
  const suggested = detail.data?.kpis.market_consensus_line ?? detail.data?.kpis.mu ?? null;
  useEffect(() => {
    if (suggested != null && detail.data?.player_id === player?.player_id) {
      setLine(propLine(suggested));
    }
  }, [suggested, detail.data?.player_id, player?.player_id]);

  const candidate = player ? { player_id: player.player_id, stat, line, side } : null;
  const duplicate = candidate ? existing.has(legKey(candidate)) : false;
  const noHistory = detail.data != null && detail.data.n_games < 3;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <PlayerPicker label="Add a player" placeholder="Search a player to add…" onSelect={setPlayer} />
        {player && (
          <span className="inline-flex h-8 items-center gap-2 rounded-full border border-line bg-surface-2 px-3 text-label text-fg">
            {player.player_name}
            {player.team && <span className="tnum text-faint">{player.team}</span>}
          </span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-4">
        <Select label="Stat" value={stat} onChange={setStat} options={stats} format={statLabel} />
        <NumberField label="Line" value={line} onChange={setLine} step={0.5} min={0} max={150} />
        <Segmented label="Side" options={SIDES} value={side} onChange={(k) => setSide(k as LegSide)} />
        <NumberField label="Leg odds" value={odds} onChange={setOdds} step={5} min={-100000} max={100000} width="w-24" />
        <Button
          disabled={disabled || !player || duplicate || noHistory || odds === 0}
          onClick={() =>
            player &&
            onAdd({ player_id: player.player_id, player_name: player.player_name, stat, line, side, odds })
          }
          title={duplicate ? "This leg is already in the parlay" : undefined}
        >
          <span aria-hidden>+</span> Add leg
        </Button>
      </div>
      {player && detail.data && (
        <p className="text-label text-faint">
          {noHistory
            ? `${player.player_name} has too few ${statLabel(stat)} games to model.`
            : `${statLabel(stat)} last ${detail.data.n_games}: mean ${fmtNum(detail.data.kpis.mu)}, book mean ${fmtLine(detail.data.kpis.market_consensus_line)}.`}
          {duplicate && " This leg is already in the parlay."}
        </p>
      )}
      {disabled && <p className="text-label text-faint">Parlays are capped at {MAX_LEGS} legs.</p>}
    </div>
  );
}

// --- Leg table ------------------------------------------------------------------

interface LegRow {
  index: number;
  input: ParlayLegInput;
  priced: ParlayLegPriced | null;
}

function legColumns(onRemove: (i: number) => void): Column<LegRow>[] {
  return [
    {
      key: "n",
      header: "#",
      render: (r) => <span className="tnum text-faint">{r.index + 1}</span>,
    },
    {
      key: "player",
      header: "Player",
      render: (r) => <span className="font-medium text-fg">{r.priced?.player_name || r.input.player_name || `#${r.input.player_id}`}</span>,
    },
    { key: "stat", header: "Stat", render: (r) => <span className="tnum text-muted">{statLabel(r.input.stat)}</span> },
    { key: "side", header: "Side", render: (r) => <SideTag side={r.input.side} /> },
    { key: "line", header: "Line", align: "right", render: (r) => <span className="tnum text-fg">{fmtLine(r.input.line)}</span> },
    { key: "odds", header: "Odds", align: "right", render: (r) => <OddsBadge odds={r.input.odds} /> },
    {
      key: "mu",
      header: "Mean ± sd",
      align: "right",
      render: (r) =>
        r.priced ? (
          <span className="tnum text-muted">
            {fmtNum(r.priced.mu)} ± {fmtNum(r.priced.sigma)}
          </span>
        ) : (
          <span className="text-faint">{DASH}</span>
        ),
    },
    {
      key: "p",
      header: "P(hit) vs implied",
      align: "right",
      width: "180px",
      render: (r) =>
        r.priced ? (
          <ProbabilityBar value={r.priced.p_hit} marker={r.priced.implied_prob} />
        ) : (
          <span className="text-faint">{DASH}</span>
        ),
    },
    {
      key: "ev",
      header: "Leg EV",
      align: "right",
      render: (r) => <Delta value={r.priced?.ev} format={fmtEv} />,
    },
    {
      key: "remove",
      header: "",
      align: "right",
      render: (r) => (
        <button
          type="button"
          aria-label={`Remove leg ${r.index + 1}`}
          onClick={(e) => {
            e.stopPropagation();
            onRemove(r.index);
          }}
          className="inline-flex h-7 w-7 items-center justify-center rounded-md text-faint transition-colors hover:bg-surface-3 hover:text-fg pointer-coarse:h-11 pointer-coarse:w-11"
        >
          ×
        </button>
      ),
    },
  ];
}

// --- View ---------------------------------------------------------------------------

export function Parlay() {
  const meta = useMeta();
  const [search, setSearch] = useSearchParams();
  // Legs live in the URL so a parlay is shareable / survives refresh.
  const legs = useMemo(() => decodeLegs(search.get("legs")), [search]);
  const [nGames, setNGames] = useState(25);
  const [bookPrice, setBookPrice] = useState<number>(0); // 0 = use the leg product

  function setLegs(next: ParlayLegInput[]) {
    setSearch(
      (prev) => {
        const p = new URLSearchParams(prev);
        if (next.length) p.set("legs", encodeLegs(next));
        else p.delete("legs");
        return p;
      },
      { replace: true },
    );
  }

  const stats = useMemo(() => meta.data?.stats ?? ["points"], [meta.data]);
  const existing = useMemo(() => new Set(legs.map(legKey)), [legs]);
  const price = useParlayPrice(legs, nGames, bookPrice === 0 ? null : bookPrice);
  const d = price.data && price.data.legs.length === legs.length ? price.data : undefined;

  const rows: LegRow[] = legs.map((input, index) => ({ index, input, priced: d?.legs[index] ?? null }));
  const heatmap = useMemo(() => (d ? correlationHeatmapOption(d) : null), [d]);
  const compare = useMemo(() => (d ? probabilityComparisonOption(d) : null), [d]);

  return (
    <Page
      title="Parlay builder"
      description="Stack 2–6 legs across players. Per-leg probabilities use each player's fitted distribution; the joint probability is correlation-aware (Monte Carlo over the legs' shared game history) and shown next to the naive independent product."
    >
      <ModelDisclaimer>
        Correlations come from recent shared games and fall back to independence when that history is thin.
      </ModelDisclaimer>

      <Card title="Add a leg" meta={`${legs.length} / ${MAX_LEGS} legs`}>
        <AddLegForm
          stats={stats}
          disabled={legs.length >= MAX_LEGS}
          existing={existing}
          onAdd={(leg) => setLegs([...legs, leg])}
        />
      </Card>

      <Card
        title="Legs"
        meta={legs.length ? undefined : "none yet"}
        actions={
          legs.length ? (
            <Button onClick={() => setLegs([])} title="Remove every leg">
              Clear
            </Button>
          ) : undefined
        }
        flush
      >
        {legs.length ? (
          <DataTable columns={legColumns((i) => setLegs(legs.filter((_, j) => j !== i)))} rows={rows} rowKey={(r) => legKey(r.input)} />
        ) : (
          <EmptyState
            compact
            title="Add at least two legs to price a parlay."
            hint="Search a player, pick a stat, line and side. Same-game legs from one player (e.g. points + rebounds) are where correlation matters most."
          />
        )}
      </Card>

      {legs.length === 1 && (
        <p className="text-label text-faint">Add one more leg to price the parlay.</p>
      )}

      {legs.length >= 2 && (
        <>
          <div className="flex flex-wrap items-center gap-4">
            <NumberField label="History (games)" value={nGames} onChange={setNGames} min={3} max={200} />
            <NumberField
              label="Book SGP price (0 = leg product)"
              value={bookPrice}
              onChange={setBookPrice}
              step={5}
              min={-100000}
              max={100000}
              width="w-24"
            />
            {price.isFetching && (
              <span className="text-label text-faint" role="status">
                Simulating…
              </span>
            )}
          </div>

          {price.isError && <ErrorState message={errorMessage(price.error, "Couldn't price this parlay.")} />}

          {!d && price.isLoading && (
            <>
              <KpiGrid cols={6}>
                <KpiRowSkeleton count={6} />
              </KpiGrid>
              <div className="panel">
                <ChartSkeleton height={220} />
              </div>
            </>
          )}

          {d && (
            <div className={`space-y-6 transition-opacity ${price.isPlaceholderData ? "opacity-60" : ""}`}>
              <KpiGrid cols={6}>
                <StatCard
                  label="Joint P (correlated)"
                  value={fmtPct(d.joint_prob)}
                  sub={`± ${fmtPct(1.96 * d.joint_prob_se)} MC · ${fmtInt(d.n_sims)} sims`}
                />
                <StatCard label="Independent product" value={fmtPct(d.independent_prob)} sub="naive baseline" />
                <StatCard
                  label="Correlation effect"
                  value={d.correlation_lift != null ? `${fmtNum(d.correlation_lift, 2)}×` : DASH}
                  sub="joint ÷ independent"
                />
                <StatCard label="Fair odds" value={fmtOdds(d.fair_american)} sub="from joint P" />
                <StatCard
                  label={d.offered_is_custom ? "Book price" : "Leg-product price"}
                  value={fmtOdds(d.offered_american)}
                  sub={d.implied_prob != null ? `implies ${fmtPct(d.implied_prob)}` : undefined}
                />
                <StatCard
                  label="EV / unit (joint)"
                  value={fmtEv(d.ev_joint)}
                  tone={signedTone(d.ev_joint)}
                  sub={`independent: ${fmtEv(d.ev_independent)}`}
                />
              </KpiGrid>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <Card title="Joint vs independent" meta="P(all legs hit)">
                  {compare && <EChart option={compare} height={180} ariaLabel="Joint probability versus independent product" />}
                  <p className="mt-2 text-label text-faint">
                    {d.correlation_fallback
                      ? `Only ${d.n_joint_games} shared games — correlation fell back toward independence.`
                      : `Correlation estimated from ${d.n_joint_games} shared games (shrunk toward independence for stability).`}
                  </p>
                </Card>
                <Card title="Leg correlation" meta="ρ between legs' stat lines">
                  {heatmap && (
                    <EChart
                      option={heatmap}
                      height={Math.max(220, 56 * d.legs.length + 80)}
                      ariaLabel="Correlation heatmap between legs"
                    />
                  )}
                </Card>
              </div>
            </div>
          )}
        </>
      )}
    </Page>
  );
}
