import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { useLineMovement } from "../api/hooks";
import type { LineMovementResponse } from "../api/types";
import { EChart } from "./EChart";
import { EmptyState } from "./EmptyState";
import { ErrorState, errorMessage } from "./Loading";
import { ChartSkeleton } from "./Skeleton";
import { CHART, grid, tooltipHtml } from "./chartTheme";
import { fmtDateTime, fmtLine, fmtSigned } from "../lib/format";

const REPLAY_TICK_MS = 650;

export function bookColor(i: number): string {
  return CHART.series[i % CHART.series.length];
}

// Forward-fill each book's line onto the shared timestamp axis so books render
// as aligned lines; reveal only up to `idx`.
export function alignSeries(data: LineMovementResponse, idx: number) {
  return data.series.map((s) => {
    const byTs = new Map(s.points.map((p) => [p.ts, p.line]));
    let last: number | null = null;
    return data.timestamps.map((ts, j) => {
      const v = byTs.get(ts);
      if (v !== undefined) last = v;
      return j <= idx ? last : null;
    });
  });
}

// Clamp a scrubber index into the current series (the data can shrink under
// a stale index when the player / stat changes).
export function clampIndex(idx: number, length: number): number {
  if (length <= 0) return 0;
  return Math.min(Math.max(0, idx), length - 1);
}

export function LineMovementPanel({ playerId, stat }: { playerId: number; stat: string }) {
  // Look back the full validated window (30d cap) so historical drift shows.
  const { data, isLoading, isError, error } = useLineMovement(playerId, stat, 720);
  const timestamps = useMemo(() => data?.timestamps ?? [], [data]);
  const last = Math.max(0, timestamps.length - 1);

  const [rawIdx, setIdx] = useState(Number.MAX_SAFE_INTEGER);
  const [playing, setPlaying] = useState(false);
  // Clamp during render — an effect-based reset runs one render too late and
  // `timestamps[idx]` would be undefined for that render.
  const idx = clampIndex(rawIdx, timestamps.length);

  // New data -> show it fully drawn, stop any replay.
  useEffect(() => {
    setIdx(Number.MAX_SAFE_INTEGER);
    setPlaying(false);
  }, [data]);

  useEffect(() => {
    if (!playing) return;
    if (idx >= last) {
      setPlaying(false);
      return;
    }
    const t = window.setTimeout(() => setIdx(idx + 1), REPLAY_TICK_MS);
    return () => window.clearTimeout(t);
  }, [playing, idx, last]);

  const option = useMemo<EChartsOption | null>(() => {
    if (!data || !timestamps.length) return null;
    const aligned = alignSeries(data, idx);
    const few = timestamps.length <= 12;
    return {
      grid: grid({ legend: true }),
      animation: !playing,
      tooltip: {
        trigger: "axis",
        formatter: (params: unknown) => {
          const arr = params as Array<{ dataIndex: number; seriesName: string; value: number | null; color: string }>;
          const j = arr[0]?.dataIndex;
          if (j === undefined) return "";
          return tooltipHtml(
            fmtDateTime(timestamps[j]),
            arr
              .filter((p) => p.value != null)
              .map((p) => ({ label: p.seriesName, value: fmtLine(p.value), color: p.color })),
          );
        },
      },
      legend: { type: "scroll", top: 0, left: 0 },
      xAxis: {
        type: "category",
        data: timestamps.map(fmtDateTime),
        boundaryGap: few,
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { formatter: (v: number) => fmtLine(v) },
      },
      series: data.series.map((s, i) => ({
        name: s.book,
        type: "line" as const,
        data: aligned[i],
        step: "end" as const,
        showSymbol: few,
        symbolSize: 6,
        connectNulls: true,
        lineStyle: { width: 2, color: bookColor(i) },
        itemStyle: { color: bookColor(i) },
        emphasis: { focus: "series" as const },
      })),
    };
  }, [data, timestamps, idx, playing]);

  if (isLoading) return <ChartSkeleton height={240} />;
  if (isError) {
    return (
      <div className="p-4">
        <ErrorState message={errorMessage(error, "Failed to load line movement.")} />
      </div>
    );
  }
  if (!data || !option) {
    return (
      <EmptyState
        compact
        title="No stored line movement for this player + stat."
        hint="Line movement replays betting_line_snapshots per book. Offseason slates have no fresh snapshots within the 30-day window; drift appears once books start posting in October."
        lastData={data?.last_snapshot_utc ? `last snapshot ${fmtDateTime(data.last_snapshot_utc)}` : null}
      />
    );
  }

  const single = timestamps.length === 1;

  return (
    <div className="space-y-4 p-4">
      {single && (
        <p className="text-label text-faint">
          One snapshot so far — each book's current main line is shown; drift
          appears as more snapshots land.
        </p>
      )}
      <EChart
        option={option}
        height={260}
        notMerge={false}
        ariaLabel={`Line movement for ${data.series.length} books across ${timestamps.length} snapshots`}
      />
      <div className="flex items-center gap-4">
        <button
          type="button"
          aria-pressed={playing}
          aria-label={playing ? "Pause replay" : "Replay line movement"}
          disabled={single}
          onClick={() => {
            if (!playing && idx >= last) setIdx(0);
            setPlaying((p) => !p);
          }}
          className="inline-flex h-8 w-24 shrink-0 items-center justify-center gap-2 rounded-md border border-line-strong bg-surface-2 text-label font-medium text-fg transition-colors enabled:hover:bg-surface-3 disabled:opacity-40 pointer-coarse:h-11"
        >
          <span aria-hidden>{playing ? "❚❚" : "▶"}</span>
          {playing ? "Pause" : "Replay"}
        </button>
        <input
          type="range"
          min={0}
          max={last}
          value={idx}
          disabled={single}
          aria-label="Snapshot"
          aria-valuetext={fmtDateTime(timestamps[idx])}
          onChange={(e) => {
            setPlaying(false);
            setIdx(Number(e.target.value));
          }}
          className="h-1 min-w-0 flex-1 cursor-pointer accent-[var(--color-accent)] disabled:cursor-default pointer-coarse:h-11"
        />
        <span className="tnum hidden w-36 shrink-0 text-right text-label text-muted sm:inline">
          {fmtDateTime(timestamps[idx])}
        </span>
      </div>
      <ul className="flex flex-wrap gap-2">
        {data.series.map((s, i) => (
          <li
            key={s.book}
            className="inline-flex h-7 items-center gap-2 rounded-full border border-line bg-surface-2 px-3 text-caption text-muted"
          >
            <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: bookColor(i) }} aria-hidden />
            {s.book}
            <span className="tnum text-fg">
              {fmtLine(s.open_line)} → {fmtLine(s.close_line)}
            </span>
            <span className="tnum text-faint">{fmtSigned(s.line_delta, 1)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
