import { useEffect, useMemo, useRef, useState } from "react";
import type { EChartsOption } from "echarts";
import { useLineMovement } from "../api/hooks";
import { EChart } from "./EChart";
import { EmptyState } from "./EmptyState";
import { Loading } from "./Loading";
import { CHART, axis, gridTight, tooltipStyle } from "./chartBase";
import { fmtNum, fmtSigned } from "../lib/format";

// A small categorical palette for per-book series (distinct, terminal-toned).
const BOOK_COLORS = [
  CHART.pos,
  CHART.info,
  CHART.warn,
  "#c792ea",
  "#f78c6c",
  "#7fdbca",
  "#ff5370",
  "#addb67",
];

function shortTs(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts.slice(5, 16);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function LineMovementPanel({
  playerId,
  stat,
}: {
  playerId: number;
  stat: string;
}) {
  // Look back the full validated window (30d cap) so historical drift shows.
  const { data, isLoading } = useLineMovement(playerId, stat, 720);
  const timestamps = data?.timestamps ?? [];
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

  // Reset the scrubber to "fully drawn" whenever the data changes.
  useEffect(() => {
    setIdx(Math.max(0, timestamps.length - 1));
    setPlaying(false);
  }, [data]);

  useEffect(() => {
    if (!playing) return;
    if (idx >= timestamps.length - 1) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => setIdx((i) => i + 1), 650);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [playing, idx, timestamps.length]);

  // Forward-fill each book's line onto the shared timestamp axis so multiple
  // books render as clean aligned lines; reveal only up to the scrubber index.
  const series = useMemo(() => {
    if (!data) return [];
    return data.series.map((s, i) => {
      const byTs = new Map(s.points.map((p) => [p.ts, p.line]));
      let last: number | null = null;
      const aligned = timestamps.map((ts) => {
        if (byTs.has(ts)) last = byTs.get(ts)!;
        return last;
      });
      const revealed = aligned.map((v, j) => (j <= idx ? v : null));
      return {
        name: s.book,
        type: "line" as const,
        data: revealed,
        showSymbol: false,
        smooth: false,
        connectNulls: true,
        lineStyle: { width: 1.75, color: BOOK_COLORS[i % BOOK_COLORS.length] },
        itemStyle: { color: BOOK_COLORS[i % BOOK_COLORS.length] },
      };
    });
  }, [data, timestamps, idx]);

  if (isLoading) return <Loading rows={4} />;
  if (!data || !timestamps.length) {
    return (
      <EmptyState
        compact
        title="No stored line movement for this player+stat."
        hint="Line-movement replays betting_line_snapshots per book. Offseason slates have no fresh snapshots within the 30-day window; drift appears once books start posting in October."
        lastData={data?.last_snapshot_utc ? `last snapshot ${shortTs(data.last_snapshot_utc)}` : null}
      />
    );
  }

  const option: EChartsOption = {
    grid: { ...gridTight, top: 24, right: 20 },
    tooltip: { trigger: "axis", ...tooltipStyle },
    legend: {
      type: "scroll",
      top: 0,
      textStyle: { color: CHART.muted, fontFamily: CHART.fontMono, fontSize: 10 },
      inactiveColor: CHART.faint,
    },
    xAxis: {
      type: "category",
      data: timestamps.map(shortTs),
      boundaryGap: false,
      ...axis(),
    },
    yAxis: { type: "value", scale: true, ...axis({ name: "line" }) },
    animationDuration: 500,
    series,
  };

  return (
    <div className="space-y-2 px-2 py-2">
      <EChart option={option} height={240} />
      <div className="flex items-center gap-3 px-2">
        <button
          onClick={() => {
            if (idx >= timestamps.length - 1) setIdx(0);
            setPlaying((p) => !p);
          }}
          className="tnum w-16 rounded border border-line bg-panel-2 px-2 py-1 text-[11px] text-fg hover:border-line-strong"
        >
          {playing ? "❚❚ pause" : "▶ replay"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, timestamps.length - 1)}
          value={idx}
          onChange={(e) => {
            setPlaying(false);
            setIdx(Number(e.target.value));
          }}
          className="flex-1 accent-pos"
        />
        <span className="tnum w-40 shrink-0 text-right text-[11px] text-muted">
          {shortTs(timestamps[idx])}
        </span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 px-2 pt-1">
        {data.series.map((s, i) => (
          <span key={s.book} className="text-[11px] text-faint">
            <span
              className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle"
              style={{ backgroundColor: BOOK_COLORS[i % BOOK_COLORS.length] }}
            />
            {s.book}{" "}
            <span className="tnum text-muted">
              {fmtNum(s.open_line)}→{fmtNum(s.close_line)}
            </span>{" "}
            <span
              className={`tnum ${
                (s.line_delta ?? 0) > 0
                  ? "text-pos"
                  : (s.line_delta ?? 0) < 0
                    ? "text-neg"
                    : "text-faint"
              }`}
            >
              {fmtSigned(s.line_delta, 1)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
