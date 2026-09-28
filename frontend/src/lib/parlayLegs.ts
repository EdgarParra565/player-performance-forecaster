import type { LegSide, ParlayLegInput } from "../api/types";

// Compact, shareable URL form for a parlay: `?legs=` of
// `<player_id>~<stat>~<line>~<side>~<odds>~<name>` joined by `|`.
// Parsing is defensive: malformed entries are dropped, never thrown.

export const MAX_LEGS = 6;

export function encodeLegs(legs: ParlayLegInput[]): string {
  return legs
    .map((l) => [l.player_id, l.stat, l.line, l.side, l.odds, encodeURIComponent(l.player_name)].join("~"))
    .join("|");
}

export function decodeLegs(raw: string | null): ParlayLegInput[] {
  if (!raw) return [];
  const out: ParlayLegInput[] = [];
  for (const part of raw.split("|").slice(0, MAX_LEGS)) {
    const [pid, stat, line, side, odds, name] = part.split("~");
    const player_id = Number(pid);
    const lineN = Number(line);
    const oddsN = Number(odds);
    if (!Number.isInteger(player_id) || player_id < 1) continue;
    if (!stat || !/^[a-z_]{2,32}$/.test(stat)) continue;
    if (!Number.isFinite(lineN)) continue;
    if (side !== "over" && side !== "under") continue;
    let player_name = "";
    try {
      player_name = decodeURIComponent(name ?? "");
    } catch {
      player_name = "";
    }
    out.push({
      player_id,
      stat,
      line: lineN,
      side: side as LegSide,
      odds: Number.isInteger(oddsN) && oddsN !== 0 ? oddsN : -110,
      player_name,
    });
  }
  return out;
}

export function legKey(l: Pick<ParlayLegInput, "player_id" | "stat" | "line" | "side">): string {
  return `${l.player_id}-${l.stat}-${l.line}-${l.side}`;
}

// Props post on the half point (no pushes): 28.7 -> 28.5, 25.0 -> 25.5.
export function propLine(v: number): number {
  return Math.max(0.5, Math.floor(v) + 0.5);
}
