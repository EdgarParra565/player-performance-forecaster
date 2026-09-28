import { describe, it, expect } from "vitest";
import type { ParlayLegInput } from "../../api/types";
import { decodeLegs, encodeLegs, legKey, MAX_LEGS, propLine } from "../parlayLegs";
import { clampInt } from "../rows";

const legs: ParlayLegInput[] = [
  { player_id: 203999, player_name: "Nikola Jokić", stat: "points", line: 25.5, side: "over", odds: -110 },
  { player_id: 2544, player_name: "LeBron James", stat: "assists", line: 7.5, side: "under", odds: 120 },
];

describe("parlay leg URL codec", () => {
  it("round-trips legs including accented names", () => {
    expect(decodeLegs(encodeLegs(legs))).toEqual(legs);
  });

  it("drops malformed entries instead of throwing", () => {
    const raw = [
      "abc~points~25.5~over~-110~X", // bad id
      "1~POINTS;drop~25.5~over~-110~X", // bad stat
      "1~points~nan~over~-110~X", // bad line
      "1~points~25.5~sideways~-110~X", // bad side
      "1~points~25.5~over~0~%E0%A4%A", // odds 0 -> default, bad URI -> ""
    ].join("|");
    expect(decodeLegs(raw)).toEqual([
      { player_id: 1, player_name: "", stat: "points", line: 25.5, side: "over", odds: -110 },
    ]);
    expect(decodeLegs(null)).toEqual([]);
  });

  it("caps at MAX_LEGS", () => {
    const many = Array.from({ length: 10 }, (_, i) => ({ ...legs[0], line: i + 0.5 }));
    expect(decodeLegs(encodeLegs(many))).toHaveLength(MAX_LEGS);
  });

  it("keys legs by player/stat/line/side", () => {
    expect(legKey(legs[0])).toBe("203999-points-25.5-over");
  });

  it("rounds suggestions onto the half point", () => {
    expect(propLine(28.7)).toBe(28.5);
    expect(propLine(25)).toBe(25.5);
    expect(propLine(0.2)).toBe(0.5);
  });
});


describe("clampInt (URL state)", () => {
  it("bounds, defaults and rejects junk", () => {
    expect(clampInt("40", 3, 200, 25)).toBe(40);
    expect(clampInt("9999", 3, 200, 25)).toBe(200);
    expect(clampInt("1", 3, 200, 25)).toBe(3);
    expect(clampInt("abc", 3, 200, 25)).toBe(25);
    expect(clampInt("2.5", 3, 200, 25)).toBe(25);
    expect(clampInt(null, 3, 200, 25)).toBe(25);
  });
});
