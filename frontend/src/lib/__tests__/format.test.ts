import { describe, it, expect } from "vitest";
import {
  fmtAgo,
  fmtDateShort,
  fmtEv,
  fmtHoursAgo,
  fmtLine,
  fmtNum,
  fmtOdds,
  fmtPct,
  fmtSigned,
  fmtSignedPct,
  parseApiDate,
} from "../format";

describe("number formatting", () => {
  it("signs American odds", () => {
    expect(fmtOdds(110)).toBe("+110");
    expect(fmtOdds(-110)).toBe("-110");
    expect(fmtOdds(null)).toBe("—");
  });

  it("formats percentages at 1dp by default", () => {
    expect(fmtPct(0.5321)).toBe("53.2%");
    expect(fmtPct(undefined)).toBe("—");
    expect(fmtPct(Number.NaN)).toBe("—");
    expect(fmtPct(Number.POSITIVE_INFINITY)).toBe("—");
  });

  it("never prints a negative zero", () => {
    expect(fmtNum(-0.04)).toBe("0.0");
    expect(fmtSigned(-0.0004)).toBe("0.000");
    expect(fmtSignedPct(-0.0001)).toBe("0.0%");
  });

  it("signs positive edges and EV", () => {
    expect(fmtSignedPct(0.042)).toBe("+4.2%");
    expect(fmtSignedPct(-0.042)).toBe("-4.2%");
    expect(fmtEv(0.758)).toBe("+0.76u");
    expect(fmtEv(-0.25)).toBe("-0.25u");
  });

  it("formats prop lines with one decimal", () => {
    expect(fmtLine(27.5)).toBe("27.5");
    expect(fmtLine(6)).toBe("6.0");
  });
});

describe("date parsing", () => {
  it("treats date-only strings as calendar dates (no UTC day shift)", () => {
    const d = parseApiDate("2026-06-13")!;
    expect(d.getFullYear()).toBe(2026);
    expect(d.getMonth()).toBe(5);
    expect(d.getDate()).toBe(13);
    expect(fmtDateShort("2026-06-13")).toBe("Jun 13");
  });

  it("treats naive DB timestamps as UTC", () => {
    const naive = parseApiDate("2026-07-22 01:25:56")!;
    const explicit = parseApiDate("2026-07-22T01:25:56+00:00")!;
    expect(naive.getTime()).toBe(explicit.getTime());
  });

  it("computes freshness from UTC timestamps", () => {
    const now = Date.parse("2026-07-22T03:25:56Z");
    expect(fmtAgo("2026-07-22 01:25:56", now)).toBe("2.0h ago");
    expect(fmtAgo(null, now)).toBe("no data");
  });

  it("formats hour ages", () => {
    expect(fmtHoursAgo(0.7)).toBe("42m ago");
    expect(fmtHoursAgo(72)).toBe("3d ago");
    expect(fmtHoursAgo(null)).toBe("—");
  });
});
