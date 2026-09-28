import { describe, it, expect } from "vitest";
import { escapeCell, toCsv } from "../csv";

describe("CSV export", () => {
  it("neutralises spreadsheet formula injection", () => {
    expect(escapeCell("=HYPERLINK(\"http://x\")")).toBe("\"'=HYPERLINK(\"\"http://x\"\")\"");
    expect(escapeCell("+1")).toBe("'+1");
    expect(escapeCell("@SUM(A1)")).toBe("'@SUM(A1)");
    expect(escapeCell("-cmd")).toBe("'-cmd");
  });

  it("leaves numbers (including negatives) alone", () => {
    expect(escapeCell(-3.5)).toBe("-3.5");
    expect(escapeCell(Number.NaN)).toBe("");
  });

  it("quotes separators, quotes and CR/LF", () => {
    expect(escapeCell("a,b")).toBe('"a,b"');
    expect(escapeCell('say "hi"')).toBe('"say ""hi"""');
    expect(escapeCell("a\rb")).toBe('"a\rb"');
  });

  it("builds header + rows", () => {
    const csv = toCsv([{ n: "LeBron", v: 1 }], [
      { key: "n", header: "player", value: (r) => r.n },
      { key: "v", header: "val", value: (r) => r.v },
    ]);
    expect(csv).toBe("player,val\nLeBron,1\n");
  });
});
