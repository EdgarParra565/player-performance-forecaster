import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { COLOR, SERIES } from "../tokens";

// index.css (@theme) and tokens.ts must describe the same palette: CSS drives
// Tailwind utilities, tokens.ts drives the ECharts canvas.
const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");

function cssVar(name: string): string | undefined {
  const m = new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6})`).exec(css);
  return m?.[1].toLowerCase();
}

function kebab(key: string): string {
  return key.replace(/([a-z])([A-Z0-9])/g, "$1-$2").toLowerCase();
}

describe("design tokens", () => {
  it.each(Object.entries(COLOR))("%s matches index.css", (key, hex) => {
    expect(cssVar(kebab(key))).toBe(hex.toLowerCase());
  });

  it("keeps polarity hues out of the categorical series palette", () => {
    const polarity = [COLOR.pos, COLOR.neg, COLOR.posDim, COLOR.negDim];
    for (const c of SERIES) expect(polarity).not.toContain(c);
  });

  it("does not load fonts from a third-party CDN", () => {
    const html = readFileSync(resolve(__dirname, "../../../index.html"), "utf8");
    expect(html).not.toMatch(/fonts\.googleapis|fonts\.gstatic/);
  });
});
