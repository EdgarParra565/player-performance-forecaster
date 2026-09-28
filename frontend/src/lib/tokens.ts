// Design tokens for code that can't read CSS variables (ECharts canvas).
// index.css `@theme` is the CSS side of the same values; the
// `tokens.test.ts` suite fails if the two drift apart.

export const COLOR = {
  // Surfaces, darkest -> most raised (layered elevation).
  base: "#0a0b0e",
  surface1: "#101217",
  surface2: "#161920",
  surface3: "#1d2129",
  surface4: "#252a34",

  // Hairlines.
  line: "#1f232b",
  lineStrong: "#2d323d",

  // Text.
  fg: "#eef1f6",
  muted: "#a0a8b6",
  faint: "#7a8291",

  // Neutral UI accent (selection, focus, active nav). NOT a polarity color.
  accent: "#8ea2ff",
  accentSoft: "#1b2040",

  // Polarity — reserved for meaning (+EV / -EV, hit / miss vs the line).
  pos: "#16dc97",
  posDim: "#0d7a55",
  posSoft: "#0d2820",
  neg: "#ff5a6e",
  negDim: "#a8303f",
  negSoft: "#2b1318",

  // Support.
  warn: "#f5b544",
  info: "#5ab4ff",
} as const;

// Categorical series palette (books, multi-series). Deliberately avoids the
// pos/neg hues so a series color is never mistaken for +EV / -EV.
export const SERIES = [
  "#8ea2ff", // periwinkle (accent)
  "#4cc9f0", // cyan
  "#f5b544", // amber
  "#c792ff", // violet
  "#f472b6", // pink
  "#fb923c", // orange
  "#cbd5e1", // slate
  "#a3e1ff", // ice
] as const;

export const FONT = {
  sans: "'Inter Variable', Inter, ui-sans-serif, system-ui, -apple-system, sans-serif",
  mono: "'JetBrains Mono Variable', 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace",
} as const;

// Type scale (px) shared with charts; CSS side is `--text-*` in index.css.
export const TYPE = {
  caption: 11,
  label: 12,
  body: 13,
  title: 15,
  heading: 22,
  kpi: 26,
} as const;
