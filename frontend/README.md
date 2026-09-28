# NBA Props Terminal — flagship UI

A standalone, visually-dense "trading terminal for sports" frontend for the NBA
player-props model. It is a **parallel** surface to the existing Streamlit web
app and Tk desktop app — those are untouched and keep working. This is the
flagship UI that will grow over the phases in `notes.txt`.

- **Backend:** a thin, **read-only** FastAPI service in [`../api/`](../api) that
  wraps the existing Python data layer (`player_charts` fetchers,
  `edge_scanner.score_prop_edges`, `DatabaseManager` consensus queries). It
  never writes to the DB and never duplicates model logic. All inputs are
  validated through `nba_model/web/input_validation.py`, exactly like the
  Streamlit dispatch.
- **Frontend:** React + TypeScript + Vite + Tailwind CSS v4, charts via Apache
  ECharts (`echarts-for-react`), data via TanStack Query hitting the API through
  Vite's dev proxy.

## Run it (two commands, two terminals)

Both sides run locally against the real SQLite DB at
`data/database/nba_data.db`.

**1. API** (from the repo root):

```bash
.venv/bin/python3 -m uvicorn api.main:app --reload --port 8000
```

Point it at a different DB with `NBA_DB_PATH=/path/to/nba_data.db`. Interactive
docs at http://localhost:8000/api/docs.

**2. Frontend** (from this `frontend/` directory):

```bash
npm install      # first time only
npm run dev
```

Open http://localhost:5173. Vite proxies every `/api/*` request to the uvicorn
service on :8000 (see `vite.config.ts`), so the browser only ever talks to Vite.
Point the proxy elsewhere with `API_PROXY_TARGET=http://localhost:8010 npm run dev`.

Other scripts: `npm run build` (typecheck + production bundle),
`npm run typecheck`, `npm run preview`.

### Production (one origin)

`npm run build`, then let uvicorn serve `dist/` next to the API:

```bash
FLAGSHIP_STATIC_DIR=$PWD/frontend/dist API_DOCS=0 \
  .venv/bin/python3 -m uvicorn api.main:app --port 8080
python -m api.smoke_check --base-url http://localhost:8080
```

Or `docker compose up flagship` (`Dockerfile.flagship`). Env vars, reverse
proxy / rate-limit setup, and the **no-auth / no-rate-limit caveats for public
deploys** are in [`docs/DEPLOYMENT.md` §15](../docs/DEPLOYMENT.md).

## What's in Phase 1

Three views, each built end-to-end against the real DB with a deliberate
empty-state (it is the NBA offseason — most live-line surfaces are empty until
October, and the UI shows "last data" / "last scrape" rather than blank panes):

1. **Slate Dashboard** (`/`) — KPI row (games in DB, players tracked, books
   producing, freshest scrape), a "top model edges" table (edge scanner in
   `full` mode), and a recent-games strip.
2. **Player Detail** (`/player`) — the flagship view. Searchable player picker,
   recent-N performance chart (rolling mean + book-mean overlay), distribution
   histogram with a fitted-normal overlay and per-book line markers, a per-book
   table with P(over)/edge/EV, and hit-rate bars.
3. **Edge Scanner** (`/edges`) — the scored slate as a dense sortable/filterable
   table (books, stats, min-edge, min-P(over), only-+EV) with the same
   `chart_mean | rolling | full` model-mode semantics as the CLI.

## What's in Phase 2 (money views)

1. **Cross-book** (`/cross-book`) — line shopping & middle candidates from the
   DFS board plus a distinct TRUE two-way arbitrage section (real posted odds
   only). KPI row (pairs / max gap / ≥min-gap / true arbs / freshest), model-mode
   / min-gap / min-books / books / stats filters, CSV export, Player-Detail jump.
   Middles are labeled candidates, never guaranteed.
2. **Line-movement replay** — a panel on Player Detail that animates
   `betting_line_snapshots` drift per book (play/pause + scrubber, per-book
   open→close deltas).
3. **Team Charts** (`/teams`) — per-game team aggregates; points carries the
   implied team total, other stats show the props-derived reference (Σ players'
   consensus lines, ≥5-player floor, clearly labeled as derived).

ECharts is now code-split (lazy-loaded) so the initial route ships ~370 KB
instead of ~1.4 MB.

## What's in Phase 3

1. **Parlay builder** (`/parlay`) — add 2–6 legs across players.
   - Shows per-leg P(hit) vs implied and the correlation-aware joint
     probability (the existing `calibrate_correlations` →
     `simulate_multi_leg_sgp` model path) next to the naive independent
     product.
   - Also shows fair / leg-product / book price, EV, and a correlation
     heatmap.
   - Legs live in the URL (`?legs=`), so a parlay is shareable.
2. **Paper trades** (`/paper-trades`) — `bet_log` picks split settled /
   pending, with W–L, win rate, CLV and estimated units.
   - Includes a calibration panel: a reliability curve plus Brier score, from
     paper trades or the hourly model predictions.
   - When bet_log is empty, it shows how to run the `bet_slip` exporter.

Both carry the standing "model estimate — not validated for real money" banner.

## Watchlist + mobile (Phase 4)

- **Watchlist** (`/watchlist`) — star (☆) a player in Player Detail, or a
  prop in the Slate / Edge Scanner / Cross-book tables.
  - The view shows the best current line, side, P, edge and EV per starred
    item from the full-model board, or "Not on the board".
  - Kept in localStorage (`lib/watchlist.ts`; no server writes). It survives
    refresh and syncs across tabs.
  - "Share link" copies `/watchlist?w=…`; opening one offers Add / Replace /
    Dismiss.
- **Responsive:**
  - Phones (< 768px): a bottom tab bar (Slate · Player · Edges · Watch ·
    More) plus a slide-in drawer with every view.
  - Tablets (768–1023px): an icon-only rail. Desktop (≥ 1024px): the full rail.
  - Touch devices get 44px targets via Tailwind's `pointer-coarse:` variant
    (desktop stays dense).
  - Tables scroll inside their cards with a sticky first column.
  - Chart axes drop overlapping labels (`hideOverlap`).

**Private deploys:** when the server sets `FLAGSHIP_ACCESS_CODE`, the app
prompts once for the code, stores it in localStorage and sends it as
`X-Access-Code` (`components/AccessGate.tsx`, `lib/access.ts`). Unset means
fully open. See `docs/DEPLOYMENT.md` §15.

## Design system

Everything visual comes from one place, so the five views read as one product:

- **Tokens:** `src/index.css` `@theme` holds the CSS side; `src/lib/tokens.ts`
  holds the same values for canvas/ECharts. `lib/__tests__/tokens.test.ts`
  fails if the two drift.
  - Layered dark surfaces: `base` → `surface-1` (cards) → `surface-2`
    (headers/inputs) → `surface-3` (active) → `surface-4`.
  - A neutral periwinkle `accent` for chrome: active nav, selection, focus rings.
  - `pos`/`neg` (green/red) are **reserved for meaning**: +EV/−EV and hit/miss
    vs the line. Over/under is a direction, not a value judgement, so
    `SideTag` renders it neutral. The categorical series palette contains no
    polarity hues (a test enforces this).
- **Type:** Inter Variable (UI, optical sizing) + JetBrains Mono Variable
  (data), self-hosted via `@fontsource-variable/*`, with no font CDN. The build
  never inlines fonts, which keeps the production CSP at `font-src 'self'`.
  - Scale: `text-caption` 11 · `label` 12 · `body` 13 · `title` 15 ·
    `heading` 22 · `kpi` 26 px.
  - Every number uses tabular figures: `.tnum` (mono, slashed zero) in tables
    and `.num` (Inter tabular) for KPI figures.
- **Spacing:** Tailwind's 4px grid only (multiples of 4/8). Controls are 32px
  tall, table rows 40px, cards 16px padded with a 24px page rhythm.
- **Shell:** `Page` (title/description/actions), `Card` (titled container;
  `tone="pos"` only for the true-arb section) and `KpiGrid` in
  `components/Page.tsx`.
- **Components:**
  - `StatCard`, `DataTable` (sticky header; sort buttons with `aria-sort`;
    keyboard-activatable rows), `ProbabilityBar` (break-even tick, `meter`
    role), `OddsBadge` (always signed), `Delta`/`SideTag`, `EmptyState`,
    `ErrorState`
  - Skeletons: `Skeleton`, `KpiSkeleton`, `TableSkeleton`, `ChartSkeleton`
  - Controls in `controls.tsx`: `Segmented`, `Chip`, `NumberField` (commits on
    blur/Enter), `Select`, `Toggle`, `Button`, `FilterRow`, `useDebounced`
- **Charts (ECharts 6):** ONE ECharts theme, `components/chartTheme.ts`, registered as
  `"terminal"` inside the lazy chunk.
  - Theme: soft gridlines, styled dark tooltips, mono axis labels, and series
    colors from tokens.
  - `grid()` uses `containLabel`, so labels don't collide with the plot edge.
  - `refLabel()` draws reference lines; `tooltipHtml()` builds formatted,
    **HTML-escaped** tooltips.
  - Option builders are pure functions in `src/charts/`, memoised in views.
  - `EChart` waits for `document.fonts.ready` before the first canvas paint
    and has its own error boundary.
- **Formatting:** only through `src/lib/format.ts`.
  - Odds are always signed; percentages are 1dp; EV is shown as `+0.76u`.
  - `-0.0` never renders.
  - Date-only strings are calendar dates (no UTC day-shift); naive DB
    timestamps are UTC.

## API surface

All read-only (the one POST, parlay pricing, only computes), rate-limited
per client IP, with `Cache-Control` (short `max-age` on GET data, `no-store`
on health):

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status + DB existence + table counts + freshness (path only with `API_EXPOSE_DB_PATH=1`) |
| `GET /api/meta` | stats / teams / seasons / books for filters |
| `GET /api/slate/kpis` | dashboard KPI row |
| `GET /api/slate/recent-games` | recent matchups strip |
| `GET /api/slate/edges` | scored edges (dashboard + Edge Scanner) |
| `GET /api/players/search` | server-side player search |
| `GET /api/players/{id}` | player detail (series, distribution, book table) |
| `GET /api/players/{id}/line-movement` | snapshot drift per book (replay) |
| `GET /api/cross-book` | line shopping / middles + TRUE two-way arb |
| `GET /api/teams/{team}/chart` | per-game team aggregates + derived reference |
| `POST /api/parlay/price` | correlation-aware parlay pricing (compute-only, nothing stored) |
| `GET /api/paper-trades` | bet_log picks + W/L/CLV summary (`status=all|pending|settled`) |
| `GET /api/paper-trades/calibration` | reliability buckets + Brier (`source=bet_log|predictions`) |

### Tests

```bash
# API (httpx TestClient against a temp seeded DB) — 99 tests (incl. db_sync + publish script)
.venv/bin/python3 -m pytest api/tests -q

# Frontend (vitest 4 + testing-library) — 99 tests: format, tokens sync, CSV,
# chart builders/theme escaping, controls, DataTable a11y, replay scrubber,
# parlay/paper-trades views, leg URL codec, access gate + client header,
# watchlist store/codec/matching, star button, mobile drawer + bottom nav
cd frontend && npm run test
```

## Screenshots

Local before/after dumps of the views live at `docs/screenshots/flagship/`
after a design pass. They are gitignored and not required to run the app.

## Notes

- ECharts is code-split behind a lazy boundary (`EChart`) with a
  `manualChunks` vendor split, so the initial route is ~380 KB (~120 KB gz)
  and the ~1 MB charting chunk loads only when a chart renders.
- A parallel agent runs live scraper tests against the same DB — treat the DB as
  read-only and expect its contents to change under you.
