# NBA Props Terminal — flagship UI (Phase 1)

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

Other scripts: `npm run build` (typecheck + production bundle),
`npm run typecheck`, `npm run preview`.

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

## Design system

A small real design system drives every view (in `src/components/` and the
`@theme` tokens in `src/index.css`):

- **Tokens** — near-black surfaces, one restrained accent per polarity
  (electric green for +EV, red for -EV), monospaced tabular numerals for every
  line/odd/probability.
- **Components** — `StatCard`, `DataTable` (sticky header + click-to-sort),
  `ProbabilityBar` (with break-even tick), `OddsBadge`, `LineSparkline`,
  `EmptyState`, `Delta`, plus shared ECharts styling in `chartBase.ts`.

## API surface

All GET, all read-only, all with `Cache-Control` (short `max-age` on data,
`no-store` on health):

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status + DB path/existence + table counts + freshness |
| `GET /api/meta` | stats / teams / seasons / books for filters |
| `GET /api/slate/kpis` | dashboard KPI row |
| `GET /api/slate/recent-games` | recent matchups strip |
| `GET /api/slate/edges` | scored edges (dashboard + Edge Scanner) |
| `GET /api/players/search` | server-side player search |
| `GET /api/players/{id}` | player detail (series, distribution, book table) |
| `GET /api/players/{id}/line-movement` | snapshot drift per book (replay) |
| `GET /api/cross-book` | line shopping / middles + TRUE two-way arb |
| `GET /api/teams/{team}/chart` | per-game team aggregates + derived reference |

### Tests

```bash
# API (httpx TestClient against a temp seeded DB) — 21 tests
.venv/bin/python3 -m pytest api/tests -q

# Frontend component tests (vitest + testing-library) — 14 tests
cd frontend && npm run test
```

## Notes

- ECharts is code-split behind a lazy boundary (`EChart` / `LineSparkline`) with
  a `manualChunks` vendor split, so the initial route no longer pays the ~1 MB
  charting cost.
- A parallel agent runs live scraper tests against the same DB — treat the DB as
  read-only and expect its contents to change under you.
