# NBA Player Performance Forecaster

Probabilistic NBA player-props modeling project focused on points forecasts, EV estimation, and backtesting.

The current baseline is designed to be reproducible offline (synthetic benchmark harness) while still supporting real NBA API data through caching when available.

## Current Status

- Core data contract is unified around lowercase stat columns: `points`, `assists`, `rebounds`, `minutes`.
- Backtest pipeline is wired end-to-end without import/signature mismatches.
- `run_model` supports executable demo modes for single props and correlated parlays.
- Deterministic smoke tests are in place for features, probability math, and backtest integration.
- Baseline benchmark artifacts are generated and saved under `nba_model/evaluation/artifacts/`.
- Per-book scraper package (`nba_model/scrapers/`) registers 20 books — 9 actively producing parsed data (PrizePicks, Underdog, Pick6, ParlayPlay for player props; BetMGM, Caesars, DraftKings, Bovada, Kalshi for team lines). Cross-book consensus rolls into both player and team charts as `book mean X.X`.
- NBA API ingest (`nba_model/data/nba_results_ingestion.py`) populates a `games` table (8K+ team-game rows) and bulk player game logs (90K+ rows across 3 seasons + 1K players) from `leaguegamefinder` / `playergamelogs`.
- Streamlit + Tk UIs both expose Player charts, Team charts, Game Results, and Player Stats Browse. All graphing inputs go through `nba_model/web/input_validation.py` (stat type / team code / season / rolling window).
- Production-ready auth + billing scaffold: native Streamlit OIDC (Google + Microsoft) sign-in, Stripe-powered Free vs Premium tiers, FastAPI webhook handler with HMAC verification, replay tolerance, idempotency, body-size cap (256 KiB), per-IP rate limiting (120/60s), slowloris timeout, and the OWASP-recommended HTTP security headers. Full threat model + 4-layer hardening pass + adversarial / data-poisoning input validation in [docs/SECURITY.md](docs/SECURITY.md).
- 182/182 tests passing (regression + stress + adversarial + scanners). `bandit` MEDIUM/HIGH baseline = 0; `pip-audit` clean; 30 consecutive stress runs flake-free.
- **Multi-sport scaffolding** in place: `sports/` package at the project root with `Sport` config dataclass + modules for `nba` (live) and stubs for `nfl`, `mlb`, `nhl`, `soccer` (sub-leagues: EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL; future: Copa Libertadores + Copa Sudamericana). Streamlit sidebar has a sport-picker — selecting a non-live sport shows a roadmap card in the main pane with that sport's stat types, sub-leagues, and open questions. Full rollout plan in [docs/MULTI_SPORT_PLAN.md](docs/MULTI_SPORT_PLAN.md).

## Repository Layout

- `nba_model/data/` - API loading, caching, and database utilities
  - `nba_results_ingestion.py` - bulk team-game + player-game-log ingest from `nba_api`
- `nba_model/model/` - feature engineering, probability, simulation, EV, parlay modeling
  - `browser_prop_parser.py` / `team_line_parser.py` - orchestrators that run each scraper's preprocessor over stored web snapshots
  - `web_text_ingestion.py` - generic fetch path (Playwright + CDP via Chrome :9222)
- `nba_model/scrapers/` - per-book scraper registry
  - `base.py` - `BookScraper` dataclass + shared regex helpers
  - `team_names.py` - canonical NBA team-name normalizer (cross-book joining)
  - one module per book: `prizepicks.py`, `underdog.py`, `pick6.py`, `parlayplay.py`, `betmgm.py`, `caesars.py`, `draftkings.py`, `bovada.py`, `kalshi.py`, plus 11 stub configs
- `nba_model/evaluation/` - backtesting and benchmark runners
- `nba_model/tests/` - smoke and integration-style tests
- `nba_model/visualization/` - distribution plotting + Player/Team chart builders (`player_charts.py` powers both UIs; also exposes `fetch_recent_games` / `fetch_player_recent_results` / `list_seasons` for the browse views)
- `nba_model/web/` - Streamlit web frontend + auth + Stripe membership glue
  - `app.py` - the Streamlit app (Player charts, Team charts, Game Results, Player Stats Browse, plus premium views)
  - `auth.py` - native OIDC wrapper + tier-based feature gating
  - `input_validation.py` - validators for all chart inputs + ingested odds
  - `subscriptions.py` - SQLite-backed subscription store
  - `stripe_helpers.py` - Checkout URL builder + signature helpers
  - `webhook_app.py` - FastAPI app that handles Stripe webhooks
  - `parlay_compare.py` - cross-comparison helpers for the Parlay analysis view
- `sports/` - **multi-sport registry** (top-level, sibling to `nba_model`)
  - `__init__.py` - `Sport` dataclass + `SPORTS` registry + `get_sport(key)` lookup
  - `nba.py` - NBA config (live)
  - `nfl.py`, `mlb.py`, `nhl.py` - per-sport stubs with stat types, ranges, team codes, open questions
  - `soccer/__init__.py` - parent + 6 sub-league `Sport` configs (EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL); commented-out Copa Libertadores + Copa Sudamericana entries for the future South American phase
- `data/DATABASE_INVENTORY.txt` - auto-generated snapshot of what's actually in the SQLite DB (refresh: `python3 -m nba_model.data.audit_db`)
- `docs/API_TO_DATABASE.md` - how data flows from APIs to SQLite and how we clean it
- `docs/MULTI_SPORT_PLAN.md` - architecture + rollout order + per-sport open questions for the NFL → MLB → NHL → soccer expansion

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Docker (web app)

The fastest way to run the Streamlit web app on any machine with Docker:

```bash
# Build once.
docker build -t nba-model .

# Open-access mode (BILLING_ENABLED=0 by default - matches the launch
# configuration; no Stripe / OIDC secrets required).
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" nba-model
```

Or via docker-compose. The `tests` and `etl-bulk` services are gated
behind the `tools` profile (one-shots, kept out of the default `up`).
The `webhook` service is gated behind the `billing` profile (Stripe-only).

```bash
# Long-running services
docker compose up streamlit                       # web UI on http://localhost:8501
docker compose --profile billing up webhook        # FastAPI Stripe webhook on :8000

# One-shot services
docker compose --profile tools run --rm tests      # full pytest suite (194 cases)
docker compose --profile tools run --rm etl-bulk   # bulk nba_api ingest
docker compose --profile tools run --rm \
  -e SEASONS="2025-26 2024-25" etl-bulk            # multi-season refresh

docker compose down                                # tear everything down
```

The image is ~1.4 GB (matplotlib + pandas/numpy/scipy + streamlit + plotly),
boots to healthy in ~2 s, and runs at ~70 MB resident memory. The container
deliberately excludes Playwright / Chromium — scraping paths assume a real
Chrome on `:9222` on the developer's host, which a container can't replicate
without losing the auth session anyway.

The `tests` service bind-mounts the whole repo (test files are excluded
from the image at build time via `.dockerignore` to keep the prod image
lean) and installs pytest at start time — the production lockfile has a
pycookiecheat ↔ cryptography conflict that drops dev packages on partial
install, so we add pytest only when actually running tests.

All four services share the bind-mounted `./data` directory, so the
SQLite DB written by `etl-bulk` is immediately visible to `streamlit`
(and to your local Python) without rebuilding.

To re-engage Stripe + OIDC billing in the container:

```bash
export BILLING_ENABLED=1
# Edit .streamlit/secrets.toml first (see .streamlit/secrets.toml.example),
# then uncomment the secrets bind-mount line in docker-compose.yml.
docker compose up streamlit webhook
```

## Run Commands

### 1) Single Prop or Parlay Demo

```bash
python3 -m nba_model.run_model --mode single --player "LeBron James" --n-games 100
python3 -m nba_model.run_model --mode parlay --player "LeBron James" --n-games 100
python3 -m nba_model.run_model --mode both --player "LeBron James" --n-games 100
```

## Security

The threat model and the full mitigation matrix live in
**[docs/SECURITY.md](docs/SECURITY.md)**. Highlights:

- **Auth**: native Streamlit OIDC (Google + Microsoft); session cookies signed
  with `cookie_secret`; `st.logout` invalidates the cookie; admin override is
  email-allowlist only.
- **Authorization**: tier (`free` vs `premium`) is the single source of
  truth, resolved on every request through `auth.current_user()`. Free-tier
  users have a server-side player allowlist + stat allowlist + N-games cap
  enforced inside dispatch (defense in depth, not just UI hiding).
- **SQL injection**: all queries parameter-bound. The single f-string SQL
  fragment (`fetch_team_chart_data`) interpolates only allowlisted
  expressions and `assert`s at the call site.
- **Webhook**: HMAC verified by `stripe.Webhook.construct_event` with a
  300s replay tolerance; body size capped at 256 KiB before buffering;
  `asyncio.wait_for` body-read timeout (slowloris defense, 408 on timeout);
  per-IP sliding-window rate limit (default 120 req / 60 s, 429 with
  `Retry-After`); idempotent on `event_id`; `TrustedHostMiddleware` honors
  `WEBHOOK_TRUSTED_HOSTS`; `/docs`, `/redoc`, `/openapi.json` all return 404.
- **HTTP security headers** (every webhook response): HSTS 2yr+preload, CSP
  `default-src 'none'`, X-Content-Type-Options nosniff, X-Frame-Options
  DENY, Referrer-Policy no-referrer, Permissions-Policy denying camera /
  mic / geo / payment / USB, Cross-Origin-Opener-Policy + Cross-Origin-
  Resource-Policy same-origin. `Server:` header overridden to "webhook" so
  scanners can't fingerprint Uvicorn.
- **Subscription DB**: `chmod 0600`, WAL mode, SQLite extensions disabled,
  one-time `journal_mode=WAL` flip behind a process-wide `_INIT_LOCK` +
  `busy_timeout=5000` on every connection (concurrent-write contention is
  flake-free), email validated by regex + length + punycode/IDN reject
  before being used as a primary key.
- **Adversarial input + data-poisoning** (AI threat model): every numeric
  user input goes through `nba_model/web/input_validation.py` (NaN / inf /
  per-stat plausible range / hard caps on n_games, n_sims, parlay legs).
  `is_plausible_betting_line` runs at insert time so scraper-poisoned rows
  ("Jokic 9999.5 points", unknown stat types, zero-odds) are dropped before
  they pollute consensus mean / EV math.
- **Logging**: webhook never logs payload bodies, emails, or customer ids -
  only event_id + event_type. Bad-signature attempts log only the exception
  type, never the raw header.
- **Dependencies**: pinned with `~=` ranges and a `requirements.lock` for
  reproducible deploys. `pip-audit` + `bandit` + secret-pattern grep run
  via `scripts/security_scan.sh` (currently clean: bandit MEDIUM/HIGH = 0,
  pip-audit "No known vulnerabilities").

Run the security regression tests + dependency / static analysis scans:

```bash
# 80+ security cases: regression (test_security.py) + stress + fuzz +
# adversarial inputs (test_security_stress.py)
.venv/bin/python3 -m pytest nba_model/tests/test_security.py \
                              nba_model/tests/test_security_stress.py -v

# pip-audit (CVEs in installed packages) + bandit (SAST) + secret-pattern grep
.venv/bin/python3 -m pip install -r requirements-dev.txt
./scripts/security_scan.sh

# crash-consistent backup of the subscriptions DB (uses SQLite .backup)
./scripts/backup_subscriptions.sh
```

## Production deployment (auth + Stripe memberships)

The web app supports **Google + Microsoft OIDC sign-in** via Streamlit's
native `st.login` (1.42+) and gates premium features behind a Stripe
subscription. See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for the full
deployment walkthrough — recommended target is **Streamlit Community Cloud**
for the app + a small FastAPI webhook handler on Render/Railway/Fly.

Quick summary of what's wired:

- `nba_model/web/auth.py` — wraps `st.user`/`st.login`/`st.logout`, resolves
  the current tier (`free` / `premium`), exposes `paywall(feature)` for
  feature-gating callers, and an admin-email override for the project owner.
- `nba_model/web/subscriptions.py` — SQLite store at
  `data/database/subscriptions.db` with one row per email; idempotent
  Stripe-event recording.
- `nba_model/web/stripe_helpers.py` — builds the Checkout URL from either a
  pre-built **Payment Link** or a server-side Checkout Session.
- `nba_model/web/webhook_app.py` — FastAPI app on `/stripe/webhook` that
  verifies signatures with the Stripe SDK and updates the subscription store
  on `checkout.session.completed`, `customer.subscription.{created,updated,
  deleted}`, `invoice.paid`, `invoice.payment_failed`.
- `.streamlit/secrets.toml.example` — full template with placeholders for
  OIDC + Stripe + admin allowlist. Copy to `secrets.toml` (gitignored).

**Free vs Premium feature matrix** (configured in `auth.py`):

| Feature                          | Free preview                          | Premium |
|----------------------------------|---------------------------------------|---------|
| Players viewable                 | `Nikola Jokic`, `LeBron James`        | All     |
| Stats                            | `points`                              | All 7   |
| Last N games                     | up to 5                               | up to 200 |
| Player charts                    | yes                                   | yes     |
| Team charts                      | yes (preview stats only)              | yes (all stats) |
| Game Results browser             | yes                                   | yes     |
| Player Stats Browse              | yes                                   | yes     |
| All-stats overview               | locked                                | yes     |
| Parlay analysis (single + multi) | locked                                | yes     |

### 1a) Player Charts Web App (Streamlit)

A browser frontend dedicated to the Player Charts feature. Pick a team / player /
stat in the left sidebar; the main pane re-renders every chart and the EV
summary on selection change. Switching player reloads everything live.

```bash
.venv/bin/python3 -m streamlit run nba_model/web/app.py
```

Then open http://localhost:8501. Layout:

- **Sidebar**: DB path, team filter, player dropdown, **view mode** radio
  (`Player charts | Team charts | Game Results | Player Stats Browse` for free,
  premium adds `All stats overview` + `Parlay analysis`), stat, last-N games, rolling
  window, distribution overlay checkboxes (normal / poisson / negative-binomial),
  "Reload DB indexes" button to bust caches. All numeric/categorical inputs
  (stat type, team code, season, n_games, rolling window) are validated through
  `nba_model/web/input_validation.py` before they hit the DB or model code; bad
  values surface as a friendly `st.error` instead of a stack trace.
- **KPI row**: games / mean / std / market median line / count of `+EV` book sides.
- **Tabs in the main pane:**
  - **Overview** — recent-N-games chart with rolling mean, plus distribution
    histogram with the selected fitted distribution overlays and a vertical
    dashed line per book at that book's current line.
  - **Splits** — home/away + rest-day-bucket chart, plus the same numbers in two
    side-by-side tables.
  - **Hit Rate + Custom Line** — historical-over-rate-per-book bar chart with a
    tick at each book's break-even, then a sortable book-lines table with
    progress-bar columns for `P(over)` and `hit_rate`, then a custom-line probe
    form (line + American odds → fitted P(over), historical over-rate, EV per
    unit for OVER and UNDER, plus a `+EV/-EV` verdict).
  - **Raw data** — the actual game-log rows used to build the charts.

The Player charts view above is one of six top-level view modes. The others:
- **Team charts** (free + premium): per-game team aggregates with a
  `book mean` reference line derived from the cross-book consensus
  `(game_total - team_spread) / 2` per book (currently `points` only — the
  only stat books surface at the lobby level). Free tier sees preview
  stats; premium sees all of `points / assists / rebounds / pra / 3pm /
  fgm / minutes`.
- **Game Results** (free): recent NBA matchups with final scores + winner.
  Filter by season / season type (Regular Season / Playoffs / Play-In /
  Pre Season) / team. Reads from `games` (populated by
  `nba_results_ingestion.py`).
- **Player Stats Browse** (free): searchable league-wide recent player
  game logs. Pair `Stat filter` + `Min value` to find e.g. "last 200 games
  where someone scored ≥ 30." Player names resolve via
  `nba_active_players_ref` (530 active players) so abbreviated rows still
  show canonical names.
- **All stats (overview)** (premium): every stat for the selected player on
  one page.
- **Parlay analysis** (premium): single-prop or multi-leg parlay with
  cross-comparison of model + chart-data + historical results.

The web app uses the same `nba_model/visualization/player_charts.py` module as
the desktop UI, so any new figure builder added there automatically becomes
available to both surfaces. The desktop UI mirrors the new browse views as
the **Game Results** and **Player Stats Browse** notebook tabs.

### 1b) Simple Desktop UI

Run from a lightweight local UI (Single Prop + Parlay with PrizePicks/Underdog inputs):

```bash
python3 -m nba_model.simple_ui
```

In the Parlay tab you can input:
- sportsbook (`prizepicks`, `underdog`, or `custom`)
- leg stats and lines
- parlay odds and odds format (`american`, `decimal`, or `multiplier`)
- bounded sliders for correlation/volatility severity, plus reset-to-default

In the Single Prop tab you can also tune:
- opponent/league defense inputs and defense sensitivity
- blowout threshold + blowout penalty
- bounded severity sliders (defense, minutes penalty, and sigma)
- selectable distribution family (`normal`, `student_t`, `binomial`, `poisson`, `exponential`, `uniform`, `lognormal`, `power_law`)
- reset-to-default button for one-click baseline restore

In the `Manual Lines Import` tab you can paste sportsbook rows and write them into
`betting_lines` for market-line backtests. Supported row formats (pipe/csv/tab):
- `player | stat | line | over_odds | under_odds`
- `player | game_date | book | stat | line | over_odds | under_odds`
- raw board text paste is also supported (auto-extracts player + matchup + line + stat blocks)
After parsing, use the table preview to verify rows and delete selected entries before saving.

#### Player Charts tab (highlighted feature)

Pick a team and player, choose a stat (`points / assists / rebounds / pra / minutes`)
and a window of recent games — the UI pulls from the local SQLite DB and renders
four interactive matplotlib charts plus an EV summary:

- **Overview view**
  - Recent-N-games bar chart with rolling-mean overlay and a dashed horizontal
    line at the median book line for that stat.
  - Distribution histogram of recent values with one or more fitted distributions
    on top (toggle any combination of normal / poisson / negative-binomial), and
    one colored vertical dashed line per book at that book's current line.
- **Splits view**
  - Home vs away mean (with sample sizes).
  - Mean by rest-day bucket (`0-1`, `2`, `3+` days since last game).
- **Hit Rate + Custom Line view**
  - Horizontal bar chart of historical over-rate vs each book's line in the last N
    games, with a tick mark at the book's break-even probability.
  - Custom-line probe: type a hypothetical line and American odds → fitted
    P(over), historical over-rate over the same window, EV per unit for OVER and
    UNDER, and a `+EV/-EV` verdict.
- **Always-visible summary text:** per-book table of `line | odds_over |
  odds_under | P(over) | EV_over | EV_under | hit%`, plus home/away and
  rest-days splits.

The team dropdown filters the player list. Player names not in the DB fall through
to the `nba_api` static lookup, so you can chart any active player even if no game
logs have been ingested for them yet.

#### Operations tab

Every back-end pipeline can be launched and monitored from the desktop UI without
touching the shell. Each section in the sub-notebook builds the right CLI args
behind the scenes, runs the subprocess in its own process group, streams stdout
live to the output panel, and exposes a Stop button (sends SIGTERM):

- **Daily ETL** — db path, season, players, all skip flags, web URLs (file or
  inline), browser auth state file, `--chrome-debug-port`, login URL,
  `--validate-session-before-etl`, `--web-text-force-poll`.
- **Web Text** — Login (headed), Validate Session, Fetch URL, Connect Chrome
  (CDP capture), Extract Chrome Cookies, Sync Active Players Ref. Shared inputs:
  target URL, auth state file, chrome debug port, user data dir.
- **Browser Parser** — urls-file, min parse confidence, max snapshots per URL.
- **Evaluation** — shared start/end date, stat types, windows, distributions;
  buttons for Real-data Benchmark, Distribution Sweep, Line Comparison, Monthly
  Diagnostics.
- **Reverse-Engineering** — source, poll seconds, thresholds, stability runs/
  tolerance, continuous toggle.

Optional flags:
- `--window` (rolling window)
- `--line` (prop line)
- `--odds` (American odds)
- `--opp-def-rating`
- `--spread`
- `--league-avg-def-rating`
- `--defense-sensitivity`
- `--blowout-threshold`
- `--blowout-penalty`
- `--distribution`
- `--defense-severity`
- `--minutes-penalty-severity`
- `--sigma-severity`
- `--correlation-severity`
- `--volatility-severity`
- `--plot` (single mode only)

### 2) Deterministic Smoke Tests (No Live API Required)

```bash
python3 -m unittest nba_model.tests.test_smoke
```

### 3) Baseline Benchmark (Deterministic, Offline)

```bash
python3 -m nba_model.evaluation.run_baseline_benchmark
```

Outputs:
- `nba_model/evaluation/artifacts/baseline_benchmark.csv`
- `nba_model/evaluation/artifacts/baseline_benchmark_summary.md`

### 4) Batch Backtest (Multi-Player / Multi-Window)

Run real-data batch backtests with significance stats (Wilson CI + z-test vs breakeven):

```bash
python3 -m nba_model.evaluation.run_batch_backtest \
  --players "LeBron James" "Stephen Curry" "Nikola Jokic" \
  --windows 5 7 10 15 \
  --stat-types points \
  --distributions normal student_t lognormal \
  --start-date 2024-11-01 \
  --end-date 2025-03-15
```

Notes:
- For points backtests, minutes projection uses spread from game rows when available, then falls back to spread aliases in `betting_lines` (`spread`, `game_spread`, `line_spread`, `vegas_spread`, etc.).

Artifacts are saved under `nba_model/evaluation/artifacts/` with timestamped filenames.

### 4b) DB Inventory Audit

`data/DATABASE_INVENTORY.txt` is an auto-generated snapshot of what's actually in
the SQLite DB. Refresh it any time with:

```bash
.venv/bin/python3 -m nba_model.data.audit_db
# or print to terminal as well:
.venv/bin/python3 -m nba_model.data.audit_db --stdout
# or print only (no file write):
.venv/bin/python3 -m nba_model.data.audit_db --no-write --stdout
```

Sections in the report: tables + row counts, `game_logs` coverage by team
(parsed from `matchup`) and date range, top players by games logged,
`betting_lines` book × stat-type matrix, `betting_line_snapshots` ranges,
`predictions` per stat, `team_defense` seasons, `web_text_snapshots` per URL,
`web_prop_cards` per book, `players` table state, `nba_active_players_ref`
freshness, and a **Known data-quality flags** section that explicitly calls out
gaps (e.g. `players.team` empty for most rows, missing prediction stat types,
orphan `betting_lines` rows).

Same audit is also reachable from the desktop UI: **Operations tab → DB Audit →
Refresh DB inventory** (writes the file *and* streams the full report to the
operations log; an "Open inventory file" button reveals it in your OS file
viewer).

> Current snapshot (re-run after each ETL): the DB has 11k+ `game_logs` rows
> covering all 30 teams via the parsed `matchup` field even though
> `players.team` is only set for 1 row. The Player Charts team-filter dropdown
> reads from `players.team`, which is why almost every team appears empty in
> the UI today — backfilling from `matchup` is on the priority list.

### 4c) Bulk NBA Results Ingestion (`leaguegamefinder` + `playergamelogs`)

Fast bulk ingest of team-game results and league-wide player game logs from
`nba_api`. Populates the `games` table (one row per team per game) and
upserts into `game_logs`. This is the canonical way to refresh historical
data — much faster than the legacy per-player `data_loader.py` path because
it issues a single league-wide request per season.

```bash
# Default: regular season + playoffs for the current 3 seasons
python3 -m nba_model.data.nba_results_ingestion

# Custom seasons + games-only refresh (no player logs)
python3 -m nba_model.data.nba_results_ingestion \
  --seasons 2025-26 2024-25 2023-24 \
  --skip-player-logs
```

Behavior:
- `season_type` (Regular Season / Playoffs / Play-In / Pre Season) is derived
  from the `SEASON_ID` prefix in `nba_api`'s response so playoff games are
  correctly tagged even though the endpoint doesn't return `SEASON_TYPE`
  directly.
- `games` uses `INSERT OR REPLACE` on `(game_id, team_id)` so re-runs cleanly
  refresh scores once the game finalizes.
- `game_logs` uses `INSERT OR IGNORE` keyed on `(player_id, game_id)`.
- Built-in retry with exponential backoff handles `nba_api`'s typical
  `ReadTimeout`/throttle errors without surfacing to the caller.
- Both passes are idempotent — safe to run on a cron / GitHub Actions
  schedule without coordinating against the daily ETL.

The frontends consume this directly:
- Streamlit: `Game Results` and `Player Stats Browse` view modes (free tier).
- Tk: `Game Results` and `Player Stats Browse` notebook tabs.
- Programmatic: `pc.fetch_recent_games(...)` and `pc.fetch_player_recent_results(...)` from `nba_model.visualization.player_charts`.

### 4d) Per-Book Scraper Registry (Cross-Book Consensus)

The `nba_model/scrapers/` package registers 20 books that the chart UIs use
to compute the cross-book "book mean" reference line (player props +
implied team totals). Each book exports a `BookScraper` describing its
domain(s), Playwright wait selectors, session markers (login-wall vs
authenticated content), and optional parser hooks.

**Currently producing parsed data (9 of 20):**

| Book | Type | Mechanism |
|---|---|---|
| PrizePicks, Underdog, Pick6, ParlayPlay | Player props (DFS pickem) | Per-book `prop_preprocess` emits `"<name> <line> <stat> <side>"` segments; generic `_CARD_PATTERNS` in `browser_prop_parser.py` extract them. Pick6's parser expands abbreviated names (`J. Brunson`) via `nba_active_players_ref`. |
| BetMGM, Caesars, DraftKings, Bovada | Team lines (spread/total/moneyline) | Per-book `team_line_extractor` returns one record per (game, market, side) using NBA team names normalized via `scrapers/team_names.py`. |
| Kalshi | Team moneylines | Decimal-odds → American conversion; index page only exposes moneyline at the lobby level. |

**Stub-only (config without parser yet):** fliff, sleeper, dabble, fanduel, betrivers, fanatics, espnbet, hardrockbet, oddsshark, vegasinsider, bettingpros. Their fetch path runs and stores snapshots; add a parser by implementing `prop_preprocess` (DFS shape) or `team_line_extractor` (sportsbook shape) in their `nba_model/scrapers/<book>.py` once an authenticated NBA-page sample is on disk.

**End-to-end flow (single book):**
```bash
# 1. Snapshot via real Chrome on :9222 (after manual login in that Chrome window)
python3 -m nba_model.model.web_text_ingestion \
  --urls "https://app.prizepicks.com/board/nba" \
  --chrome-debug-port 9222

# 2. Parse stored snapshot into web_prop_cards
python3 -m nba_model.model.browser_prop_parser \
  --urls "https://app.prizepicks.com/board/nba"

# 3. Parse stored snapshot into web_team_lines (sportsbooks)
python3 -m nba_model.model.team_line_parser \
  --urls "https://www.ma.betmgm.com/en/sports/basketball-7"
```

Cross-book consensus is read by the chart code via:
- `db.get_consensus_prop_lines(player_name, stat_type, side, since_hours, min_books)`
- `db.get_consensus_team_lines(away_team, home_team, market_type, side, since_hours, min_books)`

Both return `mean_line`, `min_line`, `max_line`, `n_books`, and the contributing-book list. The chart's `book mean X.X` reference line is derived from these — for player charts directly, for team charts via `(game_total - team_spread) / 2` per book.

### 5) Daily ETL Runner (Game Logs + Defense + Odds + Reverse Engineering)

Run a single daily pipeline job with retry + JSON report output:

```bash
python3 -m nba_model.data.daily_etl \
  --players "LeBron James" "Stephen Curry" \
  --season 2024-25 \
  --retries 2 \
  --retry-delay-seconds 1 \
  --report-dir nba_model/data/artifacts
```

Notes:
- By default, it force-refreshes game logs, updates `team_defense`, ingests odds, and then runs `reverse_engineering` in continuous mode.
- Odds step auto-skips when no API key is set (`ODDS_API_KEY`/`THE_ODDS_API_KEY`).
- Odds polling is rate-limited by default to once per 24 hours in CLI runs (`--odds-min-hours-between-polls 24`).
- Use `--force-odds-poll` to bypass that guard for a manual refresh.
- Use `--web-text-urls` (or `--web-text-urls-file`) to fetch raw visible text directly from public web pages.
- Browser parser step now runs after web-text ingestion and attempts to extract structured prop cards (`player`, `stat`, `line`, `side`, `book`, `timestamp`) into `web_prop_cards`.
- Parsed names are auto-classified with `nba_active_players_ref` as `active_nba` vs `non_nba`, and each row keeps `parse_confidence` + raw card text for auditability.
- Use `--skip-browser-parser` to disable parsing, and tune extraction with `--browser-parser-min-parse-confidence`, `--browser-parser-max-snapshots-per-url`, and `--browser-parser-max-total-snapshots`.
- For authenticated web sessions, pass `--browser-auth-state-file` and/or `--browser-user-data-dir` so web-text ingestion runs in browser context with your saved login state.
- Web-text polling is also guarded per URL by default (`--web-text-min-hours-between-polls 24`).
- Not every site will parse equally well with raw text extraction (dynamic JS pages, bot protection, and custom HTML layouts can limit coverage).
- Reverse-engineering auto-skips when the odds step is skipped/failed; use `--skip-reverse-engineering` to disable it explicitly.
- Use `--reverse-engineering-single-pass` to run one pass instead of continuous polling.
- For automation, set guardrails with `--reverse-engineering-max-runs` or `--reverse-engineering-max-wait-minutes`.
- Use `--all-db-players` to refresh every player currently in the local `players` table.
- Use `--min-players N` to auto-expand from active NBA players if the selected pool is smaller than `N`.
- Use `--skip-zero-game-players` to skip non-explicit players that currently have no local game logs.
- Use `--strict` for cron jobs to return non-zero exit code on failed/partial runs.

Example ETL run with the same continuous thresholds:

```bash
python3 -m nba_model.data.daily_etl \
  --skip-game-logs \
  --skip-team-defense \
  --reverse-engineering-source both \
  --reverse-engineering-poll-seconds 300 \
  --reverse-engineering-min-inferred-rows 25 \
  --reverse-engineering-min-book-stat-groups 2 \
  --reverse-engineering-min-player-segment-groups 5 \
  --reverse-engineering-require-stability-runs 2 \
  --reverse-engineering-stability-tolerance 0.10
```

Example to grow game-log refresh beyond the default 5-player seed:

```bash
python3 -m nba_model.data.daily_etl \
  --all-db-players \
  --min-players 150 \
  --skip-zero-game-players \
  --game-log-games 200 \
  --skip-odds \
  --skip-reverse-engineering
```

Example odds polling with once-per-day guard while the program is running:

```bash
python3 -m nba_model.data.daily_etl \
  --skip-game-logs \
  --skip-team-defense \
  --odds-min-hours-between-polls 24
```

Example direct website text ingestion (no API key) while ETL is running:

```bash
python3 -m nba_model.data.daily_etl \
  --skip-game-logs \
  --skip-team-defense \
  --skip-odds \
  --skip-reverse-engineering \
  --web-text-urls "https://example.com/sportsbook/nba/props" \
  --web-text-min-hours-between-polls 24
```

Authenticated browser-session ingestion (manual runs only):

```bash
.venv/bin/playwright install chromium

.venv/bin/python -m nba_model.data.daily_etl \
  --skip-game-logs \
  --skip-team-defense \
  --skip-odds \
  --skip-reverse-engineering \
  --web-text-urls-file data/config/web_text_urls.txt \
  --browser-auth-state-file data/config/auth/underdog_state.json \
  --browser-user-data-dir data/config/auth/underdog_profile
```

Notes:
- Browser-auth mode expects Playwright installed in the Python environment you use to run the command.
- Recommended runtime is project `.venv` (`.venv/bin/python ...`) so browser dependencies stay isolated.
- This project auto-prefers local `.playwright-browsers/` binaries when available.

#### PrizePicks — CDP Workflow (Required for Multi-Layer Bot Protection)

PrizePicks runs Cloudflare + PerimeterX + DataDome simultaneously. All three tie session cookies to the browser's TLS fingerprint, so Playwright's Chromium is blocked at the network layer regardless of JS-side stealth patches. The only reliable approach is to route everything through a **real running Chrome** via Chrome DevTools Protocol (CDP).

**One-time setup — install dependencies:**

```bash
.venv/bin/python3 -m pip install playwright-stealth pycookiecheat
```

**Step 1: Launch Chrome with remote debugging**

```bash
open -na "Google Chrome" --args \
    --remote-debugging-port=9222 \
    --user-data-dir=/tmp/pp-chrome-profile
```

**Step 2: Log in to PrizePicks** in that Chrome window. Complete any Cloudflare/location verification until you see the player props board.

**Step 3: Extract the full session (cookies + localStorage JWT)**

```bash
.venv/bin/python3 -m nba_model.model.web_text_ingestion \
  --connect-chrome https://app.prizepicks.com/board/nba \
  --browser-auth-state-file data/config/auth/prizepicks_state.json \
  --chrome-debug-port 9222
```

**Step 4: Validate the session**

```bash
.venv/bin/python3 -m nba_model.model.web_text_ingestion \
  --validate-session https://app.prizepicks.com/board/nba \
  --browser-auth-state-file data/config/auth/prizepicks_state.json \
  --chrome-debug-port 9222
```

**Step 5: Fetch/scrape through real Chrome** (keep the Chrome window open while running)

```bash
.venv/bin/python3 -m nba_model.model.web_text_ingestion \
  --urls "https://app.prizepicks.com/board/nba" \
  --browser-auth-state-file data/config/auth/prizepicks_state.json \
  --chrome-debug-port 9222 \
  --force-poll
```

Notes:
- Chrome must remain open with `--remote-debugging-port=9222` for the duration of any scrape run.
- `--connect-chrome` captures both cookies AND `localStorage` (which stores PrizePicks JWT auth tokens) — unlike `--extract-chrome-session` which only captures cookies.
- The `--chrome-debug-port` flag is also accepted by `--validate-session` to route validation through the same real Chrome.
- If Chrome closes unexpectedly, re-run Steps 1–2 and then Step 3 to refresh the session state.
- For UnderDog, Playwright + `playwright-stealth` + `--login` is usually sufficient (lighter bot protection).

Standalone direct web-text CLI:

```bash
python3 -m nba_model.model.web_text_ingestion \
  --urls "https://example.com/sportsbook/nba/props" \
  --min-hours-between-polls 24 \
  --browser-auth-state-file data/config/auth/underdog_state.json
```

Standalone browser-parser CLI (reads from `web_text_snapshots` and writes `web_prop_cards`):

```bash
python3 -m nba_model.model.browser_prop_parser \
  --urls-file data/config/web_text_urls.txt \
  --min-parse-confidence 0.50 \
  --max-snapshots-per-url 2
```

Sync active NBA players reference into DB + local file (used for filtering/classifying non-NBA names):

```bash
python3 -m nba_model.model.web_text_ingestion \
  --sync-active-players-ref \
  --active-players-output-file data/config/active_nba_players.txt
```

#### 5b) Automated Daily ETL (GitHub Actions)

This repository includes a scheduled GitHub Actions workflow (`.github/workflows/daily_etl.yml`) that runs the daily ETL once per day:

- **Schedule**: 08:00 UTC (configurable via the `cron` line in the workflow).
- **Command in CI**: `python -m nba_model.data.daily_etl --strict --skip-game-logs --skip-team-defense --skip-reverse-engineering` (stats.nba.com times out from GitHub’s network; game logs and team defense are for local or self-hosted runs).
- **Full ETL locally**: `python -m nba_model.data.daily_etl --strict`
- **Required secret**: `ODDS_API_KEY` (configure in your GitHub repository Settings → Secrets and variables → Actions). Without it, the odds step is skipped and the run still succeeds.

Daily ETL runs emit:

- A structured JSON report under `nba_model/data/artifacts/`.
- A timestamped log file under `nba_model/data/logs/` (e.g. `daily_etl_YYYYMMDD_HHMMSS.log`).

### 6) Real-Data Multi-Player Benchmark (Player/Window CIs)

```bash
python3 -m nba_model.evaluation.run_real_data_benchmark \
  --players "LeBron James" "Stephen Curry" "Nikola Jokic" "Luka Doncic" \
           "Jayson Tatum" "Giannis Antetokounmpo" "Shai Gilgeous-Alexander" \
           "Kevin Durant" "Anthony Edwards" "Damian Lillard" \
  --windows 5 7 10 15 \
  --stat-types points \
  --distributions normal \
  --start-date 2024-11-01 \
  --end-date 2025-03-15
```

Exports include per-player/window confidence interval artifacts.

### 7) Distribution Sweep Benchmark (ROI + Calibration + Significance)

```bash
python3 -m nba_model.evaluation.run_distribution_sweep \
  --windows 5 7 10 15 \
  --stat-types points assists rebounds pra \
  --distributions normal student_t binomial poisson exponential uniform lognormal power_law \
  --start-date 2024-11-01 \
  --end-date 2025-03-15
```

Exports include distribution/stat summaries for ROI, Brier score, and significance.

### 8) Line Comparison + Monthly Diagnostics

```bash
# Cross-book and model-vs-book value comparison
python3 -m nba_model.evaluation.line_comparison \
  --start-date 2024-11-01 \
  --end-date 2025-03-15 \
  --stat-types points assists rebounds pra \
  --edge-threshold 0.02

# Monthly drift, drawdown, and calibration diagnostics
python3 -m nba_model.evaluation.monthly_diagnostics \
  --start-date 2024-11-01 \
  --end-date 2025-03-15 \
  --stat-types points assists rebounds pra
```

#### Comparing to “the market” vs other forecasting models

- **Model-vs-market**: The pipeline compares **this model** to **sportsbook lines** (the “market”). `line_comparison` produces model-vs-book and book-vs-book outputs; backtests can use market lines via `--use-market-lines` and `--market-book`. So you are checking your forecasts against whatever books you ingest (DraftKings, FanDuel, etc.).
- **Other forecasting models**: There is no built-in comparison to named external models (e.g. ESPN, FiveThirtyEight, or other projection systems). To do that you would need to ingest those models’ predictions into the DB (or a separate table) and add a small comparison script that computes differences vs your model.

#### Using smaller or alternate books (e.g. Fliff, Kalshi)

Odds come from **The Odds API**, which supports many bookmakers (including Fliff, Kalshi, and other smaller or regional books). The code does not hardcode book names; it stores whatever `book` the API returns.

- **Ingest specific books**: In daily ETL or odds ingestion, pass the bookmaker keys the API expects (e.g. `fliff`, `kalshi`):
  ```bash
  python3 -m nba_model.data.daily_etl --strict --skip-game-logs --skip-team-defense \
    --bookmakers fliff kalshi
  ```
  Or with the standalone odds module:
  ```bash
  python3 -m nba_model.model.odds_ingestion \
    --db-path data/database/nba_data.db \
    --bookmakers fliff kalshi
  ```
- **Backtest / comparison using one book**: Use `--market-book` to restrict to that book’s lines:
  ```bash
  python3 -m nba_model.evaluation.run_batch_backtest \
    --use-market-lines --market-book fliff \
    --players "LeBron James" ... --stat-types points ...
  ```
  `line_comparison` uses all books present in `betting_lines`; filter by date/stat and optionally by book in downstream analysis.
- **Caveats**: (1) The Odds API’s list of bookmakers and keys can change; confirm keys (e.g. `fliff`, `kalshi`) in the API docs. (2) Kalshi is prediction-exchange style; their NBA coverage may be event-based rather than traditional player over/under lines, so availability may differ from FanDuel/DraftKings.

### 9) Prop Board Generator (All Player Props for a Game)

Generate a prop-style board for all available player props (both teams) for a given game/date, using rolling-history projections and the per-stat default distributions:

```bash
python3 -m nba_model.model.prop_board \
  --game-date 2025-03-01 \
  --home-team LAL \
  --away-team BOS \
  --stat-types points assists rebounds pra \
  --db-path data/database/nba_data.db
```

Outputs a CSV (to stdout by default) with one row per player/stat/line, including:

- projected `mu` and `sigma`
- chosen `distribution` (from `DEFAULT_DISTRIBUTION_BY_STAT`)
- model `prob_over`
- implied probabilities from book odds
- EV for both over and under.

### 10) Market Data Quality + CLV Proxy

Run basic data-quality checks on `betting_lines` and `betting_line_snapshots`:

```bash
python3 -m nba_model.evaluation.market_data_quality \
  --db-path data/database/nba_data.db \
  --max-snapshot-age-hours 6
```

This produces a JSON summary under `nba_model/evaluation/artifacts/` with counts of missing fields, duplicate snapshots, and a simple snapshot freshness flag.

Compute a simple CLV-style proxy from historical line snapshots:

```bash
python3 -m nba_model.evaluation.clv_proxy \
  --db-path data/database/nba_data.db \
  --start-date 2024-11-01 \
  --end-date 2025-03-15 \
  --stat-types points assists rebounds pra
```

This exports a CSV with open/close line/odds and implied-prob changes per `(player, game_date, stat_type, book, market_key)` that can be joined with model signals offline.

## Production defaults (from benchmarks)

Default distribution per stat type is defined in `nba_model/model/simulation.py` as `DEFAULT_DISTRIBUTION_BY_STAT`. These should be set after running the real-data benchmark and distribution sweep and reviewing artifacts:

1. **Real-data benchmark** (player/window CIs):
   ```bash
   python3 -m nba_model.evaluation.run_real_data_benchmark \
     --windows 5 7 10 15 \
     --stat-types points \
     --distributions normal \
     --start-date 2024-11-01 \
     --end-date 2025-03-15
   ```

2. **Distribution sweep** (ROI/calibration/significance by distribution and stat):
   ```bash
   python3 -m nba_model.evaluation.run_distribution_sweep \
     --windows 5 7 10 15 \
     --stat-types points assists rebounds pra \
     --start-date 2024-11-01 \
     --end-date 2025-03-15
   ```

3. Review artifacts in `nba_model/evaluation/artifacts/` (e.g. `distribution_sweep_distribution_summary_*.csv`, `distribution_sweep_summary_*.md`).
4. Pick the best distribution per stat (e.g. by `avg_roi` or significance) and update `DEFAULT_DISTRIBUTION_BY_STAT` in `nba_model/model/simulation.py`. Use `get_default_distribution(stat_type)` in code when a single default is needed for a given stat.

## Baseline Benchmark Results

Baseline benchmark now runs with the same template across:
- `points`
- `assists`
- `rebounds`
- `pra` (points + rebounds + assists)

Latest results are generated into:
- `nba_model/evaluation/artifacts/baseline_benchmark.csv`
- `nba_model/evaluation/artifacts/baseline_benchmark_summary.md`

## Known Limitations

- Benchmark metrics are synthetic and validate pipeline behavior, not live betting edge.
- Live NBA API calls require network access when cache/database does not already contain player games.
- Cross-book/model-vs-book and monthly diagnostics depend on having populated `betting_lines` and `predictions` data.
- Historical open/close line snapshots are not yet stored, so line-move analysis is still limited.

## Operational Runbook (ETL & Odds Ingestion)

- **Check scheduler health**:
  - Verify the `Daily NBA ETL` GitHub Actions workflow is running on schedule and succeeding.
  - On failures, inspect the workflow logs and the **"Show ETL failure summary"** step (which prints per-step status from the last report).
- **When daily ETL fails in GitHub Actions**:
  - Download the **etl-artifacts-** artifact from the failed run; it contains the latest ETL report JSON and logs.
  - In CI, **game_logs** and **team_defense** are skipped by design (stats.nba.com times out from GitHub runners). If the run still fails, the cause is usually the **odds** step (e.g. invalid or missing `ODDS_API_KEY`). Fix odds config or run ETL locally for full data.
  - For full ETL (game logs + team defense), run locally: `python -m nba_model.data.daily_etl --strict`.
- **Inspect ETL artifacts**:
  - JSON ETL reports are stored under `nba_model/data/artifacts/` and include per-step statuses and errors.
  - Detailed ETL logs are stored under `nba_model/data/logs/`.
- **Handling Odds API issues**:
  - Confirm `ODDS_API_KEY` is present and valid (locally via environment variable; in CI via GitHub secret).
  - Review ETL report `steps["odds"]` payload for API error messages and retry behavior.
  - If The Odds API is unavailable, rerun ETL later; game logs and team defense can still be refreshed with `--skip-odds`.
- **Recovering from partial/failed runs**:
  - Use `--strict` in automated contexts so failures surface as non-zero exit codes.
  - For manual recovery, rerun ETL for a narrower player set or with `--use-cache-for-game-logs` to reduce load.

## Next Expansion Targets

- Add real multi-player historical benchmarks from cached/live NBA data.
- Expand contextual modeling beyond current rest/travel/injury-proxy heuristics.
- Extend line comparison to include richer execution constraints and portfolio-level filters.
