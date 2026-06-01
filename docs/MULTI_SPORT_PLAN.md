# Multi-sport rollout plan

Status as of 2026-05-16. This doc captures the architecture for expanding
beyond NBA, the per-sport priority order, and the open questions to resolve
before each sport goes live.

## Architecture

The seam is **[`sports/`](../sports/)** at the project root. One module per
sport (with a nested package for soccer's sub-leagues). Each module exports
a frozen `Sport` dataclass capturing everything sport-specific:

```python
Sport(
    key="nba",                  # url slug, env var, db tag
    display_name="NBA",
    status="live" | "stub" | "planned",
    stat_types=(...),           # canonical names for player props
    stat_line_ranges={...},     # data-poisoning bounds per stat
    team_codes=(...),
    season_format="...",
    primary_books=(...),
    notes=(...),
    sub_leagues=(...,),         # only used by soccer
)
```

Anything NOT in `Sport` (distribution fitting, Monte Carlo, EV math, the
Stripe / OIDC / auth layer, the Streamlit shell, the FastAPI webhook,
input validation, etc.) is genuinely sport-agnostic and reads `Sport`
values when it needs to.

**Where the sport flows in:**

| Layer | What it reads from `Sport` |
|---|---|
| `nba_model/web/input_validation.py` | `stat_line_ranges` (data-poisoning bounds), `stat_types` (allowlist) |
| `nba_model/visualization/player_charts.py` | `stat_types`, column-mapping per stat |
| `nba_model/scrapers/<book>/<sport>.py` | per-book parsers keyed by sport |
| `data/database/schema.sql` | needs `sport` column on `game_logs`, `betting_lines`, `players`, `web_prop_cards` (or per-sport tables — see "schema decision" below) |
| `nba_model/web/app.py` sidebar | sport selector + roadmap card for stubs |

## Rollout order

The user's priority list:

| # | Sport | Status | League count | Realistic timeline |
|---|---|---|---|---|
| 1 | **NBA** | live | 1 | done |
| 2 | **NFL** | stub | 1 | ~2 weeks (ingest is easy, prop depth is uneven) |
| 3 | **MLB** | stub | 1 | ~3 weeks (batter/pitcher split adds complexity) |
| 4 | **NHL** | stub | 1 | ~2 weeks (closest shape to NBA) |
| 5a | **EPL** (soccer) | stub | 1 | ~3 weeks (first soccer league sets the pattern) |
| 5b | **La Liga** | stub | 1 | ~1 week after EPL (reuses pattern) |
| 5c | **Serie A** | stub | 1 | ~1 week after EPL |
| 5d | **Bundesliga** | stub | 1 | ~1 week after EPL |
| 5e | **Ligue 1** | stub | 1 | ~1 week after EPL |
| 5f | **UCL** | stub | 1 | ~2 weeks (two-leg ties complicate matters) |
| Future | Copa Libertadores | planned | 1 | calendar-year + Brazilian books |
| Future | Copa Sudamericana | planned | 1 | inherits Libertadores work |

## Cross-cutting decisions to make BEFORE the first non-NBA sport ships

### 1. Schema: `sport` column or per-sport tables?

**Recommendation: add `sport TEXT NOT NULL DEFAULT 'nba'`** to `game_logs`,
`betting_lines`, `betting_line_snapshots`, `players`, `web_prop_cards`,
`predictions`. Composite primary keys gain `sport`. Most queries grow a
`WHERE sport = ?` clause; the existing chart helpers can take a sport
parameter and default to `nba` to stay backward-compatible.

Why not per-sport tables? MLB has the strongest case (batter vs pitcher
diverge structurally), but even there 80% of the columns overlap. The
maintenance overhead of N parallel schemas isn't worth it for a project
this size. If MLB pitcher stats end up too different, split *only*
`mlb_pitcher_logs` as a dedicated table — keep batters in `game_logs`.

### 2. Stat-type namespacing

Stat names will collide across sports — `assists` exists in NBA, NHL, and
soccer. Two options:

- **Option A:** prefix every stat with the sport (`nba_assists`,
  `nhl_assists`, `soccer_assists`). Ugly but unambiguous.
- **Option B:** keep short names + scope by `(sport, stat_type)` in every
  index. The DB layer already uses `(player_id, stat_type)` everywhere; we
  just upgrade to `(sport, player_id, stat_type)`.

**Recommendation: Option B.** Stat names stay readable in the UI, and the
existing parameterized queries are easy to extend.

### 3. Two-way vs three-way markets

NBA / NFL / MLB / NHL are two-way (over/under, win/loss). Soccer is
three-way (home win / draw / away win). `nba_model/web/parlay_compare.py`
and the EV math in `auth.expected_value` assume binary outcomes.

**Action:** add `outcome_dim: Literal[2, 3]` to `Sport` and branch the EV
math + parlay-leg math before soccer goes live. Two-way sports keep their
existing fast path.

### 4. Player-prop scrapers per sport

Per-book scraper modules currently live at `nba_model/scrapers/<book>.py`
and assume NBA boards. For multi-sport, we move to
`nba_model/scrapers/<book>/<sport>.py` and update `get_scraper_for_url`
to resolve `(book, sport)` instead of just `book`. This is a moderate
refactor but only touches the scrapers package.

### 5. Player-name resolution across sports

`nba_active_players_ref` has 530 rows. NFL has ~1700 active players, MLB
has ~1200, NHL has ~750, soccer has tens of thousands across all leagues.
The player-id space is sport-specific (NBA IDs from `nba_api`, NFL IDs
from `nfl_data_py`, etc.).

**Action:** the existing `nba_model/scrapers/player_names.py` resolver
needs a `sport` parameter so it doesn't conflate names across sports
(both NBA and NFL have a "Lonnie Walker"; both MLB and NHL have a "Trevor
Story"-shaped namespace pollution).

### 6. Free-tier preview list per sport

Currently `auth.PREVIEW_PLAYERS = ("Nikola Jokic", "LeBron James")` and
`auth.PREVIEW_TEAMS = ("LAL", "DEN")`. Each sport will want its own pair.

**Action:** make these dicts keyed by sport — `PREVIEW_PLAYERS["nfl"] =
("Patrick Mahomes", "Christian McCaffrey")` etc.

## Per-sport open questions

### NFL

- **Ingest:** [`nfl_data_py`](https://github.com/cooperdff/nfl_data_py)
  wraps `nflverse-data` Parquet drops. Battle-tested, MIT licensed.
- **Cadence:** weekly games means rolling-mean windows behave differently
  than nightly NBA. 5-game rolling = 5 weeks of context. Defaults need
  per-sport tuning.
- **Position split:** QB / RB / WR / TE / K project entirely differently.
  We can't fit one model across positions; need a position field on
  `players` and per-position stat tables (or filter by position).
- **Bye weeks:** the game-by-game series isn't continuous. Rolling means
  + recency-decayed averages need to skip byes, not penalize them.
- **Yes/no markets:** `anytime_touchdown_scorer` is Bernoulli, not
  normal. The distribution-fitting code already supports binomial; just
  need to wire it as the default for these markets.
- **Books:** FanDuel + DraftKings have the deepest prop coverage; BetMGM
  / Caesars are decent; PrizePicks / Underdog are heavily promoted.

### MLB

- **Ingest:** [`pybaseball`](https://github.com/jldbc/pybaseball) wraps
  FanGraphs + Statcast + Baseball Reference. Comprehensive but heavier
  than `nfl_data_py`.
- **Batter vs pitcher:** entirely separate populations. Two stat-type
  lists, two model paths. Probably worth a dedicated
  `mlb_pitcher_logs` table to keep the `game_logs` shape sane.
- **Park factors:** Coors Field is +12% runs, Petco is -8%. Need a
  park-factor adjustment analogous to NBA's defense rating. Per-park
  multipliers are stable enough to hardcode initially.
- **Handedness split:** L/R pitcher matters a lot for batter projections.
  Add a `pitcher_handedness` feature to projections.
- **Lineup uncertainty:** unlike NBA where Jokic is starting unless
  injured, MLB lineups aren't confirmed until ~3 hours pre-game. Plays
  havoc with the "I bookmarked this prop yesterday" flow.
- **Books:** same lineup as NFL.

### NHL

- **Ingest:** community wrappers around `https://api.nhle.com/`. None as
  polished as `nfl_data_py`; `nhl-api-py` is the closest.
- **Goalies vs skaters:** like MLB pitcher/batter but the gap is less
  extreme — most goalie markets are saves / goals-against / wins. Keep
  skaters in `game_logs`, optionally split goalies later.
- **Shootouts:** do they count toward stats? Books differ across markets;
  codify the rule once in `nhl.py` and reference it.
- **Plus/minus:** notoriously noisy stat. Low priority for the model.
- **Books:** same lineup; NHL has shallower prop coverage than NBA.

### Soccer (parent)

- **Three-way markets:** see cross-cutting decision #3.
- **Cup competitions:** UCL knockouts are two-legged. The schema needs
  to model "tie" as a higher-level entity than "match" so aggregate
  scoring is correct.
- **Promotion / relegation:** team membership rotates yearly. We need a
  `(season, league, team_code)` membership table; the `team_codes` tuple
  in each league module is a current-season snapshot, not an enum.
- **Player turnover:** transfers cross league boundaries (Mbappé played
  for PSG, then Real Madrid). Player identity is a global graph, not
  league-scoped. Existing player-id resolver needs upgrading.
- **Prop depth in US books:** thin. bet365 + Pinnacle have the depth;
  neither is US-friendly. We may need to surface "limited US book
  coverage" as a UX caveat for soccer launches.

### Soccer (sub-leagues)

Each of EPL / La Liga / Serie A / Bundesliga / Ligue 1 shares 90% of the
work — the first one done sets the pattern, the rest are reskins.

- **EPL** ships first. Largest US betting market for any non-US sport.
- **La Liga**, **Serie A**, **Bundesliga**, **Ligue 1** are largely a
  matter of copying the EPL scraper modules + swapping team-code tables.
- **UCL** is the harder soccer entry because of two-legged ties and
  player overlap with domestic league (Mbappé has La Liga props on the
  same gameweek as UCL props).

## Future: South American competitions

Two competitions planned for a later phase:

### Copa Libertadores
- **Season format:** calendar-year (Feb – Nov), unlike European leagues.
- **Structure:** group stage + two-legged knockouts (since 2017 single-
  match final reform: final is one neutral-venue match).
- **Books with depth:** bet365, Pinnacle (both not US-friendly),
  regional Brazilian books like Betano, Stake, Sportingbet, Bet7K. Player
  prop depth is shallow even on these.
- **Team membership:** rotates yearly, ~47 clubs across CONMEBOL
  federations. Same `(season, league, team_code)` membership pattern as
  Euro leagues.

### Copa Sudamericana
- **Season format:** also calendar-year, runs alongside Libertadores
  (Sudamericana is the "Europa League equivalent").
- **Same book + team-membership caveats as Libertadores.** Implementation
  inherits everything from Libertadores once it ships.

### Stubs already wired

The `sports/soccer/__init__.py` module has commented-out `Sport` entries
for both Libertadores and Sudamericana so the layout + open-questions
notes are recorded. When the time comes, uncomment them + add them to
`SOCCER_SUB_LEAGUES`.

## What's already wired in this commit

- `sports/` package at the project root with `Sport` dataclass + nba /
  nfl / mlb / nhl / soccer modules. Soccer nests EPL, La Liga, Serie A,
  Bundesliga, Ligue 1, UCL.
- Streamlit sport selector at the top of the sidebar. NBA is the live
  option; the other four render a "coming soon" roadmap card in the main
  pane showing the planned stat types, sub-leagues, and open questions.
- URL state: `?sport=nba` etc. preserves the selection across reloads.
- 233/233 tests still pass.

## What's NOT yet wired

- Schema migrations for the `sport` column.
- Per-book scraper resolution by `(book, sport)`.
- Outcome-dimension branching for three-way soccer markets.
- Per-sport `nba_api`-equivalent ingestion.
- Free-tier preview-player dict per sport.
