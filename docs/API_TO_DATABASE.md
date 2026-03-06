# API to SQLite: Data Flow and Cleaning

Short reference for how data moves from external APIs into the SQLite DB and how we clean it.

---

## Overview

| Source | API | Tables | Entry point |
|--------|-----|--------|-------------|
| Game logs | **NBA API** (stats.nba.com) | `game_logs`, `players` | `data_loader.py` → `db_manager.insert_game_logs` |
| Team defense | **NBA API** | `team_defense` | `team_defense_ingestion.py` → `db_manager.insert_team_defense_records` |
| Betting lines & snapshots | **The Odds API** | `betting_lines`, `betting_line_snapshots` | `odds_ingestion.py` → `db_manager.insert_betting_lines_records` / `insert_betting_line_snapshots` |

All persistence goes through **`DatabaseManager`** in `nba_model/data/database/db_manager.py`. The schema is in `nba_model/data/database/schema.sql` and is applied automatically when the DB is first opened.

---

## 1. Game logs (NBA API → SQLite)

**Flow**

1. **DataLoader** (`data_loader.py`) is called (e.g. by `run_model` or daily ETL).
2. It tries, in order: **DB** (`db.get_player_games`) → **file cache** (JSON under `data/raw`) → **NBA API** (`playergamelogs.PlayerGameLogs`).
3. If data comes from the API, it is **cleaned** then written to the file cache and to the DB via `db.insert_game_logs(df)`.

**Cleaning (`_clean_game_logs`)**

- **Column mapping**: API names (e.g. `PTS`, `MIN`, `GAME_DATE`) are renamed to DB names (`points`, `minutes`, `game_date`). Only existing columns are renamed.
- **player_id**: Set from the resolved player (so every row has a consistent `player_id`).
- **home_away**: Derived from `matchup`: `'home'` if `'vs.'` in matchup, else `'away'`.
- **minutes**: If the column is string (e.g. `"32:15"`), converted to decimal minutes via `_convert_minutes` (e.g. `32 + 15/60`). Invalid/empty → `0.0`.
- **Output**: Only columns that exist in the mapping (and `home_away`) are kept, so the DataFrame matches the expected game_logs shape.

**DB insert (`insert_game_logs`)**

- Rows are **deduped** on `(player_id, game_id)` before insert.
- Only columns that exist in the `game_logs` table (and are not `game_log_id` or `created_at`) are used.
- `NaN` is converted to `None` for SQLite.
- **INSERT OR IGNORE**: duplicate `(player_id, game_id)` rows are skipped; no error.

---

## 2. Team defense (NBA API → SQLite)

**Flow**

1. **team_defense_ingestion** calls the NBA API (`LeagueDashTeamStats` with defense measure).
2. **fetch_team_defense_df** builds a DataFrame with standardized columns.
3. **populate_team_defense** passes records to `db.insert_team_defense_records(records)`.

**Cleaning**

- **Team abbreviation**: Resolved from API columns in order: `TEAM_ABBREVIATION` / `TEAM_ABBREV` (used as-is, uppercased), else `TEAM_ID` or `TEAM_NAME` mapped via `nba_api.stats.static.teams` to official abbreviation.
- **Numeric fields**: `def_rating`, `opp_ppg`, `pace` are coerced with `pd.to_numeric(..., errors="coerce")`. Optional columns (e.g. pace) can be missing in the API.
- **Rows**: Dropped where `team_abbrev` or `def_rating` is null. Season is set from the request.

**DB insert (`insert_team_defense_records`)**

- **Upsert** on `team_abbrev`: `ON CONFLICT(team_abbrev) DO UPDATE` so each team has one row per abbreviation; later runs refresh def_rating, opp_ppg, pace, last_updated.
- Records without `team_abbrev` are skipped.

---

## 3. Betting lines & snapshots (The Odds API → SQLite)

**Flow**

1. **odds_ingestion** calls The Odds API (event-level or sport-level) and gets events with `bookmakers` and `markets`.
2. **Parse** each event into normalized records: player name → player_id, game_date from `commence_time`, book, market_key, stat_type (from `PLAYER_PROP_MARKETS`), line, over/under odds. Over/under outcomes are grouped per (player, line) and merged into one row per (player, game_date, book, stat_type, line_value).
3. **validate_betting_line_records** runs on the list of records; invalid rows are dropped and counted.
4. **Dedupe**: `_dedupe_records` removes exact duplicates (same player_id, game_date, book, stat_type, line_value, over_odds, under_odds).
5. Valid records are written to **betting_lines** and optionally to **betting_line_snapshots** (with `snapshot_ts_utc`, `event_id`, `market_key`, optional `raw_payload`).

**Cleaning / validation**

- **Types**: `player_id` → int (invalid/missing → row invalid). `game_date` → normalized to `YYYY-MM-DD` (invalid → row invalid). `line_value` → float. `over_odds` / `under_odds` → int or None.
- **Required**: `player_id` > 0, valid `game_date`, non-empty `book`, `stat_type` in `{points, assists, rebounds, pra}`, non-null `line_value`. If odds are present they must coerce to int.
- **stat_type**: Lowercased and must be in `VALID_STAT_TYPES` (from `PLAYER_PROP_MARKETS`).
- **Player resolution**: Player name from the API is resolved to `player_id` via `nba_api.stats.static.players`; unresolved names are skipped (and reported) so only rows with a valid `player_id` are inserted.

**DB insert**

- **betting_lines**: Insert only if an identical row does not already exist (same player_id, game_date, book, stat_type, line_value, over_odds, under_odds). So duplicates are skipped, not updated.
- **betting_line_snapshots**: Each call appends rows (no dedupe by key); minimal validation (required fields non-null). Used for open/close and CLV-style analysis.

---

## 4. Summary table

| Data | Where cleaning happens | Dedupe / conflict |
|------|-------------------------|-------------------|
| Game logs | `DataLoader._clean_game_logs` (column map, minutes, home_away) | `INSERT OR IGNORE` on (player_id, game_id); DataFrame dedupe before insert |
| Team defense | `team_defense_ingestion.fetch_team_defense_df` (team abbrev, numerics, dropna) | Upsert on `team_abbrev` |
| Betting lines | `odds_ingestion.validate_betting_line_records` + in-memory dedupe | Insert only when no existing row with same (player_id, game_date, book, stat_type, line_value, odds) |
| Snapshots | Minimal (required fields); optional raw_payload | No dedupe; append-only |

---

## 5. Running the pipeline

- **Game logs + team defense + odds together**: `python -m nba_model.data.daily_etl` (see README for flags like `--skip-game-logs`, `--bookmakers`).
- **Game logs only**: Use `DataLoader` (e.g. via `run_model`) or call the loader then `db.insert_game_logs`.
- **Team defense only**: `python -m nba_model.data.team_defense_ingestion --season 2024-25`.
- **Odds only**: `python -m nba_model.model.odds_ingestion --db-path data/database/nba_data.db` (set `ODDS_API_KEY`).

All paths use the same SQLite DB and schema; cleaning and conflict handling are as above.
