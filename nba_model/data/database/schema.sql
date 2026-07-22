-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    team TEXT,
    position TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Game logs table (core historical data)
CREATE TABLE IF NOT EXISTS game_logs (
    game_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    game_date DATE NOT NULL,
    season TEXT NOT NULL,
    matchup TEXT,
    home_away TEXT,
    result TEXT,  -- 'W' or 'L'
    minutes REAL,
    points INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg_pct REAL,
    fg3m INTEGER,
    fg3a INTEGER,
    fg3_pct REAL,
    ftm INTEGER,
    fta INTEGER,
    ft_pct REAL,
    oreb INTEGER,
    dreb INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    plus_minus INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    UNIQUE(player_id, game_id)
);

-- Team-level NBA game results (one row per team per game = 2 rows per game).
-- Sourced from nba_api leaguegamefinder; gives us schedule + scores +
-- W/L for every team-game, which the frontend uses for the Game Results tab.
CREATE TABLE IF NOT EXISTS games (
    game_id        TEXT NOT NULL,             -- nba_api GAME_ID (10-char)
    season         TEXT NOT NULL,             -- e.g. "2024-25"
    season_type    TEXT NOT NULL,             -- "Regular Season" / "Playoffs" / "Pre Season"
    game_date      DATE NOT NULL,
    team_id        INTEGER NOT NULL,
    team_abbrev    TEXT NOT NULL,             -- e.g. "NYK"
    team_name      TEXT,                      -- e.g. "New York Knicks"
    matchup        TEXT,                      -- e.g. "NYK vs. PHI" / "NYK @ PHI"
    home_away      TEXT,                      -- 'home' / 'away'
    opponent_abbrev TEXT,                     -- parsed from matchup
    result         TEXT,                      -- 'W' / 'L' / NULL (future)
    pts            INTEGER,
    opp_pts        INTEGER,
    plus_minus     INTEGER,
    fg_pct         REAL,
    fg3_pct        REAL,
    ft_pct         REAL,
    rebounds       INTEGER,
    assists        INTEGER,
    steals         INTEGER,
    blocks         INTEGER,
    turnovers      INTEGER,
    last_updated   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, team_id)
);

-- Team defensive stats (for adjustments)
CREATE TABLE IF NOT EXISTS team_defense (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_abbrev TEXT UNIQUE NOT NULL,
    season TEXT NOT NULL,
    def_rating REAL,
    opp_ppg REAL,
    pace REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Betting lines (for tracking and backtesting)
CREATE TABLE IF NOT EXISTS betting_lines (
    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    book TEXT NOT NULL,  -- 'FanDuel', 'DraftKings', etc.
    stat_type TEXT NOT NULL,  -- 'points', 'assists', 'rebounds'
    line_value REAL NOT NULL,
    over_odds INTEGER,
    under_odds INTEGER,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT,  -- provenance: NULL = direct scrape; e.g. 'vegasinsider' = lifted off an aggregator
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- Betting line snapshots (historical open/close timeline with timestamps)
CREATE TABLE IF NOT EXISTS betting_line_snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_ts_utc TIMESTAMP NOT NULL,
    event_id        TEXT,
    game_date       DATE NOT NULL,
    player_id       INTEGER NOT NULL,
    book            TEXT NOT NULL,
    market_key      TEXT NOT NULL,
    stat_type       TEXT NOT NULL,
    line_value      REAL NOT NULL,
    over_odds       INTEGER,
    under_odds      INTEGER,
    raw_payload     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- Odds polling runs (for rate-limiting external API calls)
CREATE TABLE IF NOT EXISTS odds_poll_runs (
    poll_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    polled_at_utc      TIMESTAMP NOT NULL,
    status             TEXT NOT NULL, -- success, partial_success, failed
    records_parsed     INTEGER,
    records_valid      INTEGER,
    db_inserted        INTEGER,
    snapshots_inserted INTEGER,
    error_type         TEXT,
    error_message      TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw web text snapshots (direct URL scraping without API keys)
CREATE TABLE IF NOT EXISTS web_text_snapshots (
    snapshot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url       TEXT NOT NULL,
    fetched_at_utc   TIMESTAMP NOT NULL,
    http_status      INTEGER,
    content_type     TEXT,
    text_content     TEXT NOT NULL,
    text_length      INTEGER,
    content_sha256   TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Active NBA players reference (used for parser filtering/classification)
CREATE TABLE IF NOT EXISTS nba_active_players_ref (
    player_id      INTEGER PRIMARY KEY,
    player_name    TEXT NOT NULL UNIQUE,
    synced_at_utc  TIMESTAMP NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Parsed web prop cards extracted from visible text snapshots
CREATE TABLE IF NOT EXISTS web_prop_cards (
    card_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id           INTEGER NOT NULL,
    source_url            TEXT NOT NULL,
    book                  TEXT NOT NULL,
    observed_at_utc       TIMESTAMP NOT NULL,
    player_name           TEXT NOT NULL,
    player_classification TEXT NOT NULL, -- active_nba, non_nba
    stat_type             TEXT NOT NULL,
    line_value            REAL NOT NULL,
    side                  TEXT NOT NULL, -- over, under
    parse_confidence      REAL NOT NULL,
    raw_card_text         TEXT,
    parser_version        TEXT NOT NULL,
    record_sha256         TEXT NOT NULL UNIQUE,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_id) REFERENCES web_text_snapshots(snapshot_id)
);

-- Parsed game-level (team) markets from visible text snapshots.
-- One row per (book, game, market_type, side). For totals, side is
-- 'over'/'under' and team is NULL. For spreads/moneylines, side is
-- 'home'/'away' and team holds the canonical short name.
CREATE TABLE IF NOT EXISTS web_team_lines (
    line_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id       INTEGER NOT NULL,
    source_url        TEXT NOT NULL,
    book              TEXT NOT NULL,
    observed_at_utc   TIMESTAMP NOT NULL,
    away_team         TEXT NOT NULL,             -- canonical short name, e.g. "Knicks"
    home_team         TEXT NOT NULL,             -- canonical short name
    market_type       TEXT NOT NULL,             -- spread, total, moneyline, team_total
    side              TEXT NOT NULL,             -- home, away, over, under
    team              TEXT,                      -- canonical name when market is team-specific
    line_value        REAL,                      -- NULL for moneyline
    odds_american     INTEGER,                   -- NULL when not posted
    parse_confidence  REAL NOT NULL,
    raw_text          TEXT,
    parser_version    TEXT NOT NULL,
    record_sha256     TEXT NOT NULL UNIQUE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_id) REFERENCES web_text_snapshots(snapshot_id)
);

-- Reverse-engineered team-level priors derived from cross-book consensus.
-- One row per upcoming/recent NBA matchup; populated by
-- nba_model.model.team_line_reverse_engineering.
CREATE TABLE IF NOT EXISTS team_priors (
    away_team             TEXT NOT NULL,
    home_team             TEXT NOT NULL,
    computed_at_utc       TIMESTAMP NOT NULL,
    consensus_total       REAL,
    home_spread           REAL,
    away_spread           REAL,
    home_team_total       REAL,
    away_team_total       REAL,
    home_win_prob_devig   REAL,
    away_win_prob_devig   REAL,
    pace_factor           REAL,
    n_books               INTEGER,
    latest_observed_at    TIMESTAMP,
    PRIMARY KEY (away_team, home_team)
);

-- Model predictions (for evaluation)
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    stat_type TEXT NOT NULL,
    predicted_mean REAL,
    predicted_std REAL,
    prob_over REAL,
    line_value REAL,
    book_odds INTEGER,
    expected_value REAL,
    actual_result REAL,  -- Filled in after game
    outcome TEXT,  -- 'over', 'under', 'push', NULL (pending)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- Paper-trading bet log (WS10 measurement layer — NOT an execution layer).
-- One row per staked pick emitted by the bet-slip exporter. `status` starts
-- 'pending' and is graded against game_logs by settle_bet_log(); `clv_delta`
-- is backfilled from betting_line_snapshots when a closing snapshot exists.
CREATE TABLE IF NOT EXISTS bet_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_utc  TIMESTAMP NOT NULL,
    game_date       DATE NOT NULL,
    player_id       INTEGER,
    player_name     TEXT NOT NULL,
    stat_type       TEXT NOT NULL,
    book            TEXT,
    line            REAL NOT NULL,
    side            TEXT NOT NULL,                 -- 'over' | 'under'
    model_prob      REAL,
    implied_prob    REAL,
    edge            REAL,
    model_mode      TEXT,
    distribution    TEXT,
    kelly_fraction  REAL,
    stake_units     REAL,                          -- nullable
    status          TEXT NOT NULL DEFAULT 'pending', -- pending|won|lost|push|void
    settled_at_utc  TIMESTAMP,
    actual_value    REAL,
    clv_delta       REAL,                          -- nullable
    sport           TEXT NOT NULL DEFAULT 'nba',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- Prediction model configuration metadata (for reproducibility/auditing)
CREATE TABLE IF NOT EXISTS prediction_configs (
    config_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

-- MLB per-player game logs (long format: one row per player/game/stat).
-- MLB box scores diverge too much from the NBA-shaped game_logs table to share
-- it, so MLB lives in its own sport-tagged table (keeps MLB rows out of every
-- NBA query by construction). Keyed by MLB Stats API personId + gamePk.
CREATE TABLE IF NOT EXISTS mlb_game_logs (
    mlb_game_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport         TEXT NOT NULL DEFAULT 'mlb',
    player_id     INTEGER NOT NULL,        -- MLB Stats API personId
    player_name   TEXT,
    team          TEXT,                    -- team abbreviation (e.g. NYY)
    opponent      TEXT,                    -- opponent abbreviation
    game_pk       INTEGER NOT NULL,        -- MLB Stats API gamePk
    game_date     DATE NOT NULL,
    season        INTEGER,
    player_group  TEXT NOT NULL,           -- 'hitting' | 'pitching'
    stat_type     TEXT NOT NULL,           -- canonical sports/mlb.py stat key
    value         REAL NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, game_pk, stat_type)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_mlb_game_logs_player_stat ON mlb_game_logs(player_id, stat_type, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_mlb_game_logs_date ON mlb_game_logs(game_date DESC, game_pk);
CREATE INDEX IF NOT EXISTS idx_game_logs_player_date ON game_logs(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_bet_log_status ON bet_log(status, game_date);
CREATE INDEX IF NOT EXISTS idx_bet_log_player_date ON bet_log(player_id, game_date, stat_type);
CREATE INDEX IF NOT EXISTS idx_betting_lines_player_date ON betting_lines(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_prediction_configs_prediction_id ON prediction_configs(prediction_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_game_book_stat ON betting_line_snapshots(game_date, book, stat_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_player_date ON betting_line_snapshots(player_id, game_date, stat_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_event_book ON betting_line_snapshots(event_id, book, market_key, snapshot_ts_utc);
CREATE INDEX IF NOT EXISTS idx_odds_poll_runs_polled_at ON odds_poll_runs(polled_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_web_text_snapshots_url_time ON web_text_snapshots(source_url, fetched_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_active_players_ref_name ON nba_active_players_ref(player_name);
CREATE INDEX IF NOT EXISTS idx_web_prop_cards_snapshot ON web_prop_cards(snapshot_id, observed_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_web_prop_cards_player_stat ON web_prop_cards(player_name, stat_type, observed_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_web_prop_cards_book ON web_prop_cards(book, observed_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_web_team_lines_snapshot ON web_team_lines(snapshot_id, observed_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_web_team_lines_game ON web_team_lines(home_team, away_team, market_type, observed_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_web_team_lines_book ON web_team_lines(book, observed_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_games_date          ON games(game_date DESC, game_id);
CREATE INDEX IF NOT EXISTS idx_games_team_date     ON games(team_abbrev, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_games_season        ON games(season, season_type, game_date DESC);