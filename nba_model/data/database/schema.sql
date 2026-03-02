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

-- Prediction model configuration metadata (for reproducibility/auditing)
CREATE TABLE IF NOT EXISTS prediction_configs (
    config_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_game_logs_player_date ON game_logs(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_betting_lines_player_date ON betting_lines(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_prediction_configs_prediction_id ON prediction_configs(prediction_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_game_book_stat ON betting_line_snapshots(game_date, book, stat_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_player_date ON betting_line_snapshots(player_id, game_date, stat_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_event_book ON betting_line_snapshots(event_id, book, market_key, snapshot_ts_utc);