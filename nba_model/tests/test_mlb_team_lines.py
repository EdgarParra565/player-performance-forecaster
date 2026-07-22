"""Tests for MLB team-line extraction — validated against a LIVE capture.

The fixture below is a REAL, unmodified slice of FanDuel's MLB lobby visible
text captured through the CDP path on 2026-06-28 (two pre-game blocks). Player
/ pitcher names are public. This pins the parser to the actual book token
order (run-line / total / moneyline), not a guessed shape.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.team_line_parser import parse_and_store_web_team_lines
from nba_model.scrapers.mlb_team_lines import extract_team_lines, extract_team_lines_dk
from nba_model.scrapers.mlb_team_names import normalize_mlb_team

# Real FanDuel MLB lobby slice (captured live 2026-06-28 via CDP).
FANDUEL_MLB_REAL = (
    "San Diego Padres J Sears Chicago Cubs M Boyd "
    "+1.5 -154 +130 O 11.5 +106 -1.5 +128 -154 U 11.5 -130 "
    "same game parlay available 8:06pm ET Stats More wagers "
    "Minnesota Twins J Ryan Houston Astros M Burrows "
    "-1.5 +146 -112 O 8.5 +100 +1.5 -178 -104 U 8.5 -122 "
    "same game parlay available 8:11pm ET"
)


class MlbTeamNameTests(unittest.TestCase):
    def test_full_names_and_nicknames(self):
        self.assertEqual(normalize_mlb_team("New York Yankees"), "Yankees")
        self.assertEqual(normalize_mlb_team("Yankees"), "Yankees")
        self.assertEqual(normalize_mlb_team("Blue Jays"), "Blue Jays")
        self.assertEqual(normalize_mlb_team("Toronto Blue Jays"), "Blue Jays")
        self.assertIsNone(normalize_mlb_team("Los Angeles Lakers"))  # NBA, not MLB


class MlbTeamLineExtractorTests(unittest.TestCase):
    def setUp(self):
        self.rows = extract_team_lines(FANDUEL_MLB_REAL)

    def test_two_games_twelve_rows(self):
        self.assertEqual(len(self.rows), 12)
        games = {(r["away_team"], r["home_team"]) for r in self.rows}
        self.assertEqual(games, {("Padres", "Cubs"), ("Twins", "Astros")})

    def test_padres_cubs_markets(self):
        by = {(r["market_type"], r["side"]): r
              for r in self.rows if r["away_team"] == "Padres"}
        self.assertEqual(by[("spread", "away")]["line_value"], 1.5)
        self.assertEqual(by[("spread", "away")]["odds_american"], -154)
        self.assertEqual(by[("spread", "home")]["line_value"], -1.5)
        self.assertEqual(by[("spread", "home")]["odds_american"], 128)
        self.assertEqual(by[("total", "over")]["line_value"], 11.5)
        self.assertEqual(by[("total", "over")]["odds_american"], 106)
        self.assertEqual(by[("total", "under")]["odds_american"], -130)
        self.assertIsNone(by[("moneyline", "away")]["line_value"])
        self.assertEqual(by[("moneyline", "away")]["odds_american"], 130)
        self.assertEqual(by[("moneyline", "home")]["odds_american"], -154)

    def test_twins_astros_markets(self):
        by = {(r["market_type"], r["side"]): r
              for r in self.rows if r["away_team"] == "Twins"}
        self.assertEqual(by[("spread", "away")]["line_value"], -1.5)
        self.assertEqual(by[("spread", "away")]["odds_american"], 146)
        self.assertEqual(by[("total", "over")]["line_value"], 8.5)
        self.assertEqual(by[("moneyline", "home")]["odds_american"], -104)


# Real DraftKings MLB lobby slice (captured live 2026-07-22 via CDP). DK's token
# order differs from FanDuel — the moneyline follows each side's over/under block
# rather than sitting beside the run line — and DK renders each team as
# "<ABBREV> <Nickname>" so the abbrev + nickname both resolve to the same
# canonical name (adjacency-collapse pairing must dedupe them). Minus signs are
# the real U+2212 the site emits.
DRAFTKINGS_MLB_REAL = (
    "LA Dodgers Eric Lauer AT PHI Phillies Aaron Nola "
    "-1.5 +119 O 9.5 −118 −131 +1.5 −143 U 9.5 −102 +109 "
    "Today 6:45 PM More Bets "
    "PIT Pirates Bubba Chandler AT NY Yankees Max Fried "
    "+1.5 −127 O 9 −117 +153 -1.5 +105 U 9 −103 −186 "
    "Today 7:05 PM More Bets"
)


class DraftKingsMlbTeamLineExtractorTests(unittest.TestCase):
    def setUp(self):
        self.rows = extract_team_lines_dk(DRAFTKINGS_MLB_REAL)

    def test_two_games_twelve_rows(self):
        self.assertEqual(len(self.rows), 12)
        games = {(r["away_team"], r["home_team"]) for r in self.rows}
        self.assertEqual(games, {("Dodgers", "Phillies"), ("Pirates", "Yankees")})

    def test_dodgers_phillies_dk_token_order(self):
        by = {(r["market_type"], r["side"]): r
              for r in self.rows if r["away_team"] == "Dodgers"}
        # Run line beside its odds; moneyline pulled from AFTER the over/under.
        self.assertEqual(by[("spread", "away")]["line_value"], -1.5)
        self.assertEqual(by[("spread", "away")]["odds_american"], 119)
        self.assertEqual(by[("spread", "home")]["line_value"], 1.5)
        self.assertEqual(by[("spread", "home")]["odds_american"], -143)
        self.assertEqual(by[("total", "over")]["line_value"], 9.5)
        self.assertEqual(by[("total", "over")]["odds_american"], -118)
        self.assertEqual(by[("total", "under")]["odds_american"], -102)
        self.assertEqual(by[("moneyline", "away")]["odds_american"], -131)
        self.assertEqual(by[("moneyline", "home")]["odds_american"], 109)

    def test_pirates_yankees_markets(self):
        by = {(r["market_type"], r["side"]): r
              for r in self.rows if r["away_team"] == "Pirates"}
        self.assertEqual(by[("spread", "away")]["odds_american"], -127)
        self.assertEqual(by[("total", "over")]["line_value"], 9.0)
        self.assertEqual(by[("moneyline", "away")]["odds_american"], 153)
        self.assertEqual(by[("moneyline", "home")]["odds_american"], -186)

    def test_fanduel_order_extractor_rejects_dk_text(self):
        # The FanDuel-order block must NOT match DK's different token order.
        self.assertEqual(extract_team_lines(DRAFTKINGS_MLB_REAL), [])


class MlbTeamLineRoundTripTests(unittest.TestCase):
    URL = "https://sportsbook.fanduel.com/navigation/mlb"

    def test_parse_stores_and_stays_sport_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([{
                    "source_url": self.URL,
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "http_status": 200, "content_type": "text/html",
                    "text_content": FANDUEL_MLB_REAL,
                    "text_length": len(FANDUEL_MLB_REAL),
                    "content_sha256": "fd-mlb-real-1",
                }])

            summary = parse_and_store_web_team_lines(
                db_path=db_path, source_urls=[self.URL],
                max_snapshots_per_url=1, max_total_snapshots=5, sport="mlb",
            )
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["lines_extracted"], 12)
            self.assertEqual(summary["db_inserted"], 12)

            with DatabaseManager(db_path=db_path) as db:
                # Rows are tagged sport='mlb'.
                n_mlb = db.conn.execute(
                    "SELECT COUNT(*) FROM web_team_lines WHERE sport='mlb'").fetchone()[0]
                self.assertEqual(n_mlb, 12)
                # NBA consensus (default sport='nba') must not see MLB rows.
                nba = db.get_consensus_team_lines(market_type="total", side="over")
                self.assertEqual(len(nba), 0)
                # MLB consensus does.
                mlb = db.get_consensus_team_lines(
                    market_type="total", side="over", sport="mlb")
                self.assertEqual(len(mlb), 2)

    def test_draftkings_mlb_config_round_trip(self):
        url = "https://sportsbook.draftkings.com/leagues/baseball/mlb"
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([{
                    "source_url": url,
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "http_status": 200, "content_type": "text/html",
                    "text_content": DRAFTKINGS_MLB_REAL,
                    "text_length": len(DRAFTKINGS_MLB_REAL),
                    "content_sha256": "dk-mlb-real-1",
                }])
            summary = parse_and_store_web_team_lines(
                db_path=db_path, source_urls=[url],
                max_snapshots_per_url=1, max_total_snapshots=5, sport="mlb",
            )
            # DK config (draftkings, mlb) resolves to the DK-order extractor.
            self.assertEqual(summary["lines_extracted"], 12)
            self.assertEqual(summary["db_inserted"], 12)
            with DatabaseManager(db_path=db_path) as db:
                book = db.conn.execute(
                    "SELECT DISTINCT book FROM web_team_lines WHERE sport='mlb'"
                ).fetchone()[0]
                self.assertEqual(book, "draftkings")


if __name__ == "__main__":
    unittest.main()
