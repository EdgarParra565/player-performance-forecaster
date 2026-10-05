"""Review finding (live 2026-09-28): team_priors had no game date, so a team
with two games in the lines window got ONE prior — PHI's Oct 20 projection
used its Dec 25 total (PHI@NYK 10-20 implied 113.2 vs PHI@LAL 12-25 118.0).

Fix: extractors capture each game's date, web_team_lines / consensus /
team_priors are keyed by it, and lookups take the projection's game date.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.scrapers.game_dates import find_date_hint, resolve_game_date

NOW = datetime.now(timezone.utc).isoformat()
OBS = "2026-09-28T18:05:12+00:00"


class ResolveGameDateTests(unittest.TestCase):
    def test_tokens(self):
        cases = {
            "OCT 20": "2026-10-20", "Dec 25": "2026-12-25", "Sept 30": "2026-09-30",
            "10/20/2026": "2026-10-20", "5/8/26": "2026-05-08", "10/21": "2026-10-21",
            "2026-11-02": "2026-11-02",
        }
        for token, want in cases.items():
            self.assertEqual(resolve_game_date(token, OBS, tz=timezone.utc), want, token)

    def test_year_rolls_over_and_relative_days(self):
        self.assertEqual(resolve_game_date("Jan 3", "2026-12-20T00:00:00+00:00", tz=timezone.utc), "2027-01-03")
        self.assertEqual(resolve_game_date("Today", OBS, tz=timezone.utc), "2026-09-28")
        self.assertEqual(resolve_game_date("Tomorrow", OBS, tz=timezone.utc), "2026-09-29")
        self.assertEqual(resolve_game_date("Sat", OBS, tz=timezone.utc), "2026-10-03")  # 9/28 is a Monday
        self.assertIsNone(resolve_game_date("Feb 30", OBS, tz=timezone.utc))
        self.assertIsNone(resolve_game_date(None, OBS))

    def test_hint_finder_ignores_team_names(self):
        text = "Phoenix Suns Portland Trail Blazers +2.5 -110 OCT 21 4:10 PM"
        self.assertEqual(find_date_hint(text, 0, 44, after=40), "OCT 21")
        self.assertIsNone(find_date_hint("Phoenix Suns Sunday-less", 0, 0, after=12))


class ExtractorHintTests(unittest.TestCase):
    """The three LIVE extractors attach the date printed next to each game."""

    def _dates(self, url, text):
        from nba_model.model.team_line_parser import extract_team_lines_from_snapshot

        rows = extract_team_lines_from_snapshot(text, url, 1, OBS)
        return sorted({(r["away_team"], r["home_team"], r["game_date"]) for r in rows})

    def test_caesars(self):
        text = ("NBA Spread Money Total PHI PHI 76ers Philadelphia 76ers +5.5 -105 +162 231.5 -115 "
                "vs NYK NY Knicks New York Knicks -5.5 -115 -195 231.5 -105 OCT 20 4:10 PM "
                "Spread Money Total MIL MIL Bucks Milwaukee Bucks +5.5 -110 +180 238.5 -110 vs WAS "
                "WAS Wizards Washington Wizards -5.5 -110 -220 238.5 -110 OCT 21 4:10 PM View All")
        self.assertEqual(self._dates("https://sportsbook.caesars.com/us/ny/bet/basketball", text),
                         [("76ers", "Knicks", "2026-10-20"), ("Bucks", "Wizards", "2026-10-21")])

    def test_bovada(self):
        text = ("10/20/2026 4:00 PM +5.5 -110 -5.5 -110 +165 -195 o 231.5 -110 u 231.5 -110 "
                "Philadelphia 76ers New York Knicks + 117 12/25/2026 5:00 PM +1.5 -110 -1.5 -110 "
                "+105 -125 o 237.5 -110 u 237.5 -110 Philadelphia 76ers Los Angeles Lakers + 98")
        self.assertEqual(self._dates("https://www.bovada.lv/sports/basketball/nba", text),
                         [("76ers", "Knicks", "2026-10-20"), ("76ers", "Lakers", "2026-12-25")])

    def test_fanduel(self):
        text = ("Philadelphia 76ers New York Knicks +5.5 -105 +166 O 232.5 -110 -5.5 -115 -198 "
                "U 232.5 -110 same game parlay available Oct 20, 7:00pm ET Stats More wagers "
                "Philadelphia 76ers Los Angeles Lakers +1.5 -110 +102 O 237.5 -110 -1.5 -110 -122 "
                "U 237.5 -110 same game parlay available Dec 25, 8:00pm ET")
        self.assertEqual(self._dates("https://sportsbook.fanduel.com/navigation/nba", text),
                         [("76ers", "Knicks", "2026-10-20"), ("76ers", "Lakers", "2026-12-25")])


def _line(book, away, home, game_date, market, side, team, line, odds, ts=NOW):
    return {
        "snapshot_id": 1, "source_url": f"https://{book}.example/nba", "book": book,
        "observed_at_utc": ts, "away_team": away, "home_team": home, "game_date": game_date,
        "market_type": market, "side": side, "team": team, "line_value": line,
        "odds_american": odds, "parse_confidence": 0.9, "parser_version": "t",
        "record_sha256": f"{book}-{away}-{home}-{game_date}-{market}-{side}-{line}-{ts}",
    }


def _game(book, away, home, game_date, total, home_spread):
    return [
        _line(book, away, home, game_date, "total", "over", None, total, -110),
        _line(book, away, home, game_date, "total", "under", None, total, -110),
        _line(book, away, home, game_date, "spread", "home", home, home_spread, -110),
        _line(book, away, home, game_date, "spread", "away", away, -home_spread, -110),
    ]


class TwoGamesOneTeamTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "t.db")
        rows = []
        for book in ("bovada", "fanduel"):
            rows += _game(book, "76ers", "Knicks", "2026-10-20", 232.0, -5.5)   # PHI implied 113.25
            rows += _game(book, "76ers", "Lakers", "2026-12-25", 237.5, -1.5)   # PHI implied 118.0
        with DatabaseManager(self.db_path) as db:
            db.insert_web_team_lines(rows)
        from nba_model.model.team_line_reverse_engineering import derive_team_priors_from_consensus
        self.summary = derive_team_priors_from_consensus(since_hours=6, min_books=2, db_path=self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_each_game_gets_its_own_prior(self):
        self.assertEqual(self.summary["games_with_full_priors"], 2)
        with DatabaseManager(self.db_path) as db:
            keys = sorted(tuple(r) for r in db.conn.execute(
                "SELECT away_team, home_team, game_date FROM team_priors"))
            self.assertEqual(keys, [("PHI", "LAL", "2026-12-25"), ("PHI", "NYK", "2026-10-20")])
            oct20 = db.get_team_prior_inputs_map(game_date="2026-10-20")
            dec25 = db.get_team_prior_inputs_map(game_date="2026-12-25")
            self.assertAlmostEqual(oct20["PHI"]["implied_team_total"], 113.25)
            self.assertAlmostEqual(dec25["PHI"]["implied_team_total"], 118.0)
            self.assertNotIn("LAL", oct20)  # LAL doesn't play PHI on 10-20
            self.assertAlmostEqual(
                db.get_team_prior_inputs("PHI", "NYK", game_date="2026-10-20")["implied_team_total"], 113.25)
            # Never another game's prior.
            self.assertEqual(db.get_team_prior_inputs("PHI", "LAL", game_date="2026-10-20"), {})
            self.assertIsNone(db.get_team_prior("PHI", "NYK", game_date="2026-12-25"))

    def test_without_date_the_next_game_wins(self):
        with DatabaseManager(self.db_path) as db, \
                patch.object(DatabaseManager, "_slate_today", return_value="2026-10-01"):
            self.assertAlmostEqual(db.get_team_prior_inputs_map()["PHI"]["implied_team_total"], 113.25)
        with DatabaseManager(self.db_path) as db, \
                patch.object(DatabaseManager, "_slate_today", return_value="2026-11-01"):
            self.assertAlmostEqual(db.get_team_prior_inputs_map()["PHI"]["implied_team_total"], 118.0)
        with DatabaseManager(self.db_path) as db, \
                patch.object(DatabaseManager, "_slate_today", return_value="2027-01-10"):
            self.assertNotIn("PHI", db.get_team_prior_inputs_map())  # both games past

    def test_hourly_recompute_looks_up_its_slate_date(self):
        from nba_model.data import hourly_update
        import inspect
        src = inspect.getsource(hourly_update._run_prediction_recompute)
        self.assertIn("get_team_prior_inputs_map(game_date=today)", src)


class SameMatchupTwiceTests(unittest.TestCase):
    """A playoff-style rematch: same away/home, two dates, identical lines."""

    def test_change_only_dedupe_and_consensus_keep_games_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DatabaseManager(str(Path(tmp) / "t.db")) as db:
                rows = []
                for book in ("bovada", "fanduel"):
                    rows += _game(book, "Knicks", "76ers", "2026-05-08", 213.5, -2.5)
                    rows += _game(book, "Knicks", "76ers", "2026-05-10", 213.5, -2.5)
                res = db.insert_web_team_lines(rows)
                self.assertEqual(res["inserted"], 16)  # game 2 not "unchanged" vs game 1
                cons = db.get_consensus_team_lines(min_books=2)
                self.assertEqual(sorted({c["game_date"] for c in cons}), ["2026-05-08", "2026-05-10"])

    def test_undated_book_joins_the_only_dated_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DatabaseManager(str(Path(tmp) / "t.db")) as db:
                rows = _game("bovada", "Knicks", "76ers", "2026-10-22", 220.5, -3.5)
                rows += _game("draftkings", "Knicks", "76ers", None, 221.5, -3.5)
                db.insert_web_team_lines(rows)
                cons = db.get_consensus_team_lines(min_books=2)
                self.assertEqual({c["game_date"] for c in cons}, {"2026-10-22"})
                self.assertTrue(all(c["n_books"] == 2 for c in cons))
                # Two dated games → the undated row is ambiguous, stays apart.
                db.insert_web_team_lines(_game("fanduel", "Knicks", "76ers", "2026-10-24", 219.5, -3.5))
                dates = {c["game_date"] for c in db.get_consensus_team_lines(min_books=1)}
                self.assertEqual(dates, {"2026-10-22", "2026-10-24", None})


class GameDateMigrationTests(unittest.TestCase):
    def test_legacy_team_priors_rebuilt_and_team_lines_gain_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.db")
            DatabaseManager(path).close()
            conn = sqlite3.connect(path)
            conn.executescript("""
                DROP TABLE team_priors;
                CREATE TABLE team_priors (
                    away_team TEXT NOT NULL, home_team TEXT NOT NULL,
                    computed_at_utc TIMESTAMP NOT NULL, consensus_total REAL,
                    home_spread REAL, away_spread REAL, home_team_total REAL,
                    away_team_total REAL, home_win_prob_devig REAL,
                    away_win_prob_devig REAL, pace_factor REAL, n_books INTEGER,
                    latest_observed_at TIMESTAMP, PRIMARY KEY (away_team, home_team));
                INSERT INTO team_priors (away_team, home_team, computed_at_utc, home_team_total)
                    VALUES ('Knicks', '76ers', '2026-05-11T22:41:32+00:00', 108.0);
                ALTER TABLE web_team_lines DROP COLUMN game_date;
            """)
            conn.commit()
            conn.close()
            for _ in range(2):  # idempotent
                with DatabaseManager(path) as db:
                    rows = db.conn.execute(
                        "SELECT away_team, home_team, game_date, home_team_total FROM team_priors").fetchall()
                    pk = [r[1] for r in db.conn.execute("PRAGMA table_info(team_priors)") if r[5]]
                    wtl = {r[1] for r in db.conn.execute("PRAGMA table_info(web_team_lines)")}
                self.assertEqual(rows, [("NYK", "PHI", "", 108.0)])
                self.assertEqual(pk, ["away_team", "home_team", "game_date"])
                self.assertIn("game_date", wtl)


if __name__ == "__main__":
    unittest.main()
