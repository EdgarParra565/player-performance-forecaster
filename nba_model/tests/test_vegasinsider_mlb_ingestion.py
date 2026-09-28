"""Tests for the VegasInsider MLB props ingest into ``mlb_prop_lines``.

The fixture is a trimmed, verbatim excerpt of the REAL 2026-07-22 capture
(``web_text_snapshots`` id 155, ``vegasinsider.com/mlb/odds/player-props``):

  * Strikeouts (10-book header): Gerrit Cole (9 cells — partial, dropped),
    Reid Detmers + Emerson Hancock (10 cells — full).
  * Home Runs (6-book header): C.J. Abrams (6 cells, mixed ``o0.5`` + bare
    yes-price cells — full).
  * Total Bases (8-book header): James Wood (8 cells — full), Luis Garcia
    (5 cells — partial, dropped).

BetMGM columns hold position but are not emitted (parser's untrusted-book
rule), so the full rows yield 9 + 9 + 5 + 7 = 30 rows. Parser-only behaviour
is pinned in ``test_vegasinsider_mlb_parser``; this file covers storage,
change-only semantics, registry validation, NBA isolation and ETL wiring.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.vegasinsider_mlb_props_ingestion import (
    ingest_vegasinsider_mlb_props,
    is_valid_mlb_prop_row,
)
from nba_model.model import cross_book_arb as cba
from sports import get_sport


VI_MLB_FIXTURE = (
    "Be sure to check out the latest strikeout odds and homerun odds ! "
    "Strikeouts Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars FanDuel "
    "Fanatics Sleeper Underdog RiversCasino › › › › › › › › › › "
    "Gerrit Cole o6.5 -110 + o6.5 -137 + o10.5 +110 o6.5 -112 + o6.5 -108 "
    "o6.5 -100 + o6.5 -122 + o6.5 -137 o6.5 -114 + "
    "Reid Detmers o5.5 -165 + o6 -137 + o6.5 +105 o5.5 -156 + o5.5 -148 "
    "o6.5 +146 + o5.5 +145 + o5.5 -169 + o5.5 -137 o5.5 -141 + "
    "Emerson Hancock o5.5 -110 + o5.5 -137 + o5.5 -110 o5.5 -112 + o5.5 -113 "
    "o5.5 -113 o5.5 -115 + o5.5 -123 + o5.5 -137 o5.5 -109 + "
    "See All Home Runs Odds Time Bet365 BetMGM Caesars FanDuel HardRock "
    "Sleeper › › › › › › "
    "C.J. Abrams o0.5 +310 + +340 + +290 + +310 +325 + o0.5 +241 + "
    "See All Total Bases Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars "
    "Fanatics Sleeper Underdog › › › › › › › › "
    "James Wood o1.5 -150 + o1.5 -137 + o1.5 +105 o2.5 +113 + o1.5 +107 "
    "o1.5 -150 + o1.5 -179 + o1.5 -137 "
    "Luis Garcia o1.5 -105 + o1.5 -147 + o1.5 -152 o1.5 -155 + o1.5 -167 +"
)
EXPECTED_ROWS = 30
OBSERVED = "2026-07-22T22:52:14.730059+00:00"
VI_MLB_URL = "https://www.vegasinsider.com/mlb/odds/player-props/"


def _tmp_db(tmp: str) -> str:
    return str(Path(tmp) / "nba_data.db")


def _rows(db_path: str) -> list[dict]:
    with DatabaseManager(db_path=db_path) as db:
        cur = db.conn.execute("SELECT * FROM mlb_prop_lines ORDER BY mlb_prop_line_id")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _seed_snapshot(db_path: str, text: str, url: str = VI_MLB_URL) -> None:
    with DatabaseManager(db_path=db_path) as db:
        db.insert_web_text_snapshots([{
            "source_url": url,
            "fetched_at_utc": OBSERVED,
            "http_status": 200,
            "content_type": "text/html",
            "text_content": text,
            "text_length": len(text),
            "content_sha256": f"vi-mlb-{len(text)}",
        }])


class IngestFixtureTests(unittest.TestCase):
    def test_rows_land_sport_tagged_with_real_odds(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            summary = ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc=OBSERVED)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["parsed_rows"], EXPECTED_ROWS)
            self.assertEqual(summary["inserted"], EXPECTED_ROWS)
            self.assertEqual(summary["rejected_invalid"], 0)
            self.assertEqual(summary["game_date"], "2026-07-22")
            rows = _rows(db_path)
        self.assertEqual(len(rows), EXPECTED_ROWS)
        self.assertTrue(all(r["sport"] == "mlb" for r in rows))
        self.assertTrue(all(r["source"] == "vegasinsider" for r in rows))
        self.assertTrue(all(r["under_odds"] is None for r in rows))
        self.assertTrue(all(r["side"] == "over" for r in rows))
        self.assertNotIn("betmgm", {r["book"] for r in rows})
        # Registry-validated stat keys only.
        mlb_stats = set(get_sport("mlb").stat_types)
        self.assertTrue({r["stat_type"] for r in rows} <= mlb_stats)
        by = {(r["player_name"], r["book"], r["stat_type"]): r for r in rows}
        dk = by[("Reid Detmers", "draftkings", "strikeouts_pitcher")]
        self.assertEqual((dk["line_value"], dk["over_odds"]), (5.5, -156))
        self.assertIn(("Reid Detmers", "betrivers", "strikeouts_pitcher"), by)
        self.assertIn(("C.J. Abrams", "hardrockbet", "anytime_home_run"), by)
        # Partial rows never land.
        self.assertNotIn("Gerrit Cole", {r["player_name"] for r in rows})
        self.assertNotIn("Luis Garcia", {r["player_name"] for r in rows})

    def test_hitter_and_pitcher_groups_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc=OBSERVED)
            rows = _rows(db_path)
        groups = {}
        for r in rows:
            groups.setdefault(r["stat_group"], set()).add(r["stat_type"])
        self.assertEqual(groups["pitching"], {"strikeouts_pitcher"})
        self.assertEqual(groups["hitting"], {"total_bases"})
        self.assertEqual(groups["combined"], {"anytime_home_run"})

    def test_both_line_shapes_land(self):
        # HR mixes o0.5 cells and bare yes-prices; both are the 0.5 yes/no
        # market. Count markets are over/under.
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc=OBSERVED)
            rows = _rows(db_path)
        hr = [r for r in rows if r["stat_type"] == "anytime_home_run"]
        self.assertEqual({r["market_shape"] for r in hr}, {"yes_no"})
        self.assertTrue(all(r["line_value"] == 0.5 for r in hr))
        others = [r for r in rows if r["stat_type"] != "anytime_home_run"]
        self.assertEqual({r["market_shape"] for r in others}, {"over_under"})


class ChangeOnlyTests(unittest.TestCase):
    def test_reingest_unchanged_inserts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc=OBSERVED)
            again = ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc="2026-07-22T23:52:14+00:00")
            self.assertEqual(again["inserted"], 0)
            self.assertEqual(again["skipped_unchanged"], EXPECTED_ROWS)
            self.assertEqual(len(_rows(db_path)), EXPECTED_ROWS)

    def test_price_move_lands_one_new_row(self):
        moved = VI_MLB_FIXTURE.replace(
            "Reid Detmers o5.5 -165", "Reid Detmers o5.5 -175", 1)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc=OBSERVED)
            second = ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=moved,
                observed_at_utc="2026-07-22T23:52:14+00:00")
            self.assertEqual(second["inserted"], 1)
            rows = [r for r in _rows(db_path)
                    if r["player_name"] == "Reid Detmers" and r["book"] == "bet365"]
        self.assertEqual([r["over_odds"] for r in rows], [-165, -175])

    def test_insert_skips_incomplete_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DatabaseManager(db_path=_tmp_db(tmp)) as db:
                out = db.insert_mlb_prop_lines([
                    {"book": "bet365", "player_name": "X Y", "line_value": 1.5},
                    {"source": "vegasinsider", "book": "bet365",
                     "observed_at_utc": OBSERVED, "player_name": "X Y",
                     "stat_type": "hits", "stat_group": "hitting",
                     "market_shape": "over_under", "line_value": "nope",
                     "side": "over", "parser_version": "v",
                     "record_sha256": "abc"},
                ])
        self.assertEqual(out["inserted"], 0)


class RegistryValidationTests(unittest.TestCase):
    def test_valid_rows(self):
        self.assertTrue(is_valid_mlb_prop_row("strikeouts_pitcher", 5.5, -110))
        self.assertTrue(is_valid_mlb_prop_row("anytime_home_run", 0.5, 450))

    def test_rejects_non_mlb_stat(self):
        self.assertFalse(is_valid_mlb_prop_row("points", 25.5, -110))

    def test_rejects_out_of_range_line(self):
        self.assertFalse(is_valid_mlb_prop_row("strikeouts_pitcher", 25.5, -110))
        self.assertFalse(is_valid_mlb_prop_row("total_bases", -1, -110))

    def test_yes_no_market_must_sit_at_half(self):
        self.assertFalse(is_valid_mlb_prop_row("anytime_home_run", 1.5, 900))

    def test_rejects_bad_odds(self):
        self.assertFalse(is_valid_mlb_prop_row("total_bases", 1.5, 0))
        self.assertFalse(is_valid_mlb_prop_row("total_bases", 1.5, None))
        self.assertFalse(is_valid_mlb_prop_row("total_bases", 1.5, 99999999))


class NbaIsolationTests(unittest.TestCase):
    """MLB rows must never reach NBA tables / readers."""

    def test_no_rows_in_nba_tables_or_arb_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            ingest_vegasinsider_mlb_props(
                db_path=db_path, snapshot_text=VI_MLB_FIXTURE,
                observed_at_utc=OBSERVED)
            with DatabaseManager(db_path=db_path) as db:
                n_bl = db.conn.execute("SELECT COUNT(*) FROM betting_lines").fetchone()[0]
                n_wp = db.conn.execute("SELECT COUNT(*) FROM web_prop_cards").fetchone()[0]
            self.assertEqual((n_bl, n_wp), (0, 0))
            self.assertTrue(cba.fetch_two_way_lines(db_path=db_path).empty)

    def test_nba_ingester_ignores_mlb_snapshot(self):
        from nba_model.data.vegasinsider_odds_ingestion import ingest_vegasinsider_odds
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            _seed_snapshot(db_path, VI_MLB_FIXTURE)
            summary = ingest_vegasinsider_odds(db_path=db_path)
            self.assertEqual(summary["status"], "no_snapshot")


class StoredSnapshotAndEtlTests(unittest.TestCase):
    def test_reads_latest_stored_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            _seed_snapshot(db_path, VI_MLB_FIXTURE)
            summary = ingest_vegasinsider_mlb_props(db_path=db_path)
            self.assertEqual(summary["inserted"], EXPECTED_ROWS)
            self.assertIsNotNone(summary["snapshot_id"])
            rows = _rows(db_path)
        self.assertTrue(all(r["snapshot_id"] == summary["snapshot_id"] for r in rows))
        self.assertTrue(all(r["source_url"] == VI_MLB_URL for r in rows))

    def test_no_snapshot_and_empty_grid_are_zero_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            self.assertEqual(
                ingest_vegasinsider_mlb_props(db_path=db_path)["status"], "no_snapshot")
            _seed_snapshot(db_path, "MLB Player Props Odds 2026 Sign Up Subscribe")
            summary = ingest_vegasinsider_mlb_props(db_path=db_path)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["inserted"], 0)

    def test_hourly_and_daily_steps_ingest(self):
        from nba_model.data.daily_etl import _run_vegasinsider_mlb_props_step
        from nba_model.data.hourly_update import _run_vegasinsider_mlb_props_ingestion
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            _seed_snapshot(db_path, VI_MLB_FIXTURE)
            first = _run_vegasinsider_mlb_props_ingestion(db_path)
            self.assertEqual(first["inserted"], EXPECTED_ROWS)
            second = _run_vegasinsider_mlb_props_step(db_path)
            self.assertEqual(second["inserted"], 0)

    def test_hourly_step_order(self):
        import inspect
        from nba_model.data import hourly_update
        src = inspect.getsource(hourly_update.run_hourly_update)
        self.assertLess(src.index('"vegasinsider_ingestion"'),
                        src.index('"vegasinsider_mlb_props_ingestion"'))
        self.assertLess(src.index('"vegasinsider_mlb_props_ingestion"'),
                        src.index('"game_log_refresh"'))

    def test_daily_report_lists_step_skipped_without_urls(self):
        from nba_model.data.daily_etl import run_daily_etl
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _tmp_db(tmp)
            with DatabaseManager(db_path=db_path):
                pass
            report = run_daily_etl(
                db_path=db_path,
                skip_game_logs=True,
                skip_team_defense=True,
                skip_odds=True,
                skip_reverse_engineering=True,
                skip_bulk_results_ingest=True,
                write_report=False,
            )
        step = report["steps"]["vegasinsider_mlb_props_ingestion"]
        self.assertEqual(step["status"], "skipped")
        self.assertEqual(report["status"], "success")


if __name__ == "__main__":
    unittest.main()
