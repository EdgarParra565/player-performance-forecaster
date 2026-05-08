"""End-to-end integration tests for team-line scrape → parse → consensus."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.team_line_parser import parse_and_store_web_team_lines
from nba_model.scrapers.betmgm import extract_team_lines as betmgm_extract
from nba_model.scrapers.caesars import extract_team_lines as caesars_extract
from nba_model.scrapers.draftkings import extract_team_lines as dk_extract
from nba_model.scrapers.team_names import normalize_team


class TeamNameNormalizationTests(unittest.TestCase):
    def test_resolves_short_and_full_forms(self):
        self.assertEqual(normalize_team("Knicks"), "Knicks")
        self.assertEqual(normalize_team("NY Knicks"), "Knicks")
        self.assertEqual(normalize_team("New York Knicks"), "Knicks")
        self.assertEqual(normalize_team("NYK"), "Knicks")

    def test_handles_two_word_team_names(self):
        # "Trail Blazers" must beat the partial "Blazers" match.
        self.assertEqual(normalize_team("Portland Trail Blazers"), "Trail Blazers")
        self.assertEqual(normalize_team("Trail Blazers"), "Trail Blazers")

    def test_returns_none_for_unknown(self):
        self.assertIsNone(normalize_team("Atlanta Falcons"))
        self.assertIsNone(normalize_team(""))
        self.assertIsNone(normalize_team(None))


class BetMGMExtractorTests(unittest.TestCase):
    SAMPLE = (
        "Today • 7:10 PM • Amazon Spread Total Money Knicks 53-29 76ers 45-37 "
        "+1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"
    )

    def test_emits_six_rows_per_game(self):
        rows = betmgm_extract(self.SAMPLE)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["market_type"] for r in rows},
                         {"spread", "total", "moneyline"})
        self.assertEqual({r["away_team"] for r in rows}, {"Knicks"})
        self.assertEqual({r["home_team"] for r in rows}, {"76ers"})

    def test_parses_each_market_correctly(self):
        rows = {(r["market_type"], r["side"]): r for r in betmgm_extract(self.SAMPLE)}
        self.assertEqual(rows[("spread", "away")]["line_value"], 1.5)
        self.assertEqual(rows[("spread", "away")]["odds_american"], -110)
        self.assertEqual(rows[("spread", "home")]["line_value"], -1.5)
        self.assertEqual(rows[("total", "over")]["line_value"], 213.5)
        self.assertEqual(rows[("total", "under")]["line_value"], 213.5)
        self.assertIsNone(rows[("moneyline", "away")]["line_value"])
        self.assertEqual(rows[("moneyline", "away")]["odds_american"], 100)
        self.assertEqual(rows[("moneyline", "home")]["odds_american"], -120)


class CaesarsExtractorTests(unittest.TestCase):
    SAMPLE = (
        "Today 7:00 PM Spread Money Total NYK NY Knicks New York Knicks "
        "+1.5 -105 +105 213.5 -110 vs PHI PHI 76ers Philadelphia 76ers "
        "-1.5 -115 -125 213.5 -110"
    )

    def test_extracts_full_game(self):
        rows = caesars_extract(self.SAMPLE)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["away_team"] for r in rows}, {"Knicks"})
        self.assertEqual({r["home_team"] for r in rows}, {"76ers"})


class KalshiExtractorTests(unittest.TestCase):
    SAMPLE = (
        "Game 3: New York at Philadelphia May 8 @ 7:00PM "
        "New York 2.01 x 47 % Philadelphia 1.86 x 52 % $2,013,976 vol "
        "Game 3: Oklahoma City at Los Angeles L May 9 @ 8:30PM "
        "Oklahoma City 1.28 x 77 % Los Angeles L 3.80 x 25 %"
    )

    def test_extracts_moneylines_and_normalizes_truncated_lakers(self):
        from nba_model.scrapers.kalshi import extract_team_lines
        rows = extract_team_lines(self.SAMPLE)
        # 2 games × 2 sides each = 4 rows
        self.assertEqual(len(rows), 4)
        games = {(r["away_team"], r["home_team"]) for r in rows}
        self.assertIn(("Knicks", "76ers"), games)
        # "Los Angeles L" must resolve to Lakers via the truncation alias.
        self.assertIn(("Thunder", "Lakers"), games)
        # All rows are moneyline (Kalshi index doesn't expose spread/total).
        self.assertEqual({r["market_type"] for r in rows}, {"moneyline"})
        self.assertTrue(all(r["line_value"] is None for r in rows))

    def test_decimal_to_american_conversion(self):
        from nba_model.scrapers.kalshi import _decimal_to_american
        # Decimal 2.01 → +101 (underdog), 1.86 → -116 (favorite).
        self.assertEqual(_decimal_to_american(2.01), 101)
        self.assertEqual(_decimal_to_american(1.86), -116)
        self.assertEqual(_decimal_to_american(1.28), -357)


class DraftKingsExtractorTests(unittest.TestCase):
    # Note: real DK uses Unicode minus (U+2212). Sample uses both to verify
    # the normalization step.
    SAMPLE = (
        "Today Spread Total Moneyline NY Knicks at PHI 76ers "
        "+1.5 −108 O 213.5 −105 +102 -1.5 −112 U 213.5 −115 −122"
    )

    def test_normalizes_unicode_minus_and_extracts(self):
        rows = dk_extract(self.SAMPLE)
        self.assertEqual(len(rows), 6)
        by = {(r["market_type"], r["side"]): r for r in rows}
        # Unicode minus must have been mapped to ASCII so int() works.
        self.assertEqual(by[("spread", "away")]["odds_american"], -108)
        self.assertEqual(by[("spread", "home")]["odds_american"], -112)
        self.assertEqual(by[("total", "over")]["odds_american"], -105)
        self.assertEqual(by[("moneyline", "home")]["odds_american"], -122)


class TeamLineRoundTripTests(unittest.TestCase):
    BETMGM_TEXT = (
        "Today • 7:10 PM Spread Total Money Knicks 53-29 76ers 45-37 "
        "+1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"
    )
    CAESARS_TEXT = (
        "Today 7:00 PM Spread Money Total NYK NY Knicks New York Knicks "
        "+1.5 -105 +105 213.5 -110 vs PHI PHI 76ers Philadelphia 76ers "
        "-1.5 -115 -125 213.5 -110"
    )

    def test_two_books_same_game_produces_multi_book_consensus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([
                    {
                        "source_url": "https://sports.betmgm.com/en/sports/basketball-7",
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "http_status": 200, "content_type": "text/html",
                        "text_content": self.BETMGM_TEXT,
                        "text_length": len(self.BETMGM_TEXT),
                        "content_sha256": "betmgm-1",
                    },
                    {
                        "source_url": "https://sportsbook.caesars.com/basketball",
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "http_status": 200, "content_type": "text/html",
                        "text_content": self.CAESARS_TEXT,
                        "text_length": len(self.CAESARS_TEXT),
                        "content_sha256": "caesars-1",
                    },
                ])

            summary = parse_and_store_web_team_lines(
                db_path=db_path,
                source_urls=[
                    "https://sports.betmgm.com/en/sports/basketball-7",
                    "https://sportsbook.caesars.com/basketball",
                ],
                max_snapshots_per_url=1,
                max_total_snapshots=10,
                min_parse_confidence=0.45,
            )

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["lines_extracted"], 12)  # 2 books × 6 markets
            self.assertEqual(summary["db_inserted"], 12)

            with DatabaseManager(db_path=db_path) as db:
                consensus = db.get_consensus_team_lines(
                    away_team="Knicks", home_team="76ers",
                    market_type="spread", side="away",
                )

        self.assertEqual(len(consensus), 1)
        row = consensus[0]
        # BetMGM and Caesars both posted Knicks +1.5 → mean 1.5.
        self.assertAlmostEqual(row["mean_line"], 1.5, places=3)
        self.assertEqual(row["n_books"], 2)
        self.assertIn("betmgm", row["books"])
        self.assertIn("caesars", row["books"])

    def test_dedupe_keeps_only_latest_per_book(self):
        url = "https://sports.betmgm.com/en/sports/basketball-7"
        old_text = (
            "Today • 7:10 PM Spread Total Money Knicks 53-29 76ers 45-37 "
            "+1.5 -110 -1.5 -110 O 213.0 -110 U 213.0 -110 +100 -120"
        )
        new_text = (
            "Today • 7:10 PM Spread Total Money Knicks 53-29 76ers 45-37 "
            "+1.5 -110 -1.5 -110 O 214.0 -110 U 214.0 -110 +100 -120"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([
                    {"source_url": url, "fetched_at_utc": "2026-05-08T00:00:00+00:00",
                     "http_status": 200, "content_type": "text/html",
                     "text_content": old_text, "text_length": len(old_text),
                     "content_sha256": "old"},
                    {"source_url": url, "fetched_at_utc": "2026-05-08T01:00:00+00:00",
                     "http_status": 200, "content_type": "text/html",
                     "text_content": new_text, "text_length": len(new_text),
                     "content_sha256": "new"},
                ])

            parse_and_store_web_team_lines(
                db_path=db_path, source_urls=[url],
                max_snapshots_per_url=10, max_total_snapshots=20,
                min_parse_confidence=0.45,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.get_consensus_team_lines(
                    market_type="total", side="over",
                )

        self.assertEqual(len(rows), 1)
        # Latest snapshot was 214.0; older 213.0 must be ignored.
        self.assertAlmostEqual(rows[0]["mean_line"], 214.0, places=3)


if __name__ == "__main__":
    unittest.main()
