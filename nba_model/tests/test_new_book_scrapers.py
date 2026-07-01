"""Tests for the newly-activated FanDuel / BetRivers / ESPN BET team-line parsers.

FanDuel's format mirrors the shape documented in its module (DraftKings-like,
``@`` separator + a date/time block). BetRivers and ESPN BET were authored
against representative fixtures, NOT live authenticated snapshots — see the
``TODO(real-capture)`` notes in each scraper module. These tests pin the
parser behaviour against those committed fixtures so a real capture later can
be diffed against a known baseline.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.team_line_parser import parse_and_store_web_team_lines
from nba_model.scrapers.betrivers import extract_team_lines as betrivers_extract
from nba_model.scrapers.espnbet import extract_team_lines as espnbet_extract
from nba_model.scrapers.fanduel import extract_team_lines as fanduel_extract


# Representative single-game fixtures (whitespace-collapsed visible text).
FANDUEL_TEXT = (
    "NY Knicks @ PHI 76ers Today 7:10pm "
    "+1.5 -108 O 213.5 -110 +102 -1.5 -112 U 213.5 -110 -122"
)
BETRIVERS_TEXT = (
    "Knicks at 76ers +1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"
)
ESPNBET_TEXT = (
    "NY Knicks at PHI 76ers +1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"
)


class _ExtractorAssertMixin:
    extract = None
    text = None
    expect_spread_away_odds = None
    expect_away_ml = None
    expect_home_ml = None

    def test_emits_six_rows_one_game(self):
        rows = type(self).extract(self.text)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["away_team"] for r in rows}, {"Knicks"})
        self.assertEqual({r["home_team"] for r in rows}, {"76ers"})
        self.assertEqual(
            {r["market_type"] for r in rows}, {"spread", "total", "moneyline"}
        )

    def test_market_values(self):
        by = {(r["market_type"], r["side"]): r for r in type(self).extract(self.text)}
        self.assertEqual(by[("spread", "away")]["line_value"], 1.5)
        self.assertEqual(
            by[("spread", "away")]["odds_american"], self.expect_spread_away_odds
        )
        self.assertEqual(by[("spread", "home")]["line_value"], -1.5)
        self.assertEqual(by[("total", "over")]["line_value"], 213.5)
        self.assertEqual(by[("total", "under")]["line_value"], 213.5)
        self.assertIsNone(by[("moneyline", "away")]["line_value"])
        self.assertEqual(by[("moneyline", "away")]["odds_american"], self.expect_away_ml)
        self.assertEqual(by[("moneyline", "home")]["odds_american"], self.expect_home_ml)


class FanDuelExtractorTests(_ExtractorAssertMixin, unittest.TestCase):
    extract = staticmethod(fanduel_extract)
    text = FANDUEL_TEXT
    expect_spread_away_odds = -108
    expect_away_ml = 102
    expect_home_ml = -122


class BetRiversExtractorTests(_ExtractorAssertMixin, unittest.TestCase):
    extract = staticmethod(betrivers_extract)
    text = BETRIVERS_TEXT
    expect_spread_away_odds = -110
    expect_away_ml = 100
    expect_home_ml = -120


class ESPNBetExtractorTests(_ExtractorAssertMixin, unittest.TestCase):
    extract = staticmethod(espnbet_extract)
    text = ESPNBET_TEXT
    expect_spread_away_odds = -110
    expect_away_ml = 100
    expect_home_ml = -120


class NewBooksRoundTripTests(unittest.TestCase):
    """Snapshot → parse → store → consensus through the real parser path."""

    def test_three_books_same_game_build_consensus(self):
        urls = {
            "https://sportsbook.fanduel.com/navigation/nba": FANDUEL_TEXT,
            "https://betrivers.com/?page=sportsbook": BETRIVERS_TEXT,
            "https://espnbet.com/sport/basketball/organization/united-states/competition/nba": ESPNBET_TEXT,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            now = datetime.now(timezone.utc).isoformat()
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([
                    {
                        "source_url": url,
                        "fetched_at_utc": now,
                        "http_status": 200,
                        "content_type": "text/html",
                        "text_content": text,
                        "text_length": len(text),
                        "content_sha256": f"sha-{i}",
                    }
                    for i, (url, text) in enumerate(urls.items())
                ])

            summary = parse_and_store_web_team_lines(
                db_path=db_path,
                source_urls=list(urls),
                max_snapshots_per_url=1,
                max_total_snapshots=10,
                min_parse_confidence=0.45,
            )

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["lines_extracted"], 18)  # 3 books × 6 markets
            self.assertEqual(summary["db_inserted"], 18)

            with DatabaseManager(db_path=db_path) as db:
                consensus = db.get_consensus_team_lines(
                    away_team="Knicks", home_team="76ers",
                    market_type="spread", side="away",
                )

        self.assertEqual(len(consensus), 1)
        row = consensus[0]
        self.assertAlmostEqual(row["mean_line"], 1.5, places=3)
        self.assertEqual(row["n_books"], 3)
        for book in ("fanduel", "betrivers", "espnbet"):
            self.assertIn(book, row["books"])


if __name__ == "__main__":
    unittest.main()
