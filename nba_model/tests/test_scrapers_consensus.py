"""End-to-end integration tests: snapshot → parse → store → cross-book consensus.

These tests don't hit the network — they feed simulated visible-text strings
in the shape each book actually emits, run them through the same parser the
live ingest path uses, and then exercise ``get_consensus_prop_lines`` to
prove the averaging step works.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.browser_prop_parser import parse_and_store_web_prop_cards
from nba_model.scrapers import SCRAPERS, get_scraper_for_url, get_scraper_by_name


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScraperRegistryTests(unittest.TestCase):
    """Sanity checks on the registry itself."""

    def test_registry_exposes_all_books(self):
        names = {s.name for s in SCRAPERS}
        # Working parsers
        self.assertIn("prizepicks", names)
        self.assertIn("underdog", names)
        # Stubs (config only — no parser yet, but they must still be registered
        # so their fetch/auth path picks up the right wait selectors).
        for stub in (
            "fliff", "draftkings", "fanduel", "betmgm",
            "caesars", "betrivers", "kalshi",
        ):
            self.assertIn(stub, names)

    def test_url_lookup_resolves_subdomains(self):
        scraper = get_scraper_for_url("https://app.prizepicks.com/board/nba")
        self.assertIsNotNone(scraper)
        self.assertEqual(scraper.name, "prizepicks")

        scraper = get_scraper_for_url("https://underdogfantasy.com/pick-em")
        self.assertIsNotNone(scraper)
        self.assertEqual(scraper.name, "underdog")

    def test_stub_books_have_no_parser_but_full_session_config(self):
        for name in ("draftkings", "fanduel", "betmgm",
                     "caesars", "betrivers", "fliff", "kalshi"):
            scraper = get_scraper_by_name(name)
            self.assertIsNotNone(scraper, f"{name} missing from registry")
            # Stubs intentionally have no parser — the parser section is the
            # part that needs an authenticated text sample to pin down.
            self.assertEqual(scraper.parser_regexes, ())
            self.assertIsNone(scraper.prop_preprocess)
            # …but they must be wired to detect login walls and wait for
            # content, otherwise the fetch path can't tell what it scraped.
            self.assertTrue(scraper.wait_selectors)
            self.assertTrue(scraper.session_markers.login_wall)
            self.assertTrue(scraper.session_markers.authenticated)


class CrossBookConsensusTests(unittest.TestCase):
    """Round-trip: simulated PrizePicks + Underdog text → parser → DB → mean."""

    def _seed_active_players(self, db, names):
        db.upsert_active_players_reference(
            [
                {"player_id": 1000 + idx, "player_name": name,
                 "synced_at_utc": _utc_now_iso()}
                for idx, name in enumerate(names)
            ]
        )

    def test_consensus_averages_across_two_books(self):
        # Same player, same stat, two books with slightly different lines.
        # PrizePicks shape: "Player TEAM Stat Line More Less"
        prizepicks_text = (
            "NBA Projections "
            "LeBron James LAL Points 27.5 More Less "
            "Stephen Curry GSW Assists 6.5 More Less"
        )
        # Underdog shape (must include 'TEAM @ TEAM - <time> <TZ>'):
        underdog_text = (
            "LeBron James LAL @ DEN - 7:30 PM EDT 28.5 Points Higher 1.06x "
            "Stephen Curry GSW @ LAL - 7:30 PM EDT 7.5 Assists Higher 1.06x"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_active_players(db, ["LeBron James", "Stephen Curry"])
                db.insert_web_text_snapshots(
                    [
                        {
                            "source_url": "https://app.prizepicks.com/board/nba",
                            "fetched_at_utc": _utc_now_iso(),
                            "http_status": 200,
                            "content_type": "text/html",
                            "text_content": prizepicks_text,
                            "text_length": len(prizepicks_text),
                            "content_sha256": "pp-1",
                        },
                        {
                            "source_url": "https://underdogfantasy.com/pick-em/higher-lower/all/NBA",
                            "fetched_at_utc": _utc_now_iso(),
                            "http_status": 200,
                            "content_type": "text/html",
                            "text_content": underdog_text,
                            "text_length": len(underdog_text),
                            "content_sha256": "ud-1",
                        },
                    ]
                )

            # Run the same parser the live ingest path runs.
            summary = parse_and_store_web_prop_cards(
                db_path=db_path,
                source_urls=[
                    "https://app.prizepicks.com/board/nba",
                    "https://underdogfantasy.com/pick-em/higher-lower/all/NBA",
                ],
                max_snapshots_per_url=1,
                max_total_snapshots=10,
                min_parse_confidence=0.45,
            )

            self.assertEqual(summary["status"], "success")
            self.assertGreaterEqual(summary["cards_retained"], 4)
            self.assertEqual(summary["db_inserted"], summary["cards_retained"])

            with DatabaseManager(db_path=db_path) as db:
                consensus = db.get_consensus_prop_lines(
                    player_name="LeBron James", stat_type="points", side="over",
                )
                curry = db.get_consensus_prop_lines(
                    player_name="Stephen Curry", stat_type="assists", side="over",
                )

        self.assertEqual(len(consensus), 1)
        row = consensus[0]
        # PP=27.5, UD=28.5 → mean = 28.0; both books contributed.
        self.assertAlmostEqual(row["mean_line"], 28.0, places=3)
        self.assertEqual(row["n_books"], 2)
        self.assertIn("prizepicks", row["books"])
        self.assertIn("underdog", row["books"])

        self.assertEqual(len(curry), 1)
        # PP=6.5, UD=7.5 → mean = 7.0
        self.assertAlmostEqual(curry[0]["mean_line"], 7.0, places=3)
        self.assertEqual(curry[0]["n_books"], 2)

    def test_consensus_dedupes_same_book_to_latest_snapshot(self):
        """Two snapshots from the same book → only the newer line counts."""
        url = "https://app.prizepicks.com/board/nba"
        old_text = "LeBron James LAL Points 26.5 More Less"
        new_text = "LeBron James LAL Points 27.5 More Less"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_active_players(db, ["LeBron James"])
                # Older snapshot first.
                db.insert_web_text_snapshots(
                    [
                        {
                            "source_url": url,
                            "fetched_at_utc": "2026-04-13T00:00:00+00:00",
                            "http_status": 200,
                            "content_type": "text/html",
                            "text_content": old_text,
                            "text_length": len(old_text),
                            "content_sha256": "pp-old",
                        },
                        {
                            "source_url": url,
                            "fetched_at_utc": "2026-04-14T00:00:00+00:00",
                            "http_status": 200,
                            "content_type": "text/html",
                            "text_content": new_text,
                            "text_length": len(new_text),
                            "content_sha256": "pp-new",
                        },
                    ]
                )

            parse_and_store_web_prop_cards(
                db_path=db_path,
                source_urls=[url],
                max_snapshots_per_url=10,
                max_total_snapshots=20,
                min_parse_confidence=0.45,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.get_consensus_prop_lines(
                    player_name="LeBron James", stat_type="points", side="over",
                )

        self.assertEqual(len(rows), 1)
        # Only the latest 27.5 should count, not the average of 26.5+27.5.
        self.assertAlmostEqual(rows[0]["mean_line"], 27.5, places=3)
        self.assertEqual(rows[0]["n_books"], 1)

    def test_min_books_filter_excludes_single_source_lines(self):
        """min_books=2 drops props that only one book has posted."""
        url = "https://app.prizepicks.com/board/nba"
        text = "LeBron James LAL Points 27.5 More Less"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_active_players(db, ["LeBron James"])
                db.insert_web_text_snapshots(
                    [
                        {
                            "source_url": url,
                            "fetched_at_utc": _utc_now_iso(),
                            "http_status": 200,
                            "content_type": "text/html",
                            "text_content": text,
                            "text_length": len(text),
                            "content_sha256": "pp-only",
                        }
                    ]
                )

            parse_and_store_web_prop_cards(
                db_path=db_path,
                source_urls=[url],
                max_snapshots_per_url=1,
                max_total_snapshots=5,
                min_parse_confidence=0.45,
            )

            with DatabaseManager(db_path=db_path) as db:
                lenient = db.get_consensus_prop_lines(min_books=1)
                strict = db.get_consensus_prop_lines(min_books=2)

        self.assertEqual(len(lenient), 1)
        self.assertEqual(lenient[0]["n_books"], 1)
        self.assertEqual(strict, [])


if __name__ == "__main__":
    unittest.main()
