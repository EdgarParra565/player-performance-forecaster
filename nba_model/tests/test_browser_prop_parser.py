"""Unit tests for browser-style visible-text prop parsing."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.browser_prop_parser import (
    _canonicalize_stat,
    _infer_book_from_url,
    _preprocess_prizepicks_text,
    extract_prop_cards_from_text,
    parse_and_store_web_prop_cards,
)


class BrowserPropParserTests(unittest.TestCase):
    def test_extract_prop_cards_from_text_classifies_players(self):
        text = (
            "LeBron James Higher 27.5 Points "
            "Tom Brady Lower 0.5 Points"
        )
        cards = extract_prop_cards_from_text(
            text_content=text,
            source_url="https://app.underdogfantasy.com/pick-em/higher-lower/all/NBA",
            snapshot_id=1,
            observed_at_utc="2026-02-27T00:00:00+00:00",
            active_name_keys={"lebronjames"},
        )

        self.assertEqual(len(cards), 2)
        by_name = {row["player_name"]: row for row in cards}
        self.assertEqual(by_name["LeBron James"]["player_classification"], "active_nba")
        self.assertEqual(by_name["LeBron James"]["side"], "over")
        self.assertEqual(by_name["LeBron James"]["stat_type"], "points")
        self.assertEqual(by_name["LeBron James"]["book"], "underdog")
        self.assertEqual(by_name["Tom Brady"]["player_classification"], "non_nba")
        self.assertEqual(by_name["Tom Brady"]["side"], "under")

    def test_parse_and_store_web_prop_cards_inserts_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            source_url = "https://app.underdogfantasy.com/pick-em/higher-lower/all/NBA"
            with DatabaseManager(db_path=db_path) as db:
                db.upsert_active_players_reference(
                    [
                        {
                            "player_id": 2544,
                            "player_name": "LeBron James",
                            "synced_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                )
                db.insert_web_text_snapshots(
                    [
                        {
                            "source_url": source_url,
                            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                            "http_status": 200,
                            "content_type": "text/plain",
                            "text_content": (
                                "LeBron James Higher 27.5 Points "
                                "Tom Brady Lower 0.5 Points"
                            ),
                            "text_length": 64,
                            "content_sha256": "sha-one",
                        }
                    ]
                )

            first = parse_and_store_web_prop_cards(
                db_path=db_path,
                source_urls=[source_url],
                max_snapshots_per_url=1,
                max_total_snapshots=10,
                min_parse_confidence=0.45,
            )
            second = parse_and_store_web_prop_cards(
                db_path=db_path,
                source_urls=[source_url],
                max_snapshots_per_url=1,
                max_total_snapshots=10,
                min_parse_confidence=0.45,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    """
                    SELECT player_name, player_classification, stat_type, side
                    FROM web_prop_cards
                    ORDER BY player_name ASC
                    """
                ).fetchall()

        self.assertEqual(first["status"], "success")
        self.assertEqual(first["cards_extracted"], 2)
        self.assertEqual(first["cards_retained"], 2)
        self.assertEqual(first["db_inserted"], 2)
        self.assertEqual(second["db_inserted"], 0)
        self.assertEqual(
            rows,
            [
                ("LeBron James", "active_nba", "points", "over"),
                ("Tom Brady", "non_nba", "points", "under"),
            ],
        )

    def test_parse_and_store_web_prop_cards_skips_when_no_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            summary = parse_and_store_web_prop_cards(db_path=db_path)

        self.assertEqual(summary["status"], "skipped")
        self.assertEqual(summary["snapshots_considered"], 0)

    # --- PrizePicks-specific parsing ---

    def test_preprocess_prizepicks_text_extracts_cards(self):
        """_preprocess_prizepicks_text strips team context and returns player-stat-line-side."""
        text = (
            "LeBron James LAL Points 27.5 More Less "
            "Steph Curry GSW Assists 6.5 More Less"
        )
        preprocessed = _preprocess_prizepicks_text(text)
        self.assertIn("LeBron James", preprocessed)
        self.assertIn("27.5", preprocessed)
        self.assertIn("Steph Curry", preprocessed)
        self.assertIn("6.5", preprocessed)

    def test_extract_prop_cards_prizepicks_format(self):
        """extract_prop_cards_from_text handles PrizePicks-style 'Player TEAM Stat Line More Less' text."""
        text = (
            "NBA Projections "
            "LeBron James LAL Points 27.5 More Less "
            "Nikola Jokic DEN Rebounds 12.5 More Less "
            "Stephen Curry GSW Assists 6.5 More Less"
        )
        active_keys = {"lebronjames", "nikolajokic", "stephencurry"}
        cards = extract_prop_cards_from_text(
            text_content=text,
            source_url="https://app.prizepicks.com/board/nba",
            snapshot_id=10,
            observed_at_utc="2026-04-14T00:00:00+00:00",
            active_name_keys=active_keys,
        )
        names = {c["player_name"] for c in cards}
        self.assertIn("LeBron James", names)
        self.assertIn("Nikola Jokic", names)
        self.assertIn("Stephen Curry", names)
        by_name = {c["player_name"]: c for c in cards}
        self.assertEqual(by_name["LeBron James"]["stat_type"], "points")
        self.assertEqual(by_name["LeBron James"]["line_value"], 27.5)
        self.assertEqual(by_name["Nikola Jokic"]["stat_type"], "rebounds")
        self.assertEqual(by_name["Stephen Curry"]["stat_type"], "assists")
        for c in cards:
            self.assertEqual(c["book"], "prizepicks")
            self.assertEqual(c["player_classification"], "active_nba")

    def test_extract_prop_cards_prizepicks_login_wall_yields_nothing(self):
        """extract_prop_cards_from_text produces no valid cards from a login wall."""
        text = (
            "PrizePicks Enter Your Phone Number Enter phone number "
            "Log In Sign Up Verify your identity Create a new account"
        )
        cards = extract_prop_cards_from_text(
            text_content=text,
            source_url="https://app.prizepicks.com/board/nba",
            snapshot_id=99,
            observed_at_utc="2026-04-14T00:00:00+00:00",
            active_name_keys=set(),
        )
        self.assertEqual(cards, [])

    def test_infer_book_from_url_prizepicks(self):
        """_infer_book_from_url maps prizepicks.com correctly."""
        self.assertEqual(
            _infer_book_from_url("https://app.prizepicks.com/board/nba"),
            "prizepicks",
        )
        self.assertEqual(
            _infer_book_from_url("https://app.underdogfantasy.com/pick-em"),
            "underdog",
        )

    def test_canonicalize_stat_prizepicks_labels(self):
        """_canonicalize_stat handles PrizePicks display labels correctly."""
        self.assertEqual(_canonicalize_stat("Points"), "points")
        self.assertEqual(_canonicalize_stat("Rebounds"), "rebounds")
        self.assertEqual(_canonicalize_stat("Assists"), "assists")
        self.assertEqual(_canonicalize_stat("3PM"), "three_pointers_made")
        self.assertEqual(_canonicalize_stat("Fantasy Points"), "fantasy_points")
        self.assertIsNone(_canonicalize_stat("Unknown Stat"))

    def test_parse_and_store_prizepicks_full_flow(self):
        """Full DB round-trip: store PrizePicks snapshot then parse prop cards."""
        source_url = "https://app.prizepicks.com/board/nba"
        snapshot_text = (
            "NBA Projections "
            "LeBron James LAL Points 27.5 More Less "
            "Stephen Curry GSW Assists 6.5 More Less "
            "Nikola Jokic DEN Rebounds 12.5 More Less"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")

            with DatabaseManager(db_path=db_path) as db:
                db.upsert_active_players_reference(
                    [
                        {"player_id": 2544, "player_name": "LeBron James",
                         "synced_at_utc": datetime.now(timezone.utc).isoformat()},
                        {"player_id": 201939, "player_name": "Stephen Curry",
                         "synced_at_utc": datetime.now(timezone.utc).isoformat()},
                        {"player_id": 203999, "player_name": "Nikola Jokic",
                         "synced_at_utc": datetime.now(timezone.utc).isoformat()},
                    ]
                )
                db.insert_web_text_snapshots(
                    [
                        {
                            "source_url": source_url,
                            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                            "http_status": 200,
                            "content_type": "text/html",
                            "text_content": snapshot_text,
                            "text_length": len(snapshot_text),
                            "content_sha256": "pp-test-sha",
                        }
                    ]
                )

            summary = parse_and_store_web_prop_cards(
                db_path=db_path,
                source_urls=[source_url],
                max_snapshots_per_url=1,
                max_total_snapshots=10,
                min_parse_confidence=0.45,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    """
                    SELECT player_name, stat_type, line_value, player_classification, book
                    FROM web_prop_cards
                    ORDER BY player_name ASC
                    """
                ).fetchall()

        self.assertEqual(summary["status"], "success")
        self.assertGreaterEqual(summary["cards_retained"], 3)
        self.assertEqual(summary["db_inserted"], summary["cards_retained"])
        player_names = {r[0] for r in rows}
        self.assertIn("LeBron James", player_names)
        self.assertIn("Stephen Curry", player_names)
        self.assertIn("Nikola Jokic", player_names)
        for row in rows:
            self.assertEqual(row[3], "active_nba")
            self.assertEqual(row[4], "prizepicks")


if __name__ == "__main__":
    unittest.main()
