"""Unit tests for browser-style visible-text prop parsing."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.browser_prop_parser import (
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


if __name__ == "__main__":
    unittest.main()
