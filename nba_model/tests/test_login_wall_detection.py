"""Tests for login-wall / session-expiry detection in the parse path (task 2d).

A snapshot that is really a login wall must be skipped by the parsers instead
of having UI junk scraped into ``web_prop_cards`` / ``web_team_lines``.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.browser_prop_parser import parse_and_store_web_prop_cards
from nba_model.model.team_line_parser import parse_and_store_web_team_lines
from nba_model.model.web_text_ingestion import detect_login_wall


def _snap(url, text, sha):
    return {
        "source_url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": 200,
        "content_type": "text/html",
        "text_content": text,
        "text_length": len(text),
        "content_sha256": sha,
    }


class DetectLoginWallTests(unittest.TestCase):
    PP_URL = "https://app.prizepicks.com/board/nba"

    def test_known_book_login_phrase_flags_wall(self):
        is_wall, reason = detect_login_wall("Please Log in to PrizePicks now", self.PP_URL)
        self.assertTrue(is_wall)
        self.assertIn("Login-wall phrase", reason)

    def test_known_book_content_is_not_wall(self):
        text = "Stephen Curry 25.5 Points More LeBron James 7.5 Assists Less"
        is_wall, reason = detect_login_wall(text, self.PP_URL)
        self.assertFalse(is_wall)
        self.assertIsNone(reason)

    def test_generic_fallback_for_unknown_host(self):
        is_wall, reason = detect_login_wall(
            "sign in sign in sign in to continue", "https://unknown-book.example.com/x"
        )
        self.assertTrue(is_wall)
        self.assertIn("Generic login-wall", reason)

    def test_empty_text_is_wall(self):
        is_wall, reason = detect_login_wall("", self.PP_URL)
        self.assertTrue(is_wall)


class PropParserSkipsLoginWallTests(unittest.TestCase):
    PP_URL = "https://app.prizepicks.com/board/nba"
    # Contains a real prizepicks login-wall phrase AND text the generic card
    # regex would otherwise scrape into web_prop_cards.
    WALL_TEXT = (
        "Log in to PrizePicks Stephen Curry 25.5 Points More "
        "LeBron James 7.5 Assists Less"
    )

    def test_login_walled_snapshot_inserts_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([_snap(self.PP_URL, self.WALL_TEXT, "w1")])

            summary = parse_and_store_web_prop_cards(
                db_path=db_path, source_urls=[self.PP_URL],
                max_snapshots_per_url=1, max_total_snapshots=10,
            )

            with DatabaseManager(db_path=db_path) as db:
                count = db.conn.execute("SELECT COUNT(*) FROM web_prop_cards").fetchone()[0]

        self.assertEqual(summary["snapshots_login_walled"], 1)
        self.assertEqual(summary["cards_extracted"], 0)
        self.assertEqual(summary["db_inserted"], 0)
        self.assertEqual(count, 0)


class TeamLineParserSkipsLoginWallTests(unittest.TestCase):
    MGM_URL = "https://sports.betmgm.com/en/sports/basketball-7"
    # A REAL bare wall: generic login nav + NO game-line content (0 authenticated
    # markers), so the detector's generic-nav rule flags it. (A page that shows
    # "Log In" nav *next to* real lines is content, not a wall — see below.)
    WALL_TEXT = (
        "Log in Sign up Create account Welcome back — please log in to your "
        "account to continue. Log in Sign up"
    )
    # Content with a persistent "Log in" nav link AND real game lines: NOT a wall.
    CONTENT_TEXT = (
        "Log in Sign up Today • 7:10 PM Spread Total Money Knicks 53-29 76ers 45-37 "
        "+1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"
    )

    def test_bare_login_wall_inserts_no_team_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([_snap(self.MGM_URL, self.WALL_TEXT, "w1")])

            summary = parse_and_store_web_team_lines(
                db_path=db_path, source_urls=[self.MGM_URL],
                max_snapshots_per_url=1, max_total_snapshots=10,
            )
            with DatabaseManager(db_path=db_path) as db:
                count = db.conn.execute("SELECT COUNT(*) FROM web_team_lines").fetchone()[0]

        self.assertEqual(summary["snapshots_login_walled"], 1)
        self.assertEqual(summary["db_inserted"], 0)
        self.assertEqual(count, 0)

    def test_content_with_nav_login_is_not_walled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_web_text_snapshots([_snap(self.MGM_URL, self.CONTENT_TEXT, "c1")])

            summary = parse_and_store_web_team_lines(
                db_path=db_path, source_urls=[self.MGM_URL],
                max_snapshots_per_url=1, max_total_snapshots=10,
            )
        # Persistent nav "Log in" must NOT suppress a real, content-rich page.
        self.assertEqual(summary["snapshots_login_walled"], 0)
        self.assertEqual(summary["db_inserted"], 6)  # spread/total/ML × 2 sides


if __name__ == "__main__":
    unittest.main()
