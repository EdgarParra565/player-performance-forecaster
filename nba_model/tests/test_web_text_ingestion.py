"""Unit tests for direct web text ingestion and active-player sync helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.web_text_ingestion import (
    fetch_and_store_web_text,
    load_urls_from_file,
    sync_active_nba_players_reference,
)


class WebTextIngestionTests(unittest.TestCase):
    @patch("nba_model.model.web_text_ingestion.requests.get")
    def test_fetch_and_store_reuses_recent_snapshot(self, mock_get):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.text = (
            "<html><body>DraftKings points line 27.5"
            "<script>ignore me</script><style>ignore me</style></body></html>"
        )
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            url = "https://example.com/odds-page"

            first = fetch_and_store_web_text(
                urls=[url],
                db_path=db_path,
                min_hours_between_polls=24.0,
                force_poll=False,
                request_retries=0,
            )
            second = fetch_and_store_web_text(
                urls=[url],
                db_path=db_path,
                min_hours_between_polls=24.0,
                force_poll=False,
                request_retries=0,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    "SELECT source_url, text_content FROM web_text_snapshots"
                ).fetchall()

        self.assertEqual(first["status"], "success")
        self.assertEqual(first["fetched_count"], 1)
        self.assertEqual(first["db_inserted"], 1)
        self.assertEqual(second["status"], "success")
        self.assertEqual(second["fetched_count"], 0)
        self.assertEqual(second["skipped_recent_count"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], url)
        self.assertIn("DraftKings points line 27.5", rows[0][1])
        self.assertNotIn("ignore me", rows[0][1])
        mock_get.assert_called_once()

    @patch("nba_model.model.web_text_ingestion.requests.get")
    def test_fetch_and_store_force_poll_ignores_window(self, mock_get):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/plain"}
        response.text = "Book text snapshot"
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            url = "https://example.com/book-lines"

            first = fetch_and_store_web_text(
                urls=[url],
                db_path=db_path,
                min_hours_between_polls=24.0,
                force_poll=False,
                request_retries=0,
            )
            second = fetch_and_store_web_text(
                urls=[url],
                db_path=db_path,
                min_hours_between_polls=24.0,
                force_poll=True,
                request_retries=0,
            )

            with DatabaseManager(db_path=db_path) as db:
                total_rows = db.conn.execute(
                    "SELECT COUNT(*) FROM web_text_snapshots"
                ).fetchone()[0]

        self.assertEqual(first["fetched_count"], 1)
        self.assertEqual(second["fetched_count"], 1)
        self.assertEqual(second["skipped_recent_count"], 0)
        self.assertEqual(total_rows, 2)
        self.assertEqual(mock_get.call_count, 2)

    @patch("nba_model.model.web_text_ingestion.requests.get")
    @patch("nba_model.model.web_text_ingestion._fetch_url_text_with_browser")
    def test_fetch_and_store_uses_browser_mode_with_session_flags(
        self,
        mock_browser_fetch,
        mock_get,
    ):
        mock_browser_fetch.return_value = {
            "source_url": "https://example.com/browser-page",
            "fetched_at_utc": "2026-02-27T00:00:00+00:00",
            "http_status": 200,
            "content_type": "text/html",
            "text_content": "Rendered browser text",
            "text_length": 21,
            "content_sha256": "abc123",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            user_data_dir = str(Path(tmpdir) / "browser_profile")
            summary = fetch_and_store_web_text(
                urls=["https://example.com/browser-page"],
                db_path=db_path,
                min_hours_between_polls=None,
                force_poll=True,
                request_retries=0,
                browser_user_data_dir=user_data_dir,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    "SELECT source_url, text_content FROM web_text_snapshots"
                ).fetchall()

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["fetch_mode"], "browser")
        self.assertEqual(summary["fetched_count"], 1)
        self.assertEqual(summary["db_inserted"], 1)
        self.assertEqual(rows, [("https://example.com/browser-page", "Rendered browser text")])
        mock_browser_fetch.assert_called_once()
        mock_get.assert_not_called()

    def test_load_urls_from_file_ignores_comments_and_blanks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            url_file = Path(tmpdir) / "urls.txt"
            url_file.write_text(
                "\n".join(
                    [
                        "# sportsbook pages",
                        "https://example.com/a",
                        "",
                        "https://example.com/b",
                        "not-a-url",
                    ]
                ),
                encoding="utf-8",
            )

            urls = load_urls_from_file(str(url_file))

        self.assertEqual(
            urls,
            ["https://example.com/a", "https://example.com/b"],
        )

    @patch("nba_model.model.web_text_ingestion.nba_players.get_active_players")
    def test_sync_active_players_reference_writes_db_and_file(self, mock_get_active_players):
        mock_get_active_players.return_value = [
            {"id": 1, "full_name": "Active Player 1"},
            {"id": 2, "full_name": "Active Player 2"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            output_file = str(Path(tmpdir) / "active_nba_players.txt")
            summary = sync_active_nba_players_reference(
                db_path=db_path,
                output_file=output_file,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    "SELECT player_id, player_name FROM nba_active_players_ref ORDER BY player_id ASC"
                ).fetchall()

            file_lines = Path(output_file).read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["players_synced"], 2)
        self.assertEqual(summary["db_attempted"], 2)
        self.assertGreaterEqual(summary["db_written"], 2)
        self.assertEqual(rows, [(1, "Active Player 1"), (2, "Active Player 2")])
        self.assertEqual(file_lines, ["Active Player 1", "Active Player 2"])


if __name__ == "__main__":
    unittest.main()
