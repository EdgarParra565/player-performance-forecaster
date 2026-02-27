import unittest
from unittest.mock import MagicMock, patch

import requests

from nba_model.model.odds_ingestion import (
    _get_json,
    fetch_and_store_betting_lines,
    validate_betting_line_records,
)


class OddsIngestionValidationTests(unittest.TestCase):
    def test_validate_betting_line_records_filters_invalid_rows(self):
        records = [
            {
                "player_id": 2544,
                "player_name": "LeBron James",
                "game_date": "2025-01-15T00:00:00Z",
                "book": "FanDuel",
                "stat_type": "points",
                "line_value": 27.5,
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "player_id": None,
                "player_name": "Broken Row",
                "game_date": "not-a-date",
                "book": "",
                "stat_type": "unknown_market",
                "line_value": None,
                "over_odds": "not-int",
                "under_odds": None,
            },
        ]

        valid, summary = validate_betting_line_records(records)

        self.assertEqual(len(valid), 1)
        self.assertEqual(summary["records_received"], 2)
        self.assertEqual(summary["records_valid"], 1)
        self.assertEqual(summary["records_invalid"], 1)
        self.assertGreater(summary["invalid_reason_counts"].get("invalid_player_id", 0), 0)
        self.assertEqual(valid[0]["game_date"], "2025-01-15")


class OddsIngestionRetryTests(unittest.TestCase):
    @patch("nba_model.model.odds_ingestion.time.sleep")
    @patch("nba_model.model.odds_ingestion.requests.get")
    def test_get_json_retries_then_succeeds(self, mock_get, mock_sleep):
        first_error = requests.ConnectionError("temporary connection error")
        second_response = MagicMock()
        second_response.raise_for_status.return_value = None
        second_response.json.return_value = {"ok": True}
        mock_get.side_effect = [first_error, second_response]

        payload = _get_json(
            "https://example.com/test",
            params={"a": 1},
            timeout=1,
            retries=1,
            retry_delay_seconds=0.01,
            retry_backoff=1.0,
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()


class OddsIngestionSummaryTests(unittest.TestCase):
    @patch("nba_model.model.odds_ingestion.normalize_event_player_props")
    @patch("nba_model.model.odds_ingestion.fetch_event_player_props")
    @patch("nba_model.model.odds_ingestion.fetch_events")
    @patch("nba_model.model.odds_ingestion.DatabaseManager")
    def test_fetch_and_store_reports_validation_and_duplicates(
        self,
        mock_db_cls,
        mock_fetch_events,
        mock_fetch_event_props,
        mock_normalize_event,
    ):
        mock_fetch_events.return_value = [{"id": "event_1"}]
        mock_fetch_event_props.return_value = {"bookmakers": []}

        mock_normalize_event.return_value = (
            [
                {
                    "player_id": 2544,
                    "player_name": "LeBron James",
                    "game_date": "2025-01-15",
                    "book": "FanDuel",
                    "stat_type": "points",
                    "line_value": 27.5,
                    "over_odds": -110,
                    "under_odds": -110,
                },
                {
                    "player_id": 2544,
                    "player_name": "LeBron James",
                    "game_date": "2025-01-15",
                    "book": "FanDuel",
                    "stat_type": "points",
                    "line_value": 27.5,
                    "over_odds": -110,
                    "under_odds": -110,
                },
                {
                    "player_id": 2544,
                    "player_name": "LeBron James",
                    "game_date": "2025-01-15",
                    "book": "FanDuel",
                    "stat_type": "points",
                    "line_value": None,
                    "over_odds": -110,
                    "under_odds": -110,
                },
            ],
            ["Unknown Player"],
        )

        db = MagicMock()
        db.insert_betting_lines_records.return_value = {
            "attempted": 1,
            "inserted": 1,
            "duplicates_ignored": 0,
        }
        mock_db_cls.return_value.__enter__.return_value = db

        summary = fetch_and_store_betting_lines(
            api_key="dummy",
            sleep_seconds=0.0,
            request_retries=0,
        )

        self.assertEqual(summary["records_parsed"], 3)
        self.assertEqual(summary["records_valid"], 2)
        self.assertEqual(summary["records_invalid"], 1)
        self.assertEqual(summary["duplicates_in_payload"], 1)
        self.assertEqual(summary["db_attempted"], 1)
        self.assertEqual(summary["db_inserted"], 1)
        self.assertEqual(summary["db_duplicates_ignored"], 0)
        self.assertEqual(summary["distinct_players"], 1)
        self.assertEqual(summary["unresolved_player_names"], ["Unknown Player"])
        db.insert_player.assert_called_once_with(2544, "LeBron James")


if __name__ == "__main__":
    unittest.main()
