"""Unit tests for odds ingestion validation, retries, and summaries."""

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


class OddsApiAuthErrorTests(unittest.TestCase):
    """Review finding: HTTP 401/403/429 were swallowed into
    "success, events_processed 0"."""

    def _http_error_response(self, code):
        resp = MagicMock(status_code=code, text='{"message":"quota"}')
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"{code} Client Error for url: https://api.the-odds-api.com/x?apiKey=SECRET",
            response=resp)
        return resp

    @patch("nba_model.model.odds_ingestion.time.sleep")
    @patch("nba_model.model.odds_ingestion.requests.get")
    def test_auth_and_quota_codes_raise_typed_error_without_retry(self, mock_get, _sleep):
        from nba_model.model.odds_ingestion import OddsApiAuthError

        for code in (401, 403, 429):
            mock_get.reset_mock()
            mock_get.return_value = self._http_error_response(code)
            with self.assertRaises(OddsApiAuthError) as ctx:
                _get_json("https://example.com", params={}, retries=3, retry_delay_seconds=0)
            self.assertEqual(ctx.exception.status_code, code)
            self.assertFalse(ctx.exception.retryable)
            self.assertNotIn("SECRET", str(ctx.exception))  # key still redacted
            self.assertEqual(mock_get.call_count, 1)        # no retries
            self.assertIsInstance(ctx.exception, requests.HTTPError)

    @patch("nba_model.model.odds_ingestion.time.sleep")
    @patch("nba_model.model.odds_ingestion.requests.get")
    def test_server_errors_still_retry(self, mock_get, _sleep):
        mock_get.return_value = self._http_error_response(500)
        with self.assertRaises(requests.HTTPError):
            _get_json("https://example.com", params={}, retries=2, retry_delay_seconds=0)
        self.assertEqual(mock_get.call_count, 3)

    @patch("nba_model.model.odds_ingestion.fetch_events_from_sport_odds")
    @patch("nba_model.model.odds_ingestion.fetch_events")
    @patch("nba_model.model.odds_ingestion.DatabaseManager")
    def test_discovery_auth_error_fails_instead_of_zero_events(self, mock_db, mock_events, mock_fallback):
        from nba_model.model.odds_ingestion import OddsApiAuthError

        mock_events.side_effect = OddsApiAuthError("HTTP 401", 401)
        with self.assertRaises(OddsApiAuthError):
            fetch_and_store_betting_lines(api_key="bad", sleep_seconds=0.0, request_retries=0)
        mock_fallback.assert_not_called()
        mock_db.assert_not_called()

    @patch("nba_model.model.odds_ingestion.normalize_event_player_props")
    @patch("nba_model.model.odds_ingestion.fetch_event_player_props")
    @patch("nba_model.model.odds_ingestion.fetch_events")
    @patch("nba_model.model.odds_ingestion.DatabaseManager")
    def test_quota_mid_run_stores_parsed_events_then_fails(
        self, mock_db_cls, mock_events, mock_event_props, mock_normalize,
    ):
        from nba_model.model.odds_ingestion import OddsApiAuthError

        mock_events.return_value = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]
        mock_event_props.side_effect = [{"bookmakers": []}, OddsApiAuthError("HTTP 429", 429),
                                        {"bookmakers": []}]
        mock_normalize.return_value = ([{
            "player_id": 2544, "player_name": "LeBron James", "game_date": "2025-01-15",
            "book": "FanDuel", "stat_type": "points", "line_value": 27.5,
            "over_odds": -110, "under_odds": -110,
        }], [])
        db = MagicMock()
        db.insert_betting_lines_records.return_value = {"attempted": 1, "inserted": 1,
                                                        "duplicates_ignored": 0}
        mock_db_cls.return_value.__enter__.return_value = db
        with self.assertRaises(OddsApiAuthError) as ctx:
            fetch_and_store_betting_lines(api_key="k", sleep_seconds=0.0, request_retries=0)
        self.assertEqual(mock_event_props.call_count, 2)       # stopped at the 429
        db.insert_betting_lines_records.assert_called_once()   # event 1 kept
        self.assertEqual(ctx.exception.partial_summary["db_inserted"], 1)


if __name__ == "__main__":
    unittest.main()
