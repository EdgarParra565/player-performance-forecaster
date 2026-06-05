"""Unit tests for the shared manual-line parser used by both the Tk desktop
UI and the upcoming Streamlit "Manual Lines Import" view.

Tests target the public API in ``nba_model.model.manual_lines`` directly so
the Streamlit view can rely on the same signature without re-deriving the
heuristics."""

import unittest
from unittest.mock import MagicMock

from nba_model.model.manual_lines import (
    normalize_game_date,
    normalize_stat_type,
    parse_manual_lines_text,
    persist_manual_lines_records,
    resolve_player_identity,
)


DEFAULT_DATE = "2025-03-15"
DEFAULT_BOOK = "test_book"


class NormalizeHelpersTests(unittest.TestCase):
    def test_normalize_stat_type_aliases(self):
        self.assertEqual(normalize_stat_type("PTS"), "points")
        self.assertEqual(normalize_stat_type("Assists"), "assists")
        self.assertEqual(normalize_stat_type("reb"), "rebounds")
        self.assertEqual(normalize_stat_type("PRA"), "pra")
        self.assertEqual(
            normalize_stat_type("Points Rebounds Assists"), "pra")

    def test_normalize_stat_type_unknown_raises_without_allow_custom(self):
        with self.assertRaises(ValueError):
            normalize_stat_type("blocks")

    def test_normalize_stat_type_unknown_returns_slug_when_allowed(self):
        self.assertEqual(normalize_stat_type("3PM made", allow_custom=True), "3pm_made")

    def test_normalize_game_date_accepts_multiple_formats(self):
        self.assertEqual(normalize_game_date("2025-03-15"), "2025-03-15")
        self.assertEqual(normalize_game_date("2025/03/15"), "2025-03-15")
        self.assertEqual(normalize_game_date("03/15/2025"), "2025-03-15")

    def test_normalize_game_date_rejects_garbage(self):
        with self.assertRaises(ValueError):
            normalize_game_date("not-a-date")
        with self.assertRaises(ValueError):
            normalize_game_date("")

    def test_resolve_player_identity_returns_synthetic_for_unknown(self):
        identity = resolve_player_identity("Some Non NBA Player", allow_synthetic=True)
        self.assertGreater(identity["player_id"], 900_000_000)
        self.assertEqual(identity["player_name"], "Some Non NBA Player")

    def test_resolve_player_identity_raises_when_synthetic_disabled(self):
        with self.assertRaises(ValueError):
            resolve_player_identity("Definitely Not An NBA Name 12345", allow_synthetic=False)

    def test_resolve_player_identity_finds_known_nba_player(self):
        identity = resolve_player_identity("LeBron James")
        self.assertEqual(identity["player_name"], "LeBron James")
        self.assertEqual(identity["player_id"], 2544)


class ParseManualLinesPipeFormatTests(unittest.TestCase):
    def test_pipe_row_with_3_fields_uses_defaults(self):
        text = "LeBron James | points | 25.5"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["player_name"], "LeBron James")
        self.assertEqual(rec["stat_type"], "points")
        self.assertEqual(rec["line_value"], 25.5)
        self.assertEqual(rec["game_date"], DEFAULT_DATE)
        self.assertEqual(rec["book"], DEFAULT_BOOK)
        self.assertIsNone(rec["over_odds"])
        self.assertIsNone(rec["under_odds"])

    def test_pipe_row_with_odds_parses_american(self):
        text = "Stephen Curry | assists | 7.5 | -120 | +105"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(records[0]["over_odds"], -120)
        self.assertEqual(records[0]["under_odds"], 105)

    def test_pipe_row_with_explicit_date_book(self):
        text = "Nikola Jokic | 2025-03-10 | draftkings | rebounds | 11.5"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        rec = records[0]
        self.assertEqual(rec["game_date"], "2025-03-10")
        self.assertEqual(rec["book"], "draftkings")
        self.assertEqual(rec["stat_type"], "rebounds")
        self.assertEqual(rec["line_value"], 11.5)


class ParseManualLinesCsvAndTsvTests(unittest.TestCase):
    def test_csv_row(self):
        text = "LeBron James, points, 27.5, -110, -110"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["over_odds"], -110)
        self.assertEqual(records[0]["under_odds"], -110)

    def test_tab_separated_row(self):
        text = "Luka Doncic\tpra\t45.5"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(records[0]["stat_type"], "pra")
        self.assertEqual(records[0]["line_value"], 45.5)

    def test_header_row_is_skipped(self):
        text = (
            "player_name, stat_type, line_value\n"
            "LeBron James, points, 25.5\n"
        )
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["player_name"], "LeBron James")


class ParseManualLinesBoardStyleTests(unittest.TestCase):
    """The board-style fallback handles noisy PrizePicks-style dumps where
    each prop spans multiple lines: ``Player\\nvs OPP\\n<line>\\n<stat>``."""

    def test_simple_board_dump(self):
        text = "\n".join([
            "LeBron James",
            "vs DEN",
            "25.5",
            "Points",
            "Stephen Curry",
            "vs LAL",
            "7.5",
            "Assists",
        ])
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        names = sorted(r["player_name"] for r in records)
        self.assertEqual(names, ["LeBron James", "Stephen Curry"])
        stats = {r["player_name"]: r["stat_type"] for r in records}
        self.assertEqual(stats["LeBron James"], "points")
        self.assertEqual(stats["Stephen Curry"], "assists")

    def test_board_dedupes_repeats(self):
        text = "\n".join([
            "LeBron James", "vs DEN", "25.5", "Points",
            "LeBron James", "vs DEN", "25.5", "Points",
        ])
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)

    def test_board_filters_noise_lines(self):
        text = "\n".join([
            "Refresh Board",
            "Trending",
            "LeBron James",
            "vs DEN",
            "25.5",
            "Points",
            "Promotions",
        ])
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["player_name"], "LeBron James")

    def test_board_strips_goblin_demon_modifiers(self):
        text = "\n".join([
            "LeBron JamesGoblin",
            "vs DEN",
            "25.5",
            "Points",
        ])
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        self.assertEqual(records[0]["player_name"], "LeBron James")


class ParseManualLinesErrorTests(unittest.TestCase):
    def test_invalid_line_value_emits_error(self):
        text = "LeBron James | points | not_a_number"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(records, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("not_a_number", errors[0])

    def test_empty_text_returns_nothing(self):
        records, errors = parse_manual_lines_text("", DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(records, [])
        self.assertEqual(errors, [])

    def test_blank_default_book_falls_back_to_manual_ui(self):
        text = "LeBron James | points | 25.5"
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, "")
        self.assertEqual(errors, [])
        self.assertEqual(records[0]["book"], "manual_ui")

    def test_mixed_delimiter_and_board_text(self):
        text = (
            "LeBron James | points | 25.5\n"
            "Stephen Curry\nvs LAL\n7.5\nAssists\n"
        )
        records, errors = parse_manual_lines_text(text, DEFAULT_DATE, DEFAULT_BOOK)
        self.assertEqual(errors, [])
        names = sorted(r["player_name"] for r in records)
        self.assertEqual(names, ["LeBron James", "Stephen Curry"])


class PersistManualLinesRecordsTests(unittest.TestCase):
    def test_persist_passes_records_to_db(self):
        fake_db = MagicMock()
        fake_db.insert_betting_lines_records.return_value = {
            "inserted": 2, "duplicates_ignored": 0, "attempted": 2,
        }
        records = [
            {"player_id": 1, "player_name": "A", "game_date": "2025-01-01",
             "book": "b", "stat_type": "points", "line_value": 10.5,
             "over_odds": None, "under_odds": None},
        ]
        result = persist_manual_lines_records(records, db=fake_db)
        self.assertEqual(result["inserted"], 2)
        fake_db.insert_betting_lines_records.assert_called_once_with(records)

    def test_persist_returns_empty_summary_for_empty_input(self):
        result = persist_manual_lines_records([])
        self.assertEqual(result, {"inserted": 0, "duplicates_ignored": 0, "attempted": 0})


if __name__ == "__main__":
    unittest.main()
