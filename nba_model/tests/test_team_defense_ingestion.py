import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nba_model.data.team_defense_ingestion import (
    build_team_defense_validation_report,
    populate_team_defense,
)


MOCK_TEAMS = [
    {"id": 1610612747, "abbreviation": "LAL", "full_name": "Los Angeles Lakers", "nickname": "Lakers"},
    {"id": 1610612738, "abbreviation": "BOS", "full_name": "Boston Celtics", "nickname": "Celtics"},
    {"id": 1610612752, "abbreviation": "NYK", "full_name": "New York Knicks", "nickname": "Knicks"},
]


class TeamDefenseIngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "nba_data.db")

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("nba_model.data.team_defense_ingestion.static_teams.get_teams")
    @patch("nba_model.data.team_defense_ingestion.fetch_team_defense_df")
    def test_validation_report_flags_missing_teams(self, mock_fetch_df, mock_get_teams):
        mock_get_teams.return_value = MOCK_TEAMS
        mock_fetch_df.return_value = pd.DataFrame(
            [
                {"team_abbrev": "LAL", "season": "2024-25", "def_rating": 112.1, "opp_ppg": 110.0, "pace": 99.1},
                {"team_abbrev": "BOS", "season": "2024-25", "def_rating": 108.4, "opp_ppg": 106.8, "pace": 98.7},
            ]
        )

        inserted = populate_team_defense(season="2024-25", db_path=self.db_path)
        report = build_team_defense_validation_report(season="2024-25", db_path=self.db_path)

        self.assertEqual(inserted, 2)
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["team_count"], 2)
        self.assertEqual(report["expected_team_count"], 3)
        self.assertEqual(report["missing_teams"], ["NYK"])
        self.assertEqual(report["unexpected_teams"], [])
        self.assertFalse(report["is_complete"])
        self.assertIsNotNone(report["latest_updated"])

    @patch("nba_model.data.team_defense_ingestion.static_teams.get_teams")
    @patch("nba_model.data.team_defense_ingestion.fetch_team_defense_df")
    def test_validation_report_is_complete_with_full_coverage(self, mock_fetch_df, mock_get_teams):
        mock_get_teams.return_value = MOCK_TEAMS
        mock_fetch_df.return_value = pd.DataFrame(
            [
                {"team_abbrev": "LAL", "season": "2024-25", "def_rating": 112.1, "opp_ppg": 110.0, "pace": 99.1},
                {"team_abbrev": "BOS", "season": "2024-25", "def_rating": 108.4, "opp_ppg": 106.8, "pace": 98.7},
                {"team_abbrev": "NYK", "season": "2024-25", "def_rating": 109.2, "opp_ppg": 107.4, "pace": 97.9},
            ]
        )

        inserted = populate_team_defense(season="2024-25", db_path=self.db_path)
        report = build_team_defense_validation_report(season="2024-25", db_path=self.db_path)

        self.assertEqual(inserted, 3)
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["team_count"], 3)
        self.assertEqual(report["missing_teams"], [])
        self.assertEqual(report["unexpected_teams"], [])
        self.assertTrue(report["is_complete"])


if __name__ == "__main__":
    unittest.main()
