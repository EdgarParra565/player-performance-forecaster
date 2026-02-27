import unittest

import pandas as pd

from nba_model.model.feature_engineering import add_context_features


class ContextFeaturesTests(unittest.TestCase):
    def test_add_context_features_generates_schedule_and_travel_fields(self):
        df = pd.DataFrame(
            {
                "game_date": ["2025-01-01", "2025-01-02", "2025-01-05", "2025-01-09"],
                "matchup": ["LAL vs. BOS", "LAL @ NYK", "LAL @ MIA", "LAL vs. PHX"],
                "minutes": [35, 34, 21, 33],
                "points": [28, 24, 9, 26],
            }
        )

        out = add_context_features(df, injury_window=3)

        self.assertIn("rest_days", out.columns)
        self.assertIn("is_back_to_back", out.columns)
        self.assertIn("travel_flag", out.columns)
        self.assertIn("games_last_7d", out.columns)
        self.assertIn("injury_proxy", out.columns)
        self.assertEqual(int(out.loc[1, "is_back_to_back"]), 1)
        self.assertEqual(int(out.loc[1, "travel_flag"]), 1)
        self.assertGreaterEqual(float(out.loc[2, "injury_proxy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
