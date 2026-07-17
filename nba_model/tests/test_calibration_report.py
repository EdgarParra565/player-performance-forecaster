"""Tests for the calibration / reliability report (WS10 Phase 1).

Covers the pure bucket + Brier math and the DB loaders for both sources
(``predictions`` and ``bet_log``), including push/pending exclusion. Offline,
Windows-safe teardown.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.evaluation import calibration_report as cr


class BucketMathTests(unittest.TestCase):
    def test_bucket_index_edges(self):
        self.assertEqual(cr._bucket_index(0.0, 10), 0)
        self.assertEqual(cr._bucket_index(0.05, 10), 0)
        self.assertEqual(cr._bucket_index(0.15, 10), 1)
        self.assertEqual(cr._bucket_index(0.999, 10), 9)
        self.assertEqual(cr._bucket_index(1.0, 10), 9)  # clamped into last bucket

    def test_reliability_table_groups_by_stat_and_bucket(self):
        df = pd.DataFrame([
            # points, bucket 6 (0.6-0.7): two rows, 1 hit → realized 0.5.
            {"stat_type": "points", "pred_prob": 0.62, "hit": 1},
            {"stat_type": "points", "pred_prob": 0.66, "hit": 0},
            # points, bucket 7 (0.7-0.8): one row, hit.
            {"stat_type": "points", "pred_prob": 0.72, "hit": 1},
            # assists, bucket 3: one row, miss.
            {"stat_type": "assists", "pred_prob": 0.35, "hit": 0},
        ])
        table = cr.build_reliability_table(df, n_buckets=10)
        pts6 = table[(table["stat_type"] == "points") & (table["bucket"] == 6)].iloc[0]
        self.assertEqual(pts6["n"], 2)
        self.assertAlmostEqual(pts6["mean_pred"], 0.64, places=4)
        self.assertAlmostEqual(pts6["realized_rate"], 0.5, places=4)
        self.assertAlmostEqual(pts6["calibration_gap"], 0.5 - 0.64, places=4)
        self.assertEqual(pts6["bucket_low"], 0.6)
        self.assertEqual(pts6["bucket_high"], 0.7)
        # 3 distinct (stat, bucket) groups.
        self.assertEqual(len(table), 3)

    def test_brier_by_stat(self):
        df = pd.DataFrame([
            {"stat_type": "points", "pred_prob": 1.0, "hit": 1},   # (1-1)^2 = 0
            {"stat_type": "points", "pred_prob": 0.0, "hit": 1},   # (0-1)^2 = 1
        ])
        brier = cr.brier_by_stat(df)
        row = brier[brier["stat_type"] == "points"].iloc[0]
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["brier_score"], 0.5, places=4)

    def test_empty_frame_returns_shaped_empty(self):
        out = cr.build_reliability_table(pd.DataFrame(), n_buckets=10)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), cr.RELIABILITY_COLUMNS)


class LoadCalibrationFrameTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_predictions_source_excludes_push_and_pending(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.executemany(
                """
                INSERT INTO predictions
                    (player_id, game_date, stat_type, predicted_mean,
                     predicted_std, prob_over, line_value, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (1, "2025-04-10", "points", 25.0, 5.0, 0.62, 24.5, "over"),
                    (1, "2025-04-11", "points", 22.0, 5.0, 0.40, 23.5, "under"),
                    (1, "2025-04-12", "points", 20.0, 5.0, 0.55, 20.0, "push"),
                    (1, "2025-04-13", "points", 20.0, 5.0, 0.55, 20.0, None),
                ],
            )
            db.conn.commit()
        frame = cr.load_calibration_frame(self.db_path, source="predictions")
        self.assertEqual(len(frame), 2)  # push + pending excluded
        self.assertEqual(set(frame["hit"]), {0, 1})

    def test_bet_log_source_maps_won_lost(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_bet_log_rows([
                {"game_date": "2025-04-10", "player_id": 1,
                 "player_name": "P", "stat_type": "points", "line": 24.5,
                 "side": "over", "model_prob": 0.62, "status": "won"},
                {"game_date": "2025-04-10", "player_id": 1,
                 "player_name": "P", "stat_type": "points", "line": 30.5,
                 "side": "over", "model_prob": 0.58, "status": "lost"},
                {"game_date": "2025-04-10", "player_id": 1,
                 "player_name": "P", "stat_type": "rebounds", "line": 8.0,
                 "side": "over", "model_prob": 0.55, "status": "push"},
                {"game_date": "2025-04-10", "player_id": 1,
                 "player_name": "P", "stat_type": "assists", "line": 6.5,
                 "side": "over", "model_prob": 0.60, "status": "pending"},
            ])
        frame = cr.load_calibration_frame(self.db_path, source="bet_log")
        self.assertEqual(len(frame), 2)  # push + pending excluded
        hits = dict(zip(frame["pred_prob"].round(2), frame["hit"]))
        self.assertEqual(hits[0.62], 1)
        self.assertEqual(hits[0.58], 0)

    def test_run_calibration_report_writes_artifacts(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.executemany(
                """
                INSERT INTO predictions
                    (player_id, game_date, stat_type, predicted_mean,
                     predicted_std, prob_over, line_value, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(1, "2025-04-10", "points", 25.0, 5.0, 0.62, 24.5, "over"),
                 (1, "2025-04-11", "points", 22.0, 5.0, 0.40, 23.5, "under")],
            )
            db.conn.commit()
        out_dir = Path(self._tmp.name) / "artifacts"
        result = cr.run_calibration_report(
            db_path=self.db_path, source="predictions",
            artifact_dir=str(out_dir))
        self.assertEqual(result["settled_rows"], 2)
        self.assertTrue(Path(result["reliability_csv"]).exists())
        self.assertTrue(Path(result["brier_csv"]).exists())
        self.assertTrue(Path(result["md_path"]).exists())


if __name__ == "__main__":
    unittest.main()
