import unittest

import numpy as np
import pandas as pd

from nba_model.model.correlation_calibration import calibrate_correlations, covariance_matrix


class CorrelationCalibrationTests(unittest.TestCase):
    def test_calibrate_returns_identity_when_sample_too_small(self):
        df = pd.DataFrame(
            {
                "points": [20, 22, 24],
                "assists": [5, 6, 5],
                "rebounds": [7, 8, 9],
            }
        )
        corr = calibrate_correlations(df, ["points", "assists", "rebounds"], min_games=10)

        self.assertEqual(corr.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(corr.values), np.ones(3)))
        self.assertTrue(np.allclose(corr.values, np.eye(3)))

    def test_calibrate_applies_shrinkage(self):
        np.random.seed(7)
        base = np.random.normal(0, 1, 200)
        df = pd.DataFrame(
            {
                "points": 25 + (4.0 * base) + np.random.normal(0, 0.3, 200),
                "assists": 7 + (2.0 * base) + np.random.normal(0, 0.3, 200),
                "rebounds": 9 - (1.5 * base) + np.random.normal(0, 0.3, 200),
            }
        )

        corr_low_shrink = calibrate_correlations(
            df,
            ["points", "assists", "rebounds"],
            min_games=8,
            shrinkage=0.0,
        )
        corr_high_shrink = calibrate_correlations(
            df,
            ["points", "assists", "rebounds"],
            min_games=8,
            shrinkage=0.6,
        )

        low_mag = abs(float(corr_low_shrink.loc["points", "assists"]))
        high_mag = abs(float(corr_high_shrink.loc["points", "assists"]))
        self.assertGreater(low_mag, high_mag)
        self.assertTrue(np.allclose(np.diag(corr_high_shrink.values), np.ones(3)))

    def test_covariance_matrix_requires_std_for_each_stat(self):
        corr = pd.DataFrame(
            [[1.0, 0.2], [0.2, 1.0]],
            index=["points", "assists"],
            columns=["points", "assists"],
        )
        with self.assertRaises(KeyError):
            covariance_matrix(corr, {"points": 5.0})

    def test_covariance_matrix_psd_projection(self):
        corr = pd.DataFrame(
            [
                [1.0, 0.99, 0.99],
                [0.99, 1.0, -0.99],
                [0.99, -0.99, 1.0],
            ],
            index=["points", "assists", "rebounds"],
            columns=["points", "assists", "rebounds"],
        )
        stds = {"points": 5.0, "assists": 2.0, "rebounds": 3.0}

        cov = covariance_matrix(corr, stds, ensure_psd=True)
        eigvals = np.linalg.eigvalsh((cov + cov.T) / 2.0)
        self.assertTrue(np.all(eigvals >= -1e-8))


if __name__ == "__main__":
    unittest.main()
