"""Tests for the American-odds averaging fix + team-priors reverse-eng.

Two regression surfaces here:

1. **American-odds averaging.**  The naive ``AVG(odds_american)`` produces
   nonsense whenever the sample mixes + and -.  The fix routes through
   implied probability.  We test the conversion is round-trip-stable and
   that the consensus computes correctly on a synthetic 3-book sample.
2. **Team-priors derivation.**  Given a known set of (spread, total, ml)
   inputs, the priors module should produce the expected implied team
   totals and devigged win probabilities.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import (
    DatabaseManager,
    _american_to_implied_prob,
    _implied_prob_to_american,
)
from nba_model.model.team_line_reverse_engineering import (
    derive_team_priors_from_consensus,
)


class OddsConversionTests(unittest.TestCase):
    def test_zero_odds_returns_none(self):
        self.assertIsNone(_american_to_implied_prob(0))
        self.assertIsNone(_american_to_implied_prob(None))

    def test_pickem_round_trip(self):
        self.assertAlmostEqual(_american_to_implied_prob(100), 0.5, places=4)
        self.assertEqual(_implied_prob_to_american(0.5), 100)

    def test_favorite_round_trip(self):
        # -110 is the standard "vig" line
        p = _american_to_implied_prob(-110)
        self.assertAlmostEqual(p, 0.5238, places=3)
        self.assertEqual(_implied_prob_to_american(p), -110)

    def test_underdog_round_trip(self):
        p = _american_to_implied_prob(+150)
        self.assertAlmostEqual(p, 0.4, places=4)
        self.assertEqual(_implied_prob_to_american(p), +150)

    def test_round_trip_stable_across_range(self):
        for o in [-400, -200, -150, -110, -105, 100, 105, 110, 150, 200, 400]:
            p = _american_to_implied_prob(o)
            self.assertEqual(_implied_prob_to_american(p), o,
                             f"round-trip broke at {o}")


class TeamLineConsensusTests(unittest.TestCase):
    """End-to-end: insert 3-book team lines, check the consensus odds."""

    def _seed_three_book_moneyline(
        self, db_path: str, away="Knicks", home="76ers"
    ):
        """Insert a single (game, moneyline, away) line for 3 books with
        sign-mixed American odds — the case where naive averaging fails.
        """
        with DatabaseManager(db_path=db_path) as db:
            db.insert_web_text_snapshots([
                {
                    "source_url": f"https://book{i}.example.com/nba",
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "http_status": 200,
                    "content_type": "text/html",
                    "text_content": "x",
                    "text_length": 1,
                    "content_sha256": f"sha{i}",
                }
                for i in range(1, 4)
            ])
            db.insert_web_team_lines([
                {
                    "snapshot_id": i,
                    "source_url": f"https://book{i}.example.com/nba",
                    "book": f"book{i}",
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "away_team": away,
                    "home_team": home,
                    "market_type": "moneyline",
                    "side": "away",
                    "team": away,
                    "line_value": None,
                    "odds_american": odds,
                    "parse_confidence": 0.9,
                    "raw_text": "x",
                    "parser_version": "test",
                    "record_sha256": f"sha-{i}",
                }
                for i, odds in zip([1, 2, 3], [+100, -105, -108])
            ])

    def test_consensus_uses_implied_prob_not_raw_mean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            self._seed_three_book_moneyline(db_path)

            with DatabaseManager(db_path=db_path) as db:
                rows = db.get_consensus_team_lines(
                    away_team="Knicks", home_team="76ers",
                    market_type="moneyline", side="away",
                    min_books=2,
                )

        self.assertEqual(len(rows), 1)
        r = rows[0]
        # Raw American mean would have been (100 + -105 + -108)/3 = -37.7
        # — that's the bug.  The fixed path goes through implied prob:
        # p(+100)=0.5, p(-105)≈0.5122, p(-108)≈0.5192 → mean ≈ 0.5105
        # which round-trips to -104 American.
        self.assertAlmostEqual(r["mean_implied_prob"], 0.5105, places=3)
        self.assertEqual(r["mean_odds"], -104)
        # And the naive value is exposed for callers that want to compare.
        self.assertAlmostEqual(r["mean_odds_naive"], -37.66666666, places=2)
        self.assertEqual(r["n_books"], 3)


class TeamPriorsTests(unittest.TestCase):
    """End-to-end: full game (spread+total+ml) → priors."""

    def _seed_full_game(self, db_path: str):
        """Two books, identical Knicks vs 76ers: spread Knicks +1.5,
        total 213.5, ml Knicks +110 / 76ers -130."""
        now = datetime.now(timezone.utc).isoformat()
        with DatabaseManager(db_path=db_path) as db:
            db.insert_web_text_snapshots([
                {
                    "source_url": f"https://book{i}.example.com/nba",
                    "fetched_at_utc": now,
                    "http_status": 200,
                    "content_type": "text/html",
                    "text_content": "x",
                    "text_length": 1,
                    "content_sha256": f"sha{i}",
                }
                for i in range(1, 3)
            ])
            payload = []
            for snap_id, book in [(1, "book1"), (2, "book2")]:
                base = {
                    "snapshot_id": snap_id,
                    "source_url": f"https://book{snap_id}.example.com/nba",
                    "book": book,
                    "observed_at_utc": now,
                    "away_team": "Knicks",
                    "home_team": "76ers",
                    "parse_confidence": 0.9,
                    "raw_text": "x",
                    "parser_version": "test",
                }
                payload.extend([
                    {**base, "market_type": "spread",    "side": "away",  "team": "Knicks", "line_value":  1.5, "odds_american": -110, "record_sha256": f"{book}-sa"},
                    {**base, "market_type": "spread",    "side": "home",  "team": "76ers",  "line_value": -1.5, "odds_american": -110, "record_sha256": f"{book}-sh"},
                    {**base, "market_type": "total",     "side": "over",  "team": None,     "line_value": 213.5,"odds_american": -110, "record_sha256": f"{book}-to"},
                    {**base, "market_type": "total",     "side": "under", "team": None,     "line_value": 213.5,"odds_american": -110, "record_sha256": f"{book}-tu"},
                    {**base, "market_type": "moneyline", "side": "away",  "team": "Knicks", "line_value": None, "odds_american":  110, "record_sha256": f"{book}-ma"},
                    {**base, "market_type": "moneyline", "side": "home",  "team": "76ers",  "line_value": None, "odds_american": -130, "record_sha256": f"{book}-mh"},
                ])
            db.insert_web_team_lines(payload)

    def test_priors_derive_implied_totals_and_devig_winprob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            self._seed_full_game(db_path)

            summary = derive_team_priors_from_consensus(
                since_hours=24, min_books=2, db_path=db_path,
            )
            self.assertEqual(summary["games_with_full_priors"], 1)
            self.assertEqual(summary["db_upserted"], 1)

            with DatabaseManager(db_path=db_path) as db:
                prior = db.get_team_prior("Knicks", "76ers")

        self.assertIsNotNone(prior)
        # Implied team totals: home_total = (total - home_spread)/2
        #                                = (213.5 - (-1.5))/2 = 107.5
        # away_total = (total - away_spread)/2 = (213.5 - 1.5)/2 = 106.0
        self.assertAlmostEqual(prior["home_team_total"], 107.5, places=3)
        self.assertAlmostEqual(prior["away_team_total"], 106.0, places=3)
        # Devig win probs: book offers +110 / -130 → raw 0.4762 / 0.5652
        # → devig home = 0.5652/(0.5652+0.4762) ≈ 0.5427
        # → devig away = 1 - devig home ≈ 0.4573
        self.assertAlmostEqual(prior["home_win_prob_devig"], 0.5427, places=3)
        self.assertAlmostEqual(prior["away_win_prob_devig"], 0.4573, places=3)


if __name__ == "__main__":
    unittest.main()
