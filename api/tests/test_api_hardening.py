"""Regression tests for the API hardening pass (review_findings_web.md, W-API-*).

Covers: read-only preflight (no schema writes into a foreign/empty file),
sqlite errors -> 503, input bounds (regex search, player_id, NaN/inf floats,
book list caps, season_type), name-param trust, 404 parity, public health
body, null KPIs on empty windows, UTC-normalised freshness timestamps.
"""
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.tests.test_api_phase2 import LEBRON_ID, _seed


class HardeningTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls._tmp.name) / "nba.db")
        _seed(cls.db_path)
        os.environ["NBA_DB_PATH"] = cls.db_path
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("NBA_DB_PATH", None)
        cls._tmp.cleanup()

    def setUp(self):
        os.environ["NBA_DB_PATH"] = self.db_path

    # --- read-only guarantee -------------------------------------------------

    def test_empty_file_is_not_mounted_503_and_not_written(self):
        # A 0-byte file is what a broken/empty mount looks like.
        empty = Path(self._tmp.name) / "empty.db"
        empty.write_bytes(b"")
        os.environ["NBA_DB_PATH"] = str(empty)
        for path in ("/api/meta", "/api/slate/kpis", f"/api/players/{LEBRON_ID}"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 503, path)
            self.assertEqual(r.json()["code"], "db_not_mounted")
            self.assertIn("database file not mounted", r.json()["detail"])
        # The data layer never opened it -> no schema was created inside.
        self.assertEqual(empty.stat().st_size, 0)

    def test_missing_file_on_empty_mount_dir(self):
        # `docker run` without -v: /data exists but holds nothing.
        mount = Path(self._tmp.name) / "data"
        mount.mkdir()
        os.environ["NBA_DB_PATH"] = str(mount / "nba_data.db")
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 503)  # container HEALTHCHECK fails
        body = h.json()
        self.assertEqual((body["db_state"], body["code"], body["db_exists"]),
                         ("not_mounted", "db_not_mounted", False))
        self.assertIn("docker compose up flagship", body["detail"])
        r = self.client.get("/api/slate/kpis")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["code"], "db_not_mounted")
        self.assertEqual(list(mount.iterdir()), [])  # nothing created, no schema grown
        self.assertNotIn(self._tmp.name, h.text + r.text)

    def test_garbage_file_is_invalid_503(self):
        junk = Path(self._tmp.name) / "junk.db"
        junk.write_bytes(b"not a sqlite database at all" * 100)
        os.environ["NBA_DB_PATH"] = str(junk)
        r = self.client.get("/api/meta")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json(), {"detail": "database unavailable", "code": "db_invalid"})
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 503)
        self.assertEqual((h.json()["status"], h.json()["db_state"]), ("degraded", "invalid"))

    def test_missing_db_detail_has_no_path(self):
        os.environ["NBA_DB_PATH"] = str(Path(self._tmp.name) / "nope.db")
        r = self.client.get("/api/meta")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn(self._tmp.name, r.text)

    def test_healthy_db_reports_ok(self):
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        self.assertEqual((h.json()["db_state"], h.json()["code"]), ("ok", None))

    # --- public health body ------------------------------------------------------

    def test_health_hides_db_path_by_default(self):
        body = self.client.get("/api/health").json()
        self.assertIsNone(body["db_path"])
        self.assertTrue(body["db_exists"])

    def test_health_exposes_db_path_when_opted_in(self):
        os.environ["API_EXPOSE_DB_PATH"] = "1"
        try:
            body = self.client.get("/api/health").json()
        finally:
            os.environ.pop("API_EXPOSE_DB_PATH", None)
        self.assertEqual(body["db_path"], self.db_path)

    # --- input bounds -------------------------------------------------------

    def test_search_is_literal_not_regex(self):
        r = self.client.get("/api/players/search", params={"q": "("})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 0)
        # '.' must not act as a wildcard.
        r = self.client.get("/api/players/search", params={"q": "l.bron"})
        self.assertEqual(r.json()["count"], 0)

    def test_search_query_length_capped(self):
        r = self.client.get("/api/players/search", params={"q": "x" * 65})
        self.assertEqual(r.status_code, 422)

    def test_player_id_bounds(self):
        self.assertEqual(self.client.get("/api/players/9223372036854775808").status_code, 422)
        self.assertEqual(self.client.get("/api/players/0").status_code, 422)

    def test_non_finite_floats_rejected(self):
        for q in ("min_edge=nan", "min_edge=inf", "min_p_over=nan", "min_p_over=2"):
            self.assertEqual(self.client.get(f"/api/slate/edges?{q}").status_code, 422, q)
        self.assertEqual(self.client.get("/api/cross-book?min_gap=inf").status_code, 422)

    def test_books_list_capped(self):
        many = "&".join(f"books=b{i}" for i in range(30))
        self.assertEqual(self.client.get(f"/api/slate/edges?{many}").status_code, 400)
        long_name = "x" * 41
        self.assertEqual(self.client.get(f"/api/cross-book?books={long_name}").status_code, 400)

    def test_season_type_allowlisted(self):
        r = self.client.get("/api/slate/recent-games", params={"season_type": "Bogus"})
        self.assertEqual(r.status_code, 400)
        r = self.client.get("/api/slate/recent-games", params={"season_type": "Playoffs"})
        self.assertEqual(r.status_code, 200)

    def test_non_chartable_stat_rejected(self):
        r = self.client.get(f"/api/players/{LEBRON_ID}", params={"stat": "steals"})
        self.assertEqual(r.status_code, 400)

    def test_n_games_floor(self):
        r = self.client.get(f"/api/players/{LEBRON_ID}", params={"n_games": 1})
        self.assertEqual(r.status_code, 422)

    # --- name trust / 404 parity -------------------------------------------------

    def test_mismatched_name_rejected(self):
        r = self.client.get(f"/api/players/{LEBRON_ID}", params={"name": "Fake Guy"})
        self.assertEqual(r.status_code, 400)

    def test_matching_name_accent_and_case_insensitive(self):
        r = self.client.get(f"/api/players/{LEBRON_ID}", params={"name": "lebron JAMES"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["player_name"], "LeBron James")

    def test_line_movement_unknown_player_404(self):
        r = self.client.get("/api/players/999999/line-movement")
        self.assertEqual(r.status_code, 404)

    # --- empty windows / timestamps ----------------------------------------------

    def test_empty_team_kpis_are_null_not_zero(self):
        r = self.client.get("/api/teams/BOS/chart")
        self.assertEqual(r.status_code, 200)
        kpis = r.json()["kpis"]
        self.assertEqual(r.json()["n_games"], 0)
        self.assertIsNone(kpis["mu"])
        self.assertIsNone(kpis["sigma"])

    def test_freshest_scrape_is_utc_iso_with_offset(self):
        ts = self.client.get("/api/slate/kpis").json()["freshest_scrape_utc"]
        self.assertIsNotNone(ts)
        self.assertTrue(ts.endswith("+00:00"), ts)
        self.assertIn("T", ts)


if __name__ == "__main__":
    unittest.main()
