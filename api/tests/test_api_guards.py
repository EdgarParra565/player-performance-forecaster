"""Per-IP rate limiting + optional shared access code (W-DEP-1/2)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import guards
from api.tests.test_api_phase2 import _seed


class GuardsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls._tmp.name) / "nba.db")
        _seed(cls.db_path)
        from api.main import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, {"NBA_DB_PATH": self.db_path})
        self.env.start()
        self.client = TestClient(self.app)

    def tearDown(self):
        self.env.stop()
        guards.GUARDS.configure()

    def _configure(self, **env):
        p = patch.dict(os.environ, {k: str(v) for k, v in env.items()})
        p.start()
        self.addCleanup(p.stop)
        guards.GUARDS.configure()

    # --- rate limiting --------------------------------------------------------

    def test_heavy_endpoints_have_tight_budget_with_retry_after(self):
        self._configure(API_RATE_HEAVY_PER_MIN=3, API_RATE_READ_PER_MIN=100)
        codes = [self.client.get("/api/cross-book").status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        r = self.client.get("/api/slate/edges")  # same heavy bucket
        self.assertEqual(r.status_code, 429)
        self.assertGreaterEqual(int(r.headers["retry-after"]), 1)
        # Cheap reads are a separate, looser budget.
        self.assertEqual(self.client.get("/api/meta").status_code, 200)

    def test_read_budget_and_health_exempt(self):
        self._configure(API_RATE_READ_PER_MIN=2)
        self.assertEqual(self.client.get("/api/meta").status_code, 200)
        self.assertEqual(self.client.get("/api/slate/kpis").status_code, 200)
        self.assertEqual(self.client.get("/api/meta").status_code, 429)
        for _ in range(5):
            self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_budget_is_per_client_ip(self):
        self._configure(API_RATE_READ_PER_MIN=1)
        a = TestClient(self.app, client=("10.0.0.1", 1234))
        b = TestClient(self.app, client=("10.0.0.2", 1234))
        self.assertEqual(a.get("/api/meta").status_code, 200)
        self.assertEqual(a.get("/api/meta").status_code, 429)
        self.assertEqual(b.get("/api/meta").status_code, 200)

    def test_spoofed_forwarded_for_does_not_change_key(self):
        # Without uvicorn --proxy-headers from a trusted proxy, XFF is ignored.
        self._configure(API_RATE_READ_PER_MIN=1)
        self.assertEqual(self.client.get("/api/meta").status_code, 200)
        r = self.client.get("/api/meta", headers={"X-Forwarded-For": "1.2.3.4"})
        self.assertEqual(r.status_code, 429)

    def test_trusted_platform_ip_header(self):
        # Fly.io: Fly-Client-IP is set (and overwritten) by the edge proxy.
        self._configure(API_RATE_READ_PER_MIN=1, API_CLIENT_IP_HEADER="Fly-Client-IP")
        a = {"Fly-Client-IP": "203.0.113.7"}
        b = {"Fly-Client-IP": "198.51.100.9"}
        self.assertEqual(self.client.get("/api/meta", headers=a).status_code, 200)
        self.assertEqual(self.client.get("/api/meta", headers=a).status_code, 429)
        self.assertEqual(self.client.get("/api/meta", headers=b).status_code, 200)
        # Garbage header value -> falls back to the socket peer.
        self.assertEqual(guards.GUARDS.client_key("not-an-ip", "10.0.0.1"), "10.0.0.1")

    def test_platform_header_ignored_unless_configured(self):
        self._configure(API_RATE_READ_PER_MIN=1)
        self.assertEqual(self.client.get("/api/meta", headers={"Fly-Client-IP": "1.1.1.1"}).status_code, 200)
        self.assertEqual(self.client.get("/api/meta", headers={"Fly-Client-IP": "2.2.2.2"}).status_code, 429)

    def test_limiter_can_be_disabled(self):
        self._configure(API_RATE_LIMIT=0, API_RATE_READ_PER_MIN=1)
        for _ in range(3):
            self.assertEqual(self.client.get("/api/meta").status_code, 200)

    def test_sliding_window_frees_slots(self):
        lim = guards.SlidingWindowLimiter(2, window_seconds=10)
        self.assertIsNone(lim.hit("k", 0.0))
        self.assertIsNone(lim.hit("k", 1.0))
        self.assertAlmostEqual(lim.hit("k", 2.0), 8.0)
        self.assertIsNone(lim.hit("k", 10.5))

    # --- access code ------------------------------------------------------------

    def test_open_when_code_unset(self):
        self.assertEqual(self.client.get("/api/meta").status_code, 200)
        self.assertFalse(self.client.get("/api/health").json()["access_code_required"])

    def test_code_required_when_set(self):
        self._configure(FLAGSHIP_ACCESS_CODE="s3cret-code")
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)  # probes stay unauthenticated
        self.assertTrue(h.json()["access_code_required"])
        r = self.client.get("/api/meta")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["detail"], "access code required")
        self.assertIn("content-security-policy", r.headers)
        ok = self.client.get("/api/meta", headers={"X-Access-Code": "s3cret-code"})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.get("/api/meta", headers={"X-Access-Code": "nope"})
        self.assertEqual(bad.status_code, 401)

    def test_wrong_codes_are_brute_force_limited(self):
        self._configure(FLAGSHIP_ACCESS_CODE="right", API_RATE_AUTH_FAIL_PER_MIN=3,
                        API_RATE_READ_PER_MIN=100)
        for _ in range(3):
            self.assertEqual(
                self.client.get("/api/meta", headers={"X-Access-Code": "wrong"}).status_code, 401)
        r = self.client.get("/api/meta", headers={"X-Access-Code": "right"})
        self.assertEqual(r.status_code, 429)  # locked out even with the right code
        self.assertIn("retry-after", r.headers)

    def test_missing_code_does_not_consume_failure_budget(self):
        # A first visit (no header yet) must not lock the user out.
        self._configure(FLAGSHIP_ACCESS_CODE="right", API_RATE_AUTH_FAIL_PER_MIN=1)
        for _ in range(3):
            self.assertEqual(self.client.get("/api/meta").status_code, 401)
        ok = self.client.get("/api/meta", headers={"X-Access-Code": "right"})
        self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
