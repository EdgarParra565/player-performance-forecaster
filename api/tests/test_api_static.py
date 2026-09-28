"""Same-origin production serving (FLAGSHIP_STATIC_DIR) + security headers."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.tests.test_api_phase2 import _seed


class StaticServingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.db_path = str(root / "nba.db")
        _seed(cls.db_path)
        dist = root / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html><div id=root></div>")
        (dist / "assets" / "index-abc123.js").write_text("console.log(1)")
        (root / "secret.txt").write_text("outside the build dir")
        os.environ["NBA_DB_PATH"] = cls.db_path
        os.environ["FLAGSHIP_STATIC_DIR"] = str(dist)
        os.environ["API_DOCS"] = "0"
        import api.main
        cls.main = importlib.reload(api.main)
        cls.client = TestClient(cls.main.app)

    @classmethod
    def tearDownClass(cls):
        for k in ("NBA_DB_PATH", "FLAGSHIP_STATIC_DIR", "API_DOCS"):
            os.environ.pop(k, None)
        importlib.reload(cls.main)  # restore the default (no static) app
        cls._tmp.cleanup()

    def test_index_and_client_routes_serve_app_shell(self):
        for path in ("/", "/player/2544", "/edges"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            self.assertIn("id=root", r.text)
            self.assertEqual(r.headers["cache-control"], "no-cache")

    def test_hashed_assets_cached_immutably(self):
        r = self.client.get("/assets/index-abc123.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn("immutable", r.headers["cache-control"])

    def test_api_routes_still_win_and_unknown_api_is_404(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        r = self.client.get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("id=root", r.text)

    def test_no_path_traversal(self):
        r = self.client.get("/..%2Fsecret.txt")
        self.assertNotIn("outside the build dir", r.text)

    def test_docs_disabled_and_security_headers(self):
        self.assertEqual(self.client.get("/api/docs").status_code, 404)
        r = self.client.get("/")
        csp = r.headers["content-security-policy"]
        self.assertIn("default-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")


if __name__ == "__main__":
    unittest.main()
