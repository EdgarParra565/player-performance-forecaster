"""Tests for watchlist persistence — anonymous cross-session pins in particular.

The Streamlit-coupled surface (`get`/`add`, which touch `st.session_state`)
isn't exercised here; instead we test the decoupled persistence layer
(`_load_for`/`_save_for`, keyed by email or ``anon:<token>``) and the pure
token resolver, which is where the cross-session behavior actually lives.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nba_model.web import watchlist as wl


class _StoreTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store = str(Path(self._tmp.name) / "watchlists.json")
        self._env = patch.dict(os.environ, {"WATCHLIST_STORE_PATH": self._store})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()


class AnonTokenResolverTests(unittest.TestCase):
    def test_session_token_wins(self):
        token = wl._resolve_anon_token("abc123", "def456", mint=lambda: "fff000")
        self.assertEqual(token, "abc123")

    def test_cookie_used_on_return_visit(self):
        token = wl._resolve_anon_token(None, "def456", mint=lambda: "fff000")
        self.assertEqual(token, "def456")

    def test_mints_when_nothing_known(self):
        token = wl._resolve_anon_token(None, None, mint=lambda: "fff000")
        self.assertEqual(token, "fff000")

    def test_malformed_token_is_rejected(self):
        # A planted cookie with markup must NOT be reused (XSS-via-cookie guard).
        evil = '";</script><script>alert(1)</script>'
        token = wl._resolve_anon_token(None, evil, mint=lambda: "fff000")
        self.assertEqual(token, "fff000")
        # Non-hex session value is also ignored.
        self.assertEqual(
            wl._resolve_anon_token("not-hex!", None, mint=lambda: "fff000"),
            "fff000",
        )

    def test_default_mint_is_unique_hex(self):
        a = wl._resolve_anon_token(None, None)
        b = wl._resolve_anon_token(None, None)
        self.assertNotEqual(a, b)
        self.assertTrue(all(c in "0123456789abcdef" for c in a))


class PersistenceLayerTests(_StoreTestBase):
    def test_anon_round_trip_survives_fresh_load(self):
        # A "session" saves under the browser token...
        wl._save_for("anon:browser123", ["LeBron James", "Stephen Curry"])
        # ...a later session (fresh _load_all) reads it back.
        self.assertEqual(
            wl._load_for("anon:browser123"),
            ["LeBron James", "Stephen Curry"],
        )

    def test_cap_enforced_on_save(self):
        wl._save_for("anon:x", [f"P{i}" for i in range(40)])
        self.assertEqual(len(wl._load_for("anon:x")), wl.MAX_ITEMS)

    def test_keys_are_isolated(self):
        wl._save_for("anon:a", ["A"])
        wl._save_for("anon:b", ["B"])
        wl._save_for("paid@user.com", ["C"])
        self.assertEqual(wl._load_for("anon:a"), ["A"])
        self.assertEqual(wl._load_for("anon:b"), ["B"])
        self.assertEqual(wl._load_for("paid@user.com"), ["C"])

    def test_unknown_key_returns_empty(self):
        self.assertEqual(wl._load_for("anon:never-seen"), [])


class UserKeyTests(_StoreTestBase):
    def test_open_launch_uses_anon_token(self):
        with patch.object(wl.web_auth, "BILLING_ENABLED", False), \
                patch.object(wl, "_anon_token", return_value="tok42"):
            self.assertEqual(wl._user_key(), "anon:tok42")

    def test_billing_signed_in_uses_email(self):
        fake_user = type("U", (), {"is_authenticated": True,
                                   "email": "Paid@User.com"})()
        with patch.object(wl.web_auth, "BILLING_ENABLED", True), \
                patch.object(wl.web_auth, "current_user", return_value=fake_user):
            self.assertEqual(wl._user_key(), "paid@user.com")

    def test_billing_signed_out_is_session_only(self):
        fake_user = type("U", (), {"is_authenticated": False, "email": None})()
        with patch.object(wl.web_auth, "BILLING_ENABLED", True), \
                patch.object(wl.web_auth, "current_user", return_value=fake_user):
            self.assertIsNone(wl._user_key())


if __name__ == "__main__":
    unittest.main()
