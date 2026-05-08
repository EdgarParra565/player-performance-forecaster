"""Security regression tests.

These cover the specific attacks the SECURITY.md document promises we
defend against. Adding a new test here is the right move whenever a
security review surfaces a concrete failure mode.

Categories:
    - Input validation (email, db path, sql expressions)
    - Auth tier resolution (fail-closed)
    - Webhook handler (body size, signature, idempotency, replay)
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from nba_model.web import subscriptions


class EmailValidationTests(unittest.TestCase):
    """Defense in depth: even though OIDC verifies emails, we sanity-check
    the value before using it as a SQL key."""

    def test_accepts_normal_emails(self):
        self.assertEqual(
            subscriptions._validate_email("Alice.Smith+tag@Example.com"),
            "alice.smith+tag@example.com",
        )

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            subscriptions._validate_email("")
        with self.assertRaises(ValueError):
            subscriptions._validate_email("   ")

    def test_rejects_no_at_sign(self):
        with self.assertRaises(ValueError):
            subscriptions._validate_email("notanemail")

    def test_rejects_no_tld(self):
        with self.assertRaises(ValueError):
            subscriptions._validate_email("foo@bar")

    def test_rejects_control_chars(self):
        with self.assertRaises(ValueError):
            subscriptions._validate_email("a\x00@b.com")
        with self.assertRaises(ValueError):
            subscriptions._validate_email("a\nb@c.com")

    def test_rejects_overlong(self):
        too_long = "a" * 250 + "@b.com"
        with self.assertRaises(ValueError):
            subscriptions._validate_email(too_long)

    def test_rejects_sql_injection_payload(self):
        # If our regex regressed and someone could squeeze SQL through,
        # this would pass through. It must NOT.
        for bad in (
            "x';DROP TABLE user_subscriptions;--@evil.com",
            "x@evil.com';--",
            'x"@evil.com',
            "x;y@evil.com",
            "<script>@evil.com",
        ):
            with self.assertRaises(ValueError, msg=f"accepted: {bad!r}"):
                subscriptions._validate_email(bad)


class TierResolutionTests(unittest.TestCase):
    """Subscriptions table behavior - fail closed in every error path."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._old_env = os.environ.get("SUBSCRIPTIONS_DB_PATH")
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("SUBSCRIPTIONS_DB_PATH", None)
        else:
            os.environ["SUBSCRIPTIONS_DB_PATH"] = self._old_env
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_unknown_email_is_free(self):
        self.assertEqual(subscriptions.tier_for("nobody@example.com"), "free")

    def test_invalid_email_is_free_not_premium(self):
        # Even if we somehow ended up with a malformed email in the chain,
        # we must NOT grant premium.
        self.assertEqual(subscriptions.tier_for("not-an-email"), "free")
        self.assertEqual(subscriptions.tier_for(""), "free")
        self.assertEqual(subscriptions.tier_for(None), "free")  # type: ignore[arg-type]

    def test_round_trip_premium(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        subscriptions.upsert(
            email="alice@example.com", tier="premium",
            stripe_customer_id="cus_x", stripe_subscription_id="sub_x",
            current_period_end=future, last_event_type="checkout.session.completed",
        )
        self.assertEqual(subscriptions.tier_for("alice@example.com"), "premium")
        # Case insensitive
        self.assertEqual(
            subscriptions.tier_for("ALICE@EXAMPLE.COM"), "premium",
        )

    def test_expired_premium_downgrades_to_free(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        subscriptions.upsert(
            email="bob@example.com", tier="premium",
            current_period_end=past,
        )
        self.assertEqual(subscriptions.tier_for("bob@example.com"), "free")

    def test_explicit_cancel_is_free(self):
        subscriptions.upsert(email="carol@example.com", tier="free")
        self.assertEqual(subscriptions.tier_for("carol@example.com"), "free")

    def test_event_idempotency(self):
        first = subscriptions.record_stripe_event(
            "evt_xyz", "customer.subscription.updated", "{}",
        )
        second = subscriptions.record_stripe_event(
            "evt_xyz", "customer.subscription.updated", "{}",
        )
        self.assertTrue(first)
        self.assertFalse(second)

    def test_upsert_rejects_invalid_tier(self):
        with self.assertRaises(ValueError):
            subscriptions.upsert(email="d@example.com", tier="enterprise")

    def test_upsert_rejects_invalid_email(self):
        with self.assertRaises(ValueError):
            subscriptions.upsert(email="x';--", tier="premium")


class TeamSqlAllowlistTests(unittest.TestCase):
    """The one f-string SQL fragment must only interpolate allowlisted exprs."""

    def test_allowlisted_stats_return_safe_expr(self):
        from nba_model.visualization import player_charts as pc
        for stat in ("points", "assists", "rebounds", "pra", "ra",
                     "three_pointers_made", "field_goals_made"):
            expr = pc._team_value_sql_expr(stat)
            self.assertIsNotNone(expr, f"{stat} should have an expression")
            # No ; or -- or quote chars in the returned expression - if
            # someone wires a future stat that ends up with one of these,
            # we want the assertion in fetch_team_chart_data to bite.
            self.assertNotIn(";", expr)
            self.assertNotIn("--", expr)
            self.assertNotIn("'", expr)
            self.assertNotIn('"', expr)

    def test_unknown_stat_returns_none(self):
        from nba_model.visualization import player_charts as pc
        for stat in ("minutes", "garbage", "1; DROP TABLE", ""):
            self.assertIsNone(pc._team_value_sql_expr(stat))


class WebhookSecurityTests(unittest.TestCase):
    """End-to-end webhook attacks via FastAPI TestClient."""

    def setUp(self):
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_unit_test_secret"
        os.environ["STRIPE_SECRET_KEY"] = "sk_test_unit"
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path

        # Import lazily so module-level FastAPI setup picks up the env we just set.
        from fastapi.testclient import TestClient
        from nba_model.web import webhook_app
        self.client = TestClient(webhook_app.app)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_healthz_no_version_disclosure(self):
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json().keys()), {"status"})

    def test_docs_endpoints_disabled(self):
        # /docs, /redoc, /openapi.json all return 404 in production config
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 404, f"{path} should be 404")

    def test_missing_signature_rejected(self):
        r = self.client.post("/stripe/webhook", json={"id": "evt_1"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("missing", r.json()["detail"])

    def test_bad_signature_rejected(self):
        r = self.client.post(
            "/stripe/webhook",
            content=b'{"id":"evt_1","type":"x"}',
            headers={
                "stripe-signature": "t=1,v1=deadbeef",
                "content-type": "application/json",
            },
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "bad signature")

    def test_oversize_payload_rejected(self):
        # 1 MiB body, well over the 256 KiB cap.
        big = b'{"x":"' + b"a" * (1024 * 1024) + b'"}'
        r = self.client.post(
            "/stripe/webhook",
            content=big,
            headers={
                "stripe-signature": "t=1,v1=deadbeef",
                "content-type": "application/json",
            },
        )
        self.assertEqual(r.status_code, 413)

    def test_security_headers_present_on_every_response(self):
        # Headers should appear on healthz too (middleware applies globally).
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        for header, expected_substring in (
            ("Strict-Transport-Security", "max-age="),
            ("Content-Security-Policy", "default-src 'none'"),
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "no-referrer"),
            ("Permissions-Policy", "camera=()"),
            ("Cross-Origin-Opener-Policy", "same-origin"),
            ("Cross-Origin-Resource-Policy", "same-origin"),
        ):
            self.assertIn(header, r.headers, f"missing {header}")
            self.assertIn(
                expected_substring, r.headers[header],
                f"{header} did not contain {expected_substring!r}",
            )

    def test_server_header_does_not_leak_uvicorn_version(self):
        r = self.client.get("/healthz")
        # Our middleware overrides the Server header to "webhook" so a
        # scanner can't fingerprint Uvicorn / Python version from it.
        self.assertEqual(r.headers.get("Server"), "webhook")


class RateLimiterTests(unittest.TestCase):
    """The IPRateLimiter is a tight little class so we test it directly."""

    def test_allows_under_limit(self):
        from nba_model.web.webhook_app import IPRateLimiter
        rl = IPRateLimiter(max_requests=3, window_seconds=10)
        # First 3 hits within the window are allowed.
        self.assertTrue(rl.allow("ip1", 1.0))
        self.assertTrue(rl.allow("ip1", 1.5))
        self.assertTrue(rl.allow("ip1", 2.0))

    def test_blocks_over_limit(self):
        from nba_model.web.webhook_app import IPRateLimiter
        rl = IPRateLimiter(max_requests=3, window_seconds=10)
        for t in (1.0, 1.5, 2.0):
            self.assertTrue(rl.allow("ip1", t))
        self.assertFalse(rl.allow("ip1", 2.5))

    def test_window_slides(self):
        from nba_model.web.webhook_app import IPRateLimiter
        rl = IPRateLimiter(max_requests=2, window_seconds=10)
        self.assertTrue(rl.allow("ip1", 1.0))
        self.assertTrue(rl.allow("ip1", 2.0))
        # Same IP at t=3.0 hits the limit.
        self.assertFalse(rl.allow("ip1", 3.0))
        # Past the window: previous hits expire so the bucket is empty.
        self.assertTrue(rl.allow("ip1", 100.0))

    def test_per_ip_isolation(self):
        from nba_model.web.webhook_app import IPRateLimiter
        rl = IPRateLimiter(max_requests=1, window_seconds=10)
        self.assertTrue(rl.allow("ip1", 1.0))
        self.assertFalse(rl.allow("ip1", 1.0))
        # Different IP: not blocked.
        self.assertTrue(rl.allow("ip2", 1.0))

    def test_gc_drops_stale_buckets(self):
        from nba_model.web.webhook_app import IPRateLimiter
        rl = IPRateLimiter(max_requests=10, window_seconds=10)
        rl.allow("ip-old", 1.0)
        rl.allow("ip-new", 100.0)
        rl.gc(101.0)
        self.assertNotIn("ip-old", rl._buckets)
        self.assertIn("ip-new", rl._buckets)


if __name__ == "__main__":
    unittest.main()
