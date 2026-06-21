"""Tests for WS4 billing-hardening backend pieces.

Covers: subscriptions.aggregate_stats / _aggregate_from_rows, the STRIPE_MODE
key resolution, the webhook payment-failed alert, the session throttle core,
and the trial-window helper. No network; Stripe/webhook posters are injected.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nba_model.web import subscriptions, stripe_helpers, throttle
from nba_model.web import auth as web_auth
from nba_model.web import webhook_app


class AggregateStatsTests(unittest.TestCase):
    def test_aggregate_from_rows_counts_and_mrr(self):
        rows = [
            {"tier": "premium", "last_event_type": "invoice.paid"},
            {"tier": "premium", "last_event_type": "customer.subscription.updated"},
            {"tier": "free", "last_event_type": "customer.subscription.deleted"},
        ]
        out = subscriptions._aggregate_from_rows(rows, price_monthly=10.0)
        self.assertEqual(out["premium"], 2)
        self.assertEqual(out["free"], 1)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["churned"], 1)
        self.assertEqual(out["mrr_estimate"], 20.0)

    def test_mrr_none_without_price(self):
        out = subscriptions._aggregate_from_rows(
            [{"tier": "premium"}], price_monthly=None)
        self.assertIsNone(out["mrr_estimate"])

    def test_aggregate_stats_end_to_end_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "subs.db")
            with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}):
                subscriptions.upsert(email="a@x.com", tier="premium")
                subscriptions.upsert(email="b@x.com", tier="free")
                stats = subscriptions.aggregate_stats(price_monthly=5.0)
        self.assertEqual(stats["premium"], 1)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["mrr_estimate"], 5.0)


class StripeModeTests(unittest.TestCase):
    def test_defaults_to_test(self):
        with patch.object(stripe_helpers, "_secret",
                          side_effect=lambda *p, default=None: default):
            self.assertEqual(stripe_helpers.stripe_mode(), "test")

    def test_live_mode_prefers_scoped_key(self):
        def fake_secret(*path, default=None):
            mapping = {
                ("stripe", "mode"): "live",
                ("stripe", "secret_key_live"): "sk_live_123",
                ("stripe", "secret_key"): "sk_fallback",
            }
            return mapping.get(path, default)
        with patch.object(stripe_helpers, "_secret", side_effect=fake_secret):
            self.assertEqual(stripe_helpers.stripe_mode(), "live")
            self.assertEqual(stripe_helpers.secret_key(), "sk_live_123")

    def test_falls_back_to_unscoped_key(self):
        def fake_secret(*path, default=None):
            mapping = {
                ("stripe", "mode"): "test",
                ("stripe", "secret_key"): "sk_test_plain",
            }
            return mapping.get(path, default)
        with patch.object(stripe_helpers, "_secret", side_effect=fake_secret):
            self.assertEqual(stripe_helpers.secret_key(), "sk_test_plain")


class _FakePoster:
    def __init__(self):
        self.calls = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return type("R", (), {"status_code": 200})()


class PaymentFailedAlertTests(unittest.TestCase):
    def test_no_webhook_no_send(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BILLING_ALERT_WEBHOOK_URL", None)
            res = webhook_app.alert_payment_failed("a@x.com", poster=_FakePoster())
        self.assertFalse(res["sent"])

    def test_posts_when_configured(self):
        poster = _FakePoster()
        with patch.dict(os.environ,
                        {"BILLING_ALERT_WEBHOOK_URL": "http://hook"}):
            res = webhook_app.alert_payment_failed("paid@x.com", poster=poster)
        self.assertTrue(res["sent"])
        self.assertEqual(poster.calls[0][1]["email"], "paid@x.com")

    def test_no_email_no_send(self):
        with patch.dict(os.environ,
                        {"BILLING_ALERT_WEBHOOK_URL": "http://hook"}):
            res = webhook_app.alert_payment_failed("", poster=_FakePoster())
        self.assertFalse(res["sent"])


class ThrottleTests(unittest.TestCase):
    def test_allows_under_limit(self):
        ts = []
        for i in range(3):
            allowed, ts = throttle.check_rate(ts, now=float(i), max_calls=3,
                                              window_seconds=60.0)
            self.assertTrue(allowed)

    def test_blocks_over_limit(self):
        ts = [0.0, 1.0, 2.0]
        allowed, ts = throttle.check_rate(ts, now=3.0, max_calls=3,
                                          window_seconds=60.0)
        self.assertFalse(allowed)

    def test_window_prunes_old_calls(self):
        ts = [0.0, 1.0, 2.0]
        # At t=100 with a 60s window, all three are stale → allowed again.
        allowed, ts = throttle.check_rate(ts, now=100.0, max_calls=3,
                                          window_seconds=60.0)
        self.assertTrue(allowed)
        self.assertEqual(ts, [100.0])


class TrialTests(unittest.TestCase):
    def test_within_trial(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2025, 4, 10, tzinfo=timezone.utc)
        start = (now - timedelta(days=3)).isoformat()
        self.assertTrue(web_auth.trial_active(start, now=now, trial_days=7))

    def test_after_trial(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2025, 4, 10, tzinfo=timezone.utc)
        start = (now - timedelta(days=10)).isoformat()
        self.assertFalse(web_auth.trial_active(start, now=now, trial_days=7))

    def test_fails_closed_on_bad_input(self):
        self.assertFalse(web_auth.trial_active(None))
        self.assertFalse(web_auth.trial_active("not-a-date"))


if __name__ == "__main__":
    unittest.main()
