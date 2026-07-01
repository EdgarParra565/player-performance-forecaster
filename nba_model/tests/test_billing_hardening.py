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

    def test_all_premium_mrr_scales_linearly(self):
        rows = [{"tier": "premium"} for _ in range(7)]
        out = subscriptions._aggregate_from_rows(rows, price_monthly=14.99)
        self.assertEqual(out["premium"], 7)
        self.assertEqual(out["free"], 0)
        self.assertEqual(out["total"], 7)
        self.assertEqual(out["churned"], 0)
        # 7 * 14.99 rounded to 2dp
        self.assertEqual(out["mrr_estimate"], 104.93)

    def test_churn_matches_any_deleted_event(self):
        # `churned` is a substring match on the event-type string so the
        # admin dashboard catches both subscription cancellation and
        # explicit-delete events.
        rows = [
            {"tier": "free", "last_event_type": "customer.subscription.deleted"},
            {"tier": "free", "last_event_type": "customer.deleted"},
            {"tier": "free", "last_event_type": "invoice.payment_failed"},
            {"tier": "premium", "last_event_type": "invoice.paid"},
        ]
        out = subscriptions._aggregate_from_rows(rows, price_monthly=10.0)
        self.assertEqual(out["churned"], 2)
        self.assertEqual(out["premium"], 1)
        self.assertEqual(out["free"], 3)

    def test_empty_rows_zero_everything(self):
        out = subscriptions._aggregate_from_rows([], price_monthly=20.0)
        self.assertEqual(
            (out["premium"], out["free"], out["total"], out["churned"]),
            (0, 0, 0, 0),
        )
        # MRR = 0 * price = 0.0 (not None) so the dashboard renders "$0".
        self.assertEqual(out["mrr_estimate"], 0.0)

    def test_aggregate_stats_resolves_env_price(self):
        # `STRIPE_PRICE_MONTHLY_USD` must drive MRR when no explicit price
        # is passed — that's what the admin dashboard relies on.
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "subs.db")
            with patch.dict(os.environ, {
                "SUBSCRIPTIONS_DB_PATH": db,
                "STRIPE_PRICE_MONTHLY_USD": "9.99",
            }):
                subscriptions.upsert(email="env-priced@x.com", tier="premium")
                stats = subscriptions.aggregate_stats()
        self.assertEqual(stats["mrr_estimate"], 9.99)


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

    def test_poster_exception_does_not_raise(self):
        # The alert path must NEVER break the webhook handler — a network
        # outage at the Slack/Discord endpoint should fail closed silently.
        def raising_poster(*_args, **_kwargs):
            raise RuntimeError("upstream down")
        with patch.dict(os.environ,
                        {"BILLING_ALERT_WEBHOOK_URL": "http://hook"}):
            res = webhook_app.alert_payment_failed(
                "paid@x.com", poster=raising_poster)
        self.assertFalse(res["sent"])
        self.assertIn("post_failed", res.get("reason", ""))

    def test_payload_shape_is_minimal_and_typed(self):
        poster = _FakePoster()
        with patch.dict(os.environ,
                        {"BILLING_ALERT_WEBHOOK_URL": "http://hook"}):
            webhook_app.alert_payment_failed("paid@x.com", poster=poster)
        # Exactly one call; payload carries event + email but no internal IDs.
        self.assertEqual(len(poster.calls), 1)
        url, body = poster.calls[0]
        self.assertEqual(url, "http://hook")
        self.assertEqual(body.get("event"), "invoice.payment_failed")
        self.assertEqual(body.get("email"), "paid@x.com")
        # No raw Stripe IDs leak into the alert — keeps Slack channels clean.
        self.assertNotIn("customer_id", body)
        self.assertNotIn("subscription_id", body)


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


class CreatedAtMigrationTests(unittest.TestCase):
    """Legacy ``user_subscriptions`` rows without ``created_at`` must get
    back-filled idempotently on first connect. The trial logic depends on this.
    """

    def _fresh_tmp_db(self) -> str:
        tmp = tempfile.mkdtemp(prefix="subs_mig_")
        self.addCleanup(self._cleanup_dir, tmp)
        return str(Path(tmp) / "subs.db")

    @staticmethod
    def _cleanup_dir(path: str) -> None:
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    def test_migration_backfills_legacy_rows(self):
        import sqlite3
        db = self._fresh_tmp_db()
        # Pre-create a legacy schema row (no created_at column).
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "CREATE TABLE user_subscriptions ("
                "email TEXT PRIMARY KEY, tier TEXT, updated_at TEXT)"
            )
            conn.execute(
                "INSERT INTO user_subscriptions VALUES "
                "('legacy@x.com', 'premium', '2024-01-15T00:00:00')"
            )
            conn.commit()
        finally:
            conn.close()
        # Reset the bootstrap cache so this fresh path actually triggers
        # migration even if the test runner has touched other DBs first.
        subscriptions._BOOTSTRAPPED_PATHS.discard(db)
        # First lookup runs the bootstrap + migration.
        row = subscriptions.lookup("legacy@x.com", db_path=db)
        self.assertIsNotNone(row)
        # created_at should be back-filled from updated_at.
        self.assertEqual(row["created_at"], "2024-01-15T00:00:00")
        # Idempotency: a second call must not blow up or change the value.
        row2 = subscriptions.lookup("legacy@x.com", db_path=db)
        self.assertEqual(row2["created_at"], row["created_at"])

    def test_new_row_anchors_created_at_once(self):
        db = self._fresh_tmp_db()
        subscriptions._BOOTSTRAPPED_PATHS.discard(db)
        subscriptions.upsert(email="new@x.com", tier="free", db_path=db)
        first = subscriptions.lookup("new@x.com", db_path=db)["created_at"]
        # Subsequent upserts (e.g. Stripe webhook flipping the tier) must
        # NOT refresh created_at, otherwise the trial clock would restart.
        subscriptions.upsert(email="new@x.com", tier="premium", db_path=db,
                             last_event_type="checkout.session.completed")
        second = subscriptions.lookup("new@x.com", db_path=db)["created_at"]
        self.assertEqual(first, second)


class TouchFirstSeenTests(unittest.TestCase):
    def _fresh_tmp_db(self) -> str:
        tmp = tempfile.mkdtemp(prefix="subs_first_")
        self.addCleanup(self._cleanup_dir, tmp)
        return str(Path(tmp) / "subs.db")

    @staticmethod
    def _cleanup_dir(path: str) -> None:
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    def test_inserts_when_absent(self):
        db = self._fresh_tmp_db()
        subscriptions._BOOTSTRAPPED_PATHS.discard(db)
        self.assertTrue(
            subscriptions.touch_first_seen("first@x.com", db_path=db))
        row = subscriptions.lookup("first@x.com", db_path=db)
        self.assertEqual(row["tier"], "free")
        self.assertIsNotNone(row["created_at"])

    def test_noop_when_present(self):
        db = self._fresh_tmp_db()
        subscriptions._BOOTSTRAPPED_PATHS.discard(db)
        subscriptions.upsert(email="known@x.com", tier="premium", db_path=db)
        before = subscriptions.lookup("known@x.com", db_path=db)
        self.assertFalse(
            subscriptions.touch_first_seen("known@x.com", db_path=db))
        after = subscriptions.lookup("known@x.com", db_path=db)
        # Did not flip the tier back to free, did not touch created_at.
        self.assertEqual(after["tier"], "premium")
        self.assertEqual(after["created_at"], before["created_at"])

    def test_rejects_invalid_email(self):
        db = self._fresh_tmp_db()
        subscriptions._BOOTSTRAPPED_PATHS.discard(db)
        self.assertFalse(subscriptions.touch_first_seen("not-email", db_path=db))


class TierForTrialTests(unittest.TestCase):
    """``auth.tier_for`` wires `trial_active` into live tier resolution."""

    def _fresh_tmp_db(self) -> str:
        tmp = tempfile.mkdtemp(prefix="subs_trial_")
        self.addCleanup(self._cleanup_dir, tmp)
        path = str(Path(tmp) / "subs.db")
        subscriptions._BOOTSTRAPPED_PATHS.discard(path)
        return path

    @staticmethod
    def _cleanup_dir(path: str) -> None:
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    def _patch_billing(self, *, billing=True, trial=True):
        # Patches the module-level flags + admin set so the test owns the env.
        return [
            patch.object(web_auth, "BILLING_ENABLED", billing),
            patch.object(web_auth, "TRIAL_ENABLED", trial),
            patch.object(web_auth, "_admin_emails", return_value=set()),
        ]

    def _apply(self, patches):
        for p in patches:
            self.addCleanup(p.stop)
            p.start()

    def test_first_signin_resolves_premium_and_anchors_row(self):
        db = self._fresh_tmp_db()
        self._apply(self._patch_billing(billing=True, trial=True))
        with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}):
            self.assertEqual(web_auth.tier_for("noobie@x.com"), "premium")
            # The lookup row should now exist with a fresh created_at.
            row = subscriptions.lookup("noobie@x.com")
        self.assertIsNotNone(row)
        self.assertEqual(row["tier"], "free")  # paywall row, premium via trial

    def test_active_window_resolves_premium(self):
        from datetime import datetime, timedelta, timezone
        db = self._fresh_tmp_db()
        self._apply(self._patch_billing(billing=True, trial=True))
        with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}):
            # Seed a free row whose created_at is well inside the 7d window.
            subscriptions.upsert(email="warm@x.com", tier="free")
            recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            import sqlite3
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE user_subscriptions SET created_at = ? "
                    "WHERE email = ?", (recent, "warm@x.com"))
                conn.commit()
            self.assertEqual(web_auth.tier_for("warm@x.com"), "premium")

    def test_expired_window_falls_back_to_subscriptions_tier(self):
        from datetime import datetime, timedelta, timezone
        db = self._fresh_tmp_db()
        self._apply(self._patch_billing(billing=True, trial=True))
        with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}):
            subscriptions.upsert(email="stale@x.com", tier="free")
            old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            import sqlite3
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE user_subscriptions SET created_at = ? "
                    "WHERE email = ?", (old, "stale@x.com"))
                conn.commit()
            self.assertEqual(web_auth.tier_for("stale@x.com"), "free")

    def test_trial_flag_off_does_not_grant_premium(self):
        db = self._fresh_tmp_db()
        self._apply(self._patch_billing(billing=True, trial=False))
        with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}):
            # Brand-new email; with trial off, must resolve to free.
            self.assertEqual(web_auth.tier_for("nofree@x.com"), "free")
            # And we must not have ghost-inserted a row.
            self.assertIsNone(subscriptions.lookup("nofree@x.com"))

    def test_billing_off_short_circuits_to_premium(self):
        db = self._fresh_tmp_db()
        self._apply(self._patch_billing(billing=False, trial=True))
        with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}):
            # BILLING_ENABLED=0 must keep the open-access mode intact.
            self.assertEqual(web_auth.tier_for("anyone@x.com"), "premium")
            self.assertIsNone(subscriptions.lookup("anyone@x.com"))


if __name__ == "__main__":
    unittest.main()
