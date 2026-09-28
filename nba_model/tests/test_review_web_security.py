"""Regression tests for the three web-layer security fixes (review_findings_web.md).

1. Manual-lines "Save to DB" is admin-only regardless of BILLING_ENABLED.
2. The Stripe webhook grants premium only on an entitled subscription /
   payment status, not on the event type alone.
3. Admin identity is bound to OIDC iss+sub (or a VERIFIED email), so a
   multi-tenant Microsoft account with an attacker-chosen email claim
   ("nOAuth") can't become admin or inherit a paying user's tier.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nba_model.web import auth as web_auth
from nba_model.web import subscriptions


# ---------------------------------------------------------------------------
# 1. Manual-lines save gate
# ---------------------------------------------------------------------------
def _manual_lines_script():
    # Runs inside streamlit.testing AppTest (source-extracted), so imports
    # must be local. Module-level patches applied by the test are visible
    # because AppTest executes in-process.
    import os as _os
    from nba_model.web import app as _web_app
    _web_app._manual_lines_import_view(db_path=_os.environ["_ML_TEST_DB"])


class ManualLinesSaveGateTests(unittest.TestCase):
    def setUp(self):
        from nba_model.data.database.db_manager import DatabaseManager
        tmp = tempfile.mkdtemp(prefix="ml_gate_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.db_path = str(Path(tmp) / "nba.db")
        with DatabaseManager(db_path=self.db_path):
            pass  # create schema
        env = patch.dict(os.environ, {"_ML_TEST_DB": self.db_path})
        env.start()
        self.addCleanup(env.stop)

    def _row_count(self) -> int:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM betting_lines").fetchone()[0]

    def _run(self, *, billing: bool, admin: bool):
        from streamlit.testing.v1 import AppTest
        patches = [
            patch.object(web_auth, "BILLING_ENABLED", billing),
            patch.object(web_auth, "is_admin", return_value=admin),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        at = AppTest.from_function(_manual_lines_script, default_timeout=30)
        at.run()
        return at

    def _save_button(self, at):
        return next(b for b in at.button if b.label == "Save to DB")

    def test_billing_off_anonymous_cannot_save(self):
        # The original bug: gate was `BILLING_ENABLED and not is_admin`, so
        # open-access mode (billing off) let anyone write betting_lines.
        at = self._run(billing=False, admin=False)
        self.assertTrue(self._save_button(at).disabled)
        self.assertTrue(any("admin-only" in w.value for w in at.warning))

    def test_billing_on_anonymous_cannot_save(self):
        at = self._run(billing=True, admin=False)
        self.assertTrue(self._save_button(at).disabled)

    def test_forced_submit_is_rejected_server_side(self):
        # Even if the disabled button is bypassed, the save branch re-checks.
        at = self._run(billing=False, admin=False)
        self._save_button(at).click().run()
        self.assertEqual(self._row_count(), 0)
        self.assertTrue(any("admin-only" in e.value for e in at.error))

    def test_admin_can_save(self):
        at = self._run(billing=False, admin=True)
        btn = self._save_button(at)
        self.assertFalse(btn.disabled)
        btn.click().run()
        self.assertGreater(self._row_count(), 0)


# ---------------------------------------------------------------------------
# 2. Webhook premium grant keyed on status
# ---------------------------------------------------------------------------
_SECRET = "whsec_review_web_security"


class WebhookStatusGateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="wh_status_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.subs_db = str(Path(tmp) / "subs.db")
        subscriptions._BOOTSTRAPPED_PATHS.discard(self.subs_db)
        env = patch.dict(os.environ, {
            "STRIPE_WEBHOOK_SECRET": _SECRET,
            "STRIPE_SECRET_KEY": "sk_test_unit",
            "SUBSCRIPTIONS_DB_PATH": self.subs_db,
        })
        env.start()
        self.addCleanup(env.stop)
        from fastapi.testclient import TestClient
        from nba_model.web import webhook_app
        self.webhook_app = webhook_app
        self.client = TestClient(webhook_app.app)
        self._n = 0

    def _post(self, event_type: str, obj: dict):
        self._n += 1
        body = json.dumps({
            "id": f"evt_review_{self._n}_{time.time_ns()}",
            "object": "event",
            "type": event_type,
            "data": {"object": obj},
        }).encode()
        ts = int(time.time())
        sig = hmac.new(_SECRET.encode(), f"{ts}.".encode() + body,
                       hashlib.sha256).hexdigest()
        r = self.client.post(
            "/stripe/webhook", content=body,
            headers={"stripe-signature": f"t={ts},v1={sig}",
                     "content-type": "application/json"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r

    def _sub(self, status: str, email="payer@example.com") -> dict:
        return {"id": "sub_1", "object": "subscription", "status": status,
                "customer_email": email, "customer": None}

    def test_subscription_updated_past_due_is_not_premium(self):
        self._post("customer.subscription.created", self._sub("active"))
        self.assertEqual(subscriptions.tier_for("payer@example.com"), "premium")
        for bad in ("past_due", "unpaid", "incomplete", "canceled",
                    "incomplete_expired", "paused"):
            self._post("customer.subscription.updated", self._sub("active"))
            self._post("customer.subscription.updated", self._sub(bad))
            self.assertEqual(
                subscriptions.tier_for("payer@example.com"), "free", bad)

    def test_subscription_created_incomplete_not_premium(self):
        self._post("customer.subscription.created", self._sub("incomplete"))
        self.assertEqual(subscriptions.tier_for("payer@example.com"), "free")

    def test_trialing_is_premium(self):
        self._post("customer.subscription.created", self._sub("trialing"))
        self.assertEqual(subscriptions.tier_for("payer@example.com"), "premium")

    def test_checkout_requires_paid(self):
        obj = {"id": "cs_1", "object": "checkout.session",
               "customer_email": "cs@example.com", "payment_status": "unpaid"}
        self._post("checkout.session.completed", obj)
        self.assertEqual(subscriptions.tier_for("cs@example.com"), "free")
        obj["payment_status"] = "paid"
        self._post("checkout.session.completed", obj)
        self.assertEqual(subscriptions.tier_for("cs@example.com"), "premium")

    def _post_raw(self, event: dict):
        body = json.dumps(event).encode()
        ts = int(time.time())
        sig = hmac.new(_SECRET.encode(), f"{ts}.".encode() + body,
                       hashlib.sha256).hexdigest()
        return self.client.post(
            "/stripe/webhook", content=body,
            headers={"stripe-signature": f"t={ts},v1={sig}",
                     "content-type": "application/json"},
        )

    def _event(self, eid, etype, obj, created=None):
        ev = {"id": eid, "object": "event", "type": etype, "data": {"object": obj}}
        if created is not None:
            ev["created"] = created
        return ev

    def test_out_of_order_event_does_not_overwrite_newer_state(self):
        email = "order@example.com"
        newer = self._event("evt_new", "customer.subscription.updated",
                            {"id": "sub_9", "status": "active", "customer_email": email},
                            created=2_000)
        older = self._event("evt_old", "customer.subscription.created",
                            {"id": "sub_9", "status": "incomplete", "customer_email": email},
                            created=1_000)
        self.assertEqual(self._post_raw(newer).status_code, 200)
        r = self._post_raw(older)
        self.assertTrue(r.json().get("stale"))
        self.assertEqual(subscriptions.tier_for(email), "premium")

    def test_old_subscription_cancel_does_not_revoke_new_one(self):
        email = "resub@example.com"
        self._post_raw(self._event("evt_b", "customer.subscription.created",
                                   {"id": "sub_B", "status": "active", "customer_email": email}))
        r = self._post_raw(self._event("evt_a_del", "customer.subscription.deleted",
                                       {"id": "sub_A", "status": "canceled", "customer_email": email}))
        self.assertTrue(r.json().get("stale_subscription"))
        self.assertEqual(subscriptions.tier_for(email), "premium")

    def test_customer_lookup_failure_is_retryable_not_dropped(self):
        from unittest.mock import patch as _patch
        ev = self._event("evt_lookup", "customer.subscription.deleted",
                         {"id": "sub_1", "status": "canceled", "customer": "cus_1"})
        with _patch.object(self.webhook_app.stripe.Customer, "retrieve",
                           side_effect=RuntimeError("network down")):
            r = self._post_raw(ev)
        self.assertEqual(r.status_code, 503)
        # The event was un-recorded, so Stripe's retry is processed (not a
        # silently-acknowledged duplicate).
        with _patch.object(self.webhook_app.stripe.Customer, "retrieve",
                           return_value={"email": "lookup@example.com"}):
            r2 = self._post_raw(ev)
        self.assertEqual(r2.status_code, 200)
        self.assertNotIn("duplicate", r2.json())

    def test_processing_failure_unrecords_event(self):
        from unittest.mock import patch as _patch
        ev = self._event("evt_locked", "customer.subscription.updated",
                         {"id": "sub_1", "status": "active", "customer_email": "lock@example.com"})
        import sqlite3 as _sqlite3
        with _patch.object(subscriptions, "upsert",
                           side_effect=_sqlite3.OperationalError("database is locked")):
            r = self._post_raw(ev)
        self.assertEqual(r.status_code, 500)
        r2 = self._post_raw(ev)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(subscriptions.tier_for("lock@example.com"), "premium")

    def test_valid_deliveries_do_not_consume_rate_limit(self):
        limiter = self.webhook_app._rate_limiter
        limiter._buckets.clear()
        with patch.object(limiter, "max_requests", 3):
            for i in range(6):
                r = self._post_raw(self._event(f"evt_rl_{i}", "ping.pong", {}))
                self.assertEqual(r.status_code, 200)
            for _ in range(3):
                self.client.post("/stripe/webhook", content=b"{}",
                                 headers={"stripe-signature": "t=1,v1=bad"})
            r = self.client.post("/stripe/webhook", content=b"{}",
                                 headers={"stripe-signature": "t=1,v1=bad"})
            self.assertEqual(r.status_code, 429)
        limiter._buckets.clear()

    def test_tier_for_event_pure_mapping(self):
        f = self.webhook_app.tier_for_event
        self.assertEqual(f("customer.subscription.updated", {"status": "active"}), "premium")
        self.assertEqual(f("customer.subscription.updated", {"status": "past_due"}), "free")
        self.assertEqual(f("customer.subscription.updated", {}), "free")  # fail closed
        self.assertIsNone(f("checkout.session.completed", {"payment_status": "no_payment_required"}))
        self.assertEqual(f("invoice.payment_failed", {}), "free")
        self.assertIsNone(f("charge.refunded", {}))


# ---------------------------------------------------------------------------
# 3. Admin identity binding (nOAuth)
# ---------------------------------------------------------------------------
_GOOGLE = "https://accounts.google.com"
_MS_COMMON_TENANT = "https://login.microsoftonline.com/attacker-tenant/v2.0"
_MS_OWN_TENANT = "https://login.microsoftonline.com/owner-tenant/v2.0"


class AdminIdentityBindingTests(unittest.TestCase):
    def _as(self, *, secrets: dict, **claims):
        user = SimpleNamespace(is_logged_in=True, **claims)
        for p in (
            patch.object(web_auth, "_streamlit_user", return_value=user),
            patch.object(web_auth, "_auth_secrets", return_value=secrets),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_unverified_microsoft_email_is_not_admin(self):
        # nOAuth: attacker tenant sets email = owner's email; no email_verified.
        self._as(secrets={"admins": ["owner@example.com"]},
                 email="owner@example.com", iss=_MS_COMMON_TENANT, sub="evil")
        self.assertFalse(web_auth.is_admin())

    def test_email_verified_false_is_not_admin(self):
        self._as(secrets={"admins": ["owner@example.com"]},
                 email="owner@example.com", email_verified=False,
                 iss=_GOOGLE, sub="x")
        self.assertFalse(web_auth.is_admin())

    def test_verified_google_email_is_admin(self):
        self._as(secrets={"admins": ["owner@example.com"]},
                 email="Owner@Example.com", email_verified=True,
                 iss=_GOOGLE, sub="123")
        self.assertTrue(web_auth.is_admin())

    def test_iss_sub_binding_is_admin_without_email(self):
        self._as(secrets={"admin_identities": [{"iss": _MS_OWN_TENANT, "sub": "owner-oid"}]},
                 email=None, iss=_MS_OWN_TENANT, sub="owner-oid")
        self.assertTrue(web_auth.is_admin())

    def test_iss_sub_string_form_and_mismatched_issuer(self):
        secrets = {"admin_identities": [f"{_MS_OWN_TENANT}|owner-oid"]}
        self._as(secrets=secrets, iss=_MS_COMMON_TENANT, sub="owner-oid",
                 email="owner@example.com")
        # Same subject from a different issuer (tenant) is a different person.
        self.assertFalse(web_auth.is_admin())

    def test_trusted_single_tenant_issuer_email_is_admin(self):
        self._as(secrets={"admins": ["owner@example.com"],
                          "trusted_email_issuers": [_MS_OWN_TENANT]},
                 email="owner@example.com", iss=_MS_OWN_TENANT, sub="o")
        self.assertTrue(web_auth.is_admin())

    def test_unverified_email_cannot_inherit_paid_tier(self):
        tmp = tempfile.mkdtemp(prefix="noauth_tier_")
        self.addCleanup(shutil.rmtree, tmp, True)
        db = str(Path(tmp) / "subs.db")
        subscriptions._BOOTSTRAPPED_PATHS.discard(db)
        with patch.dict(os.environ, {"SUBSCRIPTIONS_DB_PATH": db}), \
                patch.object(web_auth, "BILLING_ENABLED", True), \
                patch.object(web_auth, "TRIAL_ENABLED", False):
            subscriptions.upsert(email="payer@example.com", tier="premium")
            self._as(secrets={}, email="payer@example.com",
                     iss=_MS_COMMON_TENANT, sub="evil")
            user = web_auth.current_user()
        self.assertTrue(user.is_authenticated)
        self.assertIsNone(user.email)
        self.assertEqual(user.tier, "free")

    def test_tier_for_admin_override_is_identity_bound(self):
        with patch.object(web_auth, "BILLING_ENABLED", True), \
                patch.object(web_auth, "TRIAL_ENABLED", False), \
                patch.object(web_auth, "_admin_emails",
                             return_value={"owner@example.com"}), \
                patch.object(subscriptions, "tier_for", return_value="free"):
            # Email string alone no longer grants premium...
            self.assertEqual(web_auth.tier_for("owner@example.com"), "free")
            # ...only the caller's identity-bound admin flag does.
            self.assertEqual(
                web_auth.tier_for("owner@example.com", admin=True), "premium")



# ---------------------------------------------------------------------------
# 4. Streamlit billing-gate / abuse fixes (review_findings_web.md W-APP-*)
# ---------------------------------------------------------------------------
class PreviewAndThrottleTests(unittest.TestCase):
    def test_preview_match_is_accent_insensitive(self):
        self.assertTrue(web_auth.is_preview_player("Nikola Jokić"))
        self.assertTrue(web_auth.is_preview_player("nikola jokic"))
        self.assertFalse(web_auth.is_preview_player("Stephen Curry"))
        self.assertFalse(web_auth.is_preview_player(None))  # type: ignore[arg-type]

    def test_scan_throttle_applies_in_open_mode(self):
        # Billing off -> everyone is "premium"; the throttle must still bite.
        from nba_model.web import app as web_app
        from nba_model.web import throttle
        calls = {"n": 0}

        def fake_limit(key, max_calls, window_seconds):
            calls["n"] += 1
            return calls["n"] <= max_calls

        with patch.object(web_auth, "is_admin", return_value=False), \
                patch.object(throttle, "rate_limit", side_effect=fake_limit), \
                patch.object(web_app.st, "warning"):
            results = [web_app._scan_throttled("edge_scan", True)
                       for _ in range(web_app.SCAN_BUDGET_PREMIUM + 1)]
        self.assertFalse(any(results[:-1]))
        self.assertTrue(results[-1])

    def test_client_ip_resolution_ignores_spoofed_xff_without_proxy(self):
        from nba_model.web import throttle
        self.assertEqual(throttle.client_ip_from("9.9.9.9", "1.1.1.1", 0), "9.9.9.9")
        # Behind one trusted proxy: right-most XFF entry is the real client.
        self.assertEqual(
            throttle.client_ip_from("10.0.0.5", "6.6.6.6, 203.0.113.7", 1), "203.0.113.7")
        self.assertIsNone(throttle.client_ip_from(None, None, 0))

    def test_new_session_does_not_reset_ip_budget(self):
        from nba_model.web import throttle
        throttle._CLIENT_STORE.clear()
        # Two "sessions" from one IP share the budget (budget keyed by IP).
        results = [throttle.client_rate_limit("203.0.113.7", "edge_scan", 2, 60.0, now=t)
                   for t in (0.0, 1.0, 2.0)]
        self.assertEqual(results, [True, True, False])
        self.assertTrue(throttle.client_rate_limit("198.51.100.1", "edge_scan", 2, 60.0, now=2.0))
        self.assertTrue(throttle.client_rate_limit("203.0.113.7", "edge_scan", 2, 60.0, now=61.5))
        throttle._CLIENT_STORE.clear()

    def test_rate_limit_prefers_ip_over_session(self):
        from nba_model.web import throttle
        with patch.object(throttle, "current_client_ip", return_value="203.0.113.9"), \
                patch.object(throttle, "client_rate_limit", return_value=False) as by_ip, \
                patch.object(throttle, "session_rate_limit", return_value=True) as by_sess:
            self.assertFalse(throttle.rate_limit("k", 1, 60.0))
        by_ip.assert_called_once()
        by_sess.assert_not_called()
        with patch.object(throttle, "current_client_ip", return_value=None), \
                patch.object(throttle, "session_rate_limit", return_value=True) as by_sess:
            self.assertTrue(throttle.rate_limit("k", 1, 60.0))
        by_sess.assert_called_once()

    def test_public_error_hides_exception_text(self):
        from nba_model.web import app as web_app
        with patch.object(web_app.st, "error") as err, \
                self.assertLogs("nba_model.web.app", level="ERROR") as logs:
            web_app._public_error("Edge scan failed",
                                  RuntimeError("no such table: /srv/secret/path.db"))
        shown = err.call_args[0][0]
        self.assertIn("Edge scan failed", shown)
        self.assertNotIn("/srv/secret", shown)
        self.assertIn("ref ", shown)
        # The real detail goes to the server log (with traceback), not the page.
        self.assertIn("/srv/secret", str(logs.records[0].exc_info[1]))

    def test_outbound_nba_api_throttle_per_client_and_global(self):
        from nba_model.web import app as web_app
        from nba_model.web import throttle
        throttle._CLIENT_STORE.clear()
        warn = patch.object(web_app.st, "warning")
        warn.start()
        self.addCleanup(warn.stop)
        with patch.object(web_auth, "is_admin", return_value=False), \
                patch.object(throttle, "current_client_ip", return_value="203.0.113.5"):
            per_client = [web_app._outbound_throttled("single_prop_nba_api")
                          for _ in range(web_app.OUTBOUND_BUDGET_PER_CLIENT + 1)]
        self.assertEqual(per_client[-1], True)
        self.assertFalse(any(per_client[:-1]))
        # Many distinct visitors: the process-wide ceiling still binds.
        throttle._CLIENT_STORE.clear()
        results = []
        with patch.object(web_auth, "is_admin", return_value=False):
            for i in range(web_app.OUTBOUND_BUDGET_GLOBAL + 1):
                with patch.object(throttle, "current_client_ip", return_value=f"198.51.100.{i}"):
                    results.append(web_app._outbound_throttled("single_prop_nba_api"))
        self.assertFalse(any(results[:-1]))
        self.assertTrue(results[-1])
        with patch.object(web_auth, "is_admin", return_value=True):
            self.assertFalse(web_app._outbound_throttled("single_prop_nba_api"))
        throttle._CLIENT_STORE.clear()

    def test_admin_is_exempt_from_scan_throttle(self):
        from nba_model.web import app as web_app
        with patch.object(web_auth, "is_admin", return_value=True):
            self.assertFalse(web_app._scan_throttled("edge_scan", False))


def _compare_script():
    import pandas as _pd
    from nba_model.web import app as _web_app
    df = _pd.DataFrame({"player_name": ["LeBron James", "Nikola Jokić", "Stephen Curry"],
                        "player_id": [2544, 203999, 201939]})
    _web_app._compare_players_view(db_path=":memory:", players_df=df,
                                   default_player="LeBron James",
                                   stat_type="points", n_games=5)


class ComparePlayersGateTests(unittest.TestCase):
    def test_free_tier_picker_lists_only_preview_players(self):
        from streamlit.testing.v1 import AppTest
        free = web_auth.CurrentUser(is_authenticated=True, email="f@x.com",
                                    name="f", tier="free")
        with patch.object(web_auth, "current_user", return_value=free):
            at = AppTest.from_function(_compare_script, default_timeout=30)
            at.run()
        options = at.multiselect[0].options
        self.assertIn("LeBron James", options)
        self.assertIn("Nikola Jokić", options)
        self.assertNotIn("Stephen Curry", options)


class WatchlistStoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="wl_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.path = Path(tmp) / "watchlists.json"
        env = patch.dict(os.environ, {"WATCHLIST_STORE_PATH": str(self.path)})
        env.start()
        self.addCleanup(env.stop)

    def test_corrupt_store_is_not_clobbered(self):
        from nba_model.web import watchlist
        self.path.write_text('{"a@x.com": ["LeBron Ja', encoding="utf-8")  # torn write
        watchlist._save_for("b@x.com", ["Stephen Curry"])
        self.assertEqual(self.path.read_text(encoding="utf-8"), '{"a@x.com": ["LeBron Ja')

    def test_atomic_save_preserves_other_keys_and_caps_length(self):
        from nba_model.web import watchlist
        watchlist._save_for("a@x.com", ["LeBron James"])
        watchlist._save_for("b@x.com", ["x" * 500])
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["a@x.com"], ["LeBron James"])
        self.assertEqual(len(data["b@x.com"][0]), watchlist.MAX_ITEM_LEN)
        # No temp files left behind.
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["watchlists.json"])


    def test_anonymous_keys_expire_signed_in_keys_do_not(self):
        from nba_model.web import watchlist
        day = 86_400
        watchlist._save_for("anon:old", ["LeBron James"], now=0)
        watchlist._save_for("owner@x.com", ["Nikola Jokić"], now=0)
        later = (watchlist.ANON_TTL_DAYS + 1) * day
        watchlist._save_for("anon:new", ["Stephen Curry"], now=later)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("anon:old", data)
        self.assertIn("anon:new", data)
        self.assertIn("owner@x.com", data)
        self.assertNotIn("anon:old", data["__seen__"])

    def test_legacy_unstamped_anon_key_gets_grace_not_deleted(self):
        from nba_model.web import watchlist
        self.path.write_text(json.dumps({"anon:legacy": ["LeBron James"]}), encoding="utf-8")
        watchlist._save_for("anon:other", ["Stephen Curry"], now=10 ** 9)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["anon:legacy"], ["LeBron James"])
        self.assertEqual(data["__seen__"]["anon:legacy"], 10 ** 9)
        self.assertEqual(watchlist._load_for("__seen__"), [])

    def test_pin_rejects_unknown_or_oversized_names(self):
        from nba_model.web import watchlist
        with patch.object(watchlist, "get", return_value=[]), \
                patch.object(watchlist.st, "session_state", {}), \
                patch.object(watchlist, "_persist") as persist:
            known = {"LeBron James"}.__contains__
            self.assertFalse(watchlist.add("<script>x</script>", is_known=known))
            self.assertFalse(watchlist.add("x" * 81, is_known=None))
            self.assertTrue(watchlist.add("LeBron James", is_known=known))
        persist.assert_called_once()



class OperationsEnvScrubTests(unittest.TestCase):
    def test_billing_and_storage_secrets_are_not_inherited(self):
        from nba_model.web import operations_panel as ops
        env = ops.scrubbed_env({
            "PATH": "/usr/bin", "ODDS_API_KEY": "keep-me", "NBA_ALERT_WEBHOOK_URL": "keep",
            "STRIPE_SECRET_KEY": "sk_live_x", "stripe_webhook_secret": "whsec",
            "AWS_SECRET_ACCESS_KEY": "s", "AWS_ACCESS_KEY_ID": "k", "BUCKET_NAME": "b",
            "SUBSCRIPTIONS_DB_URL": "postgres://u:p@h/db", "FLAGSHIP_ACCESS_CODE": "c",
            "BILLING_ALERT_WEBHOOK_URL": "https://hooks", "DB_SYNC_LOCAL_DIR": "/x",
        })
        self.assertEqual(sorted(env), ["NBA_ALERT_WEBHOOK_URL", "ODDS_API_KEY", "PATH"])

    def test_runner_uses_scrubbed_env(self):
        from nba_model.web import operations_panel as ops
        runner = ops._RunnerState()
        captured = {}

        class _FakeProc:
            stdout = iter(())
            returncode = 0

            def poll(self):
                return 0

            def wait(self):
                return 0

        def fake_popen(args, **kw):
            captured.update(kw["env"])
            return _FakeProc()

        with patch.dict(os.environ, {"STRIPE_SECRET_KEY": "sk_live_x", "ODDS_API_KEY": "k"}), \
                patch.object(ops.subprocess, "Popen", side_effect=fake_popen):
            runner.start("t", ["python", "-c", "pass"])
            runner.reader_thread.join(timeout=5)
        self.assertNotIn("STRIPE_SECRET_KEY", captured)
        self.assertEqual(captured["ODDS_API_KEY"], "k")
        self.assertEqual(captured["PYTHONUNBUFFERED"], "1")


if __name__ == "__main__":
    unittest.main()
