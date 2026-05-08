"""Stress + fuzz tests for the security layer.

This is the "we tried to break it" suite, separate from the regression tests
in test_security.py. It does the kinds of things a malicious user would:

- Pound the rate limiter with concurrent threads.
- Race two webhook events updating the same email.
- Try every malformed email shape we can think of.
- Send Stripe-flavored payloads with missing fields.
- Point the subscriptions store at a corrupt file.
- Flood it with 10k unique IPs to verify GC works.

Findings here drive code changes, not just docs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from nba_model.web import subscriptions, webhook_app


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class RateLimiterConcurrencyTests(unittest.TestCase):
    """Many threads hammering the same IP must not exceed the cap."""

    def test_thread_safety_single_ip(self):
        rl = webhook_app.IPRateLimiter(max_requests=50, window_seconds=10)
        now = 1000.0
        allowed = 0
        denied = 0
        lock = threading.Lock()

        def worker():
            nonlocal allowed, denied
            ok = rl.allow("ip1", now)
            with lock:
                if ok:
                    allowed += 1
                else:
                    denied += 1

        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(worker) for _ in range(500)]
            for f in as_completed(futures):
                f.result()

        # Exactly 50 requests should have been allowed; the rest denied.
        self.assertEqual(allowed, 50)
        self.assertEqual(allowed + denied, 500)

    def test_thread_safety_distinct_ips(self):
        rl = webhook_app.IPRateLimiter(max_requests=2, window_seconds=10)
        # 200 distinct IPs each making 5 attempts. Each IP should see exactly
        # 2 allowed + 3 denied.
        now = 1000.0
        results: list[tuple[str, bool]] = []
        rlock = threading.Lock()

        def worker(ip: str):
            nonlocal results
            for _ in range(5):
                ok = rl.allow(ip, now)
                with rlock:
                    results.append((ip, ok))

        ips = [f"10.0.0.{i}" for i in range(200)]
        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(worker, ips))
        # Group by IP and verify per-IP allow count == 2.
        per_ip = {}
        for ip, ok in results:
            per_ip.setdefault(ip, []).append(ok)
        for ip, attempts in per_ip.items():
            self.assertEqual(
                sum(attempts), 2,
                f"{ip} got {sum(attempts)} allows, expected 2",
            )


class SubscriptionConcurrencyTests(unittest.TestCase):
    """Two webhook events upserting the same email must converge."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._old = os.environ.get("SUBSCRIPTIONS_DB_PATH")
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path

    def tearDown(self):
        if self._old is None:
            os.environ.pop("SUBSCRIPTIONS_DB_PATH", None)
        else:
            os.environ["SUBSCRIPTIONS_DB_PATH"] = self._old
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_parallel_upsert_does_not_deadlock(self):
        """16 threads, each upserting 50 different emails. Should not lock up."""
        emails = [f"user{i}@example.com" for i in range(50)]

        def worker(start: int):
            for i in range(50):
                e = emails[(start + i) % 50]
                tier = "premium" if i % 2 == 0 else "free"
                subscriptions.upsert(email=e, tier=tier,
                                     last_event_type="test")

        start = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(worker, range(16)))
        elapsed = time.time() - start
        # Sanity: 16 * 50 = 800 upserts. Even at 50ms each (very pessimistic
        # under SQLite contention) that's < 40s.
        self.assertLess(
            elapsed, 60.0,
            f"parallel upserts took {elapsed:.1f}s; deadlock or thrashing?",
        )
        # Final state for every email is some valid value.
        for e in emails:
            row = subscriptions.lookup(e)
            self.assertIsNotNone(row, f"{e} missing after upserts")
            self.assertIn(row["tier"], {"free", "premium"})


# ---------------------------------------------------------------------------
# Fuzz: email validator
# ---------------------------------------------------------------------------

class EmailFuzzTests(unittest.TestCase):
    """Pathological inputs to the email validator."""

    def test_header_injection_payloads_rejected(self):
        for bad in (
            "alice@example.com\r\nX-Injected: bad",
            "alice@example.com\nbcc: hacker@evil",
            "alice@example.com\x00",
            "alice@example.com\t",
            "alice@\x00example.com",
            "alice\r@example.com",
            "alice@example.com ",
            " alice@example.com",
            "alice@ example.com",
            "alice@example .com",
        ):
            # Strip-then-validate accepts leading/trailing whitespace; we
            # want to confirm the *embedded* whitespace and control bytes
            # never pass.
            try:
                got = subscriptions._validate_email(bad)
                # If it normalizes to a clean address with no embedded
                # whitespace/control chars, that's also acceptable.
                self.assertNotIn("\r", got)
                self.assertNotIn("\n", got)
                self.assertNotIn("\x00", got)
                self.assertNotIn("\t", got)
                self.assertNotIn(" ", got)
            except ValueError:
                pass  # rejection is the preferred outcome

    def test_homoglyph_emails_rejected(self):
        # Cyrillic 'а' (U+0430) looks like Latin 'a' (U+0061). Real systems
        # have been compromised by this. Our regex restricts to ASCII so
        # any non-ASCII letter must be rejected.
        for bad in (
            "alicе@example.com",            # Cyrillic 'е'
            "аdmin@example.com",            # Cyrillic 'а'
            "admin@‮evil.com",         # right-to-left override
            "admin@xn--example.com",        # punycode is allowed by RFC
                                            # but we don't accept it
            "café@example.com",
            "user@münchen.de",
            "あ@example.com",
            "user@example.中国",
        ):
            with self.assertRaises(ValueError, msg=f"accepted: {bad!r}"):
                subscriptions._validate_email(bad)

    def test_boundary_length_emails(self):
        # Exactly 254 chars should pass; 255+ should fail.
        local = "a" * (254 - len("@b.com"))
        ok = local + "@b.com"
        self.assertEqual(len(ok), 254)
        self.assertEqual(subscriptions._validate_email(ok), ok)

        too_long = local + "x@b.com"
        self.assertEqual(len(too_long), 255)
        with self.assertRaises(ValueError):
            subscriptions._validate_email(too_long)

    def test_repeated_at_signs_rejected(self):
        for bad in ("a@b@c.com", "a@@b.com", "@a@b.com"):
            with self.assertRaises(ValueError, msg=f"accepted: {bad!r}"):
                subscriptions._validate_email(bad)

    def test_sql_metacharacters_rejected(self):
        # Each of these would be ugly if it ever reached SQL even though we
        # use parameter binding.
        for bad in (
            'a"b@c.com', "a`b@c.com", "a/*b*/@c.com", "a\\b@c.com",
            "a;b@c.com", "a||b@c.com", "a&b@c.com", "a!b@c.com",
        ):
            with self.assertRaises(ValueError, msg=f"accepted: {bad!r}"):
                subscriptions._validate_email(bad)


# ---------------------------------------------------------------------------
# Webhook payload edge cases
# ---------------------------------------------------------------------------

class WebhookPayloadEdgeTests(unittest.TestCase):
    """Exercise paths after the signature check passes (using stripe.Webhook
    with a known secret)."""

    def setUp(self):
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_stress_test"
        os.environ["STRIPE_SECRET_KEY"] = "sk_test_stress"
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path

        # Reset rate limiter so a previous test's backlog doesn't bleed in.
        webhook_app._rate_limiter._buckets.clear()

        from fastapi.testclient import TestClient
        self.client = TestClient(webhook_app.app)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _signed_post(self, body: dict, ts: int | None = None):
        """POST a body with a real Stripe-style HMAC signature."""
        import hmac
        import hashlib
        # Real Stripe events have a top-level `object: "event"` marker; the
        # SDK uses it to dispatch v1 vs v2 event types. Add it if not present.
        body = dict(body)
        body.setdefault("object", "event")
        payload = json.dumps(body).encode()
        ts = ts or int(time.time())
        secret = os.environ["STRIPE_WEBHOOK_SECRET"].encode()
        signed_payload = f"{ts}.".encode() + payload
        sig = hmac.new(secret, signed_payload, hashlib.sha256).hexdigest()
        header = f"t={ts},v1={sig}"
        return self.client.post(
            "/stripe/webhook",
            content=payload,
            headers={
                "stripe-signature": header,
                "content-type": "application/json",
            },
        )

    def test_event_with_no_email_returns_200_no_email_flag(self):
        body = {
            "id": "evt_no_email",
            "type": "customer.subscription.updated",
            "data": {"object": {}},
        }
        r = self._signed_post(body)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("no_email"))

    def test_event_type_we_dont_handle_is_ignored(self):
        body = {
            "id": "evt_unknown",
            "type": "ping.pong",
            "data": {"object": {"customer_email": "alice@example.com"}},
        }
        r = self._signed_post(body)
        self.assertEqual(r.status_code, 200)
        # Subscription was NOT changed because we don't handle ping.pong.
        self.assertEqual(subscriptions.tier_for("alice@example.com"), "free")

    def test_replay_old_event_rejected(self):
        old = int(time.time()) - 600  # 10 minutes ago, well outside default 300s
        body = {
            "id": "evt_old",
            "type": "customer.subscription.updated",
            "data": {"object": {"customer_email": "stale@example.com"}},
        }
        r = self._signed_post(body, ts=old)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "bad signature")

    def test_premium_promotion_via_signed_event(self):
        future = int(time.time()) + 30 * 24 * 3600
        body = {
            "id": "evt_promote",
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer_email": "newpremium@example.com",
                "subscription": "sub_123",
                "current_period_end": future,
            }},
        }
        r = self._signed_post(body)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            subscriptions.tier_for("newpremium@example.com"), "premium",
        )

    def test_invalid_email_in_event_does_not_poison_db(self):
        body = {
            "id": "evt_bad_email",
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer_email": "<script>alert(1)</script>@evil.com",
                "subscription": "sub_x",
                "current_period_end": int(time.time()) + 1000,
            }},
        }
        r = self._signed_post(body)
        # MUST NOT 500 (Stripe would retry forever). The event is persisted
        # in stripe_events for forensics but no row is added to
        # user_subscriptions.
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("rejected"))
        bad_email = "<script>alert(1)</script>@evil.com"
        self.assertIsNone(subscriptions.lookup(bad_email))


# ---------------------------------------------------------------------------
# DB resilience
# ---------------------------------------------------------------------------

class DBResilienceTests(unittest.TestCase):
    """Failure-mode tests against the subscription store."""

    def test_corrupt_db_does_not_grant_premium(self):
        """If subscriptions.db is unreadable we MUST fail closed (free)."""
        fd, path = tempfile.mkstemp(suffix=".db")
        # Write garbage so sqlite3 can't open it as a DB.
        with os.fdopen(fd, "wb") as f:
            f.write(b"not a sqlite database, definitely not\n")

        try:
            os.environ["SUBSCRIPTIONS_DB_PATH"] = path
            try:
                tier = subscriptions.tier_for("alice@example.com")
            except sqlite3.DatabaseError:
                # Acceptable: caller should treat the exception as 'free'.
                # We surface a real exception rather than silently granting
                # access. The Streamlit app catches and shows a paywall.
                tier = "free"
            self.assertEqual(tier, "free")
        finally:
            os.remove(path)

    def test_missing_table_recreates_schema_returns_free(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        # Open with subscriptions to create schema, then drop the user table
        # and verify a subsequent read recreates and returns free.
        os.environ["SUBSCRIPTIONS_DB_PATH"] = path
        # Force schema creation
        subscriptions.upsert(email="warm@example.com", tier="premium")
        with sqlite3.connect(path) as raw:
            raw.execute("DROP TABLE user_subscriptions")
        # Next read should recreate via _SCHEMA and return free for unknown.
        try:
            tier = subscriptions.tier_for("warm@example.com")
        finally:
            os.remove(path)
        self.assertEqual(tier, "free")

    def test_db_path_does_not_traverse_into_etc(self):
        """An env var pointing at /etc/passwd would still fail because the
        running process can't open it as a sqlite DB - but verify we don't
        crash hard.

        This is here to document the boundary: SUBSCRIPTIONS_DB_PATH is
        operator-controlled, never user-controlled, so any "exploit" via this
        path requires already-root-equivalent access on the host.
        """
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            os.environ["SUBSCRIPTIONS_DB_PATH"] = path
            # Should not crash even if the path was previously empty.
            self.assertEqual(subscriptions.tier_for("any@example.com"), "free")
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# Rate limiter resource exhaustion
# ---------------------------------------------------------------------------

class RateLimiterMemoryTests(unittest.TestCase):
    """Many distinct IPs must not blow up memory; GC must reclaim."""

    def test_many_unique_ips_bounded_after_gc(self):
        rl = webhook_app.IPRateLimiter(max_requests=5, window_seconds=10)
        # 10k distinct IPs at t=0
        for i in range(10_000):
            rl.allow(f"ip{i}", 0.0)
        self.assertGreaterEqual(len(rl._buckets), 10_000)
        # Move time forward past the window, then GC.
        rl.gc(100.0)
        # All buckets are stale -> all should have been pruned.
        self.assertEqual(len(rl._buckets), 0)

    def test_active_buckets_survive_gc(self):
        rl = webhook_app.IPRateLimiter(max_requests=5, window_seconds=10)
        rl.allow("active", 100.0)
        rl.allow("stale", 0.0)
        rl.gc(105.0)
        self.assertIn("active", rl._buckets)
        self.assertNotIn("stale", rl._buckets)


# ---------------------------------------------------------------------------
# Trusted host enforcement
# ---------------------------------------------------------------------------

class TrustedHostTests(unittest.TestCase):
    """If WEBHOOK_TRUSTED_HOSTS is set, requests with other Host headers
    must be rejected.

    Because the middleware is added at module import time, we have to spawn
    a fresh app instance with the env set to assert this."""

    def test_disallowed_host_returns_400(self):
        os.environ["WEBHOOK_TRUSTED_HOSTS"] = "webhook.example.com"
        # Build a fresh FastAPI from the same code path. The webhook_app
        # module reads the env var only at import; we invoke the same setup
        # by hand.
        from fastapi import FastAPI
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        from fastapi.testclient import TestClient
        from nba_model.web.webhook_app import SecurityHeadersMiddleware

        fresh = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        fresh.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["webhook.example.com"],
        )
        fresh.add_middleware(SecurityHeadersMiddleware)

        @fresh.get("/healthz")
        def healthz():
            return {"status": "ok"}

        c = TestClient(fresh)
        r = c.get("/healthz", headers={"host": "evil.example.com"})
        self.assertEqual(r.status_code, 400)


# ---------------------------------------------------------------------------
# Slowloris
# ---------------------------------------------------------------------------

class SlowlorisTests(unittest.TestCase):
    """Slow-stream body should hit the wall-clock timeout, not eat workers
    forever."""

    def test_body_read_timeout_returns_408(self):
        """Force a timeout by patching asyncio.wait_for to raise immediately
        when the webhook reads the body. We don't care about the actual
        duration; we want the handler to convert TimeoutError -> 408."""
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_slow"
        os.environ["STRIPE_SECRET_KEY"] = "sk_test_slow"
        fd, db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["SUBSCRIPTIONS_DB_PATH"] = db
        webhook_app._rate_limiter._buckets.clear()

        from unittest.mock import patch
        from fastapi.testclient import TestClient
        client = TestClient(webhook_app.app)

        async def fake_wait_for(coro, timeout):
            # Cancel the streaming coroutine so we don't leak a task, then
            # surface the timeout the way real asyncio.wait_for would.
            try:
                coro.close()
            except RuntimeError:
                pass
            import asyncio as _a
            raise _a.TimeoutError()

        try:
            with patch("asyncio.wait_for", side_effect=fake_wait_for):
                r = client.post(
                    "/stripe/webhook",
                    content=b"{}",
                    headers={
                        "stripe-signature": "t=1,v1=x",
                        "content-type": "application/json",
                    },
                )
            self.assertEqual(r.status_code, 408)
            self.assertEqual(r.json()["detail"], "body read timeout")
        finally:
            os.remove(db)


# ---------------------------------------------------------------------------
# Signature-header fuzz
# ---------------------------------------------------------------------------

class SignatureHeaderFuzzTests(unittest.TestCase):
    """Try every malformed Stripe-Signature header shape we can think of.
    All must produce a 400, NOT a 500 or a 200."""

    def setUp(self):
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_sig_test"
        os.environ["STRIPE_SECRET_KEY"] = "sk_test_sig"
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path
        webhook_app._rate_limiter._buckets.clear()
        from fastapi.testclient import TestClient
        self.client = TestClient(webhook_app.app)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _post_with_sig(self, sig: str, body: bytes = b"{}"):
        return self.client.post(
            "/stripe/webhook",
            content=body,
            headers={"stripe-signature": sig, "content-type": "application/json"},
        )

    def test_pathological_signatures_rejected(self):
        for sig in (
            "",                             # empty (caught earlier in handler)
            "garbage",                      # no t=, no v1=
            "t=",                           # no value
            "t=abc,v1=def",                 # non-integer t
            "t=999999999999,v1=" + "a" * 64,  # far-future t
            "t=-1,v1=" + "a" * 64,           # negative t
            "v1=abc",                       # no t=
            "t=1," * 100 + "v1=abc",         # spam header
            "t=1,v1=" + "z" * 1000,          # absurd-length sig
            "t=1,v1=NOT_HEX_AT_ALL!",       # non-hex
            "T=1,V1=ABC",                   # wrong case
            "t=1\nv1=abc",                  # embedded newline
        ):
            r = self._post_with_sig(sig)
            # 400 is the right answer for missing/bad signatures.
            self.assertIn(r.status_code, (400, 429),
                          f"unexpected {r.status_code} for {sig!r}")

    def test_signature_with_valid_format_but_wrong_secret_rejected(self):
        import hmac, hashlib
        ts = int(time.time())
        wrong_secret = b"whsec_NOT_THE_REAL_ONE"
        body = b'{"id":"evt_x","object":"event","type":"x","data":{"object":{}}}'
        signed = f"{ts}.".encode() + body
        sig = hmac.new(wrong_secret, signed, hashlib.sha256).hexdigest()
        r = self._post_with_sig(f"t={ts},v1={sig}", body=body)
        self.assertEqual(r.status_code, 400)


# ---------------------------------------------------------------------------
# Subscription expiry edge cases
# ---------------------------------------------------------------------------

class SubscriptionExpiryEdgeTests(unittest.TestCase):
    """Time-based attacks on the premium-grant logic."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_naive_iso_timestamp_treated_as_utc(self):
        """If the period-end string lacks tz info we must NOT misclassify it
        as far-future just because the naive parse defaults to localtime."""
        # 1 second AFTER now in UTC, written without tz suffix.
        future_naive = (datetime.now(timezone.utc).replace(microsecond=0)
                        + __import__("datetime").timedelta(seconds=3600)
                        ).isoformat().split("+")[0]
        subscriptions.upsert(
            email="naive@example.com", tier="premium",
            current_period_end=future_naive,
        )
        # Should still be premium (parser added UTC tz).
        self.assertEqual(subscriptions.tier_for("naive@example.com"), "premium")

    def test_garbage_period_end_does_not_break_lookup(self):
        subscriptions.upsert(
            email="garbage@example.com", tier="premium",
            current_period_end="not-a-date",
        )
        # If the period-end is unparseable we treat it as still-valid (the
        # parser just skips the expiry check). Documented in tier_for.
        self.assertEqual(
            subscriptions.tier_for("garbage@example.com"), "premium",
        )

    def test_unix_epoch_period_end_is_expired(self):
        subscriptions.upsert(
            email="epoch@example.com", tier="premium",
            current_period_end="1970-01-01T00:00:00+00:00",
        )
        self.assertEqual(subscriptions.tier_for("epoch@example.com"), "free")


# ---------------------------------------------------------------------------
# Duplicate-event flood
# ---------------------------------------------------------------------------

class DuplicateEventFloodTests(unittest.TestCase):
    """If an attacker captures one valid signed event they could replay it
    inside the tolerance window. Idempotency must hold and not let them
    escalate or DoS the DB."""

    def setUp(self):
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_dup_test"
        os.environ["STRIPE_SECRET_KEY"] = "sk_test_dup"
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["SUBSCRIPTIONS_DB_PATH"] = self.db_path
        webhook_app._rate_limiter._buckets.clear()
        from fastapi.testclient import TestClient
        self.client = TestClient(webhook_app.app)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_replayed_signed_event_marked_duplicate(self):
        import hmac, hashlib
        future = int(time.time()) + 30 * 24 * 3600
        body = json.dumps({
            "id": "evt_dup",
            "object": "event",
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer_email": "dup@example.com",
                "current_period_end": future,
            }},
        }).encode()
        ts = int(time.time())
        sig = hmac.new(
            b"whsec_dup_test", f"{ts}.".encode() + body, hashlib.sha256,
        ).hexdigest()
        headers = {
            "stripe-signature": f"t={ts},v1={sig}",
            "content-type": "application/json",
        }
        # First call processes; subsequent N must all be flagged duplicate
        # AND must NOT mutate the DB further.
        r1 = self.client.post("/stripe/webhook", content=body, headers=headers)
        self.assertEqual(r1.status_code, 200)
        self.assertNotIn("duplicate", r1.json())

        for _ in range(10):
            r = self.client.post("/stripe/webhook", content=body, headers=headers)
            if r.status_code == 429:
                # rate limiter kicked in; that's OK - means the replay flood
                # got bounced for an even stronger reason.
                break
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json().get("duplicate"))


# ---------------------------------------------------------------------------
# Adversarial numeric input fuzz (model/data integrity per OWASP AI Top 10)
# ---------------------------------------------------------------------------

class AdversarialInputTests(unittest.TestCase):
    """Defends the statistical model against the OWASP-AI threats that DO
    apply to a non-ML system: input manipulation (`evasion attacks` in the
    OWASP-ML language) and data poisoning. Our `fitted_prob_over` /
    `expected_value` / Monte Carlo paths could otherwise produce silent
    garbage or OverflowError on hostile inputs."""

    def test_validate_line_rejects_nan_inf(self):
        from nba_model.web import input_validation as iv
        for bad in (float("nan"), float("inf"), float("-inf"), "nan",
                    "inf", "-inf", "infinity"):
            with self.assertRaises(iv.ValidationError, msg=f"accepted {bad!r}"):
                iv.validate_line("points", bad)

    def test_validate_line_rejects_out_of_range(self):
        from nba_model.web import input_validation as iv
        # Way too high: 9999 points (no real game has this)
        with self.assertRaises(iv.ValidationError):
            iv.validate_line("points", 9999)
        # Negative: definitely poisoned
        with self.assertRaises(iv.ValidationError):
            iv.validate_line("points", -10)

    def test_validate_line_per_stat_ranges(self):
        from nba_model.web import input_validation as iv
        # Plausible: should pass.
        for stat, line in (("points", 25.5), ("assists", 8.5),
                           ("rebounds", 12.5), ("pra", 50.0),
                           ("three_pointers_made", 3.5),
                           ("field_goals_made", 10.5), ("minutes", 35.0)):
            self.assertEqual(iv.validate_line(stat, line), float(line))
        # Implausible per stat (would be valid for points but not for these)
        with self.assertRaises(iv.ValidationError):
            iv.validate_line("assists", 50)
        with self.assertRaises(iv.ValidationError):
            iv.validate_line("three_pointers_made", 25)
        with self.assertRaises(iv.ValidationError):
            iv.validate_line("minutes", 80)

    def test_validate_american_odds_rejects_zero_and_extremes(self):
        from nba_model.web import input_validation as iv
        with self.assertRaises(iv.ValidationError):
            iv.validate_american_odds(0)
        with self.assertRaises(iv.ValidationError):
            iv.validate_american_odds(float("inf"))
        with self.assertRaises(iv.ValidationError):
            iv.validate_american_odds(10_000_000)
        # Edge cases that should pass.
        self.assertEqual(iv.validate_american_odds(-110), -110)
        self.assertEqual(iv.validate_american_odds(+150), 150)
        self.assertEqual(iv.validate_american_odds("even"), 100)
        self.assertIsNone(iv.validate_american_odds(None))
        self.assertIsNone(iv.validate_american_odds(""))

    def test_n_games_is_capped(self):
        from nba_model.web import input_validation as iv
        self.assertEqual(iv.validate_n_games(25), 25)
        # Compute-DoS attempt: a free-tier user asks for 1M games.
        self.assertEqual(iv.validate_n_games(1_000_000),
                         iv.N_GAMES_HARD_CAP)
        with self.assertRaises(iv.ValidationError):
            iv.validate_n_games(0)
        with self.assertRaises(iv.ValidationError):
            iv.validate_n_games(float("nan"))

    def test_n_sims_is_capped_with_floor(self):
        from nba_model.web import input_validation as iv
        # Default when nothing supplied
        self.assertEqual(iv.validate_n_sims(None), 20_000)
        # Floor = 1000, attacker can't go below
        self.assertEqual(iv.validate_n_sims(10), 1000)
        # Hard cap at 200k, attacker can't go above
        self.assertEqual(iv.validate_n_sims(10_000_000), 200_000)

    def test_parlay_legs_count_bounds(self):
        from nba_model.web import input_validation as iv
        with self.assertRaises(iv.ValidationError):
            iv.validate_parlay_legs_count(1)
        with self.assertRaises(iv.ValidationError):
            iv.validate_parlay_legs_count(99)
        self.assertEqual(iv.validate_parlay_legs_count(2), 2)
        self.assertEqual(iv.validate_parlay_legs_count(6), 6)


# ---------------------------------------------------------------------------
# Ingestion data-poisoning defense
# ---------------------------------------------------------------------------

class IngestionPoisoningTests(unittest.TestCase):
    """If a scraped sportsbook page were hijacked, what would survive the
    insert path? The is_plausible_betting_line guard should drop the bad rows
    before they reach the betting_lines table."""

    def test_sane_lines_pass(self):
        from nba_model.web.input_validation import is_plausible_betting_line
        self.assertTrue(is_plausible_betting_line("points", 25.5,
                                                  over_odds=-110, under_odds=-110))
        self.assertTrue(is_plausible_betting_line("pra", 49.5))

    def test_poisoned_lines_dropped(self):
        from nba_model.web.input_validation import is_plausible_betting_line
        # Each of these would have polluted the consensus mean if accepted.
        for stat, line in (
            ("points", 9999),       # absurd
            ("points", -5),         # negative
            ("points", float("inf")),
            ("points", float("nan")),
            ("rebounds", 200),      # impossible
            ("three_pointers_made", 99),
            ("minutes", 90),        # > 60
            ("garbage_stat", 25),   # unknown stat
        ):
            self.assertFalse(
                is_plausible_betting_line(stat, line),
                f"accepted poison row: {stat}={line}",
            )

    def test_poisoned_odds_drop_row(self):
        from nba_model.web.input_validation import is_plausible_betting_line
        for over, under in (
            (0, -110),                       # 0 odds
            (float("inf"), -110),
            (10_000_000, 10_000_000),         # absurd
        ):
            self.assertFalse(
                is_plausible_betting_line("points", 25.5,
                                          over_odds=over, under_odds=under),
                f"accepted poison odds: over={over}, under={under}",
            )


# ---------------------------------------------------------------------------
# Compute-DoS defense via the validators
# ---------------------------------------------------------------------------

class ComputeBoundsTests(unittest.TestCase):
    """Ensure the validators cap any expensive request before it reaches
    the Monte Carlo / chart-data layer."""

    def test_evaluate_custom_line_with_validated_inputs_is_bounded(self):
        """fitted_prob_over with adversarial line + sigma=0 must NOT raise."""
        from nba_model.visualization.player_charts import (
            PlayerChartData, fitted_prob_over,
        )
        import numpy as np
        import pandas as pd
        # Sigma = 0 (constant series). Any line still produces a defined P.
        data = PlayerChartData(
            player_id=1, player_name="x", stat_type="points",
            games=pd.DataFrame(),
            values=np.array([20.0, 20.0, 20.0]),
        )
        self.assertIsNotNone(fitted_prob_over(data, 25.0))
        self.assertIsNotNone(fitted_prob_over(data, 15.0))
        # Empty values: returns None, not crash.
        empty = PlayerChartData(
            player_id=1, player_name="x", stat_type="points",
            games=pd.DataFrame(), values=np.array([]),
        )
        self.assertIsNone(fitted_prob_over(empty, 25.0))


if __name__ == "__main__":
    unittest.main()
