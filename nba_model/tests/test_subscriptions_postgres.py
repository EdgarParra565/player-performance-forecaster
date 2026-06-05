"""Tests for the Postgres-selectable subscription store.

Exercises:
- Env-based dispatch (``SUBSCRIPTIONS_DB_URL`` chooses Postgres; absent falls
  through to SQLite).
- Caller-override dispatch (``db_path=`` arg with a postgres URL).
- Postgres CRUD path with the driver fully mocked (no live Postgres needed).
- Friendly RuntimeError when psycopg isn't installed.
"""

from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

from nba_model.web import subscriptions


SAMPLE_PG_URL = "postgresql://user:pw@db.example.com:5432/subs"


class BackendSelectionTests(unittest.TestCase):
    def test_env_var_postgres_url_selects_postgres(self):
        with patch.dict(
            "os.environ",
            {"SUBSCRIPTIONS_DB_URL": SAMPLE_PG_URL},
            clear=False,
        ):
            self.assertEqual(subscriptions.selected_backend(), "postgres")

    def test_env_var_unset_falls_back_to_sqlite(self):
        env = {k: v for k, v in __import__("os").environ.items()
               if k != "SUBSCRIPTIONS_DB_URL"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(subscriptions.selected_backend(), "sqlite")

    def test_db_path_override_with_postgres_url(self):
        self.assertEqual(
            subscriptions.selected_backend(SAMPLE_PG_URL), "postgres"
        )

    def test_db_path_override_with_sqlite_path(self):
        self.assertEqual(
            subscriptions.selected_backend("/tmp/subscriptions_test.db"),
            "sqlite",
        )

    def test_postgresql_plus_psycopg_url_scheme_is_recognized(self):
        self.assertTrue(
            subscriptions._is_postgres_url("postgresql+psycopg://x@host/db")
        )

    def test_non_postgres_urls_treated_as_sqlite_filepaths(self):
        for value in ("", "/var/db/subs.db", "subscriptions.db", "https://x"):
            self.assertFalse(subscriptions._is_postgres_url(value))


class PostgresMissingDriverTests(unittest.TestCase):
    def test_helpful_error_when_psycopg_not_installed(self):
        # Ensure no cached psycopg lives in sys.modules for this scope.
        with patch.dict(sys.modules, {"psycopg": None}):
            with self.assertRaises(RuntimeError) as ctx:
                subscriptions._import_psycopg()
            self.assertIn("psycopg", str(ctx.exception))
            self.assertIn("pip install", str(ctx.exception))


def _install_fake_psycopg() -> tuple[ModuleType, MagicMock, MagicMock]:
    """Inject a stub `psycopg` module into sys.modules and return helpers."""
    fake = ModuleType("psycopg")

    # Cursor: tracks executed SQL + parameter tuples.
    cursor = MagicMock(name="cursor")
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchone = MagicMock(return_value=None)
    cursor.description = [
        MagicMock(name="email"), MagicMock(name="tier"),
        MagicMock(name="stripe_customer_id"),
        MagicMock(name="stripe_subscription_id"),
        MagicMock(name="current_period_end"),
        MagicMock(name="last_event_type"),
        MagicMock(name="updated_at"),
    ]
    for i, n in enumerate((
        "email", "tier", "stripe_customer_id", "stripe_subscription_id",
        "current_period_end", "last_event_type", "updated_at",
    )):
        cursor.description[i].name = n

    conn = MagicMock(name="conn")
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.cursor = MagicMock(return_value=cursor)
    conn.close = MagicMock()

    fake.connect = MagicMock(return_value=conn)

    # Errors namespace for UniqueViolation handling
    errors_mod = ModuleType("psycopg.errors")
    class UniqueViolation(Exception):
        pass
    errors_mod.UniqueViolation = UniqueViolation
    fake.errors = errors_mod

    sys.modules["psycopg"] = fake
    sys.modules["psycopg.errors"] = errors_mod
    return fake, conn, cursor


class PostgresCrudPathTests(unittest.TestCase):
    """Exercise upsert / tier_for / record_stripe_event / lookup over the
    Postgres backend with the driver stubbed — we want to verify the SQL
    parameters + dispatch shape, not the network."""

    def setUp(self):
        # Reset the per-DSN bootstrap cache so each test fires the schema
        # CREATE on its own MagicMock connection.
        subscriptions._PG_BOOTSTRAPPED_DSNS.clear()
        self._fake, self._conn, self._cursor = _install_fake_psycopg()

    def tearDown(self):
        sys.modules.pop("psycopg", None)
        sys.modules.pop("psycopg.errors", None)
        subscriptions._PG_BOOTSTRAPPED_DSNS.clear()

    def test_upsert_emits_postgres_dialect_sql(self):
        subscriptions.upsert(
            "user@example.com",
            tier="premium",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            current_period_end="2099-01-01T00:00:00+00:00",
            last_event_type="checkout.session.completed",
            db_path=SAMPLE_PG_URL,
        )
        # Find the upsert call (skip the schema-create + session-tuning calls).
        upsert_call = None
        for call in self._cursor.execute.call_args_list:
            sql = call.args[0]
            if "INSERT INTO user_subscriptions" in sql:
                upsert_call = call
                break
        self.assertIsNotNone(upsert_call)
        self.assertIn("ON CONFLICT (email) DO UPDATE", upsert_call.args[0])
        # parameter binding uses %s (Postgres style), not ? (sqlite)
        self.assertNotIn("?", upsert_call.args[0])
        params = upsert_call.args[1]
        self.assertEqual(params[0], "user@example.com")
        self.assertEqual(params[1], "premium")
        self.assertEqual(params[2], "cus_test")

    def test_tier_for_returns_premium_when_postgres_row_is_active(self):
        self._cursor.fetchone.return_value = ("premium", "2099-01-01T00:00:00+00:00")
        tier = subscriptions.tier_for("user@example.com", db_path=SAMPLE_PG_URL)
        self.assertEqual(tier, "premium")

    def test_tier_for_returns_free_when_postgres_row_expired(self):
        self._cursor.fetchone.return_value = ("premium", "1999-01-01T00:00:00+00:00")
        tier = subscriptions.tier_for("user@example.com", db_path=SAMPLE_PG_URL)
        self.assertEqual(tier, "free")

    def test_tier_for_returns_free_when_no_postgres_row(self):
        self._cursor.fetchone.return_value = None
        tier = subscriptions.tier_for("user@example.com", db_path=SAMPLE_PG_URL)
        self.assertEqual(tier, "free")

    def test_record_stripe_event_returns_false_on_unique_violation(self):
        def _execute_side_effect(sql, *args, **kwargs):
            if "INSERT INTO stripe_events" in sql:
                raise self._fake.errors.UniqueViolation("duplicate")
            return None
        self._cursor.execute.side_effect = _execute_side_effect
        result = subscriptions.record_stripe_event(
            "evt_123", "checkout.session.completed", '{"ok":1}',
            db_path=SAMPLE_PG_URL,
        )
        self.assertFalse(result)

    def test_lookup_returns_dict_row_when_postgres_row_present(self):
        self._cursor.fetchone.return_value = (
            "user@example.com", "premium", "cus_x", "sub_x",
            "2099-01-01T00:00:00+00:00", "checkout.session.completed",
            "2026-06-04T00:00:00+00:00",
        )
        row = subscriptions.lookup("user@example.com", db_path=SAMPLE_PG_URL)
        self.assertIsNotNone(row)
        self.assertEqual(row["email"], "user@example.com")
        self.assertEqual(row["tier"], "premium")
        self.assertEqual(row["stripe_customer_id"], "cus_x")


class EmailValidationStillEnforcedOnPostgresPath(unittest.TestCase):
    """The email-validation gate must run before backend dispatch so a
    garbage email never reaches the network."""

    def test_upsert_rejects_punycode_before_touching_db(self):
        # Inject a stub psycopg so we'd notice if dispatch happened.
        fake, conn, cursor = _install_fake_psycopg()
        try:
            subscriptions._PG_BOOTSTRAPPED_DSNS.clear()
            with self.assertRaises(ValueError):
                subscriptions.upsert(
                    "evil@xn--pple-43d.com",
                    tier="premium",
                    db_path=SAMPLE_PG_URL,
                )
            fake.connect.assert_not_called()
        finally:
            sys.modules.pop("psycopg", None)
            sys.modules.pop("psycopg.errors", None)
            subscriptions._PG_BOOTSTRAPPED_DSNS.clear()


if __name__ == "__main__":
    unittest.main()
