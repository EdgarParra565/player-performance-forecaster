"""Tests for Stripe helper URL builders — Customer Portal in particular.

The Stripe SDK is replaced with a fake module injected into ``sys.modules``
so ``import stripe`` inside the helpers resolves to it; no network, no real
keys. Secrets and the subscriptions backend are patched per test.
"""

import sys
import types
import unittest
from unittest.mock import patch

from nba_model.web import stripe_helpers


def _fake_stripe(portal_url="https://billing.stripe.test/session/abc",
                 list_customer_id=None, raise_on_create=False):
    """Build a stand-in `stripe` module."""
    mod = types.ModuleType("stripe")
    mod.api_key = None

    class _Session:
        last_kwargs = {}

        @staticmethod
        def create(**kwargs):
            _Session.last_kwargs = kwargs
            if raise_on_create:
                raise RuntimeError("stripe boom")
            obj = types.SimpleNamespace(url=portal_url)
            return obj

    mod.billing_portal = types.SimpleNamespace(Session=_Session)

    class _Customer:
        @staticmethod
        def list(**kwargs):
            data = []
            if list_customer_id:
                data = [types.SimpleNamespace(id=list_customer_id)]
            return types.SimpleNamespace(data=data)

    mod.Customer = _Customer
    mod._Session = _Session  # expose for assertions
    return mod


class CustomerPortalUrlTests(unittest.TestCase):
    def test_returns_none_without_secret_key(self):
        with patch.object(stripe_helpers, "secret_key", return_value=None):
            self.assertIsNone(
                stripe_helpers.customer_portal_url(email="a@b.com")
            )

    def test_uses_explicit_customer_id(self):
        fake = _fake_stripe()
        with patch.object(stripe_helpers, "secret_key", return_value="sk_test"), \
                patch.object(stripe_helpers, "portal_return_url",
                             return_value="https://app.test/back"), \
                patch.dict(sys.modules, {"stripe": fake}):
            url = stripe_helpers.customer_portal_url(customer_id="cus_123")
        self.assertEqual(url, "https://billing.stripe.test/session/abc")
        self.assertEqual(fake._Session.last_kwargs["customer"], "cus_123")
        self.assertEqual(
            fake._Session.last_kwargs["return_url"], "https://app.test/back"
        )

    def test_resolves_stored_customer_id_from_subscriptions(self):
        fake = _fake_stripe()
        from nba_model.web import subscriptions
        with patch.object(stripe_helpers, "secret_key", return_value="sk_test"), \
                patch.object(subscriptions, "lookup",
                             return_value={"stripe_customer_id": "cus_stored"}), \
                patch.dict(sys.modules, {"stripe": fake}):
            url = stripe_helpers.customer_portal_url(email="paid@user.com")
        self.assertIsNotNone(url)
        self.assertEqual(fake._Session.last_kwargs["customer"], "cus_stored")

    def test_falls_back_to_stripe_customer_lookup(self):
        fake = _fake_stripe(list_customer_id="cus_fromapi")
        from nba_model.web import subscriptions
        with patch.object(stripe_helpers, "secret_key", return_value="sk_test"), \
                patch.object(subscriptions, "lookup", return_value=None), \
                patch.dict(sys.modules, {"stripe": fake}):
            url = stripe_helpers.customer_portal_url(email="paid@user.com")
        self.assertIsNotNone(url)
        self.assertEqual(fake._Session.last_kwargs["customer"], "cus_fromapi")

    def test_returns_none_when_no_customer_resolvable(self):
        fake = _fake_stripe(list_customer_id=None)
        from nba_model.web import subscriptions
        with patch.object(stripe_helpers, "secret_key", return_value="sk_test"), \
                patch.object(subscriptions, "lookup", return_value=None), \
                patch.dict(sys.modules, {"stripe": fake}):
            url = stripe_helpers.customer_portal_url(email="unknown@user.com")
        self.assertIsNone(url)

    def test_swallows_stripe_errors(self):
        fake = _fake_stripe(raise_on_create=True)
        with patch.object(stripe_helpers, "secret_key", return_value="sk_test"), \
                patch.dict(sys.modules, {"stripe": fake}):
            url = stripe_helpers.customer_portal_url(customer_id="cus_123")
        self.assertIsNone(url)


class PortalReturnUrlTests(unittest.TestCase):
    def test_explicit_portal_return_url_wins(self):
        def fake_secret(*path, default=None):
            if path == ("stripe", "portal_return_url"):
                return "https://configured.test/return"
            return default
        with patch.object(stripe_helpers, "_secret", side_effect=fake_secret):
            self.assertEqual(
                stripe_helpers.portal_return_url(),
                "https://configured.test/return",
            )

    def test_falls_back_to_success_url(self):
        def fake_secret(*path, default=None):
            if path == ("stripe", "portal_return_url"):
                return default  # not configured -> returns the nested default
            if path == ("stripe", "success_url"):
                return "https://configured.test/success"
            return default
        with patch.object(stripe_helpers, "_secret", side_effect=fake_secret):
            self.assertEqual(
                stripe_helpers.portal_return_url(),
                "https://configured.test/success",
            )


if __name__ == "__main__":
    unittest.main()
