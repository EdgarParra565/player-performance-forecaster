"""Stripe Checkout URL helpers + webhook signature verification.

The Streamlit app only needs to BUILD a Checkout URL (one-shot link the user
clicks to upgrade); the actual subscription state is mutated by the
`webhook_app.py` FastAPI process when Stripe sends events. We store the
Checkout URL in `secrets.toml` so a single Stripe Payment Link can be reused
without keeping the Stripe SDK in the Streamlit container.

If a `stripe_secret_key` + `price_id` pair is configured instead, we'll fall
back to creating a Checkout Session at click time via the Stripe API.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st


def _secret(*path: str, default=None):
    """Get a nested secret from secrets.toml, with env-var fallback."""
    try:
        node = st.secrets
        for key in path:
            node = node[key]
        return node
    except (KeyError, FileNotFoundError, AttributeError):
        env_key = "_".join(path).upper()
        return os.environ.get(env_key, default)


def checkout_url(prefill_email: str = "") -> str:
    """Return a URL the user can click to upgrade.

    Resolution order:
        1. [stripe.payment_link_url] in secrets.toml (or STRIPE_PAYMENT_LINK_URL env)
        2. Try to create a Checkout Session via stripe-python (needs
           stripe_secret_key + price_id + success_url).
        3. A "#" placeholder if neither is configured (UI shows the button
           greyed out via try/except when this is hit).
    """
    payment_link = _secret("stripe", "payment_link_url")
    if payment_link:
        sep = "&" if "?" in payment_link else "?"
        if prefill_email:
            return (f"{payment_link}{sep}prefilled_email="
                    f"{prefill_email}")
        return payment_link

    secret_key = _secret("stripe", "secret_key")
    price_id = _secret("stripe", "price_id")
    success_url = _secret("stripe", "success_url",
                          default="https://example.com/?success=1")
    cancel_url = _secret("stripe", "cancel_url",
                         default="https://example.com/?cancelled=1")
    if secret_key and price_id:
        try:
            import stripe  # type: ignore
            stripe.api_key = secret_key
            session = stripe.checkout.Session.create(
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                customer_email=prefill_email or None,
                success_url=success_url,
                cancel_url=cancel_url,
                allow_promotion_codes=True,
            )
            return session.url or "#"
        except Exception:
            return "#"
    return "#"


def portal_return_url() -> str:
    """Where Stripe sends the user back after the Customer Portal.

    Configured via [stripe.portal_return_url] in secrets.toml or
    STRIPE_PORTAL_RETURN_URL env. Falls back to the configured success_url,
    then a placeholder.
    """
    return _secret(
        "stripe", "portal_return_url",
        default=_secret("stripe", "success_url",
                        default="https://example.com/"),
    )


def _stripe_customer_id_for(email: str, stripe_module) -> Optional[str]:
    """Resolve a Stripe customer id for an email.

    Prefers the id we already stored from webhook events (so no Stripe call
    is needed for the common case); falls back to a Stripe lookup by email.
    """
    if not email:
        return None
    try:
        from nba_model.web import subscriptions
        record = subscriptions.lookup(email)
        if record and record.get("stripe_customer_id"):
            return str(record["stripe_customer_id"])
    except Exception:
        pass
    try:
        result = stripe_module.Customer.list(email=email, limit=1)
        data = getattr(result, "data", None)
        if data is None and hasattr(result, "get"):
            data = result.get("data")
        if data:
            first = data[0]
            return first.id if hasattr(first, "id") else first.get("id")
    except Exception:
        pass
    return None


def customer_portal_url(
    email: str = "", customer_id: Optional[str] = None,
) -> Optional[str]:
    """Return a Stripe Customer Portal session URL for a paying user.

    The portal lets the user update their card, view invoices, and cancel
    without contacting us. Returns ``None`` when Stripe isn't configured or
    no customer can be resolved (callers should hide the button then).
    """
    secret = secret_key()
    if not secret:
        return None
    try:
        import stripe  # type: ignore
        stripe.api_key = secret
        cust_id = customer_id or _stripe_customer_id_for(email, stripe)
        if not cust_id:
            return None
        session = stripe.billing_portal.Session.create(
            customer=cust_id,
            return_url=portal_return_url(),
        )
        return getattr(session, "url", None) or None
    except Exception:
        return None


def stripe_mode() -> str:
    """'test' or 'live', from STRIPE_MODE env / [stripe.mode] secret.

    Defaults to 'test' so a misconfiguration can never accidentally charge a
    real card. Anything other than 'live' (case-insensitive) is treated as
    'test'.
    """
    raw = str(_secret("stripe", "mode", default="test") or "test").strip().lower()
    return "live" if raw == "live" else "test"


def _mode_scoped(name: str) -> Optional[str]:
    """Prefer a mode-scoped secret (e.g. secret_key_live) then the plain name.

    Lets an operator keep both test + live keys configured and flip with
    STRIPE_MODE instead of hand-editing keys.
    """
    mode = stripe_mode()
    scoped = _secret("stripe", f"{name}_{mode}")
    if scoped:
        return scoped
    return _secret("stripe", name)


def webhook_secret() -> Optional[str]:
    """Stripe-supplied secret used to verify webhook signatures (mode-scoped)."""
    return _mode_scoped("webhook_secret")


def secret_key() -> Optional[str]:
    """Stripe secret API key for the active mode (test/live)."""
    return _mode_scoped("secret_key")
