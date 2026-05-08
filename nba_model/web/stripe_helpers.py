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


def webhook_secret() -> Optional[str]:
    """Stripe-supplied secret used to verify webhook signatures."""
    return _secret("stripe", "webhook_secret")


def secret_key() -> Optional[str]:
    return _secret("stripe", "secret_key")
