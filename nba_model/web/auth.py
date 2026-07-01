"""Authentication + tier resolution for the Streamlit app.

Streamlit 1.42+ ships native OIDC login. We wrap it so the rest of the app
calls a small, stable surface (`current_user`, `is_authenticated`, `tier_for`,
`require_premium`, `paywall`) instead of touching `st.user` directly.

OIDC providers (Google, Microsoft) are configured in `.streamlit/secrets.toml`
- see `.streamlit/secrets.toml.example` for the template.

================================================================================
BILLING FEATURE FLAG (2026-05-11)
================================================================================
`BILLING_ENABLED` controls whether the Free/Premium paywall is active. While
we're validating product-market fit, set this to False to give every visitor
full access without sign-in. Flip to True (or set env BILLING_ENABLED=1) to
re-engage the existing Stripe + OIDC + tier-gating code — no other change
needed; all that logic is still here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from nba_model.web import subscriptions

TIER_FREE = "free"
TIER_PREMIUM = "premium"

# Master switch. When False (the launch default), every visitor is treated as
# premium and no paywall ever renders. When True, the full Free/Premium gating
# kicks back in. Env var takes precedence so deploys can flip it without a
# code change.
_FLAG = os.environ.get("BILLING_ENABLED", "").strip().lower()
BILLING_ENABLED: bool = _FLAG in {"1", "true", "yes", "on"}

# Optional first-sign-in free trial. Off by default; flip with ENABLE_TRIAL=1.
_TRIAL_FLAG = os.environ.get("ENABLE_TRIAL", "").strip().lower()
TRIAL_ENABLED: bool = _TRIAL_FLAG in {"1", "true", "yes", "on"}
TRIAL_DAYS: int = int(os.environ.get("TRIAL_DAYS", "7") or "7")


def trial_active(created_at, now=None, trial_days: int = TRIAL_DAYS) -> bool:
    """True if ``created_at`` (ISO ts of first sign-in) is within the trial.

    Pure + testable. Wiring this into live tier resolution requires a
    ``created_at`` (or ``trial_start``) timestamp on the subscription record —
    the current ``user_subscriptions`` schema only has ``updated_at``, so add
    that column (DEFAULT datetime('now')) before relying on this in production.
    Returns False on missing / unparseable input so it always fails closed.
    """
    if not created_at:
        return False
    from datetime import datetime, timedelta, timezone
    try:
        start = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now < start + timedelta(days=int(trial_days))

# Players + teams a NOT-logged-in or free-tier user is allowed to view in
# preview mode. Keep this short and high-profile so the app is still useful
# as a teaser without giving away the marquee. Only used when BILLING_ENABLED.
PREVIEW_PLAYERS: tuple[str, ...] = (
    "Nikola Jokic",
    "LeBron James",
)
PREVIEW_TEAMS: tuple[str, ...] = (
    "LAL", "DEN",
)
PREVIEW_STATS: tuple[str, ...] = ("points",)
PREVIEW_MAX_GAMES = 5


@dataclass(frozen=True)
class CurrentUser:
    is_authenticated: bool
    email: Optional[str]
    name: Optional[str]
    tier: str  # 'free' or 'premium'

    @property
    def is_premium(self) -> bool:
        return self.tier == TIER_PREMIUM


def _streamlit_user():
    """Best-effort access to st.user (1.42+). Returns None if unavailable."""
    user = getattr(st, "user", None)
    if user is None:
        return None
    return user


def is_authenticated() -> bool:
    user = _streamlit_user()
    if user is None:
        return False
    return bool(getattr(user, "is_logged_in", False))


def _admin_emails() -> set[str]:
    """Emails listed in [auth.admins] of secrets.toml are forced to premium."""
    try:
        emails = st.secrets.get("auth", {}).get("admins", [])
    except (AttributeError, FileNotFoundError):
        emails = []
    return {str(e).strip().lower() for e in emails if e}


def is_admin() -> bool:
    """True if the current user's email is in the admins allowlist.

    Used to gate developer-only UI surfaces (e.g. the DB path override) so a
    free-tier visitor can't aim the app at an arbitrary SQLite file.
    """
    user = _streamlit_user()
    if user is None or not getattr(user, "is_logged_in", False):
        return False
    email = getattr(user, "email", "") or ""
    return str(email).strip().lower() in _admin_emails()


def tier_for(email: Optional[str]) -> str:
    """Resolve a user's current tier.

    Order of precedence:
        0. BILLING_ENABLED is False (the launch default) -> everyone is premium
        1. anonymous -> free
        2. email in [auth.admins] -> premium (override for the dev/owner)
        3. ENABLE_TRIAL=1 and the email is within TRIAL_DAYS of first sign-in
           -> premium (anchors `created_at` on first call when no row exists)
        4. subscriptions table entry with active premium -> premium
        5. otherwise -> free
    """
    if not BILLING_ENABLED:
        return TIER_PREMIUM
    if not email:
        return TIER_FREE
    email_lower = email.strip().lower()
    if email_lower in _admin_emails():
        return TIER_PREMIUM
    if TRIAL_ENABLED:
        row = subscriptions.lookup(email_lower)
        if row is None:
            # First sign-in: anchor created_at NOW so the trial clock starts
            # on this very call. The row is created free-tier; the upgrade
            # path will overwrite tier=premium when Stripe webhook lands.
            subscriptions.touch_first_seen(email_lower)
            return TIER_PREMIUM
        if trial_active(row.get("created_at")):
            return TIER_PREMIUM
    return subscriptions.tier_for(email_lower)


def current_user() -> CurrentUser:
    # When billing is disabled, every visitor is treated as a (pseudo-)premium
    # user without any login state. The OIDC + Stripe code still exists and is
    # exercised the moment BILLING_ENABLED flips back on.
    if not BILLING_ENABLED:
        return CurrentUser(is_authenticated=False, email=None, name=None,
                           tier=TIER_PREMIUM)
    user = _streamlit_user()
    if user is None or not getattr(user, "is_logged_in", False):
        return CurrentUser(is_authenticated=False, email=None, name=None,
                           tier=TIER_FREE)
    email = getattr(user, "email", None)
    name = getattr(user, "name", None) or email
    return CurrentUser(
        is_authenticated=True,
        email=str(email).strip() if email else None,
        name=str(name) if name else None,
        tier=tier_for(email),
    )


def login_buttons(label_prefix: str = "Sign in with") -> None:
    """Render Google + Microsoft login buttons (requires st.login)."""
    if not hasattr(st, "login"):
        st.error(
            "This Streamlit version is too old for native OIDC. "
            "Upgrade to streamlit>=1.42."
        )
        return
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{label_prefix} Google", use_container_width=True,
                     key="login_google_btn"):
            st.login("google")
    with cols[1]:
        if st.button(f"{label_prefix} Microsoft", use_container_width=True,
                     key="login_microsoft_btn"):
            st.login("microsoft")


def render_user_card(sidebar=True) -> None:
    """Show the current user's email + tier badge + login/logout buttons.

    When BILLING_ENABLED is False (launch default), renders nothing - we
    don't want to clutter the sidebar with a 'sign in' prompt that does
    nothing useful while the paywall is disabled.
    """
    if not BILLING_ENABLED:
        return
    container = st.sidebar if sidebar else st
    user = current_user()
    if user.is_authenticated:
        with container:
            st.markdown(
                f"**Signed in:** `{user.email}`\n\n"
                f"**Tier:** "
                + (":star: Premium" if user.is_premium else ":lock: Free")
            )
            if not user.is_premium:
                container.link_button(
                    "Upgrade to Premium",
                    _checkout_url_for(user.email or ""),
                    use_container_width=True,
                )
            else:
                portal_url = _portal_url_for(user.email or "")
                if portal_url:
                    container.link_button(
                        "Manage subscription",
                        portal_url,
                        use_container_width=True,
                    )
            if hasattr(st, "logout"):
                container.button(
                    "Sign out", on_click=st.logout, key="logout_btn",
                    use_container_width=True,
                )
    else:
        with container:
            st.markdown("**Not signed in** (preview mode)")
            login_buttons()


def _checkout_url_for(email: str) -> str:
    """Return the Stripe Checkout URL configured in secrets, with email pre-fill."""
    from nba_model.web import stripe_helpers
    return stripe_helpers.checkout_url(prefill_email=email)


def _portal_url_for(email: str):
    """Return a Stripe Customer Portal URL for a premium user, or None."""
    from nba_model.web import stripe_helpers
    return stripe_helpers.customer_portal_url(email=email)


def paywall(feature: str, allow_preview: bool = True) -> None:
    """Render the standard paywall message; caller should `return` after this.

    `feature` is a short label like "Parlay analysis".
    `allow_preview` controls whether we hint at the preview the free tier gets.

    When BILLING_ENABLED is False, this is a no-op so the caller never
    actually paywalls anything (current_user().is_premium is always True
    in that mode, so callers won't reach paywall(...) anyway — this guard
    is belt-and-suspenders).
    """
    if not BILLING_ENABLED:
        return
    user = current_user()
    if not user.is_authenticated:
        st.warning(
            f":lock: **{feature}** requires a Premium membership. "
            "Sign in first, then upgrade."
        )
        login_buttons()
        return
    st.warning(
        f":lock: **{feature}** is a Premium feature. "
        f"You're signed in as `{user.email}` on the Free tier."
    )
    st.link_button(
        "Upgrade to Premium",
        _checkout_url_for(user.email or ""),
    )
    if allow_preview:
        st.caption(
            "Free preview includes: charts for "
            + ", ".join(PREVIEW_PLAYERS)
            + f" (up to last {PREVIEW_MAX_GAMES} games, points only)."
        )


def gate_player(player_name: str) -> bool:
    """Return True if the given player is viewable for the current user."""
    if current_user().is_premium:
        return True
    return player_name.strip() in PREVIEW_PLAYERS


def gate_team(team_code: str) -> bool:
    """Return True if the given team is viewable for the current user."""
    if current_user().is_premium:
        return True
    return team_code.strip().upper() in PREVIEW_TEAMS


def cap_n_games(n: int) -> int:
    """Cap last-N-games at the preview limit for non-premium users."""
    if current_user().is_premium:
        return int(n)
    return min(int(n), PREVIEW_MAX_GAMES)


def allowed_stats() -> tuple[str, ...]:
    """Stat list non-premium users can pick from."""
    if current_user().is_premium:
        return ()  # caller substitutes the full list
    return PREVIEW_STATS
