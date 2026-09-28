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

# Players a NOT-logged-in or free-tier user is allowed to view in preview
# mode. (Team charts are a free surface for everyone; the old team/stat gate
# helpers were never wired up and were removed in the open-access pass.) Keep this short and high-profile so the app is still useful
# as a teaser without giving away the marquee. Only used when BILLING_ENABLED.
PREVIEW_PLAYERS: tuple[str, ...] = (
    "Nikola Jokic",
    "LeBron James",
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


def _auth_secrets() -> dict:
    try:
        section = st.secrets.get("auth", {})
    except (AttributeError, FileNotFoundError):
        return {}
    return section or {}


def _admin_emails() -> set[str]:
    """Emails listed in [auth].admins of secrets.toml (legacy allowlist).

    SECURITY: only honoured when the signed-in identity's email is VERIFIED
    (see ``_verified_email``). Prefer ``[auth].admin_identities``.
    """
    emails = _auth_secrets().get("admins", []) or []
    return {str(e).strip().lower() for e in emails if e}


def _admin_identities() -> set[tuple[str, str]]:
    """(issuer, subject) pairs from ``[auth].admin_identities``.

    Each entry is either an inline table ``{ iss = "...", sub = "..." }`` or a
    ``"<iss>|<sub>"`` string. ``iss`` + ``sub`` together are the only OIDC
    claims that are globally unique and not user-editable, so this is the
    preferred way to bind admin rights to a person.
    """
    out: set[tuple[str, str]] = set()
    for entry in _auth_secrets().get("admin_identities", []) or []:
        iss = sub = None
        if isinstance(entry, str) and "|" in entry:
            iss, sub = entry.split("|", 1)
        else:
            try:
                iss, sub = entry.get("iss"), entry.get("sub")
            except AttributeError:
                continue
        if iss and sub:
            out.add((str(iss).strip(), str(sub).strip()))
    return out


def _trusted_email_issuers() -> set[str]:
    """Issuers whose ``email`` claim is trusted without ``email_verified``.

    Microsoft Entra ID tokens carry no ``email_verified`` claim, and in the
    multi-tenant (``/common``) configuration ANY tenant admin can set a user's
    email to an arbitrary address (the "nOAuth" pattern). Only list a
    single-tenant issuer you control here, e.g.
    ``https://login.microsoftonline.com/<your-tenant-id>/v2.0``.
    """
    issuers = _auth_secrets().get("trusted_email_issuers", []) or []
    return {str(i).strip().rstrip("/") for i in issuers if i}


def _claim(user, name: str):
    """Read an OIDC claim off ``st.user`` (attribute- or mapping-style)."""
    val = getattr(user, name, None)
    if val is None:
        try:
            val = user[name]
        except (KeyError, TypeError, AttributeError):
            val = None
    return val


def _is_true(val) -> bool:
    return val is True or str(val).strip().lower() == "true"


def _verified_email(user) -> Optional[str]:
    """The signed-in user's email, ONLY if the IdP asserts it is verified.

    Accepted when the token carries ``email_verified: true`` (Google) or the
    token's issuer is explicitly listed in ``[auth].trusted_email_issuers``.
    Otherwise returns None — an unverified email must never unlock admin,
    premium, or a Stripe customer-portal session.
    """
    email = str(_claim(user, "email") or "").strip().lower()
    if not email:
        return None
    if _is_true(_claim(user, "email_verified")):
        return email
    iss = str(_claim(user, "iss") or "").strip().rstrip("/")
    if iss and iss in _trusted_email_issuers():
        return email
    return None


def is_admin() -> bool:
    """True if the signed-in identity is an allowlisted admin.

    Used to gate developer-only UI surfaces (DB path override, Operations,
    manual-lines saves) so a visitor can't reach them.

    SECURITY: matching on the ``email`` claim alone allowed admin takeover via
    a multi-tenant Microsoft account with an attacker-chosen email. Admin is
    now granted only when either
      1. the token's (``iss``, ``sub``) pair is in ``[auth].admin_identities``, or
      2. the email is in ``[auth].admins`` AND is verified (``_verified_email``).
    See docs/SECURITY.md → "Admin identity binding".
    """
    user = _streamlit_user()
    if user is None or not getattr(user, "is_logged_in", False):
        return False
    iss = str(_claim(user, "iss") or "").strip()
    sub = str(_claim(user, "sub") or "").strip()
    if iss and sub and (iss, sub) in _admin_identities():
        return True
    email = _verified_email(user)
    return bool(email) and email in _admin_emails()


def tier_for(email: Optional[str], *, admin: bool = False) -> str:
    """Resolve a user's current tier.

    ``email`` must be a VERIFIED email (``current_user`` passes
    ``_verified_email``); ``admin`` is the caller's ``is_admin()`` result.
    The admin premium override is identity-bound — it no longer keys off the
    email string, which an attacker could claim.

    Order of precedence:
        0. BILLING_ENABLED is False (the launch default) -> everyone is premium
        1. admin (identity-bound, see ``is_admin``) -> premium
        2. anonymous / no verified email -> free
        3. ENABLE_TRIAL=1 and the email is within TRIAL_DAYS of first sign-in
           -> premium (anchors `created_at` on first call when no row exists)
        4. subscriptions table entry with active premium -> premium
        5. otherwise -> free
    """
    if not BILLING_ENABLED:
        return TIER_PREMIUM
    if admin:
        return TIER_PREMIUM
    if not email:
        return TIER_FREE
    email_lower = email.strip().lower()
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
    # Only a verified email is trusted for tier lookup, checkout prefill and
    # the Stripe customer portal (portal sessions are keyed by email).
    email = _verified_email(user)
    name = getattr(user, "name", None) or email
    return CurrentUser(
        is_authenticated=True,
        email=email,
        name=str(name) if name else None,
        tier=tier_for(email, admin=is_admin()),
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
    if user.is_authenticated and not user.email and not user.is_premium:
        with container:
            st.warning(
                "Signed in, but your identity provider did not return a "
                "verified email, so billing can't be linked to this account. "
                "Sign in with Google, or ask the operator to trust your "
                "organisation's tenant."
            )
            if hasattr(st, "logout"):
                container.button(
                    "Sign out", on_click=st.logout, key="logout_btn",
                    use_container_width=True,
                )
        return
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


def _name_key(name: str) -> str:
    import unicodedata
    folded = unicodedata.normalize("NFKD", str(name or ""))
    return "".join(c for c in folded if not unicodedata.combining(c)).casefold().strip()


_PREVIEW_KEYS = frozenset(_name_key(p) for p in PREVIEW_PLAYERS)


def is_preview_player(player_name: str) -> bool:
    """Accent/case-insensitive preview check: the DB stores "Nikola Jokić",
    the allowlist says "Nikola Jokic" — an exact match silently dropped him."""
    return _name_key(player_name) in _PREVIEW_KEYS


def gate_player(player_name: str) -> bool:
    """Return True if the given player is viewable for the current user."""
    if current_user().is_premium:
        return True
    return is_preview_player(player_name)
