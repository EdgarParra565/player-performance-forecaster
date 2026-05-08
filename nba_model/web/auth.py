"""Authentication + tier resolution for the Streamlit app.

Streamlit 1.42+ ships native OIDC login. We wrap it so the rest of the app
calls a small, stable surface (`current_user`, `is_authenticated`, `tier_for`,
`require_premium`, `paywall`) instead of touching `st.user` directly.

OIDC providers (Google, Microsoft) are configured in `.streamlit/secrets.toml`
- see `.streamlit/secrets.toml.example` for the template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from nba_model.web import subscriptions

TIER_FREE = "free"
TIER_PREMIUM = "premium"

# Players + teams a NOT-logged-in or free-tier user is allowed to view in
# preview mode. Keep this short and high-profile so the app is still useful
# as a teaser without giving away the marquee.
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
        1. anonymous -> free
        2. email in [auth.admins] -> premium (override for the dev/owner)
        3. subscriptions table entry with active premium -> premium
        4. otherwise -> free
    """
    if not email:
        return TIER_FREE
    email_lower = email.strip().lower()
    if email_lower in _admin_emails():
        return TIER_PREMIUM
    return subscriptions.tier_for(email_lower)


def current_user() -> CurrentUser:
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
    """Show the current user's email + tier badge + login/logout buttons."""
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


def paywall(feature: str, allow_preview: bool = True) -> None:
    """Render the standard paywall message; caller should `return` after this.

    `feature` is a short label like "Parlay analysis".
    `allow_preview` controls whether we hint at the preview the free tier gets.
    """
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
