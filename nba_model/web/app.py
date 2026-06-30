"""Streamlit player-charts web app.

Run with:
    .venv/bin/python3 -m streamlit run nba_model/web/app.py

Sidebar lets you switch DB / team / player / stat / window / overlays. The
main pane re-renders every chart and the EV summary on selection change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Allow `streamlit run nba_model/web/app.py` from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from nba_model.model import edge_scanner as es
from nba_model.model.simulation import SUPPORTED_DISTRIBUTIONS
from nba_model.visualization import player_charts as pc
from nba_model.visualization import plotly_charts as plc
from nba_model.web import auth as web_auth
from nba_model.web import etl_status
from nba_model.web import input_validation as iv
from nba_model.web import parlay_compare as parlay
from nba_model.web import watchlist as wl
from sports import SPORTS as _ALL_SPORTS, get_sport as _get_sport

# Available distribution-family options. We source from
# ``simulation.SUPPORTED_DISTRIBUTIONS`` so the selectors automatically match
# whatever vocabulary the model team ships, plus ``negative_binomial`` which
# the chart overlay supports directly even if the simulator hasn't added it yet.
_OVERLAY_DISTRIBUTIONS: tuple[str, ...] = tuple(
    list(SUPPORTED_DISTRIBUTIONS) + (
        ["negative_binomial"]
        if "negative_binomial" not in SUPPORTED_DISTRIBUTIONS
        else []
    )
)


# ---------------------------------------------------------------------------
# URL query-string state
# ---------------------------------------------------------------------------
#
# We use `st.query_params` (mutable dict-like, Streamlit 1.30+) so URLs like
#   https://.../?player=Nikola+Jokic&stat=points&n_games=25&view=Player+charts
# are shareable and the browser back/forward button preserves selections.
# Each input below reads its default from `_qp_get(...)` and writes back via
# `_qp_set(...)` when the user changes it.
_QP_PLAYER = "player"
_QP_TEAM   = "team"
_QP_STAT   = "stat"
_QP_NGAMES = "n_games"
_QP_VIEW   = "view"
_QP_ROLL   = "rolling"
_QP_SPORT  = "sport"


def _qp_get(key: str, default: str = "") -> str:
    """Read a single query-param value as a string."""
    try:
        return str(st.query_params.get(key, default) or default)
    except Exception:  # noqa: BLE001
        return default


def _qp_get_int(key: str, default: int) -> int:
    try:
        return int(_qp_get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _qp_set(**kwargs) -> None:
    """Write query-params; pass None / '' to clear a key.

    Wrapped because `st.query_params` raises on missing keys and we want
    set + clear to look the same to the caller.
    """
    try:
        for k, v in kwargs.items():
            if v is None or v == "":
                if k in st.query_params:
                    del st.query_params[k]
            else:
                st.query_params[k] = str(v)
    except Exception:  # noqa: BLE001
        # Older Streamlit versions in CI; silently ignore so the app still loads.
        pass


def _index_or_zero(choices, value):
    """Helper: return the index of `value` in `choices`, or 0 if absent."""
    try:
        return list(choices).index(value)
    except (ValueError, TypeError):
        return 0


def _csv_download_button(df: pd.DataFrame, label: str, filename: str,
                         key: Optional[str] = None) -> None:
    """Compact `st.download_button` wrapper that emits a CSV of `df`.

    No-op when `df` is empty (we don't want a button that downloads an empty
    file). Buttons for distinct tables need distinct `key`s if they share a
    label, but the label is usually unique enough.
    """
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
    )

DEFAULT_DB_PATH = "data/database/nba_data.db"
STAT_CHOICES = (
    "points", "assists", "rebounds", "pra",
    "three_pointers_made", "field_goals_made",
    "minutes",
)


# TTL bounds how long a running session can show a stale dropdown after the
# hourly ETL scrapes new game logs. Chart data itself (fetch_player_chart_data)
# is uncached and already reflects the latest DB on every rerun; these caches
# only back the team/player pickers, so a 5-minute TTL lets newly-scraped
# players/teams surface automatically without a manual "Reload DB indexes".
_INDEX_CACHE_TTL_SECONDS = 300


@st.cache_data(show_spinner=False, ttl=_INDEX_CACHE_TTL_SECONDS)
def _cached_teams(db_path: str) -> list[str]:
    return pc.list_teams(db_path)


@st.cache_data(show_spinner=False, ttl=_INDEX_CACHE_TTL_SECONDS)
def _cached_team_codes(db_path: str) -> list[str]:
    """Distinct team codes parsed from game_logs.matchup (works even when
    players.team is unset, which is the case for almost every row today)."""
    return pc.list_team_codes(db_path)


@st.cache_data(show_spinner=False, ttl=_INDEX_CACHE_TTL_SECONDS)
def _cached_players(db_path: str, team: str) -> pd.DataFrame:
    return pc.list_players_with_data(db_path, team=team or None)


def _resolve_player_id(players_df: pd.DataFrame, name: str):
    if players_df.empty or not name:
        return None, name
    match = players_df[players_df["player_name"].str.casefold() == name.casefold()]
    if match.empty:
        return None, name
    row = match.iloc[0]
    pid = row.get("player_id")
    try:
        pid = int(pid) if pd.notna(pid) else None
    except (TypeError, ValueError):
        pid = None
    return pid, str(row.get("player_name") or name)


def _kpi_row(data: pc.PlayerChartData) -> None:
    cols = st.columns(6)
    n = int(data.values.size)
    cols[0].metric("games", n)
    cols[1].metric("player mean (mu)", f"{data.mu:.2f}" if n else "—")
    cols[2].metric("player std (sigma)", f"{data.sigma:.2f}" if n else "—")

    consensus = pc.compute_market_consensus(data)
    if consensus["mean"] is not None:
        cols[3].metric(
            f"book mean ({consensus['n_books']} books)",
            f"{consensus['mean']:.3f}",
        )
        cols[4].metric("book sigma (across books)",
                       f"{consensus['stdev']:.3f}")
    else:
        cols[3].metric("book mean", "—")
        cols[4].metric("book sigma", "—")

    if data.book_lines is not None and not data.book_lines.empty and n:
        plus_ev = 0
        for _, row in data.book_lines.iterrows():
            try:
                lv = float(row["line_value"])
            except (TypeError, ValueError):
                continue
            p = pc.fitted_prob_over(data, lv)
            if p is None:
                continue
            ev_o = pc.expected_value(p, row.get("over_odds"))
            ev_u = pc.expected_value(1 - p, row.get("under_odds"))
            candidates = [v for v in (ev_o, ev_u) if v is not None]
            if candidates and max(candidates) > 0:
                plus_ev += 1
        cols[5].metric("+EV book sides", plus_ev)
    else:
        cols[5].metric("+EV book sides", "—")


def _books_panel(data: pc.PlayerChartData) -> None:
    """Show which books contributed + which are missing for the current selection."""
    consensus = pc.compute_market_consensus(data)
    used = consensus["books_used"]
    missing = consensus["books_missing"]
    if not used and not missing:
        return
    with st.expander(
        f"Books used: {len(used)}  |  missing: {len(missing)}", expanded=False
    ):
        c1, c2 = st.columns(2)
        c1.markdown("**Used in consensus**")
        if used:
            c1.write(", ".join(used))
        else:
            c1.info("No books contributed a line for this player + stat.")
        c2.markdown("**Missing (expected but absent)**")
        if missing:
            c2.warning(", ".join(missing))
        else:
            c2.success("All expected books present.")
        if consensus["mean"] is not None:
            st.caption(
                f"Consensus mean = {consensus['mean']:.3f}  |  "
                f"sigma across books = {consensus['stdev']:.3f}  |  "
                f"spread (max-min) = {consensus['spread']:.3f}"
            )
    _render_line_movement(data)


def _render_line_movement(data: pc.PlayerChartData) -> None:
    """Surface per-book line moves drawn from web_prop_cards history."""
    movement = getattr(data, "line_movement", None)
    if not movement:
        return
    st.markdown("**Line movement (last 48h)**")
    for mv in movement:
        icon = "📈" if mv["direction"] == "up" else "📉"
        st.caption(
            f"{icon} {mv['text']}  "
            f"(Δ {mv['delta']:+g} over {mv['n_points']} obs)"
        )


def _render_book_lines_table(data: pc.PlayerChartData) -> None:
    if data.book_lines is None or data.book_lines.empty:
        st.info("No book lines stored for this player + stat.")
        return
    consensus = pc.compute_market_consensus(data)
    pct_lookup = {p["book"].lower(): p for p in consensus["per_book"]}
    n = int(data.values.size) or 1
    rows = []
    for _, row in data.book_lines.iterrows():
        try:
            lv = float(row["line_value"])
        except (TypeError, ValueError):
            continue
        p = pc.fitted_prob_over(data, lv)
        ev_o = pc.expected_value(p, row.get("over_odds")) if p is not None else None
        ev_u = pc.expected_value(1 - p, row.get("under_odds")) if p is not None else None
        book_name = str(row.get("book") or "")
        entry = pct_lookup.get(book_name.lower())
        rows.append({
            "book": book_name,
            "line": lv,
            "% from mean": entry["pct_from_mean_str"] if entry else "",
            "delta": round(entry["delta"], 3) if entry else None,
            "over_odds": row.get("over_odds"),
            "under_odds": row.get("under_odds"),
            "P(over)": None if p is None else round(p, 4),
            "EV_over": None if ev_o is None else round(ev_o, 4),
            "EV_under": None if ev_u is None else round(ev_u, 4),
            "hit_rate": round(float((data.values > lv).sum()) / n, 4),
        })
    if not rows:
        st.info("No numeric book lines.")
        return
    book_lines_df = pd.DataFrame(rows)
    st.dataframe(
        book_lines_df,
        use_container_width=True,
        column_config={
            "P(over)": st.column_config.ProgressColumn(
                "P(over)", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "hit_rate": st.column_config.ProgressColumn(
                "hit_rate", min_value=0.0, max_value=1.0, format="%.0f%%"),
        },
        hide_index=True,
    )
    _csv_download_button(
        book_lines_df,
        label="Download book lines as CSV",
        filename=(
            f"{data.player_name.replace(' ', '_')}_"
            f"{data.stat_type}_book_lines.csv"
        ),
        key=f"csv_book_lines_{data.player_id}_{data.stat_type}",
    )
    if consensus["books_missing"]:
        st.caption(
            ":warning: Missing books for this player + stat: "
            + ", ".join(consensus["books_missing"])
        )


def _custom_line_panel(data: pc.PlayerChartData) -> None:
    with st.form("custom_line_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        default_line = (
            f"{data.market_median_line:.1f}"
            if data.market_median_line is not None else ""
        )
        line_text = c1.text_input("Hypothetical line", value=default_line)
        odds_text = c2.text_input("American odds (over)", value="-110")
        submitted = c3.form_submit_button("Evaluate")
    if not submitted:
        return
    if not line_text.strip():
        st.warning("Enter a line value.")
        return
    # SECURITY: validators reject NaN, inf, and out-of-range adversarial inputs.
    try:
        line_val = iv.validate_line(data.stat_type, line_text)
    except iv.ValidationError as exc:
        st.error(f"Invalid line: {exc}")
        return
    try:
        odds_val = iv.validate_american_odds(odds_text)
    except iv.ValidationError as exc:
        st.error(f"Invalid odds: {exc}")
        return

    res = pc.evaluate_custom_line(data, line_val, american_odds=odds_val)
    p = res["p_over"]
    cols = st.columns(4)
    cols[0].metric("fitted P(over)",
                   "—" if p is None else f"{p:.1%}")
    cols[1].metric("historical over",
                   f"{res['historical_over_rate']:.0%}",
                   help=f"{res['hits']} of {res['n']} games over {res['line']}")
    if res["ev_over_per_unit"] is not None:
        cols[2].metric("EV / unit (over)",
                       f"{res['ev_over_per_unit']:+.3f}")
        cols[3].metric("EV / unit (under)",
                       f"{res['ev_under_per_unit']:+.3f}")
        best_over = res["ev_over_per_unit"] >= res["ev_under_per_unit"]
        best_ev = max(res["ev_over_per_unit"], res["ev_under_per_unit"])
        verdict = "+EV" if best_ev > 0 else "-EV"
        side = "OVER" if best_over else "UNDER"
        st.success(f"Best side: **{side}** — {verdict} ({best_ev:+.3f} per unit)")

        # Kelly stake suggestion for the best side (full + half-Kelly).
        kelly = res["kelly_over"] if best_over else res["kelly_under"]
        if kelly and kelly > 0:
            st.caption(
                f"Kelly stake ({side}): **{kelly:.1%}** of bankroll "
                f"(half-Kelly {kelly / 2:.1%}). Full Kelly is aggressive — "
                "many bettors use ¼–½."
            )
        else:
            st.caption("Kelly stake: 0% — no edge at these odds, no bet.")

    # Probe history: keep a running list per-session so power users can
    # iterate through lines and then export the whole audit as CSV.
    from datetime import datetime as _dt
    history = st.session_state.setdefault("custom_line_history", [])
    history.append({
        "ts": _dt.utcnow().isoformat(timespec="seconds"),
        "player": data.player_name,
        "stat": data.stat_type,
        "line": line_val,
        "odds": odds_val,
        "p_over": p,
        "historical_over_rate": res["historical_over_rate"],
        "ev_over": res["ev_over_per_unit"],
        "ev_under": res["ev_under_per_unit"],
        "n_games": res["n"],
    })
    # Cap history so a long session doesn't grow unbounded.
    if len(history) > 100:
        del history[: len(history) - 100]
    with st.expander(f"Probe history ({len(history)} entries)", expanded=False):
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
        _csv_download_button(
            hist_df,
            label="Download probe history as CSV",
            filename="custom_line_probe_history.csv",
            key="csv_probe_history",
        )
        if st.button("Clear history", key="clear_probe_history"):
            st.session_state["custom_line_history"] = []
            st.rerun()


ALL_STATS_DEFAULT = (
    "points", "assists", "rebounds", "pra",
    "three_pointers_made", "field_goals_made",
)


def _all_stats_overview(
    db_path: str,
    player_id,
    player_name: str,
    n_games: int,
    overlays: tuple[str, ...],
    rolling: int,
    stat_types: tuple[str, ...] = ALL_STATS_DEFAULT,
) -> None:
    """Render one section per stat: KPI row + distribution + hit-rate side by side."""
    st.subheader(f"{player_name} — all stats overview (last {n_games} games)")

    for stat_type in stat_types:
        try:
            data = pc.fetch_player_chart_data(
                db_path=db_path,
                player_id=player_id,
                player_name=player_name,
                stat_type=stat_type,
                n_games=n_games,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{stat_type}: failed to load — {exc}")
            continue

        st.markdown(f"### {stat_type}")
        if data.values.size == 0:
            st.info(f"No game logs for {stat_type}.")
            if data.notes:
                for note in data.notes:
                    st.caption(note)
            st.divider()
            continue

        # Compact KPI row per stat
        n = int(data.values.size)
        consensus = pc.compute_market_consensus(data)
        book_mean = consensus["mean"]
        book_sigma = consensus["stdev"]
        n_books = consensus["n_books"]
        consensus_hit = (
            float((data.values > book_mean).sum()) / n
            if book_mean is not None and n else None
        )
        kpi = st.columns(6)
        kpi[0].metric("games", n)
        kpi[1].metric("player mu", f"{data.mu:.2f}")
        kpi[2].metric("player sigma", f"{data.sigma:.2f}")
        kpi[3].metric(
            f"book mean ({n_books} books)",
            f"{book_mean:.3f}" if book_mean is not None else "—",
        )
        kpi[4].metric(
            "book sigma",
            f"{book_sigma:.3f}" if book_sigma is not None else "—",
        )
        kpi[5].metric(
            "hit% @ book mean",
            f"{consensus_hit:.0%}" if consensus_hit is not None else "—",
        )

        # Books used / missing inline (so all 4 stats are scannable on one page).
        used = consensus["books_used"]
        missing = consensus["books_missing"]
        line_parts = []
        if used:
            line_parts.append(f"**Used ({len(used)}):** {', '.join(used)}")
        if missing:
            line_parts.append(
                f":warning: **Missing ({len(missing)}):** "
                + ", ".join(missing)
            )
        if line_parts:
            st.markdown("  |  ".join(line_parts))

        # Two charts side by side: distribution + hit-rate
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption("Distribution + book-line markers (hover / zoom enabled)")
            st.plotly_chart(
                plc.build_distribution_figure(data, distributions=overlays),
                use_container_width=True,
                key=f"plotly_dist_overview_{data.player_id}_{data.stat_type}",
            )
        with c_right:
            st.caption("Hit-rate vs each book line")
            st.plotly_chart(
                plc.build_hit_rate_figure(data),
                use_container_width=True,
                key=f"plotly_hitrate_overview_{data.player_id}_{data.stat_type}",
            )

        # Optional third row: recent-games trend (collapsed to save space)
        with st.expander(f"Recent games trend ({stat_type})"):
            st.plotly_chart(
                plc.build_recent_games_figure(
                    data, rolling_window=int(rolling)),
                use_container_width=True,
                key=f"plotly_recent_overview_{data.player_id}_{data.stat_type}",
            )

        st.divider()


TEAM_STAT_CHOICES = (
    "points", "rebounds", "assists", "pra",
    "three_pointers_made", "field_goals_made",
)


def _team_overview(
    db_path: str,
    team: str,
    n_games: int,
    overlays: tuple[str, ...],
    rolling: int,
    stat_types: tuple[str, ...] = TEAM_STAT_CHOICES,
) -> None:
    """All team stats on one page: per-stat KPI + recent + distribution."""
    st.subheader(
        f"{team} - team distributions across last {n_games} games "
        "(aggregated from tracked players)"
    )
    st.caption(
        ":information_source: Team values are SUMS across the players the DB has "
        "logs for. Per the audit, this DB has 5-12 players per team rather than "
        "full rosters, so absolute totals will run lower than real team totals. "
        "Distributional shape and game-to-game variance are still meaningful."
    )

    for stat_type in stat_types:
        try:
            data = pc.fetch_team_chart_data(
                db_path=db_path, team=team,
                stat_type=stat_type, n_games=n_games,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{stat_type}: failed to load - {exc}")
            continue

        st.markdown(f"### {stat_type}")
        if data.values.size == 0:
            st.info(f"No team-level rows for {stat_type}.")
            for note in data.notes:
                st.caption(note)
            st.divider()
            continue

        avg_players = (
            float(data.games["players_in_game"].mean())
            if "players_in_game" in data.games.columns else None
        )
        kpi = st.columns(5)
        kpi[0].metric("games", int(data.values.size))
        kpi[1].metric("team mu", f"{data.mu:.2f}")
        kpi[2].metric("team sigma", f"{data.sigma:.2f}")
        kpi[3].metric("min / max",
                      f"{int(data.values.min())} / {int(data.values.max())}")
        kpi[4].metric(
            "avg tracked players/game",
            f"{avg_players:.1f}" if avg_players is not None else "—",
        )

        c_left, c_right = st.columns(2)
        with c_left:
            st.caption("Recent games + rolling mean")
            st.plotly_chart(
                plc.build_recent_games_figure(
                    data, rolling_window=int(rolling)),
                use_container_width=True,
                key=f"plotly_recent_team_{data.player_id}_{data.stat_type}",
            )
        with c_right:
            st.caption("Distribution + fitted overlays (hover / zoom enabled)")
            st.plotly_chart(
                plc.build_distribution_figure(data, distributions=overlays),
                use_container_width=True,
                key=f"plotly_dist_team_{data.player_id}_{data.stat_type}",
            )

        with st.expander(f"Splits ({stat_type})"):
            st.plotly_chart(
                plc.build_splits_figure(data),
                use_container_width=True,
                key=f"plotly_splits_team_{data.player_id}_{data.stat_type}",
            )

        st.divider()


def _parlay_view(
    db_path: str,
    player_id,
    player_name: str,
    n_games: int,
) -> None:
    """Cross-compare model + chart-data + historical for single + multi-leg props."""
    st.subheader(f"Parlay analysis - {player_name}")
    st.caption(
        "Compares THREE views of the same prop: (1) the model "
        "(`run_single_prop` / `run_parlay_demo` - external NBA-API + defense + "
        "minutes adjustment), (2) chart data (local DB game logs + fitted "
        "normal), (3) historical hit-rate over the same window. "
        "For multi-leg, the chart-data path uses an independence approximation; "
        "the model uses a correlated Gaussian SGP."
    )

    sub_mode = st.radio(
        "Parlay type",
        ["Single prop", "Multi-leg parlay"],
        horizontal=True,
        index=0,
    )

    if sub_mode == "Single prop":
        _parlay_single(db_path, player_id, player_name, n_games)
    else:
        _parlay_multi(db_path, player_id, player_name, n_games)


def _parlay_single(
    db_path: str, player_id, player_name: str, n_games: int,
) -> None:
    with st.form("parlay_single_form"):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        stat = c1.selectbox("Stat", STAT_CHOICES, index=0)
        line = c2.number_input("Line", value=25.5, step=0.5, format="%.1f")
        odds = c3.number_input("American odds", value=-110, step=5)
        rolling = c4.number_input("Model rolling window",
                                  min_value=2, max_value=30, value=10, step=1)
        c5, c6, c7, c8 = st.columns(4)
        opp_def = c5.number_input("Opp def rating", value=112.5, step=0.5)
        spread = c6.number_input("Vegas spread", value=0.0, step=0.5)
        distribution = c7.selectbox(
            "Model distribution",
            list(SUPPORTED_DISTRIBUTIONS),
            index=0,
            help="Sourced from `simulation.SUPPORTED_DISTRIBUTIONS`.",
        )
        run_model_too = c8.checkbox(
            "Run external model too",
            value=False,
            help=(
                "Calls run_single_prop which hits the NBA API + applies the "
                "model's minutes/defense adjustment. Slower than chart-data "
                "only. Disable if you just want the local-DB view."
            ),
        )
        submitted = st.form_submit_button("Compare")

    if not submitted:
        return

    # SECURITY: validate adversarial numeric inputs before they feed the math.
    try:
        validated_line = iv.validate_line(stat, line)
        validated_odds = iv.validate_american_odds(odds)
        validated_ngames = iv.validate_n_games(n_games)
    except iv.ValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return

    # Always compute the chart-data + book view first (fast; uses local DB).
    chart_summary = parlay.chart_single_summary(
        db_path=db_path,
        player_id=player_id,
        player_name=player_name,
        stat_type=stat,
        line=validated_line,
        n_games=validated_ngames,
        american_odds=validated_odds,
    )

    rows = []
    rows.append({
        "method": "Chart data (local DB, fitted normal)",
        "mu": chart_summary["mu"],
        "sigma": chart_summary["sigma"],
        "P(over)": chart_summary["p_over_normal"],
        "EV / unit": chart_summary["ev_over_normal"],
        "n": chart_summary["n"],
    })
    rows.append({
        "method": f"Historical over-rate (last {chart_summary['n']} games)",
        "mu": None,
        "sigma": None,
        "P(over)": chart_summary["historical_over_rate"],
        "EV / unit": chart_summary["ev_over_historical"],
        "n": chart_summary["n"],
    })
    if chart_summary["book_mean"] is not None:
        rows.append({
            "method": (
                f"Book consensus ({len(chart_summary['books_used'])} books)"
            ),
            "mu": chart_summary["book_mean"],
            "sigma": chart_summary["book_sigma"],
            "P(over)": None,
            "EV / unit": None,
            "n": len(chart_summary["books_used"]),
        })

    model_result = None
    if run_model_too:
        try:
            from nba_model.run_model import run_single_prop
            with st.spinner(
                f"Calling run_single_prop for {player_name} (NBA API)..."
            ):
                model_result = run_single_prop(
                    player_name=player_name,
                    line=validated_line,
                    rolling_window=int(rolling),
                    american_odds=validated_odds or -110,
                    opp_def_rating=float(opp_def),
                    vegas_spread=float(spread),
                    distribution=distribution,
                )
            rows.insert(0, {
                "method": (
                    f"Model run_single_prop ({model_result['distribution']})"
                ),
                "mu": model_result["mu"],
                "sigma": model_result["sigma_adjusted"],
                "P(over)": model_result["prob_over"],
                "EV / unit": model_result["ev"],
                "n": int(rolling),
            })
        except Exception as exc:  # noqa: BLE001
            st.warning(
                f"Model run_single_prop failed: {exc}. "
                "Showing chart-data only."
            )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True, hide_index=True,
        column_config={
            "P(over)": st.column_config.ProgressColumn(
                "P(over)", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "EV / unit": st.column_config.NumberColumn(
                "EV / unit", format="%+.3f"),
            "mu": st.column_config.NumberColumn("mu", format="%.3f"),
            "sigma": st.column_config.NumberColumn("sigma", format="%.3f"),
        },
    )

    # Summary verdict
    p_norm = chart_summary["p_over_normal"]
    p_hist = chart_summary["historical_over_rate"]
    p_model = model_result["prob_over"] if model_result else None
    probs = [(name, p) for name, p in (
        ("model", p_model), ("chart", p_norm), ("hist", p_hist),
    ) if p is not None]
    if probs:
        st.markdown("**Verdict**")
        for name, p in probs:
            ev = _ev_quick(p, validated_odds)
            st.write(
                f"- `{name}` P(over) = **{p:.1%}**, EV @ "
                f"{validated_odds if validated_odds is not None else 'n/a'} "
                f"= **{ev:+.3f}**" + (" :rocket: +EV" if ev and ev > 0 else "")
            )

    # Show book lines for context
    if chart_summary["per_book"]:
        st.markdown("**Books currently posting this stat**")
        st.dataframe(
            pd.DataFrame(chart_summary["per_book"]),
            hide_index=True, use_container_width=True,
        )
    if chart_summary["books_missing"]:
        st.caption(
            ":warning: Missing books: "
            + ", ".join(chart_summary["books_missing"])
        )


def _parlay_multi(
    db_path: str, player_id, player_name: str, n_games: int,
) -> None:
    n_legs = st.slider("Number of legs", min_value=2, max_value=6, value=3)

    legs: list[parlay.LegSpec] = []
    cols_per_row = st.columns(n_legs)
    default_lines = {
        "points": 25.5, "assists": 5.5, "rebounds": 8.5, "pra": 35.5,
        "three_pointers_made": 2.5, "field_goals_made": 9.5, "minutes": 30.5,
    }
    for i, col in enumerate(cols_per_row):
        with col:
            stat = st.selectbox(
                f"Leg {i+1} stat", STAT_CHOICES,
                index=i % len(STAT_CHOICES),
                key=f"parlay_leg_{i}_stat",
            )
            line = st.number_input(
                f"Leg {i+1} line",
                value=default_lines.get(stat, 5.5),
                step=0.5, format="%.1f",
                key=f"parlay_leg_{i}_line",
            )
            legs.append(parlay.LegSpec(stat_type=stat, line=float(line)))

    c1, c2, c3 = st.columns([1, 1, 1])
    parlay_odds = c1.number_input("Parlay American odds", value=300, step=5)
    n_sims = c2.number_input("Model n_sims", value=20000, step=1000,
                             min_value=1000, max_value=200000)
    run_model_too = c3.checkbox("Run external model SGP too", value=False)

    if not st.button("Compare parlay"):
        return

    # SECURITY: validate every leg's line + the parlay-level odds + n_sims
    # before any of them feed Monte Carlo or fitted-prob math.
    try:
        iv.validate_parlay_legs_count(len(legs))
        validated_legs = [
            parlay.LegSpec(stat_type=l.stat_type,
                           line=iv.validate_line(l.stat_type, l.line))
            for l in legs
        ]
        validated_parlay_odds = iv.validate_american_odds(parlay_odds)
        validated_n_sims = iv.validate_n_sims(n_sims)
        validated_ngames = iv.validate_n_games(n_games)
    except iv.ValidationError as exc:
        st.error(f"Invalid parlay input: {exc}")
        return
    legs = validated_legs

    # Independence approximation from chart data (always run)
    chart_view = parlay.chart_independence_parlay(
        db_path=db_path,
        player_id=player_id,
        player_name=player_name,
        legs=legs,
        n_games=validated_ngames,
    )
    historical = parlay.historical_parlay_hits(
        db_path=db_path,
        player_id=player_id,
        legs=legs,
        n_games=validated_ngames,
    )

    # Per-leg breakdown table
    per_leg_rows = []
    for cv, hv in zip(chart_view["per_leg"], historical["per_leg"]):
        per_leg_rows.append({
            "leg": cv["stat"],
            "line": cv["line"],
            "mu":   cv["mu"],
            "sigma": cv["sigma"],
            "P(over) chart-data": cv["p_over"],
            "hit% historical":   hv["hit_rate"],
            "n": hv["n"],
        })
    st.markdown("**Per-leg view**")
    st.dataframe(
        pd.DataFrame(per_leg_rows),
        use_container_width=True, hide_index=True,
        column_config={
            "P(over) chart-data": st.column_config.ProgressColumn(
                "P(over) chart-data", min_value=0.0, max_value=1.0,
                format="%.1f%%"),
            "hit% historical": st.column_config.ProgressColumn(
                "hit% historical", min_value=0.0, max_value=1.0,
                format="%.0f%%"),
        },
    )

    parlay_rows = []
    parlay_rows.append({
        "method": "Chart data (independence product)",
        "P(parlay)": chart_view["product_p"],
        "EV / unit": _ev_quick(chart_view["product_p"], (validated_parlay_odds or 0)),
        "n_used": int(n_games),
    })
    parlay_rows.append({
        "method": (
            f"Historical all-hit ({historical['all_hit']}/{historical['n']})"
        ),
        "P(parlay)": historical["rate"],
        "EV / unit": _ev_quick(historical["rate"], (validated_parlay_odds or 0)),
        "n_used": historical["n"],
    })

    if run_model_too:
        try:
            from nba_model.run_model import run_parlay_demo
            with st.spinner(
                f"Calling run_parlay_demo for {player_name} (NBA API)..."
            ):
                model_res = run_parlay_demo(
                    player_name=player_name,
                    stats_cols=[leg.stat_type for leg in legs],
                    lines=[leg.line for leg in legs],
                    american_odds=(validated_parlay_odds or 0),
                    sportsbook="custom",
                    n_games=int(validated_ngames),
                    n_sims=int(validated_n_sims),
                )
            parlay_rows.insert(0, {
                "method": "Model SGP (correlated Gaussian)",
                "P(parlay)": model_res["probability"],
                "EV / unit": model_res["ev"],
                "n_used": int(n_games),
            })
        except Exception as exc:  # noqa: BLE001
            st.warning(
                f"Model run_parlay_demo failed: {exc}. Showing chart-data only."
            )

    st.markdown("**Parlay-level cross-comparison**")
    st.dataframe(
        pd.DataFrame(parlay_rows),
        use_container_width=True, hide_index=True,
        column_config={
            "P(parlay)": st.column_config.ProgressColumn(
                "P(parlay)", min_value=0.0, max_value=1.0,
                format="%.1f%%"),
            "EV / unit": st.column_config.NumberColumn(
                "EV / unit", format="%+.3f"),
        },
    )

    # Verdict
    probs = [(r["method"], r["P(parlay)"]) for r in parlay_rows
             if r["P(parlay)"] is not None]
    if probs:
        st.markdown("**Verdict**")
        for name, p in probs:
            ev = _ev_quick(p, (validated_parlay_odds or 0))
            st.write(
                f"- `{name}` P = **{p:.2%}**, EV @ {parlay_odds:+d} "
                f"= **{ev:+.3f}**" + (" :rocket: +EV" if ev > 0 else "")
            )


def _ev_quick(prob, american_odds) -> Optional[float]:
    """Tiny EV helper for the parlay view."""
    if prob is None or not american_odds:
        return None
    o = int(american_odds)
    if o == 0:
        return None
    dec = 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o))
    return float(prob * (dec - 1.0) - (1.0 - prob))


@st.cache_data(show_spinner=False)
def _cached_scanner_books(db_path: str) -> list[str]:
    """Distinct books currently in web_prop_cards, unioned with the expected set."""
    from nba_model.data.database.db_manager import DatabaseManager
    found: list[str] = []
    try:
        with DatabaseManager(db_path=db_path) as db:
            rows = db.conn.execute(
                "SELECT DISTINCT book FROM web_prop_cards "
                "WHERE player_classification = 'active_nba'"
            ).fetchall()
        found = [str(r[0]) for r in rows if r and r[0]]
    except Exception:
        found = []
    # Union with the expected DFS/sportsbook set (title-cased for display),
    # preserving any exact scraped casing first.
    seen = {b.lower() for b in found}
    for b in pc.EXPECTED_BOOKS:
        if b.lower() not in seen:
            found.append(b.title())
            seen.add(b.lower())
    return sorted(found, key=str.lower)


def _admin_dashboard_view() -> None:
    """Admin-only: subscriber counts, MRR estimate, recent churn."""
    from nba_model.web import subscriptions
    st.subheader(":bar_chart: Admin dashboard")
    st.caption(
        "Subscription health from the configured backend "
        f"(`{subscriptions.selected_backend()}`). MRR uses "
        "`STRIPE_PRICE_MONTHLY_USD` (env) × active premium."
    )
    try:
        stats = subscriptions.aggregate_stats()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load subscription stats: {exc}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Premium", stats["premium"])
    c2.metric("Free / total", f"{stats['free']} / {stats['total']}")
    c3.metric(
        "MRR estimate",
        f"${stats['mrr_estimate']:,.0f}" if stats["mrr_estimate"] is not None
        else "— (set price)",
    )
    c4.metric("Recent churn", stats["churned"],
              help="Subscriptions whose last Stripe event was a cancellation.")
    if stats["price_monthly"] is None:
        st.info(
            "Set `STRIPE_PRICE_MONTHLY_USD` to estimate MRR "
            "(e.g. `export STRIPE_PRICE_MONTHLY_USD=9.99`)."
        )


def _edge_scanner_view(db_path: str, user) -> None:
    """Book Edge Scanner: rank slate-wide props by model-vs-line edge.

    Picks one or more books, fits each player's last-N normal (μ/σ) and ranks
    every current line by how far the model's P(best side) beats the implied
    breakeven (DFS lines priced at -110 by default)."""
    st.subheader(":mag: Book Edge Scanner")
    st.caption(
        "Slate-wide model-vs-line edge: each book line vs the player's "
        "fitted-normal projection (μ = last-N mean). A line **below μ** is a "
        "soft over (P(over) > 50%). DFS books without a posted price are "
        "scored against the -110 breakeven (52.4%)."
    )

    all_books = _cached_scanner_books(db_path)
    if not all_books:
        st.warning(
            "No scraped prop cards in the DB yet. Run the web-text ETL "
            "(`nba_model.data.daily_etl` / hourly update) on the Chrome host "
            "to populate `web_prop_cards`."
        )
        return

    is_premium = bool(getattr(user, "is_premium", False))
    stat_options = (STAT_CHOICES if is_premium
                    else tuple(web_auth.PREVIEW_STATS))
    n_games_max = 200 if is_premium else web_auth.PREVIEW_MAX_GAMES

    with st.sidebar:
        st.markdown("### Edge scanner")
        books = st.multiselect(
            "Books", all_books,
            default=all_books[: min(3, len(all_books))],
            help="Only props from these books are scanned.",
            key="es_books",
        )
        stats = st.multiselect(
            "Stat types", stat_options, default=list(stat_options),
            key="es_stats",
        )
        n_games = st.slider(
            "History window (games for μ/σ)", min_value=2,
            max_value=int(n_games_max),
            value=min(25, int(n_games_max)), key="es_ngames",
        )
        model_mode_label = st.radio(
            "Model", ["Last-N mean (charts)", "Rolling window (prop board)"],
            index=0, key="es_model_mode",
        )
        model_mode = ("chart_mean" if model_mode_label.startswith("Last-N")
                      else "rolling")
        rolling_window = 10
        if model_mode == "rolling":
            rolling_window = st.slider(
                "Rolling window", min_value=3, max_value=30, value=10,
                key="es_rolling",
            )
        min_p_over = st.slider(
            "Min P(over) to show", 0.0, 1.0, 0.0, step=0.05, key="es_minp",
        )
        min_edge = st.slider(
            "Min model edge", 0.0, 0.5, 0.0, step=0.01, key="es_minedge",
        )
        only_positive_ev = st.checkbox(
            "Only +EV rows", value=False, key="es_posev",
        )
        if st.button("Refresh scan", key="es_refresh"):
            _cached_scanner_books.clear()
            st.rerun()

    if not books:
        st.info("Select at least one book in the sidebar to scan.")
        return

    # Free-tier app-layer throttle: cap full-slate scans per session so a tight
    # rerun loop can't hammer the DB. Premium is unthrottled.
    if not is_premium:
        from nba_model.web import throttle
        if not throttle.session_rate_limit("edge_scan", max_calls=8,
                                           window_seconds=60.0):
            st.warning(
                "Scan rate limit reached for the free tier (8 / minute). "
                "Wait a moment or upgrade for unthrottled scans."
            )
            return

    try:
        lines = es.fetch_latest_prop_lines(
            db_path, books=books, stat_types=stats or None,
        )
        scored = es.score_prop_edges(
            lines, db_path=db_path, n_games=int(n_games),
            model_mode=model_mode, rolling_window=int(rolling_window),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Edge scan failed: {exc}")
        return

    # Free tier: restrict to the preview player set (same gate as other views).
    if not is_premium and not scored.empty:
        preview = {p.casefold() for p in web_auth.PREVIEW_PLAYERS}
        scored = scored[
            scored["player_name"].str.casefold().isin(preview)
        ].reset_index(drop=True)

    ranked = es.top_edges(
        scored,
        min_p_over=(min_p_over or None),
        min_edge=(min_edge or None),
        only_positive_ev=only_positive_ev,
        limit=200,
    )

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Props scanned", int(len(scored)))
    k2.metric("Clearing filters", int(len(ranked)))
    pos_ev = int((scored["ev_best"].fillna(-1) > 0).sum()) if not scored.empty else 0
    k3.metric("+EV props", pos_ev)
    if not scored.empty and scored["observed_hours_ago"].notna().any():
        freshest = float(scored["observed_hours_ago"].min())
        k4.metric("Freshest line", pc._format_staleness(freshest))
    else:
        k4.metric("Freshest line", "—")

    if not is_premium:
        st.info(
            "Free preview: scanner limited to "
            f"{', '.join(web_auth.PREVIEW_PLAYERS)} "
            f"({', '.join(web_auth.PREVIEW_STATS)}, last "
            f"{web_auth.PREVIEW_MAX_GAMES} games). Upgrade for the full slate."
        )

    if ranked.empty:
        st.warning("No props cleared the current filters.")
        return

    # ---- Main table ----
    display = ranked.drop(columns=["observed_at_utc"], errors="ignore")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "p_over": st.column_config.ProgressColumn(
                "P(over)", min_value=0.0, max_value=1.0, format="%.2f",
            ),
            "model_edge": st.column_config.ProgressColumn(
                "Model edge", min_value=0.0, max_value=0.5, format="%.3f",
            ),
            "book_line": st.column_config.NumberColumn("Line", format="%.1f"),
            "model_mu": st.column_config.NumberColumn("μ (model)", format="%.1f"),
            "line_vs_mu": st.column_config.NumberColumn("Line − μ", format="%.1f"),
            "ev_best": st.column_config.NumberColumn("EV (best)", format="%.3f"),
            "consensus_mean": st.column_config.NumberColumn(
                "Consensus", format="%.1f"),
            "observed_hours_ago": st.column_config.NumberColumn(
                "Age (h)", format="%.1f"),
        },
    )
    st.caption(
        "Gold opportunities: P(over) > 0.55 **and** line < μ (soft overs). "
        "Sorted by model edge."
    )

    _csv_download_button(
        ranked, label="Download edge scan as CSV",
        filename="book_edge_scan.csv", key="csv_edge_scan",
    )

    # ---- Jump into Player Charts for a row ----
    players_in_scan = list(dict.fromkeys(ranked["player_name"].tolist()))
    jump = st.selectbox(
        "Open a player in Player Charts", ["(pick a player)"] + players_in_scan,
        index=0, key="es_jump",
    )
    if jump and jump != "(pick a player)":
        if st.button(f"Go to {jump}", key="es_jump_go"):
            row = ranked[ranked["player_name"] == jump].iloc[0]
            _qp_set(player=jump, stat=str(row["stat_type"]),
                    view="Player charts")
            st.rerun()


def _game_results_view(db_path: str) -> None:
    """List recent NBA games — final scores + winner for each matchup."""
    st.subheader("NBA Game Results")
    st.caption(
        "Sourced from nba_api leaguegamefinder. One row per game; "
        "use the filters to slice by season, season type, or team."
    )

    seasons = pc.list_seasons(db_path)
    if not seasons:
        st.warning(
            "No games in DB yet. Run "
            "`python -m nba_model.data.nba_results_ingestion --skip-player-logs` "
            "to populate."
        )
        return

    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.2, 1.0])
    with c1:
        season = st.selectbox("Season", ["(all)"] + seasons, index=1)
    with c2:
        season_type = st.selectbox(
            "Season type",
            ["(all)", "Regular Season", "Playoffs", "Play-In", "Pre Season"],
            index=0,
        )
    with c3:
        teams = ["(all)"] + pc.list_team_codes(db_path)
        team_pick = st.selectbox("Team", teams, index=0)
    with c4:
        n = int(st.number_input("Limit", min_value=10, max_value=2000,
                                value=100, step=10))

    # Validate inputs before hitting the DB so a malformed value shows a
    # friendly error instead of returning misleading rows.
    try:
        n_validated = iv.validate_n_games(n, min_value=10)
        season_arg = (None if season == "(all)"
                      else iv.validate_season(season))
        if season_type != "(all)" and season_type not in {
            "Regular Season", "Playoffs", "Play-In", "Pre Season",
        }:
            raise iv.ValidationError(f"unknown season type {season_type!r}")
        team_arg = (None if team_pick == "(all)"
                    else iv.validate_team_code(team_pick))
    except iv.ValidationError as exc:
        st.error(f"Invalid filter: {exc}")
        return

    df = pc.fetch_recent_games(
        db_path=db_path,
        n=n_validated,
        season=season_arg,
        season_type=None if season_type == "(all)" else season_type,
        team_abbrev=team_arg,
    )
    if df.empty:
        st.info("No games match those filters.")
        return

    # Compact display: pretty matchup string, score-string, winner emoji.
    show = df.copy()
    show["matchup"] = show.apply(
        lambda r: f"{r['away_abbrev']} @ {r['home_abbrev']}", axis=1,
    )
    show["score"] = show.apply(
        lambda r: (
            f"{int(r['away_pts'])}–{int(r['home_pts'])}"
            if pd.notna(r.get("away_pts")) and pd.notna(r.get("home_pts"))
            else "—"
        ),
        axis=1,
    )
    show = show[["game_date", "season", "season_type", "matchup", "score",
                 "winner", "away_pts", "home_pts", "game_id"]]
    show = show.rename(columns={"season_type": "type"})

    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(show)} game(s).")


def _player_browse_view(db_path: str) -> None:
    """Searchable / sortable table of recent league-wide player game logs."""
    st.subheader("Player Stats Browse")
    st.caption(
        "Recent player game logs across the league. Use filters to find "
        "players who hit certain marks (e.g. last games with 30+ points)."
    )

    seasons = pc.list_seasons(db_path)
    season_default = seasons[0] if seasons else None

    c1, c2, c3, c4, c5 = st.columns([1.0, 1.2, 1.2, 1.0, 1.0])
    with c1:
        season = st.selectbox(
            "Season", ["(any)"] + seasons,
            index=(1 if season_default else 0),
        )
    with c2:
        teams = ["(any)"] + pc.list_team_codes(db_path)
        team_pick = st.selectbox("Team", teams, index=0)
    with c3:
        stat = st.selectbox(
            "Stat filter", ["(none)", "points", "rebounds", "assists",
                            "steals", "blocks", "fg3m", "minutes"],
            index=0,
            help="Pair with 'Min value' to find games above a threshold.",
        )
    with c4:
        min_value_text = st.text_input(
            "Min value", value="",
            help="e.g. 30 to find games with the chosen stat ≥ 30.",
        )
    with c5:
        n = int(st.number_input("Limit", min_value=10, max_value=5000,
                                value=200, step=50))

    try:
        n_validated = iv.validate_n_games(n, min_value=10)
        season_arg = (None if season == "(any)"
                      else iv.validate_season(season))
        team_arg = (None if team_pick == "(any)"
                    else iv.validate_team_code(team_pick))
        if stat != "(none)":
            iv.validate_stat_type(stat)  # raises if unknown
        min_value = (float(min_value_text) if min_value_text.strip() else None)
    except (iv.ValidationError, ValueError) as exc:
        st.error(f"Invalid filter: {exc}")
        return

    df = pc.fetch_player_recent_results(
        db_path=db_path,
        n=n_validated,
        team_abbrev=team_arg,
        season=season_arg,
        stat=None if stat == "(none)" else stat,
        min_value=min_value if stat != "(none)" else None,
    )
    if df.empty:
        st.info("No player games match those filters.")
        return

    show = df[[
        "game_date", "player_name", "matchup", "result",
        "minutes", "points", "rebounds", "assists",
        "steals", "blocks", "turnovers", "fg3m", "plus_minus", "season",
    ]].rename(columns={"fg3m": "3pm"})
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(show)} game(s) across {show['player_name'].nunique()} player(s).")


def _render_sport_selector():
    """Sport picker in the top control bar.

    Only NBA is live today; NFL / MLB / NHL / Soccer render as "coming soon"
    options that pin a sticky warning at the top of the main pane if picked.
    Returns the selected `Sport` object.
    """
    from sports import live_sports as _live, stub_sports as _stubs
    options = list(_ALL_SPORTS)
    labels = {
        s.key: (s.display_name if s.is_live
                else f"{s.display_name} (coming soon)")
        for s in options
    }
    default_key = _qp_get(_QP_SPORT, "nba")
    with st.popover(":basketball: Sport", use_container_width=False):
        chosen_key = st.selectbox(
            "Sport",
            [s.key for s in options],
            index=_index_or_zero([s.key for s in options], default_key),
            format_func=lambda k: labels[k],
            help="NBA is live. Other sports show what we plan to ship + "
                 "their open questions; pick one to see the roadmap.",
            label_visibility="collapsed",
            key="top_sport_picker",
        )
    _qp_set(sport=chosen_key)
    sport = _get_sport(chosen_key)
    if not sport.is_live:
        with st.container():
            st.warning(
                f":construction: **{sport.display_name}** isn't live yet. "
                "The data pipeline + scrapers are planned but not built. "
                "See `docs/MULTI_SPORT_PLAN.md` for the rollout order."
            )
            with st.expander(
                f"What's planned for {sport.display_name}", expanded=False,
            ):
                st.markdown(
                    f"**Status:** {sport.status}  \n"
                    f"**Season format:** {sport.season_format or 'TBD'}  \n"
                    f"**Stat types ({len(sport.stat_types)}):** "
                    + ", ".join(sport.stat_types[:8])
                    + (" ..." if len(sport.stat_types) > 8 else "")
                )
                if sport.sub_leagues:
                    st.markdown("**Sub-leagues / competitions:**")
                    for sub in sport.sub_leagues:
                        st.markdown(
                            f"- `{sub.key}` — {sub.display_name}"
                            f" ({sub.season_format})"
                        )
                if sport.notes:
                    st.markdown("**Open questions / decisions:**")
                    for n in sport.notes:
                        st.markdown(f"- {n}")
    return sport


def _render_watchlist_popover() -> None:
    """Top-bar watchlist popover. Clicking an entry writes ``?player=`` to the
    URL and reruns so the selection-row picks up the new player.
    """
    items = wl.get()
    with st.popover(
        f":bookmark_tabs: Watchlist ({len(items)})",
        use_container_width=False,
    ):
        if items:
            for name in items:
                col_pick, col_rm = st.columns([4, 1])
                if col_pick.button(name, key=f"wl_pick_{name}",
                                   use_container_width=True):
                    _qp_set(player=name)
                    st.rerun()
                if col_rm.button(":x:", key=f"wl_rm_{name}",
                                 help=f"Remove {name}"):
                    wl.remove(name)
                    st.rerun()
        else:
            st.caption("Empty. Pin a player from the chart view below.")
        if st.button("Clear watchlist", key="wl_clear",
                     use_container_width=True, disabled=not items):
            wl.clear()
            st.rerun()


def _render_etl_status_widget() -> None:
    """Top-bar popover: data freshness signal from the latest ETL report.

    Always renders, but degrades gracefully when no report exists (the
    Streamlit Cloud deploy might not have run the ETL yet). Trust signal:
    a user can see we're pushing data updates and that they're recent.
    """
    summary = etl_status.summarize_report(etl_status.load_latest_report())
    with st.popover(
        f":satellite_antenna: Data {summary['age_text']}",
        use_container_width=False,
    ):
        if True:
            if not summary["found"]:
                st.caption(
                    "No ETL report found at `nba_model/data/artifacts/`. "
                    "The site is reading whatever's in the bundled SQLite DB."
                )
                return
            status = summary["status"]
            badge = {
                "success": ":white_check_mark: success",
                "partial_success": ":warning: partial success",
                "failed": ":x: failed",
            }.get(status, f":grey_question: {status}")
            st.markdown(
                f"**Last run:** {badge}  \n"
                f"**Finished:** {summary['age_text']}  \n"
                + (
                    f"**Elapsed:** {summary['elapsed_ms']/1000:.1f}s  \n"
                    if isinstance(summary["elapsed_ms"], (int, float))
                    else ""
                )
            )
            if summary["steps"]:
                lines = []
                for name, st_ in summary["steps"]:
                    lines.append(f"- `{name}` {etl_status.step_badge(st_)}")
                st.markdown("**Steps:**\n" + "\n".join(lines))


def _compare_players_view(
    db_path: str,
    players_df: pd.DataFrame,
    default_player: str,
    stat_type: str,
    n_games: int,
) -> None:
    """Overlay 2-3 players' distributions for the same stat.

    Pure local-DB view (no external API calls), so this is fast enough to
    re-render on every dropdown change.
    """
    st.subheader(f"Compare players — {stat_type} (last {n_games} games each)")
    st.caption(
        "Pick 2 or 3 players to overlay. Each player's histogram is "
        "density-normalized so the heights are comparable even if sample "
        "sizes differ. Dashed verticals are each player's book-consensus "
        "mean across all stored sportsbooks."
    )
    all_names = players_df["player_name"].dropna().astype(str).tolist()
    if not all_names:
        st.warning("No players in the local DB to compare.")
        return

    default_pre = [default_player] if default_player in all_names else all_names[:1]
    # Pre-fill the second slot with another famous player if available.
    for candidate in ("LeBron James", "Nikola Jokic", "Stephen Curry"):
        if candidate in all_names and candidate not in default_pre:
            default_pre.append(candidate)
            break

    picks = st.multiselect(
        "Players (2-3)",
        all_names,
        default=default_pre[:2],
        max_selections=3,
        help="2 or 3 players for a clean overlay; 4+ becomes hard to read.",
    )
    if len(picks) < 2:
        st.info("Pick at least 2 players to compare.")
        return

    # Pull data for each pick.
    datasets: list[pc.PlayerChartData] = []
    summary_rows: list[dict] = []
    for name in picks:
        pid, resolved = _resolve_player_id(players_df, name)
        try:
            d = pc.fetch_player_chart_data(
                db_path=db_path,
                player_id=pid,
                player_name=resolved,
                stat_type=stat_type,
                n_games=int(n_games),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{name}: failed to load - {exc}")
            continue
        datasets.append(d)
        consensus = pc.compute_market_consensus(d)
        summary_rows.append({
            "player": resolved,
            "n": int(d.values.size),
            "mu": round(float(d.mu), 3) if d.values.size else None,
            "sigma": round(float(d.sigma), 3) if d.values.size else None,
            "book_mean": (
                round(consensus["mean"], 3) if consensus["mean"] is not None else None
            ),
            "book_sigma": (
                round(consensus["stdev"], 3) if consensus["stdev"] is not None else None
            ),
            "n_books": consensus["n_books"],
        })

    if not datasets:
        return

    st.plotly_chart(
        plc.build_multi_player_distribution_figure(datasets),
        use_container_width=True,
        key="plotly_multi_player_overlay",
    )

    st.markdown("**Player-by-player summary**")
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    _csv_download_button(
        summary_df,
        label="Download summary as CSV",
        filename=f"compare_players_{stat_type}_n{n_games}.csv",
        key="csv_compare_summary",
    )


# ---------------------------------------------------------------------------
# Single Prop (model) view — full tuning surface, mirrors the desktop
# `Single Prop` tab. Calls `nba_model.run_model.run_single_prop`.
# ---------------------------------------------------------------------------
def _single_prop_model_view(*, default_player: str, n_games_default: int) -> None:
    st.subheader("Single prop (model)")
    st.caption(
        "Calls `nba_model.run_model.run_single_prop` — the same model the "
        "desktop Single Prop tab uses. Hits the NBA API for game logs, "
        "applies the minutes + defense + blowout adjustments, then returns "
        "P(over)+EV under the chosen distribution family. "
        "Note: the model currently uses **points** as the underlying stat; "
        "the `Stat type` field only changes the default line value."
    )

    from nba_model.run_model import (
        DEFAULT_AMERICAN_ODDS,
        DEFAULT_BLOWOUT_PENALTY,
        DEFAULT_BLOWOUT_THRESHOLD,
        DEFAULT_DEFENSE_SENSITIVITY,
        DEFAULT_LEAGUE_AVG_DEF_RATING,
        DEFAULT_OPP_DEF_RATING,
        DEFAULT_POINTS_LINE,
        DEFAULT_ROLLING_WINDOW,
        DEFAULT_SINGLE_PROP_DISTRIBUTION,
        DEFAULT_VEGAS_SPREAD,
        run_single_prop,
    )

    _single_default_lines = {
        "points": 25.5, "assists": 5.5, "rebounds": 8.5, "pra": 35.5,
    }

    with st.form("single_prop_model_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        player = c1.text_input("Player", value=default_player or "LeBron James")
        stat_type = c2.selectbox(
            "Stat type", list(_single_default_lines.keys()), index=0,
            help="Cosmetic: only changes the default line. The model itself is points-based.",
        )
        line = c3.number_input(
            "Line", value=_single_default_lines.get(stat_type, DEFAULT_POINTS_LINE),
            step=0.5, format="%.1f",
        )
        odds = c4.number_input("American odds", value=DEFAULT_AMERICAN_ODDS, step=5)

        c5, c6, c7, c8 = st.columns(4)
        rolling_window = c5.number_input(
            "Rolling window", min_value=2, max_value=30,
            value=DEFAULT_ROLLING_WINDOW, step=1,
        )
        history_games = c6.number_input(
            "History games", min_value=10, max_value=200,
            value=max(10, min(int(n_games_default), 200)), step=5,
        )
        opp_def = c7.number_input("Opp def rating",
                                   value=DEFAULT_OPP_DEF_RATING, step=0.5)
        spread = c8.number_input("Vegas spread",
                                  value=DEFAULT_VEGAS_SPREAD, step=0.5)

        st.markdown("**Distribution + tuning**")
        c9, c10 = st.columns([1.2, 1.0])
        # SECURITY/SAFETY: source dist list from SUPPORTED_DISTRIBUTIONS so
        # the selector tracks whatever the model team ships.
        dist_options = list(SUPPORTED_DISTRIBUTIONS)
        default_idx = (
            dist_options.index(DEFAULT_SINGLE_PROP_DISTRIBUTION)
            if DEFAULT_SINGLE_PROP_DISTRIBUTION in dist_options else 0
        )
        distribution = c9.selectbox(
            "Distribution family", dist_options, index=default_idx,
        )
        show_plot = c10.checkbox("Render distribution plot", value=True)

        c11, c12 = st.columns(2)
        league_avg_def = c11.slider(
            "League avg def rating",
            min_value=105.0, max_value=120.0,
            value=float(DEFAULT_LEAGUE_AVG_DEF_RATING), step=0.1,
        )
        defense_sensitivity = c12.slider(
            "Defense sensitivity",
            min_value=0.0, max_value=1.5,
            value=float(DEFAULT_DEFENSE_SENSITIVITY), step=0.01,
        )
        c13, c14 = st.columns(2)
        blowout_threshold = c13.slider(
            "Blowout threshold",
            min_value=0.0, max_value=25.0,
            value=float(DEFAULT_BLOWOUT_THRESHOLD), step=0.5,
        )
        blowout_penalty = c14.slider(
            "Blowout penalty",
            min_value=0.0, max_value=0.5,
            value=float(DEFAULT_BLOWOUT_PENALTY), step=0.01,
        )
        c15, c16, c17 = st.columns(3)
        defense_severity = c15.slider(
            "Defense severity", min_value=0.0, max_value=3.0,
            value=1.0, step=0.05,
        )
        minutes_severity = c16.slider(
            "Minutes-penalty severity", min_value=0.0, max_value=3.0,
            value=1.0, step=0.05,
        )
        sigma_severity = c17.slider(
            "Sigma severity", min_value=0.0, max_value=3.0,
            value=1.0, step=0.05,
        )

        submitted = st.form_submit_button("Run model")

    if not submitted:
        st.caption(
            "Set parameters above and click **Run model**. The first run for a "
            "new player triggers NBA-API calls and can take a few seconds."
        )
        return

    # SECURITY: validate every numeric input before it reaches the model.
    try:
        v_line = iv.validate_line(stat_type, line)
        v_odds = iv.validate_american_odds(odds)
        v_n_games = iv.validate_n_games(history_games, min_value=10)
        v_rolling = iv.validate_rolling_window(rolling_window, default=DEFAULT_ROLLING_WINDOW)
        if not (-30.0 <= float(spread) <= 30.0):
            raise iv.ValidationError(f"vegas_spread {spread} out of [-30, 30]")
        if not (80.0 <= float(opp_def) <= 130.0):
            raise iv.ValidationError(f"opp_def_rating {opp_def} out of [80, 130]")
    except iv.ValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return

    with st.spinner(f"Running model for {player} (NBA API)…"):
        try:
            result = run_single_prop(
                player_name=str(player).strip(),
                line=float(v_line),
                rolling_window=int(v_rolling),
                american_odds=int(v_odds) if v_odds is not None else DEFAULT_AMERICAN_ODDS,
                opp_def_rating=float(opp_def),
                vegas_spread=float(spread),
                league_avg_def_rating=float(league_avg_def),
                defense_sensitivity=float(defense_sensitivity),
                blowout_threshold=float(blowout_threshold),
                blowout_penalty=float(blowout_penalty),
                n_games=int(v_n_games),
                show_plot=False,
                distribution=str(distribution).strip().lower(),
                defense_severity=float(defense_severity),
                minutes_penalty_severity=float(minutes_severity),
                sigma_severity=float(sigma_severity),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model failed: {exc}")
            return

    # KPI row
    kpi = st.columns(5)
    kpi[0].metric("mu (expected)", f"{result['mu']:.2f}")
    kpi[1].metric("sigma (adjusted)", f"{result['sigma_adjusted']:.2f}")
    kpi[2].metric("P(over)", f"{result['prob_over']:.1%}")
    kpi[3].metric("Book implied", f"{result['implied_prob']:.1%}")
    ev_delta = (
        f"{(result['prob_over'] - result['implied_prob']) * 100:+.1f} pts"
    )
    kpi[4].metric("EV / unit", f"{result['ev']:+.3f}", delta=ev_delta)
    st.caption(f"Distribution: **{result['distribution']}**")

    if show_plot:
        # Build a quick density figure using the model's mu + adjusted sigma.
        import plotly.graph_objects as go
        import numpy as np
        from scipy.stats import norm
        mu = float(result["mu"])
        sigma = max(0.01, float(result["sigma_adjusted"]))
        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 240)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=norm.pdf(xs, mu, sigma), mode="lines",
            name=f"{result['distribution']} fit",
            line=dict(color="#222", width=2.5),
        ))
        fig.add_vline(
            x=float(v_line), line=dict(color="#cc4125", width=2.4, dash="dash"),
            annotation_text=f"line {v_line:.1f}", annotation_position="top right",
        )
        fig.add_vline(
            x=mu, line=dict(color="#4a86e8", width=1.6, dash="dot"),
            annotation_text=f"mu {mu:.2f}", annotation_position="top left",
        )
        fig.update_layout(
            title=f"{player} — model density ({result['distribution']})",
            xaxis_title="value", yaxis_title="density",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True,
                        key="plotly_single_model_density")

    with st.expander("Full result JSON", expanded=False):
        st.json(result)


# ---------------------------------------------------------------------------
# Manual lines import view — paste board / CSV / pipe rows, parse, save.
# Saves are admin-gated to prevent random visitors from poisoning betting_lines.
# ---------------------------------------------------------------------------
def _manual_lines_import_view(*, db_path: str) -> None:
    from datetime import datetime, timezone
    from nba_model.model.manual_lines import parse_manual_lines_text

    st.subheader("Manual lines import")
    st.caption(
        "Paste rows (pipe / CSV / tab delimited) or a noisy sportsbook board "
        "dump. The parser extracts player / stat / line / odds and previews "
        "the records before any database write."
    )

    is_admin = web_auth.is_admin()
    if not is_admin and web_auth.BILLING_ENABLED:
        st.warning(
            ":lock: Saving to the database is **admin-only** to protect the "
            "shared `betting_lines` table from poisoning. You can still paste "
            "rows and preview the parser output."
        )

    with st.form("manual_lines_form", clear_on_submit=False):
        c1, c2 = st.columns([1, 2])
        default_date = c1.text_input(
            "Default game date (YYYY-MM-DD)",
            value=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        default_book = c2.text_input(
            "Default sportsbook",
            value="Manual import",
        )
        text = st.text_area(
            "Lines / board text",
            height=240,
            value=(
                "# Example rows\n"
                "LeBron James | points | 27.5 | -115 | -105\n"
                "Stephen Curry | 2025-03-07 | FanDuel | assists | 5.5 | -110 | -120\n"
            ),
            help=(
                "Formats:\n"
                "  1) player | stat | line | over_odds | under_odds\n"
                "  2) player | game_date | book | stat | line | over_odds | under_odds\n"
                "  3) Raw board dump (auto-extract from blocks: player + matchup + line + stat)\n"
                "Stats accept aliases (pts/ast/reb/pra)."
            ),
        )
        c_parse, c_save = st.columns(2)
        parse_btn = c_parse.form_submit_button("Parse", use_container_width=True)
        save_btn = c_save.form_submit_button(
            "Save to DB",
            disabled=(web_auth.BILLING_ENABLED and not is_admin),
            use_container_width=True,
        )

    if not (parse_btn or save_btn):
        return

    try:
        records, errors = parse_manual_lines_text(
            text=text,
            default_game_date=default_date,
            default_book=default_book,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Parse failed: {exc}")
        return

    # SECURITY: reject any record that fails the betting-line plausibility
    # check before showing the user a "ready to save" view.
    plausible_records: list[dict] = []
    dropped: list[str] = []
    for rec in records:
        if iv.is_plausible_betting_line(
            rec.get("stat_type", ""),
            rec.get("line_value"),
            rec.get("over_odds"),
            rec.get("under_odds"),
        ):
            plausible_records.append(rec)
        else:
            dropped.append(
                f"{rec.get('player_name')} {rec.get('stat_type')} "
                f"{rec.get('line_value')} (failed validation)"
            )

    cols = st.columns(3)
    cols[0].metric("Parsed rows", len(records))
    cols[1].metric("Plausible rows", len(plausible_records))
    cols[2].metric("Errors / dropped", len(errors) + len(dropped))

    if records:
        st.markdown("**Parsed records (after validation)**")
        st.dataframe(
            pd.DataFrame(plausible_records),
            hide_index=True,
            use_container_width=True,
        )

    if errors:
        with st.expander(f":warning: Parse errors ({len(errors)})", expanded=False):
            for e in errors:
                st.text(e)
    if dropped:
        with st.expander(
            f":warning: Dropped by validator ({len(dropped)})", expanded=False
        ):
            for d in dropped:
                st.text(d)

    if save_btn:
        if web_auth.BILLING_ENABLED and not is_admin:
            st.error("Saving is admin-only.")
            return
        if not plausible_records:
            st.warning("Nothing to save.")
            return
        try:
            from nba_model.data.database.db_manager import DatabaseManager
            with DatabaseManager(db_path=db_path) as db:
                before = db.conn.execute(
                    "SELECT COUNT(*) FROM betting_lines"
                ).fetchone()[0]
                seen: dict[int, str] = {}
                for row in plausible_records:
                    pid = row["player_id"]
                    if pid not in seen:
                        seen[pid] = row["player_name"]
                        db.insert_player(pid, row["player_name"])
                db.insert_betting_lines_records(plausible_records)
                after = db.conn.execute(
                    "SELECT COUNT(*) FROM betting_lines"
                ).fetchone()[0]
            inserted = after - before
            st.success(
                f"Inserted **{inserted}** new `betting_lines` row(s) "
                f"(total rows now {after})."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Save failed: {exc}")


_GLOBAL_CSS = """
<style>
/* Hide the rarely-used sidebar entirely (collapsed by default; controls
   live in the top bar). The sidebar handle still shows so power-users can
   open it if a stray widget ever lands there. */
section[data-testid="stSidebar"] > div { padding-top: 0.5rem; }

/* Hero title */
h1.hero-title {
  font-weight: 700;
  letter-spacing: -0.02em;
  font-size: 2.0rem;
  margin: 0.2rem 0 0.6rem 0;
  color: #0f172a;
}
h1.hero-title .hero-suffix {
  font-weight: 500;
  font-size: 1.1rem;
  color: #64748b;
  margin-left: 0.6rem;
}

/* Section divider — subtle, used between control bar and content. */
hr.section-divider {
  border: 0; border-top: 1px solid #e5e7eb;
  margin: 0.75rem 0 1.0rem 0;
}

/* KPI cards: tighten Streamlit's default metric look. */
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 10px 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetricLabel"] {
  font-size: 0.78rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
div[data-testid="stMetricValue"] {
  font-weight: 700;
  font-size: 1.5rem;
  color: #0f172a;
}

/* Primary nav (horizontal radio styled as pill segmented control). */
div.primary-nav div[role="radiogroup"] {
  gap: 6px;
  flex-wrap: wrap;
}
div.primary-nav div[role="radiogroup"] > label {
  background: #f1f4f9;
  border: 1px solid transparent;
  border-radius: 999px;
  padding: 6px 14px;
  margin: 0 !important;
  cursor: pointer;
  font-size: 0.92rem;
  color: #334155;
  transition: background 0.12s ease, border-color 0.12s ease;
}
div.primary-nav div[role="radiogroup"] > label:hover {
  background: #e2e8f0;
}
div.primary-nav div[role="radiogroup"] > label:has(input:checked) {
  background: #5b8def;
  color: #ffffff;
  border-color: #4775d9;
}
div.primary-nav div[role="radiogroup"] > label > div:first-child {
  display: none;  /* hide the native radio dot */
}

/* Top status row spacing */
div.top-status { display: flex; gap: 6px; justify-content: flex-end; align-items: center; }

/* Plotly chart cards */
div[data-testid="stPlotlyChart"] {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 6px 8px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}

/* Tab strip polish */
button[data-testid="stTab"] {
  font-weight: 500;
}

/* Captions */
.stCaption, div[data-testid="stCaptionContainer"] { color: #64748b; }
</style>
"""


def _inject_global_css() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def _render_top_status_row(user) -> None:
    """Right-side top-bar cluster: ETL freshness + watchlist + account popovers."""
    st.markdown("<div class='top-status'>", unsafe_allow_html=True)
    cols = st.columns([1.4, 1.2, 1.2, 1.4])
    with cols[0]:
        _render_etl_status_widget()
    with cols[1]:
        _render_watchlist_popover()
    with cols[2]:
        with st.popover(":bookmark: Pin", use_container_width=False):
            cur_player = _qp_get(_QP_PLAYER, "")
            if cur_player:
                if st.button(
                    f"Pin {cur_player} to watchlist",
                    key="wl_add_current_top",
                    disabled=(cur_player in wl.get()),
                    use_container_width=True,
                ):
                    wl.add(cur_player)
                    st.rerun()
            else:
                st.caption("Select a player first.")
    with cols[3]:
        if web_auth.BILLING_ENABLED:
            with st.popover(":bust_in_silhouette: Account",
                            use_container_width=False):
                web_auth.render_user_card(sidebar=False)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="NBA Player Charts",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_global_css()

    user = web_auth.current_user()
    title_suffix = ""
    if web_auth.BILLING_ENABLED:
        title_suffix = " — Premium" if user.is_premium else " — Free preview"

    # ---- Top hero + status pills ----
    hero_l, hero_r = st.columns([5, 4])
    with hero_l:
        st.markdown(
            f"<h1 class='hero-title'>🏀 NBA Player Charts"
            f"<span class='hero-suffix'>{title_suffix}</span></h1>",
            unsafe_allow_html=True,
        )
    with hero_r:
        _render_top_status_row(user)

    if web_auth.BILLING_ENABLED and not user.is_premium:
        st.info(
            ":lock: **Free preview mode.** You can view "
            f"{', '.join(web_auth.PREVIEW_PLAYERS)} (last "
            f"{web_auth.PREVIEW_MAX_GAMES} games, "
            f"{', '.join(web_auth.PREVIEW_STATS)} only). "
            "Sign in and upgrade for all stats, all players + teams, "
            "team distributions, parlay analysis, and unlimited history."
        )

    # ---- Sport picker (top popover; roadmap card lands in main pane) ----
    active_sport = _render_sport_selector()
    if not active_sport.is_live:
        return

    # ---- DB path (hardcoded; admin gets an override popover) ----
    is_admin = web_auth.is_admin()
    db_path = DEFAULT_DB_PATH
    if is_admin:
        with st.popover(":gear: Admin", use_container_width=False):
            db_path = st.text_input(
                "DB path (admin only)", value=DEFAULT_DB_PATH,
                help="Visible because you are listed in [auth].admins.",
                key="admin_db_path",
            )
            if st.button("Reload DB indexes", key="admin_reload"):
                _cached_teams.clear()
                _cached_players.clear()

    try:
        teams = ["(any)"] + _cached_teams(db_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read teams from {db_path}: {exc}")
        return

    # ---- Primary nav (View mode = top-level pills) ----
    free_views = [
        "Player charts", "Team charts", "Compare players",
        "Line edge scanner",
        "Game Results", "Player Stats Browse",
    ]
    premium_views = [
        "Player charts", "Team charts", "Compare players",
        "Line edge scanner",
        "All stats (overview)", "Single prop (model)",
        "Parlay analysis", "Manual lines import",
        "Game Results", "Player Stats Browse",
    ]
    if web_auth.is_admin():
        premium_views = premium_views + ["Operations (admin)", "Admin (dashboard)"]
    view_options = premium_views if user.is_premium else free_views
    view_default = _qp_get(_QP_VIEW, view_options[0])

    st.markdown("<div class='primary-nav'>", unsafe_allow_html=True)
    view_mode = st.radio(
        "View",
        view_options,
        index=_index_or_zero(view_options, view_default),
        horizontal=True,
        label_visibility="collapsed",
        key="primary_nav_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    _qp_set(view=view_mode)

    is_overview      = view_mode.startswith("All stats")
    is_team_view     = view_mode.startswith("Team charts")
    is_player_view   = view_mode.startswith("Player charts")
    is_parlay_view   = view_mode.startswith("Parlay")
    is_compare_view  = view_mode.startswith("Compare")
    is_game_results  = view_mode.startswith("Game Results")
    is_player_browse = view_mode.startswith("Player Stats Browse")
    is_single_model  = view_mode.startswith("Single prop")
    is_manual_lines  = view_mode.startswith("Manual lines")
    is_operations    = view_mode.startswith("Operations")
    is_admin_dash    = view_mode.startswith("Admin")
    is_edge_scanner  = view_mode.startswith("Line edge")

    # ---- Views with their own controls (no shared selection row) ----
    if is_game_results:
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        _game_results_view(db_path)
        return
    if is_player_browse:
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        _player_browse_view(db_path)
        return
    if is_edge_scanner:
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        _edge_scanner_view(db_path, user)
        return
    if is_single_model:
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        _single_prop_model_view(
            default_player=_qp_get(_QP_PLAYER, "LeBron James"),
            n_games_default=_qp_get_int(_QP_NGAMES, 25),
        )
        return
    if is_manual_lines:
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        _manual_lines_import_view(db_path=db_path)
        return
    if is_operations:
        # Hard gate: Operations launches subprocesses on the host. Admin is
        # required REGARDLESS of BILLING_ENABLED — open-access mode must not
        # leak this surface. The pill also only renders for admins, but this
        # re-check defends against ``?view=Operations+(admin)`` deep-links.
        if not web_auth.is_admin():
            st.error(
                ":lock: The Operations console is admin-only. "
                "Sign in with an account listed in `[auth.admins]` of "
                "`secrets.toml` to access it."
            )
            return
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        from nba_model.web import operations_panel as _ops
        _ops.render_operations_panel(on_authorized=web_auth.is_admin)
        return
    if is_admin_dash:
        # Same hard admin gate as Operations (defends ?view= deep-links).
        if not web_auth.is_admin():
            st.error(":lock: The admin dashboard is admin-only.")
            return
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        _admin_dashboard_view()
        return

    # ---- Shared selection bar (Team / Player / Stat / N games + Filters) ----
    team_default = _qp_get(_QP_TEAM, teams[0])
    sel_cols = st.columns([1.3, 2.6, 1.4, 1.0, 1.0])

    with sel_cols[0]:
        team_pick = st.selectbox(
            "Team", teams,
            index=_index_or_zero(teams, team_default),
            key="top_team_select",
        )
    team_val = "" if team_pick == "(any)" else team_pick
    _qp_set(team=team_val or None)

    try:
        players_df = _cached_players(db_path, team_val)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not list players: {exc}")
        return
    if players_df.empty:
        st.warning("No players in DB for that filter.")
        return
    if "n_books" not in players_df.columns:
        players_df = players_df.assign(n_books=0)

    # ---- Filters popover holds secondary controls so the top stays clean ----
    with sel_cols[4]:
        with st.popover(":control_knobs: Filters", use_container_width=True):
            only_scraped = st.checkbox(
                "Only players with current book lines",
                value=True,
                help="Hide players who aren't in any sportsbook's slate "
                     "right now. Uncheck to browse the full active roster.",
                key="filt_only_scraped",
            )
            sort_mode = st.radio(
                "Sort players by",
                ["Book coverage (most books first)", "Alphabetical"],
                index=0,
                key="filt_sort_mode",
            )
            rolling_raw = st.number_input(
                "Rolling window", min_value=1, max_value=40, value=5, step=1,
                key="filt_rolling",
            )
            overlay_default = [d for d in ("normal",) if d in _OVERLAY_DISTRIBUTIONS]
            overlays = st.multiselect(
                "Fitted distribution families",
                list(_OVERLAY_DISTRIBUTIONS),
                default=overlay_default,
                help="Each picked family draws a fitted overlay on the "
                     "distribution chart. Sourced from "
                     "`simulation.SUPPORTED_DISTRIBUTIONS` plus "
                     "`negative_binomial`.",
                key="filt_overlays",
            )
            if not overlays:
                overlays = ["normal"]
            line_view_choice = st.radio(
                "Line-board layout",
                ["Distribution view", "Line ladder"],
                index=0,
                help="Distribution: histogram + fitted overlays + per-book "
                     "vertical markers with best-EV highlight. "
                     "Line ladder: compact one-row-per-book layout — easier "
                     "with 6+ books, drops the histogram context.",
                key="filt_line_view",
            )
            line_view_mode = (
                "ladder" if line_view_choice.lower().startswith("line ladder")
                else "distribution"
            )
    try:
        rolling = iv.validate_rolling_window(rolling_raw, default=5)
    except iv.ValidationError as exc:
        st.error(f"Invalid rolling window: {exc}")
        return

    # Filter / sort players for the dropdown.
    view_df = players_df
    if only_scraped:
        scraped = view_df[view_df["n_books"] > 0]
        if not scraped.empty:
            view_df = scraped
    if sort_mode.startswith("Book"):
        view_df = view_df.sort_values(
            ["n_books", "player_name"], ascending=[False, True],
        )
    else:
        view_df = view_df.sort_values("player_name", ascending=True)

    names = view_df["player_name"].dropna().astype(str).tolist()
    if not names:
        st.warning("No players match the current filter.")
        return

    if not user.is_premium:
        preview_set = set(web_auth.PREVIEW_PLAYERS)
        names = [n for n in names if n in preview_set]
        if not names:
            st.warning(
                "Free preview is limited to: "
                + ", ".join(web_auth.PREVIEW_PLAYERS)
                + ". They may not appear in this team filter; "
                "set Team to (any)."
            )
            return

    smart_default = names[0]
    url_player = _qp_get(_QP_PLAYER, "")
    player_default = url_player if url_player in names else smart_default

    n_books_lookup = dict(
        zip(
            view_df["player_name"].astype(str),
            view_df["n_books"].fillna(0).astype(int),
        )
    )

    def _label(n: str) -> str:
        nb = n_books_lookup.get(n, 0)
        return f"{n}  ({nb} books)" if nb else n

    with sel_cols[1]:
        player_name = st.selectbox(
            "Player", names,
            index=_index_or_zero(names, player_default),
            format_func=_label,
            help=(
                "Switching player reloads every chart on the right. "
                "Free tier: only "
                + ", ".join(web_auth.PREVIEW_PLAYERS) + "."
                if not user.is_premium
                else "Switching player reloads every chart."
            ),
            key="top_player_select",
        )
    _qp_set(player=player_name)

    stat_choices_for_user = (
        web_auth.PREVIEW_STATS if not user.is_premium else STAT_CHOICES
    )
    stat_default = _qp_get(_QP_STAT, stat_choices_for_user[0])
    with sel_cols[2]:
        stat_type_raw = st.selectbox(
            "Stat", stat_choices_for_user,
            index=_index_or_zero(stat_choices_for_user, stat_default),
            disabled=is_overview,
            help=(
                "Free preview: points only. Premium unlocks all stats."
                if not user.is_premium else None
            ),
            key="top_stat_select",
        )
    _qp_set(stat=stat_type_raw)
    try:
        stat_type = iv.validate_stat_type(
            stat_type_raw, allowed=stat_choices_for_user,
        )
    except iv.ValidationError as exc:
        st.error(f"Invalid stat: {exc}")
        return

    n_games_max = web_auth.PREVIEW_MAX_GAMES if not user.is_premium else 200
    n_games_default = min(_qp_get_int(_QP_NGAMES, 25), n_games_max)
    n_games_default = max(3, n_games_default)
    with sel_cols[3]:
        n_games_raw = st.number_input(
            "Last N", min_value=3, max_value=n_games_max,
            value=n_games_default, step=1,
            help=(
                f"Free preview capped at {web_auth.PREVIEW_MAX_GAMES}. "
                "Premium unlocks up to 200."
                if not user.is_premium else None
            ),
            key="top_n_games",
        )
    _qp_set(n_games=int(n_games_raw))
    try:
        n_games = iv.validate_n_games(n_games_raw, min_value=3)
    except iv.ValidationError as exc:
        st.error(f"Invalid N games: {exc}")
        return
    n_games = min(n_games, n_games_max)

    # Team-aggregate code picker only shows for the Team-charts view.
    if is_team_view:
        team_codes = _cached_team_codes(db_path)
        if not team_codes:
            st.warning("No teams found in game_logs. Run the daily ETL first.")
            return
        team_code_raw = st.selectbox(
            "Team aggregate (code)", team_codes, index=0,
            help="Switching team reloads every team-level chart.",
            key="top_team_code",
        )
        try:
            team_code = iv.validate_team_code(team_code_raw)
        except iv.ValidationError as exc:
            st.error(f"Invalid team code: {exc}")
            return
    else:
        team_code = None

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ---- Compare-players (uses shared selection row) ----
    if is_compare_view:
        _compare_players_view(
            db_path=db_path,
            players_df=players_df,
            default_player=player_name,
            stat_type=stat_type,
            n_games=int(n_games),
        )
        return

    # ---- Data fetch ----------------------------------------------------
    if is_team_view:
        # Team charts are now a free-tier surface alongside player charts.
        # Premium still gets the full multi-stat overlay; free tier sees the
        # same view but limited to PREVIEW_STATS.
        team_stats = (
            tuple(web_auth.PREVIEW_STATS) if not user.is_premium
            else TEAM_STAT_CHOICES
        )
        _team_overview(
            db_path=db_path,
            team=team_code or "",
            n_games=int(n_games),
            overlays=tuple(overlays),
            rolling=int(rolling),
            stat_types=team_stats,
        )
        return

    player_id, resolved_name = _resolve_player_id(players_df, player_name)

    # Belt-and-suspenders: even if a free user somehow lands on a non-preview
    # player (e.g. via deep-link), block here.
    if not user.is_premium and not web_auth.gate_player(resolved_name):
        web_auth.paywall(f"Charts for {resolved_name}")
        return

    if is_parlay_view:
        if not user.is_premium:
            web_auth.paywall("Parlay analysis")
            return
        _parlay_view(
            db_path=db_path,
            player_id=player_id,
            player_name=resolved_name,
            n_games=int(n_games),
        )
        return

    if is_overview:
        if not user.is_premium:
            web_auth.paywall("All-stats overview")
            return
        _all_stats_overview(
            db_path=db_path,
            player_id=player_id,
            player_name=resolved_name,
            n_games=int(n_games),
            overlays=tuple(overlays),
            rolling=int(rolling),
        )
        return

    try:
        data = pc.fetch_player_chart_data(
            db_path=db_path,
            player_id=player_id,
            player_name=resolved_name,
            stat_type=stat_type,
            n_games=int(n_games),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load player data: {exc}")
        return

    st.subheader(f"{resolved_name} — {stat_type}")
    _kpi_row(data)
    _books_panel(data)
    if data.values.size == 0:
        st.warning("No game logs found for this player + stat in the DB.")
        if data.notes:
            for note in data.notes:
                st.info(note)
        return

    # ---- Charts (tabs) -------------------------------------------------
    tab_overview, tab_splits, tab_hitrate, tab_movement, tab_data = st.tabs(
        ["Overview", "Splits", "Hit Rate + Custom Line",
         "Line Movement", "Raw data"]
    )

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Recent games + rolling mean + median book line")
            st.plotly_chart(
                plc.build_recent_games_figure(data, rolling_window=int(rolling)),
                use_container_width=True,
                key=f"plotly_recent_single_{data.player_id}_{data.stat_type}",
            )
        with c2:
            st.caption("Distribution + book-line markers (hover / zoom enabled)")
            st.plotly_chart(
                plc.build_distribution_figure(
                    data,
                    distributions=tuple(overlays),
                    view_mode=line_view_mode,
                ),
                use_container_width=True,
                key=f"plotly_dist_single_{data.player_id}_{data.stat_type}_{line_view_mode}",
            )

    with tab_splits:
        st.caption("home/away mean and rest-day buckets")
        st.plotly_chart(
            plc.build_splits_figure(data),
            use_container_width=True,
            key=f"plotly_splits_single_{data.player_id}_{data.stat_type}",
        )
        splits = pc.compute_home_away_split(data)
        rest = pc.compute_rest_days_split(data)
        cols = st.columns(2)
        cols[0].markdown("**Home / Away**")
        cols[0].dataframe(
            pd.DataFrame([
                {"side": k, "mean": v["mean"], "n": v["n"]}
                for k, v in splits.items() if v
            ]),
            hide_index=True, use_container_width=True,
        )
        cols[1].markdown("**By rest days**")
        cols[1].dataframe(
            pd.DataFrame([
                {"rest": k, "mean": v["mean"], "n": v["n"]}
                for k, v in rest.items()
            ]),
            hide_index=True, use_container_width=True,
        )

        # Win/loss + starter/bench splits.
        wl = pc.compute_win_loss_split(data)
        sb = pc.compute_starter_bench_split(data)
        cols2 = st.columns(2)
        cols2[0].markdown("**Win / Loss**")
        cols2[0].dataframe(
            pd.DataFrame([
                {"result": k, "mean": v["mean"], "n": v["n"]}
                for k, v in wl.items() if v
            ]),
            hide_index=True, use_container_width=True,
        )
        cols2[1].markdown("**Starter / bench** (minutes ≥ 20)")
        cols2[1].dataframe(
            pd.DataFrame([
                {"role": k, "mean": v["mean"], "n": v["n"]}
                for k, v in sb.items() if v
            ]),
            hide_index=True, use_container_width=True,
        )

        st.markdown("---")
        st.caption("Distribution spread (box / quantiles)")
        st.plotly_chart(
            plc.build_box_quantile_figure(data),
            use_container_width=True,
            key=f"plotly_box_single_{data.player_id}_{data.stat_type}",
        )
        st.caption("Calendar performance (mean by weekday / month)")
        st.plotly_chart(
            plc.build_calendar_heatmap_figure(data),
            use_container_width=True,
            key=f"plotly_calendar_single_{data.player_id}_{data.stat_type}",
        )
        st.caption("Mean by opponent (matchup split)")
        st.plotly_chart(
            plc.build_opponent_split_figure(data),
            use_container_width=True,
            key=f"plotly_opp_single_{data.player_id}_{data.stat_type}",
        )
        st.caption("Minutes & per-minute productivity (projected-minutes baseline)")
        st.plotly_chart(
            plc.build_minutes_efficiency_figure(data),
            use_container_width=True,
            key=f"plotly_minutes_single_{data.player_id}_{data.stat_type}",
        )
        with st.expander("Stat correlations (parlay context)", expanded=False):
            try:
                corr = pc.compute_correlation_matrix(
                    db_path, data.player_id,
                    ["points", "rebounds", "assists"],
                )
            except Exception:  # noqa: BLE001
                corr = pd.DataFrame()
            st.plotly_chart(
                plc.build_correlation_heatmap_figure(corr, data.player_name),
                use_container_width=True,
                key=f"plotly_corr_single_{data.player_id}",
            )

    with tab_hitrate:
        st.caption("Historical over-rate per book line; 50% reference shown")
        st.plotly_chart(
            plc.build_hit_rate_figure(data),
            use_container_width=True,
            key=f"plotly_hitrate_single_{data.player_id}_{data.stat_type}",
        )
        st.markdown("---")
        st.markdown("### Book lines + EV")
        _render_book_lines_table(data)
        st.markdown("---")
        st.markdown("### Custom line probe")
        _custom_line_panel(data)
        st.markdown("---")
        st.markdown("### Model EV vs fitted-from-data EV")
        st.caption(
            "Stored model EV (predictions table) vs the EV implied by the "
            "player's current fitted normal at the same line + odds."
        )
        try:
            ev_df = pc.fetch_model_vs_fitted_ev(db_path, data)
        except Exception as exc:  # noqa: BLE001
            ev_df = None
            st.error(f"Failed to build model-vs-fitted EV: {exc}")
        if ev_df is not None:
            st.plotly_chart(
                plc.build_model_vs_fitted_ev_figure(ev_df, data.stat_type),
                use_container_width=True,
                key=f"plotly_modelev_single_{data.player_id}_{data.stat_type}",
            )

    with tab_movement:
        st.caption(
            "Open → current line drift per book from betting_line_snapshots "
            "(last 7 days)"
        )
        try:
            snapshots = pc.fetch_line_movement_snapshots(
                db_path=db_path,
                player_id=data.player_id,
                stat_type=data.stat_type,
            )
        except Exception as exc:  # noqa: BLE001
            snapshots = None
            st.error(f"Failed to load line snapshots: {exc}")
        if snapshots is not None:
            st.plotly_chart(
                plc.build_line_movement_figure(
                    snapshots,
                    stat_type=data.stat_type,
                    current_line=data.market_consensus_line,
                ),
                use_container_width=True,
                key=f"plotly_movement_single_{data.player_id}_{data.stat_type}",
            )
            if snapshots.empty:
                st.info(
                    "No line snapshots stored yet for this player + stat. "
                    "The timeline fills in as the daily/hourly ETL writes to "
                    "betting_line_snapshots."
                )
        st.markdown("---")
        st.caption("Game-by-game book line vs actual (over/under outcomes)")
        try:
            ribbon = pc.fetch_cumulative_line_vs_actual(
                db_path, data.player_id, data.stat_type,
            )
        except Exception as exc:  # noqa: BLE001
            ribbon = None
            st.error(f"Failed to build line-vs-actual: {exc}")
        if ribbon is not None:
            st.plotly_chart(
                plc.build_line_vs_actual_ribbon_figure(ribbon, data.stat_type),
                use_container_width=True,
                key=f"plotly_ribbon_single_{data.player_id}_{data.stat_type}",
            )
        st.caption("CLV proxy — open vs current line per book")
        try:
            clv = pc.fetch_clv_proxy_by_book(
                db_path, data.player_id, data.stat_type,
            )
        except Exception as exc:  # noqa: BLE001
            clv = None
            st.error(f"Failed to build CLV proxy: {exc}")
        if clv is not None:
            st.plotly_chart(
                plc.build_clv_proxy_figure(clv, data.stat_type),
                use_container_width=True,
                key=f"plotly_clv_single_{data.player_id}_{data.stat_type}",
            )

    with tab_data:
        st.caption(f"Last {data.values.size} games")
        view_df = data.games.copy()
        view_df.insert(0, "value", data.values)
        st.dataframe(view_df, use_container_width=True, hide_index=True)
        _csv_download_button(
            view_df,
            label="Download game logs as CSV",
            filename=(
                f"{data.player_name.replace(' ', '_')}_"
                f"{data.stat_type}_last{data.values.size}_games.csv"
            ),
            key=f"csv_game_logs_{data.player_id}_{data.stat_type}",
        )
        if data.notes:
            for note in data.notes:
                st.info(note)


if __name__ == "__main__":
    main()
