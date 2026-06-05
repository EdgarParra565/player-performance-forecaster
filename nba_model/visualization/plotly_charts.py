"""Plotly chart builders for the Streamlit web app.

We keep the matplotlib builders in ``player_charts.py`` because the Tk
desktop UI embeds them via ``FigureCanvasTkAgg`` and that path is already
wired and working. The Streamlit web app prefers interactive plots (hover
for exact values, click legend to isolate per-book lines, zoom + pan,
double-click to reset) so we provide Plotly equivalents here for every
chart the web app renders.

Caller policy:
  - Streamlit:  ``st.plotly_chart(plotly_charts.build_X(data))``
  - Tk desktop: ``player_charts.build_X(data)`` (unchanged)

Public surface used by ``nba_model/web/app.py``:

  - ``build_distribution_figure(data, distributions, view_mode="distribution")``
        the marquee chart: histogram + fitted-distribution overlays + per-book
        line markers + market consensus band. ``view_mode='ladder'`` switches
        to the compact horizontal "line ladder" alternative.
  - ``build_recent_games_figure(data, rolling_window=5)``
  - ``build_hit_rate_figure(data)``
  - ``build_splits_figure(data)``
  - ``build_line_movement_figure(snapshots_df, stat_type, line=None)``
        Bonus: line-movement timeline from ``betting_line_snapshots``.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from nba_model.model.simulation import SUPPORTED_DISTRIBUTIONS
from nba_model.visualization import player_charts as pc


# --- Color tables -------------------------------------------------------------

_BOOK_COLORS: dict[str, str] = {
    "fanduel":      "#1f77b4",
    "draftkings":   "#2ca02c",
    "betmgm":       "#d62728",
    "caesars":      "#9467bd",
    "betrivers":    "#8c564b",
    "fanatics":     "#e377c2",
    "bovada":       "#bcbd22",
    "prizepicks":   "#17becf",
    "underdog":     "#ff7f0e",
    "betonline.ag": "#7f7f7f",
}

_FALLBACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _book_color(book: str, idx: int) -> str:
    return _BOOK_COLORS.get(
        (book or "").strip().lower(),
        _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)],
    )


# --- Shared theme -------------------------------------------------------------
# One Plotly look applied to every builder so charts read as one product.
# Justification for staying on Plotly (rather than adding ECharts via
# streamlit-echarts): the existing builders are already mature, the
# distribution + ladder views need fine-grained per-trace styling that Plotly
# handles cleanly, and a pip-installable ECharts wrapper would add a runtime
# dep + JS bundle weight for a single chart. A well-tuned shared template
# closes the visual-polish gap with no new dependency.

THEME = {
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "text":        "#0f172a",
    "muted":       "#475569",
    "grid":        "#eef2f7",
    "axis":        "#cbd5e1",
    "hist":        "#6366f1",
    "accent":      "#ef4444",
    "consensus":   "#0f172a",
    "ok":          "#10b981",
    "warn":        "#f59e0b",
    "bg":          "rgba(0,0,0,0)",
}

# Per-builder default heights — generous so charts dominate the viewport.
DEFAULT_HEIGHTS = {
    "distribution":   560,
    "ladder":         460,
    "recent":         480,
    "hit_rate":       420,
    "splits":         440,
    "multi_player":   520,
    "line_movement":  480,
}


def _apply_theme(
    fig,
    *,
    title: str | None = None,
    height: int = 520,
    show_legend: bool = True,
    legend_top: bool = True,
    legend_bottom: bool = False,
    hovermode: str = "closest",
) -> None:
    """Apply the shared visual theme to ``fig`` in-place.

    Centralized so every chart picks up the same fonts, palette, hover
    styling, gridlines, and legend placement. Builders may still tweak axes
    after calling this (e.g. set a custom range) — this is the baseline.
    """
    title_block = None
    if title:
        title_block = dict(
            text=title,
            font=dict(family=THEME["font_family"], size=18, color=THEME["text"]),
            x=0.0, xanchor="left",
            pad=dict(t=10, b=8),
        )
    if not show_legend:
        legend = dict()
    elif legend_bottom:
        # Busy charts (per-book overlays) wrap into several legend rows; keep
        # them below the plot so they never grow up into the top-left title.
        legend = dict(
            font=dict(size=11, color=THEME["muted"], family=THEME["font_family"]),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=THEME["grid"], borderwidth=0,
            orientation="h",
            yanchor="top", y=-0.16,
            xanchor="left", x=0.0,
        )
    else:
        legend = dict(
            font=dict(size=11, color=THEME["muted"], family=THEME["font_family"]),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=THEME["grid"], borderwidth=0,
            orientation="h" if legend_top else "v",
            yanchor="bottom" if legend_top else "top",
            y=1.02 if legend_top else 1.0,
            xanchor="left", x=0.0,
        )
    fig.update_layout(
        title=title_block,
        font=dict(family=THEME["font_family"], size=13, color=THEME["text"]),
        height=height,
        margin=dict(
            l=28, r=24,
            t=80 if title else 36,
            b=104 if legend_bottom else 48,
        ),
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=THEME["axis"],
            font=dict(family=THEME["font_family"], size=12, color=THEME["text"]),
        ),
        hovermode=hovermode,
        showlegend=show_legend,
        legend=legend,
        bargap=0.08,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=THEME["grid"], gridwidth=1,
        zerolinecolor=THEME["grid"], linecolor=THEME["axis"],
        ticks="outside", tickcolor=THEME["axis"],
        tickfont=dict(size=11, color=THEME["muted"]),
        title_font=dict(size=12, color=THEME["muted"]),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=THEME["grid"], gridwidth=1,
        zerolinecolor=THEME["grid"], linecolor=THEME["axis"],
        ticks="outside", tickcolor=THEME["axis"],
        tickfont=dict(size=11, color=THEME["muted"]),
        title_font=dict(size=12, color=THEME["muted"]),
    )


def _apply_empty_theme(fig, *, height: int = 320) -> None:
    """Minimal theme for placeholder/no-data figures — keeps annotation count
    stable (tests assert exactly one annotation)."""
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        plot_bgcolor=THEME["bg"], paper_bgcolor=THEME["bg"],
        font=dict(family=THEME["font_family"], size=13, color=THEME["muted"]),
        showlegend=False,
    )


# --- Distribution figure (marquee chart) -------------------------------------

def _fit_overlay_traces(
    mu: float,
    sigma: float,
    values: np.ndarray,
    distributions: Iterable[str],
    x_lo: float,
    x_hi: float,
):
    """Yield ``go.Scatter`` traces for each requested fitted distribution.

    Supports every entry in ``simulation.SUPPORTED_DISTRIBUTIONS`` so the web
    selector reads the canonical vocabulary, plus an explicit
    ``negative_binomial`` overlay (which simulation doesn't yet ship).
    """
    import plotly.graph_objects as go  # local: heavy import
    from scipy.stats import nbinom

    xs = np.linspace(x_lo, x_hi, 200)
    out: list = []

    canonical = {(d or "").strip().lower().replace("-", "_") for d in distributions}

    if "normal" in canonical and sigma > 0:
        out.append(go.Scatter(
            x=xs, y=norm.pdf(xs, mu, sigma),
            mode="lines",
            name=f"Normal(mu={mu:.1f}, sigma={sigma:.1f})",
            line=dict(color="#222", width=2.5),
            hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>normal</extra>",
        ))

    if "poisson" in canonical and mu > 0:
        ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
        out.append(go.Scatter(
            x=ks, y=poisson.pmf(ks, mu),
            mode="lines+markers",
            name=f"Poisson(lambda={mu:.1f})",
            line=dict(color="#cc4125", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="k=%{x}<br>pmf=%{y:.3f}<extra>poisson</extra>",
        ))

    if canonical & {"negative_binomial", "nbinom", "negbin"}:
        var = float(np.var(values, ddof=1)) if values.size >= 2 else 0.0
        if var > mu > 0:
            p = mu / var
            r = mu * p / (1 - p)
            ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
            out.append(go.Scatter(
                x=ks, y=nbinom.pmf(ks, r, p),
                mode="lines+markers",
                name="NegBin (var > mu)",
                line=dict(color="#2ca02c", width=2, dash="dot"),
                marker=dict(size=6, symbol="x"),
                hovertemplate="k=%{x}<br>pmf=%{y:.3f}<extra>negbin</extra>",
            ))

    if "student_t" in canonical and sigma > 0:
        from scipy.stats import t as student_t
        dof = max(2.0, float(values.size - 1) if values.size >= 2 else 6.0)
        scale = sigma
        out.append(go.Scatter(
            x=xs, y=student_t.pdf(xs, dof, loc=mu, scale=scale),
            mode="lines",
            name=f"Student-t(dof={dof:.0f})",
            line=dict(color="#9467bd", width=2, dash="dot"),
            hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>student_t</extra>",
        ))

    if "lognormal" in canonical and mu > 0 and sigma > 0:
        positive_mean = max(mu, 1e-3)
        variance = max(sigma * sigma, 1e-6)
        phi = np.sqrt(variance + positive_mean * positive_mean)
        log_sigma = np.sqrt(max(np.log((phi * phi) / (positive_mean * positive_mean)), 1e-9))
        log_mu = np.log((positive_mean * positive_mean) / phi)
        from scipy.stats import lognorm
        out.append(go.Scatter(
            x=xs, y=lognorm.pdf(xs, s=log_sigma, scale=float(np.exp(log_mu))),
            mode="lines",
            name="LogNormal",
            line=dict(color="#17becf", width=2, dash="longdash"),
            hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>lognormal</extra>",
        ))

    if "exponential" in canonical and mu > 0:
        from scipy.stats import expon
        scale = max(sigma, 1e-3)
        shift = max(0.0, mu - scale)
        out.append(go.Scatter(
            x=xs, y=expon.pdf(xs, loc=shift, scale=scale),
            mode="lines",
            name="Exponential",
            line=dict(color="#bcbd22", width=2, dash="dash"),
            hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>exponential</extra>",
        ))

    if "uniform" in canonical:
        half = np.sqrt(3.0) * max(sigma, 1e-6)
        low, high = mu - half, mu + half
        if high > low:
            from scipy.stats import uniform
            out.append(go.Scatter(
                x=xs, y=uniform.pdf(xs, loc=low, scale=high - low),
                mode="lines",
                name="Uniform",
                line=dict(color="#7f7f7f", width=2, dash="dashdot"),
                hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>uniform</extra>",
            ))

    if "binomial" in canonical and mu > 0 and sigma > 0:
        from scipy.stats import binom
        variance = max(sigma * sigma, 1e-6)
        p = float(np.clip(1.0 - (variance / max(mu, 1e-6)), 1e-4, 0.999))
        n_trials = int(np.clip(np.ceil(mu / p), 1, 5000))
        ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
        out.append(go.Scatter(
            x=ks, y=binom.pmf(ks, n_trials, p),
            mode="lines+markers",
            name=f"Binomial(n={n_trials}, p={p:.2f})",
            line=dict(color="#e377c2", width=2, dash="dash"),
            marker=dict(size=5, symbol="circle"),
            hovertemplate="k=%{x}<br>pmf=%{y:.3f}<extra>binomial</extra>",
        ))

    if "power_law" in canonical and mu > 0 and sigma > 0:
        positive_mean = max(mu, 1e-3)
        variance = max(sigma * sigma, 1e-6)
        ratio = variance / (positive_mean * positive_mean)
        alpha = max(2.05, 1.0 + np.sqrt(1.0 + (1.0 / max(ratio, 1e-6))))
        x_m = positive_mean * (alpha - 1.0) / alpha
        xs_pl = np.linspace(max(x_m, 1e-3), x_hi, 200)
        pdf = (alpha * x_m ** alpha) / np.power(xs_pl, alpha + 1)
        out.append(go.Scatter(
            x=xs_pl, y=pdf, mode="lines",
            name=f"PowerLaw(alpha={alpha:.2f})",
            line=dict(color="#8c564b", width=2, dash="dot"),
            hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>power_law</extra>",
        ))

    return out


def _book_row_ev(data: "pc.PlayerChartData", row: pd.Series) -> dict:
    """Compute EV+P(over) for one book row using the fitted normal."""
    try:
        lv = float(row.get("line_value"))
    except (TypeError, ValueError):
        return {"line": None, "p_over": None, "ev_over": None, "ev_under": None}
    p = pc.fitted_prob_over(data, lv)
    ev_o = pc.expected_value(p, row.get("over_odds")) if p is not None else None
    ev_u = pc.expected_value(1 - p, row.get("under_odds")) if p is not None else None
    return {"line": lv, "p_over": p, "ev_over": ev_o, "ev_under": ev_u}


def _best_ev_indexes(book_rows: list[dict]) -> tuple[Optional[int], Optional[int]]:
    """Return (best_over_idx, best_under_idx) into ``book_rows``."""
    best_over_idx = None
    best_over_ev: float | None = None
    best_under_idx = None
    best_under_ev: float | None = None
    for i, br in enumerate(book_rows):
        ev_o = br.get("ev_over")
        ev_u = br.get("ev_under")
        if ev_o is not None and (best_over_ev is None or ev_o > best_over_ev):
            best_over_ev, best_over_idx = ev_o, i
        if ev_u is not None and (best_under_ev is None or ev_u > best_under_ev):
            best_under_ev, best_under_idx = ev_u, i
    return best_over_idx, best_under_idx


def build_distribution_figure(
    data: "pc.PlayerChartData",
    distributions: tuple[str, ...] = ("normal",),
    bins: int = 14,
    view_mode: str = "distribution",
):
    """Marquee chart: histogram + fitted overlays + per-book line markers.

    Parameters
    ----------
    data:
        ``PlayerChartData`` from ``player_charts.fetch_player_chart_data``.
    distributions:
        Iterable of distribution-family names. ``simulation.SUPPORTED_DISTRIBUTIONS``
        plus ``negative_binomial`` are recognized; unknown entries are ignored.
    bins:
        Histogram bin count.
    view_mode:
        ``"distribution"`` (default): histogram + book-line vertical markers,
        sigma band, market mean line.
        ``"ladder"``: compact horizontal line ladder (one row per book on a
        shared x-axis, sorted by line value). Much easier to read with 6+
        books; loses the histogram context.
    """
    import plotly.graph_objects as go  # noqa: PLC0415  (lazy: heavy import)

    if view_mode == "ladder":
        return _build_line_ladder_figure(data)

    fig = go.Figure()
    if data.values.size == 0:
        fig.add_annotation(
            text="No game data", showarrow=False, x=0.5, y=0.5,
            xref="paper", yref="paper",
            font=dict(size=14, color=THEME["muted"]),
        )
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["distribution"])
        return fig

    mu = float(data.mu)
    sigma = float(data.sigma)
    vals = data.values

    # Histogram (density-normalized so the fitted PDFs overlay correctly).
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=max(3, int(bins)),
        histnorm="probability density",
        name=f"recent games (n={vals.size})",
        marker=dict(color=THEME["hist"], line=dict(color="white", width=1)),
        opacity=0.55,
        hovertemplate="value=%{x}<br>density=%{y:.3f}<extra></extra>",
    ))

    if sigma > 0:
        x_lo = float(min(vals.min(), mu - 4 * sigma))
        x_hi = float(max(vals.max(), mu + 4 * sigma))
    else:
        x_lo, x_hi = float(vals.min() - 1), float(vals.max() + 1)
    if data.book_lines is not None and not data.book_lines.empty:
        line_vals = pd.to_numeric(
            data.book_lines["line_value"], errors="coerce"
        ).dropna()
        if not line_vals.empty:
            x_lo = min(x_lo, float(line_vals.min()) - 2)
            x_hi = max(x_hi, float(line_vals.max()) + 2)
    if data.stat_type != "minutes":
        x_lo = max(0.0, x_lo)
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_hi = x_lo + 1

    # Fitted distribution overlays (every supported family).
    for trace in _fit_overlay_traces(mu, sigma, vals, distributions, x_lo, x_hi):
        fig.add_trace(trace)

    consensus = pc.compute_market_consensus(data)
    if consensus["mean"] is not None:
        m = consensus["mean"]
        s = consensus["stdev"] or 0.0
        if s > 0:
            fig.add_vrect(
                x0=m - s, x1=m + s, line_width=0,
                fillcolor="#444", opacity=0.10,
                annotation_text=f"+/-1sigma ({s:.3f})",
                annotation_position="top left",
                annotation=dict(font=dict(size=10, color="#444")),
            )
        fig.add_vline(
            x=m, line=dict(color="#111", width=3),
            annotation_text=(
                f"book mean {m:.3f} ({consensus['n_books']} books)"
            ),
            annotation_position="top right",
            annotation=dict(font=dict(size=10, color="#111")),
        )

    # Per-book vertical markers — gathered into a top-rail of small triangles
    # PLUS dim vertical lines, so 6+ books don't collide. Best-EV side gets a
    # gold halo so it's obvious at a glance.
    if data.book_lines is not None and not data.book_lines.empty:
        rows = data.book_lines.reset_index(drop=True)
        per_book_evs: list[dict] = []
        per_book_meta: list[dict] = []
        for idx, row in rows.iterrows():
            ev = _book_row_ev(data, row)
            ev["book"] = str(row.get("book") or "").strip()
            ev["row_idx"] = idx
            ev["over_odds"] = row.get("over_odds")
            ev["under_odds"] = row.get("under_odds")
            per_book_evs.append(ev)
        best_over_idx, best_under_idx = _best_ev_indexes(per_book_evs)

        # Per-book consensus lookup so the hover can show delta vs consensus.
        pct_lookup = {p["book"].lower(): p for p in consensus["per_book"]}

        for i, br in enumerate(per_book_evs):
            if br["line"] is None:
                continue
            book = br["book"]
            color = _book_color(book, i)
            cm = consensus["mean"]
            delta = (br["line"] - cm) if cm is not None else None
            pct_entry = pct_lookup.get(book.lower())
            pct_str = pct_entry["pct_from_mean_str"] if pct_entry else ""
            is_best_over = (i == best_over_idx)
            is_best_under = (i == best_under_idx)
            highlight = is_best_over or is_best_under
            best_label_parts = []
            if is_best_over:
                best_label_parts.append("⭐ best OVER")
            if is_best_under:
                best_label_parts.append("⭐ best UNDER")
            best_suffix = (" " + " | ".join(best_label_parts)) if best_label_parts else ""
            ev_o = br["ev_over"]
            ev_u = br["ev_under"]
            hover = (
                f"<b>{book}</b>{best_suffix}"
                f"<br>line: {br['line']:.1f}"
                + (f"  (delta vs consensus: {delta:+.2f})" if delta is not None else "")
                + (f"<br>P(over): {br['p_over']:.1%}"
                   if br["p_over"] is not None else "")
                + (f"<br>EV over: {ev_o:+.3f}"
                   if ev_o is not None else "")
                + (f"<br>EV under: {ev_u:+.3f}"
                   if ev_u is not None else "")
                + "<extra></extra>"
            )
            line_width = 4 if highlight else 1.5
            line_dash = "solid" if highlight else "dash"
            fig.add_trace(go.Scatter(
                x=[br["line"], br["line"]],
                y=[0, 1],
                mode="lines",
                name=f"{book} {br['line']:.1f}"
                     + (f"  ({pct_str})" if pct_str else "")
                     + best_suffix,
                line=dict(color=color, width=line_width, dash=line_dash),
                showlegend=True,
                yaxis="y2",
                hovertemplate=hover,
                opacity=1.0 if highlight else 0.85,
            ))
            # Top-rail triangle marker — gives a clean "ladder" at the top of
            # the plot for line-shopping, even when verticals overlap.
            fig.add_trace(go.Scatter(
                x=[br["line"]],
                y=[0.97],
                mode="markers+text",
                marker=dict(
                    color=color,
                    size=18 if highlight else 12,
                    symbol="triangle-down",
                    line=dict(color="#111", width=2 if highlight else 0.5),
                ),
                text=[book[:3].upper()],
                textposition="top center",
                textfont=dict(size=9, color=color),
                name=f"{book} marker",
                yaxis="y2",
                showlegend=False,
                hovertemplate=hover,
            ))

    title = (f"{data.player_name} - {data.stat_type} distribution "
             f"(n={data.values.size})")
    if consensus["mean"] is not None and consensus["stdev"] is not None:
        title += (f"  |  book mean={consensus['mean']:.3f}"
                  f"  sigma={consensus['stdev']:.3f}"
                  f"  ({consensus['n_books']} book"
                  f"{'s' if consensus['n_books'] != 1 else ''})")

    _apply_theme(
        fig, title=title, height=DEFAULT_HEIGHTS["distribution"],
        legend_bottom=True,
    )
    fig.update_layout(
        xaxis_title=data.stat_type,
        yaxis_title="density",
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False, showticklabels=False,
            range=[0, 1.05], visible=False,
        ),
    )
    return fig


def _build_line_ladder_figure(data: "pc.PlayerChartData"):
    """Compact horizontal "line ladder" — one row per book, sorted by line.

    Much easier to read than the overlapping-verticals view when 6+ books
    are posting a line. Highlights the best-EV over and best-EV under in
    gold/green so line-shopping is obvious at a glance.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    if (data.book_lines is None or data.book_lines.empty
            or data.values.size == 0):
        fig.add_annotation(
            text="No book lines to render", showarrow=False, x=0.5, y=0.5,
            xref="paper", yref="paper",
            font=dict(size=14, color=THEME["muted"]),
        )
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["ladder"])
        return fig

    consensus = pc.compute_market_consensus(data)
    cm = consensus["mean"]
    rows: list[dict] = []
    for idx, raw in data.book_lines.reset_index(drop=True).iterrows():
        ev = _book_row_ev(data, raw)
        if ev["line"] is None:
            continue
        rows.append({
            "book": str(raw.get("book") or "").strip(),
            "line": ev["line"],
            "p_over": ev["p_over"],
            "ev_over": ev["ev_over"],
            "ev_under": ev["ev_under"],
            "over_odds": raw.get("over_odds"),
            "under_odds": raw.get("under_odds"),
            "idx": idx,
        })
    if not rows:
        fig.add_annotation(
            text="No numeric book lines", showarrow=False, x=0.5, y=0.5,
            xref="paper", yref="paper",
            font=dict(size=14, color=THEME["muted"]),
        )
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["ladder"])
        return fig

    best_over_idx, best_under_idx = _best_ev_indexes(rows)
    best_over_row_idx = rows[best_over_idx]["idx"] if best_over_idx is not None else None
    best_under_row_idx = rows[best_under_idx]["idx"] if best_under_idx is not None else None
    rows.sort(key=lambda r: r["line"])
    # Re-locate best-EV after the sort by matching the original book-lines row.
    best_over_pos = next(
        (i for i, r in enumerate(rows)
         if best_over_row_idx is not None and r["idx"] == best_over_row_idx),
        None,
    )
    best_under_pos = next(
        (i for i, r in enumerate(rows)
         if best_under_row_idx is not None and r["idx"] == best_under_row_idx),
        None,
    )

    ys = list(range(len(rows)))
    xs = [r["line"] for r in rows]
    colors = [_book_color(r["book"], i) for i, r in enumerate(rows)]
    sizes = [
        24 if (i == best_over_pos or i == best_under_pos) else 14
        for i, _ in enumerate(rows)
    ]
    symbols = [
        "star" if (i == best_over_pos or i == best_under_pos) else "circle"
        for i, _ in enumerate(rows)
    ]
    line_colors = [
        "#111" if (i == best_over_pos or i == best_under_pos) else "rgba(0,0,0,0.3)"
        for i, _ in enumerate(rows)
    ]
    hovers = []
    for r in rows:
        delta = (r["line"] - cm) if cm is not None else None
        parts = [f"<b>{r['book']}</b>", f"line: {r['line']:.1f}"]
        if delta is not None:
            parts.append(f"delta vs consensus: {delta:+.2f}")
        if r["p_over"] is not None:
            parts.append(f"P(over): {r['p_over']:.1%}")
        if r["ev_over"] is not None:
            parts.append(f"EV over: {r['ev_over']:+.3f}")
        if r["ev_under"] is not None:
            parts.append(f"EV under: {r['ev_under']:+.3f}")
        hovers.append("<br>".join(parts) + "<extra></extra>")

    fig.add_trace(go_scatter_marker(
        x=xs, y=ys, colors=colors, sizes=sizes, symbols=symbols,
        line_colors=line_colors, hovers=hovers,
    ))

    # Per-row connection to the consensus line so the user sees the spread.
    if cm is not None:
        for i, r in enumerate(rows):
            fig.add_shape(
                type="line", x0=cm, x1=r["line"], y0=i, y1=i,
                line=dict(color=_book_color(r["book"], i), width=2),
                layer="below",
            )
        fig.add_vline(
            x=cm, line=dict(color="#111", width=2, dash="dash"),
            annotation_text=f"consensus {cm:.2f}",
            annotation_position="top right",
        )

    fig.update_yaxes(
        tickvals=ys,
        ticktext=[
            (("⭐ " if (i == best_over_pos or i == best_under_pos) else "")
             + f"{r['book']}  {r['line']:.1f}")
            for i, r in enumerate(rows)
        ],
        autorange="reversed",  # alphabetical-ish + sorted by line, top-down reads ascending
    )
    title = f"{data.player_name} - {data.stat_type} line ladder ({len(rows)} books)"
    if best_over_pos is not None:
        title += f"  |  best OVER: {rows[best_over_pos]['book']} @ {rows[best_over_pos]['line']:.1f}"
    if best_under_pos is not None:
        title += f"  |  best UNDER: {rows[best_under_pos]['book']} @ {rows[best_under_pos]['line']:.1f}"
    ladder_height = max(DEFAULT_HEIGHTS["ladder"], 70 * len(rows) + 140)
    _apply_theme(fig, title=title, height=ladder_height, show_legend=False)
    fig.update_layout(xaxis_title=data.stat_type)
    fig.update_yaxes(showgrid=False)
    return fig


def go_scatter_marker(*, x, y, colors, sizes, symbols, line_colors, hovers):
    """Single-trace scatter with per-point styling (helper for the ladder)."""
    import plotly.graph_objects as go
    return go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            color=colors,
            size=sizes,
            symbol=symbols,
            line=dict(color=line_colors, width=2),
        ),
        hovertext=hovers,
        hovertemplate="%{hovertext}",
    )


# --- Recent games + rolling mean (Plotly) ------------------------------------

def build_recent_games_figure(
    data: "pc.PlayerChartData",
    rolling_window: int = 5,
):
    """Bar of last-N values + rolling-mean overlay + book-mean reference."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if data.values.size == 0 or data.games.empty:
        fig.add_annotation(text="No game data", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper",
                           font=dict(size=14, color=THEME["muted"]))
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["recent"])
        return fig

    dates = pd.to_datetime(data.games["game_date"]).dt.strftime("%Y-%m-%d")
    vals = data.values
    n = len(vals)
    xs = list(range(n))

    fig.add_trace(go.Bar(
        x=xs, y=vals, marker_color=THEME["hist"], opacity=0.85,
        name=data.stat_type,
        customdata=list(dates),
        hovertemplate=(
            "game %{customdata}<br>"
            + data.stat_type + "=%{y}<extra></extra>"
        ),
    ))
    if rolling_window > 1 and n >= rolling_window:
        roll = pd.Series(vals).rolling(rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=xs, y=roll, mode="lines+markers",
            name=f"rolling-{rolling_window} mean",
            line=dict(color=THEME["accent"], width=2.5),
            marker=dict(size=6),
            hovertemplate=f"rolling-{rolling_window} mean: %{{y:.2f}}<extra></extra>",
        ))
    if data.market_consensus_line is not None:
        fig.add_hline(
            y=float(data.market_consensus_line),
            line=dict(color=THEME["muted"], width=1.4, dash="dash"),
            annotation_text=f"book mean {data.market_consensus_line:.1f}",
            annotation_position="top right",
        )

    step = max(1, n // 12)
    _apply_theme(
        fig,
        title=f"{data.player_name} — last {n} games ({data.stat_type})",
        height=DEFAULT_HEIGHTS["recent"],
        hovermode="x unified",
    )
    fig.update_layout(xaxis_title="game date", yaxis_title=data.stat_type)
    fig.update_xaxes(
        tickvals=xs[::step],
        ticktext=[dates.iloc[i] for i in xs[::step]],
        tickangle=-30,
    )
    return fig


# --- Hit-rate vs book lines (Plotly) -----------------------------------------

def build_hit_rate_figure(data: "pc.PlayerChartData"):
    """Horizontal bar of hit-rate (% over) per book line in last N games."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if (data.values.size == 0
            or data.book_lines is None or data.book_lines.empty):
        msg = "No book lines to compute hit-rates."
        if data.values.size == 0:
            msg = "No game data."
        fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper",
                           font=dict(size=13, color=THEME["muted"]))
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["hit_rate"])
        return fig

    n = int(data.values.size)
    rows = []
    for idx, row in data.book_lines.reset_index(drop=True).iterrows():
        try:
            lv = float(row["line_value"])
        except (TypeError, ValueError):
            continue
        hits = int(np.sum(data.values > lv))
        rate = hits / n if n else 0.0
        rows.append({
            "book": str(row.get("book") or ""),
            "line": lv,
            "hits": hits,
            "hit_rate": rate * 100.0,
            "idx": idx,
        })
    if not rows:
        fig.add_annotation(text="No numeric book lines.", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           font=dict(size=13, color=THEME["muted"]))
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["hit_rate"])
        return fig

    rows.sort(key=lambda r: r["hit_rate"])
    labels = [f"{r['book']} ({r['line']:.1f})" for r in rows]
    colors = [_book_color(r["book"], i) for i, r in enumerate(rows)]

    fig.add_trace(go.Bar(
        x=[r["hit_rate"] for r in rows],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{r['hit_rate']:.0f}%  ({r['hits']}/{n})" for r in rows],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>hit-rate: %{x:.1f}%<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=50.0, line=dict(color=THEME["muted"], dash="dot", width=1.2),
        annotation_text="50%", annotation_position="top",
    )
    hr_height = max(DEFAULT_HEIGHTS["hit_rate"], 56 * len(rows) + 140)
    _apply_theme(
        fig,
        title=f"{data.player_name} — hit-rate vs book lines ({data.stat_type}, n={n})",
        height=hr_height,
        show_legend=False,
    )
    fig.update_layout(bargap=0.22)
    fig.update_xaxes(title_text="% of last N games OVER the line", range=[0, 110])
    fig.update_yaxes(showgrid=False)
    return fig


# --- Home/away + rest-day splits (Plotly) ------------------------------------

def build_splits_figure(data: "pc.PlayerChartData"):
    """Side-by-side bars: home vs away mean, and by rest-day buckets."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("home vs away mean", "by rest days"),
        horizontal_spacing=0.12,
    )

    if data.values.size == 0 or data.games.empty:
        fig.add_annotation(text="No game data.", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           font=dict(size=13, color=THEME["muted"]))
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["splits"])
        return fig

    splits = pc.compute_home_away_split(data)
    labels, means, ns = [], [], []
    for key in ("home", "away"):
        info = splits.get(key)
        if info is not None:
            labels.append(key)
            means.append(info["mean"])
            ns.append(info["n"])
    if labels:
        fig.add_trace(go.Bar(
            x=labels, y=means,
            marker_color=[THEME["hist"], THEME["accent"]][: len(labels)],
            text=[f"n={n}" for n in ns],
            textposition="outside",
            hovertemplate=("<b>%{x}</b><br>mean: %{y:.2f}<extra></extra>"),
            name="home/away",
        ), row=1, col=1)
    else:
        fig.add_annotation(text="No home/away tag",
                           xref="x domain", yref="y domain",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#666"), row=1, col=1)

    rest = pc.compute_rest_days_split(data)
    if rest:
        fig.add_trace(go.Bar(
            x=list(rest.keys()),
            y=[v["mean"] for v in rest.values()],
            marker_color=THEME["ok"],
            text=[f"n={v['n']}" for v in rest.values()],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b> rest days<br>mean: %{y:.2f}<extra></extra>"
            ),
            name="rest days",
        ), row=1, col=2)
    else:
        fig.add_annotation(text="Not enough date history",
                           xref="x2 domain", yref="y2 domain",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#666"), row=1, col=2)

    fig.update_yaxes(title_text=data.stat_type, row=1, col=1)
    fig.update_yaxes(title_text=data.stat_type, row=1, col=2)
    _apply_theme(
        fig,
        title=f"{data.player_name} splits ({data.stat_type})",
        height=DEFAULT_HEIGHTS["splits"],
        show_legend=False,
    )
    return fig


# --- Multi-player distribution overlay (Plotly) ------------------------------

def build_multi_player_distribution_figure(
    datasets: list["pc.PlayerChartData"],
    bins: int = 18,
):
    """Plotly overlay of 2-3 players' density histograms + fitted normals."""
    import plotly.graph_objects as go

    palette = [
        "#6366f1", "#ef4444", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4",
    ]
    fig = go.Figure()
    drawn = 0
    x_lo: Optional[float] = None
    x_hi: Optional[float] = None
    stat_labels: set[str] = set()
    for i, data in enumerate(datasets):
        if data.values.size == 0:
            continue
        color = palette[i % len(palette)]
        stat_labels.add(data.stat_type)
        fig.add_trace(go.Histogram(
            x=data.values,
            nbinsx=max(3, int(bins)),
            histnorm="probability density",
            name=f"{data.player_name} (n={data.values.size})",
            marker=dict(color=color, line=dict(color="white", width=1)),
            opacity=0.40,
            hovertemplate=(
                f"<b>{data.player_name}</b><br>value=%{{x}}"
                f"<br>density=%{{y:.3f}}<extra></extra>"
            ),
        ))
        mu = float(np.mean(data.values))
        sigma = (
            float(np.std(data.values, ddof=1))
            if data.values.size >= 2 else 0.0
        )
        if sigma > 0:
            span = (mu - 4 * sigma, mu + 4 * sigma)
        else:
            span = (data.values.min() - 1, data.values.max() + 1)
        x_lo = span[0] if x_lo is None else min(x_lo, span[0])
        x_hi = span[1] if x_hi is None else max(x_hi, span[1])
        if sigma > 0:
            xs = np.linspace(span[0], span[1], 200)
            fig.add_trace(go.Scatter(
                x=xs, y=norm.pdf(xs, mu, sigma),
                mode="lines",
                name=f"  fit: {data.player_name}",
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{data.player_name}</b> fit"
                    f"<br>mu={mu:.2f}, sigma={sigma:.2f}"
                    "<extra></extra>"
                ),
            ))
        if data.market_consensus_line is not None:
            fig.add_vline(
                x=float(data.market_consensus_line),
                line=dict(color=color, width=1.6, dash="dash"),
                annotation_text=(
                    f"{data.player_name} book mean "
                    f"{data.market_consensus_line:.2f}"
                ),
                annotation_position="top",
                annotation=dict(font=dict(size=9, color=color)),
            )
        drawn += 1

    if drawn == 0:
        fig.add_annotation(
            text="No game data for any selected player",
            showarrow=False, x=0.5, y=0.5,
            xref="paper", yref="paper",
            font=dict(size=12, color=THEME["muted"]),
        )
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["multi_player"])
        return fig

    if x_lo is not None and x_hi is not None and x_hi > x_lo:
        fig.update_xaxes(range=[max(0.0, x_lo), x_hi])

    stat_label = ", ".join(sorted(stat_labels)) if stat_labels else "value"
    _apply_theme(
        fig,
        title=f"Multi-player distribution overlay ({stat_label})",
        height=DEFAULT_HEIGHTS["multi_player"],
        legend_bottom=True,
    )
    fig.update_layout(
        xaxis_title=stat_label, yaxis_title="density", barmode="overlay",
    )
    return fig


# --- Bonus: line-movement timeline -------------------------------------------

def build_line_movement_figure(
    snapshots_df: pd.DataFrame,
    stat_type: str,
    current_line: float | None = None,
):
    """Line-movement timeline from ``betting_line_snapshots``.

    Expects a DataFrame with columns: ``snapshot_ts_utc``, ``book``,
    ``stat_type``, ``line_value``, optionally ``over_odds`` / ``under_odds``.
    One trace per book; markers at each snapshot, lines connecting them in
    time order. Empty input renders an explanatory annotation rather than
    erroring.
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    if snapshots_df is None or snapshots_df.empty:
        fig.add_annotation(
            text=("No snapshots yet for this player + stat. "
                  "The line-movement timeline populates as the daily ETL "
                  "writes to betting_line_snapshots."),
            showarrow=False, x=0.5, y=0.5,
            xref="paper", yref="paper",
            font=dict(size=12, color=THEME["muted"]),
        )
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["line_movement"])
        return fig

    df = snapshots_df.copy()
    df["ts"] = pd.to_datetime(df["snapshot_ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        fig.add_annotation(text="No usable snapshot timestamps.",
                           showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper",
                           font=dict(size=12, color=THEME["muted"]))
        _apply_empty_theme(fig, height=DEFAULT_HEIGHTS["line_movement"])
        return fig

    for idx, (book, group) in enumerate(df.groupby("book", sort=False)):
        color = _book_color(str(book), idx)
        fig.add_trace(go.Scatter(
            x=group["ts"], y=group["line_value"],
            mode="lines+markers",
            name=str(book),
            line=dict(color=color, width=2, shape="hv"),
            marker=dict(size=6, color=color, line=dict(color="#111", width=0.4)),
            hovertemplate=(
                f"<b>{book}</b><br>%{{x|%Y-%m-%d %H:%M UTC}}"
                f"<br>line: %{{y}}<extra></extra>"
            ),
        ))
    if current_line is not None:
        fig.add_hline(
            y=float(current_line),
            line=dict(color="#111", width=1.4, dash="dot"),
            annotation_text=f"latest median {current_line:.2f}",
            annotation_position="top right",
        )

    _apply_theme(
        fig,
        title=f"Line movement timeline ({stat_type})",
        height=DEFAULT_HEIGHTS["line_movement"],
    )
    fig.update_layout(
        xaxis_title="snapshot time (UTC)", yaxis_title=stat_type,
    )
    return fig
