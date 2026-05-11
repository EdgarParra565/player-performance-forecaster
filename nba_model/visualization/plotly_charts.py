"""Plotly chart builders (web-only).

We keep the matplotlib builders in `player_charts.py` because the Tk desktop
UI embeds them via `FigureCanvasTkAgg` and that path is already wired and
working. The Streamlit web app prefers interactive plots (hover for exact
values, click legend to isolate per-book lines, zoom + pan, double-click to
reset) so we provide Plotly equivalents here for the marquee charts.

Caller policy:
- Streamlit:  `st.plotly_chart(plotly_charts.build_distribution_figure(data))`
- Tk desktop: `player_charts.build_distribution_figure(data)` (unchanged)

Only the distribution figure is migrated right now since it's the most
information-dense chart. Other figure types stay matplotlib — they can
follow if/when users ask for interactivity on them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from nba_model.visualization import player_charts as pc


_BOOK_COLORS = {
    "fanduel":     "#1f77b4",
    "draftkings":  "#2ca02c",
    "betmgm":      "#d62728",
    "caesars":     "#9467bd",
    "betrivers":   "#8c564b",
    "fanatics":    "#e377c2",
    "bovada":      "#bcbd22",
    "prizepicks":  "#17becf",
    "underdog":    "#ff7f0e",
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


def build_distribution_figure(
    data: pc.PlayerChartData,
    distributions: tuple[str, ...] = ("normal",),
    bins: int = 14,
):
    """Plotly version of the distribution histogram + fitted-curve overlays.

    Returns a `plotly.graph_objects.Figure`. Caller renders via
    `st.plotly_chart(fig, use_container_width=True)`. Every trace is
    individually toggleable via the legend, hover shows exact x/y values, and
    the user can zoom + pan + double-click to reset.
    """
    import plotly.graph_objects as go  # noqa: PLC0415  (lazy: heavy import)

    fig = go.Figure()
    if data.values.size == 0:
        fig.add_annotation(
            text="No game data", showarrow=False, x=0.5, y=0.5,
            xref="paper", yref="paper", font=dict(size=14, color="#666"),
        )
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    mu = float(data.mu)
    sigma = float(data.sigma)
    vals = data.values

    # Histogram (density-normalized so the fitted PDFs overlay correctly).
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=max(3, int(bins)),
        histnorm="probability density",
        name=f"recent games (n={vals.size})",
        marker=dict(color="#4a86e8", line=dict(color="white", width=1)),
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

    xs = np.linspace(x_lo, x_hi, 200)
    for dist in distributions:
        key = (dist or "").strip().lower().replace("-", "_")
        if key == "normal" and sigma > 0:
            fig.add_trace(go.Scatter(
                x=xs, y=norm.pdf(xs, mu, sigma),
                mode="lines", name=f"Normal(mu={mu:.1f}, sigma={sigma:.1f})",
                line=dict(color="#222", width=2.5),
                hovertemplate="x=%{x:.2f}<br>density=%{y:.3f}<extra>normal</extra>",
            ))
        elif key == "poisson" and mu > 0:
            ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
            fig.add_trace(go.Scatter(
                x=ks, y=poisson.pmf(ks, mu),
                mode="lines+markers", name=f"Poisson(lambda={mu:.1f})",
                line=dict(color="#cc4125", width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="k=%{x}<br>pmf=%{y:.3f}<extra>poisson</extra>",
            ))
        elif key in {"negative_binomial", "nbinom", "negbin"}:
            var = float(np.var(vals, ddof=1)) if vals.size >= 2 else 0.0
            if var > mu > 0:
                p = mu / var
                r = mu * p / (1 - p)
                ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
                fig.add_trace(go.Scatter(
                    x=ks, y=nbinom.pmf(ks, r, p),
                    mode="lines+markers", name="NegBin (var>mu)",
                    line=dict(color="#2ca02c", width=2, dash="dot"),
                    marker=dict(size=6, symbol="x"),
                    hovertemplate="k=%{x}<br>pmf=%{y:.3f}<extra>negbin</extra>",
                ))

    # Market consensus across books: bold line at mean, shaded sigma band.
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

    # Per-book markers — each is its own trace so the user can toggle them in
    # the legend.
    if data.book_lines is not None and not data.book_lines.empty:
        for idx, row in data.book_lines.reset_index(drop=True).iterrows():
            book = str(row.get("book") or "").strip()
            try:
                lv = float(row.get("line_value"))
            except (TypeError, ValueError):
                continue
            entry = next(
                (p for p in consensus["per_book"]
                 if p["book"].lower() == book.lower()),
                None,
            )
            pct = entry["pct_from_mean_str"] if entry else ""
            color = _book_color(book, idx)
            label = f"{book} {lv:.1f}" + (f"  ({pct})" if pct else "")
            # Use a tiny zero-y scatter so the marker shows in the legend
            # and the user can hover for the exact line value.
            fig.add_trace(go.Scatter(
                x=[lv, lv],
                y=[0, 1],  # full height; we set y on paper coords below
                mode="lines", name=label,
                line=dict(color=color, width=2, dash="dash"),
                showlegend=True,
                yaxis="y2",
                hovertemplate=(f"<b>{book}</b><br>line=%{{x:.1f}}"
                               f"<br>{pct}<extra></extra>")
                if pct else f"<b>{book}</b><br>line=%{{x:.1f}}<extra></extra>",
            ))

    title = (f"{data.player_name} - {data.stat_type} distribution "
             f"(n={data.values.size})")
    if consensus["mean"] is not None and consensus["stdev"] is not None:
        title += (f"  |  book mean={consensus['mean']:.3f}"
                  f"  sigma={consensus['stdev']:.3f}"
                  f"  ({consensus['n_books']} book"
                  f"{'s' if consensus['n_books'] != 1 else ''})")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=data.stat_type,
        yaxis_title="density",
        bargap=0.05,
        hovermode="x unified",
        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=10, r=10, t=60, b=10),
        # Second y-axis used by the per-book vertical lines so they span the
        # full plot height regardless of the histogram density.
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False, showticklabels=False,
            range=[0, 1], visible=False,
        ),
    )
    return fig
