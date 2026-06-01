"""Player-charts data + matplotlib figure builders for the desktop UI.

Pure functions: take inputs (or a db_path) and return matplotlib Figures.
The UI layer is responsible for embedding the figures into a Tk canvas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm, poisson, nbinom

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.scrapers.team_names import team_code_to_canonical

DISTRIBUTION_CHOICES = ("normal", "poisson", "negative_binomial")

# The set of books we'd expect to see for a major-market NBA player+stat.
# Lowercase canonical names; matched case-insensitively. Used to compute
# `books_missing` so the UI can call out coverage gaps.
EXPECTED_BOOKS: tuple[str, ...] = (
    "fanduel",
    "draftkings",
    "betmgm",
    "caesars",
    "betrivers",
    "bovada",
    "betonline.ag",
    "fanatics",
    "prizepicks",
    "underdog",
)

STAT_COLUMN_BY_TYPE = {
    "points": "points",
    "assists": "assists",
    "rebounds": "rebounds",
    "minutes": "minutes",
    "three_pointers_made": "fg3m",
    "field_goals_made": "fgm",
}

# Aliases the UI may pass; collapsed to canonical names. Books and
# web_prop_cards sometimes use shorter labels (e.g. "3pm", "fgm").
STAT_TYPE_ALIASES = {
    "3pm": "three_pointers_made",
    "three_pointers": "three_pointers_made",
    "3-pointers": "three_pointers_made",
    "threes": "three_pointers_made",
    "fgm": "field_goals_made",
    "field_goals": "field_goals_made",
    "ra": "ra",  # rebounds + assists (present in web_prop_cards)
}


@dataclass
class PlayerChartData:
    player_id: int
    player_name: str
    stat_type: str
    games: pd.DataFrame
    values: np.ndarray
    book_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_consensus_line: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def market_median_line(self) -> Optional[float]:
        """Backward-compat alias — returns the consensus (mean) line."""
        return self.market_consensus_line

    @property
    def mu(self) -> float:
        return float(np.mean(self.values)) if self.values.size else 0.0

    @property
    def sigma(self) -> float:
        if self.values.size < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))


def _canonical_stat_type(stat_type: str) -> str:
    s = (stat_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    return STAT_TYPE_ALIASES.get(s, s)


def _series_for_stat(games: pd.DataFrame, stat_type: str) -> np.ndarray:
    """Return numeric stat series, computing PRA / RA on demand."""
    stat_type = _canonical_stat_type(stat_type)
    if stat_type == "pra":
        for col in ("points", "rebounds", "assists"):
            if col not in games.columns:
                return np.array([], dtype=float)
        s = (
            games["points"].fillna(0)
            + games["rebounds"].fillna(0)
            + games["assists"].fillna(0)
        )
        return s.astype(float).to_numpy()
    if stat_type == "ra":
        for col in ("rebounds", "assists"):
            if col not in games.columns:
                return np.array([], dtype=float)
        s = games["rebounds"].fillna(0) + games["assists"].fillna(0)
        return s.astype(float).to_numpy()
    col = STAT_COLUMN_BY_TYPE.get(stat_type)
    if not col or col not in games.columns:
        return np.array([], dtype=float)
    return games[col].fillna(0).astype(float).to_numpy()


def fetch_player_chart_data(
    db_path: str,
    player_id: int,
    player_name: str,
    stat_type: str,
    n_games: int = 25,
) -> PlayerChartData:
    """Pull recent game logs + most-recent book lines for a player+stat."""
    notes: list[str] = []
    canonical = _canonical_stat_type(stat_type)
    with DatabaseManager(db_path=db_path) as db:
        games = db.get_player_games(player_id, n_games=max(1, int(n_games)))
        # ASC for plotting (oldest -> newest); db returns DESC.
        games = games.sort_values("game_date", ascending=True).reset_index(drop=True)
        values = _series_for_stat(games, canonical)

        book_lines = _fetch_latest_book_lines(
            db, player_id, canonical, player_name=player_name,
        )
        market_line = _market_consensus_line(book_lines)

    if games.empty:
        notes.append("No game logs found for this player.")
    if book_lines.empty:
        notes.append(f"No book lines found for {canonical}.")

    return PlayerChartData(
        player_id=int(player_id),
        player_name=str(player_name),
        stat_type=canonical,
        games=games,
        values=values,
        book_lines=book_lines,
        market_consensus_line=market_line,
        notes=notes,
    )


def _fetch_latest_book_lines(
    db: DatabaseManager,
    player_id: int,
    stat_type: str,
    player_name: Optional[str] = None,
    web_lookback_hours: float = 48.0,
) -> pd.DataFrame:
    """Return the most recent betting_lines + web_prop_cards row per book.

    The `player_name` argument is used to join `web_prop_cards` directly,
    avoiding the indirection through the sparse `players` table.  When it's
    not supplied we fall back to the canonical name from
    `nba_active_players_ref` (530+ rows, kept in sync by the scraper) and
    finally `players` (94 rows, mostly stale).
    """
    query = """
        WITH ranked AS (
            SELECT book, line_value, over_odds, under_odds, game_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY book
                       ORDER BY game_date DESC, scraped_at DESC
                   ) AS rn
            FROM betting_lines
            WHERE player_id = ? AND lower(stat_type) = lower(?)
        )
        SELECT book, line_value, over_odds, under_odds, game_date
        FROM ranked WHERE rn = 1
        ORDER BY book ASC
    """
    df = pd.read_sql_query(query, db.conn, params=(int(player_id), str(stat_type)))

    resolved_name = (player_name or "").strip()
    if not resolved_name:
        row = db.conn.execute(
            "SELECT player_name FROM nba_active_players_ref WHERE player_id = ?",
            (int(player_id),),
        ).fetchone()
        if row and row[0]:
            resolved_name = str(row[0])
    if not resolved_name:
        row = db.conn.execute(
            "SELECT name FROM players WHERE player_id = ?", (int(player_id),)
        ).fetchone()
        if row and row[0]:
            resolved_name = str(row[0])

    if resolved_name:
        # Latest line per (book, side) for this player+stat, restricted to the
        # configured lookback window so stale snapshots from previous slates
        # don't pollute today's consensus.
        web_query = """
            WITH ranked AS (
                SELECT book, line_value, side, observed_at_utc,
                       ROW_NUMBER() OVER (
                           PARTITION BY lower(book), lower(side)
                           ORDER BY observed_at_utc DESC
                       ) AS rn
                FROM web_prop_cards
                WHERE lower(player_name) = lower(?)
                  AND lower(stat_type) = lower(?)
                  AND observed_at_utc >= datetime('now', ?)
            )
            SELECT book, line_value, side, observed_at_utc
            FROM ranked WHERE rn = 1
        """
        web_df = pd.read_sql_query(
            web_query,
            db.conn,
            params=(resolved_name, str(stat_type),
                    f"-{float(web_lookback_hours)} hours"),
        )
        if not web_df.empty:
            agg = (
                web_df.groupby("book", as_index=False)
                .agg(line_value=("line_value", "median"),
                     game_date=("observed_at_utc", "max"))
            )
            agg["over_odds"] = None
            agg["under_odds"] = None
            agg = agg[["book", "line_value", "over_odds", "under_odds", "game_date"]]
            existing_books = set(df["book"].str.lower()) if not df.empty else set()
            agg = agg[~agg["book"].str.lower().isin(existing_books)]
            df = pd.concat([df, agg], ignore_index=True) if not agg.empty else df
    return df.sort_values("book").reset_index(drop=True)


def _market_consensus_line(book_lines: pd.DataFrame) -> Optional[float]:
    """Mean line across all books that posted a value for this player+stat.

    Each book contributes one value (the latest, deduped upstream in
    ``_fetch_latest_book_lines``).  This is the "consensus" reference line
    surfaced on the recent-games chart.
    """
    if book_lines.empty:
        return None
    vals = pd.to_numeric(book_lines["line_value"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(np.mean(vals))


def compute_market_consensus(
    data: "PlayerChartData",
    expected_books: tuple[str, ...] = EXPECTED_BOOKS,
) -> dict:
    """Aggregate book lines into a market-consensus view.

    Returns:
        {
          "mean":        float | None,    # average of book lines
          "stdev":       float | None,    # population stdev across books
          "spread":      float | None,    # max - min
          "n_books":     int,
          "books_used":  list[str],       # books that contributed
          "books_missing": list[str],     # in expected_books but absent
          "per_book": [{                  # one entry per book, mean-relative
              "book": str,
              "line": float,
              "delta": float,             # line - mean
              "pct_from_mean": float,     # (line - mean) / mean * 100
              "pct_from_mean_str": str,   # formatted to 3 decimals, signed
          }],
        }

    `expected_books` is matched case-insensitively. Books present in the data
    that aren't in the expected set are still counted in `books_used`.
    """
    out = {
        "mean": None,
        "stdev": None,
        "spread": None,
        "n_books": 0,
        "books_used": [],
        "books_missing": [],
        "per_book": [],
    }
    bl = data.book_lines
    if bl is None or bl.empty:
        out["books_missing"] = sorted(set(b.lower() for b in expected_books))
        return out

    rows = []
    for _, row in bl.iterrows():
        try:
            lv = float(row["line_value"])
        except (TypeError, ValueError):
            continue
        rows.append({"book": str(row.get("book") or "").strip(), "line": lv})
    if not rows:
        out["books_missing"] = sorted(set(b.lower() for b in expected_books))
        return out

    vals = np.array([r["line"] for r in rows], dtype=float)
    mean = float(np.mean(vals))
    stdev = float(np.std(vals, ddof=0)) if vals.size > 1 else 0.0
    spread = float(vals.max() - vals.min()) if vals.size > 1 else 0.0
    used_lower = {r["book"].lower() for r in rows if r["book"]}
    missing = sorted(b.lower() for b in expected_books if b.lower() not in used_lower)

    per_book = []
    for r in rows:
        delta = r["line"] - mean
        pct = (delta / mean * 100.0) if mean != 0 else 0.0
        per_book.append({
            "book": r["book"],
            "line": r["line"],
            "delta": float(delta),
            "pct_from_mean": float(pct),
            "pct_from_mean_str": f"{pct:+.3f}%",
        })
    per_book.sort(key=lambda p: p["pct_from_mean"])

    out.update({
        "mean": mean,
        "stdev": stdev,
        "spread": spread,
        "n_books": len(rows),
        "books_used": [r["book"] for r in rows if r["book"]],
        "books_missing": missing,
        "per_book": per_book,
    })
    return out


def build_recent_games_figure(
    data: PlayerChartData,
    rolling_window: int = 5,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> Figure:
    """Bar chart of last N values + rolling-mean overlay + book median line."""
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    if data.values.size == 0 or data.games.empty:
        ax.text(0.5, 0.5, "No game data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    dates = pd.to_datetime(data.games["game_date"]).dt.strftime("%m/%d")
    n = len(data.values)
    xs = np.arange(n)
    ax.bar(xs, data.values, color="#4a86e8", alpha=0.75, label=data.stat_type)

    if rolling_window > 1 and n >= rolling_window:
        roll = pd.Series(data.values).rolling(rolling_window, min_periods=1).mean()
        ax.plot(xs, roll, color="#cc4125", linewidth=2,
                label=f"rolling-{rolling_window} mean")

    if data.market_consensus_line is not None:
        ax.axhline(
            data.market_consensus_line, color="#666", linestyle="--",
            linewidth=1.2, label=f"book mean {data.market_consensus_line:.1f}",
        )

    tick_step = max(1, n // 10)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(dates.iloc[::tick_step], rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel(data.stat_type)
    ax.set_title(f"{data.player_name} - last {n} games ({data.stat_type})")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    return fig


_BOOK_COLORS = {
    "fanduel": "#1f77b4",
    "draftkings": "#2ca02c",
    "betmgm": "#d62728",
    "caesars": "#9467bd",
    "betrivers": "#8c564b",
    "fanatics": "#e377c2",
    "bovada": "#bcbd22",
    "prizepicks": "#17becf",
    "underdog": "#ff7f0e",
}


def _book_color(name: str, idx: int) -> str:
    base = (name or "").strip().lower()
    if base in _BOOK_COLORS:
        return _BOOK_COLORS[base]
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return palette[idx % len(palette)]


def build_multi_player_distribution_figure(
    datasets: list["PlayerChartData"],
    figsize: tuple[float, float] = (7.5, 4.0),
    bins: int = 18,
) -> Figure:
    """Overlay distribution histograms + fitted normals for 2-3 players.

    Each player gets a translucent histogram (density-normalized so heights
    are comparable across different sample sizes) and a thicker fitted-normal
    curve in the same color. Their book-consensus means are drawn as dashed
    vertical markers labeled with the player's name.

    Used by the "Multi-player compare" view in the Streamlit app to spot
    which leg is the strongest fit for a parlay.
    """
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    palette = [
        "#1f77b4", "#cc4125", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    ]
    drawn = 0
    x_lo: Optional[float] = None
    x_hi: Optional[float] = None
    for idx, data in enumerate(datasets):
        if data.values.size == 0:
            continue
        color = palette[idx % len(palette)]
        ax.hist(
            data.values, bins=max(3, int(bins)), density=True,
            color=color, alpha=0.30, edgecolor="white",
            label=f"{data.player_name} (n={data.values.size})",
        )
        mu = float(np.mean(data.values))
        sigma = float(np.std(data.values, ddof=1)) if data.values.size >= 2 else 0.0
        if sigma > 0:
            span = (mu - 4 * sigma, mu + 4 * sigma)
        else:
            span = (data.values.min() - 1, data.values.max() + 1)
        x_lo = span[0] if x_lo is None else min(x_lo, span[0])
        x_hi = span[1] if x_hi is None else max(x_hi, span[1])
        if sigma > 0:
            xs = np.linspace(span[0], span[1], 200)
            ax.plot(
                xs, norm.pdf(xs, mu, sigma),
                color=color, linewidth=2,
                label=f"  fit: mu={mu:.2f}, sigma={sigma:.2f}",
            )
        # Consensus marker.
        cm = data.market_consensus_line
        if cm is not None:
            ax.axvline(
                cm, color=color, linestyle="--", linewidth=1.4, alpha=0.85,
                label=f"  {data.player_name} book mean {cm:.2f}",
            )
        drawn += 1

    if drawn == 0:
        ax.text(0.5, 0.5, "No game data for any selected player",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    if x_lo is not None and x_hi is not None and x_hi > x_lo:
        ax.set_xlim(max(0.0, x_lo), x_hi)
    stat_label = ", ".join(
        sorted({d.stat_type for d in datasets if d.values.size})
    )
    ax.set_xlabel(stat_label or "value")
    ax.set_ylabel("density")
    ax.set_title(f"Multi-player distribution overlay ({stat_label})")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=1)
    return fig


def build_distribution_figure(
    data: PlayerChartData,
    figsize: tuple[float, float] = (7.0, 3.6),
    bins: int = 12,
    distributions: tuple[str, ...] = ("normal",),
) -> Figure:
    """Histogram + fitted distribution overlays + per-book line markers.

    `distributions` may include any of: "normal", "poisson", "negative_binomial".
    """
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    if data.values.size == 0:
        ax.text(0.5, 0.5, "No game data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    mu, sigma = data.mu, data.sigma
    vals = data.values

    ax.hist(vals, bins=max(3, int(bins)), density=True,
            color="#4a86e8", alpha=0.45, edgecolor="white", label="recent games")

    if sigma > 0:
        x_lo = min(vals.min(), mu - 4 * sigma)
        x_hi = max(vals.max(), mu + 4 * sigma)
    else:
        x_lo, x_hi = vals.min() - 1, vals.max() + 1
    if data.book_lines is not None and not data.book_lines.empty:
        line_vals = pd.to_numeric(
            data.book_lines["line_value"], errors="coerce").dropna()
        if not line_vals.empty:
            x_lo = min(x_lo, float(line_vals.min()) - 2)
            x_hi = max(x_hi, float(line_vals.max()) + 2)
    x_lo = max(0.0, x_lo) if data.stat_type != "minutes" else x_lo
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_hi = x_lo + 1

    xs = np.linspace(x_lo, x_hi, 400)
    dist_styles = {
        "normal":            ("#222",    "-",  f"Normal(mu={mu:.1f}, sigma={sigma:.1f})"),
        "poisson":           ("#cc4125", "--", f"Poisson(lambda={mu:.1f})"),
        "negative_binomial": ("#2ca02c", ":",  "NegBin (var>mu)"),
    }
    for dist in distributions:
        key = (dist or "").strip().lower().replace("-", "_")
        if key == "normal" and sigma > 0:
            ax.plot(xs, norm.pdf(xs, mu, sigma),
                    color=dist_styles["normal"][0], linestyle=dist_styles["normal"][1],
                    linewidth=2, label=dist_styles["normal"][2])
        elif key == "poisson" and mu > 0:
            ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
            ax.plot(ks, poisson.pmf(ks, mu),
                    color=dist_styles["poisson"][0], linestyle=dist_styles["poisson"][1],
                    linewidth=2, marker="o", markersize=3,
                    label=dist_styles["poisson"][2])
        elif key in {"negative_binomial", "nbinom", "negbin"}:
            var = float(np.var(vals, ddof=1)) if vals.size >= 2 else 0.0
            if var > mu > 0:
                p = mu / var
                r = mu * p / (1 - p)
                ks = np.arange(int(max(0, x_lo)), int(np.ceil(x_hi)) + 1)
                ax.plot(ks, nbinom.pmf(ks, r, p),
                        color=dist_styles["negative_binomial"][0],
                        linestyle=dist_styles["negative_binomial"][1],
                        linewidth=2, marker="x", markersize=4,
                        label=dist_styles["negative_binomial"][2])

    # Market consensus across books: bold line at mean, shaded +/- 1 stdev band.
    consensus = compute_market_consensus(data)
    consensus_lookup = {p["book"].lower(): p for p in consensus["per_book"]}
    if consensus["mean"] is not None:
        m = consensus["mean"]
        s = consensus["stdev"] or 0.0
        if s > 0:
            ax.axvspan(m - s, m + s, color="#444", alpha=0.08,
                       label=f"book +/-1sigma ({s:.3f})")
        ax.axvline(
            m, color="#111", linestyle="-", linewidth=2.4, alpha=0.95,
            label=f"book mean {m:.3f} (n={consensus['n_books']})",
        )

    if data.book_lines is not None and not data.book_lines.empty:
        for idx, row in data.book_lines.reset_index(drop=True).iterrows():
            book = str(row.get("book") or "").strip()
            try:
                lv = float(row.get("line_value"))
            except (TypeError, ValueError):
                continue
            color = _book_color(book, idx)
            pct_str = ""
            entry = consensus_lookup.get(book.lower())
            if entry is not None:
                pct_str = f" ({entry['pct_from_mean_str']})"
            ax.axvline(lv, color=color, linestyle="--", linewidth=1.6,
                       alpha=0.9, label=f"{book} {lv:.1f}{pct_str}")

    ax.set_xlabel(data.stat_type)
    ax.set_ylabel("density")
    title = (f"{data.player_name} - {data.stat_type} distribution "
             f"(n={data.values.size})")
    if consensus["mean"] is not None and consensus["stdev"] is not None:
        title += (f"  |  book mean={consensus['mean']:.3f}"
                  f"  sigma={consensus['stdev']:.3f}"
                  f"  ({consensus['n_books']} book"
                  f"{'s' if consensus['n_books'] != 1 else ''})")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=1)
    return fig


def build_hit_rate_figure(
    data: PlayerChartData,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> Figure:
    """Horizontal bar chart of hit-rate (% over) per book line in last N games.

    A reference line at 50% sits next to each bar; book-implied break-even is
    drawn as a small marker when over_odds is available.
    """
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)

    if (data.values.size == 0
            or data.book_lines is None or data.book_lines.empty):
        msg = "No book lines to compute hit-rates."
        if data.values.size == 0:
            msg = "No game data."
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    rows = []
    n = int(data.values.size)
    for _, row in data.book_lines.iterrows():
        try:
            lv = float(row["line_value"])
        except (TypeError, ValueError):
            continue
        hits = int(np.sum(data.values > lv))
        pushes = int(np.sum(data.values == lv))
        hit_rate = hits / n if n else 0.0
        breakeven = _american_odds_breakeven(row.get("over_odds"))
        rows.append({
            "book": str(row.get("book") or ""),
            "line": lv,
            "hits": hits,
            "pushes": pushes,
            "hit_rate": hit_rate,
            "breakeven": breakeven,
        })

    if not rows:
        ax.text(0.5, 0.5, "No numeric book lines.", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    df = pd.DataFrame(rows).sort_values("hit_rate", ascending=True).reset_index(drop=True)
    ys = np.arange(len(df))
    bar_colors = [_book_color(b, i) for i, b in enumerate(df["book"])]
    ax.barh(ys, df["hit_rate"] * 100.0, color=bar_colors, alpha=0.85)
    for i, row in df.iterrows():
        label = f"  {row['hit_rate']*100:.0f}%  ({row['hits']}/{n})"
        ax.text(row["hit_rate"] * 100.0, i, label, va="center",
                fontsize=8, color="#222")
        if row["breakeven"] is not None:
            ax.plot(row["breakeven"] * 100.0, i, marker="|",
                    markersize=14, color="#222", linewidth=2)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['book']} ({r['line']:.1f})" for _, r in df.iterrows()],
                       fontsize=9)
    ax.axvline(50.0, color="#888", linestyle=":", linewidth=1)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of last N games OVER the line  |  '|' = book break-even")
    ax.set_title(
        f"{data.player_name} - hit-rate vs book lines ({data.stat_type}, n={n})"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return fig


def build_rolling_ci_figure(
    data: PlayerChartData,
    rolling_window: int = 5,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> Figure:
    """Recent-games chart with a bootstrap confidence band around the rolling mean.

    For each game in the window, computes a centered rolling mean over the
    last ``rolling_window`` games and a 95 % CI by bootstrapping ``n_bootstrap``
    resamples of that window. The shaded band shows how much the rolling
    estimate can swing under resampling — useful for spotting genuine
    trend shifts vs. noise.
    """
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    if data.values.size == 0 or data.games.empty:
        ax.text(0.5, 0.5, "No game data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    values = data.values
    n = len(values)
    window = max(1, int(rolling_window))
    xs = np.arange(n)

    # Rolling mean (right-aligned to match the recent-games chart).
    roll = pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()

    # Bootstrap CI per game-position.  At each position i we have the
    # previous ``window`` values; resampling them with replacement and
    # taking the mean ``n_bootstrap`` times gives the spread of plausible
    # rolling means at that position.
    lo_pct = (1.0 - confidence) / 2.0 * 100.0
    hi_pct = (1.0 + confidence) / 2.0 * 100.0
    rng = np.random.default_rng(seed=42)  # deterministic so chart re-renders match
    lo_band = np.empty(n)
    hi_band = np.empty(n)
    for i in range(n):
        sub = values[max(0, i - window + 1): i + 1]
        if sub.size < 2:
            lo_band[i] = hi_band[i] = sub.mean() if sub.size else 0.0
            continue
        boots = rng.choice(sub, size=(n_bootstrap, sub.size), replace=True)
        means = boots.mean(axis=1)
        lo_band[i] = float(np.percentile(means, lo_pct))
        hi_band[i] = float(np.percentile(means, hi_pct))

    ax.fill_between(xs, lo_band, hi_band, color="#cc4125", alpha=0.18,
                    label=f"{int(confidence * 100)}% bootstrap CI")
    ax.plot(xs, roll, color="#cc4125", linewidth=2,
            label=f"rolling-{window} mean")
    ax.scatter(xs, values, s=18, color="#4a86e8", alpha=0.7,
               label=data.stat_type)
    if data.market_consensus_line is not None:
        ax.axhline(
            data.market_consensus_line, color="#666", linestyle="--",
            linewidth=1.0,
            label=f"book mean {data.market_consensus_line:.1f}",
        )

    dates = pd.to_datetime(data.games["game_date"]).dt.strftime("%m/%d")
    tick_step = max(1, n // 10)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(dates.iloc[::tick_step], rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel(data.stat_type)
    ax.set_title(
        f"{data.player_name} - rolling mean + {int(confidence * 100)}% CI"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    return fig


def build_trend_form_figure(
    data: PlayerChartData,
    short_window: int = 5,
    figsize: tuple[float, float] = (7.0, 2.4),
) -> Figure:
    """KPI panel: last-N vs season avg, with delta-arrows for "form" snapshot.

    Renders four mini-cards: ``last-N mean``, ``season mean``, ``delta``,
    ``last-game value``. Delta arrows (▲ / ▼) flag whether the player is
    trending hot or cold relative to their season baseline.
    """
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    if data.values.size == 0:
        ax.text(0.5, 0.5, "No game data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        return fig

    values = data.values
    short_n = max(1, min(int(short_window), values.size))
    last_n = values[-short_n:]
    last_n_mean = float(np.mean(last_n))
    season_mean = float(np.mean(values))
    delta = last_n_mean - season_mean
    last_game = float(values[-1])

    # Direction arrow + colour-coded based on delta sign.
    if delta > 0.5:
        arrow, colour = "▲", "#0a8f0a"
    elif delta < -0.5:
        arrow, colour = "▼", "#c0392b"
    else:
        arrow, colour = "•", "#555"

    cards = [
        (f"last-{short_n} mean", f"{last_n_mean:.1f}",   "#1f3a93"),
        ("season mean",          f"{season_mean:.1f}",   "#555"),
        ("delta",                f"{arrow} {delta:+.1f}", colour),
        ("last game",            f"{last_game:.1f}",     "#1f3a93"),
    ]
    for i, (label, value, value_colour) in enumerate(cards):
        x_center = (i + 0.5) / len(cards)
        ax.text(x_center, 0.7, value, transform=ax.transAxes,
                ha="center", va="center", fontsize=22, fontweight="bold",
                color=value_colour)
        ax.text(x_center, 0.25, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#666")
    ax.set_title(
        f"{data.player_name} - {data.stat_type} trend",
        fontsize=11, loc="left",
    )
    return fig


def build_defense_scatter_figure(
    data: PlayerChartData,
    db_path: str,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> Figure:
    """Player stat vs. opponent defensive rating scatter.

    For each game in ``data.games`` we look up the opponent team from the
    ``matchup`` field and join against ``team_defense.def_rating``. The
    resulting scatter shows whether the player's stat correlates with
    opponent strength — a downward slope on points-vs-def_rating is the
    expected market intuition (tough defense → lower points).
    """
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    if data.values.size == 0 or data.games.empty:
        ax.text(0.5, 0.5, "No game data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    games = data.games.copy()
    # Parse opponent abbreviation from matchup like "LAL vs. DEN" / "LAL @ DEN".
    def _extract_opp(matchup):
        if not isinstance(matchup, str):
            return None
        for sep in (" vs. ", " vs ", " @ "):
            if sep in matchup:
                return matchup.split(sep, 1)[1].strip().upper()
        return None
    games["opp_abbrev"] = games["matchup"].apply(_extract_opp) \
        if "matchup" in games.columns else None

    if games["opp_abbrev"].isna().all():
        ax.text(0.5, 0.5, "No opponent info available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    # Pull def_rating per opponent (latest season per team in team_defense).
    with DatabaseManager(db_path=db_path) as db:
        def_df = pd.read_sql_query(
            """
            SELECT team_abbrev, def_rating
            FROM team_defense
            WHERE def_rating IS NOT NULL
            """,
            db.conn,
        )
    if def_df.empty:
        ax.text(0.5, 0.5,
                "No team_defense rows. Run the daily ETL to populate.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666")
        ax.set_axis_off()
        return fig

    games = games.merge(
        def_df, left_on="opp_abbrev", right_on="team_abbrev", how="left",
    )
    games["stat_value"] = data.values
    plot_df = games.dropna(subset=["def_rating", "stat_value"])
    if plot_df.empty:
        ax.text(0.5, 0.5, "Opponent defense not joinable for these games",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666")
        ax.set_axis_off()
        return fig

    ax.scatter(plot_df["def_rating"], plot_df["stat_value"],
               s=36, alpha=0.7, color="#4a86e8", edgecolor="white")
    # Light linear-fit overlay to surface the trend.
    if len(plot_df) >= 3:
        coeffs = np.polyfit(plot_df["def_rating"], plot_df["stat_value"], 1)
        x_line = np.linspace(plot_df["def_rating"].min(),
                             plot_df["def_rating"].max(), 50)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, "--", color="#c0392b", alpha=0.6,
                label=f"slope = {coeffs[0]:+.3f}")
        ax.legend(loc="best", fontsize=9)

    ax.set_xlabel("opponent defensive rating (lower = tougher)")
    ax.set_ylabel(data.stat_type)
    ax.set_title(
        f"{data.player_name} - {data.stat_type} vs. opp def_rating "
        f"({len(plot_df)} games)"
    )
    ax.grid(linestyle=":", alpha=0.4)
    return fig


def build_splits_figure(
    data: PlayerChartData,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> Figure:
    """Two side-by-side bars: home/away mean and rest-day-bucket mean."""
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")
    if data.values.size == 0 or data.games.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No game data.", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.set_axis_off()
        return fig

    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    splits = compute_home_away_split(data)
    labels, means, counts = [], [], []
    for key in ("home", "away"):
        if splits.get(key) is not None:
            labels.append(key)
            means.append(splits[key]["mean"])
            counts.append(splits[key]["n"])
    if labels:
        bars = ax_left.bar(labels, means, color=["#4a86e8", "#cc4125"], alpha=0.85)
        for bar, n in zip(bars, counts):
            ax_left.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(),
                         f"  n={n}", ha="center", va="bottom", fontsize=8)
    else:
        ax_left.text(0.5, 0.5, "No home/away tag", ha="center", va="center",
                     transform=ax_left.transAxes, color="#666")
    ax_left.set_title("home vs away mean")
    ax_left.set_ylabel(data.stat_type)
    ax_left.grid(axis="y", linestyle=":", alpha=0.4)

    rest = compute_rest_days_split(data)
    if rest:
        ax_right.bar(list(rest.keys()), [v["mean"] for v in rest.values()],
                     color="#2ca02c", alpha=0.85)
        for i, (label, info) in enumerate(rest.items()):
            ax_right.text(i, info["mean"], f"  n={info['n']}", ha="center",
                          va="bottom", fontsize=8)
    else:
        ax_right.text(0.5, 0.5, "Not enough date history", ha="center",
                      va="center", transform=ax_right.transAxes, color="#666")
    ax_right.set_title("by rest days")
    ax_right.set_ylabel(data.stat_type)
    ax_right.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(f"{data.player_name} splits ({data.stat_type})", fontsize=11)
    return fig


# ----- Splits computations -------------------------------------------------

def compute_home_away_split(data: PlayerChartData) -> dict:
    """Return {'home': {mean,n}, 'away': {mean,n}} from games + values."""
    out: dict = {"home": None, "away": None}
    games = data.games
    if games.empty or "home_away" not in games.columns:
        return out
    series = pd.Series(data.values, index=games.index)
    for key in ("home", "away"):
        mask = games["home_away"].astype(str).str.lower() == key
        if mask.any():
            out[key] = {
                "mean": float(series[mask].mean()),
                "n": int(mask.sum()),
            }
    return out


def compute_rest_days_split(data: PlayerChartData) -> dict:
    """Bucket games by rest days since prior game and return per-bucket stats."""
    games = data.games
    if games.empty or "game_date" not in games.columns or len(games) < 2:
        return {}
    dates = pd.to_datetime(games["game_date"], errors="coerce")
    diffs = dates.diff().dt.days
    buckets = {}
    for bucket_label, mask in (
        ("0-1", diffs <= 1),
        ("2", diffs == 2),
        ("3+", diffs >= 3),
    ):
        m = mask.fillna(False)
        if m.any():
            vals = pd.Series(data.values, index=games.index)[m]
            buckets[bucket_label] = {
                "mean": float(vals.mean()),
                "n": int(m.sum()),
            }
    return buckets


# ----- Probability + EV helpers --------------------------------------------

def _american_odds_to_decimal(odds) -> Optional[float]:
    if odds is None or pd.isna(odds):
        return None
    try:
        o = int(odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    if o > 0:
        return 1.0 + o / 100.0
    return 1.0 + 100.0 / abs(o)


def _american_odds_breakeven(odds) -> Optional[float]:
    dec = _american_odds_to_decimal(odds)
    if dec is None or dec <= 1.0:
        return None
    return 1.0 / dec


def fitted_prob_over(data: PlayerChartData, line: float) -> Optional[float]:
    """Probability of going OVER `line` under fitted normal."""
    if data.values.size == 0:
        return None
    if data.sigma <= 0:
        return 1.0 if data.mu > line else (0.5 if data.mu == line else 0.0)
    return float(1.0 - norm.cdf(line, loc=data.mu, scale=data.sigma))


def expected_value(prob: float, american_odds) -> Optional[float]:
    """Return EV per 1 unit staked at the given odds and win probability."""
    dec = _american_odds_to_decimal(american_odds)
    if dec is None or prob is None:
        return None
    return float(prob * (dec - 1.0) - (1.0 - prob))


def evaluate_custom_line(
    data: PlayerChartData,
    line: float,
    american_odds: Optional[int] = None,
) -> dict:
    """Evaluate a hypothetical line: fitted P(over), historical hit-rate, EV."""
    p_over = fitted_prob_over(data, float(line))
    n = int(data.values.size)
    hits = int(np.sum(data.values > float(line))) if n else 0
    pushes = int(np.sum(data.values == float(line))) if n else 0
    hit_rate = (hits / n) if n else 0.0
    ev_over = expected_value(p_over, american_odds) if p_over is not None else None
    ev_under = (
        expected_value(1.0 - p_over, american_odds) if p_over is not None else None
    )
    return {
        "line": float(line),
        "p_over": p_over,
        "hits": hits,
        "pushes": pushes,
        "n": n,
        "historical_over_rate": hit_rate,
        "ev_over_per_unit": ev_over,
        "ev_under_per_unit": ev_under,
    }


def book_lines_staleness_summary(data: PlayerChartData) -> Optional[dict]:
    """Return age-of-most-recent-line stats for ``data.book_lines``.

    Used by both the chart-summary text and the Streamlit UI to surface
    "lines scraped 2.4 h ago" so users know whether the consensus mean
    they're looking at is fresh or pre-slate.

    Returns ``None`` when there are no book lines or no parseable timestamps.
    Otherwise returns ``{"hours_min", "hours_max", "hours_median",
    "latest_iso", "n_books"}``.
    """
    if data.book_lines is None or data.book_lines.empty:
        return None
    if "game_date" not in data.book_lines.columns:
        return None
    ts = pd.to_datetime(
        data.book_lines["game_date"], errors="coerce", utc=True,
    ).dropna()
    if ts.empty:
        return None
    now_utc = pd.Timestamp.now(tz="UTC")
    hours = (now_utc - ts).dt.total_seconds() / 3600.0
    return {
        "hours_min": float(hours.min()),
        "hours_max": float(hours.max()),
        "hours_median": float(hours.median()),
        "latest_iso": str(ts.max().isoformat()),
        "n_books": int(len(ts)),
    }


def _format_staleness(hours: float) -> str:
    """Human-friendly format for staleness: '2.4 h ago' / '12 m ago' / '3 d ago'."""
    if hours < 1.0:
        return f"{int(round(hours * 60))} m ago"
    if hours < 48.0:
        return f"{hours:.1f} h ago"
    return f"{hours / 24.0:.1f} d ago"


def book_lines_summary_text(data: PlayerChartData) -> str:
    """Render a small text table of per-book lines + over-prob under fitted normal."""
    lines: list[str] = []
    lines.append(
        f"{data.player_name}  |  {data.stat_type}  |  n={data.values.size}"
    )
    lines.append(
        f"mu={data.mu:.2f}  sigma={data.sigma:.2f}  "
        f"min={float(data.values.min()) if data.values.size else 0:.1f}  "
        f"max={float(data.values.max()) if data.values.size else 0:.1f}"
    )
    if data.book_lines is None or data.book_lines.empty:
        lines.append("No book lines available.")
        return "\n".join(lines)

    consensus = compute_market_consensus(data)
    if consensus["mean"] is not None:
        lines.append(
            f"market consensus across {consensus['n_books']} book"
            f"{'s' if consensus['n_books'] != 1 else ''}: "
            f"mean={consensus['mean']:.3f}  "
            f"sigma={consensus['stdev']:.3f}  "
            f"spread={consensus['spread']:.3f}"
        )
    # Surface staleness — useful for spotting consensus computed from
    # snapshots taken before today's slate moved.
    staleness = book_lines_staleness_summary(data)
    if staleness is not None:
        lines.append(
            f"lines scraped {_format_staleness(staleness['hours_median'])} "
            f"(median across {staleness['n_books']} book"
            f"{'s' if staleness['n_books'] != 1 else ''}; "
            f"newest {_format_staleness(staleness['hours_min'])}, "
            f"oldest {_format_staleness(staleness['hours_max'])})"
        )
    pct_lookup = {p["book"].lower(): p for p in consensus["per_book"]}

    n = int(data.values.size)
    header = (f"{'book':<14}{'line':>6}{'%fromMu':>10}"
              f"{'odds_o':>8}{'odds_u':>8}"
              f"{'P(over)':>10}{'EV_o':>9}{'EV_u':>9}{'hit%':>8}")
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in data.book_lines.iterrows():
        book_full = str(row.get("book") or "")
        book = book_full[:14]
        try:
            lv = float(row.get("line_value"))
        except (TypeError, ValueError):
            continue
        oo = row.get("over_odds")
        uo = row.get("under_odds")
        oo_text = "" if oo is None or pd.isna(oo) else str(int(oo))
        uo_text = "" if uo is None or pd.isna(uo) else str(int(uo))
        p_over = fitted_prob_over(data, lv)
        ev_o = expected_value(p_over, oo) if p_over is not None else None
        ev_u = expected_value(1.0 - p_over, uo) if p_over is not None else None
        hit = (int(np.sum(data.values > lv)) / n) if n else 0.0
        p_over_text = f"{p_over:.1%}" if p_over is not None else ""
        ev_o_text = f"{ev_o:+.2f}" if ev_o is not None else ""
        ev_u_text = f"{ev_u:+.2f}" if ev_u is not None else ""
        entry = pct_lookup.get(book_full.lower())
        pct_text = entry["pct_from_mean_str"] if entry else ""
        lines.append(
            f"{book:<14}{lv:>6.1f}{pct_text:>10}{oo_text:>8}{uo_text:>8}"
            f"{p_over_text:>10}{ev_o_text:>9}{ev_u_text:>9}{hit:>8.0%}"
        )

    if consensus["books_missing"]:
        lines.append("")
        lines.append(
            "books MISSING (expected but no current line): "
            + ", ".join(consensus["books_missing"])
        )

    splits = compute_home_away_split(data)
    rest = compute_rest_days_split(data)
    if any(splits.values()) or rest:
        lines.append("")
        lines.append("splits:")
    for key in ("home", "away"):
        s = splits.get(key)
        if s:
            lines.append(f"  {key:<6} mean={s['mean']:.2f}  n={s['n']}")
    for label, info in rest.items():
        lines.append(f"  rest {label:<4} mean={info['mean']:.2f}  n={info['n']}")

    if data.notes:
        lines.append("")
        for note in data.notes:
            lines.append(f"note: {note}")
    return "\n".join(lines)


def list_players_with_data(
    db_path: str, team: Optional[str] = None,
) -> pd.DataFrame:
    """List players that have game logs (or betting lines), filtered by team.

    Each player's team is derived from their **most recent ``game_logs``
    row's ``matchup``** — that field's first whitespace-delimited token is
    the team they played for in that game, and ``game_logs`` has full
    coverage for every active player (~530 players, all 30 teams).  We
    used to read ``players.team`` here, but that column is mostly NULL in
    the snapshot DB, which collapsed every team filter to a single roster.

    Player names come from ``nba_active_players_ref`` (530 rows, kept in
    sync by the scraper) with a fallback to the sparse ``players`` table
    so legacy rows still resolve.

    The returned frame also includes an `n_books` column counting how many
    distinct sportsbooks currently have a line for the player (across the
    union of `betting_lines` and `web_prop_cards`).  Callers can use this
    to surface players who are actively in this slate's market - e.g. the
    Streamlit sidebar offers a "Only players with current book lines"
    toggle that filters to `n_books > 0`.
    """
    team_clean = (team or "").strip()
    apply_team_filter = bool(team_clean) and team_clean.lower() not in {
        "all", "any",
    }
    params: tuple = ()
    where_clause = ""
    if apply_team_filter:
        where_clause = "AND upper(pt.team) = upper(?)"
        params = (team_clean,)

    query = f"""
        WITH player_team_history AS (
            SELECT
                g.player_id,
                upper(trim(substr(g.matchup, 1, instr(g.matchup, ' ') - 1))) AS team,
                ROW_NUMBER() OVER (
                    PARTITION BY g.player_id
                    ORDER BY g.game_date DESC, g.game_log_id DESC
                ) AS rn
            FROM game_logs g
            WHERE g.matchup IS NOT NULL AND instr(g.matchup, ' ') > 0
        ),
        player_team AS (
            SELECT player_id, team
            FROM player_team_history
            WHERE rn = 1
        )
        SELECT
            pt.player_id,
            COALESCE(r.player_name, p.name, 'Player ' || pt.player_id) AS player_name,
            pt.team
        FROM player_team pt
        LEFT JOIN nba_active_players_ref r ON r.player_id = pt.player_id
        LEFT JOIN players p ON p.player_id = pt.player_id
        WHERE COALESCE(r.player_name, p.name) IS NOT NULL
          {where_clause}
        ORDER BY player_name ASC
    """

    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(query, db.conn, params=params)

    # Backfill: pull players that appear ONLY in betting_lines (no game logs
    # yet) so the dropdown still surfaces them when the team filter is off.
    if not apply_team_filter:
        with DatabaseManager(db_path=db_path) as db:
            extras = pd.read_sql_query(
                """
                SELECT DISTINCT
                    b.player_id,
                    COALESCE(r.player_name, p.name) AS player_name,
                    NULL AS team
                FROM betting_lines b
                LEFT JOIN nba_active_players_ref r ON r.player_id = b.player_id
                LEFT JOIN players p ON p.player_id = b.player_id
                WHERE COALESCE(r.player_name, p.name) IS NOT NULL
                  AND b.player_id NOT IN (SELECT player_id FROM game_logs)
                """,
                db.conn,
            )
        if not extras.empty:
            df = pd.concat([df, extras], ignore_index=True)
            df = df.drop_duplicates(subset=["player_id"]).sort_values("player_name")

    # Attach the cross-book count: how many distinct sportsbooks currently
    # have a line for each player. `betting_lines.player_id` is the canonical
    # join key; `web_prop_cards.player_name` is name-only, so we approximate
    # by name-matching against the player_id->name map we already have.
    if not df.empty:
        with DatabaseManager(db_path=db_path) as db:
            bl_books = pd.read_sql_query(
                """
                SELECT player_id, COUNT(DISTINCT lower(book)) AS n_books
                FROM betting_lines
                WHERE book IS NOT NULL
                GROUP BY player_id
                """,
                db.conn,
            )
            wp_books = pd.read_sql_query(
                """
                SELECT lower(player_name) AS lower_name,
                       COUNT(DISTINCT lower(book)) AS n_books
                FROM web_prop_cards
                WHERE book IS NOT NULL AND player_name IS NOT NULL
                GROUP BY lower(player_name)
                """,
                db.conn,
            )
        df = df.merge(bl_books, on="player_id", how="left")
        df["n_books"] = df["n_books"].fillna(0).astype(int)
        if not wp_books.empty:
            df["__lower_name__"] = df["player_name"].str.lower()
            df = df.merge(
                wp_books, left_on="__lower_name__", right_on="lower_name",
                how="left", suffixes=("", "_wp"),
            )
            df["n_books"] = (
                df["n_books"].fillna(0).astype(int)
                + df["n_books_wp"].fillna(0).astype(int)
            )
            df = df.drop(columns=["__lower_name__", "lower_name", "n_books_wp"],
                         errors="ignore")
    else:
        df["n_books"] = 0

    return df.reset_index(drop=True)


def _team_value_sql_expr(canonical_stat: str) -> Optional[str]:
    """Return a SUM(..) SQL expression that aggregates the per-game team value.

    Returns None for stats that don't have a clean team-level aggregation
    (e.g. minutes, percentages).
    """
    column_map = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "three_pointers_made": "fg3m",
        "field_goals_made": "fgm",
    }
    if canonical_stat in column_map:
        col = column_map[canonical_stat]
        return f"SUM(COALESCE({col}, 0))"
    if canonical_stat == "pra":
        return ("SUM(COALESCE(points, 0) + COALESCE(rebounds, 0) "
                "+ COALESCE(assists, 0))")
    if canonical_stat == "ra":
        return "SUM(COALESCE(rebounds, 0) + COALESCE(assists, 0))"
    return None


def _fetch_latest_team_book_lines(
    db: DatabaseManager,
    team_code: str,
    canonical_stat: str,
    web_lookback_hours: float = 48.0,
) -> pd.DataFrame:
    """Per-book implied team-total rows for the team's most recent game.

    Books expose game total + spread on the lobby; the team's *implied total*
    is ``(game_total - team_spread) / 2`` (sign of spread naturally handles
    favorite vs. underdog).  Returned shape mirrors the player path so the
    same chart code re-uses it: book, line_value, over_odds, under_odds, game_date.

    Currently only the ``points`` stat has a meaningful book reference.  For
    other team stats (assists, rebounds, ...) the books don't surface a line
    at the lobby level, so we return an empty frame.
    """
    if canonical_stat != "points":
        return pd.DataFrame()
    canonical = team_code_to_canonical(team_code)
    if canonical is None:
        return pd.DataFrame()

    # Pull every (book, market, side) for the team's most recent game in the
    # web_team_lines window.  We compute implied_total per-book downstream.
    query = """
        WITH latest_game AS (
            SELECT away_team, home_team, MAX(observed_at_utc) AS observed_at_utc
            FROM web_team_lines
            WHERE (lower(away_team) = lower(?) OR lower(home_team) = lower(?))
              AND observed_at_utc >= datetime('now', ?)
            GROUP BY away_team, home_team
            ORDER BY observed_at_utc DESC LIMIT 1
        )
        SELECT t.book, t.market_type, t.side, t.team,
               t.line_value, t.observed_at_utc
        FROM web_team_lines t
        JOIN latest_game lg
          ON t.away_team = lg.away_team AND t.home_team = lg.home_team
        WHERE t.observed_at_utc >= datetime('now', ?)
        ORDER BY t.book ASC, t.observed_at_utc DESC
    """
    lookback = f"-{float(web_lookback_hours)} hours"
    rows = db.conn.execute(
        query, (canonical, canonical, lookback, lookback),
    ).fetchall()
    if not rows:
        return pd.DataFrame()

    # Per-book latest spread (for our team) and game total.
    by_book: dict[str, dict] = {}
    for book, market, side, team, line, obs_at in rows:
        slot = by_book.setdefault(
            book.lower(),
            {"book": book, "spread": None, "total": None, "date": obs_at},
        )
        if market == "spread" and team and team.lower() == canonical.lower():
            if slot["spread"] is None:
                slot["spread"] = float(line) if line is not None else None
        elif market == "total" and side == "over":
            if slot["total"] is None:
                slot["total"] = float(line) if line is not None else None

    out = []
    for slot in by_book.values():
        if slot["spread"] is None or slot["total"] is None:
            continue
        implied = (slot["total"] - slot["spread"]) / 2.0
        out.append(
            {
                "book": slot["book"],
                "line_value": float(implied),
                "over_odds": None,
                "under_odds": None,
                "game_date": slot["date"],
            }
        )
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("book").reset_index(drop=True)


def fetch_recent_games(
    db_path: str,
    n: int = 50,
    season: Optional[str] = None,
    season_type: Optional[str] = None,
    team_abbrev: Optional[str] = None,
) -> pd.DataFrame:
    """Frontend-facing wrapper for ``DatabaseManager.get_recent_games``.

    Returns one row per matchup with both teams' final scores and a
    ``winner`` column.  All filters are optional.
    """
    with DatabaseManager(db_path=db_path) as db:
        rows = db.get_recent_games(
            n=n, season=season, season_type=season_type, team_abbrev=team_abbrev,
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_player_recent_results(
    db_path: str,
    n: int = 200,
    player_id: Optional[int] = None,
    team_abbrev: Optional[str] = None,
    season: Optional[str] = None,
    stat: Optional[str] = None,
    min_value: Optional[float] = None,
) -> pd.DataFrame:
    """Frontend-facing wrapper for ``DatabaseManager.get_player_recent_results``.

    Useful for the "Player Stats Browse" surface to scan league-wide
    recent results, optionally filtered by player, team, season, or "show
    me players with at least N points last game".
    """
    with DatabaseManager(db_path=db_path) as db:
        rows = db.get_player_recent_results(
            n=n, player_id=player_id, team_abbrev=team_abbrev,
            season=season, stat=stat, min_value=min_value,
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def list_seasons(db_path: str) -> list[str]:
    """Distinct seasons available in the games table (newest first)."""
    with DatabaseManager(db_path=db_path) as db:
        rows = db.conn.execute(
            "SELECT DISTINCT season FROM games ORDER BY season DESC"
        ).fetchall()
    return [r[0] for r in rows if r[0]]


def list_team_codes(db_path: str) -> list[str]:
    """Return the 30 NBA team codes that appear in ``game_logs.matchup``.

    ``players.team`` is unreliable (audit shows ~93/94 NULL), so we parse
    the team from ``game_logs.matchup`` (first whitespace-delimited token).
    Filtered against the canonical NBA-team set in
    ``nba_model.web.input_validation.KNOWN_TEAM_CODES`` to drop foreign /
    exhibition codes (FIBA, EuroLeague, etc.) that ``playergamelogs``
    sometimes returns when an NBA player suited up in an international game.
    """
    # Local import to avoid cycle: input_validation imports nothing from
    # this module, but this module shouldn't pull validation at import time.
    from nba_model.web.input_validation import KNOWN_TEAM_CODES

    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT
                upper(trim(substr(matchup, 1, instr(matchup, ' ') - 1))) AS team
            FROM game_logs
            WHERE matchup IS NOT NULL AND instr(matchup, ' ') > 0
            ORDER BY team
            """,
            db.conn,
        )
    return [
        t for t in df["team"].dropna().astype(str).tolist()
        if t and t in KNOWN_TEAM_CODES
    ]


def fetch_team_chart_data(
    db_path: str,
    team: str,
    stat_type: str,
    n_games: int = 25,
) -> "PlayerChartData":
    """Aggregate game_logs into a team-level series for the figure builders.

    Sums each player-game row in the team's last N games to get one team
    value per game (e.g. team total points, team total 3PM). Returns the same
    `PlayerChartData` shape so every existing figure builder works unchanged.
    book_lines is always empty since team-totals odds aren't stored yet.
    """
    canonical = _canonical_stat_type(stat_type)
    notes: list[str] = []
    expr = _team_value_sql_expr(canonical)
    team_code = (team or "").strip().upper()

    if expr is None:
        return PlayerChartData(
            player_id=0,
            player_name=team_code,
            stat_type=canonical,
            games=pd.DataFrame(),
            values=np.array([], dtype=float),
            book_lines=pd.DataFrame(),
            market_consensus_line=None,
            notes=[f"Team-level aggregation not supported for '{canonical}'."],
        )

    # SECURITY: `expr` is interpolated into the SQL string below. It MUST come
    # from the controlled allowlist in `_team_value_sql_expr` and never from
    # user input. The assert below makes that requirement explicit so future
    # edits don't accidentally introduce a SQL-injection vector.
    _allowed_exprs = {
        "SUM(COALESCE(points, 0))",
        "SUM(COALESCE(assists, 0))",
        "SUM(COALESCE(rebounds, 0))",
        "SUM(COALESCE(fg3m, 0))",
        "SUM(COALESCE(fgm, 0))",
        "SUM(COALESCE(points, 0) + COALESCE(rebounds, 0) + COALESCE(assists, 0))",
        "SUM(COALESCE(rebounds, 0) + COALESCE(assists, 0))",
    }
    assert expr in _allowed_exprs, (
        f"Refusing to interpolate untrusted SQL expression: {expr!r}. "
        "Add it to _team_value_sql_expr's allowlist if legitimate."
    )

    with DatabaseManager(db_path=db_path) as db:
        query = f"""
            SELECT
                game_date,
                MAX(game_id)   AS game_id,
                MAX(matchup)   AS matchup,
                MAX(home_away) AS home_away,
                MAX(result)    AS result,
                {expr}         AS team_value,
                COUNT(*)       AS players_in_game
            FROM game_logs
            WHERE upper(trim(substr(matchup, 1,
                                    instr(matchup, ' ') - 1))) = ?
            GROUP BY game_date, game_id
            ORDER BY game_date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(
            query, db.conn, params=(team_code, max(1, int(n_games))),
        )

    if df.empty:
        notes.append(f"No game logs found for team '{team_code}'.")
        return PlayerChartData(
            player_id=0,
            player_name=team_code,
            stat_type=canonical,
            games=df,
            values=np.array([], dtype=float),
            book_lines=pd.DataFrame(),
            market_consensus_line=None,
            notes=notes,
        )

    # Plot oldest -> newest like the player path.
    df = df.sort_values("game_date", ascending=True).reset_index(drop=True)
    values = df["team_value"].fillna(0).astype(float).to_numpy()
    notes.append(
        f"Team values are sums across {int(df['players_in_game'].mean()):.0f} "
        "players per game on average."
    )

    # Pull cross-book consensus for the team's most recent game.  Currently
    # only the ``points`` stat has a meaningful book reference (derived from
    # game total + spread); other team stats return an empty frame.
    with DatabaseManager(db_path=db_path) as db:
        team_book_lines = _fetch_latest_team_book_lines(db, team_code, canonical)
    market_line = _market_consensus_line(team_book_lines)
    if not team_book_lines.empty and market_line is not None:
        notes.append(
            f"Book mean derived from {len(team_book_lines)} book(s): "
            f"implied team total = (game_total - team_spread) / 2."
        )

    return PlayerChartData(
        player_id=0,
        player_name=team_code,
        stat_type=canonical,
        games=df,
        values=values,
        book_lines=team_book_lines,
        market_consensus_line=market_line,
        notes=notes,
    )


def list_teams(db_path: str) -> list[str]:
    """Distinct team codes that have at least one player game log.

    The ``players.team`` column is mostly NULL (audit shows ~93/94 rows
    unset), so reading it directly used to leave the sidebar dropdown with
    a single team.  We parse the same value from ``game_logs.matchup``
    (the first whitespace-delimited token), which has full coverage for
    all 30 teams.
    """
    return list_team_codes(db_path)
