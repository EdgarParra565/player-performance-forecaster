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

DISTRIBUTION_CHOICES = ("normal", "poisson", "negative_binomial")

STAT_COLUMN_BY_TYPE = {
    "points": "points",
    "assists": "assists",
    "rebounds": "rebounds",
    "minutes": "minutes",
}


@dataclass
class PlayerChartData:
    player_id: int
    player_name: str
    stat_type: str
    games: pd.DataFrame
    values: np.ndarray
    book_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_median_line: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def mu(self) -> float:
        return float(np.mean(self.values)) if self.values.size else 0.0

    @property
    def sigma(self) -> float:
        if self.values.size < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))


def _series_for_stat(games: pd.DataFrame, stat_type: str) -> np.ndarray:
    """Return numeric stat series, computing PRA on demand."""
    stat_type = (stat_type or "").lower()
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
    with DatabaseManager(db_path=db_path) as db:
        games = db.get_player_games(player_id, n_games=max(1, int(n_games)))
        # ASC for plotting (oldest -> newest); db returns DESC.
        games = games.sort_values("game_date", ascending=True).reset_index(drop=True)
        values = _series_for_stat(games, stat_type)

        book_lines = _fetch_latest_book_lines(db, player_id, stat_type)
        market_line = _market_median_line(book_lines)

    if games.empty:
        notes.append("No game logs found for this player.")
    if book_lines.empty:
        notes.append(f"No book lines found for {stat_type}.")

    return PlayerChartData(
        player_id=int(player_id),
        player_name=str(player_name),
        stat_type=str(stat_type),
        games=games,
        values=values,
        book_lines=book_lines,
        market_median_line=market_line,
        notes=notes,
    )


def _fetch_latest_book_lines(
    db: DatabaseManager, player_id: int, stat_type: str,
) -> pd.DataFrame:
    """Return the most recent betting_lines row per book for the player+stat."""
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
    # Also pick up web_prop_cards latest if betting_lines is sparse for this stat.
    web_query = """
        WITH ranked AS (
            SELECT book, line_value, side, observed_at_utc,
                   ROW_NUMBER() OVER (
                       PARTITION BY book, side
                       ORDER BY observed_at_utc DESC
                   ) AS rn
            FROM web_prop_cards
            WHERE lower(player_name) = lower((
                    SELECT name FROM players WHERE player_id = ?))
              AND lower(stat_type) = lower(?)
        )
        SELECT book, line_value, side, observed_at_utc
        FROM ranked WHERE rn = 1
    """
    web_df = pd.read_sql_query(
        web_query, db.conn, params=(int(player_id), str(stat_type)),
    )
    if not web_df.empty:
        # Collapse over/under sides into a single row per book.
        agg = (
            web_df.groupby("book", as_index=False)
            .agg(line_value=("line_value", "median"),
                 game_date=("observed_at_utc", "max"))
        )
        agg["over_odds"] = None
        agg["under_odds"] = None
        agg = agg[["book", "line_value", "over_odds", "under_odds", "game_date"]]
        # Prefer betting_lines book entries; only add web books not already present.
        existing_books = set(df["book"].str.lower()) if not df.empty else set()
        agg = agg[~agg["book"].str.lower().isin(existing_books)]
        df = pd.concat([df, agg], ignore_index=True) if not agg.empty else df
    return df.sort_values("book").reset_index(drop=True)


def _market_median_line(book_lines: pd.DataFrame) -> Optional[float]:
    if book_lines.empty:
        return None
    vals = pd.to_numeric(book_lines["line_value"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(np.median(vals))


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

    if data.market_median_line is not None:
        ax.axhline(
            data.market_median_line, color="#666", linestyle="--",
            linewidth=1.2, label=f"book median {data.market_median_line:.1f}",
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

    if data.book_lines is not None and not data.book_lines.empty:
        for idx, row in data.book_lines.reset_index(drop=True).iterrows():
            book = str(row.get("book") or "").strip()
            try:
                lv = float(row.get("line_value"))
            except (TypeError, ValueError):
                continue
            color = _book_color(book, idx)
            ax.axvline(lv, color=color, linestyle="--", linewidth=1.6,
                       alpha=0.9, label=f"{book} {lv:.1f}")

    ax.set_xlabel(data.stat_type)
    ax.set_ylabel("density")
    ax.set_title(
        f"{data.player_name} - {data.stat_type} distribution "
        f"(n={data.values.size})"
    )
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

    n = int(data.values.size)
    header = (f"{'book':<14}{'line':>6}{'odds_o':>8}{'odds_u':>8}"
              f"{'P(over)':>10}{'EV_o':>9}{'EV_u':>9}{'hit%':>8}")
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in data.book_lines.iterrows():
        book = str(row.get("book") or "")[:14]
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
        lines.append(
            f"{book:<14}{lv:>6.1f}{oo_text:>8}{uo_text:>8}"
            f"{p_over_text:>10}{ev_o_text:>9}{ev_u_text:>9}{hit:>8.0%}"
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
    """List players that have any game logs OR betting lines, filtered by team."""
    with DatabaseManager(db_path=db_path) as db:
        if team and team.strip().lower() not in {"", "all", "any"}:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT p.player_id, p.name AS player_name, p.team
                FROM players p
                WHERE upper(p.team) = upper(?)
                ORDER BY p.name ASC
                """,
                db.conn,
                params=(team.strip(),),
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT p.player_id, p.name AS player_name, p.team
                FROM players p
                WHERE EXISTS (SELECT 1 FROM game_logs g WHERE g.player_id = p.player_id)
                   OR EXISTS (SELECT 1 FROM betting_lines b WHERE b.player_id = p.player_id)
                ORDER BY p.name ASC
                """,
                db.conn,
            )
    return df


def list_teams(db_path: str) -> list[str]:
    """Distinct non-null team codes present in the players table."""
    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(
            "SELECT DISTINCT team FROM players "
            "WHERE team IS NOT NULL AND TRIM(team) <> '' ORDER BY team",
            db.conn,
        )
    return [str(t) for t in df["team"].tolist()]
