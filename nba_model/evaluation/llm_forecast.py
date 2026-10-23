"""LLM calibration benchmark — forecast collection (WS11, measurement only).

Asks an LLM for P(over) on NBA player props and stores every answer in
``llm_forecasts``; ``calibration_report --source llm`` then scores the answers
next to the model, the market and a constant 0.5 on the exact same props. The
output is never used to pick or size bets.

Flow: select props → versioned prompt → cache lookup by prompt hash → cost
estimate → (``--dry-run`` stops here) → provider call for uncached prompts
only → strict parse (re-ask once) → insert + settle.

Rules enforced here:
  * The default ``blind`` arm's prompt carries no outcome and no market price:
    player, team, opponent, game date, stat, line, and the last-N game log for
    that stat (games strictly BEFORE the game date). The ``market`` arm adds a
    de-vigged price and only runs on props with a real two-sided price.
  * Historical props are labeled ``retro=1`` and need ``--retro``. Forward
    runs (forecast before tip-off) are the design default but need in-season
    props, so they are not built yet.
  * API keys come from the environment only. They are never logged, stored,
    cached or echoed: provider errors keep the HTTP status / exception class,
    never a response body.
  * ``--dry-run`` prints the cost estimate and a sample prompt, then stops: no
    API call, no DB write, no cache or artifact file. The DB is opened
    read-only for selection.

Prop universe (retro): settled ``predictions`` rows, one per (player, date,
stat, line). Those rows are backtest sweeps, so the defaults keep only
book-style half-point lines within ``--max-line-z`` SDs of the player's recent
mean, and drop lines that exist only as a backtest model mean
(``line_source='model'``: synthetic, and the backtest's points mean reads the
target game's minutes). The model baseline is the row run with the production
config (``get_default_distribution(stat)``, rolling window 10) when one exists.
A market price counts only if scraped before ``MARKET_PRICE_CUTOFF_UTC`` on
game day.

CLI (50-prop historical dry run):
    python -m nba_model.evaluation.llm_forecast --retro \\
        --provider anthropic --model claude-opus-5-5 --limit 50 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence
from urllib.parse import urlsplit

import requests

from nba_model.data.database.db_manager import (
    _GAMELOG_STAT_EXPR,
    DatabaseManager,
    DatabaseNotReadyError,
)
from nba_model.logging_utils import configure_logging, get_logger
from nba_model.model.odds import _implied_or_none, american_to_implied_prob

logger = get_logger(__name__)

PROMPT_VERSION = "v1"
ARMS = DatabaseManager.LLM_ARMS                     # ("blind", "market")
PROVIDERS = ("anthropic", "openai")

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_CACHE_DIR = Path("data/cache/llm_forecasts")
ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")

DEFAULT_LIMIT = 50
DEFAULT_SEED = "ws11"
DEFAULT_LAST_N = 10
DEFAULT_MIN_HISTORY = 5
DEFAULT_MAX_LINE_Z = 3.0
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_REASKS = 1
DEFAULT_MAX_COST_USD = 5.0
MAX_CONSECUTIVE_FAILURES = 5

# No real price → the -110 assumption every DFS / missing-odds path uses.
ASSUMED_PRICE_PROB = american_to_implied_prob(-110)   # 0.5238

# A market price counts only if scraped before this UTC time on game day.
# No tip times are stored and betting_lines is change-only, so a later row can
# be an in-play or post-game price (live: a 2026-07-22 backfill of May games).
# The earliest NBA tips are noon ET = 16:00 UTC under daylight time.
MARKET_PRICE_CUTOFF_UTC = "16:00:00"

_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

# Token estimate for the cost preview: chars / 3 over-counts on purpose.
CHARS_PER_TOKEN_ESTIMATE = 3.0

# USD per million tokens (input, output): Anthropic first-party list prices
# (claude-api reference, cached 2026-10-06). Any other model needs
# --price-in-per-mtok / --price-out-per-mtok before a real run.
PRICING_USD_PER_MTOK = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}

STAT_LABELS = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "pra": "points + rebounds + assists (PRA)",
    "ra": "rebounds + assists",
    "three_pointers_made": "3-pointers made",
    "field_goals_made": "field goals made",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "minutes": "minutes played",
}

SYSTEM_PROMPT_V1 = (
    "You forecast NBA player prop outcomes as calibrated probabilities. Use "
    "only the information given. Reply with one decimal number between 0 and "
    "1 and nothing else."
)
REASK_SUFFIX_V1 = (
    "\n\nYour previous reply could not be read as a probability. Reply with "
    "only one decimal number between 0 and 1."
)


# ---------------------------------------------------------------------------
# Props + prompts
# ---------------------------------------------------------------------------


class GameLine(NamedTuple):
    """One prior game in the prompt's game log."""
    game_date: str
    opponent: str          # "vs BOS" / "@ BOS"
    minutes: Optional[float]
    value: Optional[float]


@dataclass(frozen=True)
class Prop:
    player_id: int
    player_name: str
    game_date: str                       # YYYY-MM-DD
    stat_type: str
    line_value: float
    team: Optional[str] = None
    opponent: Optional[str] = None
    home: Optional[bool] = None
    history: tuple = ()                  # GameLine, most recent first
    prediction_id: Optional[int] = None  # model baseline row
    settled: bool = False
    market_prob_over: Optional[float] = None   # de-vigged REAL price only
    market_books: int = 0

    @property
    def key(self) -> tuple:
        return (self.player_id, self.game_date, self.stat_type,
                round(float(self.line_value), 4))


@dataclass(frozen=True)
class PromptSpec:
    system: str
    user: str
    prompt_version: str
    arm: str


def _fmt_num(value) -> str:
    if value is None:
        return "-"
    v = float(value)
    return str(int(v)) if v.is_integer() else f"{v:.1f}"


def _opponent_label(matchup: Optional[str]) -> str:
    team, opp, home = parse_matchup(matchup)
    if opp is None:
        return "?"
    return f"{'vs' if home else '@'} {opp}"


def parse_matchup(matchup: Optional[str], home_away: Optional[str] = None):
    """``'LAL vs. GSW'`` → ``('LAL', 'GSW', True)``; ``'LAL @ GSW'`` → away.
    Schedule facts only. ``(None, None, None)`` when unparseable."""
    tokens = str(matchup or "").replace("vs.", "vs").split()
    if len(tokens) != 3:
        return None, None, None
    team, sep, opp = tokens
    if sep.lower() == "vs":
        home = True
    elif sep == "@":
        home = False
    else:
        home = (str(home_away).lower() == "home") if home_away else None
    return team, opp, home


def build_prompt(prop: Prop, *, arm: str = "blind",
                 prompt_version: str = PROMPT_VERSION) -> PromptSpec:
    """Versioned prompt for one prop. The ``blind`` arm holds only pre-game
    facts — never the outcome, never a price; ``market`` adds the de-vigged
    real price (refused when the prop has none)."""
    if prompt_version != PROMPT_VERSION:
        raise ValueError(f"unknown prompt_version {prompt_version!r}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (use one of {ARMS})")
    if arm == "market" and prop.market_prob_over is None:
        raise ValueError("market arm needs a real two-sided price for the prop")

    label = STAT_LABELS.get(prop.stat_type, prop.stat_type.replace("_", " "))
    line = f"{float(prop.line_value):g}"
    if prop.team and prop.opponent and prop.home is not None:
        game = f"{prop.team} {'vs' if prop.home else '@'} {prop.opponent}"
    else:
        game = "matchup unknown"
    lines = [
        f"Will {prop.player_name} record more than {line} {label} in this game?",
        "",
        f"Player: {prop.player_name}" + (f" ({prop.team})" if prop.team else ""),
        f"Game: {game}, {prop.game_date}",
        f"Stat: {label}",
        f"Line: {line} (OVER means strictly more than the line)",
    ]
    if arm == "market":
        lines.append(
            f"Market: sportsbook prices imply P(over) = {prop.market_prob_over:.3f} "
            "after removing the vig.")
    lines += [
        "",
        f"{prop.player_name}'s last {len(prop.history)} games before "
        f"{prop.game_date}, most recent first:",
        f"date | opponent | minutes | {label}",
    ]
    lines += [f"{g.game_date} | {g.opponent} | {_fmt_num(g.minutes)} | {_fmt_num(g.value)}"
              for g in prop.history]
    lines += [
        "",
        "Reply with only the probability that the result is OVER the line, "
        "as a decimal number between 0 and 1.",
    ]
    return PromptSpec(system=SYSTEM_PROMPT_V1, user="\n".join(lines),
                      prompt_version=prompt_version, arm=arm)


def prompt_hash(provider_name: str, model: str, params: dict,
                spec: PromptSpec, user: Optional[str] = None) -> str:
    """Cache key: sha256 over everything that changes the reply (provider,
    model, generation params, prompt version, arm, system + user text)."""
    payload = {
        "provider": provider_name, "model": model, "params": params,
        "prompt_version": spec.prompt_version, "arm": spec.arm,
        "system": spec.system, "user": spec.user if user is None else user,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_PROB_RE = re.compile(r"(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)", re.ASCII)


def parse_probability(text) -> Optional[float]:
    """Strict: the whole (trimmed) reply must be one decimal in [0, 1] —
    ``0.62``, ``.62``, ``0``, ``1``, ``1.0``. Percentages, words, ranges or
    any extra text → ``None`` (the caller re-asks, then records 'invalid')."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not _PROB_RE.fullmatch(s):
        return None
    return float(s)


# ---------------------------------------------------------------------------
# Prop selection (read-only)
# ---------------------------------------------------------------------------


def _is_half_point(line: float) -> bool:
    return abs(line * 2 - round(line * 2)) < 1e-9 and abs(line - round(line)) > 1e-9


def _parse_config(config_json: Optional[str]) -> dict:
    try:
        cfg = json.loads(config_json) if config_json else {}
    except ValueError:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def _model_row_rank(cfg: dict, default_distribution: str, default_window: int) -> int:
    """0 = production config (default distribution + window), 1 = default
    distribution only, 2 = anything else."""
    if str(cfg.get("distribution") or "") != default_distribution:
        return 2
    return 0 if cfg.get("rolling_window") == default_window else 1


def market_prob_over(conn, player_id, game_date, stat_type, line_value):
    """De-vigged P(over) from REAL two-sided ``betting_lines`` prices at
    exactly this line: each book's latest row scraped before
    ``MARKET_PRICE_CUTOFF_UTC`` on game day, proportional de-vig (implied over
    / (implied over + implied under)), averaged across books. ``(None, 0)``
    when no book posted both sides at this line in time."""
    day = str(game_date)[:10]
    rows = conn.execute(
        """
        SELECT lower(book), over_odds, under_odds
        FROM betting_lines
        WHERE player_id = ? AND DATE(game_date) = DATE(?)
          AND lower(stat_type) = lower(?) AND ABS(line_value - ?) < 1e-6
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
          AND datetime(scraped_at) < datetime(?)
        ORDER BY scraped_at DESC, line_id DESC
        """,
        (int(player_id), day, str(stat_type), float(line_value),
         f"{day} {MARKET_PRICE_CUTOFF_UTC}"),
    ).fetchall()
    seen: set = set()
    probs = []
    for book, over_odds, under_odds in rows:
        if book in seen:
            continue
        seen.add(book)
        p_over, p_under = _implied_or_none(over_odds), _implied_or_none(under_odds)
        if p_over is None or p_under is None:
            continue
        probs.append(p_over / (p_over + p_under))
    if not probs:
        return None, 0
    return sum(probs) / len(probs), len(probs)


def _player_logs(conn, player_id: int, stats: Sequence[str]) -> list:
    """Every game_logs row for a player, newest first, with the realized value
    of each requested stat (same ``_GAMELOG_STAT_EXPR`` definitions the
    settlement uses)."""
    cols = ", ".join(f"{_GAMELOG_STAT_EXPR[s]} AS v{i}" for i, s in enumerate(stats))
    rows = conn.execute(
        f"""
        SELECT DATE(g.game_date), g.matchup, g.home_away, g.minutes, {cols}
        FROM game_logs g
        WHERE g.player_id = ?
        ORDER BY DATE(g.game_date) DESC
        """,
        (int(player_id),),
    ).fetchall()
    return [{"date": r[0], "matchup": r[1], "home_away": r[2], "minutes": r[3],
             **{s: r[4 + i] for i, s in enumerate(stats)}} for r in rows]


def select_retro_props(
    conn,
    *,
    stats: Optional[Sequence[str]] = None,
    date_range: Optional[Sequence[str]] = None,
    line_kind: str = "half",
    max_line_z: Optional[float] = DEFAULT_MAX_LINE_Z,
    min_history: int = DEFAULT_MIN_HISTORY,
    last_n: int = DEFAULT_LAST_N,
) -> tuple:
    """Settled props (over/under — pushes can't be scored) from
    ``predictions``, one per (player, date, stat, line), with the prompt facts
    attached. Returns ``(props, funnel)``; ``funnel`` counts what each filter
    dropped. Every filter uses pre-game facts only, so none of them can select
    on the outcome."""
    from nba_model.model.prop_board import DEFAULT_ROLLING_WINDOW
    from nba_model.model.simulation import get_default_distribution

    if line_kind not in ("half", "any"):
        raise ValueError("line_kind must be 'half' or 'any'")
    where = ["p.outcome IN ('over', 'under')", "p.prob_over IS NOT NULL",
             "p.line_value IS NOT NULL", "COALESCE(p.sport, 'nba') = 'nba'"]
    params: list = []
    stat_filter = [s.strip().lower() for s in (stats or []) if s and s.strip()]
    unknown = sorted(set(stat_filter) - set(_GAMELOG_STAT_EXPR))
    if unknown:
        raise ValueError(f"unsupported stats: {', '.join(unknown)}")
    if stat_filter:
        where.append(f"lower(p.stat_type) IN ({', '.join('?' * len(stat_filter))})")
        params += stat_filter
    if date_range:
        where.append("DATE(p.game_date) BETWEEN DATE(?) AND DATE(?)")
        params += [str(date_range[0]), str(date_range[1])]
    rows = conn.execute(
        f"""
        SELECT p.prediction_id, p.player_id, DATE(p.game_date), lower(p.stat_type),
               p.line_value,
               (SELECT c.config_json FROM prediction_configs c
                 WHERE c.prediction_id = p.prediction_id
                 ORDER BY c.config_id DESC LIMIT 1),
               pl.name
        FROM predictions p
        LEFT JOIN players pl ON pl.player_id = p.player_id
        WHERE {' AND '.join(where)}
        ORDER BY p.prediction_id
        """,
        params,
    ).fetchall()

    # One prop per key; the model baseline = best-ranked row, newest on ties.
    # A key whose line exists only as a backtest model mean
    # (line_source='model') is not a real prop, and the backtest's points mean
    # reads the target game's minutes (injury_proxy) — so it is dropped.
    by_key: dict = {}
    for pred_id, player_id, game_date, stat, line, cfg_json, name in rows:
        if stat not in _GAMELOG_STAT_EXPR:
            continue
        key = (int(player_id), game_date, stat, round(float(line), 4))
        cfg = _parse_config(cfg_json)
        rank = (_model_row_rank(cfg, get_default_distribution(stat),
                                DEFAULT_ROLLING_WINDOW), -int(pred_id))
        real_line = cfg.get("line_source") != "model"
        best = by_key.get(key)
        if best is None:
            by_key[key] = [rank, int(pred_id), name, real_line]
            continue
        best[3] = best[3] or real_line
        if rank < best[0]:
            best[:3] = [rank, int(pred_id), name]

    funnel = {"settled_props": len(by_key)}
    keys = [k for k in by_key if line_kind == "any" or _is_half_point(k[3])]
    funnel["line_kind"] = len(keys)
    funnel["model_line"] = sum(1 for k in keys if not by_key[k][3])
    keys = [k for k in keys if by_key[k][3]]

    logs_by_player: dict = {}
    props = []
    dropped = {"no_player_name": 0, "short_history": 0, "implausible_line": 0}
    for key in sorted(keys):
        player_id, game_date, stat, line = key
        _, pred_id, name, _ = by_key[key]
        if not name:
            dropped["no_player_name"] += 1
            continue
        if player_id not in logs_by_player:
            stats_needed = sorted({k[2] for k in keys if k[0] == player_id})
            logs_by_player[player_id] = _player_logs(conn, player_id, stats_needed)
        logs = logs_by_player[player_id]
        # The game's own row gives the matchup only (schedule facts); its box
        # score never leaves this loop.
        target = next((g for g in logs if g["date"] == game_date), None)
        team, opp, home = parse_matchup(
            target["matchup"] if target else None,
            target["home_away"] if target else None)
        prior = [g for g in logs if g["date"] < game_date][:max(int(last_n), 1)]
        history = tuple(
            GameLine(g["date"], _opponent_label(g["matchup"]), g["minutes"], g[stat])
            for g in prior)
        values = [float(g.value) for g in history if g.value is not None]
        if len(values) < max(int(min_history), 1):
            dropped["short_history"] += 1
            continue
        if max_line_z:
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            z = abs(line - statistics.fmean(values)) / max(sd, 1.0)
            if z > float(max_line_z):
                dropped["implausible_line"] += 1
                continue
        p_mkt, n_books = market_prob_over(conn, player_id, game_date, stat, line)
        props.append(Prop(
            player_id=player_id, player_name=str(name), game_date=game_date,
            stat_type=stat, line_value=float(line), team=team, opponent=opp,
            home=home, history=history, prediction_id=pred_id, settled=True,
            market_prob_over=p_mkt, market_books=n_books,
        ))
    funnel.update(dropped)
    funnel["eligible"] = len(props)
    return props, funnel


def sample_props(props: Sequence[Prop], limit: Optional[int], seed: str) -> list:
    """Deterministic pseudo-random sample: order by sha256(seed | prop key) so a
    re-run (or another model) with the same seed gets the same props."""
    def order(p: Prop) -> str:
        k = f"{seed}|{p.player_id}|{p.game_date}|{p.stat_type}|{p.line_value:g}"
        return hashlib.sha256(k.encode("utf-8")).hexdigest()

    ordered = sorted(props, key=order)
    return ordered if not limit or limit <= 0 else ordered[:int(limit)]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass
class LLMReply:
    text: str
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    refused: bool = False


class ProviderNotConfigured(RuntimeError):
    """The provider can't run: its env key is unset or its optional SDK is
    missing. Raised before any API call."""


class ProviderError(RuntimeError):
    """One request failed. Messages carry the HTTP status / exception class
    only — never a response body (provider error bodies can quote a masked
    key). ``fatal`` = retrying the run won't help (bad request, auth, model)."""

    def __init__(self, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.fatal = bool(fatal)


_FATAL_STATUSES = (400, 401, 403, 404)


def env_key_present(name: Optional[str]) -> bool:
    return bool(name) and bool(os.environ.get(name, "").strip())


def _require_env_key(name: str) -> None:
    if not env_key_present(name):
        raise ProviderNotConfigured(
            f"{name} is not set — export it in the shell (never on the command line)")


class FakeProvider:
    """Offline provider for tests and wiring checks: no network, no key.
    Replies come from ``responder(system, user)`` or a ``replies`` queue
    (str / LLMReply / exception), else ``"0.5"``."""

    name = "fake"
    env_key = None

    def __init__(self, model: str = "fake-model", replies=None,
                 responder: Optional[Callable] = None):
        self.model = model
        self.calls: list = []
        self._replies = list(replies or [])
        self._responder = responder

    def params(self) -> dict:
        return {}

    def check_ready(self) -> None:
        return None

    def complete(self, system: str, user: str) -> LLMReply:
        self.calls.append((system, user))
        if self._responder is not None:
            reply = self._responder(system, user)
        elif self._replies:
            reply = self._replies.pop(0)
        else:
            reply = "0.5"
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, LLMReply):
            return reply
        return LLMReply(text=str(reply), stop_reason="end_turn",
                        input_tokens=estimate_tokens(system + user), output_tokens=2)


class AnthropicProvider:
    """Claude through the official ``anthropic`` SDK — an optional dependency
    imported only when a real run starts. The SDK reads ANTHROPIC_API_KEY
    itself, so the key never passes through this module; its own retries
    cover 429 / 5xx / connection errors.

    Deliberately no ``temperature`` (removed or fixed on current Claude
    models) and no refusal ``fallbacks``: a fallback would answer under another
    model's name and corrupt per-model attribution, so a refusal is recorded as
    a refusal. Thinking stays at the model default; ``effort`` bounds it."""

    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-opus-5-5"
    default_effort = "low"

    def __init__(self, model: Optional[str] = None, *,
                 max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                 effort: Optional[str] = default_effort, client=None,
                 timeout: float = 120.0, max_retries: int = 2):
        self.model = model or self.default_model
        self.max_tokens = int(max_tokens)
        self.effort = effort or None
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._client = client

    def params(self) -> dict:
        return {"max_tokens": self.max_tokens, "effort": self.effort}

    def check_ready(self) -> None:
        if self._client is not None:
            return
        _require_env_key(self.env_key)
        if importlib.util.find_spec("anthropic") is None:
            raise ProviderNotConfigured(
                "the 'anthropic' package is not installed "
                "(.venv/bin/python3 -m pip install anthropic)")

    def _get_client(self):
        if self._client is None:
            self.check_ready()
            import anthropic  # optional dependency — real runs only

            self._client = anthropic.Anthropic(
                timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def complete(self, system: str, user: str) -> LLMReply:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        client = self._get_client()
        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # SDK errors: class + status only, no body
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            raise ProviderError(
                f"anthropic request failed: {type(exc).__name__}{detail}",
                fatal=status in _FATAL_STATUSES) from None
        stop = getattr(resp, "stop_reason", None)
        text = "".join(
            getattr(block, "text", "") or ""
            for block in (getattr(resp, "content", None) or [])
            if getattr(block, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        return LLMReply(text=text, stop_reason=stop,
                        input_tokens=getattr(usage, "input_tokens", None),
                        output_tokens=getattr(usage, "output_tokens", None),
                        refused=stop == "refusal")


class OpenAIProvider:
    """OpenAI Chat Completions over plain HTTPS (``requests``, already
    pinned). Any OpenAI-compatible endpoint works via ``base_url`` +
    ``env_key``. The key is read from the environment per request and goes
    only into the Authorization header — never stored on the object, logged,
    or echoed in an error."""

    name = "openai"
    default_model = None   # no safe default: pass --model

    def __init__(self, model: Optional[str], *,
                 max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                 reasoning_effort: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 env_key: str = "OPENAI_API_KEY", transport=None,
                 timeout: float = 120.0, max_retries: int = 2,
                 sleep: Callable[[float], None] = time.sleep):
        if not model:
            raise ValueError("the openai provider needs an explicit --model")
        # Both values get printed / cached, so a key pasted into either one
        # must be refused WITHOUT echoing it.
        if not _ENV_NAME_RE.fullmatch(str(env_key or "")):
            raise ValueError("--api-key-env must be the NAME of an environment "
                             "variable (letters, digits, _), not a key")
        parts = urlsplit(str(base_url or ""))
        if (parts.scheme not in ("https", "http") or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment):
            raise ValueError("--base-url must be a plain http(s) URL with no "
                             "credentials, query string or fragment")
        self.model = model
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort or None
        self.base_url = base_url.rstrip("/")
        self.env_key = env_key
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._transport = transport or requests.post
        self._sleep = sleep

    def params(self) -> dict:
        return {"max_completion_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
                "base_url": self.base_url}

    def check_ready(self) -> None:
        _require_env_key(self.env_key)

    def complete(self, system: str, user: str) -> LLMReply:
        self.check_ready()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        url = f"{self.base_url}/chat/completions"
        error = ProviderError("openai request not attempted")
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._transport(
                    url, json=body, timeout=self.timeout,
                    headers={"Authorization": "Bearer " + os.environ[self.env_key].strip(),
                             "Content-Type": "application/json"})
            except requests.RequestException as exc:
                error = ProviderError(f"openai connection error: {type(exc).__name__}")
            else:
                status = int(resp.status_code)
                if status == 200:
                    return self._parse(resp)
                error = ProviderError(f"openai HTTP {status}",
                                      fatal=status in _FATAL_STATUSES)
                if status not in (408, 409, 429) and status < 500:
                    raise error
            if attempt < self.max_retries:
                self._sleep(min(2.0 ** attempt, 30.0))
        raise error

    @staticmethod
    def _parse(resp) -> LLMReply:
        try:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
        except (ValueError, AttributeError, TypeError):
            raise ProviderError("openai returned an unreadable body") from None
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        finish = choice.get("finish_reason")
        usage = data.get("usage") or {}
        return LLMReply(text=str(content), stop_reason=finish,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        refused=bool(message.get("refusal")) or finish == "content_filter")


def make_provider(name: str, model: Optional[str] = None, *,
                  max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                  effort: Optional[str] = None, base_url: Optional[str] = None,
                  api_key_env: Optional[str] = None):
    """Build a real provider from CLI options. ``effort=None`` = the
    provider's default; ``effort='none'`` omits the parameter."""
    omit = effort is not None and str(effort).strip().lower() == "none"
    if name == "anthropic":
        if effort is None:
            effort = AnthropicProvider.default_effort
        return AnthropicProvider(model, max_tokens=max_tokens,
                                 effort=None if omit else effort)
    if name == "openai":
        kwargs = {"max_tokens": max_tokens,
                  "reasoning_effort": None if omit else effort}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key_env:
            kwargs["env_key"] = api_key_env
        return OpenAIProvider(model, **kwargs)
    raise ValueError(f"unknown provider {name!r} (use one of {PROVIDERS})")


# ---------------------------------------------------------------------------
# Cache + cost
# ---------------------------------------------------------------------------


class PromptCache:
    """One JSON file per prompt hash under ``root``, written right after each
    paid call (temp file + atomic rename), so a crashed or repeated run costs
    nothing for prompts already answered. Holds request text, reply text and
    token usage — never credentials."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> Optional[dict]:
        path = self._path(digest)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("prompt_hash") == digest else None

    def put(self, digest: str, record: dict) -> None:
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({**record, "prompt_hash": digest}, fh, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text or "") / CHARS_PER_TOKEN_ESTIMATE))


def resolve_price(provider_name: str, model: str, override=None):
    """(input, output) USD per MTok, or None when unknown."""
    if override is not None:
        return (float(override[0]), float(override[1]))
    if provider_name == "fake":
        return (0.0, 0.0)
    if provider_name == "anthropic":
        return PRICING_USD_PER_MTOK.get(model)
    return None


def estimate_cost(specs: Sequence[PromptSpec], *, max_output_tokens: int,
                  price, max_reasks: int = DEFAULT_MAX_REASKS) -> dict:
    """Upper-bound cost for ``specs`` (the uncached prompts): input ≈ chars/3,
    output = ``max_output_tokens`` per call. ``usd_worst`` also assumes every
    prompt is re-asked ``max_reasks`` times."""
    first_in = sum(estimate_tokens(s.system + s.user) for s in specs)
    reask_in = sum(estimate_tokens(s.system + s.user + REASK_SUFFIX_V1) for s in specs)
    calls = len(specs)
    out_max = calls * int(max_output_tokens)
    est = {"calls": calls, "input_tokens_est": first_in, "output_tokens_max": out_max,
           "price_in_per_mtok": None, "price_out_per_mtok": None,
           "usd_upper": None, "usd_worst": None}
    if price is not None:
        p_in, p_out = price
        est["price_in_per_mtok"], est["price_out_per_mtok"] = p_in, p_out
        est["usd_upper"] = (first_in * p_in + out_max * p_out) / 1e6
        est["usd_worst"] = est["usd_upper"] + max(int(max_reasks), 0) * (
            reask_in * p_in + out_max * p_out) / 1e6
    return est


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_run_id(retro: bool) -> str:
    return f"llm-{'retro' if retro else 'fwd'}-{datetime.now(timezone.utc):%Y%m%d}"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))[:80]


@dataclass
class _Ask:
    digest: str
    reply: LLMReply
    cached: bool


def _ask(provider, cache: PromptCache, spec: PromptSpec, user: str) -> _Ask:
    digest = prompt_hash(provider.name, provider.model, provider.params(), spec, user)
    hit = cache.get(digest)
    if hit is not None:
        r = hit.get("response") or {}
        return _Ask(digest, LLMReply(
            text=str(r.get("text") or ""), stop_reason=r.get("stop_reason"),
            input_tokens=r.get("input_tokens"), output_tokens=r.get("output_tokens"),
            refused=bool(r.get("refused"))), True)
    reply = provider.complete(spec.system, user)
    cache.put(digest, {
        "created_at_utc": _utc_now_iso(), "provider": provider.name,
        "model": provider.model, "params": provider.params(),
        "prompt_version": spec.prompt_version, "arm": spec.arm,
        "request": {"system": spec.system, "user": user},
        "response": {"text": reply.text, "stop_reason": reply.stop_reason,
                     "input_tokens": reply.input_tokens,
                     "output_tokens": reply.output_tokens, "refused": reply.refused},
    })
    return _Ask(digest, reply, False)


@dataclass
class _Tally:
    api_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    statuses: dict = field(default_factory=lambda: {"ok": 0, "invalid": 0, "refusal": 0})
    failures: int = 0

    def add(self, ask: _Ask) -> None:
        if ask.cached:
            self.cache_hits += 1
            return
        self.api_calls += 1
        self.input_tokens += int(ask.reply.input_tokens or 0)
        self.output_tokens += int(ask.reply.output_tokens or 0)


def forecast_prop(provider, cache: PromptCache, prop: Prop, *, arm: str,
                  max_reasks: int, tally: _Tally) -> dict:
    """Ask for P(over) on one prop (cache first), re-asking up to
    ``max_reasks`` times on an unreadable reply. A refusal is recorded, never
    re-asked. Returns the ``llm_forecasts`` fields that depend on the reply."""
    spec = build_prompt(prop, arm=arm)
    first = _ask(provider, cache, spec, spec.user)
    tally.add(first)
    attempts = [first.reply]
    prob = None if first.reply.refused else parse_probability(first.reply.text)
    reasks = 0
    while prob is None and not attempts[-1].refused and reasks < int(max_reasks):
        reasks += 1
        again = _ask(provider, cache, spec, spec.user + REASK_SUFFIX_V1 * reasks)
        tally.add(again)
        attempts.append(again.reply)
        prob = None if again.reply.refused else parse_probability(again.reply.text)
    status = "ok" if prob is not None else ("refusal" if attempts[-1].refused else "invalid")
    tally.statuses[status] += 1
    return {
        "prompt_hash": first.digest,
        "prompt_version": spec.prompt_version,
        "prob_over": prob,
        "parse_status": status,
        "raw_response": json.dumps(
            [{"text": a.text, "stop_reason": a.stop_reason} for a in attempts]),
    }


def run_llm_forecasts(
    db_path: str = DEFAULT_DB_PATH,
    provider=None,
    *,
    retro: bool = False,
    arm: str = "blind",
    run_id: Optional[str] = None,
    stats: Optional[Sequence[str]] = None,
    date_range: Optional[Sequence[str]] = None,
    limit: Optional[int] = DEFAULT_LIMIT,
    seed: str = DEFAULT_SEED,
    last_n: int = DEFAULT_LAST_N,
    min_history: int = DEFAULT_MIN_HISTORY,
    line_kind: str = "half",
    max_line_z: Optional[float] = DEFAULT_MAX_LINE_Z,
    max_reasks: int = DEFAULT_MAX_REASKS,
    training_cutoff: Optional[str] = None,
    cache_dir=DEFAULT_CACHE_DIR,
    artifact_dir=ARTIFACT_DIR,
    max_cost_usd: Optional[float] = DEFAULT_MAX_COST_USD,
    price_per_mtok=None,
    dry_run: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Select props, estimate cost, and (unless ``dry_run``) collect + store
    the provider's forecasts. Returns a summary dict (no secrets in it).

    Raises ``ValueError`` for an invalid request (forward run, bad arm, cost
    over ``max_cost_usd``, unknown pricing) and ``ProviderNotConfigured``
    before any API call when the provider can't run."""
    if not retro:
        raise ValueError(
            "forward runs (forecast before tip-off) need in-season props and "
            "are not built yet; pass --retro for a historical run (retro=1)")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (use one of {ARMS})")
    if provider is None:
        raise ValueError("a provider is required")
    run_id = run_id or default_run_id(retro)

    with DatabaseManager(db_path=db_path, read_only=True) as db:
        eligible, funnel = select_retro_props(
            db.conn, stats=stats, date_range=date_range, line_kind=line_kind,
            max_line_z=max_line_z, min_history=min_history, last_n=last_n)
        stored = {}
        if db.has_table("llm_forecasts"):
            stored = {
                (int(pid), day, stat, round(float(line), 4)): digest
                for pid, day, stat, line, digest in db.conn.execute(
                    "SELECT player_id, DATE(game_date), stat_type, line_value, prompt_hash "
                    "FROM llm_forecasts WHERE run_id = ? AND model_name = ? AND arm = ?",
                    (run_id, provider.model, arm))
            }
    priced = [p for p in eligible if p.market_prob_over is not None]
    pool = priced if arm == "market" else eligible
    selected = sample_props(pool, limit, seed)

    # A prop this run already answered is skipped (free, never re-inserted);
    # one answered with a different prompt or settings is a conflict — the
    # UNIQUE key would silently drop the new, paid answer.
    cache = PromptCache(cache_dir)
    spec_by_key = {p.key: build_prompt(p, arm=arm) for p in selected}
    digests = {k: prompt_hash(provider.name, provider.model, provider.params(), s)
               for k, s in spec_by_key.items()}
    conflicts = sum(1 for k in digests if stored.get(k, digests[k]) != digests[k])
    already = sum(1 for k in digests if stored.get(k) == digests[k])
    props = [p for p in selected if p.key not in stored]
    specs = [spec_by_key[p.key] for p in props]
    uncached = [s for p, s in zip(props, specs) if cache.get(digests[p.key]) is None]
    price = resolve_price(provider.name, provider.model, price_per_mtok)
    max_out = int(provider.params().get("max_tokens")
                  or provider.params().get("max_completion_tokens") or 0)
    cost = estimate_cost(uncached, max_output_tokens=max_out, price=price,
                         max_reasks=max_reasks)

    summary = {
        "run_id": run_id, "provider": provider.name, "model": provider.model,
        "params": provider.params(), "prompt_version": PROMPT_VERSION, "arm": arm,
        "retro": True, "dry_run": bool(dry_run), "seed": seed,
        "selection": {"stats": list(stats or []), "date_range": list(date_range or []),
                      "line_kind": line_kind, "max_line_z": max_line_z,
                      "min_history": min_history, "last_n": last_n, "limit": limit},
        "funnel": {**funnel, "with_real_price": len(priced)},
        "props": len(selected),
        "by_stat": {s: sum(1 for p in selected if p.stat_type == s)
                    for s in sorted({p.stat_type for p in selected})},
        "date_span": ([min(p.game_date for p in selected),
                       max(p.game_date for p in selected)] if selected else []),
        "already_stored": already,
        "conflicts": conflicts,
        "cached_prompts": len(specs) - len(uncached),
        "uncached_prompts": len(uncached),
        "cost_estimate": cost,
        "env_key": getattr(provider, "env_key", None),
        "env_key_set": env_key_present(getattr(provider, "env_key", None)),
        "sample_prompt": ({"system": specs[0].system, "user": specs[0].user}
                          if specs else None),
    }
    if dry_run:
        return summary

    if conflicts:
        raise ValueError(
            f"run {run_id!r} already holds {conflicts} of these props for "
            f"{provider.model} [{arm}] made with a different prompt or settings; "
            "use a new --run-id")
    if uncached and price is None:
        raise ValueError(
            f"no pricing for {provider.name}/{provider.model}: pass "
            "--price-in-per-mtok and --price-out-per-mtok")
    worst = cost.get("usd_worst") or 0.0
    if max_cost_usd is not None and worst > float(max_cost_usd):
        raise ValueError(
            f"worst-case cost ${worst:.2f} exceeds --max-cost-usd "
            f"{float(max_cost_usd):.2f}; lower --limit or raise the cap")
    if uncached:
        provider.check_ready()

    tally = _Tally()
    rows = []
    aborted = None
    consecutive = 0
    created = _utc_now_iso()
    for i, prop in enumerate(props, start=1):
        try:
            fields = forecast_prop(provider, cache, prop, arm=arm,
                                   max_reasks=max_reasks, tally=tally)
        except ProviderError as exc:
            tally.failures += 1
            consecutive += 1
            logger.warning("llm forecast request failed",
                           extra={"run_id": run_id, "error": str(exc)})
            if exc.fatal or consecutive >= MAX_CONSECUTIVE_FAILURES:
                aborted = str(exc)
                break
            continue
        consecutive = 0
        rows.append({
            **fields, "run_id": run_id, "provider": provider.name,
            "model_name": provider.model, "arm": arm,
            "player_id": prop.player_id, "player_name": prop.player_name,
            "game_date": prop.game_date, "stat_type": prop.stat_type,
            "line_value": prop.line_value,
            # Blind facts end the day before the game; a market price can be
            # as late as the game-day cutoff.
            "info_cutoff_utc": (f"{prop.game_date}T{MARKET_PRICE_CUTOFF_UTC}Z"
                                if arm == "market" else f"{prop.game_date}T00:00:00Z"),
            "retro": 1 if prop.settled or prop.game_date < created[:10] else 0,
            "training_cutoff": training_cutoff, "prediction_id": prop.prediction_id,
            "created_at_utc": created,
        })
        if progress and i % 25 == 0:
            progress(f"  {i}/{len(props)} props")

    insert = settle = None
    if rows:
        with DatabaseManager(db_path=db_path) as db:
            insert = db.insert_llm_forecasts(rows)
            settle = db.settle_llm_forecasts()
    actual_usd = None
    if price is not None:
        actual_usd = (tally.input_tokens * price[0] + tally.output_tokens * price[1]) / 1e6
    summary.update({
        "rows": len(rows), "aborted": aborted, "api_calls": tally.api_calls,
        "cache_hits": tally.cache_hits, "failures": tally.failures,
        "parse_status": dict(tally.statuses),
        "usage": {"input_tokens": tally.input_tokens,
                  "output_tokens": tally.output_tokens, "usd": actual_usd},
        "insert": insert, "settle": settle,
    })
    summary.pop("sample_prompt", None)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / (
        f"llm_forecast_{_safe_name(run_id)}_{_safe_name(provider.model)}_{arm}_"
        f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json")
    manifest.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["manifest_path"] = str(manifest)
    logger.info("llm forecast run complete",
                extra={"run_id": run_id, "provider": provider.name,
                       "model": provider.model, "rows": len(rows),
                       "api_calls": tally.api_calls, "failures": tally.failures})
    return summary


def settle_forecasts(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Grade pending ``llm_forecasts`` rows (writer open)."""
    with DatabaseManager(db_path=db_path) as db:
        return db.settle_llm_forecasts()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_summary(summary: dict) -> str:
    est = summary["cost_estimate"]
    f = summary["funnel"]
    head = "DRY RUN — no API call, no DB write, no files" if summary["dry_run"] else "RUN"
    lines = [
        f"LLM forecast {head}",
        f"  run_id {summary['run_id']} | {summary['provider']}/{summary['model']} | "
        f"arm {summary['arm']} | prompt {summary['prompt_version']} | params {summary['params']}",
        f"  props: {summary['props']} selected (seed {summary['seed']!r}) of {f['eligible']} eligible"
        + (f"; market arm needs a real price: {f['with_real_price']} have one"
           if summary["arm"] == "market" else ""),
        f"    funnel: {f['settled_props']} settled props → {f['line_kind']} after "
        f"line filter ({summary['selection']['line_kind']}) → -{f['model_line']} "
        f"backtest model-mean lines, -{f['no_player_name']} no name, "
        f"-{f['short_history']} short history, -{f['implausible_line']} "
        f"implausible line (|z| > {summary['selection']['max_line_z']})",
        f"    by stat: {summary['by_stat']}; dates {summary['date_span']}; "
        "retro=1 (historical: possible training-data leakage, never a headline)",
        f"  already in this run: {summary['already_stored']} (skipped); "
        f"cache: {summary['cached_prompts']} cached, {summary['uncached_prompts']} to call",
    ]
    if summary.get("conflicts"):
        lines.append(
            f"  CONFLICT: {summary['conflicts']} props are already in run "
            f"{summary['run_id']!r} with a different prompt or settings — a real run "
            "refuses; use a new --run-id")
    if est["usd_upper"] is None:
        lines.append(
            f"  cost estimate: pricing unknown for {summary['model']} — pass "
            f"--price-in-per-mtok/--price-out-per-mtok ({est['calls']} calls, "
            f"~{est['input_tokens_est']:,} input tokens, ≤{est['output_tokens_max']:,} output)")
    else:
        lines.append(
            f"  cost estimate @ ${est['price_in_per_mtok']:.2f}/${est['price_out_per_mtok']:.2f} "
            f"per MTok: {est['calls']} calls, ~{est['input_tokens_est']:,} input tokens "
            f"(chars/3), ≤{est['output_tokens_max']:,} output tokens → ≤ ${est['usd_upper']:.2f}; "
            f"with every prompt re-asked ≤ ${est['usd_worst']:.2f}")
    if summary.get("env_key"):
        lines.append(f"  {summary['env_key']}: {'set' if summary['env_key_set'] else 'NOT set'}")
    if summary["dry_run"]:
        if summary.get("sample_prompt"):
            lines += ["", "--- sample prompt (system) ---", summary["sample_prompt"]["system"],
                      "--- sample prompt (user) ---", summary["sample_prompt"]["user"]]
        return "\n".join(lines)
    lines += [
        f"  stored {summary['rows']} rows: {summary['parse_status']}; "
        f"{summary['api_calls']} API calls, {summary['cache_hits']} cache hits, "
        f"{summary['failures']} failed requests",
        f"  usage: {summary['usage']}",
        f"  insert: {summary['insert']}; settle: {summary['settle']}",
        f"  manifest: {summary.get('manifest_path')}",
    ]
    if summary.get("aborted"):
        lines.append(f"  ABORTED early: {summary['aborted']} (re-run resumes from the cache)")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect LLM P(over) forecasts for the WS11 calibration benchmark.")
    p.add_argument("--db", default=DEFAULT_DB_PATH, dest="db_path")
    p.add_argument("--provider", choices=PROVIDERS, default="anthropic")
    p.add_argument("--model", default=None,
                   help="Model id (anthropic default: claude-opus-5-5; openai: required).")
    p.add_argument("--retro", action="store_true",
                   help="Historical run on settled props (rows labeled retro=1). "
                        "Required until forward runs exist.")
    p.add_argument("--arm", choices=ARMS, default="blind",
                   help="blind = no market price in the prompt (default); market = "
                        "adds the de-vigged price (props with a real price only).")
    p.add_argument("--run-id", default=None)
    p.add_argument("--stats", nargs="*", default=None)
    p.add_argument("--date-range", nargs=2, metavar=("START", "END"), default=None)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--seed", default=DEFAULT_SEED)
    p.add_argument("--last-n", type=int, default=DEFAULT_LAST_N)
    p.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    p.add_argument("--line-kind", choices=("half", "any"), default="half")
    p.add_argument("--max-line-z", type=float, default=DEFAULT_MAX_LINE_Z,
                   help="Drop lines more than Z SDs from the player's last-N mean (0 = keep all).")
    p.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p.add_argument("--effort", default=None,
                   help="anthropic: output_config.effort (default low); openai: "
                        "reasoning_effort (default omitted); 'none' omits it.")
    p.add_argument("--base-url", default=None, help="openai-compatible endpoint")
    p.add_argument("--api-key-env", default=None,
                   help="openai: env var holding the key (default OPENAI_API_KEY)")
    p.add_argument("--training-cutoff", default=None,
                   help="Model training cutoff YYYY-MM-DD when known (else unknown = leak risk).")
    p.add_argument("--max-reasks", type=int, default=DEFAULT_MAX_REASKS)
    p.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    p.add_argument("--price-in-per-mtok", type=float, default=None)
    p.add_argument("--price-out-per-mtok", type=float, default=None)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cost estimate + a sample prompt and stop.")
    p.add_argument("--settle-only", action="store_true",
                   help="Grade pending llm_forecasts rows and exit (writer open).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging()
    if args.settle_only:
        if args.dry_run:
            print("error: --settle-only writes outcomes; it can't run with --dry-run",
                  file=sys.stderr)
            return 2
        print(f"settle_llm_forecasts: {settle_forecasts(args.db_path)}")
        return 0
    price = None
    if (args.price_in_per_mtok is None) != (args.price_out_per_mtok is None):
        print("error: pass both --price-in-per-mtok and --price-out-per-mtok", file=sys.stderr)
        return 2
    if args.price_in_per_mtok is not None:
        price = (args.price_in_per_mtok, args.price_out_per_mtok)
    try:
        provider = make_provider(
            args.provider, args.model, max_tokens=args.max_output_tokens,
            effort=args.effort, base_url=args.base_url, api_key_env=args.api_key_env)
        summary = run_llm_forecasts(
            args.db_path, provider, retro=args.retro, arm=args.arm,
            run_id=args.run_id, stats=args.stats, date_range=args.date_range,
            limit=args.limit, seed=args.seed, last_n=args.last_n,
            min_history=args.min_history, line_kind=args.line_kind,
            max_line_z=args.max_line_z or None, max_reasks=args.max_reasks,
            training_cutoff=args.training_cutoff, cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir, max_cost_usd=args.max_cost_usd,
            price_per_mtok=price, dry_run=args.dry_run, progress=print)
    except (ValueError, ProviderNotConfigured, DatabaseNotReadyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(summary))
    return 1 if summary.get("aborted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
