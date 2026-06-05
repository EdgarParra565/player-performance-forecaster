"""Streamlit-friendly subprocess runner for the admin Operations console.

The desktop UI wraps ``subprocess.Popen`` in a Tk-aware reader thread that
streams stdout into a ``ScrolledText`` widget. Streamlit reruns the whole
script on every event, so we can't keep a long-lived Tk-style poller; instead
this module:

  - Spawns a child process in its own session (so ``stop()`` can SIGTERM
    the whole group, including grand-children).
  - Pipes stdout/stderr into a background reader thread that drains into
    a thread-safe queue and an in-memory transcript.
  - Persists the runner state in ``st.session_state`` so it survives Streamlit
    reruns; the panel re-attaches on every render.
  - Provides ``render_operations_panel(...)`` which is the only thing the
    web app needs to call.

Security model
--------------
This panel runs arbitrary ``python -m <module>`` subprocesses with the project
venv. It MUST only be reachable behind ``web_auth.is_admin()``. The caller
gates the view; we don't trust the URL.

The list of available operations is hardcoded below (mirroring the desktop
``Operations`` tab) so an attacker who somehow bypasses the gate still can't
launch an arbitrary module name through a free-form field. All arguments are
shell-escaped via ``shlex`` for display, but Popen receives an argv list
directly (no shell), so injection through field values is structurally
impossible — the OS treats them as literal arguments.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

import streamlit as st


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_TRANSCRIPT_CHARS = 200_000


@dataclass
class _RunnerState:
    proc: subprocess.Popen | None = None
    reader_thread: threading.Thread | None = None
    queue: "Queue[str]" = field(default_factory=Queue)
    transcript: str = ""
    label: str = ""
    last_cmd: list[str] = field(default_factory=list)
    last_exit_code: int | None = None
    started_at: datetime | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def drain(self) -> None:
        """Pull any newly buffered lines into the transcript (non-blocking)."""
        try:
            while True:
                chunk = self.queue.get_nowait()
                self.transcript += chunk
        except Empty:
            pass
        if len(self.transcript) > _MAX_TRANSCRIPT_CHARS:
            # Keep tail; truncating the head keeps the latest output visible.
            self.transcript = (
                f"[...{len(self.transcript) - _MAX_TRANSCRIPT_CHARS} earlier bytes elided...]\n"
                + self.transcript[-_MAX_TRANSCRIPT_CHARS:]
            )

    def start(
        self,
        label: str,
        cmd_args: list[str],
        cwd: str | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> bool:
        if self.is_running():
            return False
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if env_extra:
            env.update(env_extra)
        self.label = label
        self.last_cmd = list(cmd_args)
        self.started_at = datetime.now()
        self.transcript += (
            f"\n[{self.started_at.strftime('%H:%M:%S')}] $ {label}\n"
            f"  cmd: {' '.join(shlex.quote(a) for a in cmd_args)}\n"
        )
        try:
            self.proc = subprocess.Popen(
                cmd_args,
                cwd=cwd or str(_PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.transcript += f"  ERROR: failed to launch: {exc}\n"
            self.proc = None
            return False

        def _read_loop(proc: subprocess.Popen, queue: "Queue[str]") -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    queue.put(line)
            finally:
                proc.wait()
                queue.put(f"\n[exit {proc.returncode}]\n")

        self.reader_thread = threading.Thread(
            target=_read_loop, args=(self.proc, self.queue), daemon=True,
        )
        self.reader_thread.start()
        return True

    def stop(self) -> None:
        if not self.is_running():
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)  # type: ignore[union-attr]
            self.transcript += "  >> SIGTERM sent\n"
        except Exception as exc:  # noqa: BLE001
            self.transcript += f"  >> stop failed: {exc}\n"

    def clear_transcript(self) -> None:
        self.transcript = ""
        self.last_exit_code = None


# ---------------------------------------------------------------------------
# Session-state plumbing — Streamlit recreates the module on every script run,
# but session_state survives, so we stash the live _RunnerState there.
# ---------------------------------------------------------------------------
_RUNNER_KEY = "_ops_runner_state"


def _get_runner() -> _RunnerState:
    state = st.session_state.get(_RUNNER_KEY)
    if state is None:
        state = _RunnerState()
        st.session_state[_RUNNER_KEY] = state
    return state


# ---------------------------------------------------------------------------
# Operation registry — mirrors simple_ui's Operations tab. Each entry is:
#   slug: {
#     "label": str,                # button + heading
#     "module": str,               # python -m target
#     "caveat": str | None,        # warning shown above the form (e.g. needs Chrome)
#     "fields": [(arg_flag, label, default, kind)],
#     "flags":  [(flag, label, default)],
#   }
# Kinds: "opt" (passes --flag value when non-empty), "positional" (passes value alone),
#        "spaced" (splits whitespace and passes as --flag tok1 tok2 ...).
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, dict[str, Any]] = {
    "daily_etl": {
        "label": "Daily ETL",
        "module": "nba_model.data.daily_etl",
        "caveat": None,
        "fields": [
            ("--db-path", "DB path", "data/database/nba_data.db", "opt"),
            ("--season", "Season (e.g. 2025-26)", "", "opt"),
            ("--players", "Players (space-separated)", "", "spaced"),
            ("--min-players", "Min players", "0", "opt"),
            ("--player-limit", "Player limit", "", "opt"),
            ("--game-log-games", "Game-log games", "", "opt"),
        ],
        "flags": [
            ("--skip-game-logs", "skip game logs", False),
            ("--skip-team-defense", "skip team defense", False),
            ("--skip-odds", "skip odds", False),
            ("--skip-browser-parser", "skip browser parser", False),
            ("--skip-reverse-engineering", "skip reverse-engineering", False),
            ("--all-db-players", "all DB players", False),
            ("--skip-zero-game-players", "skip zero-game players", False),
            ("--strict", "strict mode", False),
            ("--force-odds-poll", "force odds poll", False),
        ],
    },
    "web_validate": {
        "label": "Validate Web Session",
        "module": "nba_model.model.web_text_ingestion",
        "caveat": (
            "Requires a real Chrome with `--remote-debugging-port=9222` "
            "running on the *same host* as Streamlit. Streamlit Cloud cannot "
            "satisfy this — run from a local deploy."
        ),
        "fields": [
            ("--validate-session", "Target URL", "https://app.prizepicks.com/board/nba", "positional"),
            ("--browser-auth-state-file", "Auth state file", "data/config/auth/prizepicks_state.json", "opt"),
            ("--chrome-debug-port", "Chrome debug port", "9222", "opt"),
        ],
        "flags": [],
    },
    "web_fetch": {
        "label": "Fetch Web URL (force-poll)",
        "module": "nba_model.model.web_text_ingestion",
        "caveat": "Same Chrome-on-:9222 requirement as Validate.",
        "fields": [
            ("--urls", "Target URL", "https://app.prizepicks.com/board/nba", "positional"),
            ("--browser-auth-state-file", "Auth state file", "data/config/auth/prizepicks_state.json", "opt"),
            ("--chrome-debug-port", "Chrome debug port", "9222", "opt"),
        ],
        "flags": [
            ("--force-poll", "force poll", True),
        ],
    },
    "web_sync_players": {
        "label": "Sync Active Players Reference",
        "module": "nba_model.model.web_text_ingestion",
        "caveat": None,
        "fields": [
            ("--sync-active-players-ref", "(no value — flag)", "", "bare_flag"),
            ("--active-players-output-file", "Output file",
             "data/config/active_nba_players.txt", "opt"),
        ],
        "flags": [],
    },
    "browser_parser": {
        "label": "Browser Prop Parser",
        "module": "nba_model.model.browser_prop_parser",
        "caveat": "Reads prior browser-captured snapshots; needs the Chrome-on-:9222 ETL to have run.",
        "fields": [
            ("--urls-file", "URLs file", "data/config/web_text_urls.txt", "opt"),
            ("--min-parse-confidence", "Min parse confidence", "0.50", "opt"),
            ("--max-snapshots-per-url", "Max snapshots per URL", "2", "opt"),
        ],
        "flags": [],
    },
    "eval_benchmark": {
        "label": "Real-data Benchmark",
        "module": "nba_model.evaluation.run_real_data_benchmark",
        "caveat": None,
        "fields": [
            ("--start-date", "Start date (YYYY-MM-DD)", "", "opt"),
            ("--end-date", "End date (YYYY-MM-DD)", "", "opt"),
            ("--stat-types", "Stat types (space-sep)", "points assists rebounds pra", "spaced"),
            ("--windows", "Windows (space-sep)", "5 7 10 15", "spaced"),
            ("--distributions", "Distributions (space-sep)", "normal", "spaced"),
        ],
        "flags": [],
    },
    "eval_distribution_sweep": {
        "label": "Distribution Sweep",
        "module": "nba_model.evaluation.run_distribution_sweep",
        "caveat": None,
        "fields": [
            ("--start-date", "Start date (YYYY-MM-DD)", "", "opt"),
            ("--end-date", "End date (YYYY-MM-DD)", "", "opt"),
            ("--stat-types", "Stat types (space-sep)", "points assists rebounds pra", "spaced"),
            ("--windows", "Windows (space-sep)", "5 7 10 15", "spaced"),
        ],
        "flags": [],
    },
    "eval_line_compare": {
        "label": "Line Comparison",
        "module": "nba_model.evaluation.line_comparison",
        "caveat": None,
        "fields": [
            ("--start-date", "Start date (YYYY-MM-DD)", "", "opt"),
            ("--end-date", "End date (YYYY-MM-DD)", "", "opt"),
            ("--stat-types", "Stat types (space-sep)", "points assists rebounds pra", "spaced"),
            ("--edge-threshold", "Edge threshold", "0.02", "opt"),
        ],
        "flags": [],
    },
    "eval_monthly_diag": {
        "label": "Monthly Diagnostics",
        "module": "nba_model.evaluation.monthly_diagnostics",
        "caveat": None,
        "fields": [
            ("--start-date", "Start date (YYYY-MM-DD)", "", "opt"),
            ("--end-date", "End date (YYYY-MM-DD)", "", "opt"),
            ("--stat-types", "Stat types (space-sep)", "points assists rebounds pra", "spaced"),
        ],
        "flags": [],
    },
    "reverse_engineering": {
        "label": "Market Reverse-Engineering",
        "module": "nba_model.evaluation.market_reverse_engineering",
        "caveat": None,
        "fields": [
            ("--source", "Source (book/predictions/both)", "both", "opt"),
            ("--poll-seconds", "Poll seconds", "300", "opt"),
            ("--min-inferred-rows", "Min inferred rows", "25", "opt"),
            ("--min-book-stat-groups", "Min book/stat groups", "2", "opt"),
            ("--min-player-segment-groups", "Min player-segment groups", "5", "opt"),
            ("--require-stability-runs", "Stability runs", "2", "opt"),
            ("--stability-tolerance", "Stability tolerance", "0.10", "opt"),
        ],
        "flags": [
            ("--continuous", "continuous mode", True),
        ],
    },
    "db_audit": {
        "label": "DB Audit",
        "module": "nba_model.data.audit_db",
        "caveat": None,
        "fields": [
            ("--db-path", "DB path", "data/database/nba_data.db", "opt"),
            ("--output", "Output file", "data/DATABASE_INVENTORY.txt", "opt"),
        ],
        "flags": [
            ("--stdout", "also print full report", True),
        ],
    },
}


def build_command(op_slug: str, values: dict[str, str], flag_values: dict[str, bool]) -> list[str]:
    """Translate form values into a ``python -m <module> ...`` argv list.

    Centralized so tests can assert on the argv without spawning a subprocess.
    """
    if op_slug not in OPERATIONS:
        raise ValueError(f"unknown operation {op_slug!r}")
    spec = OPERATIONS[op_slug]
    args: list[str] = [sys.executable, "-m", spec["module"]]
    for flag, _label, _default, kind in spec["fields"]:
        raw = (values.get(flag, "") or "").strip()
        if kind == "bare_flag":
            # Pass the flag with no value (e.g. --sync-active-players-ref).
            args.append(flag)
        elif kind == "positional":
            if raw:
                args.extend([flag, raw])
        elif kind == "opt":
            if raw:
                args.extend([flag, raw])
        elif kind == "spaced":
            toks = [t for t in raw.split() if t]
            if toks:
                args.extend([flag, *toks])
        else:
            raise ValueError(f"unknown field kind {kind!r}")
    for flag, _label, _default in spec["flags"]:
        if flag_values.get(flag):
            args.append(flag)
    return args


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_operations_panel(*, on_authorized: Callable[[], bool] | None = None) -> None:
    """Render the admin Operations console.

    The caller is expected to have already gated the view on
    ``web_auth.is_admin()``. We accept an ``on_authorized`` callback so this
    module never imports the auth surface directly (keeps the dependency
    graph one-way: app.py imports operations_panel, never the reverse).
    """
    if on_authorized is not None and not on_authorized():
        st.error("Operations console is admin-only.")
        return

    runner = _get_runner()
    runner.drain()

    st.subheader("Operations console")
    st.caption(
        "Launches CLI pipelines as subprocesses on the host running Streamlit. "
        "Output streams below. Web-scraping operations require a real Chrome "
        "on `:9222` on the same host (Streamlit Cloud can't satisfy this — "
        "run from a local deploy)."
    )

    op_keys = list(OPERATIONS.keys())
    op_labels = {k: OPERATIONS[k]["label"] for k in op_keys}
    chosen = st.selectbox(
        "Operation",
        op_keys,
        format_func=lambda k: op_labels[k],
        key="_ops_choice",
    )
    spec = OPERATIONS[chosen]
    if spec.get("caveat"):
        st.warning(spec["caveat"])

    # --- form: text fields + boolean flags ---
    with st.form(f"ops_form_{chosen}", clear_on_submit=False):
        values: dict[str, str] = {}
        for flag, label, default, kind in spec["fields"]:
            if kind == "bare_flag":
                st.caption(f"`{flag}` is always passed for this operation.")
                values[flag] = ""
                continue
            values[flag] = st.text_input(
                f"{label} (`{flag}`)",
                value=default,
                key=f"ops_{chosen}_{flag}",
            )
        flag_values: dict[str, bool] = {}
        for flag, label, default in spec["flags"]:
            flag_values[flag] = st.checkbox(
                f"{label} (`{flag}`)",
                value=bool(default),
                key=f"ops_{chosen}_{flag}",
            )
        submitted = st.form_submit_button(
            f"▶ Run {spec['label']}",
            disabled=runner.is_running(),
            use_container_width=True,
        )

    if submitted:
        try:
            argv = build_command(chosen, values, flag_values)
        except ValueError as exc:
            st.error(str(exc))
            return
        ok = runner.start(label=spec["label"], cmd_args=argv)
        if not ok:
            st.warning("Another operation is already running. Stop it first.")
        else:
            st.success(f"Launched {spec['label']} ({argv[2]}).")
            # Trigger a rerun so the live transcript starts streaming right away.
            st.rerun()

    # --- status + controls ---
    status_cols = st.columns([2, 1, 1])
    if runner.is_running():
        status_cols[0].info(
            f"⏳ Running **{runner.label}** "
            f"(started {runner.started_at.strftime('%H:%M:%S') if runner.started_at else '—'})"
        )
    else:
        if runner.last_cmd:
            rc = (
                runner.proc.returncode
                if runner.proc is not None else runner.last_exit_code
            )
            status_cols[0].info(
                f"Idle. Last run: **{runner.label}** "
                + (f"(exit {rc})" if rc is not None else "")
            )
        else:
            status_cols[0].info("Idle. No operations launched yet.")
    if status_cols[1].button("⏹ Stop", disabled=not runner.is_running(),
                              key="ops_stop_btn"):
        runner.stop()
        st.rerun()
    if status_cols[2].button("🧹 Clear log", key="ops_clear_btn"):
        runner.clear_transcript()
        st.rerun()

    # --- live transcript ---
    st.text_area(
        "stdout / stderr",
        value=runner.transcript or "(no output yet)",
        height=420,
        disabled=True,
        key=f"ops_transcript_{chosen}",
    )

    # If a job is currently running, schedule a rerun so the transcript keeps
    # updating without the user having to manually refresh.
    if runner.is_running():
        # streamlit-autorefresh would be cleaner but is an extra dep; this is
        # the standard Streamlit pattern for live-stream views.
        st.caption(
            ":hourglass: streaming — Streamlit auto-reruns every few seconds. "
            "Click anywhere or scroll to nudge an immediate refresh."
        )
        # Use the lightweight, dependency-free auto-rerun trick.
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx  # noqa: F401
            import time as _t
            _t.sleep(1.5)
            st.rerun()
        except Exception:  # noqa: BLE001
            pass
