"""Read the most recent daily-ETL report and surface a sidebar widget.

`nba_model.data.daily_etl._write_report` drops a JSON file into
`nba_model/data/artifacts/daily_etl_report_<UTC>.json` on every run. The
widget lets a visitor see:
  - when the data was last refreshed ("4h ago")
  - overall status (success / partial_success / failed)
  - per-step status (game_logs / odds / web_text / browser_parser /
    reverse_engineering)

Pure-Python helpers live here; the Streamlit render lives in `app.py` so
we don't pull `streamlit` into a module that might be imported by the
desktop UI.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Optional

DEFAULT_REPORT_DIR = "nba_model/data/artifacts"


def _latest_report_path(report_dir: str = DEFAULT_REPORT_DIR) -> Optional[str]:
    """Return the absolute path of the most recent ETL report JSON."""
    if not os.path.isdir(report_dir):
        return None
    files = glob.glob(os.path.join(report_dir, "daily_etl_report_*.json"))
    if not files:
        return None
    # Filename embeds a sortable UTC timestamp, so lexical sort == chronological.
    return max(files)


def _parse_utc(value) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _humanize_age(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 0:
        return "in the future"
    if secs < 90:
        return f"{secs}s ago"
    mins = secs // 60
    if mins < 90:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 36:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def load_latest_report(report_dir: str = DEFAULT_REPORT_DIR) -> Optional[dict]:
    """Return the parsed report dict, or None when no report exists / is
    unreadable.  We never raise: a broken report should never break the page."""
    path = _latest_report_path(report_dir)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    data["_path"] = path
    return data


def summarize_report(report: Optional[dict]) -> dict:
    """Reduce the verbose report dict to the fields the sidebar widget shows.

    Returns:
        {
          "found":       bool,
          "status":      'success' | 'partial_success' | 'failed' | 'unknown',
          "finished_at": datetime | None,
          "age_text":    str,
          "elapsed_ms":  int | None,
          "steps":       [(name, status), ...],
          "path":        str | None,
        }
    """
    if not report:
        return {
            "found": False, "status": "unknown", "finished_at": None,
            "age_text": "no ETL report on disk", "elapsed_ms": None,
            "steps": [], "path": None,
        }
    finished = _parse_utc(report.get("finished_at_utc"))
    steps_payload = report.get("steps") or {}
    steps = []
    if isinstance(steps_payload, dict):
        for name, payload in steps_payload.items():
            status = "unknown"
            if isinstance(payload, dict):
                status = str(payload.get("status") or "unknown")
            steps.append((str(name), status))
    return {
        "found": True,
        "status": str(report.get("status") or "unknown"),
        "finished_at": finished,
        "age_text": _humanize_age(finished),
        "elapsed_ms": report.get("elapsed_ms"),
        "steps": steps,
        "path": report.get("_path"),
    }


_STEP_EMOJI = {
    "success": "[OK]",
    "partial_success": "~",
    "failed": "[FAIL]",
    "skipped": "[-]",
    "unknown": "[?]",
}


def step_badge(status: str) -> str:
    return _STEP_EMOJI.get(str(status).lower(), _STEP_EMOJI["unknown"])
