"""ETL alerting helpers (WS5d).

Two pieces, kept separate so the decision logic is unit-testable without any
network:

- ``build_alert(report)`` — pure: inspects a daily/hourly ETL report dict and
  returns an alert marker ``{"alert", "severity", "summary", "failed_steps"}``.
  Embed it in the report JSON so a future notifier can consume it even when no
  webhook is configured.
- ``maybe_send_alert(report, webhook_url, poster=...)`` — best-effort POST of
  the marker to a webhook (Slack/Discord/generic) when an alert fires. Never
  raises; returns a small status dict.

The report shapes differ between daily_etl and hourly_update, so ``build_alert``
is tolerant: it reads ``ok`` / ``status`` / ``failed_steps`` and also scans a
``steps`` mapping for per-step failure/partial markers.
"""
from __future__ import annotations

from typing import Callable, Optional

_FAILED_STATES = {"failed", "error"}
_PARTIAL_STATES = {"partial", "partial_success", "degraded"}


def _step_states(report: dict) -> list[str]:
    states: list[str] = []
    steps = report.get("steps")
    if isinstance(steps, dict):
        for v in steps.values():
            if isinstance(v, dict):
                if v.get("ok") is False:
                    states.append("failed")
                status = str(v.get("status") or "").lower()
                if status:
                    states.append(status)
                # Hourly steps nest the step's own summary under "result";
                # a step that returned status 'failed' must still alert.
                result = v.get("result")
                if isinstance(result, dict):
                    nested = str(result.get("status") or "").lower()
                    if nested:
                        states.append(nested)
    return states


def build_alert(report: dict) -> dict:
    """Return an alert marker for an ETL report. ``alert`` is True when the
    run failed or only partially succeeded."""
    report = report or {}
    failed_steps = list(report.get("failed_steps") or [])
    status = str(report.get("status") or "").lower()
    ok = report.get("ok")
    step_states = _step_states(report)

    is_failed = (
        ok is False
        or status in _FAILED_STATES
        or bool(failed_steps)
        or any(s in _FAILED_STATES for s in step_states)
    )
    is_partial = (
        status in _PARTIAL_STATES
        or any(s in _PARTIAL_STATES for s in step_states)
    )

    if is_failed:
        severity = "error"
    elif is_partial:
        severity = "warning"
    else:
        severity = "ok"

    alert = severity in {"error", "warning"}
    label = report.get("report_path") or report.get("started_at") or "ETL run"
    if severity == "error":
        summary = f"ETL FAILED: {label}"
        if failed_steps:
            summary += f" — failed steps: {', '.join(map(str, failed_steps))}"
    elif severity == "warning":
        summary = f"ETL partial: {label}"
    else:
        summary = f"ETL ok: {label}"

    return {
        "alert": bool(alert),
        "severity": severity,
        "summary": summary,
        "failed_steps": failed_steps,
    }


def maybe_send_alert(
    report: dict,
    webhook_url: Optional[str],
    *,
    poster: Optional[Callable] = None,
) -> dict:
    """Post the alert marker to ``webhook_url`` when an alert fires.

    Returns ``{"sent", "reason"|"status_code", "alert": <marker>}``. Never
    raises — alerting must not take down the ETL. ``poster`` is injected for
    tests (defaults to ``requests.post``).
    """
    marker = build_alert(report)
    if not marker["alert"]:
        return {"sent": False, "reason": "no_alert", "alert": marker}
    if not webhook_url:
        return {"sent": False, "reason": "no_webhook", "alert": marker}

    payload = {
        "text": marker["summary"],
        "severity": marker["severity"],
        "failed_steps": marker["failed_steps"],
    }
    try:
        if poster is None:
            import requests
            poster = requests.post
        resp = poster(webhook_url, json=payload, timeout=10)
        code = getattr(resp, "status_code", None)
        return {"sent": True, "status_code": code, "alert": marker}
    except Exception as exc:  # noqa: BLE001 — alerting is best-effort
        return {"sent": False, "reason": f"post_failed: {exc}", "alert": marker}


def send_session_alert(
    books: list[str],
    webhook_url: Optional[str],
    *,
    poster: Optional[Callable] = None,
) -> dict:
    """POST a "book re-login needed" alert for ``books`` to ``webhook_url``.

    Returns ``{"sent", "status_code"|"reason"}``; a non-2xx response counts as
    NOT sent so the caller's debounce can retry next tick. Never raises.
    """
    if not books:
        return {"sent": False, "reason": "no_books"}
    if not webhook_url:
        return {"sent": False, "reason": "no_webhook"}
    payload = {
        "text": (
            "Scraper re-login needed: " + ", ".join(books)
            + " — sign in again in the scraping Chrome (:9222 profile)."
        ),
        "severity": "warning",
        "kind": "session",
        "books": list(books),
    }
    try:
        if poster is None:
            import requests
            poster = requests.post
        resp = poster(webhook_url, json=payload, timeout=10)
    except Exception as exc:  # noqa: BLE001 — alerting is best-effort
        return {"sent": False, "reason": f"post_failed: {exc}"}
    code = getattr(resp, "status_code", None)
    if isinstance(code, int) and not 200 <= code < 300:
        return {"sent": False, "reason": f"http_{code}", "status_code": code}
    return {"sent": True, "status_code": code}


def notify_macos(title: str, message: str, *, runner: Optional[Callable] = None) -> dict:
    """Best-effort macOS Notification Center banner via ``osascript``.

    No-op off macOS. ``runner`` is injected for tests (defaults to
    ``subprocess.run``). Never raises.
    """
    import sys

    if runner is None:
        if sys.platform != "darwin":
            return {"sent": False, "reason": "not_macos"}
        import subprocess
        runner = subprocess.run

    def _q(text: str) -> str:
        return str(text).replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{_q(message)}" with title "{_q(title)}"'
    try:
        proc = runner(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception as exc:  # noqa: BLE001
        return {"sent": False, "reason": f"osascript_failed: {exc}"}
    rc = getattr(proc, "returncode", 0)
    return {"sent": rc == 0, "returncode": rc}
