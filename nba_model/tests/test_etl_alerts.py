"""Tests for ETL alerting (WS5d) — pure decision logic + injected webhook."""

import unittest

from nba_model.data import etl_alerts as ea


class BuildAlertTests(unittest.TestCase):
    def test_hourly_ok_run_no_alert(self):
        marker = ea.build_alert({"ok": True, "failed_steps": []})
        self.assertFalse(marker["alert"])
        self.assertEqual(marker["severity"], "ok")

    def test_hourly_failed_steps_triggers_error(self):
        marker = ea.build_alert({"ok": False, "failed_steps": ["web_text"]})
        self.assertTrue(marker["alert"])
        self.assertEqual(marker["severity"], "error")
        self.assertIn("web_text", marker["summary"])

    def test_daily_failed_status(self):
        marker = ea.build_alert({"status": "failed", "steps": {}})
        self.assertTrue(marker["alert"])
        self.assertEqual(marker["severity"], "error")

    def test_daily_partial_status_is_warning(self):
        marker = ea.build_alert({"status": "partial_success", "steps": {}})
        self.assertTrue(marker["alert"])
        self.assertEqual(marker["severity"], "warning")

    def test_step_level_failure_detected(self):
        marker = ea.build_alert({
            "status": "success",  # top-level says ok...
            "steps": {"odds": {"ok": False}},  # ...but a step failed
        })
        self.assertTrue(marker["alert"])
        self.assertEqual(marker["severity"], "error")


class _FakePoster:
    def __init__(self, code=200, raise_exc=False):
        self.code = code
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        if self.raise_exc:
            raise RuntimeError("network down")
        return type("R", (), {"status_code": self.code})()


class MaybeSendAlertTests(unittest.TestCase):
    def test_no_alert_skips_post(self):
        poster = _FakePoster()
        res = ea.maybe_send_alert({"ok": True}, "http://hook", poster=poster)
        self.assertFalse(res["sent"])
        self.assertEqual(res["reason"], "no_alert")
        self.assertEqual(poster.calls, [])

    def test_alert_without_webhook(self):
        res = ea.maybe_send_alert({"ok": False, "failed_steps": ["x"]}, None)
        self.assertFalse(res["sent"])
        self.assertEqual(res["reason"], "no_webhook")

    def test_alert_posts_to_webhook(self):
        poster = _FakePoster(code=204)
        res = ea.maybe_send_alert(
            {"ok": False, "failed_steps": ["web_text"]},
            "http://hook", poster=poster,
        )
        self.assertTrue(res["sent"])
        self.assertEqual(res["status_code"], 204)
        self.assertEqual(len(poster.calls), 1)
        self.assertEqual(poster.calls[0][1]["severity"], "error")

    def test_post_failure_is_swallowed(self):
        poster = _FakePoster(raise_exc=True)
        res = ea.maybe_send_alert(
            {"status": "failed"}, "http://hook", poster=poster,
        )
        self.assertFalse(res["sent"])
        self.assertIn("post_failed", res["reason"])


if __name__ == "__main__":
    unittest.main()
