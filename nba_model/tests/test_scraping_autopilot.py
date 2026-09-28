"""Scraping autopilot: CDP tab hygiene, per-book session health + debounced
re-login alerts, and the launchd scraping-Chrome launcher.

Everything is verified with mocks / fake binaries — no live Chrome needed.
"""

import json
import os
import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nba_model.data import etl_alerts, hourly_update, session_health
from nba_model.model import web_text_ingestion as wti

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHED_DIR = REPO_ROOT / "scripts" / "scheduler"

DK_LOBBY = "https://sportsbook.draftkings.com/leagues/basketball/nba"
PP_BOARD = "https://app.prizepicks.com/board/nba"
PP_WALL_TEXT = "Log in to PrizePicks. Enter your phone number to continue."
DK_CONTENT = "NBA lines " * 200


class TabMatchTests(unittest.TestCase):
    def test_same_host_different_path_matches(self):
        # Reuse keys on DOMAIN, so hygiene must too — not exact URL.
        self.assertTrue(wti._tab_matches_target(
            "https://sportsbook.draftkings.com/leagues/baseball/mlb?x=1", DK_LOBBY))

    def test_www_and_case_variants_match(self):
        self.assertTrue(wti._tab_matches_target(
            "https://VegasInsider.com/nba/odds/", "https://www.vegasinsider.com/nba/odds/player-props/"))

    def test_other_host_does_not_match(self):
        self.assertFalse(wti._tab_matches_target("https://mail.google.com/", DK_LOBBY))
        self.assertFalse(wti._tab_matches_target("", DK_LOBBY))
        # A sibling subdomain is a different host (pick6 != sportsbook).
        self.assertFalse(wti._tab_matches_target("https://pick6.draftkings.com/", DK_LOBBY))


class CloseTargetTabsTests(unittest.TestCase):
    def _fake_cdp(self, tabs, calls, fail_on=None):
        def _call(url, method="GET", timeout_seconds=3.0):
            calls.append((method, url))
            if fail_on and fail_on in url:
                raise OSError("boom")
            if url.endswith("/json/list"):
                return tabs
            return "Target is closing"
        return _call

    def test_closes_every_tab_on_target_host_only(self):
        tabs = [
            {"id": "A", "type": "page", "url": DK_LOBBY},
            {"id": "B", "type": "page", "url": "https://sportsbook.draftkings.com/leagues/baseball/mlb"},
            {"id": "C", "type": "page", "url": "https://mail.google.com/"},
            {"id": "D", "type": "service_worker", "url": DK_LOBBY + "/sw.js"},
        ]
        calls = []
        with patch.object(wti, "_cdp_http", side_effect=self._fake_cdp(tabs, calls)):
            out = wti.close_target_tabs_via_cdp([DK_LOBBY], 9222)
        self.assertTrue(out["ok"])
        self.assertEqual(sorted(c["id"] for c in out["closed"]), ["A", "B"])
        self.assertFalse(out["opened_blank"])
        closed_urls = [u for _, u in calls if "/json/close/" in u]
        self.assertEqual(len(closed_urls), 2)
        self.assertFalse(any(u.endswith("/C") or u.endswith("/D") for u in closed_urls))

    def test_opens_blank_tab_before_closing_the_last_pages(self):
        tabs = [{"id": "A", "type": "page", "url": PP_BOARD}]
        calls = []
        with patch.object(wti, "_cdp_http", side_effect=self._fake_cdp(tabs, calls)):
            out = wti.close_target_tabs_via_cdp([PP_BOARD], 9222)
        self.assertTrue(out["opened_blank"])
        methods = [m for m, u in calls if "/json/new" in u]
        self.assertEqual(methods, ["PUT"])
        new_idx = next(i for i, (_, u) in enumerate(calls) if "/json/new" in u)
        close_idx = next(i for i, (_, u) in enumerate(calls) if "/json/close/" in u)
        self.assertLess(new_idx, close_idx)

    def test_list_failure_is_reported_not_raised(self):
        calls = []
        with patch.object(wti, "_cdp_http", side_effect=self._fake_cdp([], calls, fail_on="/json/list")):
            out = wti.close_target_tabs_via_cdp([DK_LOBBY], 9222)
        self.assertFalse(out["ok"])
        self.assertTrue(out["errors"])

    def test_nothing_to_close(self):
        tabs = [{"id": "C", "type": "page", "url": "https://mail.google.com/"}]
        with patch.object(wti, "_cdp_http", side_effect=self._fake_cdp(tabs, [])):
            out = wti.close_target_tabs_via_cdp([DK_LOBBY], 9222)
        self.assertEqual(out["closed"], [])
        self.assertTrue(out["ok"])


def _record(url, text):
    return {
        "source_url": url, "fetched_at_utc": "2026-09-28T00:00:00+00:00",
        "http_status": 200, "content_type": None, "text_content": text,
        "text_length": len(text), "content_sha256": str(hash(text)),
    }


class FetchWithHygieneTests(unittest.TestCase):
    def _run(self, **kwargs):
        order = []

        def _hygiene(urls, port, host="127.0.0.1"):
            order.append("hygiene")
            return {"ok": True, "tabs_seen": 2, "closed": [{"id": "A", "url": DK_LOBBY}],
                    "errors": [], "opened_blank": False}

        def _fetch(url, **_kw):
            order.append(f"fetch:{url}")
            return _record(url, PP_WALL_TEXT if "prizepicks" in url else DK_CONTENT)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(wti, "close_target_tabs_via_cdp", side_effect=_hygiene) as hyg, \
                patch.object(wti, "_fetch_url_text", side_effect=_fetch):
            summary = wti.fetch_and_store_web_text(
                urls=[DK_LOBBY, PP_BOARD], db_path=str(Path(tmp) / "t.db"),
                min_hours_between_polls=None, force_poll=True, **kwargs,
            )
        return summary, order, hyg

    def test_hygiene_runs_before_any_fetch_and_lands_in_summary(self):
        summary, order, hyg = self._run(chrome_debug_port=9222, close_stale_tabs=True)
        self.assertEqual(order[0], "hygiene")
        hyg.assert_called_once()
        self.assertEqual(hyg.call_args.args[0], [DK_LOBBY, PP_BOARD])
        self.assertEqual(summary["tab_hygiene"]["closed"][0]["id"], "A")

    def test_hygiene_off_by_default_and_without_cdp(self):
        _, _, hyg = self._run(chrome_debug_port=9222)
        hyg.assert_not_called()
        _, _, hyg = self._run(close_stale_tabs=True)  # requests path: no CDP
        hyg.assert_not_called()

    def test_results_carry_session_classification(self):
        summary, _, _ = self._run(chrome_debug_port=9222, close_stale_tabs=True)
        by_url = {r["url"]: r for r in summary["results"]}
        self.assertEqual(by_url[PP_BOARD]["book"], "prizepicks")
        self.assertEqual(by_url[PP_BOARD]["session"], "login_wall")
        self.assertEqual(by_url[DK_LOBBY]["book"], "draftkings")
        self.assertEqual(by_url[DK_LOBBY]["session"], "content")

    def test_hourly_web_text_turns_hygiene_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            urls = Path(tmp) / "urls.txt"
            urls.write_text(DK_LOBBY + "\n", encoding="utf-8")
            with patch.object(wti, "fetch_and_store_web_text",
                              return_value={"status": "success"}) as fetch:
                hourly_update._run_web_text("x.db", str(urls), 9222, None)
        self.assertTrue(fetch.call_args.kwargs["close_stale_tabs"])
        self.assertEqual(fetch.call_args.kwargs["chrome_debug_port"], 9222)


def _results(**sessions):
    """{'prizepicks': 'login_wall'|'content'|'failed'} → fetch result rows."""
    urls = {"prizepicks": PP_BOARD, "draftkings": DK_LOBBY}
    rows = []
    for book, s in sessions.items():
        if s == "failed":
            rows.append({"url": urls[book], "status": "failed", "book": book,
                         "error_message": "timeout"})
        else:
            rows.append({"url": urls[book], "status": "fetched", "book": book,
                         "session": s, "session_reason": None})
    return rows


class ClassifyBooksTests(unittest.TestCase):
    def test_precedence_login_needed_over_ok_over_unreachable(self):
        rows = [
            {"url": "https://underdogsports.com/a", "status": "fetched", "book": "underdog", "session": "content"},
            {"url": "https://underdogsports.com/b", "status": "fetched", "book": "underdog", "session": "login_wall"},
            {"url": "https://kalshi.com/x", "status": "failed", "book": "kalshi"},
            {"url": "https://kalshi.com/y", "status": "fetched", "book": "kalshi", "session": "content"},
            {"url": "https://bovada.lv/x", "status": "failed", "book": "bovada"},
            {"url": "https://skip.example/x", "status": "skipped_recent"},
        ]
        books = session_health.classify_books(rows)
        self.assertEqual(books["underdog"]["status"], "login-needed")
        self.assertEqual(books["kalshi"]["status"], "ok")
        self.assertEqual(books["bovada"]["status"], "unreachable")
        self.assertEqual(len(books), 3)

    def test_http_error_page_is_unreachable_not_ok(self):
        # Live 2026-09-28: Underdog's moved URLs served a 404 whose nav passed
        # the content-marker check.
        rows = [{"url": "https://underdogsports.com/pick-em/x", "status": "fetched",
                 "book": "underdog", "session": "content", "http_status": 404}]
        books = session_health.classify_books(rows)
        self.assertEqual(books["underdog"]["status"], "unreachable")
        self.assertIn("HTTP 404", books["underdog"]["urls"][0]["reason"])

    def test_generic_nav_wall_on_public_book_is_unreachable(self):
        rows = [
            {"url": "https://kalshi.com/markets/nba", "status": "fetched", "book": "kalshi",
             "session": "login_wall", "http_status": 200,
             "session_reason": "Generic login nav 'log in' present and only 2/3 authenticated content markers found"},
            {"url": "https://pick6.draftkings.com/", "status": "fetched", "book": "pick6",
             "session": "login_wall", "http_status": 200,
             "session_reason": "Generic login nav 'log in' present and only 2/4 authenticated content markers found"},
            {"url": "https://sports.betmgm.com/x", "status": "fetched", "book": "betmgm",
             "session": "login_wall", "http_status": 200,
             "session_reason": "Login-wall phrase detected: 'where are you playing from?'"},
        ]
        books = session_health.classify_books(rows)
        self.assertEqual(books["kalshi"]["status"], "unreachable")   # public: didn't render
        self.assertIn("didn't render", books["kalshi"]["urls"][0]["reason"])
        self.assertEqual(books["pick6"]["status"], "login-needed")   # auth book
        self.assertEqual(books["betmgm"]["status"], "login-needed")  # specific phrase

    def test_unreachable_urls_when_step_raised(self):
        books = session_health.classify_books(None, unreachable_urls=[PP_BOARD, DK_LOBBY])
        self.assertEqual({b: i["status"] for b, i in books.items()},
                         {"prizepicks": "unreachable", "draftkings": "unreachable"})


class DebounceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state = str(Path(self._tmp.name) / "state.json")
        self.notify = MagicMock(return_value={"delivered": True, "retry": False})

    def tearDown(self):
        self._tmp.cleanup()

    def tick(self, **sessions):
        return session_health.run_session_health(
            _results(**sessions), self.state, notify=self.notify)

    def test_alerts_once_per_login_episode(self):
        r1 = self.tick(prizepicks="login_wall", draftkings="content")
        self.assertEqual(r1["new_login_needed"], ["prizepicks"])
        self.assertEqual(r1["books"], {"draftkings": "ok", "prizepicks": "login-needed"})
        self.assertEqual(self.notify.call_count, 1)
        self.assertEqual(self.notify.call_args.args[0], ["prizepicks"])

        r2 = self.tick(prizepicks="login_wall", draftkings="content")
        self.assertEqual(r2["new_login_needed"], [])
        r3 = self.tick(prizepicks="failed", draftkings="content")  # unreachable blip
        self.assertEqual(r3["books"]["prizepicks"], "unreachable")
        r4 = self.tick(prizepicks="login_wall", draftkings="content")
        self.assertEqual(r4["new_login_needed"], [])
        self.assertEqual(self.notify.call_count, 1)

        r5 = self.tick(prizepicks="content", draftkings="content")  # re-logged in
        self.assertEqual(r5["recovered"], ["prizepicks"])
        r6 = self.tick(prizepicks="login_wall", draftkings="content")  # new episode
        self.assertEqual(r6["new_login_needed"], ["prizepicks"])
        self.assertEqual(self.notify.call_count, 2)

    def test_failed_delivery_retries_next_tick(self):
        self.notify.return_value = {"delivered": False, "retry": True}
        self.tick(prizepicks="login_wall")
        self.notify.return_value = {"delivered": True, "retry": False}
        r2 = self.tick(prizepicks="login_wall")
        self.assertEqual(r2["new_login_needed"], ["prizepicks"])
        self.assertEqual(self.notify.call_count, 2)
        self.tick(prizepicks="login_wall")
        self.assertEqual(self.notify.call_count, 2)

    def test_notify_exception_is_contained_and_retried(self):
        self.notify.side_effect = RuntimeError("webhook down")
        r1 = self.tick(prizepicks="login_wall")
        self.assertTrue(r1["notification"]["retry"])
        self.notify.side_effect = None
        self.notify.return_value = {"delivered": True, "retry": False}
        self.assertEqual(self.tick(prizepicks="login_wall")["new_login_needed"], ["prizepicks"])

    def test_state_persists_and_corrupt_state_is_fresh(self):
        self.tick(prizepicks="login_wall")
        saved = json.loads(Path(self.state).read_text())
        self.assertTrue(saved["books"]["prizepicks"]["login_alerted"])
        Path(self.state).write_text("{not json", encoding="utf-8")
        self.assertEqual(session_health.load_state(self.state)["books"], {})

    def test_blocked_book_reports_blocked_and_never_alerts(self):
        blocked = {"prizepicks": {"kind": "geo-blocked", "note": "CA"}}
        for _ in range(3):
            r = session_health.run_session_health(
                _results(prizepicks="login_wall", draftkings="content"),
                self.state, notify=self.notify, blocked=blocked)
            self.assertEqual(r["books"]["prizepicks"], "blocked")
            self.assertEqual(r["new_login_needed"], [])
        self.notify.assert_not_called()
        self.assertEqual(r["counts"]["blocked"], 1)
        # Block lifted (page reads real content) → ok + flagged, still no alert.
        r = session_health.run_session_health(
            _results(prizepicks="content"), self.state, notify=self.notify, blocked=blocked)
        self.assertEqual(r["books"]["prizepicks"], "ok")
        self.assertEqual(r["blocked_but_ok"], ["prizepicks"])
        self.notify.assert_not_called()

    def test_step_status_is_always_success(self):
        # A degraded status would re-alert hourly via build_alert — the
        # debounced session alert is the only login-needed signal.
        r = self.tick(prizepicks="login_wall")
        self.assertEqual(r["status"], "success")


class SessionAlertChannelTests(unittest.TestCase):
    def test_webhook_non_2xx_counts_as_not_sent(self):
        poster = MagicMock(return_value=MagicMock(status_code=500))
        out = etl_alerts.send_session_alert(["prizepicks"], "https://hook", poster=poster)
        self.assertFalse(out["sent"])
        self.assertEqual(out["reason"], "http_500")
        payload = poster.call_args.kwargs["json"]
        self.assertEqual(payload["kind"], "session")
        self.assertEqual(payload["books"], ["prizepicks"])

    def test_webhook_2xx_and_no_webhook(self):
        poster = MagicMock(return_value=MagicMock(status_code=204))
        self.assertTrue(etl_alerts.send_session_alert(["x"], "https://hook", poster=poster)["sent"])
        self.assertEqual(etl_alerts.send_session_alert(["x"], None)["reason"], "no_webhook")

    def test_macos_notification_escapes_quotes(self):
        runner = MagicMock(return_value=MagicMock(returncode=0))
        out = etl_alerts.notify_macos('T "q"', 'm "q"', runner=runner)
        self.assertTrue(out["sent"])
        argv = runner.call_args.args[0]
        self.assertEqual(argv[:2], ["osascript", "-e"])
        self.assertIn('\\"q\\"', argv[2])

    def test_notifier_retry_only_when_all_configured_channels_fail(self):
        with patch.object(etl_alerts, "send_session_alert",
                          return_value={"sent": False, "reason": "http_500"}):
            out = hourly_update._session_notifier("https://hook", False)(["pp"], {})
        self.assertTrue(out["retry"])
        with patch.object(etl_alerts, "send_session_alert",
                          return_value={"sent": False, "reason": "http_500"}), \
                patch.object(etl_alerts, "notify_macos", return_value={"sent": True}):
            out = hourly_update._session_notifier("https://hook", True)(["pp"], {})
        self.assertFalse(out["retry"])
        self.assertTrue(out["delivered"])
        out = hourly_update._session_notifier(None, False)(["pp"], {})
        self.assertFalse(out["retry"])  # no channel → recorded, not retried hourly


class HourlySessionHealthIntegrationTests(unittest.TestCase):
    def _run(self, tmp, web_text):
        stub = {
            "_run_preflight": {"playwright_available": True, "chrome": {"ok": True}},
            "_run_browser_prop_parser": {}, "_run_team_line_parser": {},
            "_run_vegasinsider_ingestion": {}, "_run_vegasinsider_mlb_props_ingestion": {},
            "_run_game_log_refresh": {"players_refreshed": 0, "failures": []},
            "_run_players_table_sync": {}, "_run_reverse_engineering": {},
            "_run_outcome_settlement": {}, "_run_prediction_recompute": {"scored": 0},
        }
        patches = [patch.object(hourly_update, k, return_value=v) for k, v in stub.items()]
        patches.append(patch.object(hourly_update, "_run_web_text", **web_text))
        urls = Path(tmp) / "urls.txt"
        urls.write_text(f"{PP_BOARD}\n{DK_LOBBY}\n", encoding="utf-8")
        for p in patches:
            p.start()
        try:
            return hourly_update.run_hourly_update(
                report_dir=tmp, urls_file=str(urls), require_playwright=False,
                alert_webhook_url="https://hook",
            )
        finally:
            for p in patches:
                p.stop()

    def test_login_needed_reported_per_book_and_alerted_once(self):
        web_text = {"return_value": {"status": "success",
                                     "results": _results(prizepicks="login_wall",
                                                         draftkings="content")}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                etl_alerts, "send_session_alert",
                return_value={"sent": True, "status_code": 200}) as send:
            r1 = self._run(tmp, web_text)
            r2 = self._run(tmp, web_text)
        self.assertTrue(r1["ok"])
        self.assertEqual(r1["exit_code"], hourly_update.EXIT_OK)
        self.assertEqual(r1["session_health"],
                         {"draftkings": "ok", "prizepicks": "login-needed"})
        self.assertEqual(r1["steps"]["session_health"]["result"]["new_login_needed"], ["prizepicks"])
        self.assertEqual(r2["steps"]["session_health"]["result"]["new_login_needed"], [])
        send.assert_called_once()
        self.assertEqual(send.call_args.args[0], ["prizepicks"])
        # Login-needed doesn't degrade the run's own alert marker.
        self.assertEqual(r1["alert"]["severity"], "ok")

    def test_web_text_exception_marks_books_unreachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run(tmp, {"side_effect": RuntimeError("chrome dropped")})
        self.assertEqual(report["failed_steps"], ["web_text"])
        self.assertTrue(report["steps"]["session_health"]["ok"])
        self.assertEqual(report["session_health"],
                         {"draftkings": "unreachable", "prizepicks": "unreachable"})

    def test_cli_flags(self):
        args = hourly_update._build_parser().parse_args([])
        self.assertFalse(args.no_close_stale_tabs)
        self.assertFalse(args.no_macos_notify)
        with patch.dict(os.environ, {"NBA_ALERT_WEBHOOK_URL": "https://env-hook"}):
            args = hourly_update._build_parser().parse_args([])
        self.assertEqual(args.alert_webhook_url, "https://env-hook")


class LaunchdArtifactsTests(unittest.TestCase):
    def test_scraping_chrome_plist_keeps_alive(self):
        plist = plistlib.loads((SCHED_DIR / "com.nba.scraping-chrome.plist").read_bytes())
        self.assertEqual(plist["Label"], "com.nba.scraping-chrome")
        self.assertIs(plist["KeepAlive"], True)
        self.assertIs(plist["RunAtLoad"], True)
        self.assertTrue(plist["ProgramArguments"][0].endswith("scripts/scheduler/scraping_chrome.sh"))
        self.assertGreaterEqual(plist["ThrottleInterval"], 10)

    def test_hourly_plist_uses_wall_clock_schedule(self):
        # StartInterval counts awake time only → never fired on the idle-
        # sleeping host (live 2026-09-28). Calendar fires coalesce on wake.
        plist = plistlib.loads((SCHED_DIR / "com.nba.hourly.plist").read_bytes())
        self.assertNotIn("StartInterval", plist)
        self.assertEqual(plist["StartCalendarInterval"], {"Minute": 5})

    def test_keep_awake_plist(self):
        plist = plistlib.loads((SCHED_DIR / "com.nba.keep-awake.plist").read_bytes())
        self.assertEqual(plist["ProgramArguments"], ["/usr/bin/caffeinate", "-s"])
        self.assertIs(plist["KeepAlive"], True)

    def _run_launcher(self, env_extra):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "argv.txt"
            fake = Path(tmp) / "fake_chrome"
            fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{out}"\n', encoding="utf-8")
            fake.chmod(0o755)
            env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp,
                   "CHROME_BIN": str(fake), "CHROME_PORT": "1"}
            env.update({k: v.replace("$TMP", tmp) for k, v in env_extra.items()})
            proc = subprocess.run(["bash", str(SCHED_DIR / "scraping_chrome.sh")],
                                  env=env, capture_output=True, text=True, timeout=30)
            argv = out.read_text().splitlines() if out.exists() else []
            return proc, argv, tmp

    def test_launcher_execs_chrome_with_dedicated_profile(self):
        proc, argv, tmp = self._run_launcher({})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--remote-debugging-port=1", argv)
        profile = next(a for a in argv if a.startswith("--user-data-dir="))
        self.assertEqual(profile, f"--user-data-dir={tmp}/Library/Application Support/nba-scraping-chrome")
        self.assertNotIn("Google/Chrome", profile)

    def test_launcher_refuses_daily_profile(self):
        proc, argv, _ = self._run_launcher(
            {"NBA_SCRAPER_PROFILE_DIR": "$TMP/Library/Application Support/Google/Chrome/Default"})
        self.assertEqual(proc.returncode, 78)
        self.assertEqual(argv, [])
        self.assertIn("refusing", proc.stderr)

    def test_hourly_wrapper_no_longer_gates_python_preflight(self):
        text = (SCHED_DIR / "hourly_update.sh").read_text()
        curl_block = text.split("curl", 1)[1].split("fi", 1)[0]
        self.assertNotIn("exit 78", curl_block)
        self.assertIn('--chrome-port "${CHROME_PORT}"', text)


if __name__ == "__main__":
    unittest.main()


class SnapshotSelectionTests(unittest.TestCase):
    """Live 2026-09-28: the parsers' snapshot cap sorted by URL, so once stale
    URLs pushed the count past 20 every alphabetically-late URL (sportsbook.*,
    www.bovada, www.vegasinsider) was silently never parsed."""

    def test_cap_drops_oldest_not_alphabetically_late_urls(self):
        from nba_model.data.database.db_manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmp:
            with DatabaseManager(str(Path(tmp) / "t.db")) as db:
                rows = [(f"https://a{i:02d}.example/", f"2026-05-0{1 + i % 9}T00:00:00+00:00")
                        for i in range(25)]
                rows.append(("https://www.vegasinsider.com/nba/", "2026-09-28T17:00:00+00:00"))
                db.insert_web_text_snapshots([
                    {"source_url": u, "fetched_at_utc": ts, "http_status": 200,
                     "text_content": f"text {u}", "text_length": 10,
                     "content_sha256": f"h{u}"} for u, ts in rows])
                snaps = db.get_recent_web_text_snapshots(limit_total=20)
        urls = [s["source_url"] for s in snaps]
        self.assertEqual(len(urls), 20)
        self.assertEqual(urls[0], "https://www.vegasinsider.com/nba/")

    def test_zero_extraction_is_success_on_hourly_path(self):
        out = hourly_update._zero_extraction_is_success(
            {"status": "partial_success", "cards_extracted": 0}, "cards_extracted")
        self.assertEqual(out["status"], "success")
        self.assertTrue(out["zero_extraction"])
        self.assertEqual(out["parser_status"], "partial_success")
        kept = hourly_update._zero_extraction_is_success(
            {"status": "failed", "cards_extracted": 0}, "cards_extracted")
        self.assertEqual(kept["status"], "failed")

    def test_hourly_parsers_read_only_configured_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            urls = Path(tmp) / "urls.txt"
            urls.write_text(f"{DK_LOBBY}\n{PP_BOARD}\n", encoding="utf-8")
            with patch("nba_model.model.team_line_parser.parse_and_store_web_team_lines",
                       return_value={}) as tl, \
                    patch("nba_model.model.browser_prop_parser.parse_and_store_web_prop_cards",
                          return_value={}) as bp:
                hourly_update._run_team_line_parser("x.db", str(urls))
                hourly_update._run_browser_prop_parser("x.db", str(urls))
        self.assertEqual(tl.call_args.kwargs["source_urls"], [DK_LOBBY, PP_BOARD])
        self.assertEqual(bp.call_args.kwargs["source_urls"], [DK_LOBBY, PP_BOARD])
