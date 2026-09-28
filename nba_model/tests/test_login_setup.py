"""login_setup: target selection, passive classification, live checklist loop.

The Chrome glue (open tabs / read text / fresh verify) is exercised live
against :9222; everything decision-making is pinned here with fakes.
"""

import unittest

from nba_model.data import login_setup as ls

URLS = [
    "https://app.prizepicks.com/board/nba",
    "https://underdogsports.com/pick-em/higher-lower/all/NBA",
    "https://underdogsports.com/pick-em/higher-lower/all/NBA?filter_id=x",
    "https://pick6.draftkings.com/?sport=NBA",
    "https://parlayplay.io/",
    "https://sportsbook.draftkings.com/leagues/basketball/nba",
    "https://kalshi.com/markets/nba",
]
PP_WALL = "Log in to PrizePicks. Enter your phone number to continue."


class BuildTargetsTests(unittest.TestCase):
    def test_default_auth_books_use_curated_login_urls(self):
        targets = ls.build_login_targets(URLS)
        by_book = {t.book: t.url for t in targets}
        self.assertEqual(list(by_book), ["prizepicks", "underdog", "pick6", "parlayplay"])
        # Curated login page beats the file's (404) board URL.
        self.assertEqual(by_book["underdog"], ls.LOGIN_URLS["underdog"])
        self.assertNotIn("betrivers", by_book)  # age-blocked (21+), 2026-09-28

    def test_books_flagged_login_needed_by_last_run_are_added(self):
        state = {"books": {"kalshi": {"status": "login-needed"},
                           "draftkings": {"status": "ok"}}}
        targets = {t.book: t.url for t in ls.build_login_targets(URLS, state=state)}
        books = list(targets)
        self.assertIn("kalshi", books)
        self.assertEqual(targets["kalshi"], URLS[6])  # no curated URL → file URL
        self.assertNotIn("draftkings", books)

    def test_only_books_filter(self):
        books = [t.book for t in ls.build_login_targets(URLS, only_books=["Pick6"])]
        self.assertEqual(books, ["pick6"])

    def test_blocked_books_are_never_prompted(self):
        # Even when the last hourly run remembered them as login-needed.
        state = {"books": {"betmgm": {"status": "login-needed"},
                           "kalshi": {"status": "login-needed"}}}
        blocked = {"betmgm": {"kind": "geo-blocked", "note": ""},
                   "underdog": {"kind": "app-only", "note": ""}}
        books = [t.book for t in ls.build_login_targets(URLS, state=state, blocked=blocked)]
        self.assertNotIn("betmgm", books)
        self.assertNotIn("underdog", books)
        self.assertIn("kalshi", books)

    def test_repo_blocked_books_file(self):
        from nba_model.data import session_health
        blocked = session_health.load_blocked_books(session_health.DEFAULT_BLOCKED_BOOKS_FILE)
        self.assertEqual(blocked["betmgm"]["kind"], "geo-blocked")
        self.assertEqual(blocked["betrivers"]["kind"], "age-blocked")


class PollTests(unittest.TestCase):
    def _targets(self):
        return [ls.LoginTarget("prizepicks", URLS[0]),
                ls.LoginTarget("draftkings", URLS[5])]

    def test_polls_until_every_tab_flips_ok(self):
        texts = {"prizepicks": [PP_WALL, PP_WALL, "NBA board " * 300],
                 "draftkings": ["NBA lines " * 300] * 3}
        calls = {"n": 0}

        def read(t):
            return texts[t.book][min(calls["n"], 2)]

        def sleep(_s):
            calls["n"] += 1

        out = []
        ok = ls.poll_until_ok(self._targets(), read, poll_seconds=30,
                              timeout_seconds=600, printer=out.append,
                              sleeper=sleep, clock=lambda: 0.0)
        self.assertTrue(ok)
        self.assertEqual(calls["n"], 2)
        self.assertIn("1/2 books ok", out[0])
        self.assertIn("2/2 books ok", out[-1])
        self.assertEqual(len(out), 2)  # unchanged polls aren't re-printed

    def test_timeout_reports_remaining_books(self):
        now = {"t": 0.0}

        def sleep(s):
            now["t"] += s

        targets = self._targets()
        ok = ls.poll_until_ok(targets, lambda t: PP_WALL if t.book == "prizepicks" else "NBA lines " * 300,
                              poll_seconds=30, timeout_seconds=90, printer=lambda _x: None,
                              sleeper=sleep, clock=lambda: now["t"])
        self.assertFalse(ok)
        self.assertEqual(targets[0].status, "login-needed")
        self.assertEqual(targets[1].status, "ok")

    def test_closed_tab_and_read_errors(self):
        targets = self._targets()

        def read(t):
            if t.book == "prizepicks":
                return None
            raise RuntimeError("target closed")

        ok = ls.poll_until_ok(targets, read, poll_seconds=1, timeout_seconds=0,
                              printer=lambda _x: None, sleeper=lambda _s: None,
                              clock=lambda: 0.0)
        self.assertFalse(ok)
        self.assertEqual(targets[0].status, ls.TAB_CLOSED)
        self.assertEqual(targets[1].status, "unreachable")

    def test_empty_page_is_not_ok(self):
        status, reason = ls.classify_text(ls.LoginTarget("prizepicks", URLS[0]), "  ")
        self.assertEqual(status, "login-needed")
        self.assertIn("empty", reason)


if __name__ == "__main__":
    unittest.main()
