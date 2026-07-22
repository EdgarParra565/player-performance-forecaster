"""Tests for the Fliff iframe-reaching JS extractor + domain rebrand.

Fliff's board renders inside an emulator iframe (``getfliff.com/sports``); the
parent frame is marketing-only. ``extract_iframe_text`` must reach into that
child frame. These use lightweight fake page/frame objects so the frame-walk
logic is verified without a browser.
"""

import unittest

from nba_model.scrapers import get_scraper_for_url
from nba_model.scrapers.fliff import extract_iframe_text


class _FakeLocator:
    def __init__(self, text):
        self._text = text

    def inner_text(self, timeout=None):
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class _FakeFrame:
    def __init__(self, url, body_text=""):
        self.url = url
        self._body_text = body_text

    def locator(self, selector):
        assert selector == "body"
        return _FakeLocator(self._body_text)


class _FakePage:
    def __init__(self, frames):
        self.frames = frames


class ExtractIframeTextTests(unittest.TestCase):
    def test_reaches_board_frame_and_collapses_whitespace(self):
        page = _FakePage([
            _FakeFrame("https://sports.getfliff.com/#/sports", "marketing splash"),
            _FakeFrame(
                "https://sports.getfliff.com/sports?channelId=-333&emulator=t",
                "In Play 23   Offers 2   MLB 12\n NBA  WNBA",
            ),
        ])
        out = extract_iframe_text(page)
        self.assertEqual(out, "In Play 23 Offers 2 MLB 12 NBA WNBA")

    def test_returns_empty_when_board_frame_absent(self):
        page = _FakePage([_FakeFrame("https://sports.getfliff.com/#/sports", "x")])
        self.assertEqual(extract_iframe_text(page), "")

    def test_skips_board_frame_that_errors_and_keeps_going(self):
        page = _FakePage([
            _FakeFrame("https://getfliff.com/sports?a", RuntimeError("detached")),
            _FakeFrame("https://getfliff.com/sports?b", "real board text"),
        ])
        self.assertEqual(extract_iframe_text(page), "real board text")

    def test_handles_page_without_frames(self):
        self.assertEqual(extract_iframe_text(object()), "")


class FliffDomainTests(unittest.TestCase):
    def test_new_and_legacy_domains_resolve(self):
        self.assertEqual(
            get_scraper_for_url("https://sports.getfliff.com/x").name, "fliff")
        self.assertEqual(get_scraper_for_url("https://fliff.com/y").name, "fliff")

    def test_js_extractor_wired(self):
        s = get_scraper_for_url("https://sports.getfliff.com/x")
        self.assertIs(s.js_extractor, extract_iframe_text)


if __name__ == "__main__":
    unittest.main()
