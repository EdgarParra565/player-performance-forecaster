"""Review finding (adversarial): the PP-style player-name regexes were
super-linear on one long unspaced token — an 8,000-char 'A-A-…' token took
>6 minutes in extract_prop_cards_from_text. Name words are now anchored to a
token start and consumed possessively. Matches are unchanged (verified offline
on all 286 stored snapshots, 2026-10-10: 503 cards before and after, 0 diffs;
no live re-check — scraping is off)."""

import re
import time
import unittest

from nba_model.model import browser_prop_parser as bp
from nba_model.scrapers import prizepicks, underdog
from nba_model.scrapers.base import build_pp_style_name_pattern

LONG = "A-" * 30000                       # the finding's token
LONG_MIXED = "Ab" * 30000                 # lowercase present everywhere


class LinearTimeTests(unittest.TestCase):
    """Scope: the PP-style name regex (base + PrizePicks/Underdog/Fliff) and
    the generic card patterns. Other books' preprocessors (ParlayPlay, Pick6,
    mlb_props, nfl_props) carry their own name regexes and are still
    super-linear — recorded in review_findings_data.md, not fixed here."""

    def _fast(self, fn, budget=2.0):
        t = time.perf_counter()
        fn()
        self.assertLess(time.perf_counter() - t, budget)

    def test_generic_card_patterns_on_long_unspaced_tokens(self):
        for tok in (LONG, LONG_MIXED, f"Jayson Tatum {LONG} More 25.5 Points"):
            for pattern in bp._CARD_PATTERNS:
                self._fast(lambda: list(pattern.finditer(tok)))

    def test_book_patterns_on_long_unspaced_tokens(self):
        pp = re.compile(build_pp_style_name_pattern())
        for tok in (LONG, LONG_MIXED):
            self._fast(lambda: list(pp.finditer(tok)))
            self._fast(lambda: prizepicks.SCRAPER.prop_preprocess(tok))
            self._fast(lambda: list(underdog._PROP_RE.finditer(tok)))


class SameMatchesTests(unittest.TestCase):
    def test_names_still_match_like_before(self):
        pat = re.compile(build_pp_style_name_pattern())
        for text, want in (
            ("Karl-Anthony Towns @ PHI", "Karl-Anthony Towns"),
            ("De'Aaron Fox vs SAS", "De'Aaron Fox"),
            ("Jaren Jackson Jr. MEM", "Jaren Jackson Jr."),
        ):
            self.assertEqual(pat.search(text).group("player"), want)
        self.assertIsNone(pat.search("NYK LAL BOS"))  # all-caps tokens excluded

    def test_generic_card_still_parsed(self):
        cards = bp.extract_prop_cards_from_text(
            "LeBron James More 25.5 Points", "https://example.com", 1, "x", set())
        self.assertEqual([c["player_name"] for c in cards], ["LeBron James"])

    def test_name_cannot_start_mid_token(self):
        pat = re.compile(build_pp_style_name_pattern())
        # Old pattern could start at the 'J' inside 'xJohn'; a name word now
        # starts only at a token boundary.
        self.assertIsNone(pat.search("xJohn"))
        self.assertEqual(pat.search("x John Smith").group("player"), "John Smith")


if __name__ == "__main__":
    unittest.main()
