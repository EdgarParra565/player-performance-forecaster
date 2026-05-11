"""Tests for the shared player-name resolver used by every scraper."""

import unittest

from nba_model.scrapers.player_names import (
    normalize_name_key,
    resolve_player_name,
)


# Realistic NBA active-players sample (canonical full names).
SAMPLE = [
    "Jalen Brunson",
    "Victor Wembanyama",
    "Karl-Anthony Towns",
    "Tyrese Maxey",
    "Joel Embiid",
    "Paul George",
    "Stephen Curry",
    "Tim Hardaway Jr.",
    "CJ McCollum",
    "Devin Booker",
    "Buddy Hield",     # another Hield-suffix-ish for ambiguity tests
    "Cam Reddish",
    "Cole Anthony",
    # Two-player same-surname so ambiguity tests have teeth:
    "Jaylen Brown",
    "Jaylen Wells",
    "Mikal Bridges",
    "Miles Bridges",
]


class NormalizeKeyTests(unittest.TestCase):
    def test_strips_punctuation_and_lowercases(self):
        self.assertEqual(normalize_name_key("C.J. McCollum"), "cjmccollum")
        self.assertEqual(normalize_name_key("Karl-Anthony Towns"), "karlanthonytowns")
        self.assertEqual(normalize_name_key("Tim Hardaway Jr."), "timhardawayjr")
        self.assertEqual(normalize_name_key(""), "")
        self.assertEqual(normalize_name_key(None), "")


class ExactAndSuffixTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(resolve_player_name("Jalen Brunson", SAMPLE), "Jalen Brunson")

    def test_case_insensitive(self):
        self.assertEqual(resolve_player_name("jalen brunson", SAMPLE), "Jalen Brunson")
        self.assertEqual(resolve_player_name("JALEN BRUNSON", SAMPLE), "Jalen Brunson")

    def test_punctuation_collapse(self):
        # "C.J." in input but stored as "CJ" in the canonical list — both
        # resolve to the same row.
        self.assertEqual(resolve_player_name("C.J. McCollum", SAMPLE), "CJ McCollum")
        self.assertEqual(resolve_player_name("CJ McCollum", SAMPLE), "CJ McCollum")

    def test_suffix_dot_optional(self):
        self.assertEqual(
            resolve_player_name("Tim Hardaway Jr", SAMPLE),
            "Tim Hardaway Jr.",
        )
        self.assertEqual(
            resolve_player_name("tim hardaway jr.", SAMPLE),
            "Tim Hardaway Jr.",
        )


class AbbreviatedNameTests(unittest.TestCase):
    def test_initial_plus_surname_unique(self):
        # Only one player whose first name starts with J and last is Brunson.
        self.assertEqual(resolve_player_name("J. Brunson", SAMPLE), "Jalen Brunson")
        self.assertEqual(resolve_player_name("J Brunson", SAMPLE), "Jalen Brunson")

    def test_initial_plus_surname_ambiguous(self):
        # Two Bridges (Mikal + Miles), both first-letter-M → ambiguous.
        self.assertIsNone(resolve_player_name("M. Bridges", SAMPLE))

    def test_unique_surname_only(self):
        # Single token treated as surname.
        self.assertEqual(
            resolve_player_name("Wembanyama", SAMPLE),
            "Victor Wembanyama",
        )
        # Single token surname with multiple matches → None.
        self.assertIsNone(resolve_player_name("Bridges", SAMPLE))


class FirstCommaLastTests(unittest.TestCase):
    def test_last_comma_first_form(self):
        self.assertEqual(
            resolve_player_name("Brunson, Jalen", SAMPLE),
            "Jalen Brunson",
        )

    def test_last_comma_first_with_punctuation(self):
        self.assertEqual(
            resolve_player_name("McCollum, C.J.", SAMPLE),
            "CJ McCollum",
        )


class UnknownTests(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(resolve_player_name("", SAMPLE))
        self.assertIsNone(resolve_player_name(None, SAMPLE))

    def test_unknown_returns_none(self):
        self.assertIsNone(resolve_player_name("Foo Bar", SAMPLE))

    def test_empty_active_list(self):
        self.assertIsNone(resolve_player_name("Jalen Brunson", []))


if __name__ == "__main__":
    unittest.main()
