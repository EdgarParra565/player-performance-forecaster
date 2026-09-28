"""Parser-only tests for the VegasInsider MLB player-props grid extractor.

``MLB_FIXTURE`` is built from VERBATIM excerpts of a REAL capture
(``web_text_snapshots`` id 155, ``vegasinsider.com/mlb/odds/player-props``,
fetched 2026-07-22): one fully-aligned row + one partial row per section,
the stray ``0`` tokens, bare (line-less) Home Run prices, and the capture's
truncated tail (the 60K cap cut the page mid-row). No live data in CI.

Storage / ingestion into ``mlb_prop_lines`` is covered separately
(``test_vegasinsider_mlb_ingestion``).
"""

import unittest

from nba_model.model import web_text_ingestion as wti
from nba_model.scrapers import vegasinsider as vi
from nba_model.scrapers.mlb_props import stat_group
from sports.mlb import SPORT as MLB

_K = (
    "Strikeouts Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars FanDuel "
    "Fanatics Sleeper Underdog RiversCasino › › › › › › › › › › "
    # Partial row: 9 cells for 10 books -> unattributable, dropped.
    "Gerrit Cole o6.5 -110 + o6.5 -137 + o10.5 +110 o6.5 -112 + o6.5 -108 "
    "o6.5 -100 + o6.5 -122 + o6.5 -137 o6.5 -114 + "
    # Fully aligned (10 cells).
    "Reid Detmers o5.5 -165 + o6 -137 + o6.5 +105 o5.5 -156 + o5.5 -148 "
    "o6.5 +146 + o5.5 +145 + o5.5 -169 + o5.5 -137 o5.5 -141 + "
)
_HR = (
    "See All Home Runs Odds Time Bet365 BetMGM Caesars FanDuel HardRock "
    "Sleeper › › › › › › "
    "Randy Arozarena o0.5 +600 + +750 +475 + +480 o0.5 +525 + "
    "Mitch Garver 0 "
    "Jorbit Vivas o0.5 +700 + +700 0 +830 o0.5 +850 + "
    # Fully aligned (6 cells), mostly bare yes/no prices.
    "Cal Raleigh o0.5 +265 + +375 + +245 + +260 o0.5 +275 + o0.5 +238 + "
)
_TB = (
    "See All Total Bases Odds Time Bet365 PrizePicks BetMGM DraftKings "
    "Caesars Fanatics Sleeper Underdog › › › › › › › › "
    "Ezequiel Tovar o1.5 +160 o0.5 -185 o0.5 -190 o0.5 -185 + o0.5 -227 "
    "James Wood o1.5 -150 + o1.5 -137 + o1.5 +105 o2.5 +113 + o1.5 +107 "
    "o1.5 -150 + o1.5 -179 + o1.5 -137 "
    "Luis Garcia o1.5 -105 + o1.5 -147 + o1.5 -152 o1.5 -155 + o1.5 -167 + "
)
_RBI = (
    "See All Runs Batted In Odds Time Bet365 PrizePicks BetMGM Caesars "
    "Fanatics Sleeper Underdog › › › › › › › "
    "Rafael Devers o0.5 +132 + o0.5 +340 o0.5 +125 o0.5 +150 + o0.5 +119 + "
    "Cal Raleigh o0.5 +132 + o1.5 +210 + o0.5 +142 o0.5 +130 + o0.5 +125 + "
    # Real truncated tail of the capture.
    "Josh Bell o0.5 +"
)
MLB_FIXTURE = (
    "MLB Player Props Odds 2026 ... Be sure to check out the latest strikeout "
    "odds and homerun odds ! " + _K + _HR + _TB + _RBI
)


def _rows_for(rows, player, stat=None):
    return [
        r for r in rows
        if r["player_name"] == player and (stat is None or r["stat_type"] == stat)
    ]


class ExtractMlbOddsRowsTests(unittest.TestCase):
    def setUp(self):
        self.rows = vi.extract_mlb_odds_rows(MLB_FIXTURE)

    def test_only_fully_aligned_rows_emitted(self):
        # Detmers 10 books - BetMGM = 9; Raleigh HR 6 - BetMGM = 5;
        # James Wood 8 - BetMGM = 7; no RBI row is fully aligned.
        self.assertEqual(len(self.rows), 9 + 5 + 7)
        players = {r["player_name"] for r in self.rows}
        self.assertEqual(players, {"Reid Detmers", "Cal Raleigh", "James Wood"})

    def test_partial_and_zero_rows_dropped(self):
        for player in ("Gerrit Cole", "Randy Arozarena", "Mitch Garver",
                       "Jorbit Vivas", "Ezequiel Tovar", "Luis Garcia",
                       "Rafael Devers", "Josh Bell"):
            self.assertEqual(_rows_for(self.rows, player), [], player)
        self.assertEqual(_rows_for(self.rows, "Cal Raleigh", "rbis"), [])

    def test_strikeouts_positional_attribution(self):
        detmers = _rows_for(self.rows, "Reid Detmers")
        self.assertEqual(
            [(r["book"], r["line_value"], r["odds"]) for r in detmers],
            [
                ("bet365", 5.5, -165), ("prizepicks", 6.0, -137),
                ("draftkings", 5.5, -156), ("caesars", 5.5, -148),
                ("fanduel", 6.5, 146), ("fanatics", 5.5, 145),
                ("sleeper", 5.5, -169), ("underdog", 5.5, -137),
                ("betrivers", 5.5, -141),
            ],
        )
        self.assertTrue(all(r["stat_type"] == "strikeouts_pitcher" for r in detmers))
        self.assertTrue(all(r["market_shape"] == "over_under" for r in detmers))

    def test_home_run_is_yes_no_market_including_bare_prices(self):
        hr = _rows_for(self.rows, "Cal Raleigh", "anytime_home_run")
        self.assertEqual(
            [(r["book"], r["odds"]) for r in hr],
            [("bet365", 265), ("caesars", 245), ("fanduel", 260),
             ("hardrockbet", 275), ("sleeper", 238)],
        )
        for r in hr:
            self.assertEqual(r["market_shape"], "yes_no")
            self.assertEqual(r["line_value"], 0.5)
            self.assertEqual(r["side"], "over")  # Yes -> over

    def test_total_bases_keeps_alt_lines_per_book(self):
        wood = {r["book"]: r for r in _rows_for(self.rows, "James Wood")}
        self.assertEqual(wood["draftkings"]["line_value"], 2.5)
        self.assertEqual(wood["draftkings"]["odds"], 113)
        self.assertEqual(wood["bet365"]["line_value"], 1.5)

    def test_betmgm_is_never_emitted(self):
        self.assertNotIn("betmgm", {r["book"] for r in self.rows})

    def test_stat_types_are_registry_valid_and_groups_separate(self):
        valid = set(MLB.stat_types)
        for r in self.rows:
            self.assertIn(r["stat_type"], valid)
            self.assertIn(r["book"], {
                "bet365", "prizepicks", "draftkings", "caesars", "fanduel",
                "fanatics", "sleeper", "underdog", "betrivers", "hardrockbet",
            })
        groups = {r["stat_type"]: stat_group(r["stat_type"]) for r in self.rows}
        self.assertEqual(groups["strikeouts_pitcher"], "pitching")
        self.assertEqual(groups["total_bases"], "hitting")
        self.assertEqual(groups["anytime_home_run"], "combined")


class TruncationAndDriftTests(unittest.TestCase):
    def test_final_row_dropped_only_when_text_ends_mid_row(self):
        cut = _K.rstrip()[:-2].rstrip()  # ends on a price -> possibly cut
        self.assertEqual(_rows_for(vi.extract_mlb_odds_rows(cut), "Reid Detmers"), [])
        # "-141 +": the "+" is the betslip button after a complete price.
        button = _K.rstrip()
        self.assertEqual(
            len(_rows_for(vi.extract_mlb_odds_rows(button), "Reid Detmers")), 9)
        complete = _K + "Sign up for our newsletter"
        self.assertEqual(
            len(_rows_for(vi.extract_mlb_odds_rows(complete), "Reid Detmers")), 9)

    def test_sign_after_line_token_is_a_cut_price(self):
        # Real tail of snapshot 155: "Josh Bell o0.5 +".
        self.assertTrue(vi._ends_mid_row("Josh Bell o0.5 +"))
        self.assertFalse(vi._ends_mid_row("o5.5 -141 +"))

    def test_rejected_header_does_not_bleed_into_previous_section(self):
        # Drop one "›" from the TB header: its rows must vanish, not be parsed
        # as Home Run cells under the HR book columns.
        drifted_tb = _TB.replace("› › › › › › › › ", "› › › › › › › ", 1)
        rows = vi.extract_mlb_odds_rows(_HR + drifted_tb + " footer")
        self.assertEqual(_rows_for(rows, "James Wood"), [])
        self.assertEqual({r["player_name"] for r in rows}, {"Cal Raleigh"})

    def test_zero_book_header_is_a_boundary(self):
        text = (_HR + "See All Total Bases Odds Time › › "
                "Juan Soto o0.5 -150 o0.5 -160 footer")
        rows = vi.extract_mlb_odds_rows(text)
        self.assertEqual({r["player_name"] for r in rows}, {"Cal Raleigh"})

    def test_label_regex_is_linear(self):
        import time
        started = time.monotonic()
        vi.extract_mlb_odds_rows("word " * 12000)
        vi.extract_odds_rows("word " * 12000)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_truncated_price_never_emitted(self):
        # A full-looking row whose last price was cut ("-141" -> "-14").
        cut = _K.rstrip()[: -len("-141 +")] + "-14"
        rows = vi.extract_mlb_odds_rows(cut)
        self.assertEqual(_rows_for(rows, "Reid Detmers"), [])

    def test_header_arrow_mismatch_skips_section(self):
        drifted = _K.replace("› › › › › › › › › › ", "› › › ", 1)
        self.assertEqual(vi.extract_mlb_odds_rows(drifted + " footer"), [])

    def test_unobserved_section_label_is_skipped(self):
        other = _TB.replace("See All Total Bases", "See All Hits Allowed") + " footer"
        self.assertEqual(vi.extract_mlb_odds_rows(other), [])

    def test_empty_input(self):
        self.assertEqual(vi.extract_mlb_odds_rows(""), [])
        self.assertEqual(vi.extract_mlb_odds_rows(None), [])


class NbaGridHardeningTests(unittest.TestCase):
    def _nba_row(self, name, last_cell):
        return name + " " + "o12.5 -110 + " * 10 + last_cell

    def _header(self, stat):
        return (f"See All {stat} Odds Time Bet365 PrizePicks BetMGM DraftKings "
                "Caesars FanDuel HardRock Fanatics Sleeper Underdog RiversCasino "
                + "› " * 11)

    def test_two_digit_price_is_not_a_cell(self):
        text = self._header("Points") + self._nba_row("Victor Wembanyama", "o12.5 -11")
        self.assertEqual(vi.extract_odds_rows(text), [])

    def test_five_digit_price_not_truncated(self):
        text = (self._header("Points")
                + self._nba_row("Victor Wembanyama", "o12.5 +10000 +") + " footer")
        rows = vi.extract_odds_rows(text)
        self.assertEqual(rows[-1]["over_odds"], 10000)

    def test_rejected_header_does_not_bleed(self):
        bad_header = self._header("Rebounds").replace("RiversCasino", "ESPNBet")
        text = (self._header("Points") + self._nba_row("Jalen Brunson", "o26.5 -110 + ")
                + bad_header + self._nba_row("Victor Wembanyama", "o12.5 -110 + ")
                + " footer")
        rows = vi.extract_odds_rows(text)
        self.assertEqual({r["player_name"] for r in rows}, {"Jalen Brunson"})


class CrossSportIsolationTests(unittest.TestCase):
    def test_nba_extractor_reads_nothing_from_mlb_grid(self):
        self.assertEqual(vi.extract_odds_rows(MLB_FIXTURE), [])

    def test_mlb_extractor_reads_nothing_from_nba_grid(self):
        from nba_model.tests.test_vegasinsider_odds import VI_FIXTURE
        self.assertEqual(len(vi.extract_odds_rows(VI_FIXTURE)), 33)  # sanity
        self.assertEqual(vi.extract_mlb_odds_rows(VI_FIXTURE), [])


class PerBookTextCapTests(unittest.TestCase):
    def test_vegasinsider_raises_global_cap(self):
        url = "https://www.vegasinsider.com/mlb/odds/player-props/"
        self.assertEqual(wti._effective_max_chars(url, 60000), 250_000)

    def test_cap_never_lowered_and_unlimited_stays_unlimited(self):
        url = "https://www.vegasinsider.com/mlb/odds/player-props/"
        self.assertEqual(wti._effective_max_chars(url, 500_000), 500_000)
        self.assertEqual(wti._effective_max_chars(url, 0), 0)

    def test_other_books_keep_global_cap(self):
        self.assertEqual(
            wti._effective_max_chars("https://sportsbook.fanduel.com/navigation/mlb", 60000),
            60000,
        )
        self.assertEqual(wti._effective_max_chars("https://example.com/x", 60000), 60000)


if __name__ == "__main__":
    unittest.main()
