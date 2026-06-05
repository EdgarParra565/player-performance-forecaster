"""Tests for the Streamlit Operations console.

The panel itself needs ``streamlit.session_state`` to render, but the
``build_command`` helper is pure and easy to unit-test. We rely on it for the
"argv assembly" contract — the panel only renders the form, build_command
translates user input into the exact argv that gets handed to Popen.

Keeping this argv translation pure means an attacker can't smuggle shell
metacharacters through field values: the OS receives them as literal argv
entries (no shell expansion).
"""
from __future__ import annotations

import sys
import unittest

from nba_model.web import operations_panel as op


class BuildCommandTests(unittest.TestCase):
    def test_unknown_operation_raises(self):
        with self.assertRaises(ValueError):
            op.build_command("not_a_real_op", {}, {})

    def test_daily_etl_defaults(self):
        spec = op.OPERATIONS["daily_etl"]
        values = {flag: default for flag, _label, default, _kind in spec["fields"]}
        flag_values = {flag: bool(default) for flag, _label, default in spec["flags"]}
        argv = op.build_command("daily_etl", values, flag_values)
        self.assertEqual(argv[:3], [sys.executable, "-m", "nba_model.data.daily_etl"])
        self.assertIn("--db-path", argv)
        self.assertIn("data/database/nba_data.db", argv)

    def test_spaced_field_splits_on_whitespace(self):
        values = {
            "--start-date": "",
            "--end-date": "",
            "--stat-types": "points assists rebounds",
            "--windows": "5 7 10",
            "--distributions": "normal poisson",
        }
        argv = op.build_command("eval_benchmark", values, {})
        # The spaced flag must be followed by *each* token as a separate
        # argv entry (not the joined string).
        i = argv.index("--stat-types")
        self.assertEqual(argv[i + 1: i + 4], ["points", "assists", "rebounds"])
        i = argv.index("--windows")
        self.assertEqual(argv[i + 1: i + 4], ["5", "7", "10"])

    def test_bare_flag_always_emitted(self):
        # web_sync_players has --sync-active-players-ref as a bare_flag, so it
        # must appear regardless of any user-typed value.
        values = {
            "--sync-active-players-ref": "anything",
            "--active-players-output-file": "/tmp/players.txt",
        }
        argv = op.build_command("web_sync_players", values, {})
        self.assertIn("--sync-active-players-ref", argv)
        self.assertIn("--active-players-output-file", argv)
        self.assertIn("/tmp/players.txt", argv)

    def test_flag_checkbox_off_omits_flag(self):
        values = {
            "--source": "both",
            "--poll-seconds": "300",
            "--min-inferred-rows": "25",
            "--min-book-stat-groups": "2",
            "--min-player-segment-groups": "5",
            "--require-stability-runs": "2",
            "--stability-tolerance": "0.10",
        }
        argv_off = op.build_command("reverse_engineering", values, {"--continuous": False})
        argv_on  = op.build_command("reverse_engineering", values, {"--continuous": True})
        self.assertNotIn("--continuous", argv_off)
        self.assertIn("--continuous", argv_on)

    def test_empty_field_value_is_dropped(self):
        # Empty `--season` should not be passed at all (the CLI assumes the
        # default when the flag is absent).
        values = {
            "--db-path": "data/database/nba_data.db",
            "--season": "",
            "--players": "",
            "--min-players": "0",
            "--player-limit": "",
            "--game-log-games": "",
        }
        argv = op.build_command("daily_etl", values, {})
        self.assertNotIn("--season", argv)
        self.assertNotIn("--players", argv)
        self.assertNotIn("--player-limit", argv)
        self.assertNotIn("--game-log-games", argv)
        # --min-players had a non-empty value so it survives.
        self.assertIn("--min-players", argv)

    def test_shell_metacharacters_pass_through_as_literal_arg(self):
        # The whole reason we use argv lists and not shell strings: shell
        # metachars must NOT be interpreted. Confirm they're handed through
        # untouched so the OS treats them literally.
        evil = "value; rm -rf /"
        values = {
            "--db-path": evil,
            "--season": "",
            "--players": "",
            "--min-players": "0",
            "--player-limit": "",
            "--game-log-games": "",
        }
        argv = op.build_command("daily_etl", values, {})
        i = argv.index("--db-path")
        self.assertEqual(argv[i + 1], evil)


class OperationsRegistryTests(unittest.TestCase):
    """Sanity checks on the OPERATIONS registry itself.

    Every entry must declare a module + a `label`, fields/flags must be tuples
    of the expected arity, and the registry must include at least the
    operations the desktop UI exposes (ETL, web, evaluation, audit).
    """

    def test_required_operations_present(self):
        required = {
            "daily_etl", "web_validate", "web_fetch", "browser_parser",
            "eval_benchmark", "eval_distribution_sweep", "eval_line_compare",
            "eval_monthly_diag", "reverse_engineering", "db_audit",
        }
        self.assertTrue(required.issubset(set(op.OPERATIONS.keys())))

    def test_spec_shape(self):
        for slug, spec in op.OPERATIONS.items():
            self.assertIn("label", spec, slug)
            self.assertIn("module", spec, slug)
            for field in spec["fields"]:
                self.assertEqual(len(field), 4, slug)
            for flag in spec["flags"]:
                self.assertEqual(len(flag), 3, slug)


if __name__ == "__main__":
    unittest.main()
