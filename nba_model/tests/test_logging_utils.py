"""Tests for the shared structured-logging module (nba_model/logging_utils.py).

Scope (kept tight per the task): the JSON-lines formatter emits one valid JSON
object per record with a UTC ISO timestamp and any ``extra=`` fields, and
``configure_logging`` writes that JSON to a file, is idempotent, and honours
``LOG_LEVEL``. Redaction is explicitly NOT tested — the module does none by
design; keeping secrets out of log calls is the caller's contract.
"""

import json
import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from nba_model import logging_utils as lu


def _record(msg, *, level=logging.INFO, name="test.logger", extra=None):
    rec = logging.makeLogRecord({
        "name": name, "levelno": level,
        "levelname": logging.getLevelName(level), "msg": msg,
    })
    for key, value in (extra or {}).items():
        setattr(rec, key, value)
    return rec


class JsonLinesFormatterTests(unittest.TestCase):
    def setUp(self):
        self.fmt = lu.JsonLinesFormatter()

    def test_emits_single_valid_json_line(self):
        line = self.fmt.format(_record("hello world"))
        self.assertNotIn("\n", line)
        obj = json.loads(line)  # raises if not valid JSON
        self.assertEqual(obj["message"], "hello world")
        self.assertEqual(obj["level"], "INFO")
        self.assertEqual(obj["logger"], "test.logger")

    def test_timestamp_is_utc_iso_with_z(self):
        obj = json.loads(self.fmt.format(_record("x")))
        self.assertTrue(obj["ts"].endswith("Z"), obj["ts"])
        # Parses as an ISO-8601 instant (Z → +00:00 for fromisoformat).
        parsed = datetime.fromisoformat(obj["ts"].replace("Z", "+00:00"))
        self.assertIsNotNone(parsed.tzinfo)

    def test_extra_fields_are_serialized(self):
        obj = json.loads(self.fmt.format(_record(
            "step done",
            extra={"step": "game_logs", "rows": 1234, "duration_ms": 87},
        )))
        self.assertEqual(obj["step"], "game_logs")
        self.assertEqual(obj["rows"], 1234)
        self.assertEqual(obj["duration_ms"], 87)

    def test_nested_extra_round_trips(self):
        obj = json.loads(self.fmt.format(_record(
            "finished", extra={"step_statuses": {"game_logs": "success"}})))
        self.assertEqual(obj["step_statuses"], {"game_logs": "success"})

    def test_non_json_extra_falls_back_to_str(self):
        obj = json.loads(self.fmt.format(_record(
            "p", extra={"log_path": Path("/tmp/x.log")})))
        self.assertEqual(obj["log_path"], "/tmp/x.log")

    def test_exc_info_rendered(self):
        try:
            raise ValueError("boom")
        except ValueError:
            rec = _record("it failed", level=logging.ERROR)
            rec.exc_info = sys.exc_info()
        obj = json.loads(self.fmt.format(rec))
        self.assertIn("ValueError", obj["exc_info"])


class ConfigureLoggingTests(unittest.TestCase):
    """These mutate the root logger, so save/restore around each test."""

    def setUp(self):
        self.root = logging.getLogger()
        self._saved_handlers = list(self.root.handlers)
        self._saved_level = self.root.level
        for handler in list(self.root.handlers):
            self.root.removeHandler(handler)

    def tearDown(self):
        for handler in list(self.root.handlers):
            self.root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in self._saved_handlers:
            self.root.addHandler(handler)
        self.root.setLevel(self._saved_level)

    def test_writes_jsonl_file_with_extras(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = lu.configure_logging(
                file_prefix="unit_etl", log_dir=tmp, force=True, console=False)
            self.assertIsNotNone(path)
            self.assertTrue(path.endswith(".log"))
            self.assertTrue(Path(path).name.startswith("unit_etl_"))
            lu.get_logger("unit.test").info("hi", extra={"rows": 5})
            for handler in self.root.handlers:
                handler.flush()
            lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
            self.assertTrue(lines)
            obj = json.loads(lines[-1])
            self.assertEqual(obj["message"], "hi")
            self.assertEqual(obj["rows"], 5)
            self.assertEqual(obj["logger"], "unit.test")

    def test_no_file_prefix_returns_none(self):
        self.assertIsNone(
            lu.configure_logging(force=True, console=False))

    def test_idempotent_when_handlers_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lu.configure_logging(
                file_prefix="a", log_dir=tmp, force=True, console=False)
            # No force + handlers already present → no-op, returns existing path.
            second = lu.configure_logging(file_prefix="b", log_dir=tmp)
            self.assertIsNotNone(second)
            self.assertEqual(Path(second).name, Path(first).name)

    def test_log_level_from_env(self):
        old = os.environ.get("LOG_LEVEL")
        os.environ["LOG_LEVEL"] = "WARNING"
        try:
            lu.configure_logging(force=True, console=False)
            self.assertEqual(self.root.level, logging.WARNING)
        finally:
            if old is None:
                os.environ.pop("LOG_LEVEL", None)
            else:
                os.environ["LOG_LEVEL"] = old


if __name__ == "__main__":
    unittest.main()
