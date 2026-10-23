"""LLM forecast collector (WS11): selection, prompt, parser, cache, providers.

Offline by construction: every provider is a FakeProvider, a fake SDK client
or a fake HTTP transport, and the module blocks sockets while it runs, so a
stray real request fails the test instead of reaching the network.
"""

import contextlib
import io
import json
import logging
import os
import socket
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.evaluation import llm_forecast as lf
from nba_model.tests.llm_fixtures import (
    CURRY, EXPECTED_MARKET_PROB, LEBRON, NEXT, TARGET, TARGET_POINTS,
    add_real_prices, seed_llm_db,
)

_NET_PATCHES = []


def _no_network(*_args, **_kwargs):
    raise AssertionError("network access attempted in the test suite")


def setUpModule():
    _NET_PATCHES[:] = [mock.patch("socket.getaddrinfo", _no_network),
                       mock.patch("socket.socket.connect", _no_network),
                       mock.patch("socket.create_connection", _no_network)]
    for patcher in _NET_PATCHES:
        patcher.start()


def tearDownModule():
    for patcher in _NET_PATCHES:
        patcher.stop()


class _Db(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db_path = str(self.dir / "nba.db")
        self.ids = seed_llm_db(self.db_path)
        self.cache_dir = self.dir / "cache"
        self.artifacts = self.dir / "artifacts"

    def tearDown(self):
        self._tmp.cleanup()

    def select(self, **kwargs):
        with DatabaseManager(self.db_path, read_only=True) as db:
            return lf.select_retro_props(db.conn, **kwargs)

    def prop(self, game_date=TARGET, stat="points"):
        props, _ = self.select()
        return next(p for p in props if p.game_date == game_date and p.stat_type == stat)

    def run_fake(self, provider=None, **kwargs):
        kwargs.setdefault("retro", True)
        kwargs.setdefault("limit", None)
        return lf.run_llm_forecasts(
            self.db_path, provider or lf.FakeProvider(), cache_dir=self.cache_dir,
            artifact_dir=self.artifacts, **kwargs)

    def rows(self, sql="SELECT * FROM llm_forecasts ORDER BY forecast_id"):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql)]
        finally:
            conn.close()


class SelectionTests(_Db):
    def test_funnel_and_filters(self):
        props, funnel = self.select()
        self.assertEqual(funnel, {
            "settled_props": 10, "line_kind": 8, "model_line": 1, "no_player_name": 1,
            "short_history": 1, "implausible_line": 1, "eligible": 4})
        self.assertEqual(
            sorted((p.game_date, p.stat_type, p.line_value) for p in props),
            [(TARGET, "points", 25.5), (TARGET, "rebounds", 7.5),
             (NEXT, "points", 24.5), (NEXT, "pra", 35.5)])
        self.assertNotIn(CURRY, {p.player_id for p in props})
        _, any_lines = self.select(line_kind="any", max_line_z=None)
        # Both model-mean-only lines go (22.857 and the 7.5 assists line).
        self.assertEqual((any_lines["model_line"], any_lines["eligible"]), (2, 6))

    def test_model_baseline_is_the_production_config_row(self):
        props, _ = self.select()
        by = {(p.game_date, p.stat_type): p.prediction_id for p in props}
        self.assertEqual(by[(TARGET, "points")], self.ids["pts_prod"])     # normal, w10
        self.assertEqual(by[(TARGET, "rebounds")], self.ids["reb_prod"])   # poisson, w10
        self.assertEqual(by[(NEXT, "pra")], self.ids["pra_next"])

    def test_history_is_strictly_before_the_game_and_matchup_parsed(self):
        prop = self.prop()
        self.assertEqual((prop.team, prop.opponent, prop.home), ("LAL", "BOS", True))
        self.assertEqual(len(prop.history), lf.DEFAULT_LAST_N)
        self.assertTrue(all(g.game_date < TARGET for g in prop.history))
        self.assertEqual([g.game_date for g in prop.history],
                         sorted((g.game_date for g in prop.history), reverse=True))
        self.assertNotIn(TARGET_POINTS, [g.value for g in prop.history])
        nxt = self.prop(NEXT)
        self.assertEqual(nxt.history[0].game_date, TARGET)   # prior game is fair game
        self.assertEqual((nxt.opponent, nxt.home), ("NYK", False))

    def test_stats_and_date_range_filters(self):
        props, _ = self.select(stats=["PRA"])
        self.assertEqual([p.stat_type for p in props], ["pra"])
        props, _ = self.select(date_range=[TARGET, TARGET])
        self.assertEqual({p.game_date for p in props}, {TARGET})
        with self.assertRaises(ValueError):
            self.select(stats=["dunks"])

    def test_sample_is_deterministic_per_seed(self):
        props, _ = self.select()
        a = lf.sample_props(props, 2, "s1")
        self.assertEqual(a, lf.sample_props(list(reversed(props)), 2, "s1"))
        self.assertEqual(len(lf.sample_props(props, None, "s1")), len(props))
        self.assertEqual(
            {tuple(p.key) for p in lf.sample_props(props, 0, "x")},
            {tuple(p.key) for p in props})

    def test_market_price_is_devigged_from_pre_cutoff_two_sided_rows(self):
        self.assertIsNone(self.prop().market_prob_over)
        add_real_prices(self.db_path)   # also adds in-play / post-game rows
        prop = self.prop()
        self.assertAlmostEqual(prop.market_prob_over, EXPECTED_MARKET_PROB, places=9)
        self.assertEqual(prop.market_books, 2)


class PromptTests(_Db):
    BANNED = ("odds", "vig", "implied", "market", "sportsbook", "-110", "price")

    def test_blind_prompt_has_the_facts_and_no_outcome_or_price(self):
        add_real_prices(self.db_path)   # a price exists, the blind arm must not show it
        spec = lf.build_prompt(self.prop())
        text = (spec.system + "\n" + spec.user).lower()
        for fact in ("lebron james", "lal", "bos", TARGET, "points", "25.5"):
            self.assertIn(fact, text)
        for word in self.BANNED:
            self.assertNotIn(word, text)
        self.assertNotIn(f"| {TARGET_POINTS}", spec.user)          # the outcome
        self.assertNotIn(f"{TARGET} |", spec.user)                 # the game's own row
        self.assertNotIn(f"{EXPECTED_MARKET_PROB:.3f}", spec.user)
        self.assertEqual((spec.prompt_version, spec.arm), ("v1", "blind"))

    def test_market_arm_adds_the_price_and_needs_one(self):
        with self.assertRaises(ValueError):
            lf.build_prompt(self.prop(), arm="market")
        add_real_prices(self.db_path)
        spec = lf.build_prompt(self.prop(), arm="market")
        self.assertIn(f"P(over) = {EXPECTED_MARKET_PROB:.3f}", spec.user)
        self.assertNotIn(f"| {TARGET_POINTS}", spec.user)
        with self.assertRaises(ValueError):
            lf.build_prompt(self.prop(), arm="peek")

    def test_prompt_hash_tracks_everything_that_changes_the_reply(self):
        spec = lf.build_prompt(self.prop())
        base = lf.prompt_hash("anthropic", "m", {"max_tokens": 9}, spec)
        self.assertEqual(base, lf.prompt_hash("anthropic", "m", {"max_tokens": 9}, spec))
        for variant in (lf.prompt_hash("openai", "m", {"max_tokens": 9}, spec),
                        lf.prompt_hash("anthropic", "m2", {"max_tokens": 9}, spec),
                        lf.prompt_hash("anthropic", "m", {"max_tokens": 10}, spec),
                        lf.prompt_hash("anthropic", "m", {"max_tokens": 9}, spec,
                                       spec.user + lf.REASK_SUFFIX_V1)):
            self.assertNotEqual(base, variant)


class ParserTests(unittest.TestCase):
    def test_strict_probability(self):
        for text, want in (("0.62", 0.62), (" 0.62\n", 0.62), (".62", 0.62), ("0", 0.0),
                           ("1", 1.0), ("1.000", 1.0), ("0.5", 0.5)):
            self.assertEqual(lf.parse_probability(text), want, text)
        for text in ("62%", "0.62.", "P(over)=0.62", "about 0.6", "1.5", "-0.2", "0.",
                     "", "0.6 or 0.7", "`0.62`", None, 0.62, "1e-1", "０.６", ".６"):
            self.assertIsNone(lf.parse_probability(text), text)


class CacheAndCostTests(unittest.TestCase):
    def test_cache_roundtrip_and_tamper_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = lf.PromptCache(Path(tmp) / "c")
            digest = "ab" + "0" * 62
            self.assertIsNone(cache.get(digest))
            cache.put(digest, {"response": {"text": "0.4"}})
            self.assertEqual(cache.get(digest)["response"]["text"], "0.4")
            path = Path(tmp) / "c" / "ab" / f"{digest}.json"
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])
            path.write_text(json.dumps({"prompt_hash": "other"}), encoding="utf-8")
            self.assertIsNone(cache.get(digest))
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(cache.get(digest))

    def test_cost_estimate_upper_bound(self):
        spec = lf.PromptSpec(system="s" * 30, user="u" * 270, prompt_version="v1", arm="blind")
        est = lf.estimate_cost([spec, spec], max_output_tokens=100, price=(4.0, 20.0),
                               max_reasks=1)
        self.assertEqual(est["calls"], 2)
        self.assertEqual(est["input_tokens_est"], 200)            # 300 chars / 3, x2
        self.assertEqual(est["output_tokens_max"], 200)
        self.assertAlmostEqual(est["usd_upper"], (200 * 4 + 200 * 20) / 1e6)
        self.assertGreater(est["usd_worst"], est["usd_upper"])
        unknown = lf.estimate_cost([spec], max_output_tokens=100, price=None)
        self.assertIsNone(unknown["usd_upper"])
        self.assertEqual(lf.resolve_price("anthropic", "claude-opus-5-5"), (4.0, 20.0))
        self.assertIsNone(lf.resolve_price("openai", "anything"))
        self.assertEqual(lf.resolve_price("openai", "x", (1, 2)), (1.0, 2.0))


class RunTests(_Db):
    def test_dry_run_has_no_side_effects(self):
        provider = lf.FakeProvider()
        before = os.path.getmtime(self.db_path)
        summary = self.run_fake(provider, dry_run=True)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["props"], 4)
        self.assertEqual((summary["cached_prompts"], summary["uncached_prompts"]), (0, 4))
        self.assertIn("Will LeBron James", summary["sample_prompt"]["user"])
        self.assertEqual(provider.calls, [])
        self.assertFalse(self.cache_dir.exists())
        self.assertFalse(self.artifacts.exists())
        self.assertEqual(os.path.getmtime(self.db_path), before)
        self.assertEqual(self.rows(), [])
        self.assertIn("DRY RUN", lf.format_summary(summary))

    def test_dry_run_with_a_real_provider_needs_no_key_or_sdk(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            summary = self.run_fake(lf.make_provider("anthropic"), dry_run=True, limit=2)
        self.assertEqual(summary["model"], "claude-opus-5-5")
        self.assertEqual(summary["params"], {"max_tokens": 1024, "effort": "low"})
        self.assertFalse(summary["env_key_set"])
        est = summary["cost_estimate"]
        self.assertEqual((est["price_in_per_mtok"], est["price_out_per_mtok"]), (4.0, 20.0))
        self.assertEqual(est["output_tokens_max"], 2 * 1024)
        self.assertIn("ANTHROPIC_API_KEY: NOT set", lf.format_summary(summary))

    def test_run_inserts_settles_and_labels_retro(self):
        provider = lf.FakeProvider(
            responder=lambda s, u: "0.7" if "points" in u.splitlines()[0] else "0.4")
        summary = self.run_fake(provider, run_id="t1", training_cutoff="2024-06-30")
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["parse_status"], {"ok": 4, "invalid": 0, "refusal": 0})
        self.assertEqual(summary["settle"]["settled"], 4)
        self.assertTrue(Path(summary["manifest_path"]).is_file())
        rows = {(r["game_date"], r["stat_type"]): r for r in self.rows()}
        first = rows[(TARGET, "points")]
        self.assertEqual((first["prob_over"], first["outcome"], first["actual_result"]),
                         (0.7, "over", float(TARGET_POINTS)))
        self.assertEqual(first["prediction_id"], self.ids["pts_prod"])
        self.assertEqual(first["info_cutoff_utc"], f"{TARGET}T00:00:00Z")
        self.assertEqual(first["training_cutoff"], "2024-06-30")
        self.assertEqual(len(first["prompt_hash"]), 64)
        self.assertEqual(json.loads(first["raw_response"])[0]["text"], "0.7")
        self.assertEqual({r["retro"] for r in rows.values()}, {1})
        self.assertEqual(rows[(NEXT, "pra")]["outcome"], "under")

    def test_rerun_is_free_and_never_duplicates(self):
        provider = lf.FakeProvider()
        self.run_fake(provider, run_id="t1")
        self.assertEqual(len(provider.calls), 4)
        again = self.run_fake(provider, run_id="t1")              # already answered
        self.assertEqual((again["already_stored"], again["rows"], again["api_calls"]),
                         (4, 0, 0))
        self.assertEqual(again["cost_estimate"]["calls"], 0)
        other = self.run_fake(provider, run_id="t2")              # new run: cache hits
        self.assertEqual((other["cache_hits"], other["api_calls"]), (4, 0))
        self.assertEqual(other["insert"]["inserted"], 4)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(len(self.rows()), 8)

    def test_changed_settings_in_the_same_run_are_refused_before_any_call(self):
        provider = lf.FakeProvider()
        self.run_fake(provider, run_id="t1")
        preview = self.run_fake(provider, run_id="t1", last_n=8, dry_run=True)
        self.assertEqual(preview["conflicts"], 4)
        self.assertIn("CONFLICT", lf.format_summary(preview))
        with self.assertRaises(ValueError) as ctx:
            self.run_fake(provider, run_id="t1", last_n=8)
        self.assertIn("new --run-id", str(ctx.exception))
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(self.run_fake(provider, run_id="t2", last_n=8)["rows"], 4)

    def test_reask_then_invalid_and_refusal(self):
        def responder(system, user):
            if "PRA" in user:
                return lf.LLMReply(text="", stop_reason="refusal", refused=True)
            if "rebounds" in user.splitlines()[0]:
                return "no idea"
            if lf.REASK_SUFFIX_V1 in user:
                return "0.61"
            return "about 60%"
        provider = lf.FakeProvider(responder=responder)
        summary = self.run_fake(provider, run_id="t1")
        self.assertEqual(summary["parse_status"], {"ok": 2, "invalid": 1, "refusal": 1})
        rows = {(r["game_date"], r["stat_type"]): r for r in self.rows()}
        ok = rows[(TARGET, "points")]
        self.assertEqual((ok["parse_status"], ok["prob_over"]), ("ok", 0.61))
        self.assertEqual([a["text"] for a in json.loads(ok["raw_response"])],
                         ["about 60%", "0.61"])
        bad = rows[(TARGET, "rebounds")]
        self.assertEqual((bad["parse_status"], bad["prob_over"]), ("invalid", None))
        self.assertEqual(len(json.loads(bad["raw_response"])), 2)
        refused = rows[(NEXT, "pra")]
        self.assertEqual(refused["parse_status"], "refusal")
        self.assertEqual(len(json.loads(refused["raw_response"])), 1)   # never re-asked

    def test_provider_errors_skip_or_abort(self):
        flaky = lf.FakeProvider(replies=[lf.ProviderError("HTTP 529"), "0.5", "0.5", "0.5"])
        summary = self.run_fake(flaky, run_id="t1")
        self.assertEqual((summary["rows"], summary["failures"]), (3, 1))
        self.assertIsNone(summary["aborted"])
        fatal = lf.FakeProvider(model="fake-2",   # own cache keys: every prop is a call
                                replies=["0.5", lf.ProviderError("HTTP 401", fatal=True)])
        summary = self.run_fake(fatal, run_id="t2")
        self.assertEqual(summary["rows"], 1)                      # kept, then stopped
        self.assertEqual(summary["aborted"], "HTTP 401")
        self.assertEqual(len(self.rows("SELECT * FROM llm_forecasts WHERE run_id='t2'")), 1)

    def test_guards_fire_before_any_call(self):
        provider = lf.FakeProvider()
        with self.assertRaises(ValueError):                       # forward runs not built
            lf.run_llm_forecasts(self.db_path, provider, cache_dir=self.cache_dir)
        with self.assertRaises(ValueError):                       # cost cap
            self.run_fake(provider, price_per_mtok=(1000.0, 1000.0), max_cost_usd=0.01)
        openai = lf.OpenAIProvider("gpt-test", transport=self.fail_transport)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with self.assertRaises(ValueError):                   # unknown pricing
                self.run_fake(openai)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(lf.ProviderNotConfigured):     # no key, no call
                self.run_fake(lf.make_provider("anthropic"))
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.rows(), [])

    def test_market_arm_runs_only_on_real_prices(self):
        add_real_prices(self.db_path)
        provider = lf.FakeProvider()
        summary = self.run_fake(provider, run_id="m1", arm="market")
        self.assertEqual(summary["props"], 1)
        self.assertEqual(summary["funnel"]["with_real_price"], 1)
        self.assertIn("P(over) = 0.516", provider.calls[0][1])
        rows = self.rows()
        self.assertEqual([r["arm"] for r in rows], ["market"])
        self.assertEqual(rows[0]["info_cutoff_utc"],
                         f"{TARGET}T{lf.MARKET_PRICE_CUTOFF_UTC}Z")

    @staticmethod
    def fail_transport(*_args, **_kwargs):
        raise AssertionError("transport must not be called")


def _fake_response(status, payload=None):
    return SimpleNamespace(status_code=status, json=lambda: payload or {})


OPENAI_OK = {"choices": [{"message": {"content": "0.61"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 120, "completion_tokens": 3}}


class AnthropicAdapterTests(unittest.TestCase):
    def client(self, response=None, error=None):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            return response
        return SimpleNamespace(messages=SimpleNamespace(create=create)), calls

    def test_request_shape_and_text_blocks_only(self):
        resp = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text="0.58")],
            usage=SimpleNamespace(input_tokens=310, output_tokens=40))
        client, calls = self.client(resp)
        provider = lf.AnthropicProvider("claude-sonnet-5-5", client=client)
        reply = provider.complete("sys", "user")
        self.assertEqual((reply.text, reply.input_tokens, reply.output_tokens, reply.refused),
                         ("0.58", 310, 40, False))
        self.assertEqual(calls[0], {
            "model": "claude-sonnet-5-5", "max_tokens": 1024, "system": "sys",
            "messages": [{"role": "user", "content": "user"}],
            "output_config": {"effort": "low"}})
        no_effort = lf.make_provider("anthropic", effort="none")
        self.assertIsNone(no_effort.effort)
        self.assertEqual(lf.make_provider("anthropic", effort="high").effort, "high")

    def test_refusal_and_errors(self):
        client, _ = self.client(SimpleNamespace(stop_reason="refusal", content=[], usage=None))
        self.assertTrue(lf.AnthropicProvider(client=client).complete("s", "u").refused)
        boom = type("AuthenticationError", (Exception,), {})("invalid x-api-key SECRET")
        boom.status_code = 401
        client, _ = self.client(error=boom)
        with self.assertRaises(lf.ProviderError) as ctx:
            lf.AnthropicProvider(client=client).complete("s", "u")
        self.assertTrue(ctx.exception.fatal)
        self.assertNotIn("SECRET", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)

    def test_not_ready_without_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "  "}):
            with self.assertRaises(lf.ProviderNotConfigured) as ctx:
                lf.AnthropicProvider().check_ready()
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))


class OpenAIAdapterTests(unittest.TestCase):
    SECRET = "sk-test-SECRET-0123456789abcdef"

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": self.SECRET})
        patcher.start()
        self.addCleanup(patcher.stop)

    def provider(self, responses, **kwargs):
        calls = []

        def transport(url, **kw):
            calls.append((url, kw))
            item = responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return lf.OpenAIProvider("gpt-test", transport=transport, sleep=lambda s: None,
                                 **kwargs), calls

    def test_request_shape_and_parse(self):
        provider, calls = self.provider([_fake_response(200, OPENAI_OK)])
        reply = provider.complete("sys", "user")
        self.assertEqual((reply.text, reply.input_tokens, reply.output_tokens),
                         ("0.61", 120, 3))
        url, kw = calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(kw["headers"]["Authorization"], f"Bearer {self.SECRET}")
        self.assertEqual(kw["json"], {
            "model": "gpt-test", "max_completion_tokens": 1024,
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "user"}]})
        self.assertNotIn(self.SECRET, json.dumps(provider.params()))
        self.assertNotIn(self.SECRET, repr(vars(provider)))

    def test_retries_transient_then_fails_fast_on_auth(self):
        provider, calls = self.provider([_fake_response(429), _fake_response(503),
                                         _fake_response(200, OPENAI_OK)])
        self.assertEqual(provider.complete("s", "u").text, "0.61")
        self.assertEqual(len(calls), 3)
        body = {"error": {"message": f"Incorrect API key provided: {self.SECRET}"}}
        provider, calls = self.provider([_fake_response(401, body)])
        with self.assertRaises(lf.ProviderError) as ctx:
            provider.complete("s", "u")
        self.assertTrue(ctx.exception.fatal)
        self.assertEqual(str(ctx.exception), "openai HTTP 401")
        self.assertEqual(len(calls), 1)
        provider, calls = self.provider(
            [requests.ConnectionError(f"boom {self.SECRET}")] * 3)
        with self.assertRaises(lf.ProviderError) as ctx:
            provider.complete("s", "u")
        self.assertNotIn(self.SECRET, str(ctx.exception))
        self.assertEqual(len(calls), 3)

    def test_refusal_flag(self):
        refusal = {"choices": [{"message": {"content": None, "refusal": "no"},
                                "finish_reason": "stop"}]}
        provider, _ = self.provider([_fake_response(200, refusal)])
        self.assertTrue(provider.complete("s", "u").refused)

    def test_key_pasted_into_env_name_or_url_is_refused_without_echo(self):
        for kwargs in ({"api_key_env": self.SECRET},
                       {"base_url": f"https://user:{self.SECRET}@api.example.com/v1"},
                       {"base_url": f"https://api.example.com/v1?key={self.SECRET}"},
                       {"base_url": "ftp://api.example.com"}):
            with self.assertRaises(ValueError) as ctx:
                lf.make_provider("openai", "gpt-test", **kwargs)
            self.assertNotIn(self.SECRET, str(ctx.exception))
        ok = lf.make_provider("openai", "gpt-test", api_key_env="MY_KEY",
                              base_url="http://localhost:8000/v1")
        self.assertEqual((ok.env_key, ok.base_url), ("MY_KEY", "http://localhost:8000/v1"))
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
            code = lf.main(["--db", str(Path(tmp) / "none.db"), "--retro",
                            "--provider", "openai", "--model", "gpt-test",
                            "--api-key-env", self.SECRET, "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("--api-key-env must be the NAME", err.getvalue())
        self.assertNotIn(self.SECRET, err.getvalue())


class KeyNeverLeaksTests(_Db):
    SECRET = "sk-test-SECRET-key-must-never-appear-9f8e7d"

    def test_key_absent_from_db_cache_artifacts_logs_and_output(self):
        responses = [requests.ConnectionError(f"proxy refused {self.SECRET}")] * 3 + [
            _fake_response(401, {"error": {"message": f"bad key {self.SECRET}"}})] + [
            _fake_response(200, OPENAI_OK)] * 4

        def transport(url, **kw):
            item = responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        provider = lf.OpenAIProvider("gpt-test", transport=transport, sleep=lambda s: None)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": self.SECRET}), \
                self.assertLogs(level=logging.DEBUG) as logs, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            logging.getLogger("probe").info("start")
            first = self.run_fake(provider, run_id="k1", price_per_mtok=(1.0, 2.0))
            print(lf.format_summary(first))
            # The 401 aborted the first run; the second resumes and finishes.
            second = self.run_fake(provider, run_id="k1", price_per_mtok=(1.0, 2.0))
            print(lf.format_summary(second))
        self.assertEqual(first["aborted"], "openai HTTP 401")
        self.assertEqual((first["rows"], first["failures"]), (0, 2))
        self.assertEqual(second["rows"], 4)
        self.assertEqual(len(self.rows()), 4)
        haystacks = [json.dumps(self.rows()), out.getvalue(), json.dumps(first, default=str),
                     json.dumps(second, default=str)]
        haystacks += [json.dumps(vars(r), default=str) for r in logs.records]
        haystacks += [p.read_text(encoding="utf-8")
                      for p in self.dir.rglob("*") if p.is_file() and p.suffix == ".json"]
        self.assertTrue(any(p.suffix == ".json" for p in self.cache_dir.rglob("*")))
        for text in haystacks:
            self.assertNotIn(self.SECRET, text)
            self.assertNotIn("SECRET", text)


class CliTests(_Db):
    def test_dry_run_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            code = lf.main(["--db", self.db_path, "--retro", "--provider", "anthropic",
                            "--model", "claude-opus-5-5", "--limit", "50", "--dry-run",
                            "--cache-dir", str(self.cache_dir),
                            "--artifact-dir", str(self.artifacts)])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("DRY RUN", text)
        self.assertIn("cost estimate @ $4.00/$20.00 per MTok: 4 calls", text)
        self.assertIn("--- sample prompt (user) ---", text)
        self.assertFalse(self.cache_dir.exists())
        self.assertEqual(self.rows(), [])

    def test_settle_only_never_runs_as_a_dry_run(self):
        lf.run_llm_forecasts(self.db_path, lf.FakeProvider(), retro=True, run_id="s1",
                             limit=None, cache_dir=self.cache_dir,
                             artifact_dir=self.artifacts)
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE llm_forecasts SET actual_result = NULL, outcome = NULL")
        conn.commit()
        conn.close()
        before = os.path.getmtime(self.db_path)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(lf.main(["--db", self.db_path, "--settle-only", "--dry-run"]), 2)
        self.assertEqual(os.path.getmtime(self.db_path), before)
        self.assertEqual({r["outcome"] for r in self.rows()}, {None})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(lf.main(["--db", self.db_path, "--settle-only"]), 0)
        self.assertIn("'settled': 4", out.getvalue())

    def test_cli_refuses_forward_runs_and_half_prices(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(lf.main(["--db", self.db_path, "--dry-run"]), 2)
            self.assertEqual(lf.main(["--db", self.db_path, "--retro", "--dry-run",
                                      "--price-in-per-mtok", "1"]), 2)
            self.assertEqual(lf.main(["--db", str(self.dir / "missing.db"), "--retro",
                                      "--dry-run"]), 2)
        self.assertIn("--retro", err.getvalue())


if __name__ == "__main__":
    unittest.main()
