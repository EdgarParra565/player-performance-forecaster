"""DB delivery path (api/db_sync.py): push -> object store -> pull + atomic swap.

Mock-tested end to end with a directory-backed store and a fake S3 client
(no network, no credentials).
"""
import gzip
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api import db_sync
from api.db_sync import LocalStore, S3Store, SyncError


def _make_db(path: str, n_games: int = 3) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE games (game_id TEXT PRIMARY KEY);"
            "CREATE TABLE game_logs (player_id INT, game_id TEXT);"
            "CREATE TABLE players (player_id INT PRIMARY KEY, name TEXT);"
        )
        conn.executemany("INSERT INTO games VALUES (?)", [(f"g{i}",) for i in range(n_games)])


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = str(self.root / "mac" / "nba_data.db")
        self.dst = str(self.root / "app" / "nba_data.db")
        Path(self.src).parent.mkdir()
        _make_db(self.src)
        self.store = LocalStore(str(self.root / "bucket"))
        self.quiet = lambda *_a, **_k: None

    def tearDown(self):
        self._tmp.cleanup()

    def games(self, path):
        with sqlite3.connect(path) as conn:
            return conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]


class PushPullTests(_Base):
    def test_push_then_pull_swaps_in_identical_db(self):
        res = db_sync.push(self.src, self.store, log=self.quiet)
        self.assertEqual(res["status"], "published")
        self.assertEqual(res["row_counts"]["games"], 3)
        manifest = db_sync.read_manifest(self.store)
        self.assertEqual(manifest["key"], res["key"])

        out = db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(out["status"], "updated")
        self.assertEqual(self.games(self.dst), 3)
        self.assertEqual(db_sync.sha256_file(self.dst), manifest["sha256"])
        # Temp files cleaned up; only the DB + marker remain.
        self.assertEqual(sorted(p.name for p in Path(self.dst).parent.iterdir()),
                         ["nba_data.db", "nba_data.db.sha256"])

    def test_unchanged_push_and_pull_are_noops(self):
        db_sync.push(self.src, self.store, log=self.quiet)
        self.assertEqual(db_sync.push(self.src, self.store, log=self.quiet)["status"], "unchanged")
        db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(db_sync.pull(self.dst, self.store, log=self.quiet)["status"], "unchanged")
        self.assertEqual(len(db_sync.snapshots(self.store)), 1)

    def test_new_push_is_picked_up_and_open_readers_survive_swap(self):
        db_sync.push(self.src, self.store, log=self.quiet)
        db_sync.pull(self.dst, self.store, log=self.quiet)
        reader = sqlite3.connect(self.dst)  # an in-flight request on the old file
        with sqlite3.connect(self.src) as conn:
            conn.execute("INSERT INTO games VALUES ('g99')")
        with patch.object(db_sync, "_utc_stamp", return_value="20990101T000000Z"):
            db_sync.push(self.src, self.store, log=self.quiet)
        self.assertEqual(db_sync.pull(self.dst, self.store, log=self.quiet)["status"], "updated")
        # Old connection still reads the old inode; new ones see the new file.
        self.assertEqual(reader.execute("SELECT COUNT(*) FROM games").fetchone()[0], 3)
        reader.close()
        self.assertEqual(self.games(self.dst), 4)

    def test_dry_run_uploads_nothing(self):
        res = db_sync.push(self.src, self.store, dry_run=True, log=self.quiet)
        self.assertEqual(res["status"], "dry-run")
        self.assertIsNone(db_sync.read_manifest(self.store))

    def test_pull_with_nothing_published(self):
        self.assertEqual(db_sync.pull(self.dst, self.store, log=self.quiet)["status"], "empty")
        self.assertFalse(Path(self.dst).exists())


class SafetyTests(_Base):
    def test_missing_and_locked_db_refused(self):
        with self.assertRaises(SyncError) as ctx:
            db_sync.push(str(self.root / "nope.db"), self.store, log=self.quiet)
        self.assertEqual(ctx.exception.code, db_sync.EXIT_NO_DB)
        writer = sqlite3.connect(self.src)
        writer.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaises(SyncError) as ctx:
                db_sync.push(self.src, self.store, log=self.quiet)
            self.assertEqual(ctx.exception.code, db_sync.EXIT_LOCKED)
        finally:
            writer.rollback()
            writer.close()

    def test_push_rejects_db_missing_required_tables(self):
        bad = str(self.root / "mac" / "bad.db")
        with sqlite3.connect(bad) as conn:
            conn.execute("CREATE TABLE games (game_id TEXT)")
        with self.assertRaises(SyncError) as ctx:
            db_sync.push(bad, self.store, log=self.quiet)
        self.assertEqual(ctx.exception.code, db_sync.EXIT_INVALID)
        self.assertIsNone(db_sync.read_manifest(self.store))

    def test_tampered_download_never_replaces_live_db(self):
        db_sync.push(self.src, self.store, log=self.quiet)
        db_sync.pull(self.dst, self.store, log=self.quiet)
        before = db_sync.sha256_file(self.dst)
        # New manifest whose object doesn't match its sha256.
        m = db_sync.read_manifest(self.store)
        m["sha256"] = "0" * 64
        self.store.put_bytes("flagship/latest.json", json.dumps(m).encode())
        with self.assertRaises(SyncError) as ctx:
            db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(ctx.exception.code, db_sync.EXIT_INVALID)
        self.assertEqual(db_sync.sha256_file(self.dst), before)
        self.assertFalse(any(p.name.startswith(".") for p in Path(self.dst).parent.iterdir()))

    def test_corrupt_sqlite_payload_rejected(self):
        junk = self.root / "junk.gz"
        payload = b"definitely not sqlite" * 100
        with gzip.open(junk, "wb") as f:
            f.write(payload)
        import hashlib
        key = "flagship/snapshots/nba_data-20990101T000000Z.db.gz"
        self.store.put_file(key, str(junk))
        meta = {"key": key, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        self.store.put_bytes("flagship/latest.json", json.dumps(meta).encode())
        with self.assertRaises(SyncError) as ctx:
            db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(ctx.exception.code, db_sync.EXIT_INVALID)
        self.assertFalse(Path(self.dst).exists())

    def test_size_cap(self):
        db_sync.push(self.src, self.store, log=self.quiet)
        with self.assertRaises(SyncError):
            db_sync.pull(self.dst, self.store, max_bytes=10, log=self.quiet)

    def test_local_store_rejects_path_traversal(self):
        with self.assertRaises(SyncError):
            self.store.put_bytes("../escape.json", b"{}")


class RetentionAndRollbackTests(_Base):
    def _push_n(self, n):
        keys = []
        for i in range(n):
            with sqlite3.connect(self.src) as conn:
                conn.execute("INSERT INTO games VALUES (?)", (f"x{i}",))
            with patch.object(db_sync, "_utc_stamp", return_value=f"2099010{i}T000000Z"):
                keys.append(db_sync.push(self.src, self.store, keep=3, log=self.quiet)["key"])
        return keys

    def test_prune_keeps_newest_n(self):
        keys = self._push_n(5)
        self.assertEqual(db_sync.snapshots(self.store), keys[-3:])
        self.assertIsNone(self.store.get_bytes(keys[0] + ".json"))

    def test_rollback_one_step_then_pull(self):
        keys = self._push_n(3)
        db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(self.games(self.dst), 6)
        meta = db_sync.rollback(self.store, steps=1, log=self.quiet)
        self.assertEqual(meta["key"], keys[1])
        db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(self.games(self.dst), 5)
        db_sync.rollback(self.store, to_key=keys[2], log=self.quiet)
        db_sync.pull(self.dst, self.store, log=self.quiet)
        self.assertEqual(self.games(self.dst), 6)

    def test_rollback_bounds(self):
        self._push_n(1)
        with self.assertRaises(SyncError):
            db_sync.rollback(self.store, steps=1, log=self.quiet)
        with self.assertRaises(SyncError):
            db_sync.rollback(self.store, to_key="flagship/snapshots/nope.db.gz", log=self.quiet)


class _FakeS3:
    """Just enough of the boto3 S3 client surface S3Store uses."""

    class _Err(Exception):
        pass

    def __init__(self):
        self.objects: dict[tuple, bytes] = {}

    def upload_file(self, path, bucket, key):
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[(bucket, key)])

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803 — boto3 names
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        from botocore.exceptions import ClientError
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, _name):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [{"Key": k} for (b, k) in outer.objects
                                    if b == Bucket and k.startswith(Prefix)]}
        return P()

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop((Bucket, Key), None)


class S3AndCliTests(_Base):
    def test_s3_store_round_trip(self):
        s3 = S3Store("bucket", client=_FakeS3())
        db_sync.push(self.src, s3, log=self.quiet)
        self.assertIsNotNone(s3.get_bytes("flagship/latest.json"))
        self.assertIsNone(s3.get_bytes("flagship/nothing.json"))
        self.assertEqual(db_sync.pull(self.dst, s3, log=self.quiet)["status"], "updated")
        self.assertEqual(self.games(self.dst), 3)

    def test_store_from_env(self):
        with patch.dict(os.environ, {"DB_SYNC_LOCAL_DIR": "", "DB_SYNC_BUCKET": "", "BUCKET_NAME": ""}):
            self.assertIsNone(db_sync.store_from_env())
        with patch.dict(os.environ, {"DB_SYNC_LOCAL_DIR": str(self.root / "b")}):
            self.assertIsInstance(db_sync.store_from_env(), LocalStore)

    def test_cli_exit_codes(self):
        with patch.dict(os.environ, {"DB_SYNC_LOCAL_DIR": "", "DB_SYNC_BUCKET": "", "BUCKET_NAME": ""}):
            self.assertEqual(db_sync.main(["pull", "--if-configured"]), 0)
            self.assertEqual(db_sync.main(["push", "--db", self.src]), db_sync.EXIT_STORAGE)
        with patch.dict(os.environ, {"DB_SYNC_LOCAL_DIR": str(self.root / "bucket")}):
            self.assertEqual(db_sync.main(["push", "--db", str(self.root / "nope.db")]),
                             db_sync.EXIT_NO_DB)
            self.assertEqual(db_sync.main(["push", "--db", self.src]), 0)
            self.assertEqual(db_sync.main(["pull", "--db", self.dst]), 0)
            self.assertEqual(self.games(self.dst), 3)
            self.assertEqual(db_sync.main(["list"]), 0)

    def test_sync_once_keeps_serving_on_error(self):
        db_sync.push(self.src, self.store, log=self.quiet)
        db_sync.pull(self.dst, self.store, log=self.quiet)
        with patch.object(db_sync, "pull", side_effect=OSError("network down")):
            self.assertEqual(db_sync.sync_once(self.dst, self.store), "error")
        self.assertEqual(db_sync.STATE.last_error, "OSError")
        self.assertEqual(self.games(self.dst), 3)  # current file untouched

    def test_background_sync_not_started_when_unconfigured(self):
        with patch.dict(os.environ, {"DB_SYNC_LOCAL_DIR": "", "DB_SYNC_BUCKET": "", "BUCKET_NAME": ""}):
            self.assertIsNone(db_sync.start_background_sync(60))
        self.assertIsNone(db_sync.start_background_sync(0))

    def test_api_serves_swapped_file(self):
        # End-to-end: the API reads whatever file is at NBA_DB_PATH now.
        from fastapi.testclient import TestClient
        from api.tests.test_api_phase2 import _seed
        seeded = str(self.root / "mac" / "seeded.db")
        _seed(seeded)
        db_sync.push(seeded, self.store, log=self.quiet)
        db_sync.pull(self.dst, self.store, log=self.quiet)
        with patch.dict(os.environ, {"NBA_DB_PATH": self.dst}):
            from api.main import app
            r = TestClient(app).get("/api/slate/kpis")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.json()["players_tracked"], 0)


if __name__ == "__main__":
    unittest.main()
