"""Deliver the nightly SQLite snapshot from the Mac to the deployed flagship.

Design (docs/DEPLOYMENT.md §16): the Mac PUSHES a validated, gzipped snapshot
to S3-compatible object storage (Tigris on Fly.io); the deployed app PULLS it
on boot and every ``DB_SYNC_INTERVAL_SECONDS``, swapping it in atomically. The
API itself stays read-only — no upload endpoint exists.

Storage layout under ``DB_SYNC_PREFIX`` (default ``flagship/``)::

    snapshots/nba_data-<UTC>.db.gz        immutable snapshot
    snapshots/nba_data-<UTC>.db.gz.json   its metadata (sha256, size, rows)
    latest.json                           pointer to the live snapshot

``latest.json`` is written LAST, so a reader never sees a half-uploaded
snapshot. Rollback rewrites the pointer to an older snapshot.

Pull swap: download → temp file in the DB's directory → stream-gunzip while
hashing → verify sha256 → ``PRAGMA quick_check`` + required tables → fsync →
``os.replace`` onto ``NBA_DB_PATH``. The API opens a fresh connection per
request, so in-flight requests finish on the old inode and new requests see
the new file.

CLI (Mac side, via ``scripts/publish_db.sh --to-object-store``)::

    python -m api.db_sync push   [--db data/database/nba_data.db] [--dry-run]
    python -m api.db_sync list
    python -m api.db_sync rollback [--to KEY | --steps 1]
    python -m api.db_sync pull   [--db /data/nba_data.db]   # app side

Exit codes: 0 ok / unchanged · 2 DB missing · 3 DB locked (retry next tick)
· 5 storage error / not configured · 6 snapshot failed validation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from . import config

logger = logging.getLogger("api.db_sync")

EXIT_OK, EXIT_NO_DB, EXIT_LOCKED, EXIT_STORAGE, EXIT_INVALID = 0, 2, 3, 5, 6
CHUNK = 1024 * 1024
DEFAULT_KEEP = 7
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024  # refuse to fill the disk with a 1 GB+ file


class SyncError(RuntimeError):
    def __init__(self, message: str, code: int = EXIT_STORAGE):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

class Store(Protocol):
    def put_file(self, key: str, path: str) -> None: ...
    def get_file(self, key: str, path: str) -> None: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> Optional[bytes]: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def delete(self, key: str) -> None: ...


class LocalStore:
    """Directory-backed store (tests; also usable over a shared mount)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise SyncError(f"key escapes store root: {key!r}")
        return p

    def put_file(self, key, path):
        dest = self._p(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with open(path, "rb") as src, open(tmp, "wb") as out:
            while chunk := src.read(CHUNK):
                out.write(chunk)
        os.replace(tmp, dest)

    def get_file(self, key, path):
        src = self._p(key)
        if not src.is_file():
            raise SyncError(f"missing object {key}")
        with open(src, "rb") as s, open(path, "wb") as out:
            while chunk := s.read(CHUNK):
                out.write(chunk)

    def put_bytes(self, key, data):
        dest = self._p(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    def get_bytes(self, key):
        p = self._p(key)
        return p.read_bytes() if p.is_file() else None

    def list_keys(self, prefix):
        base = self.root
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(base)) for p in base.rglob("*")
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
            and not p.name.endswith(".part")
        )

    def delete(self, key):
        try:
            self._p(key).unlink()
        except FileNotFoundError:
            pass


class S3Store:
    """S3-compatible store (Tigris, R2, S3). Credentials come from the
    standard AWS env vars that ``fly storage create`` sets on the app:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3,
    AWS_REGION, BUCKET_NAME."""

    def __init__(self, bucket: str, client=None):
        self.bucket = bucket
        if client is None:
            import boto3  # imported lazily: only needed when syncing
            client = boto3.client(
                "s3",
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3") or None,
                region_name=os.environ.get("AWS_REGION") or "auto",
            )
        self.client = client

    def put_file(self, key, path):
        self.client.upload_file(path, self.bucket, key)

    def get_file(self, key, path):
        self.client.download_file(self.bucket, key, path)

    def put_bytes(self, key, data):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data,
                               ContentType="application/json")

    def get_bytes(self, key):
        from botocore.exceptions import ClientError
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise

    def list_keys(self, prefix):
        keys: list[str] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=key)


def store_from_env() -> Optional[Store]:
    """``DB_SYNC_LOCAL_DIR`` (tests / shared mount) or S3 via ``BUCKET_NAME``."""
    local = os.environ.get("DB_SYNC_LOCAL_DIR", "").strip()
    if local:
        return LocalStore(local)
    bucket = (os.environ.get("DB_SYNC_BUCKET") or os.environ.get("BUCKET_NAME") or "").strip()
    if bucket:
        return S3Store(bucket)
    return None


def _prefix() -> str:
    p = os.environ.get("DB_SYNC_PREFIX", "flagship/").strip()
    return p if not p or p.endswith("/") else p + "/"


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def validate_sqlite(path: str) -> dict:
    """Read-only integrity check + required tables. Returns row counts."""
    uri = f"{Path(path).resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as conn:
            ok = conn.execute("PRAGMA quick_check").fetchone()[0]
            if ok != "ok":
                raise SyncError(f"quick_check failed: {ok}", EXIT_INVALID)
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            missing = config.REQUIRED_TABLES - names
            if missing:
                raise SyncError(f"missing tables: {sorted(missing)}", EXIT_INVALID)
            return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608 — fixed names
                    for t in sorted(config.REQUIRED_TABLES)}
    except sqlite3.DatabaseError as exc:
        raise SyncError(f"not a valid SQLite database: {exc}", EXIT_INVALID) from exc


def db_is_locked(db_path: str) -> bool:
    """True if a writer holds the DB (mirrors nba_model.data.publish_db)."""
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
            return False
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return True


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_manifest(store: Store) -> Optional[dict]:
    raw = store.get_bytes(_prefix() + "latest.json")
    return json.loads(raw) if raw else None


# ---------------------------------------------------------------------------
# Push (Mac)
# ---------------------------------------------------------------------------

def push(db_path: str, store: Store, *, keep: int = DEFAULT_KEEP, dry_run: bool = False,
         log=print) -> dict:
    if not Path(db_path).is_file():
        raise SyncError(f"database not found: {db_path}", EXIT_NO_DB)
    if db_is_locked(db_path):
        raise SyncError(f"{db_path} is locked — an ETL writer is active; retry next tick",
                        EXIT_LOCKED)
    with tempfile.TemporaryDirectory(prefix="db_sync_") as tmp:
        snap = os.path.join(tmp, "snapshot.db")
        src = sqlite3.connect(db_path)
        try:
            dst = sqlite3.connect(snap)
            try:
                src.backup(dst)  # consistent online copy, even mid-read
            finally:
                dst.close()
        finally:
            src.close()
        rows = validate_sqlite(snap)
        digest = sha256_file(snap)
        current = read_manifest(store)
        if current and current.get("sha256") == digest:
            log("unchanged: live snapshot already has this content")
            return {"status": "unchanged", "key": current.get("key"), "sha256": digest}

        stamp = _utc_stamp()
        key = f"{_prefix()}snapshots/nba_data-{stamp}.db.gz"
        meta = {
            "key": key, "sha256": digest, "size": os.path.getsize(snap),
            "created_utc": stamp, "row_counts": rows, "publisher": socket.gethostname(),
        }
        if dry_run:
            log(f"dry-run: would upload {key} ({meta['size']} bytes) and repoint latest.json")
            return {"status": "dry-run", **meta}
        gz = snap + ".gz"
        with open(snap, "rb") as f_in, gzip.open(gz, "wb", compresslevel=6) as f_out:
            while chunk := f_in.read(CHUNK):
                f_out.write(chunk)
        store.put_file(key, gz)
        store.put_bytes(key + ".json", json.dumps(meta).encode())
        store.put_bytes(_prefix() + "latest.json", json.dumps(meta).encode())  # pointer LAST
        log(f"published {key} ({meta['size']} bytes, sha256 {digest[:12]}…)")
        pruned = prune(store, keep=keep, protect=key)
        if pruned:
            log(f"pruned {len(pruned)} old snapshot(s)")
        return {"status": "published", **meta, "pruned": pruned}


def snapshots(store: Store) -> list[str]:
    return [k for k in store.list_keys(_prefix() + "snapshots/") if k.endswith(".db.gz")]


def prune(store: Store, *, keep: int, protect: str) -> list[str]:
    keys = snapshots(store)
    doomed = [k for k in keys[:-keep] if k != protect] if keep > 0 else []
    for k in doomed:
        store.delete(k)
        store.delete(k + ".json")
    return doomed


def rollback(store: Store, *, to_key: Optional[str] = None, steps: int = 1, log=print) -> dict:
    keys = snapshots(store)
    current = read_manifest(store)
    if to_key is None:
        if not current or current.get("key") not in keys:
            raise SyncError("no current snapshot to roll back from")
        idx = keys.index(current["key"]) - int(steps)
        if idx < 0:
            raise SyncError(f"only {len(keys)} snapshot(s) retained; cannot go back {steps}")
        to_key = keys[idx]
    raw = store.get_bytes(to_key + ".json")
    if to_key not in keys or not raw:
        raise SyncError(f"unknown snapshot {to_key}")
    store.put_bytes(_prefix() + "latest.json", raw)
    log(f"latest.json -> {to_key} (app picks it up on its next sync)")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Pull + atomic swap (app)
# ---------------------------------------------------------------------------

def _marker(db_path: str) -> Path:
    return Path(db_path).with_name(Path(db_path).name + ".sha256")


def pull(db_path: str, store: Store, *, max_bytes: int = DEFAULT_MAX_BYTES, log=print) -> dict:
    manifest = read_manifest(store)
    if not manifest:
        log("no snapshot published yet")
        return {"status": "empty"}
    want = manifest["sha256"]
    db = Path(db_path)
    marker = _marker(db_path)
    if db.is_file() and marker.is_file() and marker.read_text().strip() == want:
        return {"status": "unchanged", "sha256": want}
    if int(manifest.get("size") or 0) > max_bytes:
        raise SyncError(f"snapshot is {manifest.get('size')} bytes > cap {max_bytes}", EXIT_INVALID)

    db.parent.mkdir(parents=True, exist_ok=True)
    gz_tmp = db.with_name(f".{db.name}.download")
    incoming = db.with_name(f".{db.name}.incoming")
    try:
        store.get_file(manifest["key"], str(gz_tmp))
        h = hashlib.sha256()
        written = 0
        with gzip.open(gz_tmp, "rb") as src, open(incoming, "wb") as out:
            while chunk := src.read(CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise SyncError("decompressed snapshot exceeds size cap", EXIT_INVALID)
                h.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if h.hexdigest() != want:
            raise SyncError("sha256 mismatch — corrupt or tampered download", EXIT_INVALID)
        validate_sqlite(str(incoming))
        os.replace(incoming, db)  # atomic on the same filesystem
        tmp_marker = marker.with_name(marker.name + ".tmp")
        tmp_marker.write_text(want)
        os.replace(tmp_marker, marker)
        log(f"swapped in {manifest['key']} ({written} bytes)")
        return {"status": "updated", "sha256": want, "key": manifest["key"]}
    finally:
        for p in (gz_tmp, incoming):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


@dataclass
class SyncState:
    last_status: str = "never"
    last_ok_utc: Optional[str] = None
    last_error: Optional[str] = None


STATE = SyncState()
_THREAD: Optional[threading.Thread] = None


def sync_once(db_path: Optional[str] = None, store: Optional[Store] = None) -> str:
    store = store or store_from_env()
    if store is None:
        STATE.last_status = "not-configured"
        return STATE.last_status
    try:
        result = pull(db_path or config.get_db_path(), store, log=logger.info)
        STATE.last_status = result["status"]
        STATE.last_ok_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        STATE.last_error = None
    except Exception as exc:  # noqa: BLE001 — keep serving the current file
        STATE.last_status = "error"
        STATE.last_error = type(exc).__name__
        logger.warning("db sync failed (%s): %s", type(exc).__name__, exc)
    return STATE.last_status


def start_background_sync(interval_seconds: float) -> Optional[threading.Thread]:
    """Poll every ``interval_seconds`` (daemon thread; one per process)."""
    global _THREAD
    if interval_seconds <= 0 or store_from_env() is None or (_THREAD and _THREAD.is_alive()):
        return None

    def loop():
        # Sync first (usually "unchanged" right after the boot-time pull) so
        # /api/health reports a real status, then poll.
        while True:
            sync_once()
            time.sleep(interval_seconds)

    if not logger.handlers:  # uvicorn doesn't configure non-uvicorn loggers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("db_sync: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    _THREAD = threading.Thread(target=loop, name="db-sync", daemon=True)
    _THREAD.start()
    return _THREAD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m api.db_sync", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_push = sub.add_parser("push", help="snapshot + upload the local DB (Mac)")
    p_push.add_argument("--db", default="data/database/nba_data.db")
    p_push.add_argument("--keep", type=int, default=int(os.environ.get("DB_SYNC_KEEP", DEFAULT_KEEP)))
    p_push.add_argument("--dry-run", action="store_true")
    p_pull = sub.add_parser("pull", help="download + atomically swap in (app)")
    p_pull.add_argument("--db", default=None)
    p_pull.add_argument("--if-configured", action="store_true",
                        help="exit 0 quietly when no storage is configured")
    sub.add_parser("list", help="list retained snapshots")
    p_rb = sub.add_parser("rollback", help="repoint latest.json to an older snapshot")
    p_rb.add_argument("--to", dest="to_key", default=None)
    p_rb.add_argument("--steps", type=int, default=1)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store = store_from_env()
    if store is None:
        if getattr(args, "if_configured", False):
            print("db sync not configured; skipping")
            return EXIT_OK
        print("FATAL: no storage configured (set BUCKET_NAME + AWS_* or DB_SYNC_LOCAL_DIR)",
              file=sys.stderr)
        return EXIT_STORAGE
    try:
        if args.cmd == "push":
            push(args.db, store, keep=args.keep, dry_run=args.dry_run)
        elif args.cmd == "pull":
            pull(args.db or config.get_db_path(), store)
        elif args.cmd == "list":
            live = (read_manifest(store) or {}).get("key")
            for k in snapshots(store):
                print(("* " if k == live else "  ") + k)
        elif args.cmd == "rollback":
            rollback(store, to_key=args.to_key, steps=args.steps)
    except SyncError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001 — storage/network failure
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_STORAGE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
