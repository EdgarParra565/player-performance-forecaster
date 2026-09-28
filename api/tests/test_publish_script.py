"""scripts/publish_db.sh --to-object-store: mock-tested against a local-dir
store (DB_SYNC_LOCAL_DIR), so no credentials or network are needed."""
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "publish_db.sh"


class PublishScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / "nba_data.db"
        with sqlite3.connect(self.db) as conn:
            conn.executescript(
                "CREATE TABLE games (game_id TEXT); CREATE TABLE game_logs (x INT);"
                "CREATE TABLE players (player_id INT); INSERT INTO games VALUES ('g1');")
        self.bucket = self.root / "bucket"
        self.env = {
            **os.environ,
            "DB_SYNC_LOCAL_DIR": str(self.bucket),
            "BUCKET_NAME": "",
            "PUBLISH_ENV_FILE": str(self.root / "absent.env"),
            "HOME": str(self.root),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def run_script(self, *args, env=None):
        return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                              env=env or self.env, timeout=60)

    def test_object_store_only_publish(self):
        r = self.run_script("--to-object-store", "--skip-git", "--db", str(self.db))
        self.assertEqual(r.returncode, 0, r.stderr)
        manifest = json.loads((self.bucket / "flagship" / "latest.json").read_text())
        self.assertTrue((self.bucket / manifest["key"]).is_file())
        self.assertEqual(manifest["row_counts"]["games"], 1)
        # Second run: unchanged, still exit 0, no new snapshot.
        r2 = self.run_script("--to-object-store", "--skip-git", "--db", str(self.db))
        self.assertEqual(r2.returncode, 0)
        self.assertIn("unchanged", r2.stdout)

    def test_dry_run_writes_nothing(self):
        r = self.run_script("--to-object-store", "--skip-git", "--dry-run", "--db", str(self.db))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.bucket.exists())

    def test_missing_db_exit_code(self):
        r = self.run_script("--to-object-store", "--skip-git", "--db", str(self.root / "nope.db"))
        self.assertEqual(r.returncode, 2)

    def test_skip_git_alone_is_refused(self):
        self.assertEqual(self.run_script("--skip-git").returncode, 2)

    def test_unconfigured_store_exit_5(self):
        env = {**self.env, "DB_SYNC_LOCAL_DIR": ""}
        r = self.run_script("--to-object-store", "--skip-git", "--db", str(self.db), env=env)
        self.assertEqual(r.returncode, 5)

    def test_env_file_sourced_and_permissions_enforced(self):
        env_file = self.root / "storage.env"
        env_file.write_text(f"DB_SYNC_LOCAL_DIR={self.root / 'from-file'}\n")
        env = {**self.env, "DB_SYNC_LOCAL_DIR": "", "PUBLISH_ENV_FILE": str(env_file)}
        os.chmod(env_file, 0o644)
        r = self.run_script("--to-object-store", "--skip-git", "--db", str(self.db), env=env)
        self.assertEqual(r.returncode, 5)
        self.assertIn("chmod 600", r.stderr)
        os.chmod(env_file, 0o600)
        r = self.run_script("--to-object-store", "--skip-git", "--db", str(self.db), env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.root / "from-file" / "flagship" / "latest.json").is_file())


if __name__ == "__main__":
    unittest.main()
