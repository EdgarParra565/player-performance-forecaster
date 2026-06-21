#!/usr/bin/env bash
# Publish the freshly-ETL'd nba_data.db to git so the read-only cloud deploy
# picks it up on its next redeploy. Implements docs/DEPLOYMENT.md §14 step 2.
#
# Thin wrapper around `nba_model.data.publish_db` — resolves the project venv
# and re-execs Python so cron / launchd can't accidentally use system python3.
# All flags pass straight through; the common one is `--dry-run`.
#
# Typical use (chained after a clean hourly run):
#   scripts/scheduler/hourly_update.sh && scripts/publish_db.sh
#
# Exit codes mirror nba_model.data.publish_db:
#   0  ok (published, or nothing changed)
#   2  database file not found
#   3  database locked — an ETL writer is still active (retry next tick)
#   4  a git command failed

set -u
set -o pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}" || {
    echo "FATAL: cannot cd to project root: ${PROJECT_ROOT}" >&2
    exit 2
}

VENV_PY="${PROJECT_ROOT}/.venv/bin/python3"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "FATAL: project venv missing at ${VENV_PY}." >&2
    echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 2
fi

exec "${VENV_PY}" -m nba_model.data.publish_db "$@"
