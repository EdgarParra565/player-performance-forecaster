#!/usr/bin/env bash
# Hourly NBA ETL runner — wrapper around `nba_model.data.hourly_update`.
#
# Designed to fire from launchd (com.nba.hourly.plist) or cron on the dev
# Mac. The Python entrypoint owns the actual pipeline; this wrapper just
# resolves the project venv and re-execs Python so the launchd / cron
# environment can't accidentally pick up system python3 (which has neither
# Playwright nor the project deps installed).
#
# Exit codes mirror nba_model.data.hourly_update:
#   0   ok
#   1   at least one ETL step failed
#   75  another run already holds the lockfile (EX_TEMPFAIL — retry next tick)
#   78  preflight failed (Chrome/Playwright/venv misconfig — needs human)

set -u
set -o pipefail

# Resolve the project root from the script's location so this works no matter
# where launchd / cron invokes us from.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}" || {
    echo "FATAL: cannot cd to project root: ${PROJECT_ROOT}" >&2
    exit 78
}

VENV_PY="${PROJECT_ROOT}/.venv/bin/python3"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "FATAL: project venv missing at ${VENV_PY}." >&2
    echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 78
fi

# Preflight: confirm Chrome CDP is reachable BEFORE we exec the Python entry
# (the Python module re-checks this too, but failing here gives a clear
# launchd-log message even when Python imports break).
CHROME_PORT="${CHROME_PORT:-9222}"
CHROME_HOST="${CHROME_HOST:-127.0.0.1}"
if ! curl --silent --fail --max-time 2 "http://${CHROME_HOST}:${CHROME_PORT}/json/version" > /dev/null; then
    echo "ERROR: Chrome CDP unreachable at http://${CHROME_HOST}:${CHROME_PORT}/json/version" >&2
    echo "Start Chrome with:" >&2
    echo "  open -na 'Google Chrome' --args --remote-debugging-port=${CHROME_PORT} --user-data-dir=/tmp/pp-chrome-profile" >&2
    echo "Then log in to PrizePicks/Underdog/DK/BetMGM/Caesars before the next hourly tick." >&2
    exit 78
fi

exec "${VENV_PY}" -m nba_model.data.hourly_update "$@"
