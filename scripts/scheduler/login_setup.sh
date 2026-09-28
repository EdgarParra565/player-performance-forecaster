#!/usr/bin/env bash
# One-time book login pass for the scraping Chrome — the owner's only manual step.
#
# Opens one tab per auth book (PrizePicks / Underdog / Pick6 / ParlayPlay /
# BetRivers + anything the last hourly run flagged login-needed) in the
# dedicated scraping Chrome, then prints a live checklist every 30s until every
# book reads ok. YOU sign in; nothing here types credentials.
#
#   scripts/scheduler/login_setup.sh                 # default 30-min timeout
#   scripts/scheduler/login_setup.sh --timeout 3600 --books prizepicks betrivers
#
# Holds the hourly lockfile while running so an hourly tick can't close the
# login tabs mid-login. Exit: 0 all ok, 3 books left, 75 lock busy, 78 no Chrome.
set -u
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}" || exit 78
VENV_PY="${PROJECT_ROOT}/.venv/bin/python3"
[[ -x "${VENV_PY}" ]] || { echo "FATAL: project venv missing at ${VENV_PY}" >&2; exit 78; }
exec "${VENV_PY}" -u -m nba_model.data.login_setup --chrome-port "${CHROME_PORT:-9222}" "$@"
