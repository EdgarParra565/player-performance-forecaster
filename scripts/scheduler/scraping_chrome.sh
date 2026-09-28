#!/usr/bin/env bash
# Keep-alive launcher for the dedicated SCRAPING Chrome (CDP on :9222).
#
# Run by launchd (com.nba.scraping-chrome.plist, KeepAlive=true): this script
# `exec`s the Chrome binary in the foreground so launchd tracks the Chrome
# process itself and relaunches it whenever it exits (crash, Cmd-Q, reboot).
#
# The profile is a DEDICATED --user-data-dir, never the owner's daily Chrome
# profile. It is persistent, so book logins survive restarts and reboots; the
# only manual act left is an occasional re-login when the hourly runner's
# session-health check alerts "re-login needed". No credentials are stored or
# typed by any script. (Chrome >= 136 also ignores --remote-debugging-port on
# the default profile, so a dedicated profile is required anyway.)
#
# Env overrides:
#   CHROME_BIN               Chrome binary (default: /Applications/Google Chrome.app/...)
#   CHROME_PORT              CDP port (default 9222)
#   NBA_SCRAPER_PROFILE_DIR  profile dir (default: ~/Library/Application Support/nba-scraping-chrome)
#   NBA_SCRAPER_POLL_SECONDS poll interval while another process owns the port (default 30)
#
# Exit codes: 78 = misconfiguration (missing binary / refused profile path).

set -u

CHROME_BIN="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
CHROME_PORT="${CHROME_PORT:-9222}"
PROFILE_DIR="${NBA_SCRAPER_PROFILE_DIR:-${HOME}/Library/Application Support/nba-scraping-chrome}"
POLL_SECONDS="${NBA_SCRAPER_POLL_SECONDS:-30}"

# Refuse the owner's daily Chrome profile (and anything inside it).
DAILY_PROFILE="${HOME}/Library/Application Support/Google/Chrome"
case "${PROFILE_DIR%/}" in
    "${DAILY_PROFILE}"|"${DAILY_PROFILE}"/*)
        echo "FATAL: refusing to use the daily Chrome profile (${PROFILE_DIR}) for scraping." >&2
        echo "Set NBA_SCRAPER_PROFILE_DIR to a dedicated directory." >&2
        exit 78
        ;;
esac

if [[ ! -x "${CHROME_BIN}" ]]; then
    echo "FATAL: Chrome binary not found/executable at ${CHROME_BIN} (set CHROME_BIN)." >&2
    exit 78
fi

# Something already answers on the CDP port (a Chrome started by hand, or a
# second copy of this agent). Launching another Chrome would just hand off and
# exit, making launchd respawn-spin. Instead wait until the port frees up, then
# exit so launchd relaunches us and we start Chrome ourselves.
if curl --silent --fail --max-time 2 "http://127.0.0.1:${CHROME_PORT}/json/version" > /dev/null 2>&1; then
    echo "CDP port ${CHROME_PORT} already served; waiting for it to free up." >&2
    while curl --silent --fail --max-time 2 "http://127.0.0.1:${CHROME_PORT}/json/version" > /dev/null 2>&1; do
        sleep "${POLL_SECONDS}"
    done
    exit 0
fi

mkdir -p "${PROFILE_DIR}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting scraping Chrome on :${CHROME_PORT} profile=${PROFILE_DIR}" >&2

exec "${CHROME_BIN}" \
    --remote-debugging-port="${CHROME_PORT}" \
    --user-data-dir="${PROFILE_DIR}" \
    --no-first-run \
    --no-default-browser-check \
    --hide-crash-restore-bubble
