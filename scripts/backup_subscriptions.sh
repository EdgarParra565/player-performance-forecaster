#!/usr/bin/env bash
# Crash-consistent backup of the subscriptions DB using SQLite's online
# backup API (NOT a raw cp - cp during a write produces a corrupt file).
#
# Use:
#     ./scripts/backup_subscriptions.sh                # writes to backups/
#     ./scripts/backup_subscriptions.sh /custom/path   # explicit dest dir
#
# Cron (run nightly at 03:15 UTC):
#     15 3 * * * cd /srv/app && ./scripts/backup_subscriptions.sh \
#                  /srv/app/backups >> /var/log/subscriptions-backup.log 2>&1
#
# The backup file is `chmod 0600` to match the source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DB="${SUBSCRIPTIONS_DB_PATH:-data/database/subscriptions.db}"
DEST_DIR="${1:-backups}"
mkdir -p "$DEST_DIR"

if [ ! -f "$DB" ]; then
    echo "no subscriptions DB at $DB; nothing to back up." >&2
    exit 0
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$DEST_DIR/subscriptions-$ts.db"

# Online backup is the canonical safe way. .backup runs through the SQLite
# driver, takes a shared lock, and writes a fresh consistent file.
sqlite3 "$DB" ".backup '$out'"
chmod 0600 "$out"

# Retention: keep last 30 backups; older are deleted. Uses lexical sort on
# the ISO timestamp in the filename.
keep=30
to_delete=$(ls -1 "$DEST_DIR"/subscriptions-*.db 2>/dev/null \
            | sort | head -n -"$keep")
if [ -n "$to_delete" ]; then
    echo "$to_delete" | xargs rm -f --
fi

echo "backup OK: $out  ($(du -h "$out" | cut -f1))"
