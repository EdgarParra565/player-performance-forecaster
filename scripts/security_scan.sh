#!/usr/bin/env bash
# Run the bundled Python security scans:
#   - pip-audit  : checks installed dependencies against the Python Packaging
#                  Advisory Database (CVE feed).
#   - bandit     : SAST against our own source for common issues
#                  (eval/exec, weak hashes, hardcoded secrets, sql via concat,
#                  etc).
#
# Use:
#     ./scripts/security_scan.sh           # both scans, fail on findings
#     ./scripts/security_scan.sh --report  # human-readable summary, no exit code
#
# CI:  add this as a required check; it returns non-zero on findings.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python3}"
if [ ! -x "$PY" ]; then
    PY="python3"
fi

REPORT_ONLY=0
if [ "${1:-}" = "--report" ]; then
    REPORT_ONLY=1
fi

fail=0

echo "==> pip-audit (CVEs in installed packages)"
if "$PY" -m pip_audit --strict; then
    echo "    pip-audit: OK"
else
    echo "    pip-audit: findings"
    fail=$((fail + 1))
fi
echo

echo "==> bandit (SAST on nba_model/)"
# -ll = report MEDIUM and HIGH severity only.
# -ii = report MEDIUM and HIGH confidence only.
# Excludes the test suite (security tests intentionally use 'assert', dummy
# secrets, etc.) and the desktop UI subprocess wiring.
if "$PY" -m bandit -ll -ii -r nba_model/ \
        --exclude "nba_model/tests,nba_model/simple_ui.py" \
        -q; then
    echo "    bandit: OK"
else
    echo "    bandit: findings"
    fail=$((fail + 1))
fi
echo

echo "==> grep for obvious leaks in tracked files"
# Cheap belt-and-suspenders sweep for common copy-paste mistakes.
PATTERN='sk_live_[A-Za-z0-9]\|whsec_[A-Za-z0-9]\|AKIA[0-9A-Z]\|-----BEGIN .* PRIVATE KEY-----'
if git ls-files | xargs grep -nE "$PATTERN" 2>/dev/null \
        | grep -v "secrets.toml.example\|SECURITY.md\|webhook_app.py\|stripe_helpers.py" ; then
    echo "    grep: possible secret in repo - review above lines"
    fail=$((fail + 1))
else
    echo "    grep: no obvious secrets"
fi

if [ "$REPORT_ONLY" -eq 1 ]; then
    echo
    echo "(report mode; not failing the script)"
    exit 0
fi

echo
if [ "$fail" -gt 0 ]; then
    echo "$fail scanner(s) reported findings."
    exit 1
fi
echo "All scans clean."
