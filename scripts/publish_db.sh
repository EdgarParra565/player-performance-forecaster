#!/usr/bin/env bash
# Publish the freshly-ETL'd nba_data.db for the deployed apps.
#
# Two targets (combinable):
#   * git (default, legacy — docs/DEPLOYMENT.md §14): wraps
#     `nba_model.data.publish_db`, which commits + pushes the DB for the
#     Streamlit Cloud deploy.
#   * object store (`--to-object-store`, flagship — docs/DEPLOYMENT.md §16):
#     wraps `api.db_sync push`, which takes a consistent .backup snapshot,
#     validates it, gzips + uploads it, then repoints latest.json. The deployed
#     flagship pulls + atomically swaps it in on its next sync.
#
# Resolves the project venv and re-execs Python so cron / launchd can't
# accidentally use system python3.
#
# Typical flagship use (chained after a clean hourly/nightly run):
#   scripts/scheduler/hourly_update.sh && scripts/publish_db.sh --to-object-store --skip-git
#
# Flags handled here (everything else passes straight through):
#   --to-object-store   also push to object storage (api.db_sync push)
#   --skip-git          don't run the git publish (flagship-only deploys)
#   --dry-run           passed to both targets; nothing is written anywhere
#   --db PATH           passed to both targets
#
# Object-store credentials: exported in the environment, or kept in
# ${PUBLISH_ENV_FILE:-$HOME/.config/nba-flagship/storage.env} (chmod 600),
# which is sourced when present. Expected keys: BUCKET_NAME,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3, AWS_REGION.
#
# Exit codes (first failure wins):
#   0  ok (published, or nothing changed)
#   2  database file not found
#   3  database locked — an ETL writer is still active (retry next tick)
#   4  a git command failed
#   5  object storage not configured / upload failed
#   6  snapshot failed validation (not uploaded)

set -u
set -o pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}" || {
    echo "FATAL: cannot cd to project root: ${PROJECT_ROOT}" >&2
    exit 2
}

VENV_PY="${PUBLISH_PYTHON:-${PROJECT_ROOT}/.venv/bin/python3}"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "FATAL: project venv missing at ${VENV_PY}." >&2
    echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 2
fi

to_store=0
skip_git=0
passthrough=()
store_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --to-object-store) to_store=1 ;;
        --skip-git) skip_git=1 ;;
        --dry-run) passthrough+=("$1"); store_args+=("$1") ;;
        --db)
            if [[ $# -lt 2 ]]; then echo "FATAL: --db needs a path" >&2; exit 2; fi
            passthrough+=("$1" "$2"); store_args+=("$1" "$2"); shift ;;
        *) passthrough+=("$1") ;;
    esac
    shift
done

if [[ ${skip_git} -eq 1 && ${to_store} -eq 0 ]]; then
    echo "FATAL: --skip-git without --to-object-store would publish nothing" >&2
    exit 2
fi

rc=0
if [[ ${skip_git} -eq 0 ]]; then
    "${VENV_PY}" -m nba_model.data.publish_db ${passthrough[@]+"${passthrough[@]}"}
    rc=$?
fi

if [[ ${to_store} -eq 1 && ${rc} -eq 0 ]]; then
    env_file="${PUBLISH_ENV_FILE:-${HOME}/.config/nba-flagship/storage.env}"
    if [[ -f "${env_file}" ]]; then
        # Refuse a world/group-readable credentials file.
        perms="$(stat -f '%Lp' "${env_file}" 2>/dev/null || stat -c '%a' "${env_file}")"
        if [[ "${perms}" != "600" && "${perms}" != "400" ]]; then
            echo "FATAL: ${env_file} must be chmod 600 (is ${perms})" >&2
            exit 5
        fi
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi
    "${VENV_PY}" -m api.db_sync push ${store_args[@]+"${store_args[@]}"}
    rc=$?
fi

exit ${rc}
