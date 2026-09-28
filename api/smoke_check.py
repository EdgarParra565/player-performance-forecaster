"""Post-deploy smoke check for the flagship UI + API (stdlib only).

    python -m api.smoke_check --base-url http://localhost:8080
    python -m api.smoke_check --base-url https://props.example.com --strict
    python -m api.smoke_check --base-url … --access-code "$FLAGSHIP_ACCESS_CODE"

Checks, against a running instance:
  - /api/health is 200, db_exists=true, and does NOT leak the server DB path
  - the SPA shell is served at / and on a client-side route (/edges)
  - core read endpoints answer 200 with the expected keys
  - bad input is a 4xx, never a 500
  - security headers (CSP, nosniff) are present
  - interactive API docs are off (--strict, i.e. public deploys)
  - when the server requires an access code: anonymous calls get 401

Exit code 0 = all passed, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


_EXTRA_HEADERS: dict[str, str] = {}


def _get(base: str, path: str, timeout: float, *, anonymous: bool = False):
    headers = {"Accept": "*/*", **({} if anonymous else _EXTRA_HEADERS)}
    req = urllib.request.Request(base.rstrip("/") + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — operator-supplied URL
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def run(base: str, *, strict: bool = False, timeout: float = 30.0) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    status, headers, body = _get(base, "/api/health", timeout)
    health = json.loads(body or b"{}") if status == 200 else {}
    check("health 200", status == 200, f"status={status}")
    check("health db_exists", health.get("db_exists") is True, str(health.get("status")))
    check("health hides db_path", health.get("db_path") in (None, ""), "")
    if health.get("access_code_required"):
        status, _, _ = _get(base, "/api/meta", timeout, anonymous=True)
        check("access code enforced (anonymous -> 401)", status == 401, f"status={status}")
        if not _EXTRA_HEADERS:
            check("access code supplied (--access-code)", False,
                  "server requires a code; pass --access-code")

    for path in ("/", "/edges"):
        status, headers, body = _get(base, path, timeout)
        check(f"SPA shell {path}", status == 200 and b'id="root"' in body, f"status={status}")
    lower = {k.lower(): v for k, v in headers.items()}
    check("CSP header", "default-src 'self'" in lower.get("content-security-policy", ""), "")
    check("nosniff header", lower.get("x-content-type-options") == "nosniff", "")

    for path, key in (
        ("/api/meta", "stats"),
        ("/api/slate/kpis", "games_in_db"),
        ("/api/slate/recent-games?n=3", "rows"),
        ("/api/slate/edges?limit=5", "rows"),
        ("/api/cross-book", "rows"),
        ("/api/players/search?q=le", "rows"),
    ):
        status, _, body = _get(base, path, timeout)
        ok = status == 200 and key in json.loads(body or b"{}")
        check(f"GET {path}", ok, f"status={status}")

    for path in ("/api/players/search?q=(", "/api/slate/edges?min_edge=nan",
                 "/api/teams/ZZZZ/chart", "/api/players/0"):
        status, _, _ = _get(base, path, timeout)
        check(f"no 500 on {path}", status < 500, f"status={status}")

    if strict:
        status, _, _ = _get(base, "/api/docs", timeout)
        check("API docs disabled", status == 404, f"status={status}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--strict", action="store_true", help="public-deploy checks (docs off)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--access-code", default="", help="value for X-Access-Code")
    args = ap.parse_args(argv)
    if args.access_code:
        _EXTRA_HEADERS["X-Access-Code"] = args.access_code
    try:
        results = run(args.base_url, strict=args.strict, timeout=args.timeout)
    except (urllib.error.URLError, OSError) as exc:
        print(f"FAIL  cannot reach {args.base_url}: {exc}")
        return 1
    failed = 0
    for name, ok, detail in results:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail and not ok else ''}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
