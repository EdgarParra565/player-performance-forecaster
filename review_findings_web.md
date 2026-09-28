# Web-half review findings (2026-09-28)

Scope: `api/**`, `frontend/**`, `nba_model/web/**`, `docs/`, Docker/compose.
Companion to `review_findings_data.md` (data half, other agent).

Format: `path:line — SEVERITY — problem — fix — STATUS`
STATUS: **FIXED** (with regression test) · **PROPOSED** (judgment call / follow-up) ·
**CROSS-BOUNDARY** (file outside this agent's ownership; not edited).

## Summary

| | FIXED | PROPOSED | CROSS-BOUNDARY |
|---|---|---|---|
| Critical | 4 | 0 | 0 |
| Major | 26 | 0 | 1 (`db_manager` schema-on-open) |
| Minor / info | 29 | 7 (6 deferred with billing) | 0 |

Test counts after this pass: **nba_model 862** (778 baseline + 28 new in
`test_review_web_security.py`; the remaining +56 came from a concurrent agent's
work in `nba_model/data|model|…`, which this pass did not touch), **api 47**
(23 → 47), **frontend vitest 66** (14 → 66).

**Update (Phase 3 pass, 2026-09-28):** W-DEP-1/2 FIXED (optional access
code + in-process per-IP limiter), error hygiene, IP-keyed throttle, echarts 6,
and the Task 3 cleanups FIXED. Stripe/webhook minors stay PROPOSED, **deferred
with billing**. Counts after that pass: **nba_model 887** (+7 in
`test_review_web_security.py`; the rest came from the concurrent data agent),
**api 71**, **frontend vitest 86**. See §7.

**Public-deploy prerequisites:** W-SEC-1…3 below are required before any public
deploy of the Streamlit app. The flagship UI/API has **no auth and no rate
limiting** yet (W-DEP-1/2), so it has to run behind a TLS reverse proxy or CDN with
per-IP limits. See `docs/DEPLOYMENT.md` §15.

---

## 1. Security

- `nba_model/web/app.py:2172` (was :2169/:2206/:2266) — **CRITICAL** — W-SEC-1: the manual-lines "Save to DB" gate was `BILLING_ENABLED and not is_admin`. With billing OFF (the launch default), any anonymous visitor could write into `betting_lines`. — Added `_manual_lines_save_allowed()` (admin-only, regardless of billing). It is used for the warning, the disabled button, and a server-side re-check in the save branch, the same pattern as the Operations gate. — **FIXED** (`ManualLinesSaveGateTests`: billing on/off, forced submit, admin can save).
- `nba_model/web/webhook_app.py:326` (was :386-393) — **CRITICAL** — W-SEC-2: premium was granted on `customer.subscription.created/updated` whatever the status, including past_due, incomplete and unpaid. `checkout.session.completed` also granted premium with no payment check. — New `tier_for_event()`:
  - subscription events → premium only for `active`/`trialing`; every other status, including unknown ones, → free (fail closed)
  - checkout → premium only when `payment_status == "paid"`; anything else makes no change, and the subscription event that follows is authoritative

  — **FIXED** (`WebhookStatusGateTests`). Two stress fixtures in `nba_model/tests/test_security_stress.py` now include `"payment_status": "paid"` (they encoded the old behaviour).
- `nba_model/web/auth.py:198` (was :119-129) — **CRITICAL** — W-SEC-3: `is_admin()`/`tier_for()` matched only the `email` claim. A multi-tenant Microsoft login allows an attacker-controlled email ("nOAuth"), which meant admin takeover (Operations runs host subprocesses) and theft of a paying user's tier and Stripe portal. — Admin is now bound to OIDC (`iss`, `sub`) via `[auth].admin_identities`, or to an email in `[auth].admins` only when it is **verified**. Verified means `email_verified: true`, or the issuer is listed in `[auth].trusted_email_issuers`. `current_user()` drops unverified emails, so they get no tier lookup, no checkout prefill and no portal. `tier_for(email, admin=…)` no longer grants premium from the email string. — **FIXED** (`AdminIdentityBindingTests`, 8 cases). Documented in `docs/SECURITY.md` → "Admin identity binding (nOAuth)"; `.streamlit/secrets.toml.example` updated.
- `frontend/src/components/LineMovementPanel.tsx` (old :147) — **CRITICAL** — Scrubbing, then switching to a player or stat with fewer snapshots, crashed the whole app (`timestamps[idx]` was undefined → `shortTs(undefined).slice`) and there was no error boundary. — The index is now clamped during render (`clampIndex`), the panel is keyed by player+stat, and the route has an `errorElement` plus a chart error boundary. — **FIXED** (`LineMovementPanel.test.tsx`).
- `nba_model/web/webhook_app.py:475` — **MAJOR** — The event id was recorded *before* it was applied. Any non-ValueError failure (such as "database is locked") returned 500, and Stripe's retry was then dropped as a duplicate: a lost cancellation. — Processing moved into `_apply_event`. On failure, `subscriptions.forget_stripe_event()` un-records the event and the handler returns 5xx. — **FIXED** (`test_processing_failure_unrecords_event`).
- `nba_model/web/webhook_app.py:252` — **MAJOR** — A failed `Customer.retrieve` (network, or a missing `STRIPE_SECRET_KEY`) was swallowed as "no email" and acknowledged with 200, so the event was lost for good. — It now raises `EmailLookupError`, which returns 503 and un-records the event so the retry is processed. The tier is decided *before* any Stripe call, so event types that change nothing skip the lookup. — **FIXED** (`test_customer_lookup_failure_is_retryable_not_dropped`).
- `nba_model/web/webhook_app.py:500` + `subscriptions.py:121` — **MAJOR** — There was no ordering guard. Stripe doesn't guarantee delivery order, so an older event could overwrite a newer one. — New `last_event_created` column (SQLite + Postgres migrations); events older than the stored value are ignored with `stale`. — **FIXED** (`test_out_of_order_event_does_not_overwrite_newer_state`).
- `nba_model/web/webhook_app.py:510` — **MAJOR** — Downgrades were applied by email even when they came from a *different* subscription. Example: the old subscription's period-end cancellation arrives after a re-subscribe. — A downgrade for a `sub_` id other than the stored one is ignored. Only `sub_…` ids are stored now, never invoice or session ids. — **FIXED** (`test_old_subscription_cancel_does_not_revoke_new_one`).
- `nba_model/web/webhook_app.py:130,380` — **MAJOR** — The per-IP limiter counted *every* request before the signature check, and `WEBHOOK_TRUSTED_PROXY_HOPS` defaults to 0, so behind a proxy all traffic shared one bucket. An unsigned flood could 429 real Stripe deliveries. — The limiter now only *peeks* up front. Only rejected requests (bad or missing signature, bad payload, oversize, slow body) count. — **FIXED** (`test_valid_deliveries_do_not_consume_rate_limit`).
- `nba_model/web/app.py` admin paths — **INFO** — Checked, no action needed: Operations (:2600) and the Admin dashboard (:2613) re-check `is_admin()` on `?view=` deep links; the DB-path override only renders for admins; view options come from per-user lists. All of these now inherit the W-SEC-3 identity binding. — no change.
- Secrets in logs — **MINOR** — `_email_from_event` logged the raw exception from `Customer.retrieve`, which can echo request details. — It now logs the exception type only. — **FIXED**. `nba_model/web/app.py` showed raw `st.error(f"…{exc}")` text (paths, SQL; 17 sites incl. the teams-list error that printed `db_path`) to visitors. — `_public_error()`: a generic message plus a short incident ref on the page, with the full exception + traceback in the server log. Our own `ValidationError` messages ("Invalid line: …") are still shown. — **FIXED** (`test_public_error_hides_exception_text`).
- `frontend/src/components/chartTheme.ts` — **MAJOR** — ECharts tooltip formatters are rendered as innerHTML, and they interpolated DB/scraped strings (`opponent`, book names) unescaped. That is stored XSS if a scraped value carries markup. — The shared `tooltipHtml()` builder escapes every label and value. — **FIXED** (`charts.test.ts`: `<img onerror>` opponent is escaped).
- `frontend/src/lib/csv.ts` — **MINOR** — CSV export allowed spreadsheet formula injection (`=HYPERLINK(…)` in a scraped name) and didn't quote `\r`. — Such cells are now prefixed with `'`, and CR is quoted. — **FIXED** (`csv.test.ts`).
- `frontend/index.html` — **MINOR** — Fonts loaded from Google Fonts: a third-party request on every visit, broken offline, and in conflict with a strict CSP. — Self-hosted via `@fontsource-variable/{inter,jetbrains-mono}`, and Vite never inlines fonts as `data:` URIs (`vite.config.ts` `assetsInlineLimit`). — **FIXED** (`tokens.test.ts` asserts no font CDN).
- `api/main.py:68` — **MAJOR** (public deploy) — The API sent no security headers. — Added CSP (`default-src 'self'`, no inline or third-party script/font/connect, `frame-ancestors 'none'`), nosniff, DENY, no-referrer, Permissions-Policy and COOP. Added optional `API_TRUSTED_HOSTS` (DNS-rebinding). `API_DOCS=0` turns off Swagger. — **FIXED** (`test_api_static.py`). Verified in a real browser: zero CSP violations across all five views.
- `frontend/package-lock.json` — **MAJOR** — `npm audit`: react-router high (RSC CSRF), plus nanoid and postcss. — `npm audit fix` (non-breaking). — **FIXED**. echarts moderate XSS: upgraded to **echarts 6.1** (`npm audit --omit=dev` = 0). The lazy `echarts` chunk is still split out, and all 7 views were re-screenshotted with zero console errors. — **FIXED**. The dev-only vitest advisory is cleared by vitest 4 (all 99 tests pass; `npm audit` = 0 including dev). — **FIXED**.

## 2. API contract

- `api/main.py:124` + `nba_model/data/database/db_manager.py:111-136` — **MAJOR** — The "read-only" API wrote: every `DatabaseManager()` open runs `schema.sql` plus migrations. Pointing `NBA_DB_PATH` at an empty file created 52 schema objects in it. — The API now runs a read-only preflight (`sqlite3 …?mode=ro`, required tables present, cached by mtime) *before* the data layer is allowed to open the file. Otherwise it returns 503 "database unavailable". — **FIXED** (`test_empty_file_is_503_and_not_written`). The root cause is in `db_manager` (schema-on-open with no read-only mode). — **CROSS-BOUNDARY** (proposed: a `read_only=True` open mode).
- `api/main.py:108` — **MAJOR** — A locked, corrupt or schema-drifted DB surfaced as a raw 500, and `/api/health` itself could 500. — A global `sqlite3.Error` handler now returns 503 with a generic detail, and health reports `degraded`. — **FIXED** (`test_garbage_file_is_503`).
- `api/services.py:293` — **MAJOR** — Player search used a regex `str.contains`: `?q=(` returned 500 and `.` acted as a wildcard. `q` was also unbounded. — Now `regex=False` with `Query(max_length=64)`. — **FIXED**.
- `api/main.py:202` — **MAJOR** — The `name` query param overrode the DB name. Book lines are matched by name, so `/api/players/1?name=Other Guy` mixed one player's logs with another's lines. — The name is always resolved from the DB. A mismatched `name` is a 400; matching is accent- and case-insensitive. The frontend no longer sends it. — **FIXED**.
- `api/services.py:600` — **MAJOR** — Line movement treated a book's **alt-line ladder inside one snapshot** as drift. For example, Jokić showed "Bovada 24.5 → 33.5, +9.0", which was fabricated. — `_main_line()` collapses each timestamp to its most balanced-odds line (median if unpriced). — **FIXED** (`test_line_movement_collapses_alt_ladder_to_main_line`).
- `api/main.py:166` — **MINOR** — `player_id` was unbounded: a 2⁶³ id gave a 500. — Now `ge=1, le=2³¹-1`. — **FIXED**.
- `api/main.py` — **MINOR** — More loose inputs, all fixed:
  - `min_edge`, `min_p_over` and `min_gap` accepted NaN/inf; they now have finite bounds (NaN/inf → 422)
  - the books list was unbounded; now ≤ 25 items of ≤ 40 characters
  - `season_type` is allowlisted
  - `steals`/`blocks`/`turnovers` returned fake 0.0 series; player and team charts now allow only chartable stats
  - `n_games ≥ 3` (σ = 0 at n = 1 flagged every line +EV)

  — **FIXED** (`test_api_hardening.py`).
- `api/main.py` `line_movement` — **MINOR** — An unknown player gave 200 with empty data, while player detail gave 404. — Now 404. — **FIXED**.
- `api/services.py:461,724` — **MINOR** — A window with no games returned `mu`/`sigma = 0.0`, which rendered as a fake projection. — Now `null`. — **FIXED**.
- `api/services.py:78,100` — **MINOR** — `_freshest_scrape` took a string `max` over mixed formats (`YYYY-MM-DD HH:MM:SS` vs ISO `T…+00:00`) and returned naive strings that browsers read as local time. `prop_lines_recent` compared ISO text against `datetime('now')`. — Values are normalized to UTC ISO with an explicit offset, and the comparison uses `datetime()`. — **FIXED** (`test_freshest_scrape_is_utc_iso_with_offset`).
- `api/main.py` health / 503 bodies — **MINOR** — They exposed the absolute server DB path. — The path is hidden unless `API_EXPOSE_DB_PATH=1`. — **FIXED**.
- `api/services.py` (per request) — **MINOR** — A single request opens `DatabaseManager` 3–6 times, each rerunning the schema script, which invites "database is locked" during ETL commits (these now map to 503). — Open one read-only connection per request. — **PROPOSED** (needs the CROSS-BOUNDARY read-only mode above).
- CORS — **INFO** — CORS only allows localhost:5173, GET, no credentials. That is fine: production is same-origin (§4). — no change.

## 3. Frontend correctness

- `frontend/src/lib/format.ts` — **MAJOR** — Date-only API strings (`"2026-06-13"`) were parsed as UTC midnight, so US users saw the day before ("Jun 12"; visible in the *before* screenshot). Naive DB timestamps were read as local time, putting freshness off by the UTC offset. — New `parseApiDate`: a date-only string is a local calendar date, a naive timestamp is UTC, an explicit offset is honoured. — **FIXED** (`format.test.ts`).
- `frontend/src/views/PlayerDetail.tsx` `NameResolver` — **MAJOR** — The `?name=` deep link fell back to `rows[0]` of a fuzzy search, which could silently load a different player's EV. — Only an exact accent- and case-insensitive match auto-selects; otherwise a "No exact match" state lists candidates. — **FIXED**.
- `frontend/src/api/hooks.ts` — **MAJOR** — `keepPreviousData` held player A's KPIs and EV table under player B's name while B loaded (same for team and line movement). — Placeholder data is now kept only for the *same* entity, and the view dims while the placeholder shows. — **FIXED**.
- `frontend/src/main.tsx` — **MAJOR** — There was no `errorElement` or error boundary, so any render error blanked the app. — Added `RouteError` and `ChartBoundary`. — **FIXED**.
- Error handling (all views, LineMovementPanel) — **MAJOR** — A failed request rendered as "no data" (line movement) or as a generic message that dropped the API `detail`. `meta` errors were ignored. Slate KPI skeletons pulsed forever on error. — `ErrorState` and `errorMessage()` now carry the API detail everywhere; skeletons show only while loading. — **FIXED**.
- `frontend/src/main.tsx` QueryClient — **MINOR** — It retried 4xx responses, and `staleTime` was 30s against the API's `max-age=60`. — No retry on 4xx; `staleTime` is 60s. — **FIXED**.
- `frontend/src/components/DataTable.tsx` — **MAJOR** (a11y) — Clickable rows weren't keyboard-reachable, sortable headers weren't buttons and had no `aria-sort`, and a null/null sort compare was inconsistent. — Rows now have `tabIndex`, Enter/Space and an accessible label; headers are `<button>`s with `aria-sort`; null/null returns 0. — **FIXED** (`controls.test.tsx`).
- Focus visibility — **MAJOR** (a11y) — Every focus ring was removed (`focus:outline-none`). — A global `:focus-visible` accent ring was added. `faint` text went from 2.8:1 to ≈4.7:1 contrast. — **FIXED**.
- Player picker — **MINOR** (a11y) — The picker had no label or combobox semantics, no keyboard navigation, and no debounce. — It is now a combobox with a listbox, arrow/Enter/Escape handling, close on blur, and a 200ms debounce. — **FIXED**.
- `frontend/src/components/controls.tsx` `NumberField` — **MINOR** — The value was clamped on every keystroke, so "12" couldn't be typed when min=2 and the field couldn't be cleared. — The field keeps a local draft and clamps on blur or Enter. — **FIXED** (`controls.test.tsx`). This also removes the per-keystroke edge scans.
- Replay / ECharts — **MINOR** — `notMerge` rebuilt every series each replay tick, and options weren't memoized, so every keystroke called `setOption`. — Replay now uses `notMerge={false}` with animation off while playing; all options are `useMemo`'d, and the builders were moved to `src/charts/`. ECharts disposal and resize are handled by echarts-for-react (no custom `init` or listeners). — **FIXED**.
- Polarity misuse — **MINOR** — UNDER was coloured red (it read as −EV); book series reused the pos green and a near-neg red; the per-book table showed a best-side Edge next to an *over* EV (contradictory green/red). — `SideTag` is neutral, the categorical palette contains no polarity hues (a test enforces this), and EV is shown for the best side. — **FIXED**.
- `rowKey` — **MINOR** — `book-player-stat` collides on alt lines. — `book_line` added to the key. — **FIXED**.
- KPI cards — **MINOR** — They showed "0" while loading or on error. — Now "—". — **FIXED**.
- TS escapes — **INFO** — No `any` or `@ts-ignore`; the `as never` markLine cast was replaced with `MarkLineComponentOption["data"]`. The remaining `!` assertions (`main.tsx` root, `hooks.ts` enabled-guarded params) are safe. — **FIXED**.
- `LineSparkline*` — **INFO** — Dead code. — Removed. — **FIXED**.
- URL state — **MINOR** — Player and stat are now URL-driven (`/player/:id?stat=`), so back/forward and sharing work. `n_games`/`rolling` are now URL state too (`?n=&roll=`, bounded by `clampInt`). — **FIXED**.

## 4. Streamlit billing-gate bypasses (app.py)

- `nba_model/web/app.py:1860` — **MAJOR** — "Compare players" (a free view) listed every player and never called `gate_player`, so free users could get past the preview paywall. — The picker is filtered to the preview set for non-premium users, and each pick is re-checked. — **FIXED** (`ComparePlayersGateTests`).
- `nba_model/web/app.py:2151` — **MAJOR** — The scan throttle applied only to non-premium users. With `BILLING_ENABLED=0` every anonymous visitor is premium, so full-model edge and cross-book scans were unthrottled for the public. — `_scan_throttled()` throttles everyone except admins (free 8/min, premium 30/min). — **FIXED** (`test_scan_throttle_applies_in_open_mode`). The budget is now keyed by **client IP** when Streamlit exposes one (`st.context.ip_address`; behind N proxies set `STREAMLIT_TRUSTED_PROXY_HOPS`, the same right-most-XFF contract as the webhook). It falls back to per session, so a new session no longer resets the budget. — **FIXED** (`test_new_session_does_not_reset_ip_budget`, XFF-spoof test).
- `nba_model/web/auth.py:416` + app.py:1184,1406,1442 — **MINOR** — `PREVIEW_PLAYERS` has "Nikola Jokic" while the DB has "Nikola Jokić", so the free preview silently showed only LeBron. — `is_preview_player()` matches accent- and case-insensitively everywhere. — **FIXED**.
- `nba_model/web/watchlist.py:74,104` — **MAJOR** — Read-modify-write without a lock or an atomic write, and a torn read parsed as `{}` and was saved back over *every* user's watchlist. — Now a process lock, a temp file plus `os.replace`, never overwriting an unparseable store, and an 80-character item cap. — **FIXED** (`WatchlistStoreTests`).
- `Dockerfile:64` — **MAJOR** — The Streamlit image didn't copy `sports/` (`app.py` imports it), so the image built but died on first load. — `COPY sports ./sports`. — **FIXED** (not image-built here: the Docker daemon was unavailable).
- `nba_model/web/app.py:2813-2828` — **MINOR** — Team charts let free users see every team; `gate_team`, `PREVIEW_TEAMS`, `cap_n_games` and `allowed_stats` are unused, so the documented policy and the code disagree. — Owner decision: open access, so the unused gates were **deleted** (`PREVIEW_TEAMS`, `gate_team`, `cap_n_games`, `allowed_stats`) and SECURITY/DEPLOYMENT/MULTI_SPORT docs updated to match. — **FIXED**.
- `nba_model/web/app.py:2419-2427` + `watchlist.py:168` — **MINOR** — Pin stores the raw `?player=` value, and every anonymous visitor gets a persisted key, so the store grows without bound. — Pins are accepted only for players in the DB (`_is_known_player` → `watchlist.add(is_known=…)`). `anon:*` keys expire after `WATCHLIST_ANON_TTL_DAYS` (default 90) without a save; legacy unstamped keys get a grace stamp rather than deletion, and signed-in keys never expire. — **FIXED** (`test_anonymous_keys_expire_signed_in_keys_do_not`, legacy-grace and pin tests).
- `nba_model/web/app.py:2083` — **MINOR** — In open mode, "Single prop (model)" lets anonymous visitors trigger outbound NBA API calls (risk of an IP ban). — `_outbound_throttled()` enforces 4 runs/min per client IP (session fallback) **and** 20/min process-wide, because stats.nba.com bans the server's IP whoever triggers the calls. Admins are exempt. — **FIXED** (`test_outbound_nba_api_throttle_per_client_and_global`).
- `nba_model/web/app.py:1607,1691` — **MINOR** — The Limit inputs allowed 2000/5000, but the validator silently caps at 200. — Widget `max_value=iv.N_GAMES_HARD_CAP`. — **FIXED**.

## 5. Webhook / billing follow-ups (judgment calls)

- `webhook_app.py:155` — **MINOR** — `WEBHOOK_TRUSTED_PROXY_HOPS` is parsed per request (a bad value 500s every webhook), and only the first X-Forwarded-For header is read. — **PROPOSED**: parse once at startup; use `getlist`. *(Deferred with billing — owner froze Stripe work 2026-09-28; app deploys with `BILLING_ENABLED=0`.)*
- `webhook_app.py` `invoice.paid` / checkout — **MINOR** — These events carry no `current_period_end`, so a premium row can have a NULL end and never expire. — **PROPOSED**: take the end from `lines[0].period.end`, or fetch the subscription. *(Deferred with billing — owner froze Stripe work 2026-09-28; app deploys with `BILLING_ENABLED=0`.)*
- `webhook_app.py:195` — **MINOR** — The secret is read from the env var only; the docstring says secrets.toml also works. — **PROPOSED**: pick one source and document it. *(Deferred with billing — owner froze Stripe work 2026-09-28; app deploys with `BILLING_ENABLED=0`.)*
- `auth.py` `tier_for` — **MINOR** — Subscription-DB errors (e.g. Postgres down) crash every signed-in page. — **PROPOSED**: fail closed to free with a warning. *(Deferred with billing — owner froze Stripe work 2026-09-28; app deploys with `BILLING_ENABLED=0`.)*
- `stripe_helpers.py:47` — **MINOR** — `prefilled_email` isn't URL-encoded (`+` becomes a space). — **PROPOSED**: use `quote()`. *(Deferred with billing — owner froze Stripe work 2026-09-28; app deploys with `BILLING_ENABLED=0`.)*
- `operations_panel.py` — **INFO** (admin-only) — Child processes inherited the full environment (Stripe/AWS keys, subscriptions DSN, access code), and their output is shown on screen. — `scrubbed_env()` denylists `STRIPE_*`, `AWS_*`, `DB_SYNC_*`, `SUBSCRIPTIONS_DB_URL`, `FLAGSHIP_ACCESS_CODE`, `BILLING_ALERT_WEBHOOK_URL` and `BUCKET_NAME`; keys the jobs need (e.g. `ODDS_API_KEY`) still pass through. — **FIXED** (`OperationsEnvScrubTests`).
- `ENABLE_TRIAL` — **INFO** — Every new Google account gets a fresh trial. — **PROPOSED**: accept this, or require a verified domain or card. *(Deferred with billing — owner froze Stripe work 2026-09-28; app deploys with `BILLING_ENABLED=0`.)*

## 6. Deploy gaps (flagship)

- W-DEP-1 — **MAJOR** — The flagship API has **no auth**: the full scored slate, which is premium content in Streamlit, is open to anyone who can reach it. — No-billing substitute: an **optional shared access code**. With `FLAGSHIP_ACCESS_CODE` set, every `/api/*` call except `/api/health` needs `X-Access-Code` (constant-time compare). The UI prompts once and stores the code in localStorage. Wrong codes have a separate brute-force budget (10/min per IP; missing codes don't count). Unset means fully open. — **FIXED** (`test_api_guards.py`, `AccessGate.test.tsx`). Real per-user auth is **deferred with billing** (DEPLOYMENT.md §15).
- W-DEP-2 — **MAJOR** — The flagship API has **no rate limit**, and full-model edge and cross-book scans are CPU-heavy. — In-process per-IP sliding-window limiter (`api/guards.py`): **heavy** budget 20/min for `/api/slate/edges`, `/api/cross-book` and `/api/parlay/*`; **read** 240/min for other `/api/*`; health and static files exempt. Returns 429 with `Retry-After`. It keys on `request.client.host`, which uvicorn rewrites from X-Forwarded-For only for `--forwarded-allow-ips` (the proxy), so it can't be spoofed; the same contract §15 documents. The Caddy/nginx example stays as defence in depth. — **FIXED** (`test_api_guards.py`: tiers, per-IP, XFF spoof, disable, window).
- `Dockerfile.flagship` — **INFO** — Written and compose-validated, but **still not built**: the Docker daemon was not running again on 2026-09-28 (`docker info` fails), so no container smoke run was done and none is claimed. The same production path was verified without Docker: `vite build` served by uvicorn with `FLAGSHIP_STATIC_DIR` passed `api/smoke_check.py --strict` 18/18, and the after-screenshots were taken against it.

## 7. Phase 3 features (2026-09-28) — security/contract notes

- `api/main.py` `POST /api/parlay/price` — **INFO** — Compute-only (nothing is persisted; a test asserts bet_log is untouched). It wraps `fitted_prob_over`, `calibrate_correlations`, `covariance_matrix`, `simulate_multi_leg_sgp` and `calculate_parlay_ev`; there is no new model math, and under legs use the standard sign flip.
  - Inputs go through input_validation: leg count 2–6, chartable stat, `validate_line` (plausibility), odds, n_games, and n_sims capped at 50k (below iv's 200k) for public CPU.
  - Duplicate legs → 400, unknown player → 404, NaN → 400, malformed body → 422.
  - The RNG is seeded per leg-set under a lock, so a parlay prices deterministically.
  - Heavy rate-limit tier; the response carries a disclaimer.
- `api/main.py` `GET /api/paper-trades`, `/api/paper-trades/calibration` — **INFO** — Read-only over `bet_log` and `calibration_report.load_calibration_frame` / `build_reliability_table` / `brier_by_stat` (the CLI report's path). `status`, `source` and `stat` are allowlisted; `limit` ≤ 500; `n_buckets` 2–20. On an empty DB it returns zero rows with null KPIs (tested).
  - "Est. units" prices wins at the logged implied probability, which includes vig, so it is labelled an estimate.
- Frontend `/parlay` and `/paper-trades` — **INFO** — Both carry the standing "Model estimate — not validated for real money; WS10 gates unpassed" banner.
  - The correlation heatmap uses a cyan↔violet diverging scale, never the EV green/red (tested).
  - Parlay legs live in the URL (`?legs=`), and decoding drops malformed entries instead of throwing.
  - The paper-trades empty state gives the `bet_slip` command; verified live (bet_log is empty on 2026-09-28).

## 8. Deployment + mobile + watchlist pass (2026-09-28)

- `api/db_sync.py` + `scripts/publish_db.sh --to-object-store` — **INFO** — DB delivery for Fly.io (DEPLOYMENT.md §16).
  - **Push:** a lock check and consistent `.backup` snapshot, validated (`quick_check` + required tables), sha256-deduped, gzipped, uploaded, and `latest.json` repointed last.
  - **Pull:** temp file → sha256 verify → validate → fsync → `os.replace`. There is still no upload endpoint; the API stays read-only.
  - The Mac credentials file must be chmod 600, or the script refuses it.
  - Tests: `test_db_sync.py` (20) and `test_publish_script.py` (6).
  - Verified against a real S3 API (moto) with the built image: boot pull, a live swap with no failed requests, rollback, and `smoke_check --strict` 18/18 (19/19 with the access code).
- `api/guards.py` — **INFO** — `API_CLIENT_IP_HEADER` (set to `Fly-Client-IP` on Fly) keys the rate limiter on the platform-overwritten header. The value must parse as an IP or the limiter falls back to the socket peer, and the header is ignored unless configured (tested). Only safe behind a proxy that always sets it; DEPLOYMENT.md says so.
- Frontend watchlist (`lib/watchlist.ts`) — **INFO** — localStorage only, with no server writes.
  - Items are sanitized on every read, so malformed storage or links are dropped, never thrown.
  - Capped at 50; `?w=` share links are decoded defensively; syncs across tabs.
  - A shared link never overwrites silently: it offers Add / Replace / Dismiss.
- Mobile — **INFO** — Phones get a bottom tab bar plus a drawer, tablets an icon rail.
  - 44px `pointer-coarse` targets; tables scroll inside their cards with a sticky first column; ECharts `hideOverlap` on axis labels.
  - Audit on all 8 views at 375 and 768 px: 0 page overflow and 0 sub-44px targets. Evidence in `docs/screenshots/flagship/mobile/`.

