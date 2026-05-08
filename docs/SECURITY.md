# Security model

This document describes the threat model for the deployed Streamlit app +
Stripe webhook handler, the mitigations that are in place, and how to report
new vulnerabilities responsibly. Pair it with [DEPLOYMENT.md](DEPLOYMENT.md)
during a deploy review.

## Reporting a vulnerability

If you find a security issue, **please do not open a public GitHub issue**.
Email the maintainer with a short description + reproduction steps and we
will respond within 72 hours.

## Trust boundaries

```
                          ┌──────────────────────────────┐
                          │         Stripe               │
                          │  - signs webhook events      │
                          │  - holds card data           │
                          └─────────────┬────────────────┘
                                        │ HTTPS (signed)
                                        ▼
┌────────────────────────┐    ┌──────────────────────────┐
│ End user (browser)     │    │ FastAPI webhook handler  │
│ - signs in via OIDC    │    │ - verifies HMAC + replay │
│ - receives session     │    │   window                 │
│   cookie               │    │ - writes subscriptions   │
└──────────┬─────────────┘    └─────────────┬────────────┘
           │ HTTPS (Streamlit)              │
           ▼                                │
┌────────────────────────┐                  │
│ Streamlit app          │                  │
│ - reads tier from      │                  │
│   subscriptions DB     │ ◄────────────────┘
│ - serves charts        │
└────────────────────────┘
```

We trust:
- The OIDC providers (Google, Microsoft) to verify email ownership.
- Stripe's HMAC + timestamp on webhook payloads.
- The local OS file system + the secrets store provided by the host
  (Streamlit Cloud secrets UI, Render env vars).

We do **not** trust:
- Anonymous and free-tier visitors (they can read this code).
- Stripe-shaped requests without a valid signature.
- The Streamlit `secrets.toml` file in transit; it is `.gitignore`d.

## Mitigations by risk category

### A01 - Broken Access Control (paywall bypass)

| Control | Where |
|---|---|
| `auth.current_user()` is the single source of truth for tier; every locked view checks it before fetching data | `nba_model/web/auth.py`, `app.py` |
| `paywall(feature)` is called inside the dispatch even when the sidebar UI already filtered the choice (defense in depth) | `app.py` `_team_overview`, `_parlay_view`, `_all_stats_overview` branches |
| Free-tier player allowlist (`PREVIEW_PLAYERS`) enforced both in the dropdown filter AND in `gate_player()` at fetch time | `auth.py` |
| Non-admin users cannot change `db_path` (UI input is hidden); production reads only the bundled SQLite file | `app.py` |
| Admin override is allowlist-based on email (read from `[auth].admins` in encrypted secrets), not on a self-claimable cookie | `auth.is_admin`, `auth._admin_emails` |

### A02 - Cryptographic failures

| Control | Where |
|---|---|
| Session cookies are signed with a 32-byte secret stored in `[auth].cookie_secret` | Streamlit native; `secrets.toml.example` |
| Stripe webhook payloads are HMAC-SHA256 verified with `stripe.Webhook.construct_event` | `webhook_app.py` |
| Replay window enforced via Stripe's built-in `tolerance` (default 300s, env-configurable) | `webhook_app.py:_tolerance_seconds` |
| Stripe API version pinned (`stripe.api_version`) so an upstream upgrade can't silently change payload shapes | `webhook_app.py` |
| TLS is terminated by the hosting layer (Streamlit Cloud / Render) - we do not run plaintext anywhere | hosting config |

### A03 - Injection

| Control | Where |
|---|---|
| Every SQL query uses parameter binding (`?` + tuple). Reviewed and grep-audited | `subscriptions.py`, `player_charts.py`, `audit_db.py` |
| The one f-string SQL fragment (`fetch_team_chart_data`) interpolates a value chosen from a hardcoded allowlist (`_team_value_sql_expr`) and an `assert expr in _allowed_exprs` check at the call site | `player_charts.py` |
| Email is validated by regex + length limit before being used as a primary key, even though the OIDC provider is trusted | `subscriptions._validate_email` |

### A04 - Insecure design

| Control | Where |
|---|---|
| Webhook handler is **idempotent on `event_id`** so Stripe retries (and attacker replays inside the tolerance window) are no-ops | `subscriptions.record_stripe_event` |
| Subscription state is "fail closed": invalid email or missing record returns `free`, never `premium` | `subscriptions.tier_for` |
| Premium expiration is enforced by checking `current_period_end` on every read, not by a background job | `subscriptions.tier_for` |
| Free tier limits are enforced by the data layer (cap N games, restrict stat list), not just by hiding UI | `auth.cap_n_games`, `auth.allowed_stats` |

### A05 - Security misconfiguration

| Control | Where |
|---|---|
| `.streamlit/secrets.toml` and `data/database/subscriptions.db` are in `.gitignore` | `.gitignore` |
| FastAPI `docs_url`, `redoc_url`, `openapi_url` are all set to `None` so the webhook host does not expose its schema | `webhook_app.py` |
| `TrustedHostMiddleware` honors `WEBHOOK_TRUSTED_HOSTS` env var to restrict accepted Host headers | `webhook_app.py` |
| Subscription DB is `chmod 0600` after every connection on POSIX hosts | `subscriptions._harden_db_file` |
| SQLite extensions are explicitly disabled on each connection | `subscriptions._connect` |
| `/healthz` returns no version or build information | `webhook_app.healthz` |
| HTTP **security headers** middleware sends `Strict-Transport-Security` (2yr+preload), `Content-Security-Policy: default-src 'none'`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy` (deny camera/mic/geo/etc), and aggressive `Cross-Origin-*` isolation on every reply | `webhook_app.SecurityHeadersMiddleware` |
| `Server:` header is overwritten to "webhook" so we don't leak Uvicorn / Python version | `webhook_app.SecurityHeadersMiddleware` |
| Per-IP **rate limiter** on `/stripe/webhook` (default 120 reqs / 60s, env-tunable). Returns 429 with `Retry-After` so Stripe's exponential-backoff retry kicks in cleanly | `webhook_app.IPRateLimiter` |

### A06 - Vulnerable + outdated components

| Control | Where |
|---|---|
| `requirements.txt` uses compatible-release (`~=`) ranges so security patches flow without breaking minors | `requirements.txt` |
| Exact versions captured in `requirements.lock` for reproducible CI/deploy | `requirements.lock` |
| Periodic upgrade procedure documented (run `pip install -r requirements.txt`, regenerate the lockfile, run pytest) | this file |
| **`pip-audit`** scans installed packages against the Python Packaging Advisory Database (CVE feed) on every CI run | `scripts/security_scan.sh` |
| **`bandit`** SAST scans `nba_model/` source on every CI run (excludes tests + the desktop UI subprocess wiring) | `scripts/security_scan.sh` |
| Cheap grep sweep for accidentally-committed secret patterns (`sk_live_*`, `whsec_*`, AWS keys, PEM bodies) on every CI run | `scripts/security_scan.sh` |

### A07 - Identification + authentication failures

| Control | Where |
|---|---|
| OIDC handles password storage, MFA, account recovery (Google + Microsoft) | provider |
| Streamlit's signed-cookie session times out per `cookie_secret` rotation | Streamlit native |
| `st.logout` clears the session client-side **and** invalidates the cookie | Streamlit native |
| App never inspects passwords or stores them | by design |
| Email is treated as an opaque identifier; we never display admin lists in error messages | `auth.is_admin`, `paywall` |

### A08 - Software + data integrity failures

| Control | Where |
|---|---|
| Webhook events are signed by Stripe; we reject anything else with HTTP 400 | `webhook_app.py` |
| Body size is capped at 256 KiB **before** buffering, preventing DoS-by-payload | `webhook_app.MAX_BODY_BYTES` |
| Subscriptions table uses `ON CONFLICT(email) DO UPDATE` for atomic upserts; SQLite WAL allows safe concurrent reads from the Streamlit app | `subscriptions.upsert` |

### A09 - Logging + monitoring

| Control | Where |
|---|---|
| Webhook logs only event id + event type + presence flags. **Email addresses, payload bodies, customer IDs are not logged** | `webhook_app.py` |
| Stripe payload is stored verbatim in the `stripe_events` table for forensic replay, but is not written to stdout | `subscriptions.record_stripe_event` |
| Bad-signature attempts and oversize payloads are logged at WARN with type only; not enough info to help an attacker tune | `webhook_app.py` |

### A10 - Server-side request forgery / file inclusion

| Control | Where |
|---|---|
| `db_path` is locked to a hardcoded value for non-admin users; admins are project-owner-allowlisted only | `app.py` |
| No `requests.get()` or `urllib.urlopen()` accepts URLs derived from end-user input on the web side | grep-audited |
| Path inputs (e.g. inventory output) only appear in the desktop UI / CLI, never the web app | `nba_model/data/audit_db.py` |

## Data retention + GDPR / CCPA

The only personal data the app stores is the **email address** the user signs
in with (used as the primary key in `user_subscriptions`) plus the verbatim
Stripe event payload kept in `stripe_events` for forensic replay. We don't
collect or store names, IP addresses, payment-card data, or behavioural
analytics.

| What | Storage | Retention |
|---|---|---|
| `user_subscriptions` row (email + tier + Stripe ids + period_end) | `subscriptions.db` | Until the user requests deletion. Inactive accounts keep their `free` row indefinitely so the same user re-signs-in without leaking premium history. |
| `stripe_events.payload` (full raw JSON Stripe sent us) | `subscriptions.db` | 30 days, then archived to backups for 1 year. We need this window to investigate billing disputes. |
| Server logs (event_id + event_type, no email or payload bodies) | host stdout | Per the host's default; Render/Railway typically rotate after 7 days. |
| Backups of `subscriptions.db` | `backups/` (or wherever `scripts/backup_subscriptions.sh` writes) | Last 30 daily snapshots; older are deleted automatically. |

**Right to erasure (GDPR Art. 17 / CCPA "Do Not Sell"):**
A user emails the project owner asking for deletion → operator runs:

```sql
DELETE FROM user_subscriptions WHERE email = ?;
DELETE FROM stripe_events
 WHERE json_extract(payload, '$.data.object.customer_email') = ?;
```

(or any equivalent admin tool that operates on the same DB). Stripe-side
deletion is the user's responsibility through Stripe's own portal; we don't
have access to delete from Stripe.

**Right to access:** the user can email the operator and receive their full
`user_subscriptions` row dumped to JSON.

## Backups + recovery

`scripts/backup_subscriptions.sh` uses SQLite's online `.backup` (NOT a raw
`cp`, which produces corrupt files when the webhook is mid-write):

```bash
./scripts/backup_subscriptions.sh           # writes to ./backups/
./scripts/backup_subscriptions.sh /mnt/dest # explicit destination dir
```

Cron example (nightly 03:15 UTC, retain 30 days):

```cron
15 3 * * * cd /srv/app && ./scripts/backup_subscriptions.sh \
            /srv/app/backups >> /var/log/subscriptions-backup.log 2>&1
```

To restore: stop the webhook, copy the backup over `subscriptions.db`,
`chmod 0600` it, restart the webhook.

## Routine security scans

```bash
# install dev tooling once
.venv/bin/python3 -m pip install -r requirements-dev.txt

# fail-fast: scans + grep sweep, exits non-zero on findings
./scripts/security_scan.sh

# human-readable, no exit code (good for first run)
./scripts/security_scan.sh --report
```

Wire `./scripts/security_scan.sh` as a required step in CI so a CVE published
against any pinned dependency or a regression in our SAST baseline fails the
build.

## Operational hardening checklist (deploy day)

- [ ] `cookie_secret` is a 32+ byte random string, **not** the placeholder.
- [ ] OAuth client secrets are pasted into the host's encrypted secrets UI, never committed.
- [ ] `[auth].admins` lists ONLY the project owner's email (not a shared mailbox).
- [ ] Stripe `webhook_secret` matches the value in the Stripe dashboard for the deployed endpoint.
- [ ] `WEBHOOK_TRUSTED_HOSTS` env var is set to the public hostname of the webhook app.
- [ ] Stripe is in **live mode**; old test-mode webhook endpoints have been deleted from the dashboard.
- [ ] `subscriptions.db` is on a persistent volume; the webhook host can write to it; the Streamlit host can read from it.
- [ ] HTTPS is enforced everywhere; HTTP redirects to HTTPS at the host or LB layer.
- [ ] `requirements.lock` was used to install (`pip install -r requirements.lock`) so transitive deps match what was tested.

## "AI" threat model (what applies, what doesn't)

It's worth being explicit about what this project's "AI" actually is, because
many of the threats from the OWASP AI Top 10 + the popular AI-attack lists
target neural networks / LLMs and don't translate to a statistical model.
This system contains:

- Rolling means, weighted averages, defense-rating adjustments
- scipy distributions (`norm`, `poisson`, `nbinom`)
- Monte Carlo simulation (numpy)
- Calibrated correlation matrix for SGP parlays

There is **no LLM, no neural network, no learned weights, no training-from-
user-data loop, and no inference of generative content**. That makes most
"AI risk" categories non-applicable. The relevant ones, and what we do:

| AI threat | Applies here? | Why / Mitigation |
|---|---|---|
| **Adversarial input manipulation** ("evasion") | Yes (mild) | Users supply line / odds / n_games / n_sims to the model. Hostile inputs (`nan`, `inf`, 9999.5 lines, n_sims=10^9) could either crash with `OverflowError` or produce silent garbage. Mitigation: `nba_model/web/input_validation.py` validates every numeric input with per-stat plausible ranges, finite-number checks, hard caps on history length and Monte Carlo iterations, and 2-6-leg cap on parlays. |
| **Data poisoning** | Yes | The `betting_lines` and `web_prop_cards` rows come from third-party scrapes; a hijacked sportsbook page could inject a "Jokic 9999.5 points" line that pollutes the consensus mean, hit-rate, and EV math. Mitigation: `is_plausible_betting_line` runs at insert time in `db_manager.insert_betting_lines_records`, dropping rows with implausible lines or odds. The drop count is logged. |
| **Compute DoS** | Yes (mild) | `n_games`, `n_sims`, parlay-leg count are user-tunable and could blow up CPU. Mitigation: hard caps (200 games, 200k sims, 6 legs) enforced server-side regardless of UI bounds, plus the per-IP webhook rate limiter for billing-side abuse. |
| **Privilege escalation via "AI agent" too much access** | No | Nothing in the model has tool-use or shell access. The webhook handler has DB-write only. |
| **Model inversion / training-data leakage** | No | Inputs are public NBA stats + market odds. No PII anywhere in the model path; user emails live in a separate DB the model never touches. |
| **Model theft / IP extraction** | Low | The "model" is open-source statistical code; there's no proprietary secret to steal. Parameter choices (defense sensitivity, volatility scaling) are visible in the repo by design. |
| **AI-generated phishing / deepfakes** | No | We generate no content delivered to third parties. |

The "what applies" rows are tested by the AdversarialInput, IngestionPoisoning,
and ComputeBounds classes in `nba_model/tests/test_security_stress.py`.

## Findings discovered by the stress suite

`nba_model/tests/test_security_stress.py` is the "we tried to break it" suite,
separate from the regression tests in `test_security.py`. It hammers the
rate limiter with concurrent threads, fuzzes the email validator with
homoglyphs / control characters / 256-byte boundaries, fires real
HMAC-signed Stripe events with malformed payloads, corrupts the DB to
verify fail-closed behavior, floods the rate limiter with 10k unique IPs
to verify GC, and slow-streams payloads to test slowloris protection.

The following **real bugs** were caught by this suite during the May 2026
review and fixed in the same pass:

| Finding | Where it would have hit prod | Fix |
|---|---|---|
| Stripe SDK 15.x's `StripeObject` no longer subclasses `dict`, so `obj.get("customer")` raised `AttributeError` on every real event. The webhook would have 500'd on every Stripe delivery, and Stripe would have retried for days. | `webhook_app._email_from_event` and `webhook_app.stripe_webhook` upsert call sites | New `_safe_get` helper that handles both `dict` and `StripeObject` cleanly. Documented in a comment so future Stripe SDK majors don't reintroduce it. |
| Punycode-encoded IDN domains (`xn--pple-43d.com` → `аpple.com`) were accepted as valid emails — homograph attack vector. | `subscriptions._validate_email` | Reject any email whose domain has an `xn--` label. Documented as a deliberate trade-off (zero legitimate IDN users for a US-focused product). |
| The webhook handler called `subscriptions.upsert()` directly without a try/except. A malformed email in a Stripe event (e.g. an attacker who registered with a bad address) would raise `ValueError`, bubble up as a 500, and Stripe would retry the same poison event forever. | `webhook_app.stripe_webhook` | Catch `ValueError` from `upsert`, log it once, return 200 with `rejected: True` so Stripe stops retrying. |
| No wall-clock timeout on the body-read loop. The 256 KiB size cap blocks the "send 1 GB body" attack, but a slowloris-style attacker could open many connections and dribble bytes for hours, exhausting worker slots without ever hitting the cap. | `webhook_app.stripe_webhook` | `asyncio.wait_for` around the chunked stream read. Default 10s, configurable via `WEBHOOK_BODY_TIMEOUT_SECONDS`. Returns 408 on timeout. |
| Unknown stat types fell through `STAT_LINE_RANGES.get(stat, default)` to a generic 1000-cap range, allowing a poisoned scraper row with `stat_type="garbage_stat"` and `line=999` to pass `is_plausible_betting_line`. | `input_validation.validate_line` | Reject any stat not in the explicit `STAT_LINE_RANGES` allowlist. Fail-closed (unknown -> rejected). |
| `database is locked` flake under 16-thread parallel upserts. Production analog: real Stripe deliveries can race for the same email, AND the Streamlit reader contends with the webhook writer. | `subscriptions._connect` | `_bootstrap_db` does the one-time `journal_mode=WAL` flip behind a process-wide `_INIT_LOCK`. Each connection then sets `busy_timeout=5000` so steady-state contention waits up to 5s instead of erroring. 30 stress runs in a row are now flake-free. |

Each fix is covered by a regression test added to the stress suite, so a
future change reintroducing any of them will fail CI.

## Known residual risks (accepted)

- **SQLite over network volume** has known correctness pitfalls. We accept
  this for the launch tier; migrating to managed Postgres is on the
  follow-up list (see `notes.txt`).
- **No rate limiting on the Streamlit app itself.** Streamlit Cloud applies
  its own per-IP throttling; if you self-host, put Cloudflare or a similar
  WAF in front.
- **Stripe Customer Portal** is not yet wired - users can't self-serve cancel
  in-app today; they'd need to email us. Adding this is on the follow-up list.
