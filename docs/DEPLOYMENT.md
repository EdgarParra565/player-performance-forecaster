# Deployment Guide

This guide walks through deploying the Streamlit web app to **Streamlit
Community Cloud** with **Google + Microsoft sign-in** and **Stripe-powered
membership tiers**.

The webhook handler that mutates the subscription state on Stripe events runs
**separately** (Streamlit Cloud doesn't accept inbound webhooks). The
recommended free option is **Render** or **Railway** for that one tiny
FastAPI service.

```
   ┌─────────────────┐                ┌─────────────────────┐
   │  Streamlit      │                │  FastAPI webhook    │
   │  Community      │                │  (Render/Railway)   │
   │  Cloud          │                │                     │
   │                 │                │  POST /stripe/...   │
   │  reads          │                │  writes             │
   │  subscriptions  │                │  subscriptions      │
   │  table          │                │  table              │
   └────────┬────────┘                └─────────┬───────────┘
            │ shared SQLite (or                  │
            │ Postgres in production)            │
            └────────────────┬───────────────────┘
                             ▼
                  data/database/subscriptions.db
```

> **Important:** SQLite-on-disk works fine for low traffic, but Streamlit
> Cloud and Render give you ephemeral disks. For a real launch you almost
> certainly want a managed Postgres (Neon, Supabase, Render Postgres) and a
> small migration to swap the `subscriptions.py` driver. The current code
> uses SQLite to keep first-time setup simple.

## 1. Prereqs

- A GitHub account with this repo pushed up.
- A Google Cloud project ([console.cloud.google.com](https://console.cloud.google.com/)).
- A Microsoft Entra (Azure AD) tenant ([portal.azure.com](https://portal.azure.com/)).
- A Stripe account ([dashboard.stripe.com](https://dashboard.stripe.com/)).
- A Streamlit Cloud account ([share.streamlit.io](https://share.streamlit.io/)).
- A Render or Railway account (free tier OK).

## 2. Create OAuth credentials

### Google

1. APIs & Services → Credentials → Create Credentials → OAuth client ID
2. Application type: **Web application**
3. Authorized redirect URI:
   `https://<YOUR-APP>.streamlit.app/oauth2callback`
4. Save `Client ID` and `Client secret`.

### Microsoft

1. Microsoft Entra ID → App registrations → New registration
2. Redirect URI (Web): `https://<YOUR-APP>.streamlit.app/oauth2callback`
3. After creation: **Certificates & secrets → New client secret**, save the
   **Value** (not the ID).
4. Save `Application (client) ID` and the secret value.

## 3. Configure Streamlit secrets

1. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`
2. Fill in real values for `[auth]`, `[auth.google]`, `[auth.microsoft]`.
3. Generate a stable cookie secret:
   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```
4. Add your own email to `[auth].admins` so you can use the app as premium
   without paying yourself.

## 4. Stripe products + webhook

### Create the membership

1. Stripe Dashboard → Products → **+ Add product**
   - Name: "NBA Probability Model — Premium"
   - Pricing: recurring monthly (e.g. $19/month).
2. Copy the **Price ID** (`price_...`).

### Choose a checkout flow (pick ONE)

**Option A — Payment Link (simplest, no Stripe SDK needed in the app)**
1. Stripe Dashboard → Payment Links → **+ New**, select the price.
2. Copy the public URL.
3. Set `[stripe].payment_link_url` in `secrets.toml`.

**Option B — Hosted Checkout Session (server-side, supports more options)**
1. Set `[stripe].secret_key`, `[stripe].price_id`, `[stripe].success_url`,
   `[stripe].cancel_url` in `secrets.toml`.
2. The app creates a Checkout Session at click-time.

### Webhook endpoint

1. Deploy `nba_model.web.webhook_app:app` to Render/Railway/Fly:
   - **Build command:** `pip install -r requirements.lock`
   - **Start command:** (note `--no-server-header` so Uvicorn doesn't leak its
     own `Server:` line; our SecurityHeadersMiddleware sets a generic one)
     ```
     uvicorn nba_model.web.webhook_app:app \
       --host 0.0.0.0 --port $PORT \
       --no-server-header --forwarded-allow-ips "*" --proxy-headers
     ```
   - **Persistent disk:** mount `/data`, set
     `SUBSCRIPTIONS_DB_PATH=/data/subscriptions.db`.
   - **Required env**: `STRIPE_WEBHOOK_SECRET`, `STRIPE_SECRET_KEY`.
   - **Recommended env**:
     - `WEBHOOK_TRUSTED_HOSTS=<your-webhook-hostname>`
     - `WEBHOOK_RATE_MAX=120`, `WEBHOOK_RATE_WINDOW_SECONDS=60`
     - `WEBHOOK_TRUSTED_PROXY_HOPS=1` (so the rate limiter keys on the real
       client IP from `X-Forwarded-For` instead of the proxy IP).
2. Health check: `https://<webhook-host>/healthz` → `{"status":"ok"}`
3. Stripe Dashboard → Developers → Webhooks → **Add endpoint**
   - URL: `https://<webhook-host>/stripe/webhook`
   - Events to send:
     - `checkout.session.completed`
     - `customer.subscription.created`
     - `customer.subscription.updated`
     - `customer.subscription.deleted`
     - `invoice.paid`
     - `invoice.payment_failed`
4. Copy the **Signing secret** (`whsec_...`) into the webhook host's env as
   `STRIPE_WEBHOOK_SECRET`. Also set `STRIPE_SECRET_KEY` (the same
   `sk_test_...`/`sk_live_...` you used elsewhere).

> Both the Streamlit app and the webhook host must read/write the **same**
> `subscriptions.db`. With SQLite on Streamlit Cloud that's not actually
> possible (no shared disk) — see the Postgres note above. For initial
> testing you can run both processes locally on the same machine.

## 5. Streamlit Community Cloud

1. Visit https://share.streamlit.io and connect this repo.
2. Set the entry point: `nba_model/web/app.py`
3. App settings → Secrets → paste the contents of your local
   `.streamlit/secrets.toml`.
4. App settings → General → Python version: 3.11+.
5. Deploy.

Once it's up, click **Sign in with Google** in the sidebar to test the
OAuth flow. Verify your admin email shows the **Premium** badge.

## 6. Smoke test the full flow

1. Sign out, sign in with a different (non-admin) email.
2. Confirm the sidebar shows :lock: Free, the player dropdown only includes
   the preview allowlist, and the View-mode radio only has "Single stat".
3. Click **Upgrade to Premium**, complete a Stripe test card
   (`4242 4242 4242 4242`, any future date, any CVC).
4. Watch your webhook logs (`docker logs` or the Render dashboard) for
   `event_type: checkout.session.completed` and an `upsert` to the DB.
5. Refresh the Streamlit tab. Sidebar should now read :star: Premium and
   all view modes should be unlocked.

## 7. Upgrading the local dev experience

Run everything locally to debug:

```bash
# terminal 1 - Streamlit
.venv/bin/python3 -m streamlit run nba_model/web/app.py

# terminal 2 - webhook handler
STRIPE_WEBHOOK_SECRET=whsec_test_... \
STRIPE_SECRET_KEY=sk_test_... \
.venv/bin/python3 -m uvicorn nba_model.web.webhook_app:app --port 8081

# terminal 3 - forward Stripe events to the local webhook
stripe listen --forward-to localhost:8081/stripe/webhook
```

`stripe listen` prints a `whsec_...` secret you can use as
`STRIPE_WEBHOOK_SECRET` for local-only testing.

## 7a. Docker for local testing + self-host deployment

The repo ships with a single-stage `Dockerfile` and a `docker-compose.yml`
that wires four services (`streamlit`, `tests`, `etl-bulk`, `webhook`).
Useful for matching the production environment exactly when reproducing a
bug, and for self-host targets (VPS / Fly / Railway / Render).

### Build + run the web app

```bash
docker build -t nba-model .

# Default = open access (BILLING_ENABLED=0, no secrets required)
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" nba-model

# Or via compose
docker compose up streamlit
```

Open http://localhost:8501. Healthcheck endpoint: `/_stcore/health`.

### Re-engage billing in the container

```bash
# 1. Fill in .streamlit/secrets.toml (see secrets.toml.example).
# 2. Uncomment the bind-mount line in docker-compose.yml:
#       - ./.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro
# 3. Boot both web + webhook services:
BILLING_ENABLED=1 \
STRIPE_WEBHOOK_SECRET=whsec_... \
STRIPE_SECRET_KEY=sk_... \
docker compose --profile billing up streamlit webhook
```

The `tests` and `etl-bulk` services are behind the `tools` profile so they
don't boot with the default `docker compose up`. Run them explicitly:

```bash
docker compose run --rm tests                    # full pytest suite
docker compose run --rm etl-bulk                 # default seasons
docker compose run --rm -e SEASONS="2025-26 2024-25" etl-bulk
```

### Production self-host on a VPS

```bash
# On the server:
git clone <your-fork> && cd nba-probability-model
docker compose --profile billing up -d streamlit webhook
# Put nginx / Caddy in front with TLS and proxy to :8501 + :8000.
```

The image is `~1.4 GB`. The runtime memory is `~70 MB` for the web service
and `~50 MB` for the webhook. Volume mounts:

| Mount | Why |
|---|---|
| `./data:/app/data` | SQLite DBs (`nba_data.db`, `subscriptions.db`) persist across restarts |
| `./.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro` | OIDC + Stripe credentials when billing is on |

> **Streamlit Community Cloud users** should ignore this section — Streamlit
> Cloud builds the image for you. Docker is only relevant for self-host
> deploys or reproducing a prod environment locally.

## 8. Free vs Premium feature matrix

| Feature                          | Free preview                          | Premium |
|----------------------------------|---------------------------------------|---------|
| Players viewable                 | `Nikola Jokic`, `LeBron James`        | All     |
| Stats                            | `points`                              | All 7   |
| Last N games                     | up to 5                               | up to 200 |
| Single-stat detailed view        | ✅                                    | ✅      |
| All-stats overview               | 🔒                                    | ✅      |
| Team distributions               | 🔒 (LAL/DEN allowlist if unlocked)    | ✅      |
| Parlay analysis (single + multi) | 🔒                                    | ✅      |
| Custom-line probe                | 🔒                                    | ✅      |
| Distribution overlays            | normal only                           | normal + poisson + neg-binomial |

The matrix is enforced in `nba_model/web/auth.py` (`PREVIEW_PLAYERS`,
`PREVIEW_STATS`, `PREVIEW_MAX_GAMES`) and only applies with
`BILLING_ENABLED=1`. Billing is frozen: the app deploys open-access, and team
charts are free for everyone. Adjust there if you change the policy.

## 9. Web app views (parity with the desktop UI)

The Streamlit app at `nba_model/web/app.py` exposes the following view modes,
selectable from the sidebar `View mode` radio:

| View mode               | What it does                                       | Gating |
|-------------------------|----------------------------------------------------|--------|
| Player charts           | Per-player distribution + recent games + splits + hit-rate + custom-line probe | Free preview (Jokic/LeBron + points) / Premium full |
| Team charts             | Per-team aggregate distributions, per-stat overview | Free (all teams) |
| Compare players         | Overlay 2–3 players' distributions for one stat   | Free / Premium |
| All stats (overview)    | One page, every stat for a selected player        | Premium |
| Single prop (model)     | Calls `run_single_prop` — same model the desktop "Single Prop" tab uses | Premium |
| Parlay analysis         | Cross-compare model + chart-data + historical for single + multi-leg | Premium |
| Manual lines import     | Paste board / CSV / pipe rows, parse, optionally save to DB | Read free; **DB save = admin** |
| Game Results            | Recent NBA games + scores                          | Free / Premium |
| Player Stats Browse     | Searchable league-wide player game logs           | Free / Premium |
| Operations (admin)      | Subprocess launcher for ETL / scrapers / evaluation / DB audit | **Admin-only** |

The fitted-distribution selector reads `simulation.SUPPORTED_DISTRIBUTIONS`
plus `negative_binomial`, so when the model team adds a family there it
appears in the UI automatically without a code change in `app.py`.

### Line-display chart upgrades

All web charts now render via Plotly (no `st.pyplot(...)` calls remain in
the NBA views), so zoom / hover / legend-toggle work everywhere. The
distribution figure has two layouts selectable from the sidebar:

- **Distribution view** (default) — histogram + per-stat fitted overlays +
  per-book vertical markers with a top-rail of triangles labeled with each
  book's three-letter tag. The best-EV-over and best-EV-under markers get
  a thick gold-bordered star so line-shopping is obvious at a glance.
- **Line ladder** — compact one-row-per-book layout sorted by line value,
  with delta-vs-consensus shown on hover. Recommended whenever 6+ books are
  posting a line; verticals start to overlap in the distribution view.

A bonus `plotly_charts.build_line_movement_figure(snapshots_df, stat_type)`
draws a line-movement timeline from `betting_line_snapshots` rows — wire it
into a custom view if your deploy is populating that table.

## 10. Admin Operations console

The desktop "Operations" tab is mirrored as a Streamlit view at
`Operations (admin)`. It lives in `nba_model/web/operations_panel.py` and
provides forms + Run buttons for:

- **Daily ETL** (`nba_model.data.daily_etl`)
- **Web-text session validate / fetch / sync active players**
- **Browser prop parser**
- **Evaluation:** real-data benchmark, distribution sweep, line comparison,
  monthly diagnostics
- **Market reverse-engineering**
- **DB audit**

Each form translates field values into argv (no shell expansion ever, by
design — see `test_operations_panel.py::test_shell_metacharacters_pass_through_as_literal_arg`)
and streams stdout/stderr into a live transcript box.

### Hard gating

`Operations (admin)` is **admin-only at all times**, including in
open-access mode (`BILLING_ENABLED=0`). The gate at the top of `is_operations`
in `app.py` returns immediately unless `web_auth.is_admin()` is true, which
requires:

1. A logged-in OIDC session (so `BILLING_ENABLED=1` *plus* a Google /
   Microsoft sign-in), AND
2. The signed-in email listed in `[auth.admins]` of `.streamlit/secrets.toml`.

If no admins are configured, the view simply doesn't appear and the
deep-link `?view=Operations+(admin)` is blocked at the gate.

### Caveats

- Scraping operations (Validate Web Session, Fetch URL, Browser Parser)
  require a real Chrome on `:9222` on the **same host running Streamlit**.
  Streamlit Community Cloud cannot satisfy this — run from a self-host
  / docker-compose deploy.
- The Docker image runs as a non-root user (`app`) with no Chrome installed;
  Operations that need Chromium will fail with a clear error in the
  transcript box rather than corrupting the DB.

## 11. Open-access vs billing modes

The launch default is **open-access**: `BILLING_ENABLED` unset (or `0`).
Every visitor is treated as Premium, no OIDC sign-in is required, the
sidebar user-card stays hidden, and the Free/Premium gating helpers in
`auth.py` short-circuit to "premium".

To flip back to billing mode:

```bash
BILLING_ENABLED=1 docker compose --profile billing up streamlit webhook
```

When `BILLING_ENABLED=1`:

- OIDC sign-in is required for all premium features.
- Stripe Checkout / webhook flow drives `subscriptions.db`.
- Admin overrides from `[auth.admins]` continue to work (this is how
  developers get full access without paying themselves).

### Important: what changes vs what doesn't

The `Operations (admin)` and Manual-Lines DB-save surfaces are admin-gated
**regardless of mode**. Open-access does NOT relax these — the worst-case
trust model for an open URL is "anonymous internet visitor," and the
Operations console launches arbitrary subprocesses (and Manual Lines
writes into the shared `betting_lines` table) which is never something we
want anonymous visitors doing.

## 12. Manual lines import (DB writes)

The web "Manual lines import" view lets anyone preview the parser output
without authentication (good for sanity-checking a scraped board), but the
**Save to DB** button is admin-only — the public must not be able to
poison the shared `betting_lines` table.

Records also pass through `nba_model.web.input_validation.is_plausible_betting_line`
before being accepted; rows that fail the per-stat range check are dropped
with a visible "dropped by validator" expander and never written.

Parsing logic lives in `nba_model.model.manual_lines` and is reused by both
the desktop Tk UI and the Streamlit view, so the two paths can't drift.

## 13. Subscription store backend (SQLite vs Postgres)

`nba_model/web/subscriptions.py` now picks its backend from the environment:

| `SUBSCRIPTIONS_DB_URL`             | Backend  | When to use                    |
|------------------------------------|----------|--------------------------------|
| _unset_ or non-URL path            | SQLite   | local dev, self-host with disk |
| `postgres://…` / `postgresql://…`  | Postgres | Streamlit Cloud / Render       |

The public API (`tier_for` / `upsert` / `record_stripe_event` / `lookup`)
is identical across both backends, so `webhook_app.py` and `auth.py` don't
need to change.

**Hosted Postgres setup (Render / Neon / Supabase / RDS):**

```bash
# 1. Install the driver in the deploy image:
pip install 'psycopg[binary]>=3.1'

# 2. Set the DSN on the deployed service (Streamlit Cloud → Secrets,
#    Render → Environment, etc.). Example DSNs:
SUBSCRIPTIONS_DB_URL=postgresql://user:pw@db.example.com:5432/subs?sslmode=require

# 3. First deploy auto-creates the user_subscriptions / stripe_events
#    tables (same idempotent CREATE-IF-NOT-EXISTS path the SQLite backend
#    uses on every connect).

# 4. Sanity-check from the deploy shell:
python3 -c "from nba_model.web import subscriptions; print(subscriptions.selected_backend())"
# expected: postgres
```

Postgres session GUCs mirror the SQLite WAL/busy_timeout tuning:
`statement_timeout=5s`, `lock_timeout=5s`,
`idle_in_transaction_session_timeout=30s`. These keep a runaway webhook
from holding row locks long enough to wedge concurrent reads from
Streamlit.

## 14. Data delivery (hourly host → cloud)

This subsection coordinates the `nba_data.db` delivery between the
always-on hourly ETL host (the dev Mac running
`scripts/scheduler/hourly_update.sh`) and any read-only cloud deploy.

**Constraint:** Streamlit Community Cloud and Render's free tiers both have
ephemeral disks. Any file the app writes is wiped at the next redeploy —
which is unacceptable for the analytics DB the scraping host populates
hour by hour. So the hourly host has to own writes; the cloud surface
needs to receive a fresh read-only snapshot.

**Implemented option (simplest reliable):** git-commit refreshed DB on
every successful hourly run.

Why this option:

- Same trust boundary as the rest of the repo — no extra S3 credentials,
  no extra IAM, no extra cost.
- Streamlit Cloud already redeploys on push to `main`, so the cloud app
  picks up the new DB on the same heartbeat as code.
- The DB is currently ~60 MB; well inside Git LFS limits (or even raw
  git, with a periodic `git gc`).
- If we outgrow git, the next-simplest swap is object storage (S3 / R2)
  with a startup hook that pulls the latest blob — the
  `subscriptions.selected_backend()` pattern (env-selected DSN, lazy
  driver import) is the model.

Pieces:

1. `scripts/scheduler/hourly_update.sh` does the work hour-by-hour;
   the JSON report under `nba_model/data/artifacts/hourly/` flags whether
   the run was clean enough to publish.
2. **`scripts/publish_db.sh`** (implemented) is the downstream "publish"
   hook. On invocation it backs up the DB, refuses to publish when the DB
   is locked (an ETL writer is still active), logs row counts for the
   headline tables, then `git add` + `commit` + `push` only when the DB
   actually changed. It's a thin wrapper that re-execs
   `.venv/bin/python3 -m nba_model.data.publish_db` so cron / launchd can't
   pick up system python.

   Install: chain it after a clean hourly run. Either append to the
   launchd wrapper / a wrapping shell line:

   ```bash
   scripts/scheduler/hourly_update.sh && scripts/publish_db.sh
   ```

   or add a second launchd job that runs `scripts/publish_db.sh` a few
   minutes after the hourly tick. Flags:

   ```bash
   scripts/publish_db.sh --dry-run                 # print plan, touch nothing
   scripts/publish_db.sh --branch data --remote origin
   scripts/publish_db.sh --message "etl: manual publish"
   ```

   Exit codes: `0` ok / nothing-changed, `2` DB missing, `3` DB locked
   (retry next tick), `4` a git command failed. Pre-publish backups land
   in `data/database/backups/nba_data_<UTC>.db`.

   Note: `data/database/*.db` is gitignored, so the hook stages with
   `git add -f` — publishing the DB blob is deliberate, not accidental.
3. Streamlit Cloud auto-redeploys on push; the app reads
   `data/database/nba_data.db` at startup as today.

**Alternative (object storage):** if commit churn becomes a problem,
swap step 2 for `aws s3 cp data/database/nba_data.db s3://…/latest.db`
and add a `STARTUP_HOOK` that pulls the latest object before the
Streamlit app boots. Same env-selection pattern as `SUBSCRIPTIONS_DB_URL`
will keep the code paths clean.

> Coordination note for Agent B: this `## 14. Data delivery` subsection
> is owned by Agent A. The Streamlit "Manual Lines Import" view and any
> billing-flow docs you add should go under their own headings (e.g.
> `## 15. Manual lines view (web)`) to avoid edit collisions.

## 15. Flagship UI (React SPA + read-only API)

The flagship terminal (`frontend/` + `api/`) deploys as **one process on one
origin**: uvicorn serves `/api/*` and the Vite build. There's no CORS, no dev
proxy and no separate web server. It sits alongside the Streamlit app and the
webhook (§7a) rather than replacing them.

> **Prerequisites for ANY public deploy (read first)**
>
> Note: `Dockerfile.flagship` builds and passes `smoke_check --strict` inside
> the container, including the §16 object-storage boot pull. The production
> target is Fly.io; see §16.
> 1. The Streamlit security fixes in `review_findings_web.md` (W-SEC-1…3:
>    manual-lines admin gate, webhook status gating, admin iss+sub binding)
>    are in this tree. Keep them. Configure `[auth].admin_identities`
>    (`docs/SECURITY.md` → "Admin identity binding").
> 2. **There are no user accounts; real auth is deferred along with billing.**
>    The owner froze Stripe/billing work on 2026-09-28: the app deploys
>    open-access (`BILLING_ENABLED=0`), and per-user auth is built only if the
>    app takes off. Until then there are two modes:
>    - **Public** (default): `FLAGSHIP_ACCESS_CODE` unset. Everything the
>      flagship serves (scored slate, cross-book, parlay pricing, paper
>      trades) is readable by anyone.
>    - **Private preview**: set `FLAGSHIP_ACCESS_CODE` to a long random
>      string (e.g. `openssl rand -base64 24`) and share it. Every `/api/*`
>      call except `/api/health` then needs `X-Access-Code`. The UI prompts
>      once and stores the code in the browser's localStorage. This is a
>      *shared secret*, not auth: anyone holding the code has full access,
>      and rotating it (restart with a new value) logs everyone out. Wrong
>      codes are brute-force limited per IP. Serve it over HTTPS only.
> 3. **Rate limiting is built in** (per client IP, in-process; see
>    below). The proxy limits in the Caddy/nginx example are defence in
>    depth, not a requirement.

### Build + run

**The image contains no database.** `data/database/nba_data.db` (~65 MB,
gitignored, excluded by `.dockerignore`) is mounted read-only at runtime:

```bash
# Easiest: compose builds the image and bind-mounts ./data/database:/data:ro
docker compose up flagship            # http://localhost:8080

# Plain docker (from the repo root):
docker build -f Dockerfile.flagship -t nba-flagship .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/data/database:/data:ro" \
  -e NBA_DB_PATH=/data/nba_data.db \
  nba-flagship

# Verify (counts must match the host DB, not an empty schema):
python -m api.smoke_check --base-url http://localhost:8080 --strict
curl -s http://localhost:8080/api/slate/kpis
```

**If you forget the mount** (`docker run` without `-v`, or a wrong host
path), `/data` is empty. The API **does not** create an empty schema. Instead:
- `/api/health` returns **503** with `db_state: "not_mounted"` and
  `code: "db_not_mounted"`, so the container `HEALTHCHECK` (and Fly's check)
  fails.
- Every data endpoint returns 503 with the same code and a safe message (no
  paths).
- The UI shows a red "Database file not mounted" notice with the commands
  above, and the header pill says "No database". The offseason empty states
  ("No scored edges right now…") appear only when the DB *is* mounted and
  simply has no current lines.
- `smoke_check` prints the reason on its `health 200` failure.

A file that exists but isn't a usable NBA DB (corrupt, or missing core
tables) gives `db_state: "invalid"` / `code: "db_invalid"`, and nothing is
written to it.

Without Docker (same production path):

```bash
(cd frontend && npm ci && npm run build)
FLAGSHIP_STATIC_DIR=$PWD/frontend/dist API_DOCS=0 \
  .venv/bin/python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8080 \
  --no-server-header --proxy-headers --forwarded-allow-ips=127.0.0.1
```

### Environment

| Var | Default | Purpose |
|---|---|---|
| `NBA_DB_PATH` | `<repo>/data/database/nba_data.db` (image: `/data/nba_data.db`) | SQLite DB to read. Mount it **read-only**; the API never writes, and it refuses (503) any file that lacks the core tables rather than letting the data layer create a schema in it. |
| `FLAGSHIP_STATIC_DIR` | unset (image: `/app/frontend/dist`) | When set, serves the SPA: hashed `assets/*` get `immutable` caching, and client routes (`/player/2544`, `/edges`) fall back to `index.html`. |
| `PORT` | `8080` (image) | Listen port. |
| `API_DOCS` | `1` locally, `0` in the image | Swagger UI at `/api/docs`. It loads from a CDN, so keep it off in production (the CSP would block it anyway). |
| `API_TRUSTED_HOSTS` | unset | CSV of allowed `Host` headers (DNS-rebinding defence). **Set this in production.** |
| `API_EXPOSE_DB_PATH` | `0` | Include the absolute DB path in `/api/health` (local debugging only). |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | IPs uvicorn trusts for `X-Forwarded-*` (your proxy). |
| `FLAGSHIP_ACCESS_CODE` | unset (open) | Optional shared access code (private-preview mode, see above). |
| `API_RATE_LIMIT` | `1` | `0` disables the in-process limiter (trusted internal deploys only). |
| `API_RATE_HEAVY_PER_MIN` | `20` | Per-IP budget for CPU-heavy calls: `/api/slate/edges`, `/api/cross-book`, `/api/parlay/*`. |
| `API_RATE_READ_PER_MIN` | `240` | Per-IP budget for every other `/api/*` call (`/api/health` and static files are exempt). |
| `API_RATE_AUTH_FAIL_PER_MIN` | `10` | Per-IP budget of WRONG access codes before a 429 lockout. |
| `API_PROXY_TARGET` | `http://localhost:8000` | **Dev only**: where `npm run dev`'s Vite proxy sends `/api`. |

Ports at a glance: Streamlit **8501**, webhook **8000** (compose), flagship
**8080**, Vite dev **5173** (proxying to API dev on **8000**).

Every response carries a strict CSP (`default-src 'self'`; no inline or
third-party scripts, fonts or connections), `nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer` and COOP. Fonts are self-hosted (`@fontsource`),
so there are no external requests.

### Rate limiting (built in)

`api/guards.py` keeps a sliding 60-second window per client IP and returns
`429` with `Retry-After` when a budget is exhausted (budgets in the table
above). The client IP is `request.client.host`: behind a reverse proxy, run
uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy IP>` (the image
does this via `FORWARDED_ALLOW_IPS`). Uvicorn then takes the client from
X-Forwarded-For **only** when the request came from that proxy. Without it,
every visitor shares the proxy's IP, and spoofed X-Forwarded-For headers
from elsewhere are ignored. Budgets are per process: with N replicas, each
IP gets N× the budget, so also rate-limit at the proxy/CDN if you scale out.

The Streamlit app throttles scans per client IP as well
(`STREAMLIT_TRUSTED_PROXY_HOPS` = number of proxies in front of it, default 0,
which uses the socket peer).

### Reverse proxy (TLS + rate limit, defence in depth) — Caddy example

```caddy
props.example.com {
    encode zstd gzip
    # Per-IP limit on the expensive scan endpoints (caddy-ratelimit plugin).
    rate_limit {
        zone scans {
            match { path /api/slate/edges* /api/cross-book* }
            key {remote_host}
            events 30
            window 1m
        }
    }
    reverse_proxy 127.0.0.1:8080
}
```

With nginx, use `limit_req_zone $binary_remote_addr zone=scans:10m rate=30r/m;`
plus `limit_req zone=scans burst=10;` on `location ~ ^/api/(slate/edges|cross-book)`.

### Smoke check

```bash
python -m api.smoke_check --base-url http://localhost:8080          # local
python -m api.smoke_check --base-url https://props.example.com --strict
python -m api.smoke_check --base-url https://props.example.com --strict \
  --access-code "$FLAGSHIP_ACCESS_CODE"   # private-preview mode
```

The smoke check verifies:
- health (DB present, no path leak)
- the SPA shell on `/` and on a client route
- the core read endpoints
- that bad input gives a 4xx, never a 500
- the CSP and nosniff headers
- (`--strict`) that API docs are off
- when the server has an access code, that anonymous API calls get 401

It exits non-zero on any failure, so it can gate a deploy.

## 16. Flagship production target: Fly.io + Tigris (decision, 2026-09-28)

**Decision:** deploy the flagship (`Dockerfile.flagship`) to **Fly.io**, as
one small machine that scales to zero when idle. The nightly/hourly SQLite
snapshot goes through **Tigris**, Fly's S3-compatible object storage. The Mac
pushes a snapshot; the app pulls it and swaps it in atomically.

### Why Fly.io (vs Railway / a small VPS)

| Need | Fly.io + Tigris | Railway | Small VPS (e.g. €4 Hetzner) |
|---|---|---|---|
| DB refresh path | `fly storage create` makes the bucket **and** injects its credentials into the app (one account, one command). The Mac uploads with boto3, and the app pulls on boot and every 10 min. | Needs a volume plus a push channel you build yourself (no bundled storage), or a third-party bucket and a second account | `rsync` to the box is simplest, but you own the box |
| TLS | Automatic on `*.fly.dev` (and custom domains via `fly certs add`) | Automatic | You configure Caddy/nginx and renewals |
| Ops burden (student) | Containers only; no OS to patch | Low | OS patching, firewall, SSH hardening, backups, all yours |
| Cost at this traffic | Pay-as-you-go. A shared-cpu-1x/512 MB machine that auto-stops when idle, plus a small bucket, should cost cents to a few dollars a month. **Check fly.io/docs/about/pricing before relying on this**; I couldn't verify current prices from here. | ~$5/mo plan after the trial | ~€4/mo flat |
| In-app per-IP rate limits | Yes. Fly's proxy sets `Fly-Client-IP` and overwrites any client value, and `API_CLIENT_IP_HEADER=Fly-Client-IP` keys the limiter on it | Yes, via X-Forwarded-For | Yes, via your proxy |

The app shape is read-only API + static SPA + one 65 MB file (9 MB gzipped)
that changes at most hourly. It fits a pull-from-object-storage design, and
Fly is the only option here where storage, credentials and TLS come from a
single CLI with no server to maintain.

### DB delivery design

```
Mac (launchd :45 hourly)                     Tigris bucket                          Fly machine
publish_db.sh --to-object-store --skip-git   flagship/
  └ api.db_sync push                          snapshots/nba_data-<UTC>.db.gz   ◄── boot: db_sync pull (before uvicorn)
      1. refuse if ETL holds a write lock      snapshots/…db.gz.json (sha256)  ◄── every 600 s: background poll
      2. sqlite .backup → consistent snapshot  latest.json  ← pointer, written LAST   1. latest.json sha == local marker? done
      3. quick_check + required tables                                                2. download .gz → temp in /data
      4. sha256 unchanged? → no upload                                                3. gunzip + sha256 verify
      5. upload .gz + metadata                                                        4. quick_check + required tables
      6. repoint latest.json                                                          5. fsync + os.replace (atomic)
      7. prune: keep newest 7                                                         6. write sha marker
```

- **Zero-downtime-ish:** `os.replace` is atomic on the same filesystem, and
  the API opens a connection per request. In-flight requests finish on the
  old inode while new requests read the new file. Verified in the container:
  60/60 requests returned 200 during a live swap.
- **Never serves a bad file:** a sha mismatch, corrupt SQLite, missing tables
  or an oversize payload (>1 GB) is rejected and the current file stays live.
  Temp files are always cleaned up.
- **The API stays read-only:** there is no upload endpoint. The only writer on
  the box is the pull, and it only writes `/data`.
- **Rollback:** `python -m api.db_sync list` then
  `python -m api.db_sync rollback --steps 1` (or `--to <key>`). The app swaps
  it in on its next poll, within 10 min; to force it immediately, run
  `fly machine restart`.
- **Cold start:** with scale-to-zero, the first request after idle boots the
  machine and pulls about 9 MB from Tigris in the same region. That takes a
  few seconds, and only on the first request.

### Mac-side credentials

After `fly storage create`, the command prints the bucket credentials. Put a
**write** key for the bucket in `~/.config/nba-flagship/storage.env`:

```bash
mkdir -p ~/.config/nba-flagship
cat > ~/.config/nba-flagship/storage.env <<'ENV'
BUCKET_NAME=<bucket name from fly storage create>
AWS_ACCESS_KEY_ID=<key id>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_ENDPOINT_URL_S3=https://fly.storage.tigris.dev
AWS_REGION=auto
ENV
chmod 600 ~/.config/nba-flagship/storage.env   # publish_db.sh refuses anything looser
```

Test once by hand (`--dry-run` first), then install the launchd job:

```bash
scripts/publish_db.sh --to-object-store --skip-git --dry-run
scripts/publish_db.sh --to-object-store --skip-git
python -m api.db_sync list          # (source the env file first)

cp scripts/scheduler/com.nba.flagship-publish.plist ~/Library/LaunchAgents/
# replace ABSOLUTE_PATH_TO_REPO with this checkout's path, then:
launchctl load -w ~/Library/LaunchAgents/com.nba.flagship-publish.plist
```

**Cadence:** the job runs at :45 every hour, 40 minutes after the hourly ETL
(`com.nba.hourly` runs at :05). An unchanged DB uploads nothing. A DB still
being written exits 3 and retries at the next :45. For nightly-only, add
`<key>Hour</key><integer>4</integer>` to the plist's `StartCalendarInterval`.

### Deploy (owner steps at the account boundary)

```bash
brew install flyctl                      # once
fly auth login                           # browser login — OWNER ONLY
fly apps create nba-props-flagship       # or another name; then update `app`
                                         # and API_TRUSTED_HOSTS in fly.toml
fly storage create --app nba-props-flagship   # Tigris bucket + app secrets
fly secrets set --app nba-props-flagship \
  FLAGSHIP_ACCESS_CODE="$(openssl rand -base64 24)"   # recommended ON for now
# Publish the FIRST snapshot before deploying: /api/health answers 503 until
# the app has a database, so `fly deploy` would fail its health check on an
# empty bucket. (Write storage.env as in "Mac-side credentials" first.)
scripts/publish_db.sh --to-object-store --skip-git
fly deploy                               # builds Dockerfile.flagship remotely
```

Then check the live site:

```bash
python -m api.smoke_check --base-url https://nba-props-flagship.fly.dev --strict \
  --access-code "<the code you set>"
curl -s https://nba-props-flagship.fly.dev/api/health   # db_sync should be updated/unchanged
```

### Operations

| Task | Command |
|---|---|
| Logs (including `db_sync: swapped in …` lines) | `fly logs` |
| Status / machines | `fly status` |
| Roll back the **code** | `fly releases` then `fly deploy --image <previous image ref>` |
| Roll back the **data** | `python -m api.db_sync rollback --steps 1` |
| Rotate the access code | `fly secrets set FLAGSHIP_ACCESS_CODE=…` (restarts; everyone re-enters it) |
| Go fully public | `fly secrets unset FLAGSHIP_ACCESS_CODE` |
| Keep one machine warm (no cold start) | `min_machines_running = 1` in fly.toml, then `fly deploy` (costs more) |
| Custom domain | `fly certs add props.example.com`, then add it to `API_TRUSTED_HOSTS` |

### Environment added for §16

| Var | Where | Purpose |
|---|---|---|
| `BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`, `AWS_REGION` | Fly secrets (set by `fly storage create`); Mac `storage.env` | Tigris / S3 access |
| `DB_SYNC_INTERVAL_SECONDS` | fly.toml (`600`) | Poll cadence; `0` = boot-time pull only |
| `DB_SYNC_PREFIX` | fly.toml (`flagship/`) | Key prefix inside the bucket |
| `DB_SYNC_KEEP` | Mac env (default 7) | Snapshots retained for rollback |
| `DB_SYNC_LOCAL_DIR` | tests / shared-mount setups | Directory-backed store instead of S3 |
| `API_CLIENT_IP_HEADER` | fly.toml (`Fly-Client-IP`) | Rate-limit key from the platform's overwritten header. **Only behind a proxy that always sets it.** |

**Verified locally (2026-09-28):** the built image, pointed at a local S3 API
(moto) with the real DB pushed by `publish_db.sh --to-object-store`:
- downloaded the snapshot on boot and passed `smoke_check --strict` (19/19
  with the access code, 18/18 without)
- swapped in a new snapshot live with no failed requests
- rolled back live

The Fly account step itself has not happened yet.
