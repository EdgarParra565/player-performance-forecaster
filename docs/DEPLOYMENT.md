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
`PREVIEW_TEAMS`, `PREVIEW_STATS`, `PREVIEW_MAX_GAMES`). Adjust there if you
change the policy.

## 9. Web app views (parity with the desktop UI)

The Streamlit app at `nba_model/web/app.py` exposes the following view modes,
selectable from the sidebar `View mode` radio:

| View mode               | What it does                                       | Gating |
|-------------------------|----------------------------------------------------|--------|
| Player charts           | Per-player distribution + recent games + splits + hit-rate + custom-line probe | Free preview (Jokic/LeBron + points) / Premium full |
| Team charts             | Per-team aggregate distributions, per-stat overview | Free preview (LAL/DEN) / Premium full |
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
2. Add a downstream "publish" hook (left as a follow-up; not on the
   critical path for billing launch) that, on a clean run, does:
   `git add data/database/nba_data.db && git commit -m "etl: hourly snapshot $(date -u +%FT%TZ)" && git push origin main`.
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
