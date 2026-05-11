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
