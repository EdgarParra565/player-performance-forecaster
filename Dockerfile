# Streamlit web app container for nba-probability-model.
#
# Build:
#   docker build -t nba-model .
#
# Run (default open-access mode; BILLING_ENABLED=false):
#   docker run --rm -p 8501:8501 \
#     -v "$(pwd)/data:/app/data" \
#     nba-model
#
# Re-enable Stripe billing + OIDC:
#   docker run --rm -p 8501:8501 \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" \
#     -e BILLING_ENABLED=1 \
#     nba-model
#
# Smaller surface: this image only carries what the web app actually serves.
# Playwright / browser scraping deps live in the desktop / ETL paths and are
# intentionally NOT included here.

FROM python:3.11-slim AS base

# Keep the image quiet about Python buffering + force UTF-8.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    BILLING_ENABLED=0

WORKDIR /app

# System packages we actually need: curl (for HEALTHCHECK), tini (PID 1).
# Skip the giant graphics stack; matplotlib only needs libfreetype which is
# already in python:3.11-slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps. We install from requirements.txt (compatible-release
# `~=` ranges) instead of requirements.lock because the lock pins
# pycookiecheat - a desktop-only dep used by the CDP scraping path - whose
# `cryptography==43.*` constraint clashes with the lock's `cryptography==48`.
# The web container needs neither pycookiecheat NOR Chromium; the `~=` ranges
# in requirements.txt give us patch+minor security updates inside one major
# version, which is enough determinism for a container we rebuild on every
# deploy.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs at runtime. Tests, scripts, backtests, and the
# desktop UI are deliberately excluded - they aren't reachable from the web
# routes and just inflate the image.
COPY nba_model ./nba_model
COPY setup.py ./

# Optional default data dir; real data is meant to be volume-mounted on top
# of this so the container is stateless.
COPY data ./data

# Streamlit defaults: bind to 8501 on all interfaces. The Streamlit healthz
# endpoint is /_stcore/health and returns 200 + body "ok".
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

# Run as a non-root user. We do this AFTER installing system + python deps
# (which need root) and AFTER COPY (which is fine as root).
RUN useradd --uid 10001 --create-home --shell /bin/false app \
 && chown -R app:app /app
USER app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "nba_model/web/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
