# Hourly NBA ETL scheduler

The deployed model needs to be "updated every hour" — that means cycling
through web-text ingestion → prop / team-line parsing → game-log refresh →
team-priors → outcome settlement → prediction recompute, on the cadence of
1 / hour. The components live in `nba_model/data/hourly_update.py`; this
directory has the shell wrapper plus a launchd plist.

## Hard constraints

- **Must run on the dev Mac (or a Mac with a residential IP).** Sportsbook
  scraping needs a real Chrome on `--remote-debugging-port=9222`. Cloudflare
  / PerimeterX / DataDome fingerprint the TLS stack, so Playwright Chromium
  alone gets blocked. **Will not work in GitHub Actions or a container.**
- **Must use the project venv** (`.venv/bin/python3`). System Python doesn't
  have Playwright and will fail preflight.
- **Chrome must be logged in.** Open the Chrome window before the first tick
  and sign in to PrizePicks / Underdog / DK / BetMGM / Caesars in that
  same profile. Sessions persist across hourly ticks via the user-data-dir.

## One-time setup

```bash
# 1. Boot the persistent Chrome window (do this once after each restart):
open -na "Google Chrome" --args \
    --remote-debugging-port=9222 \
    --user-data-dir=/tmp/pp-chrome-profile

# 2. In that Chrome window, sign in to the books that require auth
#    (PrizePicks is the strict one). Then verify CDP is up:
curl http://127.0.0.1:9222/json/version

# 3. Sanity-check the runner end-to-end (uses the venv automatically):
./scripts/scheduler/hourly_update.sh
#    Look at the JSON report it writes under nba_model/data/artifacts/hourly/
```

## Install the launchd job

```bash
# Replace the placeholders inside the plist with the absolute path of this
# checkout (the file ships with /ABSOLUTE_PATH_TO_REPO placeholders).
sed -i '' "s|/ABSOLUTE_PATH_TO_REPO|$(pwd)|g" \
    scripts/scheduler/com.nba.hourly.plist

cp scripts/scheduler/com.nba.hourly.plist ~/Library/LaunchAgents/
launchctl unload -w ~/Library/LaunchAgents/com.nba.hourly.plist 2>/dev/null
launchctl load   -w ~/Library/LaunchAgents/com.nba.hourly.plist
```

Verify:

```bash
launchctl list | grep com.nba.hourly
tail -f nba_model/data/logs/launchd_stdout.log
ls -lt nba_model/data/artifacts/hourly | head
```

Stop / uninstall:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.nba.hourly.plist
rm ~/Library/LaunchAgents/com.nba.hourly.plist
```

## Cron fallback

If you'd rather use cron (e.g. you can't use launchd on a server):

```cron
0 * * * * /ABSOLUTE_PATH_TO_REPO/scripts/scheduler/hourly_update.sh \
    >> /ABSOLUTE_PATH_TO_REPO/nba_model/data/logs/cron.log 2>&1
```

## Idempotency / overlap safety

The runner takes an fcntl lock on `/tmp/nba_hourly_update.lock`. If an
hour-long run hasn't finished when the next tick fires, the second
invocation exits with code 75 (EX_TEMPFAIL) so launchd just retries on the
next hour boundary — it does *not* tear down the in-flight run.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All steps OK. |
| `1` | At least one ETL step failed. JSON report has per-step details. |
| `75` | Another hourly run is still in flight; this tick was skipped. |
| `78` | Preflight failed — Chrome unreachable on `:9222`, or Playwright not importable, or venv missing. Needs human intervention before the next tick. |

## What gets refreshed each hour

1. **Web text** — every URL in `data/config/web_text_urls.txt` via CDP.
2. **Browser prop parser** — PrizePicks / Underdog / Pick6 / ParlayPlay cards.
3. **Team-line parser** — BetMGM / Caesars / DraftKings / Bovada / Kalshi.
4. **Game logs** — recent 10 games for the top tracked players (nba_api).
5. **Team priors** — single-pass reverse engineering over the last 6h.
6. **Outcome settlement** — settle any predictions whose games landed.
7. **Prediction recompute** — re-score every `betting_lines` row dated today
   against the latest model so the prop board reflects fresh lines.

Each step's success/failure (and a per-step duration) lands in the JSON
report at `nba_model/data/artifacts/hourly/hourly_update_<ts>.json`.
