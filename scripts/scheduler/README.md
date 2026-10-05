# Hourly NBA ETL scheduler (scraping autopilot)

> **Status: OFF since 2026-09-28 (owner).** All three agents are unloaded and
> `launchctl disable`d (California geo-blocks — see
> `data/config/blocked_books.txt`), so they won't load at login. Don't load
> them unless the owner re-enables scraping. To turn it back on:
> `launchctl enable gui/$(id -u)/<label>` for each label, then the
> `launchctl bootstrap` steps below.

The deployed model is "updated every hour": web-text ingestion → prop /
team-line parsing → VegasInsider ingestion → game-log refresh → team priors →
outcome settlement → prediction recompute, once per hour. The pipeline lives in
`nba_model/data/hourly_update.py`; this directory holds the shell wrappers and
two launchd LaunchAgents:

| Agent | What it does |
|---|---|
| `com.nba.scraping-chrome` | Keeps a **dedicated scraping Chrome** running (`--remote-debugging-port=9222`, its own persistent `--user-data-dir`). `KeepAlive=true`: relaunched after a crash, a Cmd-Q, or a reboot/login. |
| `com.nba.hourly` | Runs `hourly_update.sh` at minute :05 of every hour (wall clock) and once at load. A fire missed while the Mac slept runs once on wake. |
| `com.nba.keep-awake` | Optional: `caffeinate -s` — no *system* sleep while on AC power (display still sleeps; battery behaves normally). Without it the loop only runs while the Mac is awake. |

With both loaded the loop runs unattended. The **only** routine manual act is
signing back in to a book when the hourly run alerts "re-login needed".
Nothing here logs in for you, stores credentials, or solves captchas.

## Hard constraints

- **Dev Mac (residential IP) only.** Sportsbooks fingerprint the TLS stack
  (Cloudflare / PerimeterX / DataDome), so scraping needs a real Chrome.
  **Will not work in GitHub Actions or a container.**
- **Project venv** (`.venv/bin/python3`) — the wrapper enforces it.
- **The scraping profile is not your daily Chrome profile.** It lives at
  `~/Library/Application Support/nba-scraping-chrome` (override with
  `NBA_SCRAPER_PROFILE_DIR`); the launcher refuses the daily-profile path.
  It is persistent, so logins survive restarts. (Chrome ≥ 136 ignores
  `--remote-debugging-port` on the default profile anyway.)

## One-time setup

```bash
cd /path/to/nba-probability-model
mkdir -p nba_model/data/logs

# 1. Install both LaunchAgents with this checkout's absolute path filled in.
for agent in com.nba.scraping-chrome com.nba.hourly com.nba.keep-awake; do
  sed "s|/ABSOLUTE_PATH_TO_REPO|$(pwd)|g" "scripts/scheduler/${agent}.plist" \
    > ~/Library/LaunchAgents/${agent}.plist
done

# 2. Optional: alert webhook (Slack/Discord/generic). Edit
#    ~/Library/LaunchAgents/com.nba.hourly.plist and uncomment the
#    NBA_ALERT_WEBHOOK_URL entry. Without it, alerts still land in the JSON
#    report + log and as a macOS notification.

# 3. Start the scraping Chrome first, then the hourly job.
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nba.scraping-chrome.plist
curl -s http://127.0.0.1:9222/json/version   # CDP should answer within a few seconds

# 4. One-time login pass (the only manual step) — see "Login pass" below:
scripts/scheduler/login_setup.sh

# 5. Start the hourly loop (+ optional keep-awake).
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nba.hourly.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nba.keep-awake.plist
```

(`launchctl load -w <plist>` / `unload -w <plist>` still work on current
macOS if you prefer the legacy verbs.)

## Verify

```bash
launchctl list | grep com.nba                      # both agents, PID for chrome
tail -f nba_model/data/logs/launchd_stdout.log     # hourly runner
tail nba_model/data/logs/scraping_chrome_stderr.log
ls -lt nba_model/data/artifacts/hourly | head      # JSON reports
cat nba_model/data/artifacts/hourly/session_health_state.json
```

Each report carries a top-level `session_health` map, e.g.
`{"prizepicks": "login-needed", "draftkings": "ok", "kalshi": "unreachable"}`,
and the `web_text` step's `result.tab_hygiene` lists the stale tabs it closed.

## Stop / uninstall

`KeepAlive` relaunches Chrome if you just quit it — unload the agent instead:

```bash
launchctl bootout gui/$(id -u)/com.nba.hourly
launchctl bootout gui/$(id -u)/com.nba.keep-awake
launchctl bootout gui/$(id -u)/com.nba.scraping-chrome
rm ~/Library/LaunchAgents/com.nba.{hourly,keep-awake,scraping-chrome}.plist
```

To pause scraping temporarily, boot out `com.nba.hourly` only.

## What runs unattended each hour

1. **Preflight** — Playwright importable + CDP reachable on `:9222`. If Chrome
   is down the run writes a report, fires the alert, exits 78; launchd will
   usually have Chrome back up for the next tick.
2. **Tab hygiene** (default ON; `--no-close-stale-tabs` to skip) — closes every
   open tab on a target URL's HOST. The CDP fetcher reuses any tab on the same
   domain *without navigating*, which silently stores a stale page under a
   different URL; closing them forces a fresh navigation per URL.
3. **Web text** — every URL in `data/config/web_text_urls.txt` via CDP.
4. **Session health** — each fetched page is classified with the book's
   `session_markers` (`detect_login_wall`, the same check the parsers use to
   skip walled snapshots). Per book: `ok` / `login-needed` (any page walled) /
   `unreachable` (no page fetched). **Alerts fire only on a NEW login-needed**:
   the state file remembers which books were already alerted; a book re-arms
   once it is seen `ok` again, `unreachable` blips don't re-fire, and a failed
   webhook delivery retries next tick. Channels: `NBA_ALERT_WEBHOOK_URL` /
   `--alert-webhook-url`, plus a macOS notification (`--no-macos-notify` to
   silence). The step never fails the run, so it can't spam the hourly alert.
5. Browser prop parser → team-line parser → VegasInsider (NBA grid →
   `betting_lines`, MLB props → `mlb_prop_lines`) → game logs + players.team
   sync → team priors (6h window) → outcome settlement → prediction recompute
   (optional `--settle-bet-log`).

## Login pass (`login_setup.sh`) — the one manual act

```bash
scripts/scheduler/login_setup.sh                      # all auth books, 30-min timeout
scripts/scheduler/login_setup.sh --books pick6        # after a single re-login alert
scripts/scheduler/login_setup.sh --timeout 3600
```

It opens one tab per auth book (PrizePicks, Underdog, Pick6, ParlayPlay, plus
any book the last hourly run flagged `login-needed`) in the scraping Chrome,
then reprints a checklist every 30s as you sign in:

```
--- 17:48:32Z  4/4 books ok ---
[ok]   prizepicks  ok
[ .. ] pick6       login-needed   (Generic login nav 'log in' present ...)
```

It only reads page text (no navigation, no typing). When every tab reads ok it
re-checks each book in a fresh tab and exits 0; `--timeout` exits 3 listing
what's left; Ctrl-C is safe. While it runs it holds the hourly lockfile, so a
tick that fires mid-login exits 75 instead of closing your login tabs.

## Blocked books (`data/config/blocked_books.txt`)

Books the owner can't fix by logging in — geo (`betmgm`, `underdog` from
California), age (`betrivers`, `hardrockbet`: 21+) — are listed here. They
report `blocked` in `session_health`, never alert, and are never prompted by
`login_setup.sh`. If a blocked book's page ever reads real content, the report
shows it `ok` and lists it under `blocked_but_ok`. Delete a line to put a book
back (e.g. when your location changes).

## Re-login runbook

1. Alert says e.g. "Scraper re-login needed: prizepicks".
2. Run `scripts/scheduler/login_setup.sh --books prizepicks` and sign in in the
   tab it opens (or just sign in in the scraping Chrome window).
3. Next hourly report shows the book `ok`; the alert re-arms.

Public books (sportsbooks, Kalshi, VegasInsider) never raise a re-login alert
from a merely generic "Log in" nav link — a capture where the board didn't
render reports `unreachable` instead.

Offseason caveat: an AUTH book's lobby with a "Log in" nav link and no live
lines can classify `login-needed` (generic-nav rule). Expect at most one alert
per book per episode; it clears when real content returns.

## Scheduling gotchas (learned live 2026-09-28)

- `StartInterval` timers count only AWAKE time, so on a Mac that idle-sleeps
  (this one: `sleep 1`) an hourly interval job effectively never fires — hence
  `StartCalendarInterval`.
- `launchctl kickstart gui/$(id -u)/com.nba.hourly` runs a tick now (same path
  as the schedule) — handy for manual verification.
- Check the schedule: `launchctl print gui/$(id -u)/com.nba.hourly | grep -A8
  "event triggers"` shows the `Minute => 5` calendar trigger; `runs = N`
  counts fires since load.

## Idempotency / overlap safety

The runner takes an fcntl lock on `/tmp/nba_hourly_update.lock`. If a run
hasn't finished when the next tick fires (or `login_setup.sh` is running), the
second invocation exits 75 (EX_TEMPFAIL) and launchd tries again next hour —
the in-flight run continues.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All steps OK (a login-needed book does not change this). |
| `1` | At least one ETL step failed. JSON report has per-step details. |
| `75` | Another hourly run is still in flight; this tick was skipped. |
| `78` | Preflight failed — venv missing (shell), or Chrome CDP unreachable / Playwright missing (Python; report + alert written). |

`scraping_chrome.sh` exits 78 for a missing Chrome binary or a refused
(daily) profile path; launchd keeps retrying every 30s (`ThrottleInterval`).

## Cron fallback (hourly job only)

```cron
0 * * * * /ABSOLUTE_PATH_TO_REPO/scripts/scheduler/hourly_update.sh \
    >> /ABSOLUTE_PATH_TO_REPO/nba_model/data/logs/cron.log 2>&1
```

Cron can't keep Chrome alive; start `scripts/scheduler/scraping_chrome.sh`
yourself (or still use the Chrome LaunchAgent).
