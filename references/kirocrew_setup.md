---
name: morning-briefing
description: Generate a daily AI research + market + news briefing (arxiv papers, tech blogs, a stock watchlist, and industry news) rendered to a Kindle-optimized PDF. Use when the user wants to set up an automated morning briefing, a daily research digest, or a scheduled knowledge feed. Runs as a plain Python pipeline; optionally uses Amazon Bedrock for AI synthesis, but degrades to a zero-cost deterministic mode with no LLM. Delivery via Gmail/Kindle, Telegram, or hand off the generated PDF to KiroCrew.
---

# Morning Briefing (KiroCrew)

This is the **KiroCrew-oriented** guide for the Atlas Morning Briefing pipeline.
The pipeline itself is plain Python (see the repo `README.md` for the full feature
set); this file covers **installing it as a KiroCrew skill and scheduling it**.

## Install (one time)

```bash
# 1. Clone anywhere (this becomes ATLAS_HOME)
git clone https://github.com/JunjieTang-D1/atlas-morning-briefing.git
cd atlas-morning-briefing

# 2. Isolated venv + deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Config + secrets (both are gitignored)
cp config.yaml.example config.yaml   # edit topics / stocks / delivery
# put API keys in a dotenv file, e.g. .env:
#   FINNHUB_API_KEY=...
#   BRAVE_API_KEY=...
#   GMAIL_USER=...            (optional, for Kindle/email delivery)
#   GMAIL_APP_PASSWORD=...    (optional)
#   TELEGRAM_BOT_TOKEN=...    (optional, for Telegram delivery)
#   TELEGRAM_CHAT_ID=...      (optional)
```

`run_briefing.sh` is now portable -- it resolves its own directory, so it works
from any checkout. Override paths via env if needed:
`ATLAS_HOME`, `ATLAS_ENV_FILE`, `ATLAS_VENV`, `ATLAS_CONFIG`.

On Windows (PowerShell), create the venv with `py -3.12 -m venv venv` and install
with `.\venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Verify it runs

```bash
# Dry run: build the PDF locally, no email/Telegram delivery
./run_briefing.sh --dry-run
# Windows (run_briefing.sh is bash-only):
#   .\venv\Scripts\python.exe scripts\briefing_runner.py --config config.yaml --dry-run
```

## Schedule it in KiroCrew (zero LLM tokens)

The pipeline is deterministic, so schedule it with a **script cron**: KiroCrew runs a
Python function directly and spawns no agent turn, so it costs **zero LLM tokens**.
`integrations/kirocrew/atlas_briefing_cron.py` works on macOS, Linux and Windows. It
runs the pipeline from the repo's `venv`, loads `$ATLAS_HOME/.env`, and reports the
newest PDF plus the run counts to the dashboard.

```bash
# 1. Script crons must live in the KiroCrew crons folder
mkdir -p ~/.kiro/crew/crons
cp integrations/kirocrew/atlas_briefing_cron.py ~/.kiro/crew/crons/

# 2. Try it once (runs the real pipeline, prints the report)
kirocrew cron preview "~/.kiro/crew/crons/atlas_briefing_cron.py:run" -m ""

# 3. Register: daily 06:50 Berlin time, dry run (PDF only)
kirocrew cron add "atlas-briefing" "" --cron "50 6 * * *" --timezone Europe/Berlin \
  --script "~/.kiro/crew/crons/atlas_briefing_cron.py:run" \
  --timeout 900 --timeout-secs 960 --no-persistent-session --minimal-context
```

Notes:
- The cron message selects the mode: `deliver` sends email/Kindle, anything else
  (including `""`) is a dry run.
- The repo is expected at `~/atlas-morning-briefing`. For another location set
  `ATLAS_HOME` in the gateway environment (`~/.kiro/crew/.env`) or edit
  `DEFAULT_ATLAS_HOME` in the copied script.
- KiroCrew expands the quoted `~` itself, so the same commands work in PowerShell
  (use `New-Item`/`Copy-Item` for step 1).
- A `command=` cron pointing at `run_briefing.sh` also works on macOS/Linux, but
  KiroCrew refuses command crons on Windows (they need a POSIX `/bin/sh`).
- For delivery through KiroCrew instead of Gmail/Telegram, keep the dry run and have an
  agent `file_send` the newest PDF (see "Delivery").

## Dashboard app (optional)

`apps/atlas-briefing` adds a read-only dashboard page. See its `README.md` for
install, trust, and the `ATLAS_HOME` / `ATLAS_ARTIFACTS_HOME` /
`ATLAS_RESEARCH_HOME` path variables.

## Delivery options

| Channel | How | Secrets |
|---|---|---|
| Kindle / email | `run_briefing.sh` with no `--dry-run`, or cron message `deliver` | `GMAIL_USER`, `GMAIL_APP_PASSWORD` |
| Telegram | `python3 scripts/send_briefing_telegram.py --chat-id <id>` | `TELEGRAM_BOT_TOKEN` (+ `--chat-id` or `TELEGRAM_CHAT_ID`) |
| KiroCrew (PDF to dashboard/outbox) | run `--dry-run`, then `file_send` the newest PDF | none |

## Zero-cost mode

Set `bedrock.enabled: false` in `config.yaml` to run with no LLM at all: papers are
scored by TF-IDF, top-N by original order, deterministic PDF. Fetches + delivery still
work; you just lose the AI summaries/synthesis.

## Security notes

- Secrets are read from environment / a gitignored `.env`; nothing is hardcoded.
- `config.yaml` and `.env` are gitignored -- keep them out of commits.
- The pipeline is **not production-hardened** (see repo `README.md` disclaimer): it is a
  personal/prototype tool. Do not point it at production credentials or untrusted config.
