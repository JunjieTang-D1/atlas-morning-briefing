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

## Verify it runs

```bash
# Dry run: build the PDF locally, no email/Telegram delivery
./run_briefing.sh --dry-run
```

## Schedule it in KiroCrew (zero LLM tokens)

The pipeline is deterministic, so schedule it with a **`command=` cron** -- this runs
the shell directly and costs **zero KiroCrew LLM tokens** (no agent turn is spawned).

Register with the `cron_add` MCP tool:

```
cron_add(
  name    = "morning-briefing",
  command = "/ABSOLUTE/PATH/TO/atlas-morning-briefing/run_briefing.sh",
  cron_expr = "50 6 * * *",       # 06:50 daily (set timezone below)
  timezone  = "Europe/Berlin",
  timeout   = 600,                 # pipeline runs ~10-60s + LLM calls
)
```

Notes:
- Use an **absolute path** to `run_briefing.sh`.
- The script loads `$ATLAS_HOME/.env` automatically, so the cron does not need the
  API keys in its own environment -- they live in the gitignored `.env`.
- For delivery through KiroCrew instead of Gmail/Telegram, run with `--dry-run` in the
  cron and have an agent `file_send` the newest `Atlas-Briefing-*.pdf` (see "Delivery").

## Delivery options

| Channel | How | Secrets |
|---|---|---|
| Kindle / email | default (`run_briefing.sh` with no `--dry-run`) | `GMAIL_USER`, `GMAIL_APP_PASSWORD` |
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
