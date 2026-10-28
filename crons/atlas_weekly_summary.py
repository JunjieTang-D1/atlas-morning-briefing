#!/usr/bin/env python3
"""Atlas Briefing — weekly executive-summary generator (LLM-written).

A deterministic script cron that assembles the last 7 days of archived
briefings and asks a TOP-TIER kiro-cli model (claude-opus-5.5) to write a real
executive summary, then caches the result to the atlas-briefing app's data dir.
The app's /weekly endpoint serves this cached text instantly (the LLM call is
far too slow for a per-request, 30s-polled dashboard endpoint), falling back to
its own regex synthesis when the cache is missing.

Why a script cron and not an agent cron: the output must be a clean prose blob
written to a specific file on a schedule, with no conversational state — a
code-based cron that shells `kiro-cli chat` once and writes the file is the
deterministic path. Registered to run daily after the briefing pipeline.

ctx contract (code cron): read ctx.message (unused here), raise Done(msg) to
deliver+remove, Report(msg) to deliver+keep, Skip() to retry. We keep the job
(Report) and keep delivery silent — this is a cache-refresh, not a notification.
"""
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARTIFACTS_HOME = Path("/home/ec2-user/.kiro/crew/artifacts")
# The installed app's data dir — the /weekly endpoint reads the cache from here.
APP_DATA_DIR = Path("/home/ec2-user/.kiro/crew/apps/atlas-briefing/data")
CACHE_FILE = APP_DATA_DIR / "weekly_summary.json"
MODEL = "claude-opus-5.5"  # top-tier; user asked for the top model on this summary


def _list_recent(days: int = 7) -> list[dict]:
    out = []
    if not ARTIFACTS_HOME.exists():
        return out
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    for d in ARTIFACTS_HOME.iterdir():
        m = re.match(r"atlas-briefing-(\d{4})-(\d{2})-(\d{2})$", d.name) if d.is_dir() else None
        if not m:
            continue
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if date >= cutoff:
            out.append({"slug": d.name, "date": date})
    out.sort(key=lambda x: x["date"], reverse=True)
    return out


def _section(md: str, header: str) -> str:
    marker = f"## {header}"
    if marker not in md:
        return ""
    body = md.split(marker, 1)[1]
    return body.split("\n## ", 1)[0].strip() if "\n## " in body else body.strip()


def _assemble_corpus(recent: list[dict]) -> str:
    """Compact the week's briefings to the sections an executive summary needs,
    keeping the prompt well within context while preserving real signal."""
    parts = []
    for b in recent:
        f = ARTIFACTS_HOME / b["slug"] / "current.html"
        try:
            md = f.read_text()
        except Exception:
            continue
        parts.append(f"=== {b['date']} ===")
        for sec in ("Executive Summary", "Top Papers", "AI & Tech News", "Financial Market Overview"):
            body = _section(md, sec)
            if body:
                parts.append(f"## {sec}\n{body[:2500]}")
    return "\n\n".join(parts)


_PROMPT = """You are writing the WEEKLY EXECUTIVE SUMMARY for a daily AI-research \
briefing dashboard. Below are the last 7 days of briefings (research papers, \
tech news, and market data), newest first.

Write a tight, structured executive summary of the WEEK (not any single day), \
in AT MOST 250 words total. Use EXACTLY these three sections, each a markdown \
H3 header on its own line followed by 2-3 sentences:

### Research & Industry
The dominant theme of the week and how it evolved; name specific papers/events.

### Top Papers
The most important or highest-rated papers and what they signify.

### Markets
The market picture — notable moves and their drivers.

Rules: concrete and specific, executive tone, no filler, no "this week we saw", \
do not restate counts. Keep the WHOLE thing under 250 words. Output ONLY the \
three sections with their headers, nothing else.

BRIEFINGS:
"""


def run(ctx):
    from kiro_crew.cron_script import Report, Skip  # type: ignore

    recent = _list_recent(7)
    if not recent:
        raise Skip("no briefings in the last 7 days yet")

    corpus = _assemble_corpus(recent)
    prompt = _PROMPT + corpus

    # kiro-cli may not be on the cron subprocess PATH; resolve it explicitly.
    import shutil
    cli = shutil.which("kiro-cli") or "/home/ec2-user/.local/bin/kiro-cli"

    try:
        proc = subprocess.run(
            [cli, "chat", "--no-interactive", "--model", MODEL, prompt],
            capture_output=True, text=True, timeout=240,
        )
        summary = (proc.stdout or "").strip()
    except Exception as e:  # noqa: BLE001
        raise Skip(f"LLM call failed, will retry: {e}")

    if not summary or len(summary) < 80:
        raise Skip("LLM returned too little; will retry next cycle")

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({
        "summary": summary,
        "model": MODEL,
        "window_days": 7,
        "date_range": f"{recent[-1]['date']} to {recent[0]['date']}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "briefings_used": [b["date"] for b in recent],
    }, indent=2))

    # Keep the job; this is a silent cache refresh, not a user notification.
    raise Report(f"atlas weekly summary refreshed ({len(summary)} chars, {MODEL})")
