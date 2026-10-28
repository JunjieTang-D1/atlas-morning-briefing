#!/usr/bin/env python3
"""Atlas Morning Briefing -- KiroCrew script cron (deterministic pipeline).

Delivery note (verified against this build's source 2026-08-29):
  There is NO session="telegram". ctx.notify(session="origin") injects the
  message into the cron OWNER's dashboard session as a turn; if that owner
  session is a Telegram-originated session (origin-mirror bound, see
  kiro_crew.telegram.transport_dispatch._bind_origin_mirror), the resulting
  reply is mirrored back to the Telegram chat. Therefore this cron MUST be
  registered from the Telegram session (@KiroCrewJTBot) for Telegram delivery.
  raise Report(...) alone reaches only the dashboard/Slack fallback, never
  Telegram -- so it is kept only as a last-resort surface.
"""
import json
from datetime import date
from pathlib import Path

ATLAS_HOME = Path("/home/ec2-user/atlas-morning-briefing")  # EC2 path
RUNNER = ATLAS_HOME / "run_briefing.sh"


def _newest(pattern: str):
    files = sorted(ATLAS_HOME.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _tg_clean(text: str) -> str:
    """Strip markdown that Telegram renders inconsistently.

    Bold markers are the main offender: a stray/unbalanced `*` (or `**`) makes
    Telegram fail to parse the whole message and fall back to showing every
    marker literally. Safest path for cron delivery is to remove bold markup
    entirely (emoji section headers carry the visual weight instead), drop `###`
    headings, and flatten `[title](url)` -> `title — url`.
    """
    import re
    out_lines = []
    for line in text.splitlines():
        s = re.sub(r"^\s*#{1,6}\s*", "", line)                            # drop ### headings
        s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 — \2", s)  # [t](u) -> t — u
        s = s.replace("**", "").replace("__", "")                         # remove bold pairs
        s = re.sub(r"(?<!\w)\*(?!\s)([^*\n]+?)\*(?!\w)", r"\1", s)         # remove *italic/bold*
        s = s.replace("`", "")                                            # drop stray code ticks
        out_lines.append(s)
    return "\n".join(out_lines)


def _section(md: str, header: str) -> str:
    marker = f"## {header}"
    if marker not in md:
        return ""
    body = md.split(marker, 1)[1]
    if "\n## " in body:
        body = body.split("\n## ", 1)[0]
    return body.strip()


def counts_desc(status: dict) -> str:
    """Short human counts line for the artifact description."""
    return (
        f"papers {status.get('papers_found', '?')}, "
        f"news {status.get('news_found', '?')}, "
        f"blogs {status.get('blogs_found', '?')}, "
        f"stocks {status.get('stocks_fetched', '?')}"
    )


def _market_to_prose(market: str) -> str:
    """Convert the markdown market table into Telegram-friendly inline text.

    Telegram's Markdown parse mode does not support tables, so `| Ticker |...`
    rows render as garbled literal pipes. Flatten each data row into a
    `SYMBOL $price (±change%)` line instead.
    """
    lines = []
    for raw in market.splitlines():
        raw = raw.strip()
        if not raw.startswith("|"):
            # Keep any prose (e.g. market_trend) as-is.
            if raw:
                lines.append(raw)
            continue
        cells = [c.strip() for c in raw.strip("|").split("|")]
        # Skip header row and the |---|---| separator.
        if not cells or cells[0].lower() in ("ticker", "") or set(cells[0]) <= {"-"}:
            continue
        symbol = cells[0] if len(cells) > 0 else ""
        price = cells[1] if len(cells) > 1 else ""
        change = cells[2] if len(cells) > 2 else ""
        driver = cells[3] if len(cells) > 3 else ""
        seg = f"• {symbol}"
        if price and price != "—":
            seg += f" {price}"
        if change and change not in ("—", ""):
            seg += f" ({change})"
        if driver and driver not in ("—", ""):
            seg += f" — {driver}"
        lines.append(seg)
    return "\n".join(lines).strip()


def _tg_send_all(messages) -> bool:
    """Send each message DIRECTLY to Telegram via the Bot API, verbatim.

    Reads bot_token + the first allowed_user_ids entry (the DM chat_id) from
    ~/.kiro/crew/config.json and POSTs sendMessage per chunk with urllib — no
    gateway, no LLM turn, no paraphrasing. Plaintext (no parse_mode) so bare
    markup never trips a Telegram 400. Returns True only if every chunk was
    accepted; False (any failure / missing config) lets the caller fall back
    to the dashboard surface.
    """
    import json as _json
    import urllib.request
    import urllib.error

    cfg_path = Path.home() / ".kiro/crew/config.json"
    try:
        tg = _json.loads(cfg_path.read_text()).get("telegram", {})
    except Exception:
        return False
    token = tg.get("bot_token") or tg.get("token")
    ids = tg.get("allowed_user_ids") or []
    if not token or not ids:
        return False
    chat_id = ids[0]
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    all_ok = True
    for msg in messages:
        payload = _json.dumps({
            "chat_id": chat_id,
            "text": msg,
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    all_ok = False
        except Exception:
            all_ok = False
    return all_ok


def run(ctx):
    import subprocess

    from kiro_crew.cron_script import Report  # noqa: F401

    proc = subprocess.run(
        ["/bin/bash", str(RUNNER), "--dry-run"],
        capture_output=True, text=True, timeout=540,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-800:]
        raise Report(f"⚠️ Atlas briefing run FAILED (exit {proc.returncode}).\n{tail}")

    status = {}
    sp = ATLAS_HOME / "status.json"
    if sp.exists():
        try:
            status = json.loads(sp.read_text())
        except Exception:
            status = {}

    md_path = _newest("Atlas-Briefing-*.md")
    if not md_path:
        raise Report("⚠️ Atlas briefing: pipeline ok but no markdown file found.")
    md = md_path.read_text()
    today = date.today().isoformat()

    kb_status = "ok"
    try:
        ctx.call_tool(
            "kirocrew-core", "knowledge_add_document",
            {"title": f"Atlas Morning Briefing {today}", "content": md,
             "source_uri": str(md_path), "reason": "daily morning briefing archive"},
        )
    except Exception as e:  # noqa: BLE001
        kb_status = f"failed: {e}"

    # Archive the day's briefing as a dated ARTIFACT under morning_briefing/ so
    # past editions are browsable/recoverable even though run_briefing.sh's
    # `rm -f Atlas-Briefing-*` wipes the on-disk file each morning. Independent
    # of the KB path above (which is gated by knowledge.auto_add_documents and
    # may be disabled). Stable per-day slug => a same-day re-run versions in
    # place rather than duplicating.
    archive_status = "ok"
    try:
        ctx.call_tool(
            "kirocrew-core", "artifact_save",
            {
                "name": f"Atlas Briefing {today}",
                "slug": f"atlas-briefing-{today}",
                "kind": "markdown",
                "folder": "morning_briefing",
                "content": f"# Atlas Morning Briefing — {today}\n\n"
                           f"*Archived from the daily 08:00 CET run. Source: {md_path}*\n\n---\n\n"
                           + md,
                "description": f"Archived Atlas Morning Briefing for {today} ({counts_desc(status)}). "
                               f"Filed under morning_briefing/ for historical reference.",
                "tags": ["atlas-briefing", "morning-briefing", "archive", today[:7]],
            },
        )
    except Exception as e:  # noqa: BLE001
        # A same-day re-run hits a slug collision (artifact_save refuses a taken
        # slug); fall back to artifact_update so the re-run refreshes in place.
        try:
            ctx.call_tool(
                "kirocrew-core", "artifact_update",
                {"slug": f"atlas-briefing-{today}",
                 "content": f"# Atlas Morning Briefing — {today}\n\n"
                            f"*Archived from the daily 08:00 CET run. Source: {md_path}*\n\n---\n\n"
                            + md},
            )
            archive_status = "updated"
        except Exception as e2:  # noqa: BLE001
            archive_status = f"failed: {e2}"

    counts = (
        f"papers {status.get('papers_found', '?')} · "
        f"news {status.get('news_found', '?')} · "
        f"blogs {status.get('blogs_found', '?')} · "
        f"stocks {status.get('stocks_fetched', '?')}"
    )
    exec_sum = _section(md, "Executive Summary")
    market = _section(md, "Financial Market Overview")
    news = _section(md, "AI & Tech News")
    papers = _section(md, "Top Papers")
    blogs = _section(md, "Blog Updates")

    # Build ONE complete briefing body (all sections joined), then split into
    # the FEWEST Telegram-safe messages. Sections are joined with a blank line
    # so _fit() can pack multiple sections per message up to the 3800 limit;
    # only genuinely long runs spill into a 2nd/3rd message. This keeps the
    # briefing readable as a continuous document rather than 7 fragments.
    sections = [f"📊 今日简报 — {today}\n{counts}"]
    if exec_sum:
        sections.append("📝 Executive Summary\n" + _tg_clean(exec_sum))
    if market:
        sections.append("💰 Financial Market\n" + _tg_clean(_market_to_prose(market)))
    if news:
        sections.append("📰 AI & Tech News\n" + _tg_clean(news))
    if papers:
        sections.append("📑 Top Papers\n" + _tg_clean(papers))
    if blogs:
        sections.append("✍️ Blog Updates\n" + _tg_clean(blogs))
    sections.append(f"知识库归档: {kb_status} · PDF: {md_path.with_suffix('.pdf').name}")

    body = "\n\n".join(sections)

    # Pack the whole body into as few <=3800-char messages as possible,
    # splitting only on paragraph boundaries (never mid-line/mid-item).
    def _fit(text: str, limit: int = 3800):
        out, cur = [], ""
        for para in text.split("\n\n"):
            piece = para if not cur else cur + "\n\n" + para
            if len(piece) <= limit:
                cur = piece
            else:
                if cur:
                    out.append(cur)
                while len(para) > limit:
                    out.append(para[:limit])
                    para = para[limit:]
                cur = para
        if cur:
            out.append(cur)
        return out or [text[:limit]]

    messages = _fit(body)

    # Delivery: send each message DIRECTLY to Telegram via the Bot API.
    # This deliberately bypasses ctx.notify(session="origin"), which injects
    # each message as an LLM turn into the owner session — that path let the
    # session agent paraphrase/merge/drop/repeat sections across the multi-part
    # briefing. A direct sendMessage POST relays each chunk BYTE-FOR-BYTE with
    # no agent in the loop, so what _fit() packs is exactly what arrives.
    delivered = _tg_send_all(messages)

    # Fallback surface only if the direct Telegram send delivered nothing
    # (missing token / API error): dashboard notification + Slack DM.
    if not delivered:
        raise Report(messages[0])
