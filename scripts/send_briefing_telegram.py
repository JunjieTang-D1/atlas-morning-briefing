#!/usr/bin/env python3
"""
Send today's morning briefing PDF + summary to Telegram.

Usage: python3 send_briefing_telegram.py --chat-id <CHAT_ID>

Looks for the newest Atlas-Briefing-*.pdf in the briefing directory and sends it
as a document, then sends the markdown executive summary as separate text message(s).

Configuration (all via environment, no hardcoded personal values):
  TELEGRAM_BOT_TOKEN   Bot token (required).
  TELEGRAM_CHAT_ID     Target chat ID (used if --chat-id is not passed).
  ATLAS_BRIEFING_DIR   Directory holding the generated briefing files
                       (default: the repo root, i.e. the parent of this script's dir).
  ATLAS_ENV_FILE       Optional dotenv file to load keys from (default: none).
"""
import os
import sys
import json
import argparse
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

# Repo root = parent of the scripts/ directory this file lives in.
_REPO_ROOT = Path(__file__).resolve().parent.parent
BRIEFING_DIR = Path(os.environ.get("ATLAS_BRIEFING_DIR", str(_REPO_ROOT)))
ENV_FILE = Path(os.environ["ATLAS_ENV_FILE"]) if os.environ.get("ATLAS_ENV_FILE") else None


def load_env():
    if not ENV_FILE or not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def newest(glob_pattern):
    files = sorted(BRIEFING_DIR.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def extract_summary(md_path, max_chars=4000):
    """Extract sections from briefing markdown and return a list of message strings,
    each <= max_chars (Telegram per-message limit is 4096).
    """
    if not md_path or not md_path.exists():
        return []
    text = md_path.read_text()

    def grab_section(t, header):
        if header not in t:
            return None
        s = t.split(header, 1)[1]
        if "\n## " in s:
            s = s.split("\n## ", 1)[0]
        return s.strip()

    exec_sum = grab_section(text, "## Executive Summary")
    top_papers = grab_section(text, "## Top Papers")

    blocks = []
    if exec_sum:
        blocks.append("📊 *Atlas Morning Briefing*\n\n" + exec_sum)
    if top_papers:
        blocks.append("*📑 Top Papers*\n\n" + top_papers)

    if not blocks:
        return [text[:max_chars]]

    messages = []
    current = ""
    sep = "\n\n---\n\n"
    for b in blocks:
        if len(b) > max_chars:
            b = b[: max_chars - 50] + "\n\n_(truncated -- see PDF)_"
        if current and len(current) + len(sep) + len(b) <= max_chars:
            current = current + sep + b
        else:
            if current:
                messages.append(current)
            current = b
    if current:
        messages.append(current)
    return messages


def telegram_send_document(token, chat_id, pdf_path, caption=None):
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    # multipart/form-data
    boundary = "----telegrambriefingboundary"
    parts = []

    def add_field(name, value):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

    add_field("chat_id", str(chat_id))
    if caption:
        add_field("caption", caption[:1024])  # Telegram caption limit
        add_field("parse_mode", "Markdown")

    # File
    file_bytes = pdf_path.read_bytes()
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{pdf_path.name}"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def telegram_send_text(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": str(chat_id), "text": text[:4096], "parse_mode": "Markdown"}
    ).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        # If markdown parse failed, retry plain
        if "can't parse entities" in body.lower() or e.code == 400:
            data = urllib.parse.urlencode({"chat_id": str(chat_id), "text": text[:4096]}).encode()
            req2 = urllib.request.Request(url, data=data, method="POST")
            try:
                with urllib.request.urlopen(req2, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except Exception as ee:
                return {"ok": False, "error": f"retry failed: {ee}"}
        return {"ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"),
                    help="Target Telegram chat ID (or set TELEGRAM_CHAT_ID)")
    ap.add_argument("--pdf-only", action="store_true", help="Send only the PDF, no summary text")
    args = ap.parse_args()

    load_env()
    # Re-read chat id after load_env in case it came from the dotenv file.
    if not args.chat_id:
        args.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not args.chat_id:
        print("❌ No chat id: pass --chat-id or set TELEGRAM_CHAT_ID", file=sys.stderr)
        sys.exit(1)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    pdf = newest("Atlas-Briefing-*.pdf")
    md = newest("Atlas-Briefing-*.md")
    if not pdf:
        print(f"❌ No PDF found in {BRIEFING_DIR}", file=sys.stderr)
        sys.exit(2)

    print(f"📎 PDF:  {pdf.name} ({pdf.stat().st_size // 1024}KB)")
    print(f"📝 MD:   {md.name if md else 'N/A'}")
    print(f"💬 To:   chat_id={args.chat_id}")

    # Send PDF with caption = first ~200 chars of summary
    caption = f"📊 Atlas Morning Briefing — {pdf.stem.replace('Atlas-Briefing-', '')}"
    r1 = telegram_send_document(token, args.chat_id, pdf, caption=caption)
    if r1.get("ok"):
        print(f"✅ PDF sent (message_id={r1.get('result', {}).get('message_id')})")
    else:
        print(f"❌ PDF send failed: {r1.get('error', r1)}", file=sys.stderr)
        sys.exit(3)

    # Send summary text as separate message(s)
    if not args.pdf_only and md:
        messages = extract_summary(md)
        if messages:
            for idx, msg in enumerate(messages, 1):
                r2 = telegram_send_text(token, args.chat_id, msg)
                if r2.get("ok"):
                    print(f"✅ Summary part {idx}/{len(messages)} sent (message_id={r2.get('result', {}).get('message_id')})")
                else:
                    # Non-fatal; PDF already delivered
                    print(f"⚠️ Summary part {idx}/{len(messages)} send failed: {r2.get('error', r2)}", file=sys.stderr)

    print("✅ Telegram delivery complete")


if __name__ == "__main__":
    main()
