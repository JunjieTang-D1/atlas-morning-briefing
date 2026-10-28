"""Atlas briefing watchdog — self-healing safety net for the daily 08:00 CEST job.

Context: on 2026-10-04 the scheduled briefing cron (job e5d3fd58) silently did
not fire — the gateway process never restarted, there was no OOM/disk-full,
and no per-run error text was recoverable from the available tooling, so the
exact root cause is UNKNOWN. Rather than chase an unreproducible miss, this
watchdog makes a missed/failed fire self-healing: it runs a safe margin after
the expected time, checks for today's real output on disk, and if absent,
triggers the real job itself and notifies the owner. Deterministic, no LLM.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

ATLAS_HOME = Path("/home/ec2-user/atlas-morning-briefing")
BRIEFING_CRON_ID = "e5d3fd58"  # the real atlas-morning-briefing job


def run(ctx):
    today = date.today().isoformat().replace("-", ".")
    expected = ATLAS_HOME / f"Atlas-Briefing-{today}.md"

    if expected.exists():
        # Already produced today (normal case) -- nothing to do, stay silent.
        return

    # No md for today ~30min after the expected 08:00 CEST fire. Self-heal:
    # trigger the real job directly rather than just notifying and waiting.
    try:
        ctx.call_tool("kirocrew-cron", "cron_trigger", {"job_id": BRIEFING_CRON_ID})
        ctx.notify(
            f"⚠️ Atlas briefing watchdog: today's briefing ({today}) was missing "
            f"~30min after the expected 08:00 CEST fire -- triggered it now as a "
            f"self-heal. If this keeps recurring, the underlying scheduler miss "
            f"needs investigation (root cause was not determinable on 2026-10-04)."
        )
    except Exception as e:  # noqa: BLE001
        ctx.notify(
            f"⚠️ Atlas briefing watchdog: today's briefing ({today}) is missing AND "
            f"the self-heal trigger itself failed ({e!r}). Needs manual attention."
        )
