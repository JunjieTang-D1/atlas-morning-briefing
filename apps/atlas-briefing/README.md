# Atlas Briefing

Dashboard for the Atlas Morning Briefing pipeline.

## What it shows
- **Pipeline Performance** — last run's paper/news/blog/stock counts, error state, configured topic count.
- **Hot Topics** — keyword frequency across the last 7 days of archived briefings' Top Papers + News sections.
- **Weekly Summary** — per-day paper/news counts for the last 7 archived editions.
- **Knowledge Graph** — a lightweight day↔topic co-occurrence graph (not a full semantic KG) over the last week.

## What it does NOT do
This app is read-only. It does not own or modify the briefing pipeline, its
cron jobs, or its config — those remain:
- `~/.kiro/crew/crons/atlas_brief.py` (job `e5d3fd58`, daily 08:00 CEST)
- `~/.kiro/crew/crons/atlas_watchdog.py` (job `4538f3af`, self-heal, 06:30 UTC)
- `/home/ec2-user/atlas-morning-briefing/config.yaml`

It reads: `status.json` and `config.yaml` from the briefing repo, and archived
daily-briefing artifacts filed under the `morning_briefing/` folder (slugs
`atlas-briefing-YYYY-MM-DD`).

## Permissions
- `network: false` — no outbound calls; everything is local filesystem + artifact reads.
- `cron: false` — this app schedules nothing; the existing crons are untouched.
- `mcpTools`: knowledge-base + artifact read tools only.
