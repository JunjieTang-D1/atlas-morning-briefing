# Atlas Briefing

Dashboard for the Atlas Morning Briefing pipeline.

## What it shows
- **Pipeline Performance** — last run's paper/news/blog/stock counts, error state, configured topic count.
- **Hot Topics** — keyword frequency across the last 7 days of archived briefings' Top Papers + News sections.
- **Weekly Summary** — per-day paper/news counts for the last 7 archived editions.
- **Knowledge Graph** — a lightweight day↔topic co-occurrence graph (not a full semantic KG) over the last week.

## What it does NOT do
This app is read-only. It does not own or modify the briefing pipeline, its
cron jobs, or its config. Schedule the pipeline separately (see
`references/kirocrew_setup.md`).

It reads: `status.json` and `config.yaml` from the briefing repo, and archived
daily-briefing artifacts in the Kiro Crew artifact store (slugs
`atlas-briefing-YYYY-MM-DD`).

## Install

```bash
kirocrew app install ~/atlas-morning-briefing/apps/atlas-briefing
kirocrew app enable atlas-briefing   # third-party apps must be trusted first
```

Grant trust from the consent dialog on the dashboard **Apps** page, or add
`atlas-briefing` to `agent.apps_trusted` in Settings.

## Paths

| Variable | Default | Read for |
|---|---|---|
| `ATLAS_HOME` | `~/atlas-morning-briefing` | `status.json`, `config.yaml` |
| `ATLAS_ARTIFACTS_HOME` | `$KIROCREW_HOME/artifacts` (`~/.kiro/crew/artifacts`) | archived briefings |
| `ATLAS_RESEARCH_HOME` | `~/workplace/research-papers` | Research Topics panel (empty if missing) |

Set any of them in the gateway's environment, for example in
`~/.kiro/crew/.env`, then restart the gateway.

## Permissions
- `network: false` — no outbound calls; everything is local filesystem + artifact reads.
- `cron: false` — this app schedules nothing; the existing crons are untouched.
- `mcpTools`: knowledge-base + artifact read tools only.
