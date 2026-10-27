# Experiments (unmaintained)

Code kept for reference. Nothing in the production pipeline imports it, it is not
part of the installed package, and it has no tests.

| File | What it is |
|---|---|
| `briefing_runner_v2.py` + `workers/` | v0.2 coordinator + parallel-workers architecture |
| `benchmark_v1_v2.py` | Speed / quality / cost benchmark of `scripts/briefing_runner.py` vs v0.2 |
| `benchmark_analysis.py` | Static estimate of the same comparison, without running either pipeline |

The production entry point is `scripts/briefing_runner.py` (via `run_briefing.sh`).

**Known issue:** as of 2026-10-27, `briefing_runner_v2.py --dry-run` fails. The news
worker passes the whole config dict to `NewsAggregator`, so every Brave query errors,
and the PDF step then raises `ValueError: format not resolved` in ReportLab.
It also ignores `bedrock.enabled: false`: the coordinator and every worker build a
`BedrockClient` unconditionally, so a "zero-cost" config still makes paid Bedrock calls
whenever AWS credentials are available. Fix these before relying on the v0.2 runner or
its benchmark.

Run from the repo root, for example:

```bash
python experiments/briefing_runner_v2.py --config config.yaml --dry-run
```
