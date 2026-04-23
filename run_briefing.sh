#!/bin/bash
cd /home/ubuntu/.openclaw/workspace/atlas-morning-briefing
source venv/bin/activate
source ~/.openclaw/.env
rm -f Atlas-Briefing-*.md Atlas-Briefing-*.pdf status.json
python3 scripts/briefing_runner.py --config config.yaml
