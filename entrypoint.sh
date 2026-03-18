#!/bin/bash
set -e

CRON_SCHEDULE="${CRON_SCHEDULE:-50 5 * * *}"
DRY_RUN="${DRY_RUN:-false}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
VENV="/app/.venv/bin/python3"

DRY_RUN_FLAG=""
if [ "$DRY_RUN" = "true" ]; then
  DRY_RUN_FLAG="--dry-run"
fi

# Build the run command using the poetry venv python
RUN_CMD="cd /app && $VENV scripts/briefing_runner.py --config /app/config/config.yaml --log-level $LOG_LEVEL $DRY_RUN_FLAG >> /app/logs/briefing.log 2>&1"

# Export all env vars so cron can pick them up
printenv | grep -v "no_proxy" > /etc/environment

# Write cron job
echo "$CRON_SCHEDULE root . /etc/environment; $RUN_CMD" > /etc/cron.d/briefing
chmod 0644 /etc/cron.d/briefing
crontab /etc/cron.d/briefing

echo "Atlas Morning Briefing container started"
echo "Schedule: $CRON_SCHEDULE"
echo "Dry run:  $DRY_RUN"
echo "Log level: $LOG_LEVEL"

# Run once immediately if requested
if [ "${RUN_ON_START:-false}" = "true" ]; then
  echo "Running briefing immediately..."
  cd /app && $VENV scripts/briefing_runner.py --config /app/config/config.yaml --log-level "$LOG_LEVEL" $DRY_RUN_FLAG 2>&1 | tee -a /app/logs/briefing.log
fi

# Start cron in foreground
echo "Starting cron daemon..."
cron -f
