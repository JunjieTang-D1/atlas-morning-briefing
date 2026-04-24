"""HIPAA Audit Log — append-only PHI access logging."""
import json
import datetime
from pathlib import Path
from typing import Optional


class AuditLogger:
    """Append-only audit log for PHI access events."""

    def __init__(self, log_dir: str = "security/audit_log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"audit_{datetime.date.today().isoformat()}.jsonl"

    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str = "success",
        details: Optional[str] = None,
    ) -> dict:
        """Log a PHI access event. Returns the log entry."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,
            "details": details,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def read_log(self) -> list:
        """Read all entries from today\'s log."""
        if not self.log_file.exists():
            return []
        entries = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
