"""Claude Code Worker Simulator

Simulates what a Claude Code session does:
- Receives task assignment
- Writes code to workspace
- Runs tests
- Self-debugs: if tests fail, reads error, fixes code, reruns
"""
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result of a Claude Code worker execution."""
    story_id: str
    agent_role: str
    files_written: list[str] = field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    self_repair_count: int = 0
    self_repair_log: list[str] = field(default_factory=list)
    final_status: str = "pending"  # "pass", "fail", "blocked"
    error_message: str = ""
    duration_ms: int = 0


class ClaudeCodeWorker:
    """Simulates a Claude Code session that writes, tests, and self-debugs code."""

    def __init__(self, workspace_dir: Path, max_repair_attempts: int = 3):
        self.workspace_dir = workspace_dir
        self.max_repair_attempts = max_repair_attempts
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, story: Dict[str, Any], agent_role: str) -> WorkerResult:
        """Execute a user story: write code → test → self-debug if needed."""
        start = time.time()
        result = WorkerResult(
            story_id=story["id"],
            agent_role=agent_role,
        )

        resource = story.get("resource", "")
        test_file = story.get("test_file", "")

        # Step 1: Generate code
        code = self._generate_code(story)
        code_path = self.workspace_dir / resource
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)
        result.files_written.append(str(resource))
        logger.info(f"[{agent_role}] Wrote {resource}")

        # Step 2: Generate test
        test_code = self._generate_test(story)
        if test_file:
            test_path = self.workspace_dir / test_file
            test_path.parent.mkdir(parents=True, exist_ok=True)
            # Append if test file exists (QA may share)
            if test_path.exists():
                existing = test_path.read_text()
                test_code = existing + "\n\n" + test_code
            test_path.write_text(test_code)
            result.files_written.append(str(test_file))

        # Step 3: Run tests with self-debug loop
        code, attempts = self.self_debug_loop(code, code_path, test_file, result)

        result.duration_ms = int((time.time() - start) * 1000)
        return result

    def self_debug_loop(
        self,
        code: str,
        code_path: Path,
        test_file: str,
        result: WorkerResult,
    ) -> Tuple[str, int]:
        """Write → Test → Fix → Retest loop."""
        for attempt in range(self.max_repair_attempts + 1):
            test_output = self._run_tests(test_file)
            result.tests_run += 1

            if test_output["passed"]:
                result.tests_passed += test_output["num_passed"]
                result.tests_failed += test_output["num_failed"]
                result.final_status = "pass"
                if attempt > 0:
                    result.self_repair_log.append(
                        f"Attempt {attempt}: Fixed — {test_output['num_passed']} passed"
                    )
                logger.info(f"[{result.agent_role}] Tests passed on attempt {attempt + 1}")
                return code, attempt

            # Self-debug: read error, fix code
            if attempt < self.max_repair_attempts:
                result.self_repair_count += 1
                error_msg = test_output.get("error", "Unknown error")
                result.self_repair_log.append(
                    f"Attempt {attempt + 1}: FAIL — {error_msg}"
                )
                code = self._fix_code(code, error_msg, result.story_id)
                code_path.write_text(code)
                logger.info(
                    f"[{result.agent_role}] Self-repair attempt {attempt + 1}: {error_msg[:80]}"
                )
            else:
                result.tests_passed += test_output["num_passed"]
                result.tests_failed += test_output["num_failed"]
                result.final_status = "fail"
                result.error_message = test_output.get("error", "Max repair attempts exceeded")

        return code, self.max_repair_attempts

    def _generate_code(self, story: Dict[str, Any]) -> str:
        """Generate realistic Python code for a user story."""
        story_id = story["id"]
        title = story["title"]
        resource = story.get("resource", "")

        # Generate code based on story type
        if "api" in resource.lower() or "api" in title.lower():
            return self._gen_api_code(story)
        elif "model" in resource.lower() or "ner" in title.lower():
            return self._gen_ml_code(story)
        elif "frontend" in resource.lower() or "dashboard" in title.lower():
            return self._gen_frontend_code(story)
        elif "audit" in resource.lower() or "security" in resource.lower():
            return self._gen_security_code(story)
        elif "docker" in resource.lower() or "Dockerfile" in resource:
            return self._gen_dockerfile(story)
        else:
            return self._gen_generic_code(story)

    def _gen_api_code(self, story: Dict) -> str:
        return '''"""Patient Search API — HIPAA-compliant endpoint."""
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

audit_logger = logging.getLogger("audit")
app = FastAPI(title="Clinical AI Platform")


class Patient(BaseModel):
    id: str
    name: str
    mrn: str
    dob: str


# Synthetic data (no real PHI)
PATIENTS = [
    Patient(id="1", name="Jane Doe", mrn="MRN-001", dob="1985-03-15"),
    Patient(id="2", name="John Smith", mrn="MRN-002", dob="1972-08-22"),
    Patient(id="3", name="Alice Johnson", mrn="MRN-003", dob="1990-11-03"),
]


@app.get("/api/v1/patients", response_model=List[Patient])
async def search_patients(
    name: Optional[str] = Query(None),
    mrn: Optional[str] = Query(None),
):
    """Search patients by name or MRN. All access is audit-logged."""
    audit_logger.info(f"PHI_ACCESS search name={name} mrn={mrn}")

    results = PATIENTS
    if name:
        results = [p for p in results if name.lower() in p.name.lower()]
    if mrn:
        results = [p for p in results if mrn == p.mrn]

    if not results:
        raise HTTPException(status_code=404, detail="No patients found")

    return results
'''

    def _gen_ml_code(self, story: Dict) -> str:
        return '''"""Clinical NER Model Wrapper — entity extraction from clinical text."""
from typing import Dict, List, Any
import json
import logging

audit_logger = logging.getLogger("audit")


class ClinicalNER:
    """Extracts clinical entities from text using Bedrock Claude."""

    ENTITY_TYPES = ["diagnosis", "medication", "procedure", "lab_result"]

    def __init__(self):
        self.model_id = "anthropic.claude-sonnet-4-20250514"

    def extract_entities(self, clinical_text: str) -> List[Dict[str, Any]]:
        """Extract clinical entities from text.

        Args:
            clinical_text: Raw clinical note text

        Returns:
            List of entities with type, text, and span
        """
        audit_logger.info("PHI_ACCESS clinical_ner extract_entities")

        # Simulated extraction (in production, calls Bedrock)
        entities = []
        keywords = {
            "diagnosis": ["hypertension", "diabetes", "pneumonia", "fracture"],
            "medication": ["metformin", "lisinopril", "amoxicillin", "ibuprofen"],
            "procedure": ["x-ray", "mri", "ct scan", "blood test"],
        }

        text_lower = clinical_text.lower()
        for entity_type, terms in keywords.items():
            for term in terms:
                start = text_lower.find(term)
                if start >= 0:
                    entities.append({
                        "type": entity_type,
                        "text": term,
                        "start": start,
                        "end": start + len(term),
                        "confidence": 0.95,
                    })

        return entities

    def filter_phi(self, text: str) -> str:
        """Remove PHI from text before processing."""
        import re
        # Mask SSN patterns
        text = re.sub(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "[SSN-REDACTED]", text)
        # Mask phone numbers
        text = re.sub(r"\\b\\d{3}[-.\\s]\\d{3}[-.\\s]\\d{4}\\b", "[PHONE-REDACTED]", text)
        return text
'''

    def _gen_frontend_code(self, story: Dict) -> str:
        return '''"""Frontend component stubs for Clinical Dashboard (Python test helpers)."""


def render_patient_list(patients: list) -> str:
    """Render patient list as HTML table for testing."""
    if not patients:
        return "<div class=\\"empty\\">No patients found</div>"

    rows = []
    for p in patients:
        rows.append(
            f"<tr><td>{p.get('name', '')}</td>"
            f"<td>{p.get('mrn', '')}</td>"
            f"<td>{p.get('dob', '')}</td></tr>"
        )
    return f"<table><thead><tr><th>Name</th><th>MRN</th><th>DOB</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_search_bar() -> str:
    """Render search bar component."""
    return '<input type="text" id="patient-search" placeholder="Search patients..." />'
'''

    def _gen_security_code(self, story: Dict) -> str:
        return '''"""HIPAA Audit Log — append-only PHI access logging."""
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
            f.write(json.dumps(entry) + "\\n")
        return entry

    def read_log(self) -> list:
        """Read all entries from today\\'s log."""
        if not self.log_file.exists():
            return []
        entries = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
'''

    def _gen_dockerfile(self, story: Dict) -> str:
        return '''# Multi-stage Docker build for Clinical AI Platform
# HIPAA-compliant: runs as non-root, minimal attack surface

FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

FROM python:3.11-slim
WORKDIR /app
RUN useradd -m -r appuser
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
'''

    def _gen_generic_code(self, story: Dict) -> str:
        return f'''"""Auto-generated for {story["title"]}."""


def main():
    """Placeholder implementation."""
    print("Feature: {story["title"]}")
    return True


if __name__ == "__main__":
    main()
'''

    def _generate_test(self, story: Dict[str, Any]) -> str:
        """Generate pytest test code for a story."""
        story_id = story["id"]
        title = story["title"]
        resource = story.get("resource", "")

        if "api" in resource.lower():
            return f'''"""Tests for {title}."""
import pytest


def test_{story_id.lower().replace("-", "_")}_success():
    """Test successful patient search."""
    # Simulated test
    assert True


def test_{story_id.lower().replace("-", "_")}_not_found():
    """Test patient not found."""
    assert True


def test_{story_id.lower().replace("-", "_")}_invalid_input():
    """Test invalid input handling."""
    assert True
'''
        elif "model" in resource.lower() or "ner" in title.lower():
            return f'''"""Tests for {title}."""
import pytest


def test_{story_id.lower().replace("-", "_")}_extraction():
    """Test entity extraction from clinical text."""
    assert True


def test_{story_id.lower().replace("-", "_")}_phi_filtering():
    """Test PHI is filtered before processing."""
    assert True
'''
        else:
            return f'''"""Tests for {title}."""
import pytest


def test_{story_id.lower().replace("-", "_")}_basic():
    """Test basic functionality."""
    assert True
'''

    def _run_tests(self, test_file: str) -> Dict[str, Any]:
        """Run pytest on generated tests."""
        if not test_file:
            return {"passed": True, "num_passed": 1, "num_failed": 0}

        test_path = self.workspace_dir / test_file
        if not test_path.exists():
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 1,
                "error": f"Test file not found: {test_file}",
            }

        try:
            proc = subprocess.run(
                ["python3", "-m", "pytest", str(test_path), "-v", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_dir),
            )
            # Parse pytest output
            output = proc.stdout + proc.stderr
            passed = proc.returncode == 0

            # Count passed/failed from output
            num_passed = output.count(" PASSED")
            num_failed = output.count(" FAILED")
            if num_passed == 0 and passed:
                num_passed = 1  # At least 1 if exit code 0

            return {
                "passed": passed,
                "num_passed": num_passed,
                "num_failed": num_failed,
                "error": output if not passed else "",
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 1,
                "error": "Test execution timed out (30s)",
            }
        except Exception as e:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 1,
                "error": str(e),
            }

    def _fix_code(self, code: str, error_msg: str, story_id: str) -> str:
        """Self-debug: attempt to fix code based on error message."""
        # Pattern-based fixes (simulates Claude Code's self-repair)
        if "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            # Add missing import
            if "fastapi" in error_msg:
                code = "from fastapi import FastAPI\n" + code
            elif "pydantic" in error_msg:
                code = "from pydantic import BaseModel\n" + code
        elif "SyntaxError" in error_msg:
            # Try to fix common syntax issues
            code = code.replace("def (", "def func(")
        elif "NameError" in error_msg:
            # Add missing variable initialization
            code = "# Auto-fixed: added missing references\n" + code
        elif "AssertionError" in error_msg or "assert" in error_msg.lower():
            # Fix test assertion
            pass  # Can't easily fix without context

        return code
