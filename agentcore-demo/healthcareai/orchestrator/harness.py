"""Harness Engineering Kit

Quality validation tools that run AFTER Claude Code worker finishes.
"""
import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone


class PretestValidator:
    """Pre-test validation: static analysis before running tests."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)

    def validate(self, file_path: str, expected_imports: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate a Python file before testing.

        Returns:
            (is_valid, errors)
        """
        errors = []
        full_path = self.workspace_dir / file_path

        if not full_path.exists():
            errors.append(f"File not found: {file_path}")
            return (False, errors)

        # Read file
        try:
            content = full_path.read_text()
        except Exception as e:
            errors.append(f"Cannot read file: {e}")
            return (False, errors)

        # Check imports
        try:
            tree = ast.parse(content)
            imports = self._extract_imports(tree)
            missing = [imp for imp in expected_imports if imp not in imports]
            if missing:
                errors.append(f"Missing imports: {', '.join(missing)}")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return (False, errors)

        # Check for hardcoded secrets
        if self._contains_secrets(content):
            errors.append("Hardcoded secrets detected (API_KEY, TOKEN, PASSWORD)")

        # Check for PHI in plain text (very basic check)
        if self._contains_phi_issues(content):
            errors.append("Potential PHI exposure: unencrypted PHI fields in logs")

        return (len(errors) == 0, errors)

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imported module names from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])
        return imports

    def _contains_secrets(self, content: str) -> bool:
        """Check for hardcoded secrets."""
        pattern = r'(?i)(api_key|secret|password|token)\s*=\s*["\'][^"\']{10,}'
        return bool(re.search(pattern, content))

    def _contains_phi_issues(self, content: str) -> bool:
        """Check for potential PHI exposure (very basic heuristic)."""
        # Look for logging PHI fields without masking
        phi_fields = ["patient_name", "mrn", "ssn", "dob"]
        for field in phi_fields:
            if re.search(rf'print\(.*{field}', content):
                return True
            if re.search(rf'logger\.\w+\(.*{field}.*\)', content):
                return True
        return False


class NightlyTests:
    """Nightly test runner: executes pytest on workspace."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)

    def run(self) -> Dict:
        """
        Run pytest on all test files in workspace.

        Returns:
            {
                "status": "pass" | "fail",
                "total": int,
                "passed": int,
                "failed": int,
                "output": str,
            }
        """
        test_dir = self.workspace_dir / "tests"
        if not test_dir.exists():
            return {
                "status": "fail",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "output": "No tests directory found",
            }

        # Run pytest
        try:
            result = subprocess.run(
                ["pytest", str(test_dir), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.workspace_dir,
            )

            # Parse output
            output = result.stdout + result.stderr
            passed, failed, total = self._parse_pytest_output(output)

            return {
                "status": "pass" if result.returncode == 0 else "fail",
                "total": total,
                "passed": passed,
                "failed": failed,
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "fail",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "output": "Tests timed out after 60s",
            }
        except Exception as e:
            return {
                "status": "fail",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "output": f"Test execution error: {e}",
            }

    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int]:
        """Parse pytest output to extract test counts."""
        # Look for lines like: "===== 3 passed, 1 failed in 0.23s ====="
        match = re.search(r'(\d+) passed', output)
        passed = int(match.group(1)) if match else 0

        match = re.search(r'(\d+) failed', output)
        failed = int(match.group(1)) if match else 0

        total = passed + failed
        return (passed, failed, total)


class FeedbackInjector:
    """Feedback injector: converts Day N errors into Day N+1 instructions."""

    def __init__(self):
        self.feedback_log: List[Dict] = []

    def inject_feedback(self, day: int, test_results: Dict) -> List[str]:
        """
        Generate feedback instructions from test failures.

        Returns:
            List of feedback instructions for next day
        """
        instructions = []

        if test_results["status"] == "fail":
            output = test_results["output"]

            # Extract error messages
            errors = self._extract_errors(output)

            for error in errors:
                instruction = self._error_to_instruction(error)
                instructions.append(instruction)

                # Log feedback
                self.feedback_log.append({
                    "day": day,
                    "error": error,
                    "instruction": instruction,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return instructions

    def _extract_errors(self, output: str) -> List[str]:
        """Extract error messages from pytest output."""
        errors = []

        # Look for FAILED lines
        for line in output.split("\n"):
            if "FAILED" in line or "ERROR" in line:
                errors.append(line.strip())

        # Also look for assertion errors
        error_pattern = r'(AssertionError|AttributeError|ImportError|NameError|TypeError): (.+)'
        for match in re.finditer(error_pattern, output):
            errors.append(f"{match.group(1)}: {match.group(2)}")

        return errors[:5]  # Limit to 5 errors

    def _error_to_instruction(self, error: str) -> str:
        """Convert an error message to a repair instruction."""
        if "ImportError" in error or "ModuleNotFoundError" in error:
            return "Add missing import statements at the top of the file"

        if "AssertionError" in error:
            return "Fix assertion logic to match expected behavior"

        if "AttributeError" in error:
            return "Check method/attribute names for typos"

        if "TypeError" in error:
            return "Fix function signature or argument types"

        if "NameError" in error:
            return "Define missing variable or function"

        return f"Fix error: {error[:100]}"

    def get_feedback_log(self) -> List[Dict]:
        """Return all feedback entries."""
        return self.feedback_log


def validate_and_test(
    workspace_dir: str,
    file_path: str,
    expected_imports: List[str],
) -> Dict:
    """
    Run full harness validation on a file.

    Returns:
        {
            "pretest_valid": bool,
            "pretest_errors": List[str],
            "test_results": Dict,
        }
    """
    validator = PretestValidator(workspace_dir)
    is_valid, errors = validator.validate(file_path, expected_imports)

    result = {
        "pretest_valid": is_valid,
        "pretest_errors": errors,
        "test_results": None,
    }

    # Only run tests if pretest passed
    if is_valid:
        tester = NightlyTests(workspace_dir)
        result["test_results"] = tester.run()

    return result
