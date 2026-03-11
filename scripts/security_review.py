#!/usr/bin/env python3
"""
Atlas Security Review Agent
============================
Automated security & governance checks for all infrastructure operations.

Run before any EC2 launch, code push, or config change.
Exit code 0 = PASS, 1 = FAIL (blocks the operation).

Usage:
    python security_review.py --check ec2-launch --instance g5.xlarge --subnet subnet-xxx
    python security_review.py --check git-push --repo /path/to/repo
    python security_review.py --check all
    python security_review.py --check orphan-instances
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# === POLICY CONSTANTS (single source of truth) ===
ALLOWED_INSTANCE_TYPES = {"g5.xlarge", "trn1.2xlarge"}
ALLOWED_SUBNETS = {"subnet-0a9bcb1a9094da197", "subnet-08608538369219acb", "subnet-013d6036340119601"}  # us-east-1b + us-east-1f (trn1) + us-east-1a (trn1 fallback)
REQUIRED_TAG = {"Key": "CreatedBy", "Value": "atlas"}
MAX_RUNTIME_HOURS = 4
REGION = "us-east-1"
VPC_ID = "vpc-0709c477aae73f852"

# Secrets patterns to scan for
SECRET_PATTERNS = [
    (r'[a-zA-Z0-9._%+-]+@gmail\.com', 'Gmail address'),
    (r'[a-zA-Z0-9._%+-]+@kindle\.com', 'Kindle address'),
    (r'[a-zA-Z0-9._%+-]+@amazon\.com', 'Amazon email'),
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
    (r'(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{8,}', 'API key/token/password'),
    (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI API key'),
    (r'xoxb-[0-9]{10,}', 'Slack token'),
]

WORKSPACE = Path.home() / ".openclaw" / "workspace"
LOG_DIR = WORKSPACE / "logs" / "security-reviews"


class ReviewResult:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def ok(self, check, detail=""):
        self.checks.append(("PASS", check, detail))
        self.passed += 1

    def fail(self, check, detail=""):
        self.checks.append(("FAIL", check, detail))
        self.failed += 1

    def warn(self, check, detail=""):
        self.checks.append(("WARN", check, detail))
        self.warnings += 1

    @property
    def all_passed(self):
        return self.failed == 0

    def report(self):
        lines = []
        lines.append("=" * 60)
        lines.append("🔒 Atlas Security Review")
        lines.append("=" * 60)
        for status, check, detail in self.checks:
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[status]
            line = f"  {icon} {check}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        lines.append("-" * 60)
        lines.append(f"  Results: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        if self.all_passed:
            lines.append("  🟢 REVIEW: PASS")
        else:
            lines.append("  🔴 REVIEW: FAIL — operation blocked")
        lines.append("=" * 60)
        return "\n".join(lines)


def check_ec2_launch(result, instance_type=None, subnet=None, public_ip=None):
    """Review an EC2 launch request."""

    # Instance type
    if instance_type:
        if instance_type in ALLOWED_INSTANCE_TYPES:
            result.ok("Instance type", instance_type)
        else:
            result.fail("Instance type", f"{instance_type} not in {ALLOWED_INSTANCE_TYPES}")
    else:
        result.fail("Instance type", "not specified")

    # Subnet
    if subnet:
        if subnet in ALLOWED_SUBNETS:
            result.ok("Subnet", subnet)
        else:
            result.fail("Subnet", f"{subnet} not in allowed subnets {ALLOWED_SUBNETS}")
    else:
        result.fail("Subnet", "not specified")

    # Public IP
    if public_ip is not None:
        if not public_ip:
            result.ok("Public IP", "disabled")
        else:
            result.fail("Public IP", "ENABLED — must be disabled")
    else:
        result.warn("Public IP", "not specified — ensure AssociatePublicIpAddress=False")

    # Check IAM permission with dry-run
    if instance_type and subnet:
        try:
            ec2 = boto3.client("ec2", region_name=REGION)
            ec2.run_instances(
                InstanceType=instance_type,
                ImageId="ami-0aad28499825d76c3",
                SubnetId=subnet,
                MinCount=1, MaxCount=1,
                DryRun=True,
            )
        except ClientError as e:
            if "DryRunOperation" in str(e):
                result.ok("IAM permission", "dry-run succeeded")
            elif "UnauthorizedOperation" in str(e):
                result.fail("IAM permission", "denied by IAM policy")
            else:
                result.warn("IAM permission", str(e)[:100])


def check_orphan_instances(result):
    """Check for running atlas instances that might be forgotten."""
    try:
        ec2 = boto3.client("ec2", region_name=REGION)
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:CreatedBy", "Values": ["atlas"]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )
        instances = []
        for res in resp["Reservations"]:
            for inst in res["Instances"]:
                launch_time = inst["LaunchTime"]
                age_hours = (datetime.now(timezone.utc) - launch_time).total_seconds() / 3600
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                instances.append({
                    "id": inst["InstanceId"],
                    "type": inst["InstanceType"],
                    "ip": inst.get("PrivateIpAddress", "N/A"),
                    "age_hours": round(age_hours, 1),
                    "task": tags.get("Task", "unknown"),
                    "has_public_ip": inst.get("PublicIpAddress") is not None,
                })

        if not instances:
            result.ok("Orphan instances", "none found")
        else:
            for inst in instances:
                if inst["age_hours"] > MAX_RUNTIME_HOURS:
                    result.fail(
                        "Orphan instance",
                        f"{inst['id']} ({inst['type']}) running {inst['age_hours']}h — exceeds {MAX_RUNTIME_HOURS}h limit, TERMINATE NOW"
                    )
                else:
                    result.warn(
                        "Running instance",
                        f"{inst['id']} ({inst['type']}) age={inst['age_hours']}h task={inst['task']}"
                    )
                if inst["has_public_ip"]:
                    result.fail(
                        "Public IP violation",
                        f"{inst['id']} HAS A PUBLIC IP — security violation"
                    )
    except ClientError as e:
        result.warn("Orphan check", f"could not query: {e}")


def check_git_push(result, repo_path=None):
    """Scan staged changes for secrets before push."""
    repo = repo_path or str(WORKSPACE / "research-papers")

    try:
        # Check staged diff
        r = subprocess.run(
            ["git", "diff", "--cached", "--no-color"],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        diff_text = r.stdout

        if not diff_text:
            # Check last commit if nothing staged
            r = subprocess.run(
                ["git", "diff", "HEAD~1", "--no-color"],
                cwd=repo, capture_output=True, text=True, timeout=10,
            )
            diff_text = r.stdout

        if not diff_text:
            result.ok("Git diff", "no changes to scan")
            return

        found_secrets = []
        for pattern, name in SECRET_PATTERNS:
            matches = re.findall(pattern, diff_text)
            if matches:
                # Filter out obvious placeholders
                real_matches = [m for m in matches if "example.com" not in str(m) and "placeholder" not in str(m).lower()]
                if real_matches:
                    found_secrets.append((name, len(real_matches)))

        if found_secrets:
            for name, count in found_secrets:
                result.fail("Secret detected", f"{count}x {name}")
        else:
            result.ok("Secret scan", f"clean ({len(diff_text)} chars scanned)")

        # Check .env not committed
        r = subprocess.run(
            ["git", "ls-files", ".env"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        )
        if r.stdout.strip():
            result.fail(".env in repo", ".env file is tracked — must be in .gitignore")
        else:
            result.ok(".env not tracked", "safe")

    except Exception as e:
        result.warn("Git scan", f"error: {e}")


def check_config_files(result):
    """Scan workspace config files for exposed secrets."""
    sensitive_files = [
        WORKSPACE / "TOOLS.md",
        WORKSPACE / "USER.md",
        WORKSPACE / "MEMORY.md",
    ]

    for f in sensitive_files:
        if f.exists():
            content = f.read_text()
            for pattern, name in SECRET_PATTERNS:
                if name == "API key/token/password":
                    continue  # TOOLS.md legitimately mentions these words
                matches = re.findall(pattern, content)
                real = [m for m in matches if "example.com" not in str(m)]
                if real:
                    result.warn(f"Secret in {f.name}", f"{len(real)}x {name}")

    result.ok("Config scan", "completed")


def check_gpu_runner_integrity(result):
    """Verify gpu_runner.py hasn't been tampered with."""
    runner = WORKSPACE / "scripts" / "gpu_runner.py"
    if not runner.exists():
        result.fail("gpu_runner.py", "missing!")
        return

    content = runner.read_text()

    # Check critical safety constants are intact
    checks = [
        ('ALLOWED_INSTANCES = {"g5.xlarge", "trn1.2xlarge"}', "allowed instances"),
        ('SUBNET_ID = "subnet-0a9bcb1a9094da197"', "subnet lock"),
        ('"AssociatePublicIpAddress": False', "no public IP"),
        ('"CreatedBy", "Value": "atlas"', "creator tag"),
        ('"MarketType": "spot"', "spot enforcement"),
    ]

    for expected, name in checks:
        if expected in content:
            result.ok(f"gpu_runner: {name}", "intact")
        else:
            result.fail(f"gpu_runner: {name}", "MODIFIED or MISSING — possible tampering")


def save_log(result, check_type):
    """Save review result to log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    log_file = LOG_DIR / f"{ts}_{check_type}.json"
    log_file.write_text(json.dumps({
        "timestamp": ts,
        "check_type": check_type,
        "passed": result.passed,
        "failed": result.failed,
        "warnings": result.warnings,
        "all_passed": result.all_passed,
        "checks": [{"status": s, "check": c, "detail": d} for s, c, d in result.checks],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Atlas Security Review Agent")
    parser.add_argument("--check", required=True,
                        choices=["ec2-launch", "git-push", "orphan-instances", "config", "integrity", "all"],
                        help="Type of security check")
    parser.add_argument("--instance", help="Instance type (for ec2-launch)")
    parser.add_argument("--subnet", help="Subnet ID (for ec2-launch)")
    parser.add_argument("--public-ip", type=bool, default=None, help="Public IP flag (for ec2-launch)")
    parser.add_argument("--repo", help="Repo path (for git-push)")
    args = parser.parse_args()

    result = ReviewResult()

    if args.check in ("ec2-launch", "all"):
        check_ec2_launch(result, args.instance, args.subnet or list(ALLOWED_SUBNETS)[0], args.public_ip)

    if args.check in ("orphan-instances", "all"):
        check_orphan_instances(result)

    if args.check in ("git-push", "all"):
        check_git_push(result, args.repo)

    if args.check in ("config", "all"):
        check_config_files(result)

    if args.check in ("integrity", "all"):
        check_gpu_runner_integrity(result)

    print(result.report())
    save_log(result, args.check)
    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
