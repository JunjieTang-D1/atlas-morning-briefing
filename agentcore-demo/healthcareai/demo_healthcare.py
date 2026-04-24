#!/usr/bin/env python3
"""
AgentCore Healthcare Demo - HIPAA-Compliant Clinical AI Platform
Simulates a 6-agent sprint team building clinical AI with dual governance layers.

Reference: Rede Mater Dei de Saúde deployed 12 agents on AgentCore, 517% ROI

Run: python demo_healthcare.py
"""

import time
import random
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich import box

console = Console()


class HealthcareDemo:
    """Simulates a 2-day sprint with 6 AI agents building a clinical AI platform."""

    def __init__(self):
        self.agents = [
            {"name": "Backend Engineer", "role": "backend_engineer", "scope": "src/backend/"},
            {"name": "Frontend Engineer", "role": "frontend_engineer", "scope": "src/frontend/"},
            {"name": "ML Engineer", "role": "ml_engineer", "scope": "src/models/"},
            {"name": "QA Engineer", "role": "qa_engineer", "scope": "tests/"},
            {"name": "DevOps Engineer", "role": "devops_engineer", "scope": "infrastructure/"},
            {"name": "Security Engineer", "role": "security_engineer", "scope": "security/"},
        ]
        self.metrics = {
            "policy_violations_blocked": 0,
            "pretest_errors_caught": 0,
            "phi_exposures_prevented": 0,
            "test_pass_rate_day1": 0,
            "test_pass_rate_day2": 0,
            "stories_completed": 0,
        }

    def print_header(self):
        """Print demo header."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]AgentCore Healthcare Demo[/bold cyan]\n"
            "[white]HIPAA-Compliant Clinical AI Platform[/white]\n"
            "[dim]6 AI Agents × 2-Day Sprint × Dual Governance[/dim]",
            border_style="cyan",
            padding=(1, 4)
        ))
        console.print("\n")

    def print_governance_layers(self):
        """Display governance architecture."""
        console.print("[bold yellow]⚡ Three-Layer Architecture[/bold yellow]")
        console.print("├─ [green]Layer 3: Harness Engineering Kit[/green] (Quality)")
        console.print("│  └─ PretestValidator, nightly tests, feedback injection")
        console.print("├─ [yellow]Layer 2: AgentCore Managed Harness + Cedar[/yellow] (Governance)")
        console.print("│  └─ Orchestration, Cedar policies, Firecracker isolation")
        console.print("└─ [cyan]Layer 1: Claude Code Workers[/cyan] (Execution)")
        console.print("   └─ Write → Run → Fail → Fix → Retest → Pass\n")
        time.sleep(1.5)

    def agent_register(self):
        """Simulate agent registration."""
        console.print("[bold green]✓ Agent Registration[/bold green]")
        console.print("  [dim]AgentCore Managed Harness → Planner + Judge (Opus 4.6)[/dim]")
        console.print("  [dim]Claude Code Sessions → 4 Workers (Sonnet 4.6)[/dim]\n")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            for agent in self.agents:
                task = progress.add_task(
                    f"Registering {agent['name']}...",
                    total=None
                )
                time.sleep(0.3)
                progress.stop_task(task)
                engine = "AgentCore" if agent['name'] in ["Planner", "Judge"] else "Claude Code"
                console.print(f"  [green]✓[/green] {agent['name']} → {engine} session → scope: {agent['scope']}")

        console.print()
        time.sleep(0.5)

    def planner_assigns_stories(self):
        """Simulate sprint planning."""
        console.print("[bold green]✓ Sprint Planning[/bold green]")
        stories = [
            ("US-101", "Backend Engineer", "Patient Risk Scoring API"),
            ("US-102", "Frontend Engineer", "Clinical Dashboard UI"),
            ("US-103", "ML Engineer", "Readmission Prediction Model"),
            ("US-104", "QA Engineer", "Integration Test Suite"),
            ("US-105", "DevOps Engineer", "CI/CD Pipeline with HIPAA Controls"),
            ("US-106", "Security Engineer", "Audit Logging Infrastructure"),
        ]

        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Story ID", style="dim")
        table.add_column("Assigned To")
        table.add_column("Feature")

        for story_id, agent, feature in stories:
            table.add_row(story_id, agent, feature)

        console.print(table)
        console.print()
        time.sleep(1.0)

    def day1_development(self):
        """Simulate Day 1 development with Cedar block and PretestValidator catch."""
        console.print("[bold magenta]📅 DAY 1: Development Sprint[/bold magenta]\n")
        time.sleep(0.5)

        # Story 1: Backend Engineer - Success with Claude Code self-debug
        console.print("[bold yellow]🔥 Story US-101: Backend Engineer[/bold yellow]")
        console.print("  [dim]Spawning Claude Code session...[/dim]")
        time.sleep(0.3)
        console.print("  [green]✓[/green] Cedar policy check: src/backend/ → ALLOW")
        console.print("  [cyan]  Claude Code:[/cyan] Writing src/backend/risk_api.py...")
        time.sleep(0.3)
        console.print("  [cyan]  Claude Code:[/cyan] Running pytest tests/test_risk_api.py...")
        time.sleep(0.3)
        console.print("  [cyan]  Claude Code:[/cyan] [green]12/12 tests passing[/green]")
        console.print("  [green]✓[/green] Harness validation passed")
        console.print("  [green]✓[/green] Code committed\n")
        time.sleep(0.5)

        # Story 2: Frontend Engineer - Cedar Policy BLOCK (dramatic moment!)
        console.print("[bold yellow]🔥 Story US-102: Frontend Engineer → Clinical Dashboard[/bold yellow]")
        console.print("  [dim]Writing code...[/dim]")
        time.sleep(0.4)

        console.print("\n[bold red]🚨 CEDAR POLICY VIOLATION[/bold red]")
        console.print(Panel(
            "[red]Decision: DENY[/red]\n\n"
            "[white]Reason:[/white] frontend_engineer attempted to write to [yellow]src/backend/auth.py[/yellow]\n\n"
            "[dim]Policy: Cross-scope writes are prohibited\n"
            "Agent role: frontend_engineer\n"
            "Target scope: src/backend/ (outside permitted scope)[/dim]",
            title="[bold red]AgentCore Gateway[/bold red]",
            border_style="red",
            padding=(1, 2)
        ))
        self.metrics["policy_violations_blocked"] += 1
        self.metrics["phi_exposures_prevented"] += 1
        console.print("  [red]✗[/red] Story blocked by governance layer\n")
        time.sleep(2.0)

        # Story 3: ML Engineer - Claude Code self-debug loop (dramatic!)
        console.print("[bold yellow]🔥 Story US-103: ML Engineer → Readmission Prediction[/bold yellow]")
        console.print("  [dim]Spawning Claude Code session...[/dim]")
        time.sleep(0.3)
        console.print("  [green]✓[/green] Cedar policy check: src/models/ → ALLOW")
        console.print("  [cyan]  Claude Code:[/cyan] Writing src/models/readmission_model.py...")
        time.sleep(0.3)
        console.print("  [cyan]  Claude Code:[/cyan] Running pytest...")
        time.sleep(0.3)
        console.print("  [red]  Claude Code:[/red] ✗ ImportError: No module named 'sklearn'")
        console.print("  [cyan]  Claude Code:[/cyan] [dim]Self-debugging... adding scikit-learn to deps[/dim]")
        time.sleep(0.4)
        console.print("  [cyan]  Claude Code:[/cyan] Running pytest... [green]PASS[/green]")
        console.print("  [green]✓[/green] Claude Code self-repaired\n")
        time.sleep(0.3)
        console.print("  [dim]Running harness validation...[/dim]")
        time.sleep(0.5)

        console.print("\n[bold red]🛑 HARNESS CATCHES WHAT CLAUDE CODE MISSED[/bold red]")
        console.print(Panel(
            "[red]Errors Found:[/red]\n\n"
            "1. [yellow]src/models/readmission_model.py[/yellow]\n"
            "   → PHI field 'patient_ssn' accessed without encryption\n\n"
            "[dim]Claude Code fixed the import error itself.[/dim]\n"
            "[dim]But the harness caught the HIPAA violation that Claude Code didn't see.[/dim]\n\n"
            "[bold]This is why you need BOTH layers.[/bold]",
            title="[bold red]PretestValidator (Harness Layer)[/bold red]",
            border_style="red",
            padding=(1, 2)
        ))
        self.metrics["pretest_errors_caught"] += 1
        console.print("  [red]✗[/red] Story blocked by harness — PHI fix required\n")
        time.sleep(2.0)

        # Stories 4-6: Remaining work proceeds
        self._execute_story("US-104", "QA Engineer", "tests/integration/test_api.py", success=True)
        self._execute_story("US-105", "DevOps Engineer", "infrastructure/pipeline.yaml", success=True)
        self._execute_story("US-106", "Security Engineer", "security/audit_logger.py", success=True)

        self.metrics["stories_completed"] = 3

        # Day 1 nightly tests
        self._run_nightly_tests(day=1, pass_rate=87.9)

    def day2_development(self):
        """Simulate Day 2 with feedback injection and improvements."""
        console.print("\n[bold magenta]📅 DAY 2: Feedback & Refinement[/bold magenta]\n")
        time.sleep(0.5)

        # Feedback injection
        console.print("[bold cyan]🔄 Feedback Injection (Harness Engineering Kit)[/bold cyan]")
        console.print(Panel(
            "[yellow]Feedback from Day 1:[/yellow]\n\n"
            "1. Frontend Engineer: Scope violation → corrected to src/frontend/auth_handler.py\n"
            "2. ML Engineer: Missing dependency + PHI encryption issue → dependencies updated, encryption added\n\n"
            "[green]Action:[/green] Agents retrying with corrections applied",
            border_style="cyan",
            padding=(1, 2)
        ))
        time.sleep(1.5)

        # Retry Story 2 - Success
        console.print("\n[bold yellow]🔄 Retry US-102: Frontend Engineer → Clinical Dashboard[/bold yellow]")
        console.print("  [dim]Writing to src/frontend/dashboard.tsx...[/dim]")
        time.sleep(0.4)
        console.print("  [green]✓[/green] Cedar policy check passed")
        console.print("  [green]✓[/green] Code committed\n")
        self.metrics["stories_completed"] += 1
        time.sleep(0.5)

        # Retry Story 3 - Success
        console.print("[bold yellow]🔄 Retry US-103: ML Engineer → Readmission Prediction[/bold yellow]")
        console.print("  [dim]Updated dependencies in requirements.txt...[/dim]")
        time.sleep(0.3)
        console.print("  [dim]Added PHI encryption layer...[/dim]")
        time.sleep(0.3)
        console.print("  [dim]Running pre-test validation...[/dim]")
        time.sleep(0.4)
        console.print("  [green]✓[/green] Pre-test validation passed")
        console.print("  [green]✓[/green] HIPAA compliance verified")
        console.print("  [green]✓[/green] Code committed\n")
        self.metrics["stories_completed"] += 1
        time.sleep(0.5)

        # Day 2 nightly tests show improvement
        self._run_nightly_tests(day=2, pass_rate=96.4)

    def _execute_story(self, story_id, agent_name, file_path, success=True):
        """Execute a single story."""
        console.print(f"[bold yellow]🔥 Story {story_id}: {agent_name}[/bold yellow]")
        console.print(f"  [dim]Writing to {file_path}...[/dim]")
        time.sleep(0.4)

        if success:
            console.print("  [green]✓[/green] Cedar policy check passed")
            console.print("  [green]✓[/green] Pre-test validation passed")
            console.print("  [green]✓[/green] Code committed\n")
        else:
            console.print("  [red]✗[/red] Validation failed\n")

        time.sleep(0.3)

    def _run_nightly_tests(self, day, pass_rate):
        """Run nightly tests and display results."""
        console.print(f"[bold cyan]🌙 Day {day} Nightly Tests[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Running integration test suite...", total=None)
            time.sleep(1.0)

        total = 247
        passed = int(total * pass_rate / 100)
        failed = total - passed

        console.print(f"  Total tests: [white]{total}[/white]")
        console.print(f"  Passed: [green]{passed}[/green]")
        console.print(f"  Failed: [red]{failed}[/red]")
        console.print(f"  Pass rate: [{'green' if pass_rate > 90 else 'yellow'}]{pass_rate}%[/{'green' if pass_rate > 90 else 'yellow'}]\n")

        if day == 1:
            self.metrics["test_pass_rate_day1"] = pass_rate
        else:
            self.metrics["test_pass_rate_day2"] = pass_rate

        time.sleep(1.0)

    def print_final_dashboard(self):
        """Print final results dashboard."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]Sprint Complete - Results Dashboard[/bold green]",
            border_style="green",
            padding=(1, 4)
        ))
        console.print()

        # Metrics table
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Metric", style="white", width=40)
        table.add_column("Value", justify="right", style="bold green")

        table.add_row("Stories Completed", f"{self.metrics['stories_completed']}/6")
        table.add_row("Policy Violations Blocked", f"{self.metrics['policy_violations_blocked']}")
        table.add_row("Pre-test Errors Caught", f"{self.metrics['pretest_errors_caught']}")
        table.add_row("PHI Exposures Prevented", f"{self.metrics['phi_exposures_prevented']}")
        table.add_row("Test Pass Rate (Day 1)", f"{self.metrics['test_pass_rate_day1']}%")
        table.add_row("Test Pass Rate (Day 2)", f"{self.metrics['test_pass_rate_day2']}%")
        table.add_row(
            "Test Improvement",
            f"+{self.metrics['test_pass_rate_day2'] - self.metrics['test_pass_rate_day1']:.1f}%",
            style="bold green"
        )

        console.print(table)
        console.print()

        # Key achievements
        console.print("[bold cyan]🎯 Key Achievements[/bold cyan]")
        console.print("  [green]✓[/green] Zero HIPAA violations reached production")
        console.print("  [green]✓[/green] 100% of policy violations caught at gateway")
        console.print("  [green]✓[/green] Pre-test validation prevented CI/CD failures")
        console.print("  [green]✓[/green] Feedback loop improved test pass rate by 8.5 points")
        console.print("  [green]✓[/green] All 6 agents delivered features on schedule\n")

        # Reference case studies
        console.print(Panel(
            "[bold white]Real-World References:[/bold white]\n\n"
            "[cyan]Rede Mater Dei de Saúde[/cyan] (Brazil — Healthcare)\n"
            "• 12 agents on AgentCore → [green]517% ROI[/green] in 4 months\n"
            "• Trust & Compliance Layer = our harness + Cedar\n\n"
            "[cyan]BGL[/cyan] (Australia — Financial Services)\n"
            "• Claude Agent SDK on AgentCore → text-to-SQL analytics\n"
            "• SKILL.md + CLAUDE.md pattern = our domain packs\n"
            "• Firecracker microVM isolation for compliance",
            title="[bold cyan]Case Studies[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()

    def run(self):
        """Run the complete demo."""
        self.print_header()
        self.print_governance_layers()
        self.agent_register()
        self.planner_assigns_stories()
        self.day1_development()
        self.day2_development()
        self.print_final_dashboard()

        # Verification
        if self.metrics["stories_completed"] == 5 and self.metrics["test_pass_rate_day2"] > 90:
            console.print("[bold green]VERIFICATION: PASS[/bold green] ✓\n", style="bold green")
            return 0
        else:
            console.print(
                f"[bold red]VERIFICATION: FAIL[/bold red] - "
                f"Stories: {self.metrics['stories_completed']}/5, "
                f"Pass rate: {self.metrics['test_pass_rate_day2']}%\n",
                style="bold red"
            )
            return 1


if __name__ == "__main__":
    demo = HealthcareDemo()
    exit(demo.run())
