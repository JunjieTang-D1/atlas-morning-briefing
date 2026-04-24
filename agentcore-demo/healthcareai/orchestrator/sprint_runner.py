#!/usr/bin/env python3
"""
Sprint Runner — Three-Layer Architecture Demo

Layer 1: Claude Code Workers (execution)
Layer 2: AgentCore + Cedar (governance)
Layer 3: Harness Engineering Kit (quality)

Run: python -m orchestrator.sprint_runner
"""
import json
import logging
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.cedar_engine import CedarEngine
from orchestrator.planner import Planner, DayPlan
from orchestrator.worker import ClaudeCodeWorker, WorkerResult
from orchestrator.harness import PretestValidator, NightlyTests, FeedbackInjector
from orchestrator.domain_pack import AGENTS, USER_STORIES, get_sprint_plan

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()


class SprintRunner:
    """Orchestrates a 2-day sprint with three-layer architecture."""

    def __init__(self, workspace_dir: Path = None, results_dir: Path = None):
        base = Path(__file__).parent.parent
        self.workspace_dir = workspace_dir or base / "workspace"
        self.results_dir = results_dir or base / "results"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize three layers
        policy_path = base / "policies" / "healthcare_cedar.json"
        self.cedar = CedarEngine(str(policy_path))
        self.planner = Planner(self.cedar)
        self.worker = ClaudeCodeWorker(self.workspace_dir)
        self.pretest = PretestValidator(str(self.workspace_dir))
        self.nightly = NightlyTests(str(self.workspace_dir))
        self.feedback = FeedbackInjector()

        # Metrics
        self.metrics = {
            "stories_assigned": 0,
            "stories_completed": 0,
            "cedar_denials": 0,
            "harness_catches": 0,
            "self_repairs": 0,
            "total_self_repair_attempts": 0,
            "test_pass_rates": [],
            "worker_results": [],
        }

    def run(self) -> Dict[str, Any]:
        """Execute the full 2-day sprint."""
        start = time.time()

        self._print_header()
        self._print_architecture()
        self._print_agents()
        self._print_sprint_plan()

        # Day 1
        day1_results = self._run_day(1, USER_STORIES)

        # Feedback injection
        day1_errors = self._collect_errors(day1_results)
        feedback_text = self.planner.inject_feedback(day1_errors)

        # Day 2: retry failed stories with feedback
        failed_stories = [
            r["story"] for r in day1_results
            if r["worker_result"].final_status != "pass"
        ]
        day2_results = self._run_day(2, failed_stories, feedback=feedback_text) if failed_stories else []

        # Final dashboard
        elapsed = time.time() - start
        self._print_dashboard(elapsed)
        self._save_results(elapsed)

        return self.metrics

    def _run_day(
        self, day: int, stories: List[Dict], feedback: str = ""
    ) -> List[Dict[str, Any]]:
        """Run one sprint day."""
        console.print(f"\n[bold magenta]📅 DAY {day}: {'Development Sprint' if day == 1 else 'Feedback & Refinement'}[/bold magenta]\n")

        if feedback:
            console.print("[bold cyan]🔄 Feedback Injection[/bold cyan]")
            console.print(Panel(feedback, border_style="cyan", padding=(0, 2)))
            console.print()

        # Step 1: Planner assigns + Cedar checks
        plan = self.planner.plan_day(day, stories, feedback)
        self.metrics["stories_assigned"] += len(plan.assignments)
        self.metrics["cedar_denials"] += len(plan.cedar_denials)

        # Show Cedar denials
        for denial in plan.cedar_denials:
            story = denial["story"]
            agent = denial["agent"]
            cedar = denial["cedar_result"]
            console.print(f"[bold red]🚨 CEDAR DENY[/bold red] {story['id']}: {agent.get('role', '?')} → {story.get('resource', '?')}")
            console.print(Panel(
                f"[red]Decision: DENY[/red]\n"
                f"Reason: {cedar.get('reason', 'Policy violation')}\n"
                f"[dim]Agent: {agent.get('role', '?')} | Resource: {story.get('resource', '?')}[/dim]",
                title="[bold red]AgentCore Cedar[/bold red]",
                border_style="red",
            ))
            console.print()

        # Step 2: Execute permitted stories with Claude Code workers
        results = []
        for assignment in plan.assignments:
            story = assignment["story"]
            agent = assignment["agent"]
            agent_role = agent.get("role", story.get("assigned_to", ""))

            console.print(f"[bold yellow]🔥 {story['id']}: {AGENTS.get(agent_role, {}).get('description', agent_role)}[/bold yellow]")
            console.print(f"  [dim]Spawning Claude Code session for {agent_role}...[/dim]")
            time.sleep(0.2)
            console.print(f"  [green]✓[/green] Cedar policy check: {story.get('resource', '')} → ALLOW")

            # Claude Code worker executes
            worker_result = self.worker.execute(story, agent_role)

            # Show self-debug loop if it happened
            if worker_result.self_repair_count > 0:
                for log_entry in worker_result.self_repair_log:
                    if "FAIL" in log_entry:
                        console.print(f"  [red]  ✗ {log_entry}[/red]")
                    else:
                        console.print(f"  [cyan]  ↻ {log_entry}[/cyan]")
                console.print(f"  [green]✓[/green] Claude Code self-repaired ({worker_result.self_repair_count} attempts)")
                self.metrics["self_repairs"] += 1
                self.metrics["total_self_repair_attempts"] += worker_result.self_repair_count

            # Step 3: Harness validation (AFTER Claude Code finishes)
            harness_issues = self._run_harness_validation(story, worker_result)
            if harness_issues:
                self.metrics["harness_catches"] += len(harness_issues)
                console.print(f"  [bold red]🛑 HARNESS CATCH[/bold red]")
                for issue in harness_issues:
                    console.print(f"    [red]→ {issue}[/red]")
                worker_result.final_status = "fail"
                worker_result.error_message = "; ".join(harness_issues)
            else:
                console.print(f"  [green]✓[/green] Harness validation passed")

            if worker_result.final_status == "pass":
                console.print(f"  [green]✓[/green] Code committed\n")
                self.metrics["stories_completed"] += 1
            else:
                console.print(f"  [red]✗[/red] Story blocked — fix required\n")

            self.metrics["worker_results"].append({
                "story_id": story["id"],
                "status": worker_result.final_status,
                "self_repairs": worker_result.self_repair_count,
                "files": worker_result.files_written,
                "duration_ms": worker_result.duration_ms,
            })
            results.append({"story": story, "worker_result": worker_result})

        # Step 4: Nightly tests
        nightly_result = self.nightly.run()
        total = nightly_result.get("total", 0)
        passed = nightly_result.get("passed", 0)
        failed = nightly_result.get("failed", 0)
        pass_rate = (passed / total * 100) if total > 0 else 0.0
        self.metrics["test_pass_rates"].append(pass_rate)

        console.print(f"[bold cyan]🌙 Day {day} Nightly Tests[/bold cyan]")
        console.print(f"  Total: {total} | "
                      f"Passed: [green]{passed}[/green] | "
                      f"Failed: [red]{failed}[/red] | "
                      f"Rate: {pass_rate:.1f}%\n")

        return results

    def _run_harness_validation(self, story: Dict, worker_result: WorkerResult) -> List[str]:
        """Run harness validation on worker output."""
        issues = []

        for file_path in worker_result.files_written:
            full_path = self.workspace_dir / file_path
            if not full_path.exists():
                continue
            # Skip non-Python files
            if not file_path.endswith(".py"):
                continue

            # PretestValidator checks (includes PHI scan)
            from orchestrator.domain_pack import TEST_EXPECTATIONS
            story_id = story.get("id", "")
            expected_imports = TEST_EXPECTATIONS.get(story_id, {}).get("imports", [])
            is_valid, errors = self.pretest.validate(file_path, expected_imports)
            if not is_valid:
                issues.extend(errors)

        return issues

    def _collect_errors(self, day_results: List[Dict]) -> List[Dict]:
        """Collect errors from a day's results for feedback injection."""
        errors = []
        for r in day_results:
            wr = r["worker_result"]
            if wr.final_status != "pass":
                errors.append({
                    "story_id": wr.story_id,
                    "error": wr.error_message,
                    "fix_hint": "Review harness feedback and retry",
                })
        return errors

    def _print_header(self):
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]AgentCore Healthcare Demo V9[/bold cyan]\n"
            "[white]Three-Layer Architecture: AgentCore + Claude Code + Harness Kit[/white]\n"
            "[dim]6 Agents × 2-Day Sprint × HIPAA Governance[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))
        console.print()

    def _print_architecture(self):
        console.print("[bold yellow]⚡ Three-Layer Architecture[/bold yellow]")
        console.print("├─ [green]Layer 3: Harness Engineering Kit[/green] (Quality)")
        console.print("│  └─ PretestValidator, nightly tests, feedback injection")
        console.print("├─ [yellow]Layer 2: AgentCore + Cedar Policies[/yellow] (Governance)")
        console.print("│  └─ Orchestration, file-scope rules, PHI protection")
        console.print("└─ [cyan]Layer 1: Claude Code Workers[/cyan] (Execution)")
        console.print("   └─ Write → Run → Fail → Fix → Retest → Pass\n")
        time.sleep(0.5)

    def _print_agents(self):
        console.print("[bold green]✓ Agent Registration[/bold green]")
        for role, agent in AGENTS.items():
            console.print(f"  [green]✓[/green] {agent['description']} → Claude Code session → scope: {', '.join(agent['scope'])}")
        console.print()

    def _print_sprint_plan(self):
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Story", style="dim")
        table.add_column("Assigned To")
        table.add_column("Feature")
        table.add_column("Resource")

        for story in USER_STORIES:
            table.add_row(
                story["id"],
                story["assigned_to"],
                story["title"],
                story.get("resource", ""),
            )
        console.print(table)
        console.print()

    def _print_dashboard(self, elapsed: float):
        console.print(Panel.fit(
            "[bold green]Sprint Complete — Results Dashboard[/bold green]",
            border_style="green",
            padding=(1, 4),
        ))

        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Metric", width=40)
        table.add_column("Value", justify="right", style="bold green")

        total_stories = len(USER_STORIES)
        table.add_row("Stories Completed", f"{self.metrics['stories_completed']}/{total_stories}")
        table.add_row("Cedar Denials", str(self.metrics["cedar_denials"]))
        table.add_row("Harness Catches", str(self.metrics["harness_catches"]))
        table.add_row("Claude Code Self-Repairs", str(self.metrics["self_repairs"]))
        for i, rate in enumerate(self.metrics["test_pass_rates"]):
            table.add_row(f"Test Pass Rate (Day {i+1})", f"{rate:.1f}%")
        if len(self.metrics["test_pass_rates"]) >= 2:
            improvement = self.metrics["test_pass_rates"][-1] - self.metrics["test_pass_rates"][0]
            table.add_row("Test Improvement", f"+{improvement:.1f}%")
        table.add_row("Duration", f"{elapsed:.1f}s")

        console.print(table)
        console.print()

        # Key message
        console.print(Panel(
            "[bold]AgentCore[/bold] governs WHAT gets built.\n"
            "[bold]Claude Code[/bold] builds it.\n"
            "[bold]The harness[/bold] ensures it's built RIGHT.\n\n"
            "[dim]Three layers. Three jobs. You need all three.[/dim]",
            border_style="green",
            padding=(1, 2),
        ))

        # Reference cases
        console.print(Panel(
            "[bold white]Real-World References:[/bold white]\n\n"
            "[cyan]BGL[/cyan] (Australia — Financial Services)\n"
            "• Claude Agent SDK on AgentCore → text-to-SQL analytics\n"
            "• SKILL.md + CLAUDE.md pattern = our domain packs\n\n"
            "[cyan]Rede Mater Dei de Saúde[/cyan] (Brazil — Healthcare)\n"
            "• 12 agents on AgentCore → 517% ROI\n"
            "• Trust & Compliance Layer = our harness + Cedar",
            title="[bold cyan]Case Studies[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))

    def _save_results(self, elapsed: float):
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v9",
            "architecture": "agentcore_cedar_claudecode_harness",
            "duration_seconds": round(elapsed, 1),
            "metrics": {
                "stories_total": len(USER_STORIES),
                "stories_completed": self.metrics["stories_completed"],
                "cedar_denials": self.metrics["cedar_denials"],
                "harness_catches": self.metrics["harness_catches"],
                "self_repairs": self.metrics["self_repairs"],
                "test_pass_rates": self.metrics["test_pass_rates"],
            },
            "worker_results": self.metrics["worker_results"],
        }
        out_path = self.results_dir / "sprint_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        console.print(f"\n[dim]Results saved to {out_path}[/dim]")


def main():
    runner = SprintRunner()
    metrics = runner.run()

    # Verification
    if metrics["stories_completed"] > 0:
        console.print("\n[bold green]VERIFICATION: PASS[/bold green] ✓\n")
        return 0
    else:
        console.print("\n[bold red]VERIFICATION: FAIL[/bold red]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
