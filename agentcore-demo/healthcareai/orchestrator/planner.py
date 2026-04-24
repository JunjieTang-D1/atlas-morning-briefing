"""Sprint Planner — assigns stories to workers based on Cedar policy checks."""
import logging
from typing import Dict, Any, List
from pathlib import Path

from orchestrator.cedar_engine import CedarEngine
from orchestrator.domain_pack import AGENTS, USER_STORIES, get_sprint_plan

logger = logging.getLogger(__name__)


class DayPlan:
    """Plan for a single sprint day."""

    def __init__(self, day: int):
        self.day = day
        self.assignments: List[Dict[str, Any]] = []
        self.cedar_denials: List[Dict[str, Any]] = []

    def add_assignment(self, story: Dict, agent: Dict, cedar_result: Dict):
        self.assignments.append({
            "story": story,
            "agent": agent,
            "cedar_result": cedar_result,
        })

    def add_denial(self, story: Dict, agent: Dict, cedar_result: Dict):
        self.cedar_denials.append({
            "story": story,
            "agent": agent,
            "cedar_result": cedar_result,
        })


class Planner:
    """Assigns user stories to workers with Cedar policy validation."""

    def __init__(self, cedar_engine: CedarEngine):
        self.cedar = cedar_engine

    def plan_day(
        self,
        day: int,
        stories: List[Dict],
        feedback: str = "",
    ) -> DayPlan:
        """Create a day plan by assigning stories and checking Cedar policies."""
        plan = DayPlan(day=day)

        for story in stories:
            agent_role = story.get("assigned_to", "")
            agent = AGENTS.get(agent_role, {})
            resource = story.get("resource", "")

            # Cedar policy check
            decision, reason = self.cedar.check(
                principal=agent_role,
                action="write",
                resource=resource,
            )
            cedar_result = {"decision": decision, "reason": reason}

            if decision == "Permit":
                plan.add_assignment(story, agent, cedar_result)
                logger.info(
                    f"Day {day}: {agent_role} → {story['id']} ({resource}) — PERMIT"
                )
            else:
                plan.add_denial(story, agent, cedar_result)
                logger.warning(
                    f"Day {day}: {agent_role} → {story['id']} ({resource}) — DENY: {cedar_result['reason']}"
                )

        return plan

    def inject_feedback(self, day_errors: List[Dict]) -> str:
        """Generate feedback from Day N errors for Day N+1 planning."""
        if not day_errors:
            return ""

        lines = [f"Feedback from previous day ({len(day_errors)} issues):"]
        for err in day_errors:
            lines.append(f"  - {err.get('story_id', '?')}: {err.get('error', 'Unknown')}")
            if err.get("fix_hint"):
                lines.append(f"    Fix hint: {err['fix_hint']}")

        return "\n".join(lines)
