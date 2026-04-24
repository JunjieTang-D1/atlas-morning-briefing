"""Cedar Policy Engine

Loads Cedar-style policies and evaluates authorization decisions.
"""
import json
import fnmatch
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple
from datetime import datetime, timezone


Decision = Literal["Permit", "Deny"]


class CedarEngine:
    """Cedar-style policy engine for agent authorization."""

    def __init__(self, policy_path: str):
        """Load policies from JSON file."""
        self.policy_path = Path(policy_path)
        with open(self.policy_path, "r") as f:
            policies = json.load(f)

        self.allow_rules = policies.get("allow_rules", [])
        self.deny_rules = policies.get("deny_rules", [])
        self.audit_log: List[Dict] = []

    def check(
        self,
        principal: str,
        action: str,
        resource: str,
        context: Dict | None = None,
    ) -> Tuple[Decision, str]:
        """
        Evaluate authorization decision.

        Returns:
            (Decision, reason)
        """
        context = context or {}

        # Cedar semantics: deny rules take precedence
        for rule in self.deny_rules:
            if self._matches_deny_rule(principal, action, resource, rule, context):
                reason = f"DENY-{rule['id']}: {rule['description']}"
                self._log_decision(principal, action, resource, "Deny", reason)
                return ("Deny", reason)

        # Check allow rules
        for rule in self.allow_rules:
            if self._matches_allow_rule(principal, action, resource, rule):
                reason = f"ALLOW-{rule['id']}: {rule['description']}"
                self._log_decision(principal, action, resource, "Permit", reason)
                return ("Permit", reason)

        # Default deny
        reason = "No matching allow rule"
        self._log_decision(principal, action, resource, "Deny", reason)
        return ("Deny", reason)

    def _matches_allow_rule(
        self, principal: str, action: str, resource: str, rule: Dict
    ) -> bool:
        """Check if request matches an allow rule."""
        # Check principal
        if rule["principal"] != "*" and rule["principal"] != principal:
            return False

        # Check action
        rule_actions = rule["action"].split(",")
        if action not in rule_actions:
            return False

        # Check resource (glob pattern)
        rule_resources = rule["resource"].split(",")
        for pattern in rule_resources:
            if self._match_pattern(resource, pattern.strip()):
                return True

        return False

    def _matches_deny_rule(
        self,
        principal: str,
        action: str,
        resource: str,
        rule: Dict,
        context: Dict,
    ) -> bool:
        """Check if request matches a deny rule."""
        # Check principal (supports negation with !)
        rule_principal = rule["principal"]
        if rule_principal.startswith("!"):
            # Negative match: deny everyone EXCEPT this principal
            allowed_principal = rule_principal[1:]
            if principal == allowed_principal:
                return False
        elif rule_principal != "*" and rule_principal != principal:
            return False

        # Check action
        rule_actions = rule.get("action", "").split(",")
        if action and not any(action == a.strip() for a in rule_actions):
            # If actions specified and don't match, skip
            if rule_actions != [""]:
                return False

        # Check resource pattern (if specified)
        rule_resource = rule.get("resource", "")
        if rule_resource and not self._match_pattern(resource, rule_resource):
            return False

        # Check conditions (simplified evaluation)
        condition = rule.get("condition")
        if condition:
            # We only handle simple pattern matching for now
            if "content.matches_pattern" in condition:
                # Extract pattern from condition
                match = re.search(r"matches_pattern\('(.+?)'\)", condition)
                if match:
                    pattern = match.group(1)
                    content = context.get("content", "")
                    if not re.search(pattern, content):
                        return False

        return True

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match a file path against a glob pattern."""
        # Handle ** wildcards
        pattern = pattern.replace("**", "*")
        return fnmatch.fnmatch(path, pattern)

    def _log_decision(
        self, principal: str, action: str, resource: str, decision: Decision, reason: str
    ) -> None:
        """Log authorization decision to audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "principal": principal,
            "action": action,
            "resource": resource,
            "decision": decision,
            "reason": reason,
        }
        self.audit_log.append(entry)

    def get_audit_log(self) -> List[Dict]:
        """Return audit log entries."""
        return self.audit_log

    def save_audit_log(self, output_path: str) -> None:
        """Save audit log to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.audit_log, f, indent=2)
