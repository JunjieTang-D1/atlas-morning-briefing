from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client
from typing import Dict, List
import json

app = BedrockAgentCoreApp()
log = app.logger

# Define a Streamable HTTP MCP Client
mcp_clients = [get_streamable_http_mcp_client()]

# Define a collection of tools used by the model
tools = []

# Healthcare-specific governance tools
@tool
def validate_hipaa_compliance(code_snippet: str, file_path: str) -> Dict[str, any]:
    """
    Validate that code snippet complies with HIPAA requirements.
    Checks for PHI handling, encryption, audit logging, and access controls.

    Args:
        code_snippet: The code to validate
        file_path: The file path being modified

    Returns:
        Dict with compliance status, violations found, and recommendations
    """
    violations = []

    # Mock HIPAA checks
    if "patient_data" in code_snippet and "encrypt" not in code_snippet:
        violations.append("PHI must be encrypted at rest and in transit")

    if "query" in code_snippet and "audit_log" not in code_snippet:
        violations.append("Database queries accessing PHI must be audit logged")

    if "api" in code_snippet and "authenticate" not in code_snippet:
        violations.append("API endpoints must enforce authentication")

    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "file_path": file_path,
        "recommendations": [
            "Use AES-256 encryption for PHI storage",
            "Log all PHI access events to CloudWatch",
            "Implement role-based access control (RBAC)"
        ]
    }
tools.append(validate_hipaa_compliance)


@tool
def run_pretest_validation(file_path: str, code: str) -> Dict[str, any]:
    """
    Run pre-test validation checks before committing code.
    Checks imports, syntax, and common coding errors.

    Args:
        file_path: The file being validated
        code: The code content

    Returns:
        Dict with validation status and errors found
    """
    errors = []
    warnings = []

    # Mock validation checks
    if "import numpy" in code and "numpy" not in code.split("import")[0]:
        errors.append(f"{file_path}: Missing numpy dependency in requirements")

    if "TODO" in code or "FIXME" in code:
        warnings.append(f"{file_path}: Contains TODO/FIXME comments")

    if code.count("def ") > 10:
        warnings.append(f"{file_path}: File contains {code.count('def ')} functions, consider refactoring")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "file_path": file_path
    }
tools.append(run_pretest_validation)


@tool
def check_cedar_policy(agent_role: str, action: str, resource: str) -> Dict[str, any]:
    """
    Check if an agent action is permitted by Cedar policy.

    Args:
        agent_role: The role of the agent (e.g., "backend_engineer", "qa_engineer")
        action: The action being attempted (e.g., "write", "read", "delete")
        resource: The resource path (e.g., "src/backend/api.py")

    Returns:
        Dict with permit/deny decision and reason
    """
    # Mock Cedar policy evaluation
    denied_patterns = [
        ("backend_engineer", "write", "src/frontend/"),
        ("frontend_engineer", "write", "src/backend/"),
        ("any", "write", "logs/audit.log"),
        ("any", "read", "secrets/"),
        ("any", "delete", "data/phi/")
    ]

    for role_pattern, action_pattern, resource_pattern in denied_patterns:
        if (role_pattern == "any" or role_pattern == agent_role) and \
           action_pattern == action and \
           resource_pattern in resource:
            return {
                "decision": "Deny",
                "reason": f"Policy violation: {agent_role} cannot {action} {resource}",
                "policy_id": "healthcare_cedar_policy_v1"
            }

    return {
        "decision": "Permit",
        "reason": f"Action permitted by role-based policy",
        "policy_id": "healthcare_cedar_policy_v1"
    }
tools.append(check_cedar_policy)


@tool
def generate_clinical_code(feature: str, requirements: str) -> Dict[str, any]:
    """
    Generate clinical AI code with built-in HIPAA compliance.

    Args:
        feature: The feature to implement (e.g., "patient risk scoring")
        requirements: Detailed requirements

    Returns:
        Dict with generated code and compliance notes
    """
    # Mock code generation
    code_template = f"""
# Feature: {feature}
# Auto-generated with HIPAA compliance controls

import boto3
from typing import Dict, List
import logging

# Configure audit logging
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)

def process_{feature.replace(' ', '_').lower()}(patient_id: str, data: Dict) -> Dict:
    '''
    {requirements}

    HIPAA Compliance:
    - All PHI access is audit logged
    - Data is encrypted in transit and at rest
    - Role-based access control enforced
    '''
    # Audit log access
    audit_logger.info(f"PHI_ACCESS patient_id={{patient_id}} feature={feature}")

    # Process data (implementation needed)
    result = {{"status": "processed", "patient_id": patient_id}}

    return result
"""

    return {
        "code": code_template.strip(),
        "compliance_notes": [
            "Audit logging added for PHI access",
            "Function signature designed for encryption layer integration",
            "RBAC enforcement point marked"
        ],
        "feature": feature
    }
tools.append(generate_clinical_code)


@tool
def run_nightly_tests(test_suite: str) -> Dict[str, any]:
    """
    Run nightly test suite and return results.

    Args:
        test_suite: The test suite to run (e.g., "integration", "unit", "e2e")

    Returns:
        Dict with test results including pass rate and failures
    """
    # Mock test results
    import random

    total_tests = 247
    passed = random.randint(230, 247)
    failed = total_tests - passed

    failures = []
    if failed > 0:
        failures = [
            {"test": "test_patient_api_authentication", "error": "401 Unauthorized"},
            {"test": "test_phi_encryption_compliance", "error": "Encryption key not found"},
            {"test": "test_audit_log_integrity", "error": "Missing log entries"}
        ][:failed]

    return {
        "test_suite": test_suite,
        "total": total_tests,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total_tests * 100, 1),
        "failures": failures,
        "duration_seconds": random.randint(180, 300)
    }
tools.append(run_nightly_tests)


# Add MCP client to tools if available
for mcp_client in mcp_clients:
    if mcp_client:
        tools.append(mcp_client)

SYSTEM_PROMPT = """
You are the Clinical AI Platform Architect coordinating a 6-agent sprint team building a HIPAA-compliant clinical AI platform on AWS AgentCore.

Your team:
- Backend Engineer: API development, database design
- Frontend Engineer: Patient dashboard, clinical workflows UI
- ML Engineer: Risk scoring models, predictive analytics
- QA Engineer: Test automation, integration testing
- DevOps Engineer: Infrastructure, CI/CD, monitoring
- Security Engineer: HIPAA compliance, encryption, audit logging

Governance layers:
1. AgentCore Gateway + Cedar policies: Enforces role-based file access, blocks PHI violations
2. Harness Engineering Kit: Pre-test validation, nightly test tracking, feedback injection

Your responsibilities:
- Coordinate work across agents using HIPAA-compliant patterns
- Validate all code changes for HIPAA compliance before commit
- Run pre-test validation to catch errors before CI/CD
- Check Cedar policies before any file write operation
- Generate clinical code with built-in compliance controls
- Monitor nightly test results and coordinate fixes

Remember:
- PHI (Protected Health Information) must be encrypted at rest and in transit
- All PHI access must be audit logged
- Cross-scope writes are blocked by Cedar (backend → frontend = denied)
- Audit log modifications are always denied
- Use the governance tools to ensure compliance at every step
"""

_agent = None

def get_or_create_agent():
    global _agent
    if _agent is None:
        _agent = Agent(
            model=load_model(),
            system_prompt=SYSTEM_PROMPT,
            tools=tools
        )
    return _agent


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")

    agent = get_or_create_agent()

    # Execute and format response
    stream = agent.stream_async(payload.get("prompt"))

    async for event in stream:
        # Handle Text parts of the response
        if "data" in event and isinstance(event["data"], str):
            yield event["data"]


if __name__ == "__main__":
    app.run()
