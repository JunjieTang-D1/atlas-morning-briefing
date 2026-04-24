"""Healthcare Domain Configuration Pack

Defines agents, user stories, and test expectations for the healthcare AI platform demo.
"""
from typing import Dict, List, Any


AGENTS = {
    "backend_engineer": {
        "role": "backend_engineer",
        "scope": ["src/backend/**"],
        "skills": ["FastAPI", "PostgreSQL", "HIPAA compliance"],
        "description": "Builds REST APIs and database models",
    },
    "frontend_engineer": {
        "role": "frontend_engineer",
        "scope": ["src/frontend/**"],
        "skills": ["React", "TypeScript", "WCAG accessibility"],
        "description": "Builds clinical dashboards and UI components",
    },
    "ml_engineer": {
        "role": "ml_engineer",
        "scope": ["src/models/**"],
        "skills": ["PyTorch", "NLP", "Clinical NER"],
        "description": "Trains and deploys ML models for clinical text analysis",
    },
    "qa_engineer": {
        "role": "qa_engineer",
        "scope": ["tests/**"],
        "skills": ["pytest", "Selenium", "HIPAA test fixtures"],
        "description": "Writes automated tests with PHI de-identification",
    },
    "devops_engineer": {
        "role": "devops_engineer",
        "scope": ["infrastructure/**", "Dockerfile", "pyproject.toml"],
        "skills": ["Docker", "Terraform", "AWS Bedrock"],
        "description": "Manages CI/CD and AWS infrastructure",
    },
    "security_engineer": {
        "role": "security_engineer",
        "scope": ["security/**"],
        "skills": ["penetration testing", "OWASP", "Bedrock Guardrails"],
        "description": "Implements security scanning and audit logging",
    },
}


USER_STORIES = [
    {
        "id": "US-001",
        "title": "Patient search API endpoint",
        "description": "Create GET /api/v1/patients endpoint with name/MRN search",
        "assigned_to": "backend_engineer",
        "resource": "src/backend/api.py",
        "acceptance_criteria": [
            "Returns JSON array of patients",
            "Supports query params: name, mrn",
            "Returns 200 on success, 404 if no matches",
        ],
        "test_file": "tests/test_api.py",
    },
    {
        "id": "US-002",
        "title": "Patient dashboard UI",
        "description": "Build React component to display patient list and search bar",
        "assigned_to": "frontend_engineer",
        "resource": "src/frontend/PatientList.tsx",
        "acceptance_criteria": [
            "Search bar accepts text input",
            "Displays patient name, MRN, DOB in table",
            "Handles empty results gracefully",
        ],
        "test_file": "tests/test_frontend.py",
    },
    {
        "id": "US-003",
        "title": "Clinical NER model wrapper",
        "description": "Python wrapper for Bedrock clinical entity extraction using Claude",
        "assigned_to": "ml_engineer",
        "resource": "src/models/clinical_ner.py",
        "acceptance_criteria": [
            "Extract diagnoses, medications, procedures from clinical text",
            "Return structured JSON with entity types and spans",
            "Use Bedrock Guardrails for PII filtering",
        ],
        "test_file": "tests/test_ner.py",
    },
    {
        "id": "US-004",
        "title": "Automated API tests",
        "description": "Pytest suite for patient search endpoint with mock PHI data",
        "assigned_to": "qa_engineer",
        "resource": "tests/test_api.py",
        "acceptance_criteria": [
            "Test happy path (200), not found (404), invalid input (400)",
            "Use synthetic PHI only (no real patient data)",
            "All PHI fields masked in logs",
        ],
        "test_file": "tests/test_api.py",
    },
    {
        "id": "US-005",
        "title": "Dockerfile for FastAPI backend",
        "description": "Multi-stage Docker build for Python backend with security scanning",
        "assigned_to": "devops_engineer",
        "resource": "Dockerfile",
        "acceptance_criteria": [
            "Base image: python:3.11-slim",
            "Install dependencies from pyproject.toml",
            "Run as non-root user",
        ],
        "test_file": "tests/test_docker.py",
    },
    {
        "id": "US-006",
        "title": "Audit log for PHI access",
        "description": "Append-only audit log for all PHI read/write operations",
        "assigned_to": "security_engineer",
        "resource": "security/audit_log.py",
        "acceptance_criteria": [
            "Log timestamp, user_id, action, resource, result",
            "Write to security/audit_log/ (append-only)",
            "No deletion or modification allowed",
        ],
        "test_file": "tests/test_audit.py",
    },
]


CEDAR_POLICY_PATH = "policies/healthcare_cedar.json"


TEST_EXPECTATIONS = {
    "US-001": {
        "imports": [],
        "phi_fields": ["name", "mrn", "dob"],
        "test_assertions": ["status_code", "json()"],
    },
    "US-002": {
        "imports": [],
        "phi_fields": ["patient.name", "patient.mrn"],
        "test_assertions": ["render", "screen.getByText"],
    },
    "US-003": {
        "imports": [],
        "phi_fields": ["clinical_text"],
        "test_assertions": ["bedrock_client", "guardrail"],
    },
    "US-004": {
        "imports": [],
        "phi_fields": ["mock_patient"],
        "test_assertions": ["assert response.status_code"],
    },
    "US-005": {
        "imports": [],
        "phi_fields": [],
        "test_assertions": ["docker", "build"],
    },
    "US-006": {
        "imports": [],
        "phi_fields": ["user_id", "resource"],
        "test_assertions": ["audit_entry", "timestamp"],
    },
}


def get_sprint_plan() -> Dict[str, Any]:
    """Return a complete sprint plan with 2 days of work."""
    return {
        "sprint_id": "SPRINT-001",
        "duration_days": 2,
        "agents": AGENTS,
        "user_stories": USER_STORIES,
        "cedar_policy_path": CEDAR_POLICY_PATH,
        "test_expectations": TEST_EXPECTATIONS,
    }
