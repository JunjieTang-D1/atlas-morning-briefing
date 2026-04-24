"""Clinical NER Model Wrapper — entity extraction from clinical text."""
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
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]", text)
        # Mask phone numbers
        text = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE-REDACTED]", text)
        return text
