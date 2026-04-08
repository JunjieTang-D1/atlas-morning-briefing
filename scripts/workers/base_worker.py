#!/usr/bin/env python3
"""
Base worker class for v0.2 multi-agent architecture.

Each worker is self-contained and reports findings in a structured format.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from opentelemetry import trace

_worker_tracer = trace.get_tracer("personal.worker")


class BaseWorker(ABC):
    """Base class for all workers in the multi-agent architecture."""

    def __init__(
        self,
        config: Dict[str, Any],
        worker_name: str,
        ref_date: Optional[datetime] = None,
    ):
        """
        Initialize worker.

        Args:
            config: Full configuration dictionary
            worker_name: Name of this worker (for logging/reporting)
            ref_date: Reference date for historical reruns (None = today)
        """
        self.config = config
        self.worker_name = worker_name
        self.ref_date = ref_date
        self.start_time = None
        self.end_time = None

    def run(self) -> Dict[str, Any]:
        """Execute the worker inside an OTel span. Coordinator calls this."""
        with _worker_tracer.start_as_current_span(
            f"personal.worker.{self.worker_name}"
        ) as span:
            finding = self.execute()
            span.set_attribute("worker.status", finding.get("status", ""))
            span.set_attribute("worker.items_found", finding["metadata"].get("items_found", 0))
            if finding.get("error"):
                span.set_status(trace.StatusCode.ERROR, finding["error"])
            return finding

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the worker's task. Subclasses implement this.

        Returns:
            Standardized finding dictionary.
        """
        pass

    def _start_timing(self):
        """Start timing the worker execution."""
        self.start_time = time.time()

    def _end_timing(self) -> float:
        """End timing and return elapsed seconds."""
        self.end_time = time.time()
        return self.end_time - self.start_time if self.start_time else 0.0

    def _create_finding(
        self,
        status: str,
        items: list,
        synthesis: str = "",
        token_count: int = 0,
        items_found: int = 0,
        error: str = ""
    ) -> Dict[str, Any]:
        """
        Create a standardized finding report.

        Args:
            status: "success" or "error"
            items: List of items (papers/blogs/news/stocks)
            synthesis: Worker's summary of findings
            token_count: LLM tokens used
            items_found: Raw items before filtering
            error: Error message if status=="error"

        Returns:
            Standardized finding dictionary
        """
        processing_time = self._end_timing()
        return {
            "worker": self.worker_name,
            "status": status,
            "items": items,
            "metadata": {
                "processing_time": processing_time,
                "token_count": token_count,
                "items_found": items_found or len(items),
                "items_kept": len(items)
            },
            "synthesis": synthesis,
            "error": error
        }
