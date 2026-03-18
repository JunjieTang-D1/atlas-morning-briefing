#!/usr/bin/env python3
"""
ArXiv paper PDF downloader.

Downloads PDFs for top-scored papers above a configurable threshold.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PaperDownloader:
    """Downloads ArXiv paper PDFs for high-scoring papers."""

    def __init__(
        self,
        output_dir: str = "paper_downloads",
        min_score: float = 8.0,
        max_papers: int = 5,
        delay: float = 1.0,
    ):
        """
        Initialize PaperDownloader.

        Args:
            output_dir: Directory to save downloaded PDFs.
            min_score: Minimum paper score to trigger download.
            max_papers: Maximum number of papers to download per run.
            delay: Delay in seconds between downloads (be nice to arxiv).
        """
        self.output_dir = Path(output_dir)
        self.min_score = min_score
        self.max_papers = max_papers
        self.delay = delay

    def _arxiv_id_from_url(self, url: str) -> str:
        """Extract arxiv ID from a URL like http://arxiv.org/abs/2506.12345v1."""
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        return match.group(1) if match else ""

    def _build_pdf_url(self, paper: Dict[str, Any]) -> str:
        """Get or construct the PDF URL for a paper."""
        pdf_link = paper.get("pdf_link", "")
        if pdf_link:
            # Ensure it ends with .pdf
            if not pdf_link.endswith(".pdf"):
                pdf_link += ".pdf"
            # Ensure https
            return pdf_link.replace("http://", "https://")

        # Fallback: construct from arxiv_url or id
        arxiv_url = paper.get("arxiv_url", "") or paper.get("id", "")
        arxiv_id = self._arxiv_id_from_url(arxiv_url)
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return ""

    def _safe_filename(self, title: str, arxiv_id: str) -> str:
        """Create a filesystem-safe filename from paper title and ID."""
        clean = re.sub(r"[^\w\s-]", "", title)
        clean = re.sub(r"\s+", "_", clean).strip("_")[:80]
        if arxiv_id:
            return f"{arxiv_id}_{clean}.pdf"
        return f"{clean}.pdf"

    def download_papers(
        self, scored_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Download PDFs for papers scoring above the threshold.

        Args:
            scored_papers: Papers with a 'score' field (already sorted desc).

        Returns:
            List of dicts with download results:
              [{"title": ..., "score": ..., "path": ..., "success": bool}, ...]
        """
        eligible = [
            p for p in scored_papers
            if p.get("score", 0) >= self.min_score
        ][:self.max_papers]

        if not eligible:
            logger.info(
                f"No papers above download threshold ({self.min_score})"
            )
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Downloading {len(eligible)} papers "
            f"(score >= {self.min_score}) to {self.output_dir}"
        )

        results = []
        for i, paper in enumerate(eligible):
            title = paper.get("title", "untitled")
            score = paper.get("score", 0)
            pdf_url = self._build_pdf_url(paper)

            if not pdf_url:
                logger.warning(f"No PDF URL for: {title}")
                results.append({
                    "title": title, "score": score,
                    "path": None, "success": False,
                })
                continue

            arxiv_id = self._arxiv_id_from_url(
                paper.get("arxiv_url", "") or paper.get("id", "")
            )
            filename = self._safe_filename(title, arxiv_id)
            dest = self.output_dir / filename

            # Skip if already downloaded
            if dest.exists() and dest.stat().st_size > 1000:
                logger.info(f"Already downloaded: {filename}")
                results.append({
                    "title": title, "score": score,
                    "path": str(dest), "success": True,
                })
                continue

            try:
                logger.info(
                    f"[{i+1}/{len(eligible)}] Downloading: {title[:60]}... "
                    f"(score: {score:.1f})"
                )
                resp = requests.get(pdf_url, timeout=60, stream=True)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "pdf" not in content_type and "octet" not in content_type:
                    logger.warning(
                        f"Unexpected content-type '{content_type}' for {title}"
                    )

                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                size_kb = dest.stat().st_size / 1024
                logger.info(f"Saved: {filename} ({size_kb:.0f} KB)")
                results.append({
                    "title": title, "score": score,
                    "path": str(dest), "success": True,
                })

            except Exception as e:
                logger.error(f"Failed to download '{title}': {e}")
                results.append({
                    "title": title, "score": score,
                    "path": None, "success": False,
                })

            # Rate-limit to be polite to arxiv
            if i < len(eligible) - 1:
                time.sleep(self.delay)

        downloaded = sum(1 for r in results if r["success"])
        logger.info(f"Downloaded {downloaded}/{len(eligible)} papers")
        return results
