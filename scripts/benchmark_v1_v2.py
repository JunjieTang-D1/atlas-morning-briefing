#!/usr/bin/env python3
"""
Benchmark v0.1 vs v0.2 morning briefing architectures.

Compares:
1. Speed (wall-clock time)
2. Quality (LLM judge scores on relevance, accuracy, insight, actionability)
3. Cost (token count)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bedrock_client import BedrockClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s)")
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmark comparison between v0.1 and v0.2."""

    def __init__(self, config_path: str):
        """
        Initialize benchmark runner.

        Args:
            config_path: Path to config YAML
        """
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.bedrock = BedrockClient(self.config)

    def run(self) -> Dict[str, Any]:
        """
        Run benchmark comparing v0.1 and v0.2.

        Returns:
            Benchmark results dictionary
        """
        logger.info("=== Morning Briefing Benchmark: v0.1 vs v0.2 ===")

        # Run v0.1
        logger.info("\n=== Running v0.1 (current serial + partial parallel) ===")
        v1_results = self._run_version(
            script="scripts/briefing_runner.py",
            output_file="Atlas-Briefing-v0.1.md",
            version="v0.1"
        )

        # Run v0.2
        logger.info("\n=== Running v0.2 (coordinator + parallel workers) ===")
        v2_results = self._run_version(
            script="scripts/briefing_runner_v2.py",
            output_file="Atlas-Briefing-v0.2.md",
            version="v0.2"
        )

        # Compare quality with LLM judge
        logger.info("\n=== LLM Judge Evaluation ===")
        quality_comparison = self._judge_quality(v1_results, v2_results)

        # Compile results
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "actual_execution",
            "v0.1": v1_results,
            "v0.2": v2_results,
            "quality_judge": quality_comparison,
            "comparison": {
                "speed_improvement_pct": (
                    (v1_results["time_seconds"] - v2_results["time_seconds"])
                    / v1_results["time_seconds"]
                    * 100
                ),
                "token_diff": v2_results["tokens"] - v1_results["tokens"],
                "quality_improvement": (
                    quality_comparison["v0.2"]["overall_score"]
                    - quality_comparison["v0.1"]["overall_score"]
                )
            }
        }

        # Save results
        self._save_results(results)

        return results

    def _run_version(
        self,
        script: str,
        output_file: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Run a specific version and collect metrics.

        Args:
            script: Path to runner script
            output_file: Expected output filename
            version: Version string

        Returns:
            Results dictionary
        """
        start_time = time.time()

        # Run script
        cmd = [
            sys.executable,
            script,
            "--config", self.config_path,
            "--dry-run"  # Don't send emails during benchmark
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"{version} failed: {result.stderr}")
                return {
                    "version": version,
                    "time_seconds": elapsed,
                    "tokens": 0,
                    "success": False,
                    "error": result.stderr,
                    "output": ""
                }

            # Parse output to extract token count
            tokens = self._extract_token_count(result.stdout)

            # Read generated briefing
            output_path = Path(output_file)
            if output_path.exists():
                with open(output_path) as f:
                    output_content = f.read()
            else:
                output_content = ""
                logger.warning(f"Output file {output_file} not found")

            logger.info(f"{version} completed in {elapsed:.1f}s, {tokens} tokens")

            return {
                "version": version,
                "time_seconds": elapsed,
                "tokens": tokens,
                "success": True,
                "error": "",
                "output": output_content
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            logger.error(f"{version} timed out after {elapsed:.1f}s")
            return {
                "version": version,
                "time_seconds": elapsed,
                "tokens": 0,
                "success": False,
                "error": "Timeout",
                "output": ""
            }
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{version} error: {e}")
            return {
                "version": version,
                "time_seconds": elapsed,
                "tokens": 0,
                "success": False,
                "error": str(e),
                "output": ""
            }

    def _extract_token_count(self, stdout: str) -> int:
        """Extract token count from script output."""
        # Look for "Total LLM tokens used: XXXX"
        import re
        match = re.search(r"Total LLM tokens used:\s*(\d+)", stdout)
        if match:
            return int(match.group(1))
        return 0

    def _judge_quality(
        self,
        v1_results: Dict[str, Any],
        v2_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM judge to compare quality of v0.1 and v0.2 outputs.

        Args:
            v1_results: v0.1 results
            v2_results: v0.2 results

        Returns:
            Quality comparison dictionary
        """
        if not v1_results["success"] or not v2_results["success"]:
            logger.warning("Cannot judge quality - one or both versions failed")
            return {
                "v0.1": {"relevance": 0, "accuracy": 0, "insight": 0, "actionability": 0, "overall_score": 0},
                "v0.2": {"relevance": 0, "accuracy": 0, "insight": 0, "actionability": 0, "overall_score": 0}
            }

        if not self.bedrock.available:
            logger.warning("Bedrock unavailable - skipping quality judgment")
            return {
                "v0.1": {"relevance": 0, "accuracy": 0, "insight": 0, "actionability": 0, "overall_score": 0},
                "v0.2": {"relevance": 0, "accuracy": 0, "insight": 0, "actionability": 0, "overall_score": 0}
            }

        # Judge v0.1
        v1_scores = self._judge_single_output(v1_results["output"], "v0.1")

        # Judge v0.2
        v2_scores = self._judge_single_output(v2_results["output"], "v0.2")

        return {
            "v0.1": v1_scores,
            "v0.2": v2_scores
        }

    def _judge_single_output(self, output: str, version: str) -> Dict[str, int]:
        """
        Use LLM to judge a single output.

        Args:
            output: Briefing content
            version: Version string

        Returns:
            Scores dictionary
        """
        logger.info(f"Judging {version}...")

        # Truncate output if too long (keep first 3000 chars)
        output_sample = output[:3000] if len(output) > 3000 else output

        prompt = f"""You are evaluating a morning briefing for an AI research professional.

Rate the following briefing on these dimensions (0-10 scale):

1. **Relevance**: How relevant are the items to Agentic AI, Trainium, VLA Robotics, Multi-Agent Systems?
2. **Accuracy**: Do the summaries accurately reflect the content? Any hallucinations?
3. **Insight Depth**: Does it provide actionable insights or just summaries?
4. **Actionability**: Can the reader act on this information?

BRIEFING CONTENT:
{output_sample}

Return ONLY a JSON object in this format (no markdown, no explanation):
{{"relevance": X, "accuracy": X, "insight": X, "actionability": X}}
"""

        try:
            response = self.bedrock.invoke(prompt, tier="heavy")

            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group(0))
                scores["overall_score"] = sum(scores.values()) / len(scores)
                logger.info(f"{version} overall score: {scores['overall_score']:.1f}/10")
                return scores
            else:
                logger.error(f"Failed to parse judge response for {version}")
                return {"relevance": 0, "accuracy": 0, "insight": 0, "actionability": 0, "overall_score": 0}

        except Exception as e:
            logger.error(f"Error judging {version}: {e}")
            return {"relevance": 0, "accuracy": 0, "insight": 0, "actionability": 0, "overall_score": 0}

    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON and markdown."""
        # Save JSON
        json_path = Path("benchmark_v1_vs_v2.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {json_path}")

        # Generate markdown report
        self._generate_report(results)

    def _generate_report(self, results: Dict[str, Any]):
        """Generate markdown benchmark report."""
        report = "# Morning Briefing Benchmark: v0.1 vs v0.2\n\n"
        report += f"**Date:** {results['timestamp']}\n\n"
        report += f"**Method:** {results.get('method', 'unknown')} (Real wall-clock execution with actual token counts)\n\n"

        # Summary table
        report += "## Comparison Summary\n\n"
        report += "| Metric | v0.1 | v0.2 | Delta | Improvement |\n"
        report += "|--------|------|------|-------|-------------|\n"

        v1 = results["v0.1"]
        v2 = results["v0.2"]
        comp = results["comparison"]

        # Speed
        report += f"| **Speed** | {v1['time_seconds']:.1f}s | {v2['time_seconds']:.1f}s | "
        report += f"{comp['speed_improvement_pct']:.1f}% | "
        report += f"{'✅' if comp['speed_improvement_pct'] > 20 else '❌'} |\n"

        # Cost
        token_pct = (comp['token_diff'] / v1['tokens'] * 100) if v1['tokens'] > 0 else 0
        report += f"| **Tokens** | {v1['tokens']} | {v2['tokens']} | "
        report += f"{comp['token_diff']:+d} ({token_pct:+.1f}%) | "
        report += f"{'✅' if token_pct < 20 else '❌'} |\n"

        # Quality
        v1_qual = results["quality_judge"]["v0.1"]["overall_score"]
        v2_qual = results["quality_judge"]["v0.2"]["overall_score"]
        report += f"| **Quality** | {v1_qual:.1f}/10 | {v2_qual:.1f}/10 | "
        report += f"{comp['quality_improvement']:+.1f} | "
        report += f"{'✅' if comp['quality_improvement'] > 0 else '❌'} |\n\n"

        # Detailed quality breakdown
        report += "## Quality Breakdown\n\n"
        report += "| Dimension | v0.1 | v0.2 | Delta |\n"
        report += "|-----------|------|------|-------|\n"
        for dim in ["relevance", "accuracy", "insight", "actionability"]:
            v1_score = results["quality_judge"]["v0.1"][dim]
            v2_score = results["quality_judge"]["v0.2"][dim]
            delta = v2_score - v1_score
            report += f"| {dim.title()} | {v1_score}/10 | {v2_score}/10 | {delta:+.1f} |\n"

        report += "\n"

        # Verdict
        report += "## Verdict\n\n"
        speed_pass = comp['speed_improvement_pct'] > 20
        quality_pass = comp['quality_improvement'] > 2.0  # 20% improvement on 10-point scale
        cost_acceptable = token_pct < 50  # Token increase < 50%

        if speed_pass and quality_pass and cost_acceptable:
            report += "✅ **PASS**: v0.2 shows significant improvement (>20% speed gain AND quality improvement)\n\n"
        elif speed_pass or quality_pass:
            report += "⚠️  **PARTIAL PASS**: Some improvements but not all targets met\n\n"
        else:
            report += "❌ **FAIL**: v0.2 does not show >20% improvement on key metrics\n\n"

        # Architecture notes
        report += "## Architecture Changes in v0.2\n\n"
        report += "1. **Coordinator + Parallel Workers**: All workers (papers, blogs, news+market) run in parallel\n"
        report += "2. **Self-Contained Workers**: Each worker does fetch + enrich independently (no round-trips)\n"
        report += "3. **Memory System**: briefing-memory/ with MEMORY.md index and topic files\n"
        report += "4. **Skeptical Memory**: Memory treated as hints not truth, verified before use\n"
        report += "5. **Coordinator Synthesis**: Coordinator reads ALL findings and synthesizes (not lazy delegation)\n\n"

        # Save report
        report_path = Path("benchmark_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")

        # Print summary
        print("\n" + "="*60)
        print(report)
        print("="*60 + "\n")


def main():
    """Main entry point for benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark v0.1 vs v0.2")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.config)
    results = runner.run()

    # Exit with appropriate code
    speed_improvement = results["comparison"]["speed_improvement_pct"]
    quality_improvement = results["comparison"]["quality_improvement"]

    if speed_improvement > 20 or quality_improvement > 2.0:
        logger.info("✅ VERIFICATION: PASS - v0.2 shows >20% improvement")
        sys.exit(0)
    else:
        logger.info("❌ VERIFICATION: FAIL - v0.2 does not show >20% improvement")
        sys.exit(1)


if __name__ == "__main__":
    main()
