"""
Evidence aggregation and final prediction classification.
"""

import logging
from typing import List, Tuple, Dict

from src.graph_verifier import VerificationResult

logger = logging.getLogger(__name__)


def aggregate_evidence(results: List[Tuple[VerificationResult, str]]) -> Tuple[Dict[str, int], str]:
    """
    Aggregate verification results from multiple claims.

    Args:
        results: List of (VerificationResult, explanation) tuples

    Returns:
        Tuple of (counts_dict, rationale)
        - counts_dict: {"true": N, "false": M, "unknown": K}
        - rationale: Brief summary string like "2 true, 2 false"
    """
    if not results:
        logger.warning("No verification results provided")
        return {"true": 0, "false": 0, "unknown": 0}, "No claims to evaluate"

    # Count results by type
    counts = {
        "true": sum(1 for r, _ in results if r == VerificationResult.TRUE),
        "false": sum(1 for r, _ in results if r == VerificationResult.FALSE),
        "unknown": sum(1 for r, _ in results if r == VerificationResult.UNKNOWN),
    }

    total = len(results)

    # Generate rationale
    parts = []
    if counts["true"] > 0:
        parts.append(f"{counts['true']} true")
    if counts["false"] > 0:
        parts.append(f"{counts['false']} false")
    if counts["unknown"] > 0:
        parts.append(f"{counts['unknown']} unknown")

    rationale = ", ".join(parts) if parts else "No results"

    logger.info(f"Aggregated {total} claims: {rationale}")

    return counts, rationale
