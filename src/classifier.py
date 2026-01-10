"""
Evidence aggregation and final prediction classification.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def aggregate_evidence(consistency_results: List[Tuple[bool, str]]) -> Tuple[int, str]:
    """
    Aggregate consistency results from multiple claims into a final prediction.
    
    Uses majority voting: if >50% of claims are consistent, prediction=1, else 0.
    Generates a concise rationale summarizing the key findings.
    
    Args:
        consistency_results: List of (is_consistent, explanation) tuples
        
    Returns:
        Tuple of (prediction: int, rationale: str)
        - prediction: 0 (inconsistent) or 1 (consistent)
        - rationale: Brief 1-2 line explanation
    """
    if not consistency_results:
        logger.warning("No consistency results provided")
        return 0, "No claims to evaluate"
    
    # Count consistent and inconsistent claims
    consistent_count = sum(1 for is_consistent, _ in consistency_results if is_consistent)
    total_count = len(consistency_results)
    inconsistent_count = total_count - consistent_count
    
    # Majority voting
    prediction = 1 if consistent_count > (total_count / 2) else 0
    
    # Generate rationale
    if prediction == 1:
        if consistent_count == total_count:
            rationale = f"All {total_count} claims are consistent with the evidence."
        else:
            rationale = f"{consistent_count} out of {total_count} claims are consistent with the evidence."
    else:
        if inconsistent_count == total_count:
            rationale = f"All {total_count} claims are inconsistent with the evidence."
        else:
            rationale = f"{inconsistent_count} out of {total_count} claims are inconsistent with the evidence."
    
    # Add key explanations if available
    key_explanations = []
    for is_consistent, explanation in consistency_results:
        if explanation and len(explanation) > 10:
            # Include explanations for inconsistent claims (more informative)
            if not is_consistent:
                key_explanations.append(explanation[:100])
            elif len(key_explanations) < 2:  # Also include some consistent ones
                key_explanations.append(explanation[:100])
    
    if key_explanations:
        # Add a summary of key findings
        summary = " ".join(key_explanations[:2])  # Use first 2 explanations
        if len(summary) > 150:
            summary = summary[:150] + "..."
        rationale += f" Key findings: {summary}"
    
    logger.info(f"Aggregated {total_count} claims: prediction={prediction}, consistent={consistent_count}, inconsistent={inconsistent_count}")
    
    return prediction, rationale
