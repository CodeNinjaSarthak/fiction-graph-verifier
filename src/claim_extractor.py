"""
Claim extraction from backstory text.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def parse_backstory(backstory: str) -> List[str]:
    """
    Extract atomic claims from backstory text.

    Handles:
    - Line breaks (each line can be a claim)
    - Multi-sentence paragraphs
    - Trailing punctuation
    - Empty lines

    Args:
        backstory: Full backstory text

    Returns:
        List of individual claim strings
    """
    if not backstory or not backstory.strip():
        logger.warning("Empty backstory provided")
        return []

    claims = []

    # Split on line breaks first
    lines = backstory.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split on sentence-ending punctuation, keeping the punctuation
        # Uses lookbehind to split after .!? followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', line)

        for part in parts:
            sentence = part.strip()
            if not sentence:
                continue

            # Filter only truly degenerate cases (less than 3 words)
            word_count = len(sentence.split())
            if word_count < 3:
                logger.debug(f"Filtering short sentence ({word_count} words): {sentence}")
                continue

            # Must contain at least one letter
            if not re.search(r'[a-zA-Z]', sentence):
                continue

            claims.append(sentence)

    logger.info(f"Extracted {len(claims)} claims from backstory")

    if not claims:
        logger.warning("No valid claims extracted from backstory")
        # Only use fallback if there's content
        if backstory.strip():
            return [backstory.strip()]

    return claims
