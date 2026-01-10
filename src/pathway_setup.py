"""
Text chunking utilities.
(Vector store code removed - replaced by graph-based system)
"""

import logging

logger = logging.getLogger(__name__)


def chunk_by_paragraphs(text: str, min_len: int = 300, max_len: int = 800) -> list:
    """
    Split text into chunks based on paragraphs.

    Aggregates paragraphs to keep chunks between min_len and max_len characters.

    Args:
        text: Text to chunk
        min_len: Minimum chunk length in characters
        max_len: Maximum chunk length in characters

    Returns:
        List of text chunks
    """
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunks = []
    buf = ""

    for p in paras:
        if len(buf) + len(p) <= max_len:
            buf += " " + p
        else:
            if len(buf) >= min_len:
                chunks.append(buf.strip())
            buf = p

    if len(buf) >= min_len:
        chunks.append(buf.strip())

    return chunks
