"""
Retrieval utilities for finding relevant passages using hybrid search.
"""

import logging
import re
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def extract_keywords(text: str) -> List[str]:
    """
    Extract potential character names and important keywords from text.

    Args:
        text: Text to extract keywords from

    Returns:
        List of potential keywords (capitalized words, likely names)
    """
    # Find capitalized words (potential character names)
    words = re.findall(r"\b[A-Z][a-z]+\b", text)
    # Filter out common words
    common_words = {"The", "This", "That", "There", "Then", "They", "These", "Those"}
    keywords = [w for w in words if w not in common_words and len(w) > 2]
    return list(set(keywords))  # Remove duplicates


def keyword_match_score(query: str, passage: str) -> float:
    """
    Calculate keyword matching score between query and passage.

    Args:
        query: Query text
        passage: Passage text

    Returns:
        Keyword match score (0.0 to 1.0)
    """
    query_keywords = set(extract_keywords(query))
    passage_keywords = set(extract_keywords(passage))

    if not query_keywords:
        return 0.0

    # Calculate Jaccard similarity
    intersection = query_keywords & passage_keywords
    union = query_keywords | passage_keywords

    if not union:
        return 0.0

    return len(intersection) / len(union)


def retrieve_passages(
    claim: str,
    store: dict,
    k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[str]:
    """
    Retrieve relevant passages using hybrid search (semantic + keyword).

    Args:
        claim: Claim to find relevant passages for
        store: Vector store dictionary with embeddings and chunks
        k: Number of top passages to return
        semantic_weight: Weight for semantic similarity (0.0 to 1.0)
        keyword_weight: Weight for keyword matching (0.0 to 1.0)

    Returns:
        List of top-k most relevant passages
    """
    if not store.get("chunks") or not store.get("embeddings"):
        logger.warning("Vector store is empty")
        return []

    logger.debug(f"Retrieving passages for claim: {claim[:100]}...")

    # Get embedding model and generate query embedding
    model = store["model"]
    query_embedding = model.encode([claim], convert_to_numpy=True)[0]

    # Get all stored embeddings and chunks
    stored_embeddings = np.array(store["embeddings"])
    chunks = store["chunks"]

    # Calculate semantic similarity scores
    query_embedding_tensor = l2_normalize(query_embedding.reshape(1, -1))
    stored_embeddings_tensor = l2_normalize(stored_embeddings)

    semantic_scores = np.dot(
        stored_embeddings_tensor, query_embedding_tensor.T
    ).flatten()

    # Normalize semantic scores to [0, 1]
    semantic_scores = (semantic_scores - semantic_scores.min()) / (
        semantic_scores.max() - semantic_scores.min() + 1e-8
    )

    # Calculate keyword match scores
    keyword_scores = np.array([keyword_match_score(claim, chunk) for chunk in chunks])

    # Combine scores with weights
    combined_scores = (semantic_weight * semantic_scores) + (
        keyword_weight * keyword_scores
    )

    # Get top-k indices
    top_indices = np.argsort(combined_scores)[::-1][:k]

    # Retrieve top-k passages
    retrieved_passages = [chunks[idx] for idx in top_indices]

    logger.debug(f"Retrieved {len(retrieved_passages)} passages")

    return retrieved_passages
