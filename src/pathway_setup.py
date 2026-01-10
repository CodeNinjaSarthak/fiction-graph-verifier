"""
Pathway vector store setup and indexing utilities.
"""

import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# In-memory vector store (will be replaced with Pathway integration)
# This is a simplified implementation that can be enhanced with actual Pathway API
_vector_store: Dict[str, Any] = {"embeddings": [], "chunks": [], "metadata": []}


def initialize_pathway_store(config: dict) -> Dict[str, Any]:
    """
    Initialize Pathway vector store for document indexing.

    Args:
        config: Configuration dictionary with Pathway settings

    Returns:
        Dictionary representing the vector store (to be replaced with Pathway Index)
    """
    logger.info("Initializing Pathway vector store...")

    # Initialize embedding model
    embedding_model_name = config.get("models", {}).get(
        "embedding", "sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info(f"Loading embedding model: {embedding_model_name}")

    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        raise

    store = {
        "model": embedding_model,
        "embeddings": [],
        "chunks": [],
        "metadata": [],
        "config": config,
    }

    logger.info("Pathway vector store initialized")
    return store


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        # Move start position with overlap
        start = end - chunk_overlap
        if start >= text_length:
            break

    return chunks


def chunk_by_paragraphs(text, min_len=300, max_len=800):
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


def index_novel(novel_text: str, book_id: str, store: Dict[str, Any]) -> None:
    """
    Index a novel into the Pathway vector store.

    Chunks the text, generates embeddings, and stores them with metadata.

    Args:
        novel_text: Full text of the novel
        book_id: Unique identifier for the book
        store: Vector store dictionary (from initialize_pathway_store)
    """
    logger.info(f"Indexing novel {book_id}...")

    config = store.get("config", {})
    chunks = chunk_by_paragraphs(novel_text)

    logger.info(f"Created {len(chunks)} chunks for book {book_id}")

    if not chunks:
        logger.warning(f"No chunks created for book {book_id}")
        return

    # Generate embeddings
    model = store["model"]
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")

    try:
        embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

    # Store chunks, embeddings, and metadata
    start_idx = len(store["chunks"])
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        store["chunks"].append(chunk)
        store["embeddings"].append(embedding)
        store["metadata"].append(
            {"book_id": book_id, "chunk_index": start_idx + i, "chunk_size": len(chunk)}
        )

    logger.info(f"Successfully indexed {len(chunks)} chunks for book {book_id}")


def get_store_stats(store: Dict[str, Any]) -> Dict[str, int]:
    """
    Get statistics about the vector store.

    Args:
        store: Vector store dictionary

    Returns:
        Dictionary with statistics
    """
    return {
        "total_chunks": len(store.get("chunks", [])),
        "total_embeddings": len(store.get("embeddings", [])),
        "unique_books": len(set(m.get("book_id") for m in store.get("metadata", []))),
    }
