"""
Main orchestration script for the consistency checking pipeline.
"""

import os
import sys
import yaml
import csv
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import download_gutenberg_book, strip_gutenberg_metadata
from src.pathway_setup import chunk_by_paragraphs
from src.claim_extractor import parse_backstory
from src.consistency_checker import load_model
from src.graph_schema import WorldGraph
from src.graph_builder import build_graph_from_book, merge_graphs
from src.graph_verifier import verify_claim, VerificationResult
from src.classifier import aggregate_evidence


def setup_logging(log_file: str = "pipeline.log") -> None:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_test_case(
    story_id: str,
    backstory: str,
    graph: WorldGraph,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> tuple:
    """
    Process a single test case through the graph-based pipeline.

    Args:
        story_id: Unique identifier for the test case
        backstory: Backstory text to check
        graph: WorldGraph with indexed knowledge
        model: Loaded LLM model (used for claim parsing only)
        tokenizer: Loaded tokenizer
        config: Configuration dictionary

    Returns:
        Tuple of (story_id, counts_dict, rationale)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing test case: {story_id}")

    # Step 1: Extract claims from backstory
    claims = parse_backstory(backstory)
    logger.info(f"Extracted {len(claims)} claims from backstory")

    if not claims:
        logger.warning(f"No claims extracted for {story_id}")
        return story_id, {"true": 0, "false": 0, "unknown": 0}, "No valid claims found in backstory"

    # Step 2: Verify each claim against the graph
    results = []
    for claim in claims:
        result, explanation = verify_claim(claim, graph, model, tokenizer)
        results.append((result, explanation))
        logger.info(f"Claim: '{claim[:50]}...' -> {result.value}: {explanation[:50]}...")

    # Step 3: Aggregate results
    counts, rationale = aggregate_evidence(results)

    logger.info(f"Test case {story_id}: {counts}")

    return story_id, counts, rationale


def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting Consistency Checking Pipeline")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

    # Step 1: Download/load novels from Gutenberg
    book_ids = config.get("gutenberg_books", [])
    if not book_ids:
        logger.warning("No Gutenberg book IDs specified in config")
        book_ids = [1342]  # Default to Pride and Prejudice

    logger.info(f"Downloading {len(book_ids)} books from Project Gutenberg...")
    novels = {}
    for book_id in book_ids:
        try:
            raw_text = download_gutenberg_book(book_id)
            clean_text = strip_gutenberg_metadata(raw_text)
            novels[str(book_id)] = clean_text
            logger.info(
                f"Downloaded and cleaned book {book_id} ({len(clean_text)} characters)"
            )
        except Exception as e:
            logger.error(f"Failed to download book {book_id}: {str(e)}")
            continue

    if not novels:
        logger.error("No novels downloaded. Exiting.")
        sys.exit(1)

    # Step 2: Load LLM model ONCE (needed for graph building and claim parsing)
    model_name = config.get("models", {}).get("llm", "Qwen/Qwen2-0.5B")
    device = config.get("models", {}).get("device", "auto")

    logger.info(f"Loading LLM model: {model_name}...")
    try:
        model, tokenizer = load_model(model_name, device)
        logger.info("LLM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM model: {str(e)}")
        sys.exit(1)

    # Step 3: Build knowledge graph from novels
    logger.info("Building knowledge graph from novels...")
    graph = WorldGraph()

    for book_id, novel_text in novels.items():
        try:
            logger.info(f"Processing book {book_id}...")
            chunks = chunk_by_paragraphs(novel_text)
            logger.info(f"Created {len(chunks)} chunks for book {book_id}")

            book_graph = build_graph_from_book(chunks, book_id, model, tokenizer)
            merge_graphs(graph, book_graph)

            logger.info(f"Book {book_id}: extracted {len(book_graph.entities)} entities, "
                       f"{len(book_graph.relations)} relations")
        except Exception as e:
            logger.error(f"Failed to build graph for book {book_id}: {str(e)}")
            continue

    stats = graph.get_stats()
    logger.info(f"Knowledge graph built: {stats['total_entities']} entities, "
               f"{stats['total_relations']} relations")

    # Step 4: Process test cases
    results = []

    # Hardcoded test case: Pride and Prejudice
    test_case = {
        "story_id": "alice_003",
        "backstory": """
    Alice enters Wonderland by falling down a rabbit hole.
    She has tea with the Mad Hatter.
    The Cheshire Cat is Alice's brother.
    The Queen of Hearts loves gardening peacefully.
    """,
    }

    logger.info("Processing hardcoded test case...")
    try:
        story_id, counts, rationale = process_test_case(
            test_case["story_id"],
            test_case["backstory"],
            graph,
            model,
            tokenizer,
            config,
        )
        results.append(
            {"story_id": story_id, "counts": str(counts), "rationale": rationale}
        )
        logger.info(f"Test case {story_id} completed: {rationale}")
    except Exception as e:
        logger.error(f"Error processing test case: {str(e)}")

    # Step 5: Save results to CSV
    results_file = "results.csv"
    logger.info(f"Writing results to {results_file}...")

    try:
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["story_id", "counts", "rationale"]
            )
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Results written to {results_file}")
        logger.info(f"Total test cases processed: {len(results)}")
    except Exception as e:
        logger.error(f"Failed to write results: {str(e)}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
