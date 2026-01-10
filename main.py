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
from src.pathway_setup import initialize_pathway_store, index_novel
from src.claim_extractor import parse_backstory
from src.retrieval import retrieve_passages
from src.consistency_checker import load_model, check_consistency_batch
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
    store: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> tuple:
    """
    Process a single test case through the pipeline.

    Args:
        story_id: Unique identifier for the test case
        backstory: Backstory text to check
        store: Vector store with indexed novels
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        config: Configuration dictionary

    Returns:
        Tuple of (story_id, prediction, rationale)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing test case: {story_id}")

    # Step 1: Extract claims from backstory
    claims = parse_backstory(backstory)
    logger.info(f"Extracted {len(claims)} claims from backstory")

    if not claims:
        logger.warning(f"No claims extracted for {story_id}")
        return story_id, 0, "No valid claims found in backstory"

    # Step 2: Retrieve passages for each claim
    top_k = config.get("retrieval", {}).get("top_k", 5)
    passages_list = []
    for claim in claims:
        passages = retrieve_passages(claim, store, k=top_k)
        passages_list.append(passages)
        logger.debug(f"Retrieved {len(passages)} passages for claim: {claim[:50]}...")

    # Step 3: Check consistency with LLM
    max_tokens = config.get("inference", {}).get("max_tokens", 256)
    batch_size = config.get("inference", {}).get("batch_size", 4)
    device = config.get("models", {}).get("device", "auto")

    logger.info(f"Checking consistency for {len(claims)} claims...")
    consistency_results = check_consistency_batch(
        claims,
        passages_list,
        model,
        tokenizer,
        max_new_tokens=max_tokens,
        batch_size=batch_size,
        device=device,
    )

    # Step 4: Aggregate results
    prediction, rationale = aggregate_evidence(consistency_results)

    logger.info(f"Test case {story_id}: prediction={prediction}")

    return story_id, prediction, rationale


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

    # Step 2: Initialize Pathway vector store
    try:
        store = initialize_pathway_store(config)
        logger.info("Pathway vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        sys.exit(1)

    # Step 3: Index all novels
    logger.info("Indexing novels into vector store...")
    for book_id, novel_text in novels.items():
        try:
            index_novel(novel_text, book_id, store)
        except Exception as e:
            logger.error(f"Failed to index book {book_id}: {str(e)}")
            continue

    logger.info(f"Indexed {len(novels)} novels into vector store")

    # Step 4: Load LLM model ONCE
    model_name = config.get("models", {}).get("llm", "microsoft/phi-2")
    device = config.get("models", {}).get("device", "auto")

    logger.info(f"Loading LLM model: {model_name}...")
    try:
        model, tokenizer = load_model(model_name, device)
        logger.info("LLM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM model: {str(e)}")
        sys.exit(1)

    # Step 5: Process test cases
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
        story_id, prediction, rationale = process_test_case(
            test_case["story_id"],
            test_case["backstory"],
            store,
            model,
            tokenizer,
            config,
        )
        results.append(
            {"story_id": story_id, "prediction": prediction, "rationale": rationale}
        )
        logger.info(f"Test case {story_id} completed: prediction={prediction}")
    except Exception as e:
        logger.error(f"Error processing test case: {str(e)}")

    # Step 6: Save results to CSV
    results_file = "results.csv"
    logger.info(f"Writing results to {results_file}...")

    try:
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["story_id", "prediction", "rationale"]
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
