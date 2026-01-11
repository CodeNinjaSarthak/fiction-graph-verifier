"""
Main orchestration script for the consistency checking pipeline.

Two-tier architecture:
1. World is pre-built in Colab (using build_world_colab.py)
2. Local verification loads world and checks claims
"""

import os
import sys
import yaml
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.consistency_checker import load_model
from src.graph_schema import WorldGraph
from src.graph_builder import load_graph_from_world, verify_world_integrity
from src.graph_verifier import verify_claim, VerificationResult
from src.classifier import aggregate_evidence
from src.claim_extractor import parse_backstory
import dotenv

dotenv.load_dotenv()


def setup_logging(log_file: str = "pipeline.log") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {
            "models": {"llm": "Qwen/Qwen2-0.5B", "device": "auto"},
            "world_dir": "world",
        }
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def verify_single_claim(
    claim: str,
    graph: WorldGraph,
    model: Any = None,
    tokenizer: Any = None,
    provider: str = "gemini",
    gemini_model: str = "gemini-1.5-flash",
) -> tuple:
    """
    Verify a single claim against the world graph.

    Args:
        claim: The claim to verify
        graph: Loaded WorldGraph
        model: LLM model for claim parsing (optional, required for local provider)
        tokenizer: Tokenizer (optional, required for local provider)
        provider: "gemini" or "local"
        gemini_model: Gemini model name

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    return verify_claim(claim, graph, model, tokenizer, provider, gemini_model)


def process_backstory(
    story_id: str,
    backstory: str,
    graph: WorldGraph,
    model: Any = None,
    tokenizer: Any = None,
    provider: str = "gemini",
    gemini_model: str = "gemini-1.5-flash",
) -> tuple:
    """
    Process a backstory with multiple claims.

    Args:
        story_id: Unique identifier
        backstory: Backstory text with multiple claims
        graph: Loaded WorldGraph
        model: LLM model for claim parsing (optional, required for local provider)
        tokenizer: Tokenizer (optional, required for local provider)
        provider: "gemini" or "local"
        gemini_model: Gemini model name

    Returns:
        Tuple of (story_id, counts_dict, rationale)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing backstory: {story_id}")

    # Extract claims from backstory
    claims = parse_backstory(backstory)
    logger.info(f"Extracted {len(claims)} claims")

    if not claims:
        return story_id, {"true": 0, "false": 0, "unknown": 0}, "No valid claims found"

    # Verify each claim
    results = []
    for claim in claims:
        result, explanation = verify_claim(
            claim, graph, model, tokenizer, provider, gemini_model
        )
        results.append((result, explanation))
        logger.info(f"Claim: '{claim[:50]}...' -> {result.value}")

    # Aggregate results
    counts, rationale = aggregate_evidence(results)
    logger.info(f"Result: {counts}")

    return story_id, counts, rationale


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Narrative Consistency Verification System"
    )
    parser.add_argument("--claim", type=str, help="Single claim to verify")
    parser.add_argument(
        "--backstory", type=str, help="Backstory with multiple claims to verify"
    )
    parser.add_argument(
        "--world-dir",
        type=str,
        default="world",
        help="Directory containing world JSON files",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="LLM model name for claim parsing"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Narrative Consistency Verification System")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)

    # Check world directory exists
    world_dir = args.world_dir
    if not os.path.exists(world_dir):
        logger.error(f"World directory not found: {world_dir}")
        logger.error("Please run build_world_colab.py first to create the world.")
        sys.exit(1)

    # Load world graph
    logger.info(f"Loading world from: {world_dir}")
    try:
        graph = load_graph_from_world(world_dir)
        stats = graph.get_stats()
        logger.info(
            f"World loaded: {stats['total_entities']} entities, "
            f"{stats['total_events']} events, {stats['total_edges']} edges"
        )

        # Verify integrity
        warnings = verify_world_integrity(graph)
        if warnings:
            logger.warning(f"World integrity warnings: {len(warnings)}")
            for w in warnings[:5]:
                logger.warning(f"  - {w}")
    except Exception as e:
        logger.error(f"Failed to load world: {e}")
        sys.exit(1)

    # Get claim parser provider settings
    parser_config = config.get("claim_parser", {})
    provider = parser_config.get("provider", "gemini")
    gemini_config = parser_config.get("gemini", {})
    gemini_model = gemini_config.get("model", "gemini-1.5-flash")

    # Load local model only if using local provider
    model, tokenizer = None, None
    if provider == "local":
        local_config = parser_config.get("local", {})
        model_name = args.model or local_config.get("model", "Qwen/Qwen2.5-3B-Instruct")
        device = local_config.get("device", "auto")

        logger.info(f"Loading local LLM model: {model_name}")
        try:
            model, tokenizer = load_model(model_name, device)
            logger.info("Local LLM model loaded (used for claim parsing only)")
        except Exception as e:
            logger.error(f"Failed to load local LLM model: {e}")
            sys.exit(1)
    else:
        logger.info(f"Using {provider} API for claim parsing (model: {gemini_model})")

    # Process claims
    if args.claim:
        # Single claim mode
        logger.info(f"Verifying claim: {args.claim}")
        result, explanation = verify_single_claim(
            args.claim, graph, model, tokenizer, provider, gemini_model
        )
        print(f"\n{'=' * 60}")
        print(f"CLAIM: {args.claim}")
        print(f"RESULT: {result.value.upper()}")
        print(f"EXPLANATION: {explanation}")
        print(f"{'=' * 60}\n")

    elif args.backstory:
        # Backstory mode
        story_id, counts, rationale = process_backstory(
            "cli_backstory",
            args.backstory,
            graph,
            model,
            tokenizer,
            provider,
            gemini_model,
        )
        print(f"\n{'=' * 60}")
        print(f"BACKSTORY VERIFICATION")
        print(f"TRUE: {counts.get('true', 0)}")
        print(f"FALSE: {counts.get('false', 0)}")
        print(f"UNKNOWN: {counts.get('unknown', 0)}")
        print(f"RATIONALE: {rationale}")
        print(f"{'=' * 60}\n")

    else:
        # Demo mode with hardcoded test case
        logger.info("Running demo with hardcoded test case...")

        test_backstory = """
Alice enters Wonderland by falling down a rabbit hole.
She has tea with the Mad Hatter.
The Cheshire Cat is Alice's brother.
The Queen of Hearts loves gardening peacefully.
"""

        story_id, counts, rationale = process_backstory(
            "alice_test",
            test_backstory,
            graph,
            model,
            tokenizer,
            provider,
            gemini_model,
        )

        print(f"\n{'=' * 60}")
        print("DEMO VERIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Backstory:")
        for line in test_backstory.strip().split("\n"):
            if line.strip():
                print(f"  - {line.strip()}")
        print(f"\nResults:")
        print(f"  TRUE: {counts.get('true', 0)}")
        print(f"  FALSE: {counts.get('false', 0)}")
        print(f"  UNKNOWN: {counts.get('unknown', 0)}")
        print(f"\nRationale: {rationale}")
        print(f"{'=' * 60}\n")

        # Save to CSV
        results_file = "results.csv"
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["story_id", "counts", "rationale"])
            writer.writeheader()
            writer.writerow(
                {"story_id": story_id, "counts": str(counts), "rationale": rationale}
            )
        logger.info(f"Results written to {results_file}")

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
