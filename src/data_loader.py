"""
Data loading utilities for Gutenberg books and backstories.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

try:
    from gutenbergpy import textget, gutenbergcache
except ImportError:
    textget = None


logger = logging.getLogger(__name__)


def download_gutenberg_book(book_id: int, cache_dir: str = "./cache/books") -> str:
    if textget is None:
        raise ImportError("gutenbergpy import failed")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cached_file = cache_path / f"{book_id}.txt"
    if cached_file.exists():
        logger.info(f"Loading book {book_id} from cache: {cached_file}")
        return cached_file.read_text(encoding="utf-8")

    try:
        logger.info(f"Downloading book {book_id} from Project Gutenberg...")

        raw = textget.get_text_by_id(book_id)
        if not raw:
            raise ValueError(f"No text returned for book ID {book_id}")

        text = raw.decode("utf-8", errors="ignore")

        cached_file.write_text(text, encoding="utf-8")
        logger.info(f"Cached book {book_id} to {cached_file}")

        return text

    except Exception as e:
        logger.error(f"Error downloading book {book_id}: {e}")
        raise ValueError(f"Failed to download book {book_id}") from e


def strip_gutenberg_metadata(text: str) -> str:
    """
    Remove Project Gutenberg headers and footers from text.

    Args:
        text: Raw text from Project Gutenberg

    Returns:
        Cleaned text without headers and footers
    """
    # Pattern to match Gutenberg header (starts with *** START OF...)
    header_pattern = r"\*\*\* START OF.*?\*\*\*"
    # Pattern to match Gutenberg footer (ends with *** END OF...)
    footer_pattern = r"\*\*\* END OF.*?\*\*\*"

    # Remove header
    text = re.sub(header_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove footer
    text = re.sub(footer_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove common Gutenberg metadata lines
    lines_to_remove = [
        r"^Project Gutenberg.*$",
        r"^This eBook.*$",
        r"^Title:.*$",
        r"^Author:.*$",
        r"^Release Date:.*$",
        r"^Language:.*$",
        r"^Character set encoding:.*$",
    ]

    for pattern in lines_to_remove:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Clean up multiple blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def load_backstory(file_path: str) -> str:
    """
    Load backstory from a JSON or text file.

    Args:
        file_path: Path to backstory file (JSON or text)

    Returns:
        Backstory text as a string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Backstory file not found: {file_path}")

    try:
        # Try loading as JSON first
        content = path.read_text(encoding="utf-8")
        try:
            data = json.loads(content)
            # If JSON, extract text from common keys
            if isinstance(data, dict):
                # Try common keys
                for key in ["backstory", "text", "content", "story"]:
                    if key in data:
                        return str(data[key])
                # If no common key, return the whole dict as string
                return json.dumps(data, indent=2)
            elif isinstance(data, str):
                return data
            else:
                return str(data)
        except json.JSONDecodeError:
            # Not JSON, treat as plain text
            return content

    except Exception as e:
        logger.error(f"Error loading backstory from {file_path}: {str(e)}")
        raise ValueError(f"Failed to load backstory: {str(e)}")
