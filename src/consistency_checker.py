"""
LLM model loading utilities.
(Consistency checking logic removed - replaced by graph-based verification)
"""

import logging
import torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Global model and tokenizer (loaded once, reused)
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def detect_device() -> str:
    """
    Auto-detect the best available device (GPU or CPU).

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = 'cpu'
        logger.info("Using CPU device")

    return device


def load_model(model_name: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load transformer model and tokenizer (loads once, reuses global variables).

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2-0.5B")
        device: Device to load model on ("auto", "cuda", "cpu", "mps")

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer

    # Return cached model if already loaded
    if _model is not None and _tokenizer is not None:
        logger.info("Reusing already loaded model")
        return _model, _tokenizer

    # Auto-detect device if needed
    if device == "auto":
        device = detect_device()

    logger.info(f"Loading model: {model_name} on device: {device}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto" if device != 'cpu' else None,
            low_cpu_mem_usage=True
        )

        if device != 'cpu' and not hasattr(model, 'device_map'):
            model = model.to(device)

        model.eval()  # Set to evaluation mode

        # Cache globally
        _model = model
        _tokenizer = tokenizer

        logger.info(f"Model {model_name} loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise
