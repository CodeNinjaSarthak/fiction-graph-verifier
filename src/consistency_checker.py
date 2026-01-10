"""
LLM-based consistency checking between claims and evidence.
"""

import logging
import torch
from typing import List, Tuple, Optional
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
        model_name: HuggingFace model name (e.g., "microsoft/phi-2")
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


def create_prompt(claim: str, passages: List[str]) -> str:
    """
    Create a prompt for consistency checking.
    
    Args:
        claim: The claim to check
        passages: Evidence passages from the novel
        
    Returns:
        Formatted prompt string
    """
    evidence_text = "\n".join([f"- {passage}" for passage in passages])
    
    prompt = f"""Backstory Claim: {claim}

Evidence from Novel:
{evidence_text}

Question: Is the claim consistent with the evidence?
Answer with Yes or No, then explain in one sentence.

Answer:"""
    
    return prompt


def parse_model_output(output: str) -> Tuple[bool, str]:
    """
    Parse model output to extract Yes/No answer and explanation.
    
    Args:
        output: Raw model output text
        
    Returns:
        Tuple of (is_consistent: bool, explanation: str)
    """
    output_lower = output.lower().strip()
    
    # Check for "yes" or "no" in the output
    if output_lower.startswith('yes'):
        is_consistent = True
    elif output_lower.startswith('no'):
        is_consistent = False
    elif 'yes' in output_lower[:50] and 'no' not in output_lower[:50]:
        is_consistent = True
    elif 'no' in output_lower[:50] and 'yes' not in output_lower[:50]:
        is_consistent = False
    else:
        # Default: try to infer from context
        # If output contains positive words, assume consistent
        positive_words = ['consistent', 'correct', 'true', 'accurate', 'matches']
        negative_words = ['inconsistent', 'incorrect', 'false', 'contradicts', 'does not match']
        
        pos_count = sum(1 for word in positive_words if word in output_lower)
        neg_count = sum(1 for word in negative_words if word in output_lower)
        
        is_consistent = pos_count > neg_count
    
    # Extract explanation (everything after Yes/No)
    explanation = output.strip()
    # Try to clean up the explanation
    if explanation.lower().startswith(('yes', 'no')):
        explanation = explanation[3:].strip()
    if explanation.startswith(':'):
        explanation = explanation[1:].strip()
    
    # Limit explanation length
    if len(explanation) > 200:
        explanation = explanation[:200] + "..."
    
    return is_consistent, explanation


def check_consistency(
    claim: str,
    passages: List[str],
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_new_tokens: int = 256,
    device: str = "auto"
) -> Tuple[bool, str]:
    """
    Check if a claim is consistent with evidence passages using LLM.
    
    Args:
        claim: Claim to check
        passages: List of evidence passages
        model: Pre-loaded model (if None, will use global model)
        tokenizer: Pre-loaded tokenizer (if None, will use global tokenizer)
        max_new_tokens: Maximum tokens to generate
        device: Device to run inference on
        
    Returns:
        Tuple of (is_consistent: bool, explanation: str)
    """
    # Use global model/tokenizer if not provided
    if model is None or tokenizer is None:
        if _model is None or _tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        model = _model
        tokenizer = _tokenizer
    
    if not passages:
        logger.warning("No passages provided for consistency check")
        return False, "No evidence available"
    
    # Create prompt
    prompt = create_prompt(claim, passages)
    
    # Auto-detect device if needed
    if device == "auto":
        device = detect_device()
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to device
        if hasattr(model, 'device'):
            model_device = next(model.parameters()).device
        else:
            model_device = device
        
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated part
        prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        new_text = generated_text[prompt_length:].strip()
        
        # Parse output
        is_consistent, explanation = parse_model_output(new_text)
        
        logger.debug(f"Consistency check: {is_consistent} - {explanation[:50]}...")
        
        return is_consistent, explanation
        
    except Exception as e:
        logger.error(f"Error during consistency check: {str(e)}")
        return False, f"Error: {str(e)}"


def check_consistency_batch(
    claims: List[str],
    passages_list: List[List[str]],
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "auto"
) -> List[Tuple[bool, str]]:
    """
    Check consistency for multiple claims in batches.
    
    Args:
        claims: List of claims to check
        passages_list: List of passage lists (one per claim)
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        max_new_tokens: Maximum tokens to generate
        batch_size: Number of claims to process in parallel
        device: Device to run inference on
        
    Returns:
        List of (is_consistent, explanation) tuples
    """
    results = []
    
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i + batch_size]
        batch_passages = passages_list[i:i + batch_size]
        
        batch_results = []
        for claim, passages in zip(batch_claims, batch_passages):
            result = check_consistency(
                claim, passages, model, tokenizer, max_new_tokens, device
            )
            batch_results.append(result)
        
        results.extend(batch_results)
        logger.info(f"Processed batch {i // batch_size + 1}/{(len(claims) + batch_size - 1) // batch_size}")
    
    return results
