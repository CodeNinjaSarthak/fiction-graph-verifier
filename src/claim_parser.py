"""
Parse natural language claims into (subject, predicate, object) triples.
Uses LLM for semantic understanding of claims.
"""

import json
import re
import logging
from typing import List, Tuple, Dict, Any
import torch

from src.graph_schema import normalize_entity_name, normalize_predicate

logger = logging.getLogger(__name__)


# Prompt for parsing claims into triples
CLAIM_TO_TRIPLE_PROMPT = """Convert the following claim into (subject, predicate, object) triples.

Guidelines:
- subject: The main entity performing the action or having the attribute
- predicate: The action or relationship (use simple verb phrases like: enters, falls_into, has_tea_with, is_sibling_of, loves, has_trait)
- object: The entity receiving the action, or the attribute/location
- negated: Set to true ONLY if the claim explicitly says something does NOT happen

For personality/behavior claims like "X loves Y peacefully", extract BOTH the action AND any traits:
- {{"subject": "X", "predicate": "loves", "object": "Y", "negated": false}}
- {{"subject": "X", "predicate": "has_trait", "object": "peaceful", "negated": false}}

Claim: "{claim}"

Return ONLY a valid JSON array. No explanation.

Output format:
[
  {{"subject": "entity1", "predicate": "action", "object": "entity2", "negated": false}}
]

Triples:"""


# Few-shot examples for better parsing
CLAIM_PARSING_EXAMPLES = """
Example 1:
Claim: "Alice enters Wonderland by falling down a rabbit hole."
Triples:
[
  {{"subject": "Alice", "predicate": "enters", "object": "Wonderland", "negated": false}},
  {{"subject": "Alice", "predicate": "falls_into", "object": "rabbit hole", "negated": false}}
]

Example 2:
Claim: "The Cheshire Cat is Alice's brother."
Triples:
[
  {{"subject": "Cheshire Cat", "predicate": "is_sibling_of", "object": "Alice", "negated": false}}
]

Example 3:
Claim: "The Queen of Hearts loves gardening peacefully."
Triples:
[
  {{"subject": "Queen of Hearts", "predicate": "loves", "object": "gardening", "negated": false}},
  {{"subject": "Queen of Hearts", "predicate": "has_trait", "object": "peaceful", "negated": false}}
]

Example 4:
Claim: "She has tea with the Mad Hatter."
Triples:
[
  {{"subject": "She", "predicate": "has_tea_with", "object": "Mad Hatter", "negated": false}}
]
"""


def parse_llm_json(output: str) -> List[Dict[str, Any]]:
    """Parse JSON array from LLM output."""
    if not output:
        return []

    # Try to find JSON array in output
    match = re.search(r'\[[\s\S]*?\]', output)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole output
    try:
        result = json.loads(output.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    logger.debug(f"Failed to parse JSON from claim parsing output: {output[:200]}")
    return []


def generate_text(
    prompt: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 256
) -> str:
    """Generate text from LLM given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        if hasattr(model, 'device'):
            model_device = next(model.parameters()).device
        else:
            model_device = 'cpu'

        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        new_text = generated_text[len(prompt_decoded):].strip()

        return new_text

    except Exception as e:
        logger.error(f"Error generating text for claim parsing: {str(e)}")
        return ""


def resolve_pronouns(triples: List[Tuple[str, str, str, bool]], claim: str) -> List[Tuple[str, str, str, bool]]:
    """
    Attempt to resolve pronouns in triples by looking at the claim context.
    This is a simple heuristic - the graph verifier will do final resolution.
    """
    # Common pronouns that might need resolution
    pronouns = {'she', 'he', 'they', 'it', 'her', 'him', 'them'}

    resolved = []
    for subject, predicate, obj, negated in triples:
        # Check if subject is a pronoun
        if subject.lower() in pronouns:
            # Try to find a proper noun in the claim that could be the antecedent
            # This is a simple heuristic - look for capitalized words
            words = claim.split()
            for word in words:
                if word[0].isupper() and word.lower() not in pronouns:
                    # Found a potential antecedent
                    subject = word.rstrip('.,!?')
                    break

        resolved.append((subject, predicate, obj, negated))

    return resolved


def extract_traits_from_claim(claim: str) -> List[str]:
    """
    Extract adjectives/traits mentioned in a claim.
    Used for trait conflict detection.
    """
    # Common trait words
    trait_patterns = [
        r'\b(peaceful|peacefully)\b',
        r'\b(violent|violently)\b',
        r'\b(cruel|cruelly)\b',
        r'\b(kind|kindly)\b',
        r'\b(gentle|gently)\b',
        r'\b(angry|angrily)\b',
        r'\b(happy|happily)\b',
        r'\b(loving|lovingly)\b',
        r'\b(aggressive|aggressively)\b',
    ]

    traits = []
    claim_lower = claim.lower()
    for pattern in trait_patterns:
        match = re.search(pattern, claim_lower)
        if match:
            # Normalize to base form
            trait = match.group(1)
            if trait.endswith('ly'):
                trait = trait[:-2]
            if trait.endswith('ful'):
                trait = trait[:-3]
            traits.append(trait)

    return traits


def parse_claim_to_triples(
    claim: str,
    model: Any,
    tokenizer: Any
) -> List[Tuple[str, str, str, bool]]:
    """
    Parse a natural language claim into structured triples.

    Args:
        claim: Natural language claim string
        model: Loaded LLM model
        tokenizer: Loaded tokenizer

    Returns:
        List of (subject, predicate, object, negated) tuples
    """
    if not claim.strip():
        return []

    prompt = CLAIM_TO_TRIPLE_PROMPT.format(claim=claim)
    output = generate_text(prompt, model, tokenizer, max_new_tokens=256)

    triples = []
    parsed = parse_llm_json(output)

    for item in parsed:
        if not isinstance(item, dict):
            continue

        subject = item.get('subject', '').strip()
        predicate = item.get('predicate', '').strip()
        obj = item.get('object', '').strip()

        if not subject or not predicate or not obj:
            continue

        negated = item.get('negated', False)
        if isinstance(negated, str):
            negated = negated.lower() == 'true'

        # Normalize
        subject_norm = normalize_entity_name(subject)
        predicate_norm = normalize_predicate(predicate)
        obj_norm = normalize_entity_name(obj)

        triples.append((subject_norm, predicate_norm, obj_norm, bool(negated)))

    # Try to resolve pronouns
    triples = resolve_pronouns(triples, claim)

    # Extract traits from the claim for conflict detection
    traits = extract_traits_from_claim(claim)

    logger.debug(f"Parsed claim '{claim[:50]}...' into {len(triples)} triples")

    return triples
