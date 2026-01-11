"""
Parse natural language claims into event-structured queries.
Supports subgraph pattern matching for verification.

Multi-provider support:
- gemini: Google Gemini API (reliable, requires API key)
- local: Local HuggingFace models (offline, may be less accurate)
"""

import json
import re
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import torch

from src.graph_schema import normalize_entity_name, normalize_predicate

logger = logging.getLogger(__name__)

# Gemini API (lazy import to avoid dependency if not used)
_genai = None


# ============ Query Data Structures ============


@dataclass
class EventQuery:
    """Represents a query pattern for event-based matching."""

    verb: str  # The action verb
    agent: Optional[str] = None  # Who performs the action
    patient: Optional[str] = None  # Who receives the action
    destination: Optional[str] = None  # Where it leads to
    location: Optional[str] = None  # Where it happens
    instrument: Optional[str] = None  # With what
    negated: bool = False


@dataclass
class RelationQuery:
    """Represents a query for entity-entity relations."""

    subject: str
    predicate: str
    object: str
    negated: bool = False


@dataclass
class TraitQuery:
    """Represents a query about character traits."""

    entity: str
    trait: str
    negated: bool = False


@dataclass
class ClaimQuery:
    """Parsed claim containing multiple query patterns."""

    event_queries: List[EventQuery] = field(default_factory=list)
    relation_queries: List[RelationQuery] = field(default_factory=list)
    trait_queries: List[TraitQuery] = field(default_factory=list)
    raw_claim: str = ""


# ============ Prompts ============

CLAIM_TO_QUERY_PROMPT = """Parse this claim into structured queries for verification.

Guidelines:
1. EVENT patterns: Actions with participants
   - verb: The action (fall, meet, drink, chase, etc.)
   - agent: Who performs the action
   - patient: Who receives the action (optional)
   - destination: Where the action leads to (optional)
   - location: Where the action happens (optional)

2. RELATION patterns: Relationships between entities
   - subject, predicate, object format
   - Use predicates like: sibling_of, parent_of, married_to, loves, hates, knows

3. TRAIT patterns: Character personality traits
   - entity: Who has the trait
   - trait: The trait name (peaceful, violent, kind, cruel, etc.)

Examples:

Claim: "Alice fell into a rabbit hole"
{{"events": [{{"verb": "fall", "agent": "Alice", "destination": "rabbit_hole"}}], "relations": [], "traits": []}}

Claim: "The Cheshire Cat is Alice's brother"
{{"events": [], "relations": [{{"subject": "Cheshire Cat", "predicate": "sibling_of", "object": "Alice"}}], "traits": []}}

Claim: "The Queen of Hearts loves gardening peacefully"
{{"events": [], "relations": [{{"subject": "Queen of Hearts", "predicate": "loves", "object": "gardening"}}], "traits": [{{"entity": "Queen of Hearts", "trait": "peaceful"}}]}}

Claim: "Alice met the Mad Hatter at the tea party"
{{"events": [{{"verb": "meet", "agent": "Alice", "patient": "Mad Hatter", "location": "tea_party"}}], "relations": [], "traits": []}}

Claim: "The Queen did not kill anyone"
{{"events": [{{"verb": "kill", "agent": "Queen", "negated": true}}], "relations": [], "traits": []}}

Claim: "{claim}"

Return ONLY valid JSON. No explanation:"""


# Legacy prompt for backward compatibility
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


# ============ JSON Parsing ============


def parse_llm_json_object(output: str) -> Dict[str, Any]:
    """Parse JSON object from LLM output."""
    if not output:
        return {}

    # Try to find JSON object in output
    match = re.search(r"\{[\s\S]*\}", output)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole output
    try:
        result = json.loads(output.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    logger.debug(f"Failed to parse JSON object from output: {output[:200]}")
    return {}


def parse_llm_json(output: str) -> List[Dict[str, Any]]:
    """Parse JSON array from LLM output."""
    if not output:
        return []

    # Try to find JSON array in output
    match = re.search(r"\[[\s\S]*?\]", output)
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


# ============ Text Generation ============


def generate_text_local(
    prompt: str, model: Any, tokenizer: Any, max_new_tokens: int = 512
) -> str:
    """Generate text from local HuggingFace LLM given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        if hasattr(model, "device"):
            model_device = next(model.parameters()).device
        else:
            model_device = "cpu"

        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )
        new_text = generated_text[len(prompt_decoded) :].strip()

        return new_text

    except Exception as e:
        logger.error(f"Error generating text with local model: {str(e)}")
        return ""


def generate_text_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Generate structured JSON using Gemini API.

    Args:
        prompt: The prompt to send to Gemini
        model_name: Gemini model name (default: gemini-1.5-flash)

    Returns:
        Generated text (JSON string)
    """
    global _genai

    # Lazy import - try new google.genai first, fall back to deprecated google.generativeai
    if _genai is None:
        try:
            from google import genai

            _genai = genai
        except ImportError:
            try:
                import google.generativeai as genai

                _genai = genai
            except ImportError:
                raise ImportError(
                    "Neither google-genai nor google-generativeai installed. "
                    "Run: pip install google-genai"
                )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get a key from https://aistudio.google.com/apikey"
        )

    try:
        # New google.genai API
        if hasattr(_genai, "Client"):
            client = _genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0,
                    "max_output_tokens": 512,
                    "response_mime_type": "application/json",
                },
            )
            return response.text
        else:
            # Old google.generativeai API (deprecated)
            _genai.configure(api_key=api_key)
            model = _genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=_genai.GenerationConfig(
                    temperature=0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            return response.text

    except Exception as e:
        logger.error(f"Error generating text with Gemini: {str(e)}")
        raise


# Legacy alias for backward compatibility
def generate_text(
    prompt: str, model: Any, tokenizer: Any, max_new_tokens: int = 256
) -> str:
    """Legacy wrapper - calls generate_text_local."""
    return generate_text_local(prompt, model, tokenizer, max_new_tokens)


# ============ Query Parsing ============


def parse_claim_to_queries(
    claim: str,
    model: Any = None,
    tokenizer: Any = None,
    provider: str = "gemini",
    gemini_model: str = "gemini-1.5-flash",
) -> ClaimQuery:
    """
    Parse a natural language claim into structured queries.

    Args:
        claim: Natural language claim string
        model: HuggingFace model (required for local provider)
        tokenizer: Tokenizer (required for local provider)
        provider: "gemini" or "local"
        gemini_model: Gemini model name (when using gemini provider)

    Returns:
        ClaimQuery with event, relation, and trait queries
    """
    if not claim.strip():
        return ClaimQuery(raw_claim=claim)

    prompt = CLAIM_TO_QUERY_PROMPT.format(claim=claim)

    # Generate output based on provider
    try:
        if provider == "gemini":
            output = generate_text_gemini(prompt, gemini_model)
        elif provider == "local":
            if not model or not tokenizer:
                raise ValueError("Local provider requires model and tokenizer")
            output = generate_text_local(prompt, model, tokenizer, max_new_tokens=512)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        logger.error(f"Claim parsing failed with {provider}: {e}")
        return ClaimQuery(raw_claim=claim)

    parsed = parse_llm_json_object(output)

    event_queries = []
    for e in parsed.get("events", []):
        if not isinstance(e, dict):
            continue
        verb = e.get("verb", "")
        if not verb:
            continue

        event_queries.append(
            EventQuery(
                verb=normalize_predicate(verb),
                agent=normalize_entity_name(e.get("agent", "")) or None,
                patient=normalize_entity_name(e.get("patient", "")) or None,
                destination=normalize_entity_name(e.get("destination", "")) or None,
                location=normalize_entity_name(e.get("location", "")) or None,
                instrument=normalize_entity_name(e.get("instrument", "")) or None,
                negated=bool(e.get("negated", False)),
            )
        )

    relation_queries = []
    for r in parsed.get("relations", []):
        if not isinstance(r, dict):
            continue
        subject = r.get("subject", "")
        predicate = r.get("predicate", "")
        obj = r.get("object", "")
        if not subject or not predicate or not obj:
            continue

        relation_queries.append(
            RelationQuery(
                subject=normalize_entity_name(subject),
                predicate=normalize_predicate(predicate),
                object=normalize_entity_name(obj),
                negated=bool(r.get("negated", False)),
            )
        )

    trait_queries = []
    for t in parsed.get("traits", []):
        if not isinstance(t, dict):
            continue
        entity = t.get("entity", "")
        trait = t.get("trait", "")
        if not entity or not trait:
            continue

        trait_queries.append(
            TraitQuery(
                entity=normalize_entity_name(entity),
                trait=trait.lower().strip(),
                negated=bool(t.get("negated", False)),
            )
        )

    # Also extract traits from the claim text directly
    additional_traits = extract_traits_from_claim(claim)
    # Find the main subject from event or relation queries
    main_subject = None
    if event_queries and event_queries[0].agent:
        main_subject = event_queries[0].agent
    elif relation_queries:
        main_subject = relation_queries[0].subject

    if main_subject:
        for trait in additional_traits:
            # Check if this trait is already in trait_queries
            if not any(
                tq.trait == trait and tq.entity == main_subject for tq in trait_queries
            ):
                trait_queries.append(
                    TraitQuery(entity=main_subject, trait=trait, negated=False)
                )

    return ClaimQuery(
        event_queries=event_queries,
        relation_queries=relation_queries,
        trait_queries=trait_queries,
        raw_claim=claim,
    )


# ============ Legacy Functions (Backward Compatibility) ============


def resolve_pronouns(
    triples: List[Tuple[str, str, str, bool]], claim: str
) -> List[Tuple[str, str, str, bool]]:
    """
    Attempt to resolve pronouns in triples by looking at the claim context.
    This is a simple heuristic - the graph verifier will do final resolution.
    """
    # Common pronouns that might need resolution
    pronouns = {"she", "he", "they", "it", "her", "him", "them"}

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
                    subject = word.rstrip(".,!?")
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
        r"\b(peaceful|peacefully)\b",
        r"\b(violent|violently)\b",
        r"\b(cruel|cruelly)\b",
        r"\b(kind|kindly)\b",
        r"\b(gentle|gently)\b",
        r"\b(angry|angrily)\b",
        r"\b(happy|happily)\b",
        r"\b(loving|lovingly)\b",
        r"\b(aggressive|aggressively)\b",
        r"\b(brave|bravely)\b",
        r"\b(cowardly)\b",
        r"\b(honest|honestly)\b",
        r"\b(dishonest|dishonestly)\b",
        r"\b(calm|calmly)\b",
    ]

    traits = []
    claim_lower = claim.lower()
    for pattern in trait_patterns:
        match = re.search(pattern, claim_lower)
        if match:
            # Normalize to base form
            trait = match.group(1)
            if trait.endswith("ly"):
                trait = trait[:-2]
            if trait.endswith("ful"):
                trait = trait[:-3]
            traits.append(trait)

    return traits


def parse_claim_to_triples(
    claim: str,
    model: Any = None,
    tokenizer: Any = None,
    provider: str = "gemini",
    gemini_model: str = "gemini-1.5-flash",
) -> List[Tuple[str, str, str, bool]]:
    """
    Parse a natural language claim into structured triples (legacy interface).

    Args:
        claim: Natural language claim string
        model: Loaded LLM model (required for local provider)
        tokenizer: Loaded tokenizer (required for local provider)
        provider: "gemini" or "local"
        gemini_model: Gemini model name (when using gemini provider)

    Returns:
        List of (subject, predicate, object, negated) tuples
    """
    if not claim.strip():
        return []

    prompt = CLAIM_TO_TRIPLE_PROMPT.format(claim=claim)

    # Generate output based on provider
    try:
        if provider == "gemini":
            output = generate_text_gemini(prompt, gemini_model)
        elif provider == "local":
            if not model or not tokenizer:
                raise ValueError("Local provider requires model and tokenizer")
            output = generate_text_local(prompt, model, tokenizer, max_new_tokens=256)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        logger.error(f"Triple parsing failed with {provider}: {e}")
        return []

    triples = []
    parsed = parse_llm_json(output)

    for item in parsed:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject", "").strip()
        predicate = item.get("predicate", "").strip()
        obj = item.get("object", "").strip()

        if not subject or not predicate or not obj:
            continue

        negated = item.get("negated", False)
        if isinstance(negated, str):
            negated = negated.lower() == "true"

        # Normalize
        subject_norm = normalize_entity_name(subject)
        predicate_norm = normalize_predicate(predicate)
        obj_norm = normalize_entity_name(obj)

        triples.append((subject_norm, predicate_norm, obj_norm, bool(negated)))

    # Try to resolve pronouns
    triples = resolve_pronouns(triples, claim)

    logger.debug(f"Parsed claim '{claim[:50]}...' into {len(triples)} triples")

    return triples


def queries_to_triples(query: ClaimQuery) -> List[Tuple[str, str, str, bool]]:
    """
    Convert ClaimQuery to legacy triple format.
    Useful for backward compatibility.

    WARNING: This function is NOT compatible with reified event graphs.
    EventQuery objects should NOT be converted to triples - they must be
    verified using verify_event_query() which matches against the event graph
    structure: entity --role--> event --role--> entity

    This function should only be used for legacy code paths that expect
    simple (subject, predicate, object) triples.
    """
    triples = []

    # Convert event queries to triples
    for eq in query.event_queries:
        if eq.agent and eq.destination:
            triples.append((eq.agent, eq.verb, eq.destination, eq.negated))
        elif eq.agent and eq.patient:
            triples.append((eq.agent, eq.verb, eq.patient, eq.negated))
        elif eq.agent and eq.location:
            triples.append((eq.agent, eq.verb, eq.location, eq.negated))

    # Convert relation queries to triples
    for rq in query.relation_queries:
        triples.append((rq.subject, rq.predicate, rq.object, rq.negated))

    # Convert trait queries to triples
    for tq in query.trait_queries:
        triples.append((tq.entity, "has_trait", tq.trait, tq.negated))

    return triples
