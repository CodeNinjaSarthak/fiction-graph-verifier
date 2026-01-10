"""
LLM-based extraction of entities and relations from book text.
Builds a WorldGraph from text chunks.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
import torch

from src.graph_schema import (
    Entity, EntityType, Relation, WorldGraph,
    normalize_entity_name, normalize_predicate
)

logger = logging.getLogger(__name__)


# Prompt for extracting entities from text
ENTITY_EXTRACTION_PROMPT = """Extract all named entities from the following text passage.

For each entity, identify:
1. name: The entity's name as it appears in text
2. type: One of CHARACTER, LOCATION, or OBJECT
3. aliases: Any alternate names or pronouns used for this entity

Return ONLY a valid JSON array. No explanation or other text.

Text:
{chunk_text}

Output format example:
[
  {{"name": "Alice", "type": "CHARACTER", "aliases": ["she", "the girl"]}},
  {{"name": "Wonderland", "type": "LOCATION", "aliases": []}},
  {{"name": "rabbit hole", "type": "LOCATION", "aliases": ["the hole"]}}
]

Entities:"""


# Prompt for extracting relations from text
RELATION_EXTRACTION_PROMPT = """Extract relationships and events between entities from this text.

Known entities in this passage: {entity_list}

For each relationship, identify:
1. subject: The entity performing the action
2. predicate: The action or relationship (use simple verb phrases like: falls_into, has_tea_with, meets, threatens, orders)
3. object: The entity receiving the action, or an attribute/trait
4. negated: true only if the text explicitly says something does NOT happen

Also extract character traits as: {{"subject": "Character", "predicate": "has_trait", "object": "trait_name"}}
Example traits: violent, cruel, kind, curious, angry, peaceful

Return ONLY a valid JSON array. No explanation.

Text:
{chunk_text}

Output format example:
[
  {{"subject": "Alice", "predicate": "falls_into", "object": "rabbit hole", "negated": false}},
  {{"subject": "Queen of Hearts", "predicate": "has_trait", "object": "violent", "negated": false}},
  {{"subject": "Queen of Hearts", "predicate": "orders", "object": "execution", "negated": false}}
]

Relations:"""


def parse_llm_json(output: str) -> List[Dict[str, Any]]:
    """
    Parse JSON array from LLM output.
    Handles cases where LLM includes extra text around JSON.
    """
    if not output:
        return []

    # Try to find JSON array in output
    # First try to find array brackets
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

    logger.debug(f"Failed to parse JSON from LLM output: {output[:200]}")
    return []


def generate_text(
    prompt: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 512
) -> str:
    """Generate text from LLM given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        # Move to model's device
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

        # Extract only newly generated text
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        new_text = generated_text[len(prompt_decoded):].strip()

        return new_text

    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return ""


def extract_entities(
    chunk_text: str,
    model: Any,
    tokenizer: Any,
    existing_entities: Optional[Dict[str, Entity]] = None
) -> List[Entity]:
    """
    Extract entities from a text chunk using LLM.

    Args:
        chunk_text: Text passage to extract entities from
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        existing_entities: Already known entities for context

    Returns:
        List of Entity objects
    """
    if not chunk_text.strip():
        return []

    # Truncate chunk if too long
    chunk_text = chunk_text[:2000]

    prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk_text)
    output = generate_text(prompt, model, tokenizer, max_new_tokens=512)

    entities = []
    parsed = parse_llm_json(output)

    for item in parsed:
        if not isinstance(item, dict):
            continue

        name = item.get('name', '').strip()
        if not name:
            continue

        type_str = item.get('type', 'OBJECT').upper()
        try:
            entity_type = EntityType[type_str]
        except KeyError:
            entity_type = EntityType.OBJECT

        aliases = set(item.get('aliases', []))
        canonical = normalize_entity_name(name)

        entity = Entity(
            canonical_name=canonical,
            display_name=name,
            entity_type=entity_type,
            aliases=aliases
        )
        entities.append(entity)

    logger.debug(f"Extracted {len(entities)} entities from chunk")
    return entities


def extract_relations(
    chunk_text: str,
    entities: List[Entity],
    model: Any,
    tokenizer: Any
) -> List[Relation]:
    """
    Extract relations from a text chunk using LLM.

    Args:
        chunk_text: Text passage to extract relations from
        entities: List of known entities in this chunk
        model: Loaded LLM model
        tokenizer: Loaded tokenizer

    Returns:
        List of Relation objects
    """
    if not chunk_text.strip():
        return []

    # Create entity list string for prompt
    entity_names = [e.display_name for e in entities]
    entity_list = ", ".join(entity_names) if entity_names else "No specific entities identified"

    # Truncate chunk if too long
    chunk_text = chunk_text[:2000]

    prompt = RELATION_EXTRACTION_PROMPT.format(
        chunk_text=chunk_text,
        entity_list=entity_list
    )
    output = generate_text(prompt, model, tokenizer, max_new_tokens=512)

    relations = []
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

        relation = Relation(
            subject=normalize_entity_name(subject),
            predicate=normalize_predicate(predicate),
            object=normalize_entity_name(obj),
            negated=bool(negated),
            source_text=chunk_text[:200]
        )
        relations.append(relation)

    logger.debug(f"Extracted {len(relations)} relations from chunk")
    return relations


def build_graph_from_book(
    chunks: List[str],
    book_id: str,
    model: Any,
    tokenizer: Any,
    progress_callback: Optional[callable] = None
) -> WorldGraph:
    """
    Build a WorldGraph from book text chunks.

    Args:
        chunks: List of text chunks from the book
        book_id: Identifier for the book
        model: LLM model for extraction
        tokenizer: Tokenizer for the model
        progress_callback: Optional callback for progress updates

    Returns:
        WorldGraph with extracted entities and relations
    """
    graph = WorldGraph()
    total_chunks = len(chunks)

    logger.info(f"Building graph from {total_chunks} chunks for book {book_id}")

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total_chunks)

        # Extract entities
        entities = extract_entities(chunk, model, tokenizer, graph.entities)

        # Add entities to graph with book_id
        for entity in entities:
            entity.book_id = book_id
            # Check if entity already exists
            if entity.canonical_name in graph.entities:
                # Merge aliases
                existing = graph.entities[entity.canonical_name]
                existing.aliases.update(entity.aliases)
                # Re-register aliases
                for alias in entity.aliases:
                    graph.alias_to_canonical[alias.lower()] = entity.canonical_name
            else:
                graph.add_entity(entity)

        # Extract relations
        relations = extract_relations(chunk, entities, model, tokenizer)

        # Add relations to graph
        for relation in relations:
            relation.book_id = book_id
            graph.add_relation(relation)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{total_chunks} chunks, "
                       f"{len(graph.entities)} entities, {len(graph.relations)} relations")

    logger.info(f"Built graph for book {book_id}: "
               f"{len(graph.entities)} entities, {len(graph.relations)} relations")

    return graph


def merge_graphs(target: WorldGraph, source: WorldGraph) -> None:
    """
    Merge source graph into target graph.
    Handles entity deduplication and alias merging.

    Args:
        target: Graph to merge into (modified in place)
        source: Graph to merge from
    """
    # Merge entities
    for canonical, entity in source.entities.items():
        if canonical in target.entities:
            # Merge aliases
            target.entities[canonical].aliases.update(entity.aliases)
            for alias in entity.aliases:
                target.alias_to_canonical[alias.lower()] = canonical
        else:
            target.add_entity(entity)

    # Merge relations
    for relation in source.relations:
        target.add_relation(relation)

    logger.info(f"Merged graphs: target now has {len(target.entities)} entities, "
               f"{len(target.relations)} relations")
