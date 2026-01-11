"""
Load WorldGraph from compiled JSON files.
The world is built offline (in Colab) and loaded at runtime.

This replaces LLM-based extraction with file-based loading.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.graph_schema import (
    Entity, EntityType, Event, Edge, Trait, WorldGraph,
    WorldType, get_relation_world_type,
    normalize_entity_name, normalize_predicate
)
from src.trait_ontology import get_trait_axis_name

logger = logging.getLogger(__name__)


def _parse_polarity_conf(polarity_conf):
    """Parse polarity and confidence from trait data."""
    if isinstance(polarity_conf, list):
        polarity = polarity_conf[0] if polarity_conf else 1
        confidence = polarity_conf[1] if len(polarity_conf) > 1 else None
        time_range = polarity_conf[2] if len(polarity_conf) > 2 else None
    else:
        polarity = polarity_conf
        confidence = None
        time_range = None
    return polarity, confidence, time_range


def load_graph_from_world(world_dir: str = "world") -> WorldGraph:
    """
    Load a WorldGraph from JSON files in the world directory.

    This is the main entry point for loading a pre-compiled world.
    The world is built by build_world_colab.py on GPU and downloaded.

    Expected files:
        - world/entities.json
        - world/events.json
        - world/edges.json
        - world/traits.json
        - world/chapters.json (optional metadata)

    Args:
        world_dir: Path to directory containing JSON files

    Returns:
        Populated WorldGraph

    Raises:
        FileNotFoundError: If required files are missing
    """
    graph = WorldGraph()
    world_path = Path(world_dir)

    if not world_path.exists():
        raise FileNotFoundError(f"World directory not found: {world_dir}")

    # Load entities
    entities_file = world_path / "entities.json"
    if entities_file.exists():
        with open(entities_file, "r", encoding="utf-8") as f:
            entities_data = json.load(f)

        for entity_id, entity_dict in entities_data.items():
            # Parse entity type
            type_str = entity_dict.get("type", "object").upper()
            try:
                entity_type = EntityType[type_str]
            except KeyError:
                entity_type = EntityType.OBJECT

            # Create entity
            entity = Entity(
                canonical_name=entity_id,
                display_name=entity_dict.get("display_name", entity_id),
                entity_type=entity_type,
                aliases=set(entity_dict.get("aliases", []))
            )
            graph.add_entity(entity)

        logger.info(f"Loaded {len(graph.entities)} entities from {entities_file}")
    else:
        logger.warning(f"Entities file not found: {entities_file}")

    # Load events
    events_file = world_path / "events.json"
    if events_file.exists():
        with open(events_file, "r", encoding="utf-8") as f:
            events_data = json.load(f)

        for event_id, event_dict in events_data.items():
            event = Event(
                event_id=event_id,
                verb=event_dict.get("verb", ""),
                time=event_dict.get("time", 0),
                chapter=event_dict.get("chapter", 0),
                source_text=event_dict.get("source_text", "")
            )
            graph.add_event(event)

        logger.info(f"Loaded {len(graph.events)} events from {events_file}")
    else:
        logger.warning(f"Events file not found: {events_file}")

    # Load edges
    edges_file = world_path / "edges.json"
    if edges_file.exists():
        with open(edges_file, "r", encoding="utf-8") as f:
            edges_data = json.load(f)

        for edge_entry in edges_data:
            # Handle both list format [src, rel, tgt] and dict format
            if isinstance(edge_entry, list):
                source = edge_entry[0]
                relation = edge_entry[1]
                target = edge_entry[2]
                negated = edge_entry[3] if len(edge_entry) > 3 else False
            else:
                source = edge_entry.get("source", "")
                relation = edge_entry.get("relation", "")
                target = edge_entry.get("target", "")
                negated = edge_entry.get("negated", False)

            # Determine world type for this relation
            world_type = get_relation_world_type(relation)

            edge = Edge(
                source=source,
                relation=normalize_predicate(relation),
                target=target,
                world_type=world_type,
                negated=bool(negated)
            )
            graph.add_edge(edge)

        logger.info(f"Loaded {len(graph.edges)} edges from {edges_file}")
    else:
        logger.warning(f"Edges file not found: {edges_file}")

    # Load traits
    traits_file = world_path / "traits.json"
    if traits_file.exists():
        with open(traits_file, "r", encoding="utf-8") as f:
            traits_data = json.load(f)

        trait_count = 0
        for entity_id, entity_traits in traits_data.items():
            for key, value in entity_traits.items():
                # Handle nested structure: {axis: {trait: [polarity, conf]}}
                # or flat structure: {trait: [polarity, conf]}
                if isinstance(value, dict):
                    # Nested: key is axis_name, value is {trait_name: [polarity, conf]}
                    axis = key
                    for trait_name, polarity_conf in value.items():
                        polarity, confidence, time_range = _parse_polarity_conf(polarity_conf)
                        trait = Trait(
                            entity_id=entity_id,
                            trait_name=trait_name,
                            axis=axis,
                            polarity=polarity,
                            confidence=confidence,
                            time_range=time_range
                        )
                        graph.add_trait(trait)
                        trait_count += 1
                else:
                    # Flat: key is trait_name, value is [polarity, conf]
                    trait_name = key
                    polarity, confidence, time_range = _parse_polarity_conf(value)
                    axis = get_trait_axis_name(trait_name) or "unknown"
                    trait = Trait(
                        entity_id=entity_id,
                        trait_name=trait_name,
                        axis=axis,
                        polarity=polarity,
                        confidence=confidence,
                        time_range=time_range
                    )
                    graph.add_trait(trait)
                    trait_count += 1

        logger.info(f"Loaded {trait_count} traits for {len(traits_data)} entities")
    else:
        logger.warning(f"Traits file not found: {traits_file}")

    # Load chapters metadata (optional)
    chapters_file = world_path / "chapters.json"
    if chapters_file.exists():
        with open(chapters_file, "r", encoding="utf-8") as f:
            chapters_data = json.load(f)
        logger.info(f"Loaded {len(chapters_data)} chapters metadata")

    # Log final stats
    stats = graph.get_stats()
    logger.info(f"World loaded: {stats}")

    return graph


def verify_world_integrity(graph: WorldGraph) -> list:
    """
    Verify the integrity of a loaded world graph.

    Checks for:
    - Dangling edge references (entities/events that don't exist)
    - Trait axis consistency
    - Event time ordering

    Args:
        graph: The loaded WorldGraph

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check for dangling edge references
    for i, edge in enumerate(graph.edges):
        # Check source exists (could be entity or event)
        if edge.source not in graph.entities and edge.source not in graph.events:
            warnings.append(f"Edge {i}: source '{edge.source}' not found in entities or events")

        # Check target exists
        if edge.target not in graph.entities and edge.target not in graph.events:
            # Target might be a literal value (like "gardening"), so just warn
            if not edge.target.startswith("E"):
                pass  # Acceptable for non-event targets
            else:
                warnings.append(f"Edge {i}: target event '{edge.target}' not found")

    # Check event time ordering
    events_ordered = graph.get_events_in_order()
    for i in range(1, len(events_ordered)):
        if events_ordered[i].time <= events_ordered[i-1].time:
            if events_ordered[i].time == events_ordered[i-1].time:
                pass  # Same time is OK (simultaneous events)
            else:
                warnings.append(
                    f"Event time ordering issue: {events_ordered[i-1].event_id} (t={events_ordered[i-1].time}) "
                    f"should come before {events_ordered[i].event_id} (t={events_ordered[i].time})"
                )

    # Check for trait axis conflicts within same entity
    from src.trait_ontology import are_traits_conflicting
    for entity_id, traits in graph.traits.items():
        for i, t1 in enumerate(traits):
            for t2 in traits[i+1:]:
                if are_traits_conflicting(t1.trait_name, t2.trait_name):
                    warnings.append(
                        f"Entity '{entity_id}' has conflicting traits: "
                        f"'{t1.trait_name}' and '{t2.trait_name}'"
                    )

    return warnings


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

    # Merge events
    for event_id, event in source.events.items():
        if event_id not in target.events:
            target.add_event(event)

    # Merge edges
    existing_edges = {(e.source, e.relation, e.target) for e in target.edges}
    for edge in source.edges:
        key = (edge.source, edge.relation, edge.target)
        if key not in existing_edges:
            target.add_edge(edge)
            existing_edges.add(key)

    # Merge traits
    for entity_id, traits in source.traits.items():
        for trait in traits:
            # Check if trait already exists
            existing = target.traits.get(entity_id, [])
            if not any(t.trait_name == trait.trait_name for t in existing):
                target.add_trait(trait)

    logger.info(f"Merged graphs: target now has {len(target.entities)} entities, "
               f"{len(target.events)} events, {len(target.edges)} edges")


def save_graph_to_world(graph: WorldGraph, world_dir: str = "world") -> None:
    """
    Save a WorldGraph to JSON files.
    Useful for debugging or creating test worlds.

    Args:
        graph: The WorldGraph to save
        world_dir: Directory to save files to
    """
    world_path = Path(world_dir)
    world_path.mkdir(parents=True, exist_ok=True)

    # Save entities
    entities_data = {}
    for entity_id, entity in graph.entities.items():
        entities_data[entity_id] = {
            "id": entity.canonical_name,
            "type": entity.entity_type.value,
            "display_name": entity.display_name,
            "aliases": list(entity.aliases)
        }

    with open(world_path / "entities.json", "w", encoding="utf-8") as f:
        json.dump(entities_data, f, indent=2)

    # Save events
    events_data = {}
    for event_id, event in graph.events.items():
        events_data[event_id] = {
            "id": event.event_id,
            "verb": event.verb,
            "time": event.time,
            "chapter": event.chapter
        }

    with open(world_path / "events.json", "w", encoding="utf-8") as f:
        json.dump(events_data, f, indent=2)

    # Save edges
    edges_data = []
    for edge in graph.edges:
        edges_data.append([edge.source, edge.relation, edge.target])

    with open(world_path / "edges.json", "w", encoding="utf-8") as f:
        json.dump(edges_data, f, indent=2)

    # Save traits
    traits_data = {}
    for entity_id, traits in graph.traits.items():
        traits_data[entity_id] = {}
        for trait in traits:
            traits_data[entity_id][trait.trait_name] = [trait.polarity, trait.confidence]

    with open(world_path / "traits.json", "w", encoding="utf-8") as f:
        json.dump(traits_data, f, indent=2)

    logger.info(f"Saved world to {world_dir}")
