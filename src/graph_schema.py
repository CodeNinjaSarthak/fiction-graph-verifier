"""
Core data structures for the Narrative Knowledge Graph.
Extended with Event support, trait axis/polarity, and relation world-types.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from enum import Enum


class EntityType(Enum):
    CHARACTER = "character"
    LOCATION = "location"
    OBJECT = "object"


class WorldType(Enum):
    """Determines inference behavior for missing relations."""
    CLOSED = "closed"  # Missing relation = FALSE (e.g., sibling_of)
    OPEN = "open"      # Missing relation = UNKNOWN (e.g., likes)


# Relation world-type registry
# Closed-world: if both entities are known but relation is absent, it's FALSE
# Open-world: if relation is absent, it's UNKNOWN
RELATION_WORLD_TYPES: Dict[str, WorldType] = {
    # Closed-world relations (family, formal relationships)
    "sibling_of": WorldType.CLOSED,
    "is_sibling_of": WorldType.CLOSED,
    "brother_of": WorldType.CLOSED,
    "is_brother_of": WorldType.CLOSED,
    "sister_of": WorldType.CLOSED,
    "is_sister_of": WorldType.CLOSED,
    "married_to": WorldType.CLOSED,
    "is_married_to": WorldType.CLOSED,
    "parent_of": WorldType.CLOSED,
    "is_parent_of": WorldType.CLOSED,
    "child_of": WorldType.CLOSED,
    "is_child_of": WorldType.CLOSED,
    "mother_of": WorldType.CLOSED,
    "is_mother_of": WorldType.CLOSED,
    "father_of": WorldType.CLOSED,
    "is_father_of": WorldType.CLOSED,
    "son_of": WorldType.CLOSED,
    "is_son_of": WorldType.CLOSED,
    "daughter_of": WorldType.CLOSED,
    "is_daughter_of": WorldType.CLOSED,

    # Open-world relations (can be unknown)
    "likes": WorldType.OPEN,
    "loves": WorldType.OPEN,
    "hates": WorldType.OPEN,
    "knows": WorldType.OPEN,
    "helps": WorldType.OPEN,
    "threatens": WorldType.OPEN,
    "fears": WorldType.OPEN,
    "respects": WorldType.OPEN,
    "trusts": WorldType.OPEN,
    "works_with": WorldType.OPEN,
    "lives_in": WorldType.OPEN,
    "owns": WorldType.OPEN,
}


def get_relation_world_type(predicate: str) -> WorldType:
    """Get the world type for a relation predicate."""
    pred_normalized = normalize_predicate(predicate)
    return RELATION_WORLD_TYPES.get(pred_normalized, WorldType.OPEN)


@dataclass
class Entity:
    """A node in the knowledge graph representing a named entity."""
    canonical_name: str          # Normalized name: "alice", "mad_hatter"
    display_name: str            # Original name: "Alice", "Mad Hatter"
    entity_type: EntityType
    aliases: Set[str] = field(default_factory=set)  # {"she", "the girl"}
    book_id: str = ""


@dataclass
class Event:
    """A node representing an action/event with temporal ordering."""
    event_id: str              # "E1", "E2", etc.
    verb: str                  # "fall", "meet", "drink"
    time: int                  # Monotonic sequence number
    chapter: int = 0           # Chapter number
    source_text: str = ""      # Original text snippet


@dataclass
class Trait:
    """A character trait with axis-based conflict detection."""
    entity_id: str             # Who has the trait
    trait_name: str            # "violent", "kind", etc.
    axis: str                  # "violence", "kindness", etc.
    polarity: int              # +1 or -1
    time_range: Optional[List[Optional[int]]] = None  # [start, end] or [start, null]
    confidence: Optional[float] = None


@dataclass
class Edge:
    """A typed edge in the knowledge graph."""
    source: str                # Entity or Event ID
    relation: str              # Relation type
    target: str                # Entity or Event ID
    world_type: WorldType = WorldType.OPEN
    negated: bool = False
    source_text: str = ""


# Keep Relation for backward compatibility (deprecated)
@dataclass
class Relation:
    """An edge in the knowledge graph (deprecated - use Edge instead)."""
    subject: str
    predicate: str
    object: str
    negated: bool = False
    source_text: str = ""
    book_id: str = ""


@dataclass
class WorldGraph:
    """
    Enhanced knowledge graph with event support.
    """
    entities: Dict[str, Entity] = field(default_factory=dict)
    events: Dict[str, Event] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    traits: Dict[str, List[Trait]] = field(default_factory=dict)  # entity_id -> [Trait]

    # Indices for fast lookup
    alias_to_canonical: Dict[str, str] = field(default_factory=dict)
    source_index: Dict[str, List[int]] = field(default_factory=dict)
    target_index: Dict[str, List[int]] = field(default_factory=dict)
    relation_index: Dict[str, List[int]] = field(default_factory=dict)

    # Backward compatibility: keep old relation storage
    relations: List[Relation] = field(default_factory=list)
    subject_index: Dict[str, List[int]] = field(default_factory=dict)
    object_index: Dict[str, List[int]] = field(default_factory=dict)
    predicate_index: Dict[str, List[int]] = field(default_factory=dict)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph and register its aliases."""
        self.entities[entity.canonical_name] = entity

        # Register canonical name as alias
        self.alias_to_canonical[entity.canonical_name.lower()] = entity.canonical_name
        self.alias_to_canonical[entity.display_name.lower()] = entity.canonical_name

        # Register all aliases
        for alias in entity.aliases:
            self.alias_to_canonical[alias.lower()] = entity.canonical_name

    def add_event(self, event: Event) -> None:
        """Add an event to the graph."""
        self.events[event.event_id] = event

    def add_edge(self, edge: Edge) -> int:
        """Add an edge and update indices."""
        idx = len(self.edges)
        self.edges.append(edge)

        # Update source index
        if edge.source not in self.source_index:
            self.source_index[edge.source] = []
        self.source_index[edge.source].append(idx)

        # Update target index
        if edge.target not in self.target_index:
            self.target_index[edge.target] = []
        self.target_index[edge.target].append(idx)

        # Update relation index
        rel_key = edge.relation.lower()
        if rel_key not in self.relation_index:
            self.relation_index[rel_key] = []
        self.relation_index[rel_key].append(idx)

        return idx

    def add_trait(self, trait: Trait) -> None:
        """Add a trait for an entity."""
        if trait.entity_id not in self.traits:
            self.traits[trait.entity_id] = []
        self.traits[trait.entity_id].append(trait)

    def add_relation(self, relation: Relation) -> int:
        """Add a relation to the graph and update indices (backward compat)."""
        idx = len(self.relations)
        self.relations.append(relation)

        # Update subject index
        if relation.subject not in self.subject_index:
            self.subject_index[relation.subject] = []
        self.subject_index[relation.subject].append(idx)

        # Update object index
        if relation.object not in self.object_index:
            self.object_index[relation.object] = []
        self.object_index[relation.object].append(idx)

        # Update predicate index
        pred_key = relation.predicate.lower()
        if pred_key not in self.predicate_index:
            self.predicate_index[pred_key] = []
        self.predicate_index[pred_key].append(idx)

        return idx

    def resolve_entity(self, name: str) -> Optional[str]:
        """Resolve an alias or name to its canonical entity name."""
        if not name:
            return None
        normalized = normalize_entity_name(name)
        return self.alias_to_canonical.get(normalized) or self.alias_to_canonical.get(name.lower())

    def get_edges_from(self, source: str) -> List[Edge]:
        """Get all edges originating from a node."""
        if source not in self.source_index:
            return []
        return [self.edges[i] for i in self.source_index[source]]

    def get_edges_to(self, target: str) -> List[Edge]:
        """Get all edges pointing to a node."""
        if target not in self.target_index:
            return []
        return [self.edges[i] for i in self.target_index[target]]

    def get_entity_traits(self, entity_id: str) -> List[Trait]:
        """Get all traits for an entity."""
        canonical = self.resolve_entity(entity_id)
        if not canonical:
            canonical = entity_id
        return self.traits.get(canonical, [])

    def get_event_by_time(self, time: int) -> Optional[Event]:
        """Get event at a specific time point."""
        for event in self.events.values():
            if event.time == time:
                return event
        return None

    def get_events_in_order(self) -> List[Event]:
        """Get all events sorted by time."""
        return sorted(self.events.values(), key=lambda e: e.time)

    def get_relations_for_subject(self, subject: str) -> List[Relation]:
        """Get all relations where entity is the subject (backward compat)."""
        canonical = self.resolve_entity(subject)
        if not canonical or canonical not in self.subject_index:
            return []
        return [self.relations[i] for i in self.subject_index[canonical]]

    def get_relations_for_object(self, obj: str) -> List[Relation]:
        """Get all relations where entity is the object (backward compat)."""
        canonical = self.resolve_entity(obj)
        if not canonical or canonical not in self.object_index:
            return []
        return [self.relations[i] for i in self.object_index[canonical]]

    def query_relation(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Relation]:
        """
        Query for relations matching the pattern (backward compat).
        None values act as wildcards.
        """
        # Resolve entity aliases
        subj_canonical = self.resolve_entity(subject) if subject else None
        obj_canonical = self.resolve_entity(obj) if obj else None

        # Start with smallest index set for efficiency
        candidate_indices: Optional[Set[int]] = None

        if subj_canonical and subj_canonical in self.subject_index:
            candidate_indices = set(self.subject_index[subj_canonical])

        if obj_canonical and obj_canonical in self.object_index:
            obj_indices = set(self.object_index[obj_canonical])
            if candidate_indices is None:
                candidate_indices = obj_indices
            else:
                candidate_indices &= obj_indices

        if predicate:
            pred_key = predicate.lower()
            if pred_key in self.predicate_index:
                pred_indices = set(self.predicate_index[pred_key])
                if candidate_indices is None:
                    candidate_indices = pred_indices
                else:
                    candidate_indices &= pred_indices
            else:
                # Predicate not found, return empty
                return []

        if candidate_indices is None:
            # No constraints specified, return all
            return list(self.relations)

        return [self.relations[i] for i in candidate_indices]

    def query_edges(
        self,
        source: Optional[str] = None,
        relation: Optional[str] = None,
        target: Optional[str] = None
    ) -> List[Edge]:
        """
        Query for edges matching the pattern.
        None values act as wildcards.
        """
        # Resolve entity aliases
        src_canonical = self.resolve_entity(source) if source else None
        tgt_canonical = self.resolve_entity(target) if target else None

        # Start with smallest index set for efficiency
        candidate_indices: Optional[Set[int]] = None

        if src_canonical and src_canonical in self.source_index:
            candidate_indices = set(self.source_index[src_canonical])
        elif source and source in self.source_index:
            # Try direct lookup (for event IDs)
            candidate_indices = set(self.source_index[source])

        if tgt_canonical and tgt_canonical in self.target_index:
            tgt_indices = set(self.target_index[tgt_canonical])
            if candidate_indices is None:
                candidate_indices = tgt_indices
            else:
                candidate_indices &= tgt_indices
        elif target and target in self.target_index:
            tgt_indices = set(self.target_index[target])
            if candidate_indices is None:
                candidate_indices = tgt_indices
            else:
                candidate_indices &= tgt_indices

        if relation:
            rel_key = relation.lower()
            if rel_key in self.relation_index:
                rel_indices = set(self.relation_index[rel_key])
                if candidate_indices is None:
                    candidate_indices = rel_indices
                else:
                    candidate_indices &= rel_indices
            else:
                return []

        if candidate_indices is None:
            return list(self.edges)

        return [self.edges[i] for i in candidate_indices]

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the graph."""
        unique_books = set()
        for entity in self.entities.values():
            if entity.book_id:
                unique_books.add(entity.book_id)
        for relation in self.relations:
            if relation.book_id:
                unique_books.add(relation.book_id)

        return {
            "total_entities": len(self.entities),
            "total_events": len(self.events),
            "total_edges": len(self.edges),
            "total_relations": len(self.relations),
            "total_aliases": len(self.alias_to_canonical),
            "total_traits": sum(len(t) for t in self.traits.values()),
            "unique_books": len(unique_books),
        }


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name to canonical form.

    Examples:
        "The Mad Hatter" -> "mad_hatter"
        "Alice" -> "alice"
        "Queen of Hearts" -> "queen_of_hearts"
    """
    if not name:
        return ""

    name = name.lower().strip()
    # Remove leading articles
    name = re.sub(r'^(the|a|an)\s+', '', name)
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove possessive 's
    name = re.sub(r"'s$", '', name)

    return name


def normalize_predicate(predicate: str) -> str:
    """
    Normalize predicate to canonical form.

    Examples:
        "fell into" -> "fell_into"
        "has tea with" -> "has_tea_with"
    """
    if not predicate:
        return ""

    pred = predicate.lower().strip()
    # Replace spaces with underscores
    pred = re.sub(r'\s+', '_', pred)

    return pred
