"""
Core data structures for the Narrative Knowledge Graph.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from enum import Enum


class EntityType(Enum):
    CHARACTER = "character"
    LOCATION = "location"
    OBJECT = "object"


@dataclass
class Entity:
    """A node in the knowledge graph representing a named entity."""
    canonical_name: str          # Normalized name: "alice", "mad_hatter"
    display_name: str            # Original name: "Alice", "Mad Hatter"
    entity_type: EntityType
    aliases: Set[str] = field(default_factory=set)  # {"she", "the girl"}
    book_id: str = ""


@dataclass
class Relation:
    """An edge in the knowledge graph representing a relationship or event."""
    subject: str                 # Canonical entity name
    predicate: str               # Action/relationship: "fell_into", "has_tea_with"
    object: str                  # Canonical entity name or attribute
    negated: bool = False        # True if "does NOT do X"
    source_text: str = ""        # Original text snippet for reference
    book_id: str = ""


@dataclass
class WorldGraph:
    """
    The complete knowledge graph for one or more books.
    Stores entities, relations, and indices for fast lookup.
    """
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    alias_to_canonical: Dict[str, str] = field(default_factory=dict)
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

    def add_relation(self, relation: Relation) -> int:
        """Add a relation to the graph and update indices. Returns relation index."""
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

    def get_relations_for_subject(self, subject: str) -> List[Relation]:
        """Get all relations where entity is the subject."""
        canonical = self.resolve_entity(subject)
        if not canonical or canonical not in self.subject_index:
            return []
        return [self.relations[i] for i in self.subject_index[canonical]]

    def get_relations_for_object(self, obj: str) -> List[Relation]:
        """Get all relations where entity is the object."""
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
        Query for relations matching the pattern.
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
            "total_relations": len(self.relations),
            "total_aliases": len(self.alias_to_canonical),
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
