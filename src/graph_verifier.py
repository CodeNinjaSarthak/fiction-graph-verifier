"""
Graph-based verification of claims.
Queries the WorldGraph to determine if claims are True, False, or Unknown.
NO LLM is used here - this is pure graph lookup.
"""

import logging
from typing import Tuple, List, Optional, Any, Set
from enum import Enum

from src.graph_schema import WorldGraph, EntityType, normalize_entity_name, normalize_predicate
from src.claim_parser import parse_claim_to_triples, extract_traits_from_claim

logger = logging.getLogger(__name__)


class VerificationResult(Enum):
    TRUE = "true"       # Relation found in graph
    FALSE = "false"     # Conflicting relation exists or known negative
    UNKNOWN = "unknown" # No evidence either way


# Predicate synonyms for flexible matching
PREDICATE_SYNONYMS = {
    "is_sibling_of": ["is_brother_of", "is_sister_of", "brother_of", "sister_of", "sibling_of"],
    "is_parent_of": ["is_mother_of", "is_father_of", "parent_of", "mother_of", "father_of"],
    "is_child_of": ["is_son_of", "is_daughter_of", "child_of", "son_of", "daughter_of"],
    "falls_into": ["fell_into", "dropped_into", "tumbled_into", "fell_down"],
    "enters": ["entered", "goes_into", "went_into", "arrives_at", "arrived_at"],
    "has_tea_with": ["drinks_tea_with", "takes_tea_with", "had_tea_with"],
    "meets": ["met", "encounters", "encountered"],
    "loves": ["adores", "is_fond_of", "liked"],
    "hates": ["despises", "loathes", "dislikes"],
    "threatens": ["threatened", "menaces", "menaced"],
    "orders": ["ordered", "commands", "commanded"],
}

# Conflicting predicates - if A->B exists with pred1, then A->B with pred2 is false
CONFLICTING_PREDICATES = {
    "is_sibling_of": ["is_parent_of", "is_child_of", "is_married_to"],
    "loves": ["hates", "despises", "dislikes"],
    "hates": ["loves", "adores"],
    "is_alive": ["is_dead"],
    "is_dead": ["is_alive"],
}

# Symmetric relations - if A->B, then B->A is also true
SYMMETRIC_RELATIONS = {"is_sibling_of", "is_married_to", "is_friend_of", "is_enemy_of"}

# Family relations - if both entities are known characters but no relation exists, it's FALSE
FAMILY_RELATIONS = {
    "is_sibling_of", "is_brother_of", "is_sister_of",
    "is_parent_of", "is_child_of", "is_mother_of", "is_father_of",
    "sibling_of", "brother_of", "sister_of", "parent_of", "child_of"
}

# Trait conflicts - traits that cannot coexist
TRAIT_CONFLICTS = {
    "violent": ["peaceful", "gentle", "kind", "calm"],
    "cruel": ["loving", "caring", "peaceful", "kind"],
    "peaceful": ["violent", "aggressive", "cruel"],
    "gentle": ["violent", "aggressive", "cruel"],
    "kind": ["cruel", "violent", "mean"],
    "angry": ["calm", "peaceful"],
    "aggressive": ["peaceful", "gentle", "calm"],
}


def get_predicate_synonyms(predicate: str) -> Set[str]:
    """Get all synonyms for a predicate including itself."""
    pred_norm = normalize_predicate(predicate)
    synonyms = {pred_norm}

    for canonical, syn_list in PREDICATE_SYNONYMS.items():
        if pred_norm == canonical or pred_norm in syn_list:
            synonyms.add(canonical)
            synonyms.update(syn_list)

    return synonyms


def are_predicates_conflicting(pred1: str, pred2: str) -> bool:
    """Check if two predicates are mutually exclusive."""
    p1 = normalize_predicate(pred1)
    p2 = normalize_predicate(pred2)

    if p1 in CONFLICTING_PREDICATES:
        if p2 in CONFLICTING_PREDICATES[p1]:
            return True
    if p2 in CONFLICTING_PREDICATES:
        if p1 in CONFLICTING_PREDICATES[p2]:
            return True

    return False


def check_trait_conflict(
    graph: WorldGraph,
    subject: str,
    claim_traits: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Check if claim traits conflict with known character traits in the graph.

    Args:
        graph: The WorldGraph to query
        subject: The subject entity
        claim_traits: Traits mentioned in the claim

    Returns:
        Tuple of (has_conflict, explanation)
    """
    subj_canonical = graph.resolve_entity(subject)
    if not subj_canonical:
        return False, None

    # Get known traits for this entity
    known_traits = []
    for relation in graph.get_relations_for_subject(subj_canonical):
        if relation.predicate == "has_trait" and not relation.negated:
            known_traits.append(relation.object)

    # Check for conflicts
    for known in known_traits:
        known_lower = known.lower()
        if known_lower in TRAIT_CONFLICTS:
            for claim_trait in claim_traits:
                claim_trait_lower = claim_trait.lower()
                if claim_trait_lower in TRAIT_CONFLICTS[known_lower]:
                    return True, f"Character has trait '{known}' which conflicts with '{claim_trait}'"

    return False, None


def verify_triple(
    graph: WorldGraph,
    subject: str,
    predicate: str,
    obj: str,
    negated: bool = False
) -> Tuple[VerificationResult, str]:
    """
    Verify a single triple against the knowledge graph.

    Args:
        graph: The WorldGraph to query
        subject: Subject entity name
        predicate: Predicate/relation
        obj: Object entity name
        negated: Whether the claim asserts the relation does NOT exist

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    # Resolve entity aliases
    subj_canonical = graph.resolve_entity(subject)
    obj_canonical = graph.resolve_entity(obj)

    # If either entity is not found, might still be able to verify
    if subj_canonical is None:
        # Entity completely unknown
        return VerificationResult.UNKNOWN, f"Entity '{subject}' not found in graph"

    if obj_canonical is None:
        # Object not found - could be an attribute like "gardening"
        obj_canonical = normalize_entity_name(obj)

    # Normalize predicate and get synonyms
    pred_normalized = normalize_predicate(predicate)
    pred_synonyms = get_predicate_synonyms(pred_normalized)

    # Query for matching relations using all synonyms
    matching_relations = []
    for synonym in pred_synonyms:
        matches = graph.query_relation(
            subject=subj_canonical,
            predicate=synonym,
            obj=obj_canonical
        )
        matching_relations.extend(matches)

    # Also check symmetric relations in reverse
    if pred_normalized in SYMMETRIC_RELATIONS:
        for synonym in pred_synonyms:
            reverse_matches = graph.query_relation(
                subject=obj_canonical,
                predicate=synonym,
                obj=subj_canonical
            )
            matching_relations.extend(reverse_matches)

    # Determine verification result
    for rel in matching_relations:
        if not negated and not rel.negated:
            # Claim says X, graph says X -> TRUE
            return VerificationResult.TRUE, f"Found: {rel.subject} {rel.predicate} {rel.object}"
        if negated and rel.negated:
            # Claim says NOT X, graph says NOT X -> TRUE
            return VerificationResult.TRUE, f"Found negation: {rel.subject} does not {rel.predicate} {rel.object}"
        if not negated and rel.negated:
            # Claim says X, graph says NOT X -> FALSE
            return VerificationResult.FALSE, f"Graph says {rel.subject} does NOT {rel.predicate} {rel.object}"
        if negated and not rel.negated:
            # Claim says NOT X, graph says X -> FALSE
            return VerificationResult.FALSE, f"Graph says {rel.subject} {rel.predicate} {rel.object}"

    # Check for conflicting predicates
    subject_relations = graph.get_relations_for_subject(subj_canonical)
    for rel in subject_relations:
        # Check if same object but conflicting predicate
        rel_obj = graph.resolve_entity(rel.object) or rel.object
        if rel_obj == obj_canonical:
            if are_predicates_conflicting(pred_normalized, rel.predicate):
                return VerificationResult.FALSE, f"Conflicting relation: {rel.subject} {rel.predicate} {rel.object}"

    # Special handling for family relations between known characters
    if pred_normalized in FAMILY_RELATIONS:
        subj_entity = graph.entities.get(subj_canonical)
        obj_entity = graph.entities.get(obj_canonical)

        if (subj_entity and obj_entity and
            subj_entity.entity_type == EntityType.CHARACTER and
            obj_entity.entity_type == EntityType.CHARACTER):
            # Both are known characters but no family relation exists
            return VerificationResult.FALSE, f"No family relation between {subject} and {obj}"

    # No evidence found
    return VerificationResult.UNKNOWN, f"No evidence for {subject} {predicate} {obj}"


def verify_claim(
    claim: str,
    graph: WorldGraph,
    model: Any,
    tokenizer: Any
) -> Tuple[VerificationResult, str]:
    """
    Verify a natural language claim against the knowledge graph.

    Uses LLM only for parsing the claim into triples.
    The actual verification is done by querying the graph.

    Args:
        claim: Natural language claim string
        graph: The WorldGraph to query
        model: LLM model for parsing claim to triples
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    # Parse claim into triples using LLM
    triples = parse_claim_to_triples(claim, model, tokenizer)

    if not triples:
        return VerificationResult.UNKNOWN, "Could not parse claim into triples"

    # Extract traits from claim for conflict checking
    claim_traits = extract_traits_from_claim(claim)

    # Verify each triple
    results = []
    for subject, predicate, obj, negated in triples:
        result, explanation = verify_triple(graph, subject, predicate, obj, negated)
        results.append((result, explanation))

        # Check trait conflicts for the subject
        if claim_traits and result != VerificationResult.FALSE:
            has_conflict, conflict_explanation = check_trait_conflict(graph, subject, claim_traits)
            if has_conflict:
                results.append((VerificationResult.FALSE, conflict_explanation))

    # Aggregate results:
    # If ANY triple is FALSE -> claim is FALSE
    # If ALL triples are TRUE -> claim is TRUE
    # Otherwise -> UNKNOWN
    has_false = any(r == VerificationResult.FALSE for r, _ in results)
    all_true = all(r == VerificationResult.TRUE for r, _ in results)

    if has_false:
        false_explanations = [e for r, e in results if r == VerificationResult.FALSE]
        return VerificationResult.FALSE, "; ".join(false_explanations[:2])
    elif all_true:
        return VerificationResult.TRUE, "All components verified in graph"
    else:
        unknown_explanations = [e for r, e in results if r == VerificationResult.UNKNOWN]
        return VerificationResult.UNKNOWN, "; ".join(unknown_explanations[:2])
