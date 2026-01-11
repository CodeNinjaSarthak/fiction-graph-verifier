"""
Enhanced graph-based verification with subgraph matching,
trait axis conflicts, and closed-world semantics.

NO LLM is used for verification - this is pure graph lookup.
"""

import logging
from typing import Tuple, List, Optional, Any, Set
from enum import Enum

from src.graph_schema import (
    WorldGraph, EntityType, Event, Edge, Trait,
    WorldType, get_relation_world_type,
    normalize_entity_name, normalize_predicate
)
from src.claim_parser import (
    ClaimQuery, EventQuery, RelationQuery, TraitQuery,
    parse_claim_to_queries, parse_claim_to_triples, extract_traits_from_claim
)
from src.trait_ontology import are_traits_conflicting, get_trait_axis

logger = logging.getLogger(__name__)


class VerificationResult(Enum):
    TRUE = "true"       # Relation found in graph
    FALSE = "false"     # Conflicting relation exists or known negative
    UNKNOWN = "unknown" # No evidence either way


# Verb synonyms for flexible matching
VERB_SYNONYMS = {
    "fall": ["fell", "drop", "tumble", "plunge", "falls_into", "fell_into"],
    "meet": ["met", "encounter", "encountered", "greet", "greeted"],
    "drink": ["drank", "sip", "sipped", "consume", "consumed"],
    "eat": ["ate", "devour", "devoured", "consume", "consumed"],
    "enter": ["entered", "go_into", "went_into", "arrive", "arrived"],
    "run": ["ran", "sprint", "sprinted", "dash", "dashed"],
    "chase": ["chased", "pursue", "pursued", "follow", "followed"],
    "kill": ["killed", "murder", "murdered", "slay", "slew"],
    "love": ["loved", "adore", "adored"],
    "hate": ["hated", "despise", "despised", "loathe", "loathed"],
    "help": ["helped", "assist", "assisted", "aid", "aided"],
    "threaten": ["threatened", "menace", "menaced"],
    "order": ["ordered", "command", "commanded"],
}

# Predicate synonyms for relations
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
SYMMETRIC_RELATIONS = {"is_sibling_of", "is_married_to", "is_friend_of", "is_enemy_of",
                       "sibling_of", "married_to", "friend_of", "enemy_of"}


def get_verb_synonyms(verb: str) -> Set[str]:
    """Get all synonyms for a verb including itself."""
    verb_norm = normalize_predicate(verb)
    synonyms = {verb_norm}

    for canonical, syn_list in VERB_SYNONYMS.items():
        if verb_norm == canonical or verb_norm in syn_list:
            synonyms.add(canonical)
            synonyms.update(syn_list)

    return synonyms


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


# ============ SUBGRAPH MATCHING ============

def verify_event_query(
    graph: WorldGraph,
    query: EventQuery
) -> Tuple[VerificationResult, str]:
    """
    Verify an event query using subgraph matching.

    Pattern: agent --agent--> Event(verb) --destination--> target

    Example: "Alice fell into a rabbit hole"
    Matches: alice --agent--> E1(verb=fall) --destination--> rabbit_hole

    Args:
        graph: The WorldGraph to query
        query: EventQuery with verb, agent, patient, destination, location

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    verb_synonyms = get_verb_synonyms(query.verb)

    # Find events with matching verb
    matching_events = []
    query_verb_norm = normalize_predicate(query.verb)
    for event_id, event in graph.events.items():
        event_verb_norm = normalize_predicate(event.verb)
        # Check if normalized verbs match or if either verb is in the synonym set
        if (event_verb_norm in verb_synonyms or 
            event.verb in verb_synonyms or 
            query_verb_norm == event_verb_norm or
            query.verb == event.verb):
            matching_events.append(event)

    if not matching_events:
        if query.negated:
            return VerificationResult.TRUE, f"No events with verb '{query.verb}' found (negated claim confirmed)"
        return VerificationResult.UNKNOWN, f"No events with verb '{query.verb}' found"

    # For each matching event, check if the edge pattern matches
    for event in matching_events:
        edges_to_event = graph.get_edges_to(event.event_id)
        edges_from_event = graph.get_edges_from(event.event_id)

        # Track which constraints are satisfied
        agent_matched = query.agent is None
        patient_matched = query.patient is None
        destination_matched = query.destination is None
        location_matched = query.location is None
        instrument_matched = query.instrument is None

        # Check agent constraint
        if query.agent:
            agent_canonical = graph.resolve_entity(query.agent) or normalize_entity_name(query.agent)
            for edge in edges_to_event:
                edge_source_norm = normalize_entity_name(edge.source)
                if edge.relation == "agent" and (edge.source == agent_canonical or edge_source_norm == agent_canonical):
                    agent_matched = True
                    break

        # Check destination constraint
        if query.destination:
            dest_canonical = graph.resolve_entity(query.destination) or normalize_entity_name(query.destination)
            for edge in edges_from_event:
                edge_target_norm = normalize_entity_name(edge.target)
                if edge.relation == "destination" and (edge.target == dest_canonical or edge_target_norm == dest_canonical):
                    destination_matched = True
                    break

        # Check patient constraint
        if query.patient:
            patient_canonical = graph.resolve_entity(query.patient) or normalize_entity_name(query.patient)
            for edge in edges_from_event:
                edge_target_norm = normalize_entity_name(edge.target)
                if edge.relation == "patient" and (edge.target == patient_canonical or edge_target_norm == patient_canonical):
                    patient_matched = True
                    break

        # Check location constraint
        if query.location:
            loc_canonical = graph.resolve_entity(query.location) or normalize_entity_name(query.location)
            for edge in edges_from_event:
                edge_target_norm = normalize_entity_name(edge.target)
                if edge.relation == "location" and (edge.target == loc_canonical or edge_target_norm == loc_canonical):
                    location_matched = True
                    break

        # Check instrument constraint
        if query.instrument:
            inst_canonical = graph.resolve_entity(query.instrument) or normalize_entity_name(query.instrument)
            for edge in edges_to_event:
                edge_source_norm = normalize_entity_name(edge.source)
                if edge.relation == "instrument" and (edge.source == inst_canonical or edge_source_norm == inst_canonical):
                    instrument_matched = True
                    break

        # All constraints satisfied?
        if agent_matched and patient_matched and destination_matched and location_matched and instrument_matched:
            if query.negated:
                return VerificationResult.FALSE, f"Event found but claim says it didn't happen: {event.event_id}"
            return VerificationResult.TRUE, f"Matched event {event.event_id}: {event.verb}"

    # No matching event with all constraints
    if query.negated:
        return VerificationResult.TRUE, "No matching event pattern found (negated claim confirmed)"
    return VerificationResult.UNKNOWN, f"No event matching full pattern for '{query.verb}'"


# ============ RELATION VERIFICATION ============

def verify_relation_query(
    graph: WorldGraph,
    query: RelationQuery
) -> Tuple[VerificationResult, str]:
    """
    Verify a relation query with closed/open world semantics.

    - Closed world (sibling_of): missing relation between known entities = FALSE
    - Open world (likes): missing relation = UNKNOWN

    Args:
        graph: The WorldGraph to query
        query: RelationQuery with subject, predicate, object

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    subj_canonical = graph.resolve_entity(query.subject)
    obj_canonical = graph.resolve_entity(query.object) or normalize_entity_name(query.object)

    if not subj_canonical:
        return VerificationResult.UNKNOWN, f"Entity '{query.subject}' not found"

    # Get all predicate synonyms
    pred_synonyms = get_predicate_synonyms(query.predicate)

    # Check for direct edge
    edges = graph.get_edges_from(subj_canonical)
    for edge in edges:
        edge_rel_norm = normalize_predicate(edge.relation)
        if edge_rel_norm in pred_synonyms and edge.target == obj_canonical:
            if query.negated and not edge.negated:
                return VerificationResult.FALSE, f"Claim says NOT, but relation exists: {edge.source} {edge.relation} {edge.target}"
            if not query.negated and not edge.negated:
                return VerificationResult.TRUE, f"Found: {edge.source} {edge.relation} {edge.target}"
            if not query.negated and edge.negated:
                return VerificationResult.FALSE, f"Relation explicitly negated: {edge.source} NOT {edge.relation} {edge.target}"
            if query.negated and edge.negated:
                return VerificationResult.TRUE, f"Negation confirmed: {edge.source} NOT {edge.relation} {edge.target}"

    # Check symmetric relations in reverse direction
    pred_norm = normalize_predicate(query.predicate)
    if pred_norm in SYMMETRIC_RELATIONS:
        reverse_edges = graph.get_edges_from(obj_canonical)
        for edge in reverse_edges:
            edge_rel_norm = normalize_predicate(edge.relation)
            if edge_rel_norm in pred_synonyms and edge.target == subj_canonical:
                if not query.negated:
                    return VerificationResult.TRUE, f"Found symmetric: {edge.source} {edge.relation} {edge.target}"
                else:
                    return VerificationResult.FALSE, f"Symmetric relation exists but claim says NOT"

    # Check for conflicting predicates
    for edge in edges:
        if edge.target == obj_canonical:
            if are_predicates_conflicting(pred_norm, edge.relation):
                if not query.negated:
                    return VerificationResult.FALSE, f"Conflicting relation exists: {edge.source} {edge.relation} {edge.target}"

    # No relation found - apply closed/open world semantics
    world_type = get_relation_world_type(query.predicate)

    if world_type == WorldType.CLOSED:
        # Closed world: both entities must be known characters
        subj_entity = graph.entities.get(subj_canonical)
        obj_entity = graph.entities.get(obj_canonical)

        if subj_entity and obj_entity:
            # Both are known entities - absence means FALSE
            if subj_entity.entity_type == EntityType.CHARACTER and obj_entity.entity_type == EntityType.CHARACTER:
                if query.negated:
                    return VerificationResult.TRUE, f"No {query.predicate} relation between known characters (closed world, negated confirmed)"
                return VerificationResult.FALSE, f"No {query.predicate} relation between known characters (closed world)"

    # Open world: cannot determine
    if query.negated:
        return VerificationResult.UNKNOWN, f"Cannot confirm negation of '{query.predicate}' in open world"
    return VerificationResult.UNKNOWN, f"No evidence for {query.subject} {query.predicate} {query.object}"


# ============ TRAIT VERIFICATION ============

def verify_trait_query(
    graph: WorldGraph,
    query: TraitQuery
) -> Tuple[VerificationResult, str]:
    """
    Verify trait query with axis-based conflict detection.

    Conflict detection uses trait ontology:
    - Same axis, opposite polarity = conflict
    - Example: has "violent" (+1 violence) but claim says "peaceful" (-1 violence)

    Args:
        graph: The WorldGraph to query
        query: TraitQuery with entity and trait

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    entity_canonical = graph.resolve_entity(query.entity)
    if not entity_canonical:
        return VerificationResult.UNKNOWN, f"Entity '{query.entity}' not found"

    entity_traits = graph.get_entity_traits(entity_canonical)

    # Check for exact match or axis conflict
    for known_trait in entity_traits:
        # Exact match
        if known_trait.trait_name.lower() == query.trait.lower():
            if query.negated:
                return VerificationResult.FALSE, f"Entity has trait '{query.trait}' but claim says NOT"
            return VerificationResult.TRUE, f"Entity has trait '{query.trait}'"

        # Axis conflict check using ontology
        if are_traits_conflicting(known_trait.trait_name, query.trait):
            if query.negated:
                # Claim says "NOT peaceful" and entity has "violent" - consistent
                return VerificationResult.TRUE, f"Trait conflict confirms negation: has '{known_trait.trait_name}'"
            return VerificationResult.FALSE, f"Trait conflict: has '{known_trait.trait_name}' but claim says '{query.trait}'"

    # No trait information found
    return VerificationResult.UNKNOWN, f"No trait information for '{query.trait}' on entity"


# ============ TEMPORAL VERIFICATION ============

def verify_temporal_order(
    graph: WorldGraph,
    event1_id: str,
    event2_id: str,
    expected_order: str  # "before" or "after"
) -> Tuple[VerificationResult, str]:
    """
    Verify temporal ordering between two events.

    Args:
        graph: The WorldGraph
        event1_id: ID of first event
        event2_id: ID of second event
        expected_order: "before" (event1 < event2) or "after" (event1 > event2)

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    event1 = graph.events.get(event1_id)
    event2 = graph.events.get(event2_id)

    if not event1:
        return VerificationResult.UNKNOWN, f"Event '{event1_id}' not found"
    if not event2:
        return VerificationResult.UNKNOWN, f"Event '{event2_id}' not found"

    if expected_order == "before":
        if event1.time < event2.time:
            return VerificationResult.TRUE, f"{event1_id} (t={event1.time}) is before {event2_id} (t={event2.time})"
        elif event1.time > event2.time:
            return VerificationResult.FALSE, f"{event1_id} (t={event1.time}) is NOT before {event2_id} (t={event2.time})"
        else:
            return VerificationResult.UNKNOWN, f"Events have same time: {event1.time}"
    elif expected_order == "after":
        if event1.time > event2.time:
            return VerificationResult.TRUE, f"{event1_id} (t={event1.time}) is after {event2_id} (t={event2.time})"
        elif event1.time < event2.time:
            return VerificationResult.FALSE, f"{event1_id} (t={event1.time}) is NOT after {event2_id} (t={event2.time})"
        else:
            return VerificationResult.UNKNOWN, f"Events have same time: {event1.time}"

    return VerificationResult.UNKNOWN, f"Invalid expected_order: {expected_order}"


# ============ LEGACY TRIPLE VERIFICATION ============

def check_trait_conflict(
    graph: WorldGraph,
    subject: str,
    claim_traits: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Check if claim traits conflict with known character traits in the graph.
    Uses the trait ontology for axis-based conflict detection.

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
    known_traits = graph.get_entity_traits(subj_canonical)

    # Check for conflicts using ontology
    for known in known_traits:
        for claim_trait in claim_traits:
            if are_traits_conflicting(known.trait_name, claim_trait):
                return True, f"Character has trait '{known.trait_name}' which conflicts with '{claim_trait}'"

    return False, None


def verify_triple(
    graph: WorldGraph,
    subject: str,
    predicate: str,
    obj: str,
    negated: bool = False
) -> Tuple[VerificationResult, str]:
    """
    Verify a single triple against the knowledge graph (legacy interface).

    Args:
        graph: The WorldGraph to query
        subject: Subject entity name
        predicate: Predicate/relation
        obj: Object entity name
        negated: Whether the claim asserts the relation does NOT exist

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    # Convert to RelationQuery and use new verification
    query = RelationQuery(
        subject=subject,
        predicate=predicate,
        object=obj,
        negated=negated
    )
    return verify_relation_query(graph, query)


# ============ MAIN VERIFICATION FUNCTION ============

def verify_claim(
    claim: str,
    graph: WorldGraph,
    model: Any = None,
    tokenizer: Any = None,
    provider: str = "gemini",
    gemini_model: str = "gemini-1.5-flash"
) -> Tuple[VerificationResult, str]:
    """
    Verify a natural language claim against the knowledge graph.

    Uses:
    1. Subgraph matching for event patterns
    2. Relation verification with closed/open world semantics
    3. Trait conflict detection using axis ontology

    The LLM is used ONLY for parsing the claim into structured queries.
    All verification is done by querying the graph.

    Args:
        claim: Natural language claim string
        graph: The WorldGraph to query
        model: LLM model for parsing claim (optional, required for local provider)
        tokenizer: Tokenizer for the model (optional, required for local provider)
        provider: "gemini" or "local" - which provider to use for claim parsing
        gemini_model: Gemini model name (when using gemini provider)

    Returns:
        Tuple of (VerificationResult, explanation)
    """
    # Parse claim into structured queries
    query = parse_claim_to_queries(claim, model, tokenizer, provider, gemini_model)

    if not query.event_queries and not query.relation_queries and not query.trait_queries:
        # Fallback to legacy triple parsing
        triples = parse_claim_to_triples(claim, model, tokenizer, provider, gemini_model)
        if not triples:
            return VerificationResult.UNKNOWN, "Could not parse claim into queries"

        # Verify using legacy approach
        results = []
        claim_traits = extract_traits_from_claim(claim)

        for subject, predicate, obj, negated in triples:
            result, explanation = verify_triple(graph, subject, predicate, obj, negated)
            results.append((result, explanation))

            # Check trait conflicts
            if claim_traits and result != VerificationResult.FALSE:
                has_conflict, conflict_explanation = check_trait_conflict(graph, subject, claim_traits)
                if has_conflict:
                    results.append((VerificationResult.FALSE, conflict_explanation))

        return aggregate_results(results)

    # Verify using new query-based approach
    results = []

    # Verify event queries
    for event_query in query.event_queries:
        result, explanation = verify_event_query(graph, event_query)
        results.append((result, explanation))

    # Verify relation queries
    for rel_query in query.relation_queries:
        result, explanation = verify_relation_query(graph, rel_query)
        results.append((result, explanation))

    # Verify trait queries
    for trait_query in query.trait_queries:
        result, explanation = verify_trait_query(graph, trait_query)
        results.append((result, explanation))

    return aggregate_results(results)


def aggregate_results(
    results: List[Tuple[VerificationResult, str]]
) -> Tuple[VerificationResult, str]:
    """
    Aggregate multiple verification results.

    Logic:
    - If ANY result is FALSE -> claim is FALSE
    - If ALL results are TRUE -> claim is TRUE
    - Otherwise -> claim is UNKNOWN

    Args:
        results: List of (VerificationResult, explanation) tuples

    Returns:
        Aggregated (VerificationResult, explanation)
    """
    if not results:
        return VerificationResult.UNKNOWN, "No verification results"

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
