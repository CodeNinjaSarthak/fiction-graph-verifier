"""
Trait ontology for narrative consistency checking.
Defines trait axes and polarity mappings for conflict detection.

Trait conflicts are detected when two traits share the same axis
but have opposite polarities (e.g., "violent" vs "peaceful" on the VIOLENCE axis).
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TraitAxis(Enum):
    """Semantic axes for character traits."""
    VIOLENCE = "violence"
    EMPATHY = "empathy"
    HONESTY = "honesty"
    COURAGE = "courage"
    KINDNESS = "kindness"
    TEMPERAMENT = "temperament"
    INTELLIGENCE = "intelligence"
    SOCIABILITY = "sociability"


@dataclass
class TraitMapping:
    """Maps a trait word to its axis and polarity."""
    axis: TraitAxis
    polarity: int  # +1 or -1


# Core trait ontology: trait_name -> (axis, polarity)
TRAIT_ONTOLOGY: Dict[str, TraitMapping] = {
    # Violence axis (+1 = violent, -1 = peaceful)
    "violent": TraitMapping(TraitAxis.VIOLENCE, +1),
    "aggressive": TraitMapping(TraitAxis.VIOLENCE, +1),
    "brutal": TraitMapping(TraitAxis.VIOLENCE, +1),
    "fierce": TraitMapping(TraitAxis.VIOLENCE, +1),
    "savage": TraitMapping(TraitAxis.VIOLENCE, +1),
    "peaceful": TraitMapping(TraitAxis.VIOLENCE, -1),
    "gentle": TraitMapping(TraitAxis.VIOLENCE, -1),
    "calm": TraitMapping(TraitAxis.VIOLENCE, -1),
    "docile": TraitMapping(TraitAxis.VIOLENCE, -1),
    "meek": TraitMapping(TraitAxis.VIOLENCE, -1),

    # Empathy axis (+1 = empathetic, -1 = cold)
    "empathetic": TraitMapping(TraitAxis.EMPATHY, +1),
    "compassionate": TraitMapping(TraitAxis.EMPATHY, +1),
    "caring": TraitMapping(TraitAxis.EMPATHY, +1),
    "sympathetic": TraitMapping(TraitAxis.EMPATHY, +1),
    "understanding": TraitMapping(TraitAxis.EMPATHY, +1),
    "cold": TraitMapping(TraitAxis.EMPATHY, -1),
    "indifferent": TraitMapping(TraitAxis.EMPATHY, -1),
    "callous": TraitMapping(TraitAxis.EMPATHY, -1),
    "heartless": TraitMapping(TraitAxis.EMPATHY, -1),
    "unfeeling": TraitMapping(TraitAxis.EMPATHY, -1),

    # Honesty axis (+1 = honest, -1 = dishonest)
    "honest": TraitMapping(TraitAxis.HONESTY, +1),
    "truthful": TraitMapping(TraitAxis.HONESTY, +1),
    "sincere": TraitMapping(TraitAxis.HONESTY, +1),
    "frank": TraitMapping(TraitAxis.HONESTY, +1),
    "candid": TraitMapping(TraitAxis.HONESTY, +1),
    "dishonest": TraitMapping(TraitAxis.HONESTY, -1),
    "deceptive": TraitMapping(TraitAxis.HONESTY, -1),
    "lying": TraitMapping(TraitAxis.HONESTY, -1),
    "deceitful": TraitMapping(TraitAxis.HONESTY, -1),
    "treacherous": TraitMapping(TraitAxis.HONESTY, -1),

    # Courage axis (+1 = brave, -1 = cowardly)
    "brave": TraitMapping(TraitAxis.COURAGE, +1),
    "courageous": TraitMapping(TraitAxis.COURAGE, +1),
    "fearless": TraitMapping(TraitAxis.COURAGE, +1),
    "bold": TraitMapping(TraitAxis.COURAGE, +1),
    "valiant": TraitMapping(TraitAxis.COURAGE, +1),
    "cowardly": TraitMapping(TraitAxis.COURAGE, -1),
    "timid": TraitMapping(TraitAxis.COURAGE, -1),
    "fearful": TraitMapping(TraitAxis.COURAGE, -1),
    "scared": TraitMapping(TraitAxis.COURAGE, -1),
    "spineless": TraitMapping(TraitAxis.COURAGE, -1),

    # Kindness axis (+1 = kind, -1 = cruel)
    "kind": TraitMapping(TraitAxis.KINDNESS, +1),
    "loving": TraitMapping(TraitAxis.KINDNESS, +1),
    "benevolent": TraitMapping(TraitAxis.KINDNESS, +1),
    "generous": TraitMapping(TraitAxis.KINDNESS, +1),
    "warm": TraitMapping(TraitAxis.KINDNESS, +1),
    "cruel": TraitMapping(TraitAxis.KINDNESS, -1),
    "mean": TraitMapping(TraitAxis.KINDNESS, -1),
    "malicious": TraitMapping(TraitAxis.KINDNESS, -1),
    "spiteful": TraitMapping(TraitAxis.KINDNESS, -1),
    "vicious": TraitMapping(TraitAxis.KINDNESS, -1),

    # Temperament axis (+1 = even-tempered, -1 = volatile)
    "patient": TraitMapping(TraitAxis.TEMPERAMENT, +1),
    "composed": TraitMapping(TraitAxis.TEMPERAMENT, +1),
    "serene": TraitMapping(TraitAxis.TEMPERAMENT, +1),
    "angry": TraitMapping(TraitAxis.TEMPERAMENT, -1),
    "irritable": TraitMapping(TraitAxis.TEMPERAMENT, -1),
    "volatile": TraitMapping(TraitAxis.TEMPERAMENT, -1),
    "hot-tempered": TraitMapping(TraitAxis.TEMPERAMENT, -1),

    # Intelligence axis (+1 = smart, -1 = foolish)
    "intelligent": TraitMapping(TraitAxis.INTELLIGENCE, +1),
    "clever": TraitMapping(TraitAxis.INTELLIGENCE, +1),
    "wise": TraitMapping(TraitAxis.INTELLIGENCE, +1),
    "smart": TraitMapping(TraitAxis.INTELLIGENCE, +1),
    "foolish": TraitMapping(TraitAxis.INTELLIGENCE, -1),
    "stupid": TraitMapping(TraitAxis.INTELLIGENCE, -1),
    "dim": TraitMapping(TraitAxis.INTELLIGENCE, -1),

    # Sociability axis (+1 = outgoing, -1 = withdrawn)
    "friendly": TraitMapping(TraitAxis.SOCIABILITY, +1),
    "outgoing": TraitMapping(TraitAxis.SOCIABILITY, +1),
    "sociable": TraitMapping(TraitAxis.SOCIABILITY, +1),
    "shy": TraitMapping(TraitAxis.SOCIABILITY, -1),
    "withdrawn": TraitMapping(TraitAxis.SOCIABILITY, -1),
    "reclusive": TraitMapping(TraitAxis.SOCIABILITY, -1),

    # Additional common traits
    "curious": TraitMapping(TraitAxis.INTELLIGENCE, +1),
    "mad": TraitMapping(TraitAxis.TEMPERAMENT, -1),
    "crazy": TraitMapping(TraitAxis.TEMPERAMENT, -1),
    "eccentric": TraitMapping(TraitAxis.SOCIABILITY, -1),
}


def get_trait_axis(trait: str) -> Optional[Tuple[TraitAxis, int]]:
    """
    Get the axis and polarity for a trait word.

    Args:
        trait: The trait name (e.g., "violent", "peaceful")

    Returns:
        Tuple of (TraitAxis, polarity) or None if trait not in ontology
    """
    trait_lower = trait.lower().strip()
    if trait_lower in TRAIT_ONTOLOGY:
        mapping = TRAIT_ONTOLOGY[trait_lower]
        return (mapping.axis, mapping.polarity)
    return None


def get_trait_axis_name(trait: str) -> Optional[str]:
    """
    Get just the axis name for a trait.

    Args:
        trait: The trait name

    Returns:
        Axis name as string (e.g., "violence") or None
    """
    result = get_trait_axis(trait)
    if result:
        return result[0].value
    return None


def are_traits_conflicting(trait1: str, trait2: str) -> bool:
    """
    Check if two traits are on the same axis with opposite polarities.

    Args:
        trait1: First trait name
        trait2: Second trait name

    Returns:
        True if traits conflict (same axis, opposite polarity)

    Examples:
        are_traits_conflicting("violent", "peaceful") -> True
        are_traits_conflicting("violent", "cruel") -> False (different axes)
        are_traits_conflicting("violent", "aggressive") -> False (same polarity)
    """
    info1 = get_trait_axis(trait1)
    info2 = get_trait_axis(trait2)

    if info1 is None or info2 is None:
        return False

    axis1, polarity1 = info1
    axis2, polarity2 = info2

    return axis1 == axis2 and polarity1 != polarity2


def get_conflicting_traits(trait: str) -> list:
    """
    Get all traits that conflict with the given trait.

    Args:
        trait: The trait to find conflicts for

    Returns:
        List of conflicting trait names
    """
    info = get_trait_axis(trait)
    if info is None:
        return []

    axis, polarity = info
    conflicts = []

    for other_trait, mapping in TRAIT_ONTOLOGY.items():
        if mapping.axis == axis and mapping.polarity != polarity:
            conflicts.append(other_trait)

    return conflicts
