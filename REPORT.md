# Graph-Based Narrative Consistency Verification

**Kharagpur Data Science Hackathon - Track A**

---

## 1. Overall Approach

We verify whether natural language claims about fictional characters are consistent with knowledge extracted from source novels. The key insight: **separate world-building from verification** to eliminate hallucination in the verification step.

### Two-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: World Building (Colab/GPU - Expensive, Once)      │
│                                                              │
│  Novel → Chunking → LLM Extraction → Structured JSON Graph  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Verification (Local/CPU - Fast, Many Times)       │
│                                                              │
│  Claim → Parse to Query → Graph Lookup (NO LLM) → TRUE/FALSE│
└─────────────────────────────────────────────────────────────┘
```

**Why this matters:** Verification uses pure graph operations—no LLM inference, no hallucination risk. The LLM is only used for:
1. Extracting knowledge from novels (Phase 1)
2. Parsing claims into structured queries (Phase 2)

### Knowledge Representation

We extract four types of knowledge:

| Type | Example | Storage |
|------|---------|---------|
| **Entities** | alice, mad_hatter, rabbit_hole | entities.json |
| **Events** | E1: fall(agent=alice, destination=rabbit_hole) | events.json |
| **Relations** | (alice, sibling_of, sister) | edges.json |
| **Traits** | alice: curious (+1 curiosity axis) | traits.json |

---

## 2. Handling Long Context

Novels are long (100K+ tokens). Our approach:

### Chunking Strategy

```python
CHUNK_SIZE = 6000  # ~6KB per chunk
OVERLAP = 500      # Context continuity
```

- Split novel into manageable pieces
- Preserve chapter boundaries for temporal ordering
- Overlap ensures entity references aren't lost at boundaries

### Batch Processing

```python
BATCH_SIZE = 6  # chunks per batch
```

- Process 6 chunks in parallel per LLM call
- Reduces memory footprint vs. processing entire novel
- Enables 4-bit quantization to fit in 8GB RAM

### Monotonic Event Tracking

```python
class EventTimeTracker:
    def next_event_id(self) -> str:
        self.counter += 1
        return f"E{self.counter}"  # E1, E2, E3...
```

Events maintain temporal ordering across all chunks. If E42 happens in Chapter 3 and E78 in Chapter 5, we know E42 precedes E78.

### Indexed Graph Lookups

```python
class WorldGraph:
    source_index: Dict[str, List[Edge]]   # O(1) by source
    target_index: Dict[str, List[Edge]]   # O(1) by target
    relation_index: Dict[str, List[Edge]] # O(1) by relation type
```

At query time, we don't scan all edges—hash indices enable constant-time lookups regardless of graph size.

---

## 3. Distinguishing Causal Signals from Noise

This is the core challenge. Three mechanisms:

### A. Closed-World vs Open-World Semantics

Not all relations are equal:

| Type | Relations | Missing Edge Means |
|------|-----------|-------------------|
| **Closed-World** | sibling_of, parent_of, married_to, owns | **FALSE** |
| **Open-World** | likes, knows, fears, trusts | **UNKNOWN** |

**Example:**
- Claim: "Alice is Bob's sister"
- Both alice and bob exist in graph, but no sibling_of edge
- Result: **FALSE** (family relations are closed-world)

vs.

- Claim: "Alice knows Bob"
- No knows edge found
- Result: **UNKNOWN** (knowledge relations are open-world)

**Why this matters:** Family relationships are typically stated explicitly in novels. Emotional relationships might be implied but never stated.

### B. Trait Conflict Detection via Ontology

Traits mapped to semantic axes with polarity:

```python
TRAIT_ONTOLOGY = {
    "violent":   TraitMapping(axis="violence",    polarity=+1),
    "peaceful":  TraitMapping(axis="violence",    polarity=-1),
    "cruel":     TraitMapping(axis="kindness",    polarity=-1),
    "kind":      TraitMapping(axis="kindness",    polarity=+1),
    # ... 50+ traits across 8 axes
}
```

**Conflict Rule:** Same axis + opposite polarity = contradiction

```python
def are_traits_conflicting(trait1: str, trait2: str) -> bool:
    t1, t2 = TRAIT_ONTOLOGY[trait1], TRAIT_ONTOLOGY[trait2]
    return t1.axis == t2.axis and t1.polarity != t2.polarity
```

**Example:**
- Graph has: mad_hatter → violent (+1 violence)
- Claim: "The Mad Hatter is peaceful"
- peaceful = -1 violence, same axis, opposite polarity
- Result: **FALSE**

### C. Subgraph Pattern Matching for Events

Claims about actions become graph patterns:

```
Claim: "Alice fell into a rabbit hole"

Pattern:
  alice ──agent──→ Event(verb=fall) ──destination──→ rabbit_hole

Verification:
  1. Find events with verb="fall" or synonyms
  2. Check if any has incoming agent edge from alice
  3. Check if any has outgoing destination edge to rabbit_hole
  4. ALL constraints must match → TRUE, else UNKNOWN
```

### D. Synonym Resolution

Natural language varies. We normalize:

```python
VERB_SYNONYMS = {
    "fall": ["fell", "drop", "tumble", "plunge"],
    "kill": ["killed", "murder", "slay", "assassinate"],
}

PREDICATE_SYNONYMS = {
    "sibling_of": ["brother_of", "sister_of", "is_sibling_of"],
}
```

### E. Conflicting Predicate Detection

Some relations are mutually exclusive:

```python
CONFLICTING_PREDICATES = {
    "loves": ["hates", "despises"],
    "sibling_of": ["parent_of", "child_of"],  # Can't be both
}
```

If graph has (alice, loves, bob) and claim says "Alice hates Bob" → **FALSE**

---

## 4. Key Limitations & Failure Cases

### Limitation 1: Entity Resolution Failures

**Problem:** Same entity, different names.

```
Novel: "The Cheshire Cat grinned."
       "It vanished slowly."

Claim: "The Cat vanished"
```

"The Cat", "Cheshire Cat", "It" must all resolve to same entity. Our alias system helps but isn't perfect.

**Current mitigation:** Alias mapping in entities.json:
```json
{"canonical": "cheshire_cat", "aliases": ["the_cat", "cat", "cheshire"]}
```

**Failure case:** Pronouns ("it", "he", "she") are explicitly filtered—we cannot track them.

### Limitation 2: Implicit Information

**Problem:** Novels imply more than they state.

```
Novel: "Alice's sister sat reading on the bank."
Claim: "Alice has a sister"
```

The relation is implied by "Alice's sister" but never stated as "X is Alice's sister." Our extraction might miss this if the LLM doesn't recognize the possessive construction.

**Failure case:** Heavy reliance on LLM extraction quality.

### Limitation 3: Temporal Reasoning Gaps

**Problem:** We track event order but not precise time.

```
Claim: "Alice fell BEFORE meeting the Caterpillar"
```

We know E12 (fall) and E45 (meet caterpillar), and E12 < E45, so this is TRUE. But:

```
Claim: "Alice fell IMMEDIATELY BEFORE meeting the Caterpillar"
```

We can't verify "immediately"—many events might occur between E12 and E45.

### Limitation 4: Negation Handling

**Problem:** Negated claims are tricky.

```
Novel: "Alice did not drink the poison."
Claim: "Alice drank poison" → should be FALSE
```

We store a `negated` flag on edges, but:
- Extraction must correctly identify negation
- Double negatives ("not unlikely") are error-prone

### Limitation 5: Counterfactual & Hypothetical Statements

**Problem:** Novels contain non-factual statements.

```
Novel: "If Alice had drunk the poison, she would have shrunk."
```

This is hypothetical—Alice didn't actually drink it. But naive extraction might create:
```
alice --agent--> drink --patient--> poison
```

**Current mitigation:** LLM prompts explicitly exclude hypotheticals, but accuracy varies.

### Limitation 6: First-Person & Dialogue Attribution

**Problem:** Who said what?

```
Novel: "I am the Queen!" shouted the woman in red.
```

Does this mean `woman_in_red = queen`? Or is she lying? Our system treats stated claims as ground truth within the novel's world.

### Limitation 7: Scale & Coverage

**Quantitative limits:**
- Chunking may miss cross-chapter entity references
- 3B parameter model has extraction quality ceiling
- Trait ontology covers ~50 traits; unusual traits default to UNKNOWN

---

## 5. Summary

| Aspect | Approach |
|--------|----------|
| **Architecture** | Two-phase: GPU world-building + CPU verification |
| **Long Context** | 6KB chunking + batch processing + indexed lookups |
| **Signal vs Noise** | Closed/open world semantics + trait ontology + pattern matching |
| **Core Strength** | No hallucination in verification (pure graph ops) |
| **Core Weakness** | Extraction quality bounds overall accuracy |

The system trades recall for precision: we'd rather return UNKNOWN than hallucinate an answer. This is appropriate for consistency verification where false positives (claiming something is true when it contradicts the source) are worse than false negatives.

---

## Appendix A: File Structure

```
kdsh/
├── main.py                    # Entry point for verification
├── build_world_colab.py       # World building (run in Colab)
├── config.yaml                # Model and path configuration
├── src/
│   ├── graph_schema.py        # Entity, Event, Edge, Trait classes
│   ├── graph_builder.py       # Load/save/merge graphs
│   ├── graph_verifier.py      # Verification logic
│   ├── claim_parser.py        # NL → structured query
│   ├── trait_ontology.py      # Axis-based conflict detection
│   ├── claim_extractor.py     # Split backstory into claims
│   └── classifier.py          # Aggregate results
└── world/                     # Generated knowledge graph
    ├── entities.json
    ├── events.json
    ├── edges.json
    ├── traits.json
    └── chapters.json
```

## Appendix B: Example Verification Trace

**Claim:** "The Mad Hatter is Alice's brother"

```
1. Parse claim → RelationQuery(subject="mad_hatter",
                               predicate="sibling_of",
                               object="alice")

2. Lookup entities:
   - mad_hatter: EXISTS ✓
   - alice: EXISTS ✓

3. Check relation type:
   - sibling_of → CLOSED_WORLD

4. Search edges:
   - source_index["mad_hatter"] → [Edge(mad_hatter, host_of, tea_party), ...]
   - No sibling_of edge to alice found

5. Apply closed-world semantics:
   - Both entities exist
   - Relation is closed-world
   - Edge not found → FALSE

6. Result: FALSE
   Rationale: "No sibling relationship found between mad_hatter and alice
              in closed-world relation set"
```
