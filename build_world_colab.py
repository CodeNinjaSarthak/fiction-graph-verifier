# !pip install transformers torch accelerate -q
# !pip install sentencepiece tqdm bitsandbytes -q

import json
import re
import os
import logging
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ CONFIGURATION ============


@dataclass
class WorldBuilderConfig:
    """Configuration for the world builder."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 1024
    use_4bit: bool = False

    # Chunking settings
    chunk_size: int = 6000
    chunk_overlap: int = 0

    # Batching settings
    batch_size: int = 6

    # Chapter detection patterns
    chapter_patterns: List[str] = field(
        default_factory=lambda: [
            r"^CHAPTER\s+[IVXLCDM]+\.?\s*",
            r"^CHAPTER\s+\d+\.?\s*",
            r"^Chapter\s+[A-Z][a-z]+\.?\s*",
            r"^[IVXLCDM]+\.\s+",
            r"^\d+\.\s+",
        ]
    )

    # Output directory
    output_dir: str = "world"


# ============ TRAIT ONTOLOGY ============

TRAIT_AXES = {
    "violence": {
        "violent": 1,
        "brutal": 1,
        "aggressive": 1,
        "peaceful": -1,
        "gentle": -1,
        "calm": -1,
    },
    "empathy": {
        "kind": 1,
        "compassionate": 1,
        "caring": 1,
        "cruel": -1,
        "heartless": -1,
        "callous": -1,
    },
    "courage": {
        "brave": 1,
        "courageous": 1,
        "fearless": 1,
        "cowardly": -1,
        "timid": -1,
        "fearful": -1,
    },
    "intelligence": {
        "intelligent": 1,
        "smart": 1,
        "clever": 1,
        "wise": 1,
        "foolish": -1,
        "stupid": -1,
        "naive": -1,
    },
    "honesty": {
        "honest": 1,
        "truthful": 1,
        "sincere": 1,
        "deceitful": -1,
        "dishonest": -1,
        "lying": -1,
    },
    "social": {
        "friendly": 1,
        "sociable": 1,
        "outgoing": 1,
        "shy": -1,
        "withdrawn": -1,
        "antisocial": -1,
    },
}


# ============ RELATION TYPES ============

RELATION_TYPES = {
    "closed_world": [
        "parent_of",
        "child_of",
        "sibling_of",
        "married_to",
        "spouse_of",
        "owns",
        "part_of",
    ],
    "open_world": [
        "knows",
        "likes",
        "loves",
        "hates",
        "helps",
        "threatens",
        "fears",
        "trusts",
        "admires",
        "agent",
        "patient",
        "destination",
        "location",
        "instrument",
    ],
}


# ============ VALID VERBS LIST ============

VALID_VERB_PREFIXES = {
    "fall",
    "meet",
    "drink",
    "eat",
    "run",
    "walk",
    "talk",
    "say",
    "tell",
    "ask",
    "go",
    "come",
    "see",
    "look",
    "find",
    "take",
    "give",
    "get",
    "make",
    "do",
    "think",
    "know",
    "want",
    "try",
    "use",
    "work",
    "call",
    "feel",
    "leave",
    "put",
    "mean",
    "keep",
    "let",
    "begin",
    "seem",
    "help",
    "show",
    "hear",
    "play",
    "turn",
    "start",
    "hold",
    "bring",
    "happen",
    "write",
    "sit",
    "stand",
    "lose",
    "pay",
    "send",
    "allow",
    "stay",
    "speak",
    "lead",
    "read",
    "grow",
    "open",
    "close",
    "carry",
    "set",
    "learn",
    "change",
    "end",
    "pass",
    "expect",
    "decide",
    "appear",
    "buy",
    "wait",
    "serve",
    "die",
    "build",
    "spend",
    "cut",
    "reach",
    "kill",
    "raise",
    "laugh",
    "cry",
    "shout",
    "jump",
    "hide",
    "fight",
    "attack",
    "defend",
    "escape",
    "chase",
    "catch",
    "throw",
    "push",
    "pull",
    "break",
    "fix",
    "create",
    "destroy",
    "enter",
    "exit",
    "arrive",
    "depart",
    "return",
    "follow",
    "lead",
    "guide",
    "teach",
    "learn",
    "understand",
    "explain",
    "describe",
    "argue",
    "agree",
    "disagree",
    "promise",
    "refuse",
    "accept",
    "reject",
    "offer",
    "request",
    "demand",
    "bow",
    "kneel",
    "rise",
    "ascend",
    "descend",
    "climb",
    "crawl",
    "swim",
    "fly",
    "sail",
    "ride",
    "drive",
    "travel",
    "journey",
    "visit",
    "explore",
    "discover",
    "search",
    "hunt",
    "gather",
    "collect",
    "save",
    "protect",
    "harm",
    "hurt",
    "heal",
    "cure",
    "poison",
    "infect",
    "behead",
    "execute",
    "arrest",
    "capture",
    "release",
    "free",
    "imprison",
    "punish",
    "reward",
    "celebrate",
    "mourn",
    "worry",
    "fear",
    "love",
    "hate",
    "like",
    "dislike",
    "enjoy",
    "suffer",
    "engage",
    "dispute",
    "repeat",
    "crowd",
    "surprise",
    "tuck",
    "back",
    "appear",
    "collect",
}


def is_valid_verb(verb: str) -> bool:
    """Check if a verb is valid (not a fragment)."""
    if not verb or len(verb) < 3:
        return False

    verb_lower = verb.lower().strip()

    # Check against known verbs
    for valid_prefix in VALID_VERB_PREFIXES:
        if verb_lower.startswith(valid_prefix):
            return True

    # Reject obvious non-verbs
    if verb_lower in {
        "over",
        "out",
        "in",
        "up",
        "down",
        "back",
        "away",
        "body",
        "head",
        "hand",
        "eye",
        "face",
        "thing",
        "place",
        "time",
        "way",
        "man",
        "woman",
    }:
        return False

    # Check if it looks like a verb (ends in common verb patterns)
    if re.match(r"^[a-z]+(?:e[ds]?|ing|s)$", verb_lower):
        return True

    return False


# ============ CHAPTER DETECTION ============


def detect_chapters(
    text: str, config: WorldBuilderConfig
) -> List[Tuple[str, int, int]]:
    """Detect chapter boundaries in novel text."""
    chapters = []
    patterns = [
        re.compile(p, re.MULTILINE | re.IGNORECASE) for p in config.chapter_patterns
    ]

    markers = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            markers.append((match.start(), match.group().strip()))

    markers.sort(key=lambda x: x[0])

    seen_positions = set()
    unique_markers = []
    for pos, title in markers:
        if pos not in seen_positions:
            seen_positions.add(pos)
            unique_markers.append((pos, title))

    for i, (start, title) in enumerate(unique_markers):
        if i + 1 < len(unique_markers):
            end = unique_markers[i + 1][0]
        else:
            end = len(text)
        chapters.append((title, start, end))

    if not chapters:
        chapters = [("Full Text", 0, len(text))]

    logger.info(f"Detected {len(chapters)} chapters")
    return chapters


# ============ CHUNKING ============


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 0) -> List[str]:
    """Split text into chunks for LLM processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ============ EVENT TIME TRACKER - FIXED ============


class EventTimeTracker:
    """Tracks monotonic event time across chunks and chapters."""

    def __init__(self):
        self.current_time = 0
        self.event_id_counter = 0
        self.chapter_events: Dict[int, List[str]] = {}

    def get_next_event_id_and_time(self) -> Tuple[str, int]:
        """Get next available event ID and time."""
        self.event_id_counter += 1
        self.current_time += 1
        return f"E{self.event_id_counter}", self.current_time

    def record_chapter_event(self, chapter_num: int, event_id: str):
        """Record an event for a chapter."""
        if chapter_num not in self.chapter_events:
            self.chapter_events[chapter_num] = []
        self.chapter_events[chapter_num].append(event_id)


# ============ LLM EXTRACTION  ============

EXTRACTION_PROMPT = """You are extracting structured narrative information from a novel passage.

CRITICAL INSTRUCTIONS:
1. NEVER use pronouns (he, she, they, it, him, her, his, their) as entity IDs
2. ALWAYS use the canonical entity name (e.g., "alice", "mad_hatter", "rabbit")
3. Extract ONLY events that are explicitly present - do not invent events
4. Use the exact event ID format provided: E<number>
5. Traits MUST include the axis name from the trait ontology

Extract the following as JSON:

1. **entities**: Named characters, locations, and objects
   - id: lowercase_snake_case identifier (e.g., "alice", "mad_hatter", "rabbit_hole")
   - type: "character", "location", or "object"
   - NO PRONOUNS - use canonical names only

2. **events**: Actions or happenings that ACTUALLY OCCUR in the passage
   - Start with event ID {start_event_id} and increment: {start_event_id}, E{next_num}, E{next_num_plus}, etc.
   - Starting time: {start_time}
   - verb: The action in present tense (e.g., "fall", "meet", "drink", "run", "say")
   - ONLY include real actions - no fragments like "over", "back", "body"
   - If there are no events in the passage, return an empty events array

3. **edges**: Relationships connecting entities and events
   Format: [source_id, relation_type, target_id]
   - Event IDs must be UPPERCASE (e.g., "E631", not "e631")
   - Entity IDs must be lowercase_snake_case

   Event roles (open-world):
   - "agent": who performs the action (entity -> event)
   - "patient": who receives the action (event -> entity)
   - "destination": where the action leads (event -> entity)
   - "location": where the action happens (event -> entity)
   - "instrument": with what (event -> entity)

   Entity relations (CLOSED-WORLD):
   - "parent_of", "child_of", "sibling_of", "married_to", "spouse_of"

   Entity relations (OPEN-WORLD):
   - "knows", "likes", "loves", "hates", "helps", "threatens"

4. **traits**: Character personality traits
   Format: [entity_id, trait_name, axis_name, polarity]
   - axis_name MUST be from: {trait_axes}
   - polarity: 1 for positive pole, -1 for negative pole

Known entities from previous chunks: {known_entities}

Passage:
---
{passage_text}
---

Return ONLY valid JSON:
{{
  "entities": [{{"id": "alice", "type": "character"}}],
  "events": [{{"id": "E631", "verb": "fall", "time": 631}}],
  "edges": [["alice", "agent", "E631"], ["E631", "destination", "rabbit_hole"]],
  "traits": [["queen_of_hearts", "violent", "violence", 1]]
}}"""


def generate_text_batch(
    prompts: List[str], model, tokenizer, max_new_tokens: int = 1024
) -> List[str]:
    """
    Generate text from LLM for a batch of prompts (FIXED - no EOS stopping).
    """
    import torch

    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # FIXED: No EOS-based early stopping
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                # No eos_token_id stopping - let it generate fully
            )

        results = []
        for i, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            prompt_decoded = tokenizer.decode(
                inputs["input_ids"][i], skip_special_tokens=True
            )
            new_text = generated_text[len(prompt_decoded) :].strip()
            results.append(new_text)

        return results

    except Exception as e:
        logger.error(f"Error generating text batch: {e}")
        return [""] * len(prompts)


def parse_llm_json(output: str) -> Dict[str, Any]:
    """Parse JSON from LLM output using bracket matching."""
    if not output:
        return {}

    # Find first { and matching }
    start = output.find("{")
    if start == -1:
        return {}

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(output)):
        char = output[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON
                    json_str = output[start : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON parse error: {e}")
                        return {}

    # Try parsing whole output as fallback
    try:
        result = json.loads(output.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    logger.debug(f"Failed to parse JSON from: {output[:200]}")
    return {}


def extract_from_chunks_batch(
    chunks: List[str],
    model,
    tokenizer,
    time_tracker: EventTimeTracker,
    known_entities: List[str],
    chapter_num: int,
) -> List[Dict[str, Any]]:
    """
    Extract entities, events, edges, and traits from multiple chunks (FIXED).
    """
    all_prompts = []
    all_start_ids = []
    all_start_times = []

    for chunk_text in chunks:
        # Get starting event ID and time for this chunk
        start_event_id, start_time = time_tracker.get_next_event_id_and_time()
        all_start_ids.append(start_event_id)
        all_start_times.append(start_time)

        # Extract event number
        event_num = int(start_event_id[1:])

        trait_axes_str = ", ".join(TRAIT_AXES.keys())

        prompt = EXTRACTION_PROMPT.format(
            start_event_id=start_event_id,
            next_num=event_num + 1,
            next_num_plus=event_num + 2,
            start_time=start_time,
            trait_axes=trait_axes_str,
            known_entities=", ".join(known_entities[:20]) if known_entities else "None",
            passage_text=chunk_text[:5500],
        )

        all_prompts.append(prompt)

    # Generate batch
    outputs = generate_text_batch(all_prompts, model, tokenizer)

    # Parse and validate each output
    results = []
    for i, (output, start_event_id, start_time) in enumerate(
        zip(outputs, all_start_ids, all_start_times)
    ):
        parsed = parse_llm_json(output)

        # Validate and filter events
        valid_events = []
        for event in parsed.get("events", []):
            if isinstance(event, dict):
                event_id = event.get("id", "")
                verb = event.get("verb", "")
                event_time = event.get("time", 0)

                # FIXED: Validate event
                # 1. Must have uppercase E prefix
                if not event_id.startswith("E"):
                    continue

                # 2. Must have valid verb
                if not is_valid_verb(verb):
                    logger.debug(f"Rejected invalid verb: {verb}")
                    continue

                # 3. Time must be reasonable
                if event_time < start_time or event_time > start_time + 100:
                    continue

                valid_events.append(
                    {
                        "id": event_id,
                        "verb": verb,
                        "time": event_time,
                        "chapter": chapter_num,
                    }
                )
                time_tracker.record_chapter_event(chapter_num, event_id)

        results.append(
            {
                "entities": parsed.get("entities", []),
                "events": valid_events,
                "edges": parsed.get("edges", []),
                "traits": parsed.get("traits", []),
            }
        )

    return results


# ============ WORLD MERGER - FIXED ============

PRONOUNS = {
    "he",
    "she",
    "they",
    "it",
    "him",
    "her",
    "his",
    "their",
    "its",
    "himself",
    "herself",
    "itself",
    "themselves",
    "hers",
    "theirs",
}


class WorldMerger:
    """Merges extracted data across chunks, handling deduplication."""

    def __init__(self):
        self.entities: Dict[str, Dict] = {}
        self.events: Dict[str, Dict] = {}
        self.edges: Set[Tuple] = set()
        self.traits: Dict[Tuple, Tuple] = {}
        self.chapters: Dict[int, Dict] = {}

    def normalize_entity_id(self, name: str) -> str:
        """Normalize entity ID to canonical form (lowercase only)."""
        if not name:
            return ""
        name = name.lower().strip()

        if name in PRONOUNS:
            return ""

        name = re.sub(r"^(the|a|an)\s+", "", name)
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"'s$", "", name)
        return name

    def merge_chunk(self, chunk_data: Dict, chapter_num: int):
        """Merge a single chunk's extracted data."""
        # Entities
        for entity in chunk_data.get("entities", []):
            if not isinstance(entity, dict):
                continue
            entity_id = self.normalize_entity_id(entity.get("id", ""))
            if not entity_id:
                continue

            if entity_id not in self.entities:
                self.entities[entity_id] = {
                    "id": entity_id,
                    "type": entity.get("type", "object").lower(),
                    "display_name": entity.get("id", entity_id),
                    "aliases": [],
                }

        # Events - FIXED: No normalization, keep uppercase
        for event in chunk_data.get("events", []):
            if not isinstance(event, dict):
                continue
            event_id = event.get("id", "")

            # FIXED: Don't normalize event IDs
            if event_id and event_id.startswith("E") and event_id not in self.events:
                self.events[event_id] = {
                    "id": event_id,
                    "verb": event.get("verb", "unknown"),
                    "time": event.get("time", 0),
                    "chapter": event.get("chapter", chapter_num),
                }

        # Edges - FIXED: Don't normalize event IDs
        for edge in chunk_data.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 3:
                src = str(edge[0])
                rel = str(edge[1]).lower().replace(" ", "_")
                tgt = str(edge[2])

                # FIXED: Only normalize if not an event ID
                if not src.startswith("E"):
                    src = self.normalize_entity_id(src)
                if not tgt.startswith("E"):
                    tgt = self.normalize_entity_id(tgt)

                if src and rel and tgt:
                    self.edges.add((src, rel, tgt))

        # Traits
        for trait in chunk_data.get("traits", []):
            if isinstance(trait, (list, tuple)) and len(trait) >= 4:
                entity_id = self.normalize_entity_id(str(trait[0]))
                trait_name = str(trait[1]).lower()
                axis_name = str(trait[2]).lower()
                polarity = trait[3]

                if entity_id and trait_name and axis_name and axis_name in TRAIT_AXES:
                    self.traits[(entity_id, trait_name, axis_name)] = (polarity, None)

        # Update chapter metadata
        if chapter_num not in self.chapters:
            self.chapters[chapter_num] = {
                "num": chapter_num,
                "first_event": None,
                "last_event": None,
            }

    def set_chapter_title(self, chapter_num: int, title: str):
        """Set chapter title."""
        if chapter_num in self.chapters:
            self.chapters[chapter_num]["title"] = title
        else:
            self.chapters[chapter_num] = {"num": chapter_num, "title": title}

    def set_chapter_events(self, chapter_num: int, event_ids: List[str]):
        """Set chapter event range."""
        if event_ids:
            if chapter_num in self.chapters:
                self.chapters[chapter_num]["first_event"] = event_ids[0]
                self.chapters[chapter_num]["last_event"] = event_ids[-1]

    def get_known_entities(self) -> List[str]:
        """Get list of known entity IDs."""
        return list(self.entities.keys())


# ============ OUTPUT FUNCTIONS ============


def save_world(merger: WorldMerger, output_dir: str = "world"):
    """Save merged world data to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save entities.json
    with open(output_path / "entities.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in merger.entities.items()}, f, indent=2, ensure_ascii=False
        )
    logger.info(f"Saved {len(merger.entities)} entities")

    # Save events.json
    with open(output_path / "events.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in merger.events.items()}, f, indent=2, ensure_ascii=False
        )
    logger.info(f"Saved {len(merger.events)} events")

    # Save edges.json
    with open(output_path / "edges.json", "w", encoding="utf-8") as f:
        json.dump(list(merger.edges), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(merger.edges)} edges")

    # Save traits.json
    traits_data = {}
    for (entity_id, trait_name, axis_name), (polarity, conf) in merger.traits.items():
        if entity_id not in traits_data:
            traits_data[entity_id] = {}
        if axis_name not in traits_data[entity_id]:
            traits_data[entity_id][axis_name] = {}
        traits_data[entity_id][axis_name][trait_name] = [polarity, conf]

    with open(output_path / "traits.json", "w", encoding="utf-8") as f:
        json.dump(traits_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved traits for {len(traits_data)} entities")

    # Save trait_axes.json
    with open(output_path / "trait_axes.json", "w", encoding="utf-8") as f:
        json.dump(TRAIT_AXES, f, indent=2, ensure_ascii=False)

    # Save relation_types.json
    with open(output_path / "relation_types.json", "w", encoding="utf-8") as f:
        json.dump(RELATION_TYPES, f, indent=2, ensure_ascii=False)

    # Save chapters.json
    with open(output_path / "chapters.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in merger.chapters.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved {len(merger.chapters)} chapters")


# ============ MODEL LOADING ============


def load_model_and_tokenizer(config: WorldBuilderConfig):
    """Load model and tokenizer with optional 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if config.use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            logger.info("Attempting 4-bit quantization...")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✓ Model loaded with 4-bit quantization")
        except Exception as e:
            logger.warning(f"⚠ 4-bit failed, using FP16: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


# ============ MAIN BUILD FUNCTION ============


def build_world(
    novel_text: str,
    config: Optional[WorldBuilderConfig] = None,
    model=None,
    tokenizer=None,
):
    """Build a world graph from novel text (FIXED VERSION)."""
    if config is None:
        config = WorldBuilderConfig()

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config)

    chapters = detect_chapters(novel_text, config)
    merger = WorldMerger()
    time_tracker = EventTimeTracker()

    for chapter_num, (title, start, end) in enumerate(
        tqdm(chapters, desc="Processing chapters"), 1
    ):
        chapter_text = novel_text[start:end]
        merger.set_chapter_title(chapter_num, title)

        chunks = chunk_text(chapter_text, config.chunk_size, config.chunk_overlap)
        logger.info(f"Chapter {chapter_num}: '{title[:50]}' - {len(chunks)} chunks")

        batch_size = config.batch_size
        for batch_start in tqdm(
            range(0, len(chunks), batch_size),
            desc=f"  Chapter {chapter_num} batches",
            leave=False,
        ):
            batch_end = min(batch_start + batch_size, len(chunks))
            chunk_batch = chunks[batch_start:batch_end]

            batch_results = extract_from_chunks_batch(
                chunk_batch,
                model,
                tokenizer,
                time_tracker,
                merger.get_known_entities(),
                chapter_num,
            )

            for chunk_data in batch_results:
                merger.merge_chunk(chunk_data, chapter_num)

        if chapter_num in time_tracker.chapter_events:
            event_ids = time_tracker.chapter_events[chapter_num]
            merger.set_chapter_events(chapter_num, event_ids)

        logger.info(
            f"  Chapter complete: {len(merger.entities)} entities, {len(merger.events)} events"
        )

    logger.info(
        f"World building complete: {len(merger.entities)} entities, {len(merger.events)} events"
    )
    return merger


def build_world_from_file(
    filepath: str,
    output_dir: str = "world",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    use_4bit: bool = True,
):
    """Build world from a text file and save (FIXED VERSION)."""
    logger.info(f"Reading novel from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        novel_text = f.read()
    logger.info(f"Novel length: {len(novel_text)} characters")

    config = WorldBuilderConfig(
        model_name=model_name,
        output_dir=output_dir,
        use_4bit=use_4bit,
        chunk_size=6000,
        batch_size=6,
    )

    merger = build_world(novel_text, config)
    save_world(merger, output_dir)

    logger.info("=" * 50)
    logger.info("WORLD BUILD COMPLETE (FIXED VERSION)")
    logger.info(f"Output: {output_dir}/")
    logger.info("=" * 50)


if __name__ == "__main__":
    print(
        """
    Usage:
    build_world_from_file("/content/your_novel.txt", output_dir="/content/world")
    """
    )
