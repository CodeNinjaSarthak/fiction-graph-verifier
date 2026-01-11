# Narrative Consistency Verification System

**Kharagpur Data Science Hackathon - Track A**

A graph-based verification system that checks if claims about fictional characters are consistent with knowledge extracted from source novels. Unlike traditional LLM-based approaches, this system uses structured knowledge graphs for verification, eliminating hallucination in the verification step.

## Architecture

```
[User Claims] --> [Claim Parser (LLM)] --> [Structured Queries]
                                                   |
                                                   v
[Pre-built World Graph] <-- [Graph Verifier] --> [TRUE/FALSE/UNKNOWN]
        |
        +-- entities.json (characters, locations, objects)
        +-- events.json (actions with temporal ordering)
        +-- edges.json (relationships and event roles)
        +-- traits.json (character personality traits)
```

**Two-Tier Design:**
1. **World Building (Colab/GPU)**: Extract knowledge graph from novels using LLM
2. **Verification (Local/CPU)**: Query the graph to verify claims - no LLM needed

## Key Features

- **Graph-based verification**: No LLM hallucination in the verification step
- **Closed-world semantics**: Family relations (sibling_of, parent_of) use closed-world assumption - if not in graph, it's FALSE
- **Open-world semantics**: Emotional relations (likes, knows) use open-world assumption - if not in graph, it's UNKNOWN
- **Trait conflict detection**: Axis-based ontology detects contradictions (violent vs peaceful)
- **Temporal reasoning**: Event ordering verification

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key (if using Gemini)

```bash
export GOOGLE_API_KEY="your-api-key"
```

### 3. Run verification

```bash
# Demo mode with sample claims
python main.py

# Single claim
python main.py --claim "Alice fell into a rabbit hole"

# Multiple claims (backstory)
python main.py --backstory "Alice fell into a rabbit hole. The Mad Hatter is violent."
```

## World Data Structure

The `world/` folder contains the pre-built knowledge graph:

| File | Description |
|------|-------------|
| `entities.json` | Characters, locations, objects with aliases |
| `events.json` | Actions with verb, timestamp, and chapter |
| `edges.json` | Relationships: agent, patient, destination, sibling_of, etc. |
| `traits.json` | Character traits with axis and polarity |
| `chapters.json` | Chapter metadata |
| `relation_types.json` | Closed-world vs open-world relation definitions |
| `trait_axes.json` | Trait axis definitions (violence, courage, etc.) |

## Building a New World

Use `build_world_colab.py` in Google Colab for GPU-accelerated extraction:

```python
# In Colab with GPU runtime
!pip install transformers torch accelerate bitsandbytes

from build_world_colab import build_world_from_file

build_world_from_file(
    "/content/your_novel.txt",
    output_dir="/content/world",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    use_4bit=True
)
```

Download the generated `world/` folder and place it in the project root.

## Verification Logic

### Relation Types

| Type | Relations | Missing Edge Means |
|------|-----------|-------------------|
| **Closed-world** | sibling_of, parent_of, married_to, owns | FALSE |
| **Open-world** | likes, knows, helps, fears, agent, patient | UNKNOWN |

### Trait Conflict Detection

Traits on the same axis with opposite polarity create conflicts:

| Axis | Positive (+1) | Negative (-1) |
|------|---------------|---------------|
| violence | violent, aggressive | peaceful, gentle |
| courage | brave, fearless | cowardly, timid |
| honesty | truthful, sincere | dishonest, deceptive |
| empathy | kind, compassionate | cruel, callous |

Example: If a character has trait "violent" (+1 violence), claiming they are "peaceful" (-1 violence) returns FALSE.

## Output Format

Results are saved to `results.csv`:

```csv
story_id,counts,rationale
alice_test,"{'true': 2, 'false': 1, 'unknown': 1}","2 claims verified true, 1 false, 1 unknown"
```

## Project Structure

```
kdsh/
├── main.py                    # Orchestration & CLI
├── build_world_colab.py       # World builder (Colab/GPU)
├── config.yaml                # Configuration (provider, model settings)
├── requirements.txt           # Dependencies
├── src/
│   ├── __init__.py            # Package init
│   ├── graph_schema.py        # Core data structures (WorldGraph, Entity, Event, Edge)
│   ├── graph_builder.py       # Load world from JSON files
│   ├── graph_verifier.py      # Verification engine (verify_claim)
│   ├── claim_parser.py        # LLM-based claim parsing (Gemini/local)
│   ├── claim_extractor.py     # Backstory sentence splitting
│   ├── trait_ontology.py      # Trait axis & conflict detection
│   ├── classifier.py          # Result aggregation
│   └── consistency_checker.py # Model loading utilities
└── world/                     # Pre-built knowledge graph
    ├── entities.json
    ├── events.json
    ├── edges.json
    ├── traits.json
    └── ...
```

## Configuration

Edit `config.yaml` to switch between claim parsing providers:

```yaml
claim_parser:
  provider: "gemini"  # Options: "gemini", "local"

  gemini:
    model: "gemini-1.5-flash"
    # Set GOOGLE_API_KEY environment variable

  local:
    model: "Qwen/Qwen2.5-3B-Instruct"
    device: "auto"
```

- **gemini**: Uses Google's Gemini API (requires `GOOGLE_API_KEY` env var)
- **local**: Uses local HuggingFace model (requires GPU for reasonable speed)

## Requirements

- Python 3.8+
- 8GB RAM (16GB recommended)
- For Gemini provider: `GOOGLE_API_KEY` environment variable
- For local provider: GPU recommended (CPU works but slower)

## License

Kharagpur Data Science Hackathon - Track A
