# Kharagpur Data Science Hackathon - Track A: Modular Python Pipeline

A modular consistency checking pipeline that verifies backstory claims against Project Gutenberg novels using Pathway framework for vector search and local transformer models (phi-2) for consistency checking.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure settings:**
   Edit `config.yaml` to specify Gutenberg book IDs and model settings.

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

Results will be saved to `results.csv`.

## System Requirements

- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU**: Optional but recommended for faster inference (CUDA-compatible or Apple Silicon)
- **Python**: 3.8 or higher
- **Disk Space**: ~5GB for models and cached books

## Installation

### 1. Create and activate virtual environment

```bash
# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download transformer models (~2-5GB). Ensure you have sufficient disk space and a stable internet connection.

### 3. Verify installation

```bash
python -c "import pathway; import transformers; print('Installation successful!')"
```

## Configuration

Edit `config.yaml` to customize the pipeline:

### Gutenberg Books

```yaml
gutenberg_books:
  - 1342  # Pride and Prejudice
  - 11    # Alice in Wonderland
```

### Model Settings

```yaml
models:
  embedding: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
  llm: "microsoft/phi-2"  # Language model for consistency checking
  device: "auto"  # auto, cuda, cpu, or mps
```

### Pathway Settings

```yaml
pathway:
  chunk_size: 500      # Characters per chunk
  chunk_overlap: 50    # Overlap between chunks
```

### Inference Settings

```yaml
inference:
  max_tokens: 256      # Maximum tokens to generate
  batch_size: 4        # Batch size for processing claims
```

### Retrieval Settings

```yaml
retrieval:
  top_k: 5  # Number of passages to retrieve per claim
```

## Finding Gutenberg Books

1. Visit [Project Gutenberg](https://www.gutenberg.org/)
2. Search for books by title or author
3. Open the book's page
4. The book ID is in the URL: `https://www.gutenberg.org/ebooks/1342` → ID is `1342`

### Popular Book IDs

- `1342` - Pride and Prejudice (Jane Austen)
- `11` - Alice's Adventures in Wonderland (Lewis Carroll)
- `84` - Frankenstein (Mary Shelley)
- `2701` - Moby Dick (Herman Melville)
- `74` - The Adventures of Tom Sawyer (Mark Twain)

## Usage

### Basic Usage

Run the pipeline with default settings:

```bash
python main.py
```

The pipeline will:
1. Download specified Gutenberg books
2. Index them into the vector store
3. Process test cases
4. Generate `results.csv`

### Sample Test Case

The pipeline includes a hardcoded test case using Pride and Prejudice. To add your own test cases, modify `main.py` or extend it to read from a file.

### Output Format

Results are saved to `results.csv` with the following columns:

- `story_id`: Unique identifier for the test case
- `prediction`: `0` (inconsistent) or `1` (consistent)
- `rationale`: Brief explanation of the prediction

Example:

```csv
story_id,prediction,rationale
test_001,1,All 5 claims are consistent with the evidence. Key findings: The evidence confirms Elizabeth Bennet's character development...
```

## Troubleshooting

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce `batch_size` in `config.yaml` (try `1` or `2`)
2. Use a smaller model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
3. Set `device: "cpu"` in `config.yaml` (slower but uses less memory)

### Model Download Issues

**Error**: `ConnectionError` or `TimeoutError` during model download

**Solution**:
1. Check internet connection
2. Manually download models using HuggingFace CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli download microsoft/phi-2
   ```
3. Set `HF_HOME` environment variable to specify cache directory

### Pathway Setup Errors

**Error**: `ImportError: No module named 'pathway'`

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall: `pip install --upgrade pathway>=0.7.0`
3. Check Python version: `python --version` (requires 3.8+)

### Gutenberg Download Failures

**Error**: `ValueError: Failed to download book`

**Solution**:
1. Check book ID is valid on Project Gutenberg
2. Verify internet connection
3. Check if book is available in your region
4. Try a different book ID

### Slow Performance

**Solutions**:
1. Use GPU: Set `device: "cuda"` in `config.yaml`
2. Reduce `chunk_size` in Pathway settings
3. Reduce `top_k` in retrieval settings
4. Use smaller embedding model

## Model Selection

### Recommended Models

1. **microsoft/phi-2** (2.7B parameters)
   - Good balance of quality and speed
   - Recommended for most use cases

2. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B parameters)
   - Faster inference, lower memory
   - Good for testing or limited resources

3. **stabilityai/stablelm-3b-4e1t** (3B parameters)
   - Higher quality, slower inference
   - Good for production use

### Changing Models

Edit `config.yaml`:

```yaml
models:
  llm: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change this line
```

### Embedding Models

The default `sentence-transformers/all-MiniLM-L6-v2` is fast and efficient. Alternatives:

- `sentence-transformers/all-mpnet-base-v2` (higher quality, slower)
- `sentence-transformers/all-MiniLM-L12-v2` (balanced)

## Project Structure

```
project_root/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Gutenberg download & text processing
│   ├── pathway_setup.py        # Pathway vector store initialization
│   ├── claim_extractor.py      # Backstory parsing & claim extraction
│   ├── retrieval.py            # Hybrid search (semantic + keyword)
│   ├── consistency_checker.py  # LLM-based consistency checking
│   └── classifier.py          # Evidence aggregation & final prediction
├── main.py                     # Orchestration & test case execution
├── requirements.txt            # All dependencies
├── config.yaml                 # Configuration
├── README.md                   # This file
├── results.csv                 # Generated output
├── pipeline.log                # Log file
└── cache/
    └── books/                  # Cached Gutenberg books
```

## Code Quality

- **Type Hints**: All functions include type hints
- **Docstrings**: Google-style docstrings for all modules
- **Logging**: Comprehensive logging (not print statements)
- **Error Handling**: Try-except blocks at module boundaries
- **Config-Driven**: No hardcoded values

## Performance Optimizations

- Model loaded once, reused for all inferences
- Batch processing for multiple claims
- `torch.no_grad()` for inference efficiency
- Incremental CSV writing
- Local caching of downloaded books

## License

This project is part of the Kharagpur Data Science Hackathon - Track A.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log file: `pipeline.log`
3. Verify configuration in `config.yaml`
