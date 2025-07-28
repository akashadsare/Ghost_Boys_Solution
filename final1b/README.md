# Persona-Driven Document Intelligence Solution

## Overview
This solution processes document collections to extract and prioritize the most relevant sections based on a specific persona and their job-to-be-done using advanced semantic search and AI summarization.

## Features
- **Semantic Search**: Uses sentence transformers for intelligent content matching
- **Persona-Aware Processing**: Tailors results to specific user roles and tasks
- **Document Diversity**: Ensures representation across all input documents
- **Intelligent Summarization**: Generates contextual summaries using T5 model
- **Performance Optimized**: Multi-threading, caching, and batch processing
- **CPU-Only Execution**: No GPU requirements, runs on standard hardware

## Models Used
- **Sentence Transformer**: `all-MiniLM-L6-v2` - For semantic search and text embeddings
- **Summarization**: `t5-small` - For persona-aware text summarization
- **Total Model Size**: <1GB (meets hackathon constraints)
- **Offline Execution**: Models pre-downloaded during Docker build, no internet required

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Build the image (downloads models for offline execution)
docker build -t document-intelligence .

# Run processing (no internet required)
docker run --rm -v $(pwd):/app document-intelligence
```

### Option 2: Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download models (required for offline execution)
python download_models.py

# Run processing
python process_collections.py
```

## Input Structure
```
Collection X/
├── challenge1b_input.json    # Persona and job definition
├── challenge1b_output.json   # Generated results
└── PDFs/                     # Document collection
    ├── document1.pdf
    └── document2.pdf
```

## Output Format
The solution generates `challenge1b_output.json` files containing:
- **Metadata**: Input documents, persona, job-to-be-done, timestamp
- **Extracted Sections**: Top 5 ranked sections with titles and page numbers
- **Subsection Analysis**: Persona-aware summaries for each section

## Performance
- **Processing Time**: <60 seconds for 3-10 documents
- **Model Size**: <1GB total (all-MiniLM-L6-v2 + T5-small)
- **Memory Usage**: ~2GB RAM during processing
- **Caching**: Automatic caching for repeated runs

## Architecture
- **Text Extraction**: PyMuPDF for PDF processing
- **Semantic Search**: Sentence Transformers with cosine similarity
- **Summarization**: T5-small with persona context
- **Parallel Processing**: ThreadPoolExecutor for performance
- **Caching**: Pickle-based caching system

## Files
- `process_collections.py` - Main processing script
- `download_models.py` - Model pre-download utility
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `approach_explanation.md` - Detailed methodology