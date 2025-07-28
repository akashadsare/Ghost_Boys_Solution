#!/usr/bin/env python3
"""Download and cache models for offline execution."""

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download and cache models for offline use."""
    try:
        logger.info("Downloading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer model downloaded successfully")
        
        logger.info("Downloading summarization model...")
        summarizer = pipeline("summarization", model="t5-small", device=-1)
        logger.info("Summarization model downloaded successfully")
        
        logger.info("All models downloaded and cached successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise

if __name__ == "__main__":
    download_models()