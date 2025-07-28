import json
import logging
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import fitz  
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(data: str) -> str:
    """Generate cache key from data."""
    return hashlib.md5(data.encode()).hexdigest()

def load_from_cache(cache_key: str) -> Any:
    """Load data from cache."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None

def save_to_cache(cache_key: str, data: Any) -> None:
    """Save data to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF pages with caching."""
    cache_key = get_cache_key(f"pdf_{pdf_path}_{pdf_path.stat().st_mtime}")
    cached_result = load_from_cache(cache_key)
    if cached_result:
        return cached_result
    
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            if text:
                texts.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        doc.close()
        save_to_cache(cache_key, texts)
        return texts
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return []

def extract_section_title(text: str) -> str:
    """Extract section title with improved heuristics."""
    lines = text.split('\n')
    
    # Look for headers (all caps, title case, or bold indicators)
    for line in lines[:10]:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
            
        # Skip obvious non-titles
        if (stripped.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.', 'Page', 'Figure', 'Table', 'www.', 'http')) or
            stripped.isdigit() or len(stripped) > 120 or
            stripped.count('.') > 3 or stripped.count(',') > 3):
            continue
            
        # Good title indicators
        if (15 <= len(stripped) <= 80 and
            (stripped.isupper() or stripped.istitle() or 
             any(word.isupper() for word in stripped.split()[:3])) and
            not stripped.endswith(('.', '!', '?', ':')) and
            stripped.count(' ') >= 1):
            return stripped
    
    # Fallback: first substantial line
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and 10 <= len(stripped) <= 100:
            return stripped[:75] + "..." if len(stripped) > 75 else stripped
            
    return "Untitled Section"

def process_pdf_batch(pdf_paths_and_docs: List[tuple], pdfs_dir: Path) -> List[Dict[str, Any]]:
    """Process multiple PDFs in parallel."""
    all_texts = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_doc = {executor.submit(extract_text_from_pdf, pdfs_dir / doc_info["filename"]): doc_info 
                        for doc_info in pdf_paths_and_docs if (pdfs_dir / doc_info["filename"]).exists()}
        
        for future in as_completed(future_to_doc):
            doc_info = future_to_doc[future]
            try:
                pdf_texts = future.result()
                for text_info in pdf_texts:
                    all_texts.append({
                        "document": doc_info["filename"],
                        "page_number": text_info["page_number"],
                        "text": text_info["text"]
                    })
            except Exception as e:
                logger.warning(f"Failed to process {doc_info['filename']}: {e}")
    return all_texts

def process_collection(collection_dir: Path, model: SentenceTransformer, summarizer) -> bool:
    """Process a single collection."""
    try:
        input_file = collection_dir / "challenge1b_input.json"
        output_file = collection_dir / "challenge1b_output.json"
        pdfs_dir = collection_dir / "PDFs"
        
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            return False
            
        if not pdfs_dir.exists():
            logger.warning(f"PDFs directory not found: {pdfs_dir}")
            return False

        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        persona = input_data["persona"]["role"]
        job_to_be_done = input_data["job_to_be_done"]["task"]
        query = f"As a {persona}, I need to {job_to_be_done}. What are the most relevant sections from these documents?"

        # Process PDFs in parallel
        all_texts = process_pdf_batch(input_data["documents"], pdfs_dir)

        if not all_texts:
            logger.error(f"No text extracted from {collection_dir}")
            return False

        # Enhanced semantic search
        try:
            texts_for_embedding = [item["text"] for item in all_texts]
            
            # Create enhanced query
            enhanced_query = f"{persona} {job_to_be_done}"
            
            corpus_cache_key = get_cache_key(f"corpus_{''.join(texts_for_embedding[:2])}")
            corpus_embeddings = load_from_cache(corpus_cache_key)
            
            if corpus_embeddings is None:
                corpus_embeddings = model.encode(texts_for_embedding, convert_to_tensor=True, batch_size=8)
                save_to_cache(corpus_cache_key, corpus_embeddings)
            
            query_cache_key = get_cache_key(f"query_{enhanced_query}")
            query_embedding = load_from_cache(query_cache_key)
            
            if query_embedding is None:
                query_embedding = model.encode(enhanced_query, convert_to_tensor=True)
                save_to_cache(query_cache_key, query_embedding)
            
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(30, len(all_texts)))
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return False

        # Better section extraction with diversity
        extracted_sections = []
        subsection_analysis = []
        doc_scores = {}  # Track best score per document
        
        # Group results by document and get best from each
        for score, idx in zip(top_results[0], top_results[1]):
            item = all_texts[idx]
            doc_name = item["document"]
            
            if doc_name not in doc_scores or score > doc_scores[doc_name]["score"]:
                doc_scores[doc_name] = {
                    "score": float(score),
                    "item": item,
                    "idx": idx
                }
        
        # Sort by score and take top 5
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
        
        for rank, (doc_name, doc_data) in enumerate(sorted_docs, 1):
            item = doc_data["item"]
            section_title = extract_section_title(item["text"])
            
            extracted_sections.append({
                "document": item["document"],
                "section_title": section_title,
                "importance_rank": rank,
                "page_number": item["page_number"]
            })
            
            # Enhanced summarization
            text_to_summarize = item["text"][:800]
            if len(text_to_summarize) > 50:
                summary_cache_key = get_cache_key(f"summary_{persona}_{text_to_summarize[:200]}")
                refined_text = load_from_cache(summary_cache_key)
                
                if refined_text is None:
                    try:
                        context_text = f"For {persona}: {text_to_summarize}"
                        summary = summarizer(context_text[:400], max_length=80, min_length=15, do_sample=False)
                        refined_text = summary[0]["summary_text"]
                        save_to_cache(summary_cache_key, refined_text)
                    except Exception as e:
                        logger.warning(f"Summarization failed for {item['document']}: {e}")
                        refined_text = item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"]
            else:
                refined_text = text_to_summarize
            
            subsection_analysis.append({
                "document": item["document"],
                "refined_text": refined_text,
                "page_number": item["page_number"]
            })

        output_data = {
            "metadata": {
                "input_documents": [d["filename"] for d in input_data["documents"]],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Successfully processed {collection_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing collection {collection_dir}: {e}")
        return False

def process_collections(base_path: str = ".", num_collections: int = 3):
    try:
        # Load models once
        logger.info("Loading models...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer = pipeline("summarization", model="t5-small", device=-1)
        logger.info("Models loaded successfully")

        base_collection_path = Path(base_path)
        if not base_collection_path.exists():
            logger.error(f"Base collection path does not exist: {base_collection_path}")
            return

        # Process collections in parallel
        collection_dirs = [base_collection_path / f"Collection {i}" for i in range(1, num_collections + 1)]
        successful_collections = 0
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_dir = {executor.submit(process_collection, collection_dir, model, summarizer): collection_dir 
                           for collection_dir in collection_dirs}
            
            for future in as_completed(future_to_dir):
                if future.result():
                    successful_collections += 1

        logger.info(f"Processing complete. {successful_collections}/{num_collections} collections processed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in process_collections: {e}")
        raise

if __name__ == "__main__":
    process_collections()