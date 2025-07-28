import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
import re
import unicodedata
from ultralytics import YOLO
import time
import concurrent.futures
import threading
import numpy as np
import gc
from functools import lru_cache
import torch
import psutil

# AGGRESSIVE OPTIMIZATION CONFIGURATION
YOLO_MODEL_PATH = "/app/models/custom_yolo_model.pt"
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.4
IMAGE_DPI = 100
TARGET_CLASSES = ['Title', 'Section-header']

# Ultra-aggressive settings
NUM_THREADS = min(psutil.cpu_count(logical=False) or 4, 8)
BATCH_SIZE = 16
MAX_CACHE_SIZE = 1000
YOLO_IMGSZ = 320

# Memory and processing optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Thread-local storage for models
thread_local_data = threading.local()

def get_thread_local_model():
    """Get or create thread-local model instance"""
    if not hasattr(thread_local_data, 'model'):
        thread_local_data.model = YOLO(YOLO_MODEL_PATH)
        thread_local_data.model.model.eval()
        thread_local_data.model.to('cpu')
        if hasattr(thread_local_data.model.model, 'fuse'):
            thread_local_data.model.model.fuse()
        thread_local_data.model_names = thread_local_data.model.names
    return thread_local_data.model, thread_local_data.model_names

def pdf_to_images_ultra_fast(pdf_path, dpi=IMAGE_DPI):
    """Ultra-fast PDF to images with minimal memory allocation"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        images = [None] * total_pages

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
            img_data = pix.pil_tobytes(format="JPEG", optimize=True, quality=75)
            images[page_num] = Image.open(io.BytesIO(img_data))
            pix = None
            page = None
        doc.close()
        print(f"Converted {total_pages} pages at {dpi} DPI")
        return images, total_pages
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return [], 0

@lru_cache(maxsize=MAX_CACHE_SIZE)
def ultra_fast_text_extraction_v2(pdf_path, page_num, x1, y1, x2, y2, img_dpi):
    """Even faster text extraction with aggressive caching"""
    try:
        doc = fitz.open(pdf_path)
        pdf_page = doc.load_page(page_num)
        scale = img_dpi / 72.0
        rect = fitz.Rect(x1/scale, y1/scale, x2/scale, y2/scale).intersect(pdf_page.rect)
        text = pdf_page.get_textbox(rect).strip()
        doc.close()
        return text
    except Exception as e:
        return ""

def find_document_title_on_first_page(pdf_path):
    """
    Analyzes all text on the first page to find the most likely title
    based on the largest font size.
    """
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return ""

        page = doc.load_page(0)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]

        max_font_size = 0
        potential_title = ""

        if not blocks:
            doc.close()
            return ""

        # Find the maximum font size on the page
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]

        # Collect all text with that maximum font size
        title_candidates = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            if abs(span["size"] - max_font_size) < 0.1:
                                title_candidates.append(span["text"].strip())

        doc.close()
        potential_title = " ".join(title_candidates)

        # Basic cleaning and validation
        potential_title = re.sub(r'\s+', ' ', potential_title).strip()
        if len(potential_title) > 5 and len(potential_title) < 200:
             print(f"Robustly detected title: '{potential_title}'")
             return potential_title

        return ""
    except Exception as e:
        print(f"Error during robust title detection: {e}")
        return ""

def process_batch_ultra_speed(batch_data_with_pdf):
    """Ultra-speed batch processing with thread-local models"""
    pdf_path, page_batch, img_dpi = batch_data_with_pdf
    try:
        model, model_names = get_thread_local_model()
        images = [item['image'] for item in page_batch]
        page_nums = [item['page_num'] for item in page_batch]

        results = model.predict(
            images, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False,
            save=False, show=False, augment=False, half=False, device='cpu',
            max_det=20, agnostic_nms=True, classes=[7], imgsz=YOLO_IMGSZ
        )

        batch_results = []
        for page_num, yolo_result in zip(page_nums, results):
            page_elements = []
            if hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                boxes = yolo_result.boxes
                coords = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                for j in range(len(coords)):
                    if float(confidences[j]) >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = coords[j].astype(int)
                        text = ultra_fast_text_extraction_v2(
                            pdf_path, page_num, int(x1), int(y1), int(x2), int(y2), img_dpi
                        )
                        if text and len(text.strip()) > 1:
                            page_elements.append({
                                "page_num": page_num + 1,
                                "text": text,
                                "y0_coord": int(y1)
                            })
            batch_results.append((page_num, page_elements))
        return batch_results
    except Exception as e:
        print(f"Batch processing error: {e}")
        return [(item['page_num'], []) for item in page_batch]
def detect_language(text):
    """Simple language detection for multilingual support"""
    # Devanagari script (Hindi, Marathi, Sanskrit)
    if re.search(r'[\u0900-\u097F]', text):
        # Check for Marathi-specific characters or patterns
        if re.search(r'[\u0933\u0934\u0931\u0930]', text):  # ळ, ऴ, र्, र patterns common in Marathi
            return 'mr'
        return 'hi'
    # Japanese characters (Hiragana, Katakana, Kanji)
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
        return 'ja'
    # Chinese characters
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    # Korean characters
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return 'ko'
    # Arabic characters
    elif re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    # Cyrillic characters (Russian, etc.)
    elif re.search(r'[\u0400-\u04FF]', text):
        return 'ru'
    return 'en'

def lightning_fast_heading_classifier(header_text):
    """Enhanced multilingual heading classification supporting H1-H6 levels"""
    text = header_text.strip()
    lang = detect_language(text)
    
    # Handle multi-line text
    if '\n' in text:
        lines = text.split('\n')
        first_line = lines[0].strip()
        if first_line:
            text = first_line
    
    # Pattern 1: Universal numbered sections (1, 1.1, 1.1.1, etc.)
    number_match = re.match(r'^(\d+(?:\.\d+)*)', text)
    if number_match:
        number_part = number_match.group(1)
        dot_count = number_part.count('.')
        
        if dot_count == 0:  # "1", "2", etc.
            return "H1"
        elif dot_count == 1:  # "1.1", "2.3", etc.
            return "H2"
        elif dot_count >= 2:  # "1.1.1", "1.1.1.1", etc.
            return "H3"
    
    # Pattern 2: Language-specific numbering patterns
    if lang == 'ja':
        # Japanese chapter markers: 第1章, 第2節, etc.
        if re.match(r'^第[\d一二三四五六七八九十]+[章節項]', text):
            if '章' in text[:5]:
                return "H1"
            elif '節' in text[:5]:
                return "H2"
            elif '項' in text[:5]:
                return "H3"
        
        # Japanese bullet points: ・, ○, ●
        if re.match(r'^[・○●]', text):
            return "H3"
    
    elif lang in ['hi', 'mr']:
        # Hindi/Marathi chapter markers: अध्याय १, भाग २, etc.
        if re.match(r'^(अध्याय|प्रकरण|भाग|खंड)[\s\d०१२३४५६७८९]+', text):
            return "H1"
        
        # Hindi/Marathi section markers
        if re.match(r'^(विभाग|उपविभाग|अनुभाग)[\s\d०१२३४५६७८९]+', text):
            return "H2"
        
        # Devanagari bullet points: •, ०, १, २
        if re.match(r'^[•०१२३४५६७८९]\s', text):
            return "H3"
    
    # Pattern 3: Roman numerals
    roman_match = re.match(r'^([IVX]+)\.?\s', text.upper())
    if roman_match:
        return "H2"
    
    # Pattern 4: Letters (A, B, C or a, b, c)
    letter_match = re.match(r'^([A-Za-z])\.?\s', text)
    if letter_match:
        if letter_match.group(1).isupper():
            return "H2"
        else:
            return "H3"
    
    # Pattern 5: Universal bullet points
    if re.match(r'^[-•·▪▫◦‣⁃・○●]\s', text):
        return "H3"
    
    # Pattern 6: Multilingual keywords for major sections (H1)
    h1_keywords = {
        'en': ['abstract', 'introduction', 'background', 'literature review',
               'methodology', 'methods', 'results', 'discussion', 'conclusion',
               'references', 'bibliography', 'appendix', 'acknowledgments',
               'executive summary', 'overview', 'summary'],
        'hi': ['सार', 'सारांश', 'परिचय', 'प्रस्तावना', 'पृष्ठभूमि', 'विधि', 'पद्धति', 
               'परिणाम', 'चर्चा', 'निष्कर्ष', 'संदर्भ', 'ग्रंथसूची', 'परिशिष्ट', 'आभार'],
        'mr': ['सारांश', 'प्रस्तावना', 'पार्श्वभूमी', 'पद्धती', 'परिणाम', 'चर्चा', 
               'निष्कर्ष', 'संदर्भ', 'परिशिष्ट', 'आभार'],
        'ja': ['要約', '概要', '序論', '導入', '背景', '手法', '方法', '結果', 
               '考察', '議論', '結論', '参考文献', '付録', '謝辞', 'まとめ'],
        'zh': ['摘要', '概述', '引言', '背景', '方法', '结果', '讨论', '结论', 
               '参考文献', '附录', '致谢', '总结'],
        'ko': ['요약', '개요', '서론', '배경', '방법', '결과', '토론', '결론', 
               '참고문헌', '부록', '감사의 말'],
        'ar': ['ملخص', 'مقدمة', 'خلفية', 'منهجية', 'نتائج', 'مناقشة', 'خاتمة', 
               'مراجع', 'ملحق'],
        'ru': ['аннотация', 'введение', 'методы', 'результаты', 'обсуждение', 
               'заключение', 'литература', 'приложение']
    }
    
    text_lower = text.lower()
    if lang in h1_keywords:
        if any(keyword in text_lower for keyword in h1_keywords[lang]):
            return "H1"
    
    # Pattern 7: Multilingual subsection keywords (H2)
    h2_keywords = {
        'en': ['objectives', 'scope', 'limitations', 'assumptions', 'findings',
               'analysis', 'implementation', 'evaluation', 'recommendations',
               'future work', 'related work'],
        'hi': ['उद्देश्य', 'क्षेत्र', 'सीमाएं', 'मान्यताएं', 'खोजें', 'विश्लेषण', 
               'कार्यान्वयन', 'मूल्यांकन', 'सिफारिशें', 'भविष्य का कार्य', 'संबंधित कार्य'],
        'mr': ['उद्दिष्टे', 'व्याप्ती', 'मर्यादा', 'गृहीतके', 'शोध', 'विश्लेषण', 
               'अंमलबजावणी', 'मूल्यमापन', 'शिफारसी', 'भावी कार्य'],
        'ja': ['目的', '範囲', '制限', '仮定', '発見', '分析', '実装', '評価', 
               '推奨', '今後の課題', '関連研究'],
        'zh': ['目标', '范围', '限制', '假设', '发现', '分析', '实现', '评估', 
               '建议', '未来工作', '相关工作'],
        'ko': ['목표', '범위', '제한', '가정', '발견', '분석', '구현', '평가', 
               '권장사항', '향후 작업'],
        'ar': ['أهداف', 'نطاق', 'قيود', 'افتراضات', 'نتائج', 'تحليل', 'تنفيذ', 
               'تقييم', 'توصيات'],
        'ru': ['цели', 'область', 'ограничения', 'предположения', 'находки', 
               'анализ', 'реализация', 'оценка', 'рекомендации']
    }
    
    if lang in h2_keywords:
        if any(keyword in text_lower for keyword in h2_keywords[lang]):
            return "H2"
    
    # Pattern 8: Text length and structure heuristics
    text_len = len(text)
    
    # Adjust length thresholds for different languages
    if lang in ['ja', 'zh', 'ko']:  # CJK languages are more compact
        short_threshold = 10
        medium_threshold = 25
        long_threshold = 40
    elif lang in ['hi', 'mr']:  # Devanagari script is moderately compact
        short_threshold = 15
        medium_threshold = 35
        long_threshold = 50
    else:
        short_threshold = 20
        medium_threshold = 50
        long_threshold = 60
    
    if text_len < short_threshold:
        return "H3"
    
    if text_len < medium_threshold:
        # Check for question format (universal)
        if text.endswith('?') or text.endswith('？'):
            return "H3"
        # Check for "How to", "What is", etc. patterns
        if re.match(r'^(how to|what is|why|when|where)', text_lower):
            return "H3"
        # Japanese question patterns
        if lang == 'ja' and (text.endswith('か') or 'とは' in text):
            return "H3"
        # Hindi/Marathi question patterns
        if lang in ['hi', 'mr'] and (text.endswith('?') or 'क्या' in text or 'कैसे' in text):
            return "H3"
        return "H2"
    
    # Pattern 9: All caps (often major headings)
    if text.isupper() and text_len > 5:
        return "H1"
    
    # Pattern 10: Title case detection (mainly for Latin scripts)
    if lang == 'en':
        words = text.split()
        if len(words) > 1:
            title_case_count = sum(1 for word in words if word[0].isupper() and len(word) > 2)
            if title_case_count >= len(words) * 0.7:  # 70% of words are title case
                if text_len > long_threshold:
                    return "H1"
                else:
                    return "H2"
    
    # Pattern 11: Default classification based on position and length
    if text_len > long_threshold:
        return "H1"
    elif text_len > medium_threshold:
        return "H2"
    else:
        return "H3"


'''
def lightning_fast_heading_classifier(header_text):
    """Lightning fast heading classification"""
    text = header_text.strip()
    if '\n' in text:
        lines = text.split('\n', 1)
        first_line = lines[0].strip()
        if re.match(r'^\d+$', first_line): return "H1"
        if re.match(r'^\d+\.\d+$', first_line): return "H2"
        text = first_line

    if text[:2].isdigit():
        return "H2" if '.' in text[:5] else "H1"
    if any(kw in text[:20].lower() for kw in ['intro', 'concl', 'method', 'result', 'discuss', 'ref']):
        return "H1"
    return "H2"
''' 

def process_pdf_ultra_performance(pdf_path):
    """Ultra-performance PDF processing targeting 5 pages/second"""
    print("Starting ultra-performance PDF processing...")
    start_time = time.time()

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return {"title": "", "outline": []}

    document_title = find_document_title_on_first_page(pdf_path)

    all_images, total_pages = pdf_to_images_ultra_fast(pdf_path)
    if not all_images:
        return {"title": document_title, "outline": []}

    print(f"Processing {total_pages} pages with {NUM_THREADS} threads, batch size {BATCH_SIZE}")

    batches = []
    for i in range(0, len(all_images), BATCH_SIZE):
        batch_items = [{'page_num': j, 'image': all_images[j]} for j in range(i, min(i + BATCH_SIZE, len(all_images)))]
        batches.append((pdf_path, batch_items, IMAGE_DPI))

    all_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_batch = {executor.submit(process_batch_ultra_speed, batch_data): i for i, batch_data in enumerate(batches)}
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = future.result(timeout=45)
                for page_num, elements in batch_results:
                    all_results[page_num] = elements
            except Exception as exc:
                print(f"Batch error: {exc}")

    all_images = None
    gc.collect()

    all_headers = [element for page_num in sorted(all_results.keys()) for element in all_results[page_num]]
    all_headers.sort(key=lambda h: (h['page_num'], h['y0_coord']))

    final_outline = [{"level": lightning_fast_heading_classifier(h['text']), "text": h['text'], "page": h['page_num']} for h in all_headers]

    end_time = time.time()
    processing_time = end_time - start_time
    rate = total_pages / processing_time if processing_time > 0 else 0

    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Processing rate: {rate:.1f} pages/second")
    print(f"Found {len(final_outline)} headers")
    if rate >= 5.0: print("🎯 TARGET ACHIEVED: 5+ pages/second!")
    else: print(f"📈 Progress: {(rate/5.0)*100:.1f}% of target speed")

    return {"title": document_title, "outline": final_outline}

def apply_system_optimizations():
    """Apply system-level optimizations"""
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

def main():
    apply_system_optimizations()
    input_dir = "/app/input"
    output_dir = "/app/output"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found. Please ensure PDFs are mounted.")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_filename)
        output_filename = os.path.splitext(pdf_filename)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\nProcessing '{pdf_filename}'...")
        try:
            final_output = process_pdf_ultra_performance(pdf_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            print(f"Successfully processed '{pdf_filename}'. Output saved to '{output_path}'")
        except Exception as e:
            print(f"Error processing '{pdf_filename}': {e}")

if __name__ == "__main__":
    main()
