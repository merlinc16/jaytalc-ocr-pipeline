#!/Users/mc/jaytalc/venv/bin/python3
"""
Complete OCR Pipeline: Document AI text + Tesseract positioning + JBIG2 compression

Supports both PDF and multi-page TIFF input.

TRUE HYBRID APPROACH:
- Document AI OCR for high-quality text recognition
- Tesseract hOCR for pixel-accurate text positioning (proven to work)
- JBIG2 compression for ABBYY-identical output
- Clean text extraction via pdftotext/Tika
"""

import os
import re
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import pikepdf
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from openai import OpenAI

# Configuration
PROJECT_ID = "toxicdocs"
LOCATION = "us"
DPI = 300
PAGE_WIDTH = 612   # PDF points (letter)
PAGE_HEIGHT = 792


@dataclass
class Word:
    """Word with bounding box (pixel coordinates)."""
    text: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class PageOCR:
    """OCR data for a page."""
    page_num: int
    words: List[Word] = field(default_factory=list)
    width: int = 2550
    height: int = 3300


def get_tiff_dimensions(tiff_path: Path) -> Tuple[int, int]:
    """Get TIFF dimensions."""
    result = subprocess.run(
        ["magick", "identify", "-format", "%w %h", str(tiff_path)],
        capture_output=True, text=True, check=True
    )
    w, h = result.stdout.strip().split()
    return int(w), int(h)


def preprocess_pdf(pdf_path: Path, work_dir: Path) -> List[Path]:
    """Convert PDF to 1-bit B&W TIFFs at 300 DPI."""
    pdf_path = Path(pdf_path).resolve()

    # PDF to PNG
    png_prefix = work_dir / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(DPI), str(pdf_path), str(png_prefix)],
        capture_output=True, check=True
    )

    png_files = sorted(work_dir.glob("page-*.png")) or sorted(work_dir.glob("page*.png"))

    # PNG to 1-bit TIFF
    tiff_files = []
    for png in png_files:
        tiff = work_dir / f"{png.stem}.tif"
        subprocess.run(
            ["magick", str(png), "-auto-threshold", "otsu",
             "-depth", "1", "-compress", "none", str(tiff)],
            capture_output=True, check=True
        )
        tiff_files.append(tiff)

    return tiff_files


def preprocess_tiff(tiff_path: Path, work_dir: Path) -> List[Path]:
    """Convert multi-page TIFF to 1-bit B&W TIFFs at 300 DPI."""
    tiff_path = Path(tiff_path).resolve()

    # Get page count
    result = subprocess.run(
        ["magick", "identify", str(tiff_path)],
        capture_output=True, text=True
    )
    page_count = len(result.stdout.strip().split('\n'))

    tiff_files = []
    for i in range(page_count):
        output_tiff = work_dir / f"page-{i+1:03d}.tif"

        # Extract page, resize to 300 DPI letter size if needed, convert to 1-bit
        # Using [i] to select specific page from multi-page TIFF
        subprocess.run(
            ["magick", f"{tiff_path}[{i}]",
             "-density", str(DPI),
             "-resize", "2550x3300",  # 8.5x11 at 300 DPI
             "-auto-threshold", "otsu",
             "-depth", "1", "-compress", "none", str(output_tiff)],
            capture_output=True, check=True
        )
        tiff_files.append(output_tiff)

    return tiff_files


def preprocess_input(input_path: Path, work_dir: Path) -> List[Path]:
    """Convert PDF or TIFF to 1-bit B&W TIFFs."""
    input_path = Path(input_path).resolve()
    suffix = input_path.suffix.lower()

    if suffix == '.pdf':
        return preprocess_pdf(input_path, work_dir)
    elif suffix in ('.tif', '.tiff'):
        return preprocess_tiff(input_path, work_dir)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def parse_hocr_bbox(title_attr: str) -> Tuple[int, int, int, int]:
    """Extract bbox from hOCR title attribute."""
    match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title_attr)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0, 0)


def run_tesseract_hocr(tiff_files: List[Path], work_dir: Path) -> List[PageOCR]:
    """Run tesseract to get pixel-accurate word positions via hOCR."""
    ocr_results = []

    for i, tiff_file in enumerate(tiff_files):
        hocr_base = work_dir / f"tess_{i:04d}"
        subprocess.run(
            ["tesseract", str(tiff_file), str(hocr_base), "-l", "eng", "hocr"],
            capture_output=True, check=True
        )

        hocr_file = Path(f"{hocr_base}.hocr")
        width, height = get_tiff_dimensions(tiff_file)
        page_ocr = PageOCR(page_num=i + 1, width=width, height=height)

        if hocr_file.exists():
            tree = ET.parse(hocr_file)
            root = tree.getroot()

            for elem in root.iter():
                if 'ocrx_word' in elem.get('class', ''):
                    bbox = parse_hocr_bbox(elem.get('title', ''))
                    text = ''.join(elem.itertext()).strip()
                    if text and bbox[2] > bbox[0]:
                        page_ocr.words.append(Word(
                            text=text, x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]
                        ))

        ocr_results.append(page_ocr)

    return ocr_results


def run_document_ai_ocr(
    tiff_files: List[Path],
    docai_client: documentai.DocumentProcessorServiceClient,
    processor_name: str
) -> List[PageOCR]:
    """Run Document AI OCR to get high-quality text with positions."""
    ocr_results = []

    for i, tiff_file in enumerate(tiff_files):
        with open(tiff_file, "rb") as f:
            content = f.read()

        width, height = get_tiff_dimensions(tiff_file)

        raw_doc = documentai.RawDocument(content=content, mime_type="image/tiff")
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_doc)
        result = docai_client.process_document(request=request)
        document = result.document

        page_ocr = PageOCR(page_num=i + 1, width=width, height=height)

        for page in document.pages:
            for token in page.tokens:
                text = ""
                if token.layout.text_anchor and token.layout.text_anchor.text_segments:
                    for seg in token.layout.text_anchor.text_segments:
                        start = int(seg.start_index) if seg.start_index else 0
                        end = int(seg.end_index)
                        text += document.text[start:end]
                text = text.strip()
                if not text:
                    continue

                if token.layout.bounding_poly.normalized_vertices:
                    verts = token.layout.bounding_poly.normalized_vertices
                    x1 = int(min(v.x for v in verts) * width)
                    y1 = int(min(v.y for v in verts) * height)
                    x2 = int(max(v.x for v in verts) * width)
                    y2 = int(max(v.y for v in verts) * height)
                    page_ocr.words.append(Word(text=text, x1=x1, y1=y1, x2=x2, y2=y2))

        ocr_results.append(page_ocr)

    return ocr_results


def merge_ocr_data(tess_data: List[PageOCR], docai_data: List[PageOCR]) -> List[PageOCR]:
    """
    Merge tesseract positions with Document AI text.

    For each tesseract word, find the overlapping Document AI word and use its text.
    This gives us: tesseract's pixel-accurate positions + Document AI's superior text.
    """
    merged = []

    for tess_page, docai_page in zip(tess_data, docai_data):
        merged_page = PageOCR(
            page_num=tess_page.page_num,
            width=tess_page.width,
            height=tess_page.height
        )

        # Build spatial index of Document AI words
        docai_words = docai_page.words

        for tess_word in tess_page.words:
            # Find best matching Document AI word by overlap
            best_match = None
            best_overlap = 0

            for docai_word in docai_words:
                # Calculate overlap area
                x_overlap = max(0, min(tess_word.x2, docai_word.x2) - max(tess_word.x1, docai_word.x1))
                y_overlap = max(0, min(tess_word.y2, docai_word.y2) - max(tess_word.y1, docai_word.y1))
                overlap = x_overlap * y_overlap

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = docai_word

            # Use Document AI text with tesseract position
            if best_match and best_overlap > 0:
                merged_page.words.append(Word(
                    text=best_match.text,  # Document AI text (better quality)
                    x1=tess_word.x1,       # Tesseract position (pixel-accurate)
                    y1=tess_word.y1,
                    x2=tess_word.x2,
                    y2=tess_word.y2
                ))
            else:
                # No match found, use tesseract text as fallback
                merged_page.words.append(tess_word)

        merged.append(merged_page)

    return merged


def postprocess_ocr_with_llm(ocr_data: List[PageOCR], openai_client: OpenAI) -> List[PageOCR]:
    """
    Use GPT-4o-mini to fix common OCR errors while preserving word positions.
    Processes page by page, sending words in reading order.
    """
    if not openai_client:
        return ocr_data

    corrected = []

    for page in ocr_data:
        if not page.words:
            corrected.append(page)
            continue

        # Build text with word indices for mapping back
        words_text = [w.text for w in page.words]
        original_text = " ".join(words_text)

        # Skip if too few words
        if len(words_text) < 5:
            corrected.append(page)
            continue

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an OCR error correction assistant. Fix common OCR errors in the text while preserving the exact word count and order.

Common OCR errors to fix:
- "rn" misread as "m" or vice versa (e.g., "bum" → "burn", "modem" → "modern")
- "l" confused with "1" or "I"
- "0" confused with "O"
- "cl" misread as "d"
- Split words that should be joined
- Joined words that should be split (maintain word count by using hyphens if needed)
- Obvious misspellings from character confusion

CRITICAL: You must return EXACTLY the same number of words as the input. Do not add or remove words. If unsure, keep the original word."""
                    },
                    {
                        "role": "user",
                        "content": f"Fix OCR errors in this text ({len(words_text)} words). Return exactly {len(words_text)} words:\n\n{original_text}"
                    }
                ],
                temperature=0.1,
                max_tokens=min(len(original_text) * 2, 16000)
            )

            corrected_text = response.choices[0].message.content.strip()
            corrected_words = corrected_text.split()

            # Only use correction if word count matches
            if len(corrected_words) == len(words_text):
                new_page = PageOCR(
                    page_num=page.page_num,
                    width=page.width,
                    height=page.height
                )
                for i, word in enumerate(page.words):
                    new_page.words.append(Word(
                        text=corrected_words[i],
                        x1=word.x1,
                        y1=word.y1,
                        x2=word.x2,
                        y2=word.y2
                    ))
                corrected.append(new_page)
            else:
                # Word count mismatch, use original
                corrected.append(page)

        except Exception as e:
            print(f"    Warning: LLM post-processing failed for page {page.page_num}: {e}")
            corrected.append(page)

    return corrected


def detect_columns(words: List[Word], page_width: int, page_height: int) -> List[List[Word]]:
    """
    Detect multiple columns by finding vertical gaps in word positions.
    Uses histogram analysis to find gutters between columns.
    Returns list of word lists, one per column, ordered left to right.
    """
    if not words or len(words) < 5:
        return [words] if words else []

    # Build histogram of word coverage across X axis
    # Each bin represents a small horizontal slice
    num_bins = 100
    bin_width = page_width / num_bins
    histogram = [0] * num_bins

    for word in words:
        start_bin = int(word.x1 / bin_width)
        end_bin = int(word.x2 / bin_width)
        for b in range(max(0, start_bin), min(num_bins, end_bin + 1)):
            histogram[b] += 1

    # Find gaps (bins with zero or very low coverage)
    # These represent gutters between columns
    threshold = max(histogram) * 0.05  # Gaps have < 5% of max coverage

    # Find continuous regions of low coverage (gutters)
    gutters = []
    in_gutter = False
    gutter_start = 0

    for i, count in enumerate(histogram):
        if count <= threshold:
            if not in_gutter:
                in_gutter = True
                gutter_start = i
        else:
            if in_gutter:
                gutter_end = i
                # Only count as gutter if it's wide enough (at least 2% of page)
                if gutter_end - gutter_start >= 2:
                    gutter_center = (gutter_start + gutter_end) / 2 * bin_width
                    # Don't count gutters at page edges
                    if gutter_center > page_width * 0.05 and gutter_center < page_width * 0.95:
                        gutters.append(gutter_center)
                in_gutter = False

    # If no gutters found, return single column
    if not gutters:
        return [words]

    # Sort gutters and use them as column boundaries
    gutters.sort()

    # Create columns based on boundaries
    boundaries = [0] + gutters + [page_width]
    columns = []

    for i in range(len(boundaries) - 1):
        left_bound = boundaries[i]
        right_bound = boundaries[i + 1]
        col_words = [w for w in words if left_bound <= (w.x1 + w.x2) / 2 < right_bound]
        if col_words:
            columns.append(col_words)

    return columns if columns else [words]


def group_into_lines(page_ocr: PageOCR) -> List[Tuple[str, int, int, int, int]]:
    """
    Group words into lines with proper column handling.
    Processes each column separately, then outputs column by column.
    Returns (text, x1, y1, x2, y2) tuples.
    """
    if not page_ocr.words:
        return []

    # First detect columns
    columns = detect_columns(page_ocr.words, page_ocr.width, page_ocr.height)

    closing_punct = set('.,;:!?)]\'"')
    opening_punct = set('(["\'')
    result = []

    # Process each column separately
    for column_words in columns:
        # Sort by y center, then x within column
        words = sorted(column_words, key=lambda w: ((w.y1 + w.y2) / 2, w.x1))

        # Group into lines within this column
        lines_raw = []
        current_line = []
        y_threshold = page_ocr.height * 0.012

        for word in words:
            y_center = (word.y1 + word.y2) / 2
            if not current_line:
                current_line.append(word)
            else:
                curr_y = (current_line[0].y1 + current_line[0].y2) / 2
                if abs(y_center - curr_y) < y_threshold:
                    current_line.append(word)
                else:
                    current_line.sort(key=lambda w: w.x1)
                    lines_raw.append(current_line)
                    current_line = [word]

        if current_line:
            current_line.sort(key=lambda w: w.x1)
            lines_raw.append(current_line)

        # Process lines with punctuation fixes
        for line_words in lines_raw:
            text_parts = []
            for word in line_words:
                t = word.text
                if text_parts and all(c in closing_punct or c == '-' for c in t):
                    text_parts[-1] += t
                elif text_parts and all(c in opening_punct for c in text_parts[-1]):
                    text_parts[-1] += t
                elif text_parts and text_parts[-1].endswith('-'):
                    text_parts[-1] += t
                else:
                    text_parts.append(t)

            line_text = ' '.join(text_parts)
            x1 = min(w.x1 for w in line_words)
            y1 = min(w.y1 for w in line_words)
            x2 = max(w.x2 for w in line_words)
            y2 = max(w.y2 for w in line_words)
            result.append((line_text, x1, y1, x2, y2))

    return result


def compress_to_jbig2(tiff_files: List[Path], work_dir: Path) -> Tuple[List[Path], Optional[Path]]:
    """Compress TIFFs to JBIG2."""
    original_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        rel_tiffs = [f.name for f in tiff_files]
        subprocess.run(["jbig2", "-s", "-p", "-b", "output"] + rel_tiffs,
                      capture_output=True, check=True)
        jbig2_files = sorted(work_dir.glob("output.*"))
        sym_file = work_dir / "output.sym"
        data_files = [f for f in jbig2_files if f.suffix != '.sym']
        return data_files, sym_file if sym_file.exists() else None
    finally:
        os.chdir(original_dir)


def create_searchable_pdf(
    jbig2_files: List[Path],
    sym_file: Optional[Path],
    ocr_data: List[PageOCR],
    output_path: Path
):
    """Create PDF with JBIG2 images and positioned text layer."""
    pdf = pikepdf.Pdf.new()

    # Read global symbols
    globals_stream = None
    if sym_file and sym_file.exists():
        with open(sym_file, "rb") as f:
            globals_stream = pikepdf.Stream(pdf, f.read())

    # Courier font
    font_dict = pikepdf.Dictionary({
        "/Type": pikepdf.Name.Font,
        "/Subtype": pikepdf.Name.Type1,
        "/BaseFont": pikepdf.Name.Courier,
    })

    for jbig2_file, page_ocr in zip(jbig2_files, ocr_data):
        with open(jbig2_file, "rb") as f:
            jbig2_data = f.read()

        # Create JBIG2 image
        image_stream = pikepdf.Stream(pdf, jbig2_data)
        image_stream["/Type"] = pikepdf.Name.XObject
        image_stream["/Subtype"] = pikepdf.Name.Image
        image_stream["/Width"] = page_ocr.width
        image_stream["/Height"] = page_ocr.height
        image_stream["/ColorSpace"] = pikepdf.Name.DeviceGray
        image_stream["/BitsPerComponent"] = 1
        image_stream["/Filter"] = pikepdf.Name.JBIG2Decode

        if globals_stream:
            image_stream["/DecodeParms"] = pikepdf.Dictionary({"/JBIG2Globals": globals_stream})

        # Build content stream
        content_parts = [f"q {PAGE_WIDTH} 0 0 {PAGE_HEIGHT} 0 0 cm /Im0 Do Q"]

        # Add text layer - place each word individually for accurate selection
        if page_ocr.words:
            scale_x = PAGE_WIDTH / page_ocr.width
            scale_y = PAGE_HEIGHT / page_ocr.height

            content_parts.append("BT")
            content_parts.append("3 Tr")  # Invisible

            for word in page_ocr.words:
                text = word.text
                if not text:
                    continue

                pdf_x = word.x1 * scale_x
                pdf_y = PAGE_HEIGHT - (word.y2 * scale_y)  # Flip Y, use bottom of box
                word_width = (word.x2 - word.x1) * scale_x
                word_height = (word.y2 - word.y1) * scale_y

                font_size = max(1, min(word_height * 0.85, 72))
                natural_width = len(text) * font_size * 0.6
                h_scale = (word_width / natural_width * 100) if natural_width > 0 else 100

                escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

                content_parts.append(f"/F1 {font_size:.2f} Tf")
                content_parts.append(f"{h_scale:.1f} Tz")
                content_parts.append(f"1 0 0 1 {pdf_x:.2f} {pdf_y:.2f} Tm")
                content_parts.append(f"({escaped}) Tj")

            content_parts.append("ET")

        # Add page
        pdf.add_blank_page(page_size=(PAGE_WIDTH, PAGE_HEIGHT))
        page = pdf.pages[-1]
        page.Contents = pikepdf.Stream(pdf, "\n".join(content_parts).encode())
        page.Resources = pikepdf.Dictionary({
            "/XObject": pikepdf.Dictionary({"/Im0": image_stream}),
            "/Font": pikepdf.Dictionary({"/F1": font_dict}),
        })

    pdf.save(output_path, linearize=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)


def process_single_file(
    input_path: Path,
    output_dir: Path,
    docai_client: documentai.DocumentProcessorServiceClient,
    processor_name: str,
    openai_client: Optional[OpenAI] = None
) -> Tuple[bool, Path, str]:
    """Process a single PDF or TIFF through the hybrid pipeline."""
    input_path = Path(input_path)
    output_path = output_dir / f"{input_path.stem}_out.pdf"

    use_llm = openai_client is not None
    total_steps = 6 if use_llm else 5

    print(f"Processing: {input_path.name}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            # Step 1: Pre-process to 1-bit TIFFs
            print(f"  [1/{total_steps}] Converting to 1-bit B&W TIFFs...")
            tiff_files = preprocess_input(input_path, work_dir)
            print(f"        {len(tiff_files)} page(s)")

            # Step 2: Run tesseract hOCR (for pixel-accurate positions)
            print(f"  [2/{total_steps}] Running tesseract for positions...")
            tess_data = run_tesseract_hocr(tiff_files, work_dir)
            tess_words = sum(len(p.words) for p in tess_data)
            print(f"        {tess_words} positions found")

            # Step 3: Run Document AI OCR (for high-quality text)
            print(f"  [3/{total_steps}] Running Document AI for text...")
            docai_data = run_document_ai_ocr(tiff_files, docai_client, processor_name)
            docai_words = sum(len(p.words) for p in docai_data)
            print(f"        {docai_words} words recognized")

            # Step 4: Merge - tesseract positions + Document AI text
            print(f"  [4/{total_steps}] Merging OCR data...")
            merged_data = merge_ocr_data(tess_data, docai_data)

            # Step 5 (optional): LLM post-processing
            if use_llm:
                print(f"  [5/{total_steps}] LLM error correction (GPT-4o-mini)...")
                merged_data = postprocess_ocr_with_llm(merged_data, openai_client)
                print(f"        {sum(len(p.words) for p in merged_data)} words corrected")

            # Step 5/6: JBIG2 compression
            step_num = 6 if use_llm else 5
            print(f"  [{step_num}/{total_steps}] Creating JBIG2 PDF...")
            jbig2_files, sym_file = compress_to_jbig2(tiff_files, work_dir)
            create_searchable_pdf(jbig2_files, sym_file, merged_data, output_path)

            in_kb = input_path.stat().st_size / 1024
            out_kb = output_path.stat().st_size / 1024
            pct = (1 - out_kb / in_kb) * 100
            print(f"        {in_kb:.1f} KB -> {out_kb:.1f} KB ({pct:.1f}% reduction)")

            return True, output_path, ""

    except Exception as e:
        import traceback
        return False, output_path, f"{e}\n{traceback.format_exc()}"


def process_batch(input_paths, output_dir, docai_client, processor_name, openai_client=None, batch_size=4):
    """Process PDFs/TIFFs in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(process_single_file, f, output_dir, docai_client, processor_name, openai_client): f
            for f in input_paths
        }
        for future in as_completed(futures):
            input_file = futures[future]
            try:
                results.append((input_file, *future.result()))
            except Exception as e:
                results.append((input_file, False, Path(), str(e)))
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Document AI OCR + hOCR alignment + JBIG2 compression (PDF and TIFF support)'
    )
    parser.add_argument('files', nargs='*', help='PDF or TIFF files to process')
    parser.add_argument('--processor-id', required=True, help='Document AI processor ID')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Parallel processes')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files')
    parser.add_argument('--llm-postprocess', action='store_true',
                        help='Use GPT-4o-mini to fix OCR errors (requires OPENAI_API_KEY)')
    args = parser.parse_args()

    # Supported extensions
    supported_ext = ('.pdf', '.tif', '.tiff')

    # Find input files
    if args.files:
        input_files = [Path(f) for f in args.files
                       if f.lower().endswith(supported_ext) and not f.endswith('_out.pdf')]
    else:
        input_files = sorted([f for f in Path('.').iterdir()
                              if f.suffix.lower() in supported_ext and not f.name.endswith('_out.pdf')])

    if args.limit:
        input_files = input_files[:args.limit]

    if not input_files:
        print("No PDF or TIFF files found")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Document AI
    print(f"Initializing Document AI...")
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor_name = client.processor_path(PROJECT_ID, LOCATION, args.processor_id)

    # Initialize OpenAI client if LLM post-processing enabled
    openai_client = None
    if args.llm_postprocess:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: --llm-postprocess requires OPENAI_API_KEY environment variable")
            return 1
        openai_client = OpenAI(api_key=api_key)
        print(f"LLM post-processing: enabled (GPT-4o-mini)")

    pdf_count = sum(1 for f in input_files if f.suffix.lower() == '.pdf')
    tiff_count = len(input_files) - pdf_count

    print(f"\n{'='*60}")
    print(f"Processing {len(input_files)} file(s) ({pdf_count} PDF, {tiff_count} TIFF)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    results = process_batch(input_files, args.output_dir, client, processor_name, openai_client, args.batch_size)

    # Summary
    success = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(success)} succeeded, {len(failed)} failed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed files:")
        for pdf, _, _, err in failed:
            print(f"  {pdf.name}: {err[:100]}")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
