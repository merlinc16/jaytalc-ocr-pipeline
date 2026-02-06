#!/Users/mc/jaytalc/venv/bin/python3
"""
Complete OCR Pipeline: Document AI + hOCR alignment + JBIG2 compression

Produces ABBYY-compatible PDFs with:
- Document AI OCR (high accuracy text recognition)
- hOCR-based text layer (perfect text selection alignment)
- JBIG2 compression (ABBYY-identical structure)
- Clean text extraction via pdftotext/Tika
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.sax.saxutils import escape

import pikepdf
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai

# Configuration
PROJECT_ID = "toxicdocs"
LOCATION = "us"
DPI = 300
PAGE_WIDTH_PX = 2550  # 8.5" * 300 DPI
PAGE_HEIGHT_PX = 3300  # 11" * 300 DPI


@dataclass
class Word:
    """Word with bounding box (pixel coordinates)."""
    text: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Line:
    """Line with words and bounding box."""
    text: str
    words: List[Word]
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class PageOCR:
    """OCR data for a page."""
    page_num: int
    lines: List[Line] = field(default_factory=list)
    width: int = PAGE_WIDTH_PX
    height: int = PAGE_HEIGHT_PX


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


def run_document_ai_ocr(
    tiff_files: List[Path],
    docai_client: documentai.DocumentProcessorServiceClient,
    processor_name: str
) -> List[PageOCR]:
    """Run Document AI OCR and return structured data with line grouping and punctuation fixes."""
    ocr_results = []

    closing_punct = set('.,;:!?)]\'"')
    opening_punct = set('(["\'')

    for i, tiff_file in enumerate(tiff_files):
        with open(tiff_file, "rb") as f:
            content = f.read()

        width, height = get_tiff_dimensions(tiff_file)

        raw_doc = documentai.RawDocument(content=content, mime_type="image/tiff")
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_doc)
        result = docai_client.process_document(request=request)
        document = result.document

        # Extract words with pixel coordinates
        words = []
        for page in document.pages:
            for token in page.tokens:
                # Get text
                text = ""
                if token.layout.text_anchor and token.layout.text_anchor.text_segments:
                    for seg in token.layout.text_anchor.text_segments:
                        start = int(seg.start_index) if seg.start_index else 0
                        end = int(seg.end_index)
                        text += document.text[start:end]
                text = text.strip()
                if not text:
                    continue

                # Get bounding box in pixels
                if token.layout.bounding_poly.normalized_vertices:
                    verts = token.layout.bounding_poly.normalized_vertices
                    x1 = int(min(v.x for v in verts) * width)
                    y1 = int(min(v.y for v in verts) * height)
                    x2 = int(max(v.x for v in verts) * width)
                    y2 = int(max(v.y for v in verts) * height)
                    words.append(Word(text=text, x1=x1, y1=y1, x2=x2, y2=y2))

        # Group words into lines (handle multi-column layouts)
        if not words:
            ocr_results.append(PageOCR(page_num=i+1, width=width, height=height))
            continue

        words.sort(key=lambda w: ((w.y1 + w.y2) / 2, w.x1))

        lines_raw = []
        current_line = []
        y_threshold = height * 0.012  # 1.2% of page height
        x_gap_threshold = width * 0.15  # 15% of page width = column gap

        for word in words:
            y_center = (word.y1 + word.y2) / 2
            if not current_line:
                current_line.append(word)
            else:
                curr_y = (current_line[0].y1 + current_line[0].y2) / 2
                # Check if same line AND not a column gap
                last_word = max(current_line, key=lambda w: w.x2)
                x_gap = word.x1 - last_word.x2

                if abs(y_center - curr_y) < y_threshold and x_gap < x_gap_threshold:
                    current_line.append(word)
                else:
                    current_line.sort(key=lambda w: w.x1)
                    lines_raw.append(current_line)
                    current_line = [word]

        if current_line:
            current_line.sort(key=lambda w: w.x1)
            lines_raw.append(current_line)

        # Process lines with punctuation fixes
        lines = []
        for line_words in lines_raw:
            # Build text with punctuation handling
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

            lines.append(Line(text=line_text, words=line_words, x1=x1, y1=y1, x2=x2, y2=y2))

        ocr_results.append(PageOCR(page_num=i+1, lines=lines, width=width, height=height))

    return ocr_results


def generate_hocr(ocr_data: List[PageOCR], work_dir: Path) -> List[Path]:
    """Generate hOCR XML files from OCR data."""
    hocr_files = []

    for page in ocr_data:
        hocr_path = work_dir / f"page_{page.page_num:04d}.hocr"

        hocr_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <title>OCR Output</title>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
</head>
<body>
  <div class="ocr_page" title="bbox 0 0 {page.width} {page.height}">
'''

        for j, line in enumerate(page.lines):
            line_bbox = f"bbox {line.x1} {line.y1} {line.x2} {line.y2}"
            escaped_text = escape(line.text)
            hocr_content += f'    <span class="ocr_line" title="{line_bbox}">\n'

            # Add individual words for better text selection
            for k, word in enumerate(line.words):
                word_bbox = f"bbox {word.x1} {word.y1} {word.x2} {word.y2}"
                escaped_word = escape(word.text)
                hocr_content += f'      <span class="ocrx_word" title="{word_bbox}">{escaped_word}</span>\n'

            hocr_content += f'    </span>\n'

        hocr_content += '''  </div>
</body>
</html>'''

        with open(hocr_path, 'w', encoding='utf-8') as f:
            f.write(hocr_content)

        hocr_files.append(hocr_path)

    return hocr_files


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


def create_final_pdf(jbig2_files: List[Path], sym_file: Optional[Path],
                     ocr_data: List[PageOCR], output_path: Path):
    """Create final PDF with JBIG2 images and Document AI text layer."""
    pdf = pikepdf.Pdf.new()

    # Read global symbols
    globals_stream = None
    globals_data = None
    if sym_file and sym_file.exists():
        with open(sym_file, "rb") as f:
            globals_data = f.read()
        globals_stream = pikepdf.Stream(pdf, globals_data)

    # Font for invisible text
    font = pikepdf.Dictionary({
        "/Type": pikepdf.Name.Font,
        "/Subtype": pikepdf.Name.Type1,
        "/BaseFont": pikepdf.Name.Helvetica,
    })

    PDF_W, PDF_H = 612, 792  # Letter size in points

    for jbig2_file, page_ocr in zip(jbig2_files, ocr_data):
        # Create JBIG2 image
        with open(jbig2_file, "rb") as f:
            jbig2_data = f.read()

        img = pikepdf.Stream(pdf, jbig2_data)
        img["/Type"] = pikepdf.Name.XObject
        img["/Subtype"] = pikepdf.Name.Image
        img["/Width"] = page_ocr.width
        img["/Height"] = page_ocr.height
        img["/ColorSpace"] = pikepdf.Name.DeviceGray
        img["/BitsPerComponent"] = 1
        img["/Filter"] = pikepdf.Name.JBIG2Decode

        if globals_stream:
            img["/DecodeParms"] = pikepdf.Dictionary({"/JBIG2Globals": globals_stream})

        # Build content stream
        content = [f"q {PDF_W} 0 0 {PDF_H} 0 0 cm /Im0 Do Q"]

        # Add text layer
        if page_ocr.lines:
            content.append("BT")
            content.append("3 Tr")  # Invisible

            scale_x = PDF_W / page_ocr.width
            scale_y = PDF_H / page_ocr.height

            for line in page_ocr.lines:
                # Position at line start, baseline at bottom of bbox
                pdf_x = line.x1 * scale_x
                pdf_y = PDF_H - (line.y2 * scale_y)

                # Font size from line height
                line_h = (line.y2 - line.y1) * scale_y
                font_size = max(4, min(line_h * 0.8, 72))

                # Escape text
                text = line.text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

                content.append(f"/F1 {font_size:.1f} Tf")
                content.append(f"1 0 0 1 {pdf_x:.2f} {pdf_y:.2f} Tm")
                content.append(f"({text}) Tj")

            content.append("ET")

        # Create page
        pdf.add_blank_page(page_size=(PDF_W, PDF_H))
        page = pdf.pages[-1]
        page.Contents = pikepdf.Stream(pdf, "\n".join(content).encode())
        page.Resources = pikepdf.Dictionary({
            "/XObject": pikepdf.Dictionary({"/Im0": img}),
            "/Font": pikepdf.Dictionary({"/F1": font}),
        })

    pdf.save(output_path, linearize=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    pdf.close()


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    docai_client: documentai.DocumentProcessorServiceClient,
    processor_name: str
) -> Tuple[bool, Path, str]:
    """Process a single PDF through the complete pipeline."""
    pdf_path = Path(pdf_path)
    output_path = output_dir / f"{pdf_path.stem}_out.pdf"

    print(f"Processing: {pdf_path.name}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            # Step 1: Pre-process to 1-bit TIFFs
            print(f"  [1/5] Converting to 1-bit B&W TIFFs...")
            tiff_files = preprocess_pdf(pdf_path, work_dir)
            print(f"        {len(tiff_files)} page(s)")

            # Step 2: Document AI OCR
            print(f"  [2/5] Running Document AI OCR...")
            ocr_data = run_document_ai_ocr(tiff_files, docai_client, processor_name)
            total_words = sum(len(line.words) for page in ocr_data for line in page.lines)
            print(f"        {total_words} words extracted")

            # Step 3: JBIG2 compression
            print(f"  [3/4] Compressing to JBIG2...")
            jbig2_files, sym_file = compress_to_jbig2(tiff_files, work_dir)

            # Step 4: Create final PDF
            print(f"  [4/4] Creating searchable PDF...")
            create_final_pdf(jbig2_files, sym_file, ocr_data, output_path)

            in_kb = pdf_path.stat().st_size / 1024
            out_kb = output_path.stat().st_size / 1024
            pct = (1 - out_kb / in_kb) * 100
            print(f"        {in_kb:.1f} KB -> {out_kb:.1f} KB ({pct:.1f}% reduction)")

            return True, output_path, ""

    except Exception as e:
        import traceback
        return False, output_path, f"{e}\n{traceback.format_exc()}"


def process_batch(pdf_paths, output_dir, docai_client, processor_name, batch_size=4):
    """Process PDFs in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf, output_dir, docai_client, processor_name): pdf
            for pdf in pdf_paths
        }
        for future in as_completed(futures):
            pdf = futures[future]
            try:
                results.append((pdf, *future.result()))
            except Exception as e:
                results.append((pdf, False, Path(), str(e)))
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Document AI OCR + hOCR alignment + JBIG2 compression'
    )
    parser.add_argument('files', nargs='*', help='PDF files to process')
    parser.add_argument('--processor-id', required=True, help='Document AI processor ID')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Parallel processes')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files')
    args = parser.parse_args()

    # Find PDFs
    if args.files:
        pdfs = [Path(f) for f in args.files if f.endswith('.pdf') and not f.endswith('_out.pdf')]
    else:
        pdfs = sorted([f for f in Path('.').glob('*.pdf') if not f.name.endswith('_out.pdf')])

    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print("No PDFs found")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Document AI
    print(f"Initializing Document AI...")
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor_name = client.processor_path(PROJECT_ID, LOCATION, args.processor_id)

    print(f"\n{'='*60}")
    print(f"Processing {len(pdfs)} PDF(s)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    results = process_batch(pdfs, args.output_dir, client, processor_name, args.batch_size)

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
