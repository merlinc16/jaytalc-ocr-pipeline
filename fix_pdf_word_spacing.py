#!/Users/mc/jaytalc/venv/bin/python3
"""
Fix PDF word spacing for Preview.app copy-paste.

Problem: Our OCR pipeline places each word as a separate invisible text (Tj)
command positioned over the scanned image. When words are positioned close
together, Preview.app merges them during copy-paste (e.g., "publicmenace",
"highlycame"). Other extractors like pdftotext handle this correctly.

Fix: Append a space character to each word's text in the Tj command, e.g.,
  (hello) Tj  ->  (hello ) Tj

Since the text layer uses invisible rendering (3 Tr), this doesn't affect
the visual appearance but forces text extractors to include word separators.

Usage:
  # Fix a single PDF (overwrites in place):
  python fix_pdf_word_spacing.py input.pdf

  # Fix a single PDF to a new file:
  python fix_pdf_word_spacing.py input.pdf -o output.pdf

  # Fix all PDFs in a directory tree:
  python fix_pdf_word_spacing.py /path/to/pdfs/ --recursive

  # Dry run (count fixes without modifying):
  python fix_pdf_word_spacing.py /path/to/pdfs/ --recursive --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import pikepdf


def fix_word_spacing(pdf_path, output_path=None, dry_run=False):
    """Fix word spacing in a single PDF.

    Appends a space to every word's Tj command in the content stream.

    Args:
        pdf_path: Path to input PDF.
        output_path: Path to save fixed PDF. If None, overwrites in place.
        dry_run: If True, count fixes without saving.

    Returns:
        Number of words modified.
    """
    overwrite = output_path is None or str(output_path) == str(pdf_path)
    pdf = pikepdf.open(str(pdf_path), allow_overwriting_input=overwrite)

    total_modified = 0

    for page in pdf.pages:
        try:
            content = page.Contents.read_bytes().decode('latin-1')
        except Exception:
            continue

        lines = content.split('\n')
        new_lines = list(lines)
        modified = 0

        for i, line in enumerate(lines):
            line_s = line.strip()
            m = re.match(r'\((.+)\) Tj$', line_s)
            if m:
                word_text = m.group(1)
                new_tj = f'({word_text} ) Tj'
                new_lines[i] = new_lines[i].replace(line_s, new_tj)
                modified += 1

        if modified > 0 and not dry_run:
            new_content = '\n'.join(new_lines).encode('latin-1')
            page.Contents = pikepdf.Stream(pdf, new_content)

        total_modified += modified

    if not dry_run:
        save_path = str(output_path) if output_path else str(pdf_path)
        pdf.save(save_path)

    pdf.close()
    return total_modified


def process_directory(dir_path, recursive=False, dry_run=False):
    """Fix word spacing in all PDFs in a directory."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(Path(dir_path).glob(pattern))

    if not pdf_files:
        print(f"No PDFs found in {dir_path}")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Processing {len(pdf_files)} PDFs...")

    total_files = 0
    total_words = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files):
        try:
            words = fix_word_spacing(pdf_path, dry_run=dry_run)
            total_words += words
            total_files += 1
            print(f"  {i+1}/{len(pdf_files)}  {words:5d} words  {pdf_path.name}")
        except Exception as e:
            failed += 1
            print(f"  {i+1}/{len(pdf_files)}  FAILED  {pdf_path.name}: {e}")

    action = "Would fix" if dry_run else "Fixed"
    print(f"\n{action} {total_words} words across {total_files} files ({failed} failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Fix PDF word spacing for Preview.app copy-paste"
    )
    parser.add_argument("input", help="PDF file or directory to fix")
    parser.add_argument("-o", "--output", help="Output file (single file mode only)")
    parser.add_argument(
        "--recursive", action="store_true",
        help="Process all PDFs in subdirectories"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count fixes without modifying files"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output = Path(args.output) if args.output else None
        words = fix_word_spacing(input_path, output, dry_run=args.dry_run)
        dest = args.output or str(input_path)
        action = "Would fix" if args.dry_run else "Fixed"
        print(f"{action} {words} words in {dest}")
    elif input_path.is_dir():
        if args.output:
            print("Error: --output not supported for directory mode", file=sys.stderr)
            sys.exit(1)
        process_directory(input_path, args.recursive, args.dry_run)
    else:
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
