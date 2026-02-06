#!/Users/mc/jaytalc/venv/bin/python3
"""
OCR Pipeline using ocrmypdf for proper text alignment.

This script processes PDF files using ocrmypdf (tesseract backend) to create
searchable PDFs with:
- Correct text layer alignment for proper text selection
- JBIG2 compression for small file sizes
- High optimization level
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
) -> Tuple[bool, Path, str]:
    """
    Process a single PDF through ocrmypdf.

    Steps:
    1. Convert PDF to images (handles problematic PDFs)
    2. Run ocrmypdf with JBIG2 optimization

    Returns: (success, output_path, error_message)
    """
    pdf_path = Path(pdf_path)
    output_path = output_dir / f"{pdf_path.stem}_out.pdf"

    print(f"Processing: {pdf_path.name}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Step 1: Convert PDF to PNG images (handles problematic PDFs)
            print(f"  Converting to images...")
            png_prefix = temp_dir / "page"
            result = subprocess.run(
                ["pdftoppm", "-png", "-r", "300", str(pdf_path), str(png_prefix)],
                capture_output=True,
                text=True,
                check=True
            )

            # Find generated PNG files
            png_files = sorted(temp_dir.glob("page-*.png"))
            if not png_files:
                png_files = sorted(temp_dir.glob("page*.png"))

            if not png_files:
                return False, output_path, "No pages generated from PDF"

            print(f"    Generated {len(png_files)} page(s)")

            # Step 2: Convert PNGs to a single PDF
            print(f"  Creating intermediate PDF...")
            intermediate_pdf = temp_dir / "intermediate.pdf"

            # Use ImageMagick to combine PNGs into PDF
            magick_cmd = ["magick"] + [str(f) for f in png_files] + [str(intermediate_pdf)]
            subprocess.run(magick_cmd, capture_output=True, check=True)

            # Step 3: Run ocrmypdf with JBIG2 optimization
            print(f"  Running OCR and optimization...")
            ocr_result = subprocess.run(
                [
                    "ocrmypdf",
                    "--output-type", "pdf",
                    "--optimize", "3",
                    "--jbig2-lossy",
                    "--skip-text",  # Don't re-OCR if text exists
                    "--force-ocr",  # Force OCR even on image PDFs
                    str(intermediate_pdf),
                    str(output_path)
                ],
                capture_output=True,
                text=True
            )

            if ocr_result.returncode != 0:
                # Try without --jbig2-lossy if it fails
                ocr_result = subprocess.run(
                    [
                        "ocrmypdf",
                        "--output-type", "pdf",
                        "--optimize", "2",
                        "--force-ocr",
                        str(intermediate_pdf),
                        str(output_path)
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )

            if output_path.exists():
                input_size = pdf_path.stat().st_size / 1024
                output_size = output_path.stat().st_size / 1024
                reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0
                print(f"    Done: {input_size:.1f} KB -> {output_size:.1f} KB ({reduction:.1f}% {'reduction' if reduction > 0 else 'increase'})")
                return True, output_path, ""
            else:
                return False, output_path, "Output file not created"

    except subprocess.CalledProcessError as e:
        return False, output_path, f"Command failed: {e.stderr}"
    except Exception as e:
        return False, output_path, str(e)


def process_batch(
    pdf_paths: List[Path],
    output_dir: Path,
    batch_size: int = 4
) -> List[Tuple[Path, bool, Path, str]]:
    """
    Process documents in parallel batches.

    Returns list of (input_path, success, output_path, error_message)
    """
    results = []

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf, output_dir): pdf
            for pdf in pdf_paths
        }

        for future in as_completed(futures):
            pdf = futures[future]
            try:
                success, output_path, error = future.result()
                results.append((pdf, success, output_path, error))
            except Exception as e:
                results.append((pdf, False, Path(), str(e)))

    return results


def verify_dependencies():
    """Check that all required external tools are available."""
    dependencies = [
        ("pdftoppm", "poppler"),
        ("magick", "imagemagick"),
        ("ocrmypdf", "ocrmypdf"),
    ]

    missing = []
    for cmd, package in dependencies:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            missing.append(f"{cmd} (install with: brew install {package})")

    if missing:
        print("Missing required dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='OCR PDF files using ocrmypdf with proper text alignment'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='PDF files to process (if none specified, processes all PDFs in current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for processed PDFs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Number of documents to process concurrently (default: 4)'
    )
    args = parser.parse_args()

    # Verify dependencies
    verify_dependencies()

    # Find PDF files to process
    if args.files:
        pdf_files = [Path(f) for f in args.files if f.lower().endswith('.pdf')]
        # Exclude already-processed files
        pdf_files = [f for f in pdf_files if not f.name.endswith('_out.pdf')]
    else:
        pdf_files = list(Path('.').glob('*.pdf'))
        pdf_files = [f for f in pdf_files if not f.name.endswith('_out.pdf')]

    if not pdf_files:
        print("No PDF files found to process.")
        sys.exit(1)

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdf_files)} PDF file(s) to process")
    print(f"Output directory: {output_dir}")

    # Process files
    print(f"\nProcessing with batch size: {args.batch_size}")
    results = process_batch(pdf_files, output_dir, args.batch_size)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [(p, o) for p, s, o, e in results if s]
    failed = [(p, e) for p, s, o, e in results if not s]

    print(f"\nSuccessfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed files:")
        for pdf_file, error in failed:
            print(f"  {pdf_file.name}: {error}")

    print("\nDone!")
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
