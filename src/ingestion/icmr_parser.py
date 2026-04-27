"""
ICMR Guidelines PDF Parser
==========================
Reads PDF guideline documents from data/raw/icmr/ and converts
them to clean structured JSON for use in the RAG corpus.

PDFs must be manually downloaded into data/raw/icmr/ first.

Run from the project root:
    python -m src.ingestion.icmr_parser
"""

import json
import re
from pathlib import Path

import pdfplumber

# --- Configuration ---

INPUT_DIR = Path("data/raw/icmr")
OUTPUT_DIR = Path("data/raw/icmr_parsed")

# Minimum content length to keep a parsed document
MIN_CONTENT_LENGTH = 500


# --- Helper functions ---

def clean_extracted_text(text):
    """Clean up common PDF extraction noise."""
    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Normalize bullets
    text = re.sub(r"[\u2022\u00b7\u25cf\u25cb\u25e6\u25aa\u25ab]", "-", text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Strip whitespace from each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove resulting empty triplet lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_text(pdf_path):
    """Extract all text from a PDF, page by page, then clean it."""
    print(f"  -> Parsing: {pdf_path.name}")
    pages_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"     ({num_pages} pages)")

            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)

                if page_num % 25 == 0:
                    print(f"     ...processed {page_num}/{num_pages} pages")

    except Exception as e:
        print(f"     [X] Failed to parse: {e}")
        return None

    full_text = "\n\n".join(pages_text)
    cleaned = clean_extracted_text(full_text)
    return cleaned


def make_record(pdf_path, content):
    """Build a structured JSON record for one parsed PDF."""
    doc_id = pdf_path.stem
    title = doc_id.replace("_", " ").replace("-", " ").strip()

    return {
        "source": "ICMR",
        "doc_id": doc_id,
        "title": title,
        "filename": pdf_path.name,
        "content": content,
        "content_length": len(content),
    }


def save_record(record, output_dir):
    """Save a parsed PDF as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{record['doc_id']}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"     [OK] Saved: {filepath.name} ({record['content_length']:,} chars)")


# --- Main routine ---

def main():
    print("=" * 60)
    print("ICMR Guidelines PDF Parser")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"\n[!] Input folder doesn't exist: {INPUT_DIR}")
        print("    Create the folder and add PDF files first.")
        return

    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"\n[!] No PDF files found in {INPUT_DIR}")
        print("    Download ICMR guideline PDFs into that folder first.")
        return

    print(f"\n[1/2] Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"    - {pdf.name} ({size_mb:.2f} MB)")

    print(f"\n[2/2] Extracting text from PDFs...")
    saved_count = 0
    for pdf_path in pdf_files:
        content = extract_pdf_text(pdf_path)

        if content is None:
            continue

        if len(content) < MIN_CONTENT_LENGTH:
            print(f"     [!] Skipping (too short, possibly scanned): {pdf_path.name}")
            continue

        record = make_record(pdf_path, content)
        save_record(record, OUTPUT_DIR)
        saved_count += 1

    print(f"\n[Done] Total saved: {saved_count} documents")
    print(f"  Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()