"""
Document Chunker
================
Splits all documents in data/raw/ into smaller, retrievable chunks
and saves them as a single JSONL file in data/processed/.

Each chunk records its source, original document ID, position,
chunking strategy, and the chunk text itself.

Run from the project root:
    python -m src.processing.chunker
"""

import json
from pathlib import Path
from typing import Iterator

import tiktoken

# --- Configuration ---

# Source folders (each contains JSON records)
SOURCE_DIRS = {
    "WHO":    Path("data/raw/who"),
    "PubMed": Path("data/raw/pubmed"),
    "ICMR":   Path("data/raw/icmr_parsed"),
}

# Where the chunks go
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

# Chunking parameters (in tokens, not characters)
CHUNK_SIZE = 500       # ~375 words; one dense medical paragraph
CHUNK_OVERLAP = 75     # ~15% overlap; preserves context across boundaries

# Which chunking strategy to use for v1
STRATEGY = "fixed_size"  # later: "recursive", "section_aware"

# Tokenizer - cl100k_base is what GPT-4 / most modern LLMs use
ENCODER = tiktoken.get_encoding("cl100k_base")


# --- Helper functions ---

def count_tokens(text: str) -> int:
    """Return the number of tokens in a piece of text."""
    return len(ENCODER.encode(text))


def chunk_fixed_size(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Iterator[str]:
    """
    Split text into chunks of `chunk_size` tokens with `overlap`
    tokens of overlap between consecutive chunks.

    Yields chunk strings one at a time (memory-efficient for large docs).
    """
    tokens = ENCODER.encode(text)
    total = len(tokens)

    if total == 0:
        return

    start = 0
    step = chunk_size - overlap  # how far to move forward each iteration

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODER.decode(chunk_tokens)
        yield chunk_text

        # If we've reached the end, stop
        if end == total:
            break

        start += step


def load_documents(source_dirs: dict) -> Iterator[dict]:
    """
    Iterate over all JSON document files across all source folders.
    Yields one document dict at a time.
    """
    for source_name, dir_path in source_dirs.items():
        if not dir_path.exists():
            print(f"  [!] Skipping missing folder: {dir_path}")
            continue

        json_files = list(dir_path.glob("*.json"))
        print(f"  -> {source_name}: {len(json_files)} documents")

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    # Tag with the file's actual location for traceability
                    doc["_source_file"] = str(json_path)
                    yield doc
            except Exception as e:
                print(f"  [X] Error loading {json_path.name}: {e}")


def make_chunk_record(
    chunk_text: str,
    chunk_index: int,
    parent_doc: dict,
    strategy: str,
) -> dict:
    """Build a JSON record for one chunk."""
    # Build a unique chunk ID combining source + doc identifier + index
    source = parent_doc.get("source", "unknown")
    doc_identifier = (
        parent_doc.get("pmid")        # PubMed
        or parent_doc.get("doc_id")   # ICMR
        or parent_doc.get("url", "").rstrip("/").split("/")[-1]  # WHO
        or "unknown"
    )
    chunk_id = f"{source.lower()}_{doc_identifier}_{chunk_index:04d}"

    return {
        "chunk_id": chunk_id,
        "source": source,
        "parent_title": parent_doc.get("title", ""),
        "parent_url": parent_doc.get("url", ""),
        "parent_doc_id": doc_identifier,
        "chunk_index": chunk_index,
        "strategy": strategy,
        "token_count": count_tokens(chunk_text),
        "text": chunk_text,
    }


# --- Main routine ---

def main():
    print("=" * 60)
    print("Document Chunker")
    print("=" * 60)
    print(f"Strategy: {STRATEGY}")
    print(f"Chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP} tokens")

    # Make sure output folder exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] Scanning source folders...")
    documents = list(load_documents(SOURCE_DIRS))
    print(f"\n  Total documents loaded: {len(documents)}")

    if not documents:
        print("\n[!] No documents found. Run the scrapers first.")
        return

    print(f"\n[2/2] Chunking documents -> {OUTPUT_FILE}")
    total_chunks = 0
    docs_processed = 0
    docs_skipped = 0
    chunks_per_source = {"WHO": 0, "PubMed": 0, "ICMR": 0}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")

            if not content or count_tokens(content) < 50:
                docs_skipped += 1
                continue

            chunk_count_for_doc = 0
            for chunk_index, chunk_text in enumerate(chunk_fixed_size(content)):
                # Skip very small final chunks (< 100 tokens)
                if count_tokens(chunk_text) < 100:
                    continue

                record = make_chunk_record(chunk_text, chunk_index, doc, STRATEGY)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
                chunk_count_for_doc += 1

            if chunk_count_for_doc > 0:
                docs_processed += 1
                if source in chunks_per_source:
                    chunks_per_source[source] += chunk_count_for_doc

            # Progress indicator every 50 documents
            if docs_processed % 50 == 0 and docs_processed > 0:
                print(f"  ...processed {docs_processed} documents, {total_chunks} chunks so far")

    print(f"\n[Done]")
    print(f"  Documents processed: {docs_processed}")
    print(f"  Documents skipped (too short): {docs_skipped}")
    print(f"  Total chunks: {total_chunks}")
    print(f"\n  Chunks per source:")
    for src, count in chunks_per_source.items():
        print(f"    {src}: {count}")
    print(f"\n  Output: {OUTPUT_FILE.resolve()}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()