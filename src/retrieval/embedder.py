"""
Chunk Embedder
==============
Loads the chunks produced by the chunker and embeds each one into
a 384-dimensional vector using sentence-transformers/all-MiniLM-L6-v2.

Saves embeddings + metadata to data/processed/embeddings.npz so
we can load them quickly in the retrieval step.

Run from the project root:
    python -m src.retrieval.embedder
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---

INPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("data/processed/embeddings.npz")

# The embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size for encoding (how many chunks to embed at once)
# Higher = faster but uses more RAM. 32 is safe for any laptop.
BATCH_SIZE = 32


# --- Helper functions ---

def load_chunks(input_file: Path) -> list[dict]:
    """Load all chunks from the JSONL file into memory."""
    print(f"  -> Loading chunks from {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {input_file}\n"
            f"   Run the chunker first: python -m src.processing.chunker"
        )

    chunks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"  -> Loaded {len(chunks)} chunks")
    return chunks


def load_model(model_name: str) -> SentenceTransformer:
    """
    Load the embedding model. First run downloads ~90MB to a local cache.
    Subsequent runs load from cache (fast).
    """
    print(f"\n  -> Loading model: {model_name}")
    print(f"     (First run downloads ~90MB; subsequent runs are instant)")
    model = SentenceTransformer(model_name)
    print(f"     Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(model: SentenceTransformer, chunks: list[dict]) -> np.ndarray:
    """
    Embed all chunks in batches. Returns an (N, 384) numpy array
    where N = number of chunks and 384 = embedding dimension.
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"\n  -> Embedding {len(texts)} chunks (batch size {BATCH_SIZE})...")

    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine similarity = dot product (faster later)
    )
    elapsed = time.time() - start_time

    print(f"\n  -> Done in {elapsed:.1f}s ({len(texts) / elapsed:.0f} chunks/sec)")
    print(f"     Output shape: {embeddings.shape}")
    return embeddings


def save_embeddings(
    output_file: Path,
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:
    """
    Save embeddings + chunk metadata to a single .npz file.
    .npz is numpy's compressed multi-array format — efficient and fast to load.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract metadata into parallel arrays (one entry per chunk)
    chunk_ids = np.array([c["chunk_id"] for c in chunks])
    sources = np.array([c["source"] for c in chunks])
    parent_titles = np.array([c["parent_title"] for c in chunks])
    parent_urls = np.array([c["parent_url"] for c in chunks])
    parent_doc_ids = np.array([c["parent_doc_id"] for c in chunks])
    texts = np.array([c["text"] for c in chunks])

    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        sources=sources,
        parent_titles=parent_titles,
        parent_urls=parent_urls,
        parent_doc_ids=parent_doc_ids,
        texts=texts,
    )

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\n  -> Saved to {output_file}")
    print(f"     File size: {file_size_mb:.2f} MB")


# --- Main routine ---

def main():
    print("=" * 60)
    print("Chunk Embedder")
    print("=" * 60)

    # Step 1: load chunks
    print(f"\n[1/3] Loading chunks...")
    chunks = load_chunks(INPUT_FILE)

    # Step 2: load model + embed
    print(f"\n[2/3] Loading embedding model and encoding chunks...")
    model = load_model(MODEL_NAME)
    embeddings = embed_chunks(model, chunks)

    # Sanity check: every chunk got embedded
    assert len(embeddings) == len(chunks), \
        f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"

    # Step 3: save
    print(f"\n[3/3] Saving embeddings...")
    save_embeddings(OUTPUT_FILE, embeddings, chunks)

    print(f"\n[Done] Embedding complete.")


if __name__ == "__main__":
    main()