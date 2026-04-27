"""
FAISS Indexer
=============
Loads pre-computed embeddings from embeddings.npz and builds a
FAISS index on disk for fast similarity search.

FAISS stores the index as a binary file; chunk metadata is stored
separately in a small JSON sidecar file.

Run from the project root:
    python -m src.retrieval.indexer
"""

import json
import time
from pathlib import Path

import numpy as np
import faiss

# --- Configuration ---

EMBEDDINGS_FILE = Path("data/processed/embeddings.npz")
INDEX_DIR = Path("data/processed/faiss")
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"


# --- Helper functions ---

def load_embeddings(filepath: Path):
    """Load all chunks + embeddings from the .npz file."""
    print(f"  -> Loading {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {filepath}\n"
            f"   Run the embedder first: python -m src.retrieval.embedder"
        )
    data = np.load(filepath, allow_pickle=True)
    print(f"  -> Loaded {len(data['chunk_ids'])} chunks")
    return data


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index optimized for cosine similarity.
    Because our embeddings are L2-normalized, inner product
    is mathematically equivalent to cosine similarity.
    """
    n, dim = embeddings.shape
    print(f"\n  -> Building FAISS index (dim={dim}, vectors={n})")

    index = faiss.IndexFlatIP(dim)
    vectors = np.ascontiguousarray(embeddings.astype(np.float32))
    index.add(vectors)

    print(f"  -> Index contains {index.ntotal} vectors")
    return index


def save_metadata(data, filepath: Path) -> None:
    """Save chunk metadata as a parallel-array JSON file."""
    metadata = {
        "chunk_ids":      [str(x) for x in data["chunk_ids"]],
        "sources":        [str(x) for x in data["sources"]],
        "parent_titles":  [str(x) for x in data["parent_titles"]],
        "parent_urls":    [str(x) for x in data["parent_urls"]],
        "parent_doc_ids": [str(x) for x in data["parent_doc_ids"]],
        "texts":          [str(x) for x in data["texts"]],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"  -> Saved metadata to {filepath}")


# --- Main routine ---

def main():
    print("=" * 60)
    print("FAISS Indexer")
    print("=" * 60)

    print(f"\n[1/3] Loading pre-computed embeddings...")
    data = load_embeddings(EMBEDDINGS_FILE)

    print(f"\n[2/3] Building FAISS index...")
    start = time.time()
    index = build_faiss_index(data["embeddings"])
    elapsed = time.time() - start
    print(f"  -> Built in {elapsed:.2f}s")

    print(f"\n[3/3] Saving to {INDEX_DIR}")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    save_metadata(data, METADATA_FILE)

    index_size_mb = INDEX_FILE.stat().st_size / 1024 / 1024
    metadata_size_mb = METADATA_FILE.stat().st_size / 1024 / 1024

    print(f"\n[Done]")
    print(f"  Index file: {INDEX_FILE} ({index_size_mb:.2f} MB)")
    print(f"  Metadata file: {METADATA_FILE} ({metadata_size_mb:.2f} MB)")
    print(f"  Total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()