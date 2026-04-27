"""
Semantic Retriever (FAISS)
==========================
Provides a clean interface to query the FAISS index and return
the top-k most semantically similar chunks.

Usage as CLI:
    python -m src.retrieval.retriever "your question here"
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---

INDEX_DIR = Path("data/processed/faiss")
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRetriever:
    """Loads the FAISS index, metadata, and embedding model once."""

    def __init__(
        self,
        index_file: Path = INDEX_FILE,
        metadata_file: Path = METADATA_FILE,
        model_name: str = MODEL_NAME,
    ):
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_file}\n"
                f"   Run the indexer first: python -m src.retrieval.indexer"
            )

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_file))

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
        """Return the top-k most semantically similar chunks for a query."""
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32).reshape(1, -1)

        n_to_fetch = top_k * 5 if source_filter else top_k

        scores, indices = self.index.search(query_embedding, n_to_fetch)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue

            source = self.metadata["sources"][idx]
            if source_filter and source != source_filter:
                continue

            results.append({
                "chunk_id": self.metadata["chunk_ids"][idx],
                "text": self.metadata["texts"][idx],
                "score": float(score),
                "metadata": {
                    "source": source,
                    "parent_title": self.metadata["parent_titles"][idx],
                    "parent_url": self.metadata["parent_urls"][idx],
                    "parent_doc_id": self.metadata["parent_doc_ids"][idx],
                },
            })

            if len(results) >= top_k:
                break

        return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.retriever \"your question here\"")
        print("Example: python -m src.retrieval.retriever \"symptoms of type 2 diabetes\"")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 5

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    retriever = SemanticRetriever()
    results = retriever.search(query, top_k=top_k)

    print(f"\nTop {top_k} retrieved chunks:\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] Score: {r['score']:.3f} | Source: {r['metadata']['source']}")
        print(f"    Title: {r['metadata']['parent_title']}")
        print(f"    URL: {r['metadata']['parent_url']}")
        print(f"    Excerpt: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()