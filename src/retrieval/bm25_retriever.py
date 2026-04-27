"""
BM25 Keyword Retriever
======================
Builds an in-memory BM25 index over the same chunks indexed by
FAISS, and provides a search interface for keyword-based retrieval.

BM25 catches exact medical terminology, drug names, codes, and
acronyms that pure semantic search may miss.

Usage from another module:
    from src.retrieval.bm25_retriever import BM25Retriever
    retriever = BM25Retriever()
    results = retriever.search("HbA1c", top_k=5)

Usage as CLI:
    python -m src.retrieval.bm25_retriever "your query here"
"""

import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

# --- Configuration ---

# Reuse the same metadata file as the FAISS retriever
METADATA_FILE = Path("data/processed/faiss/metadata.json")


# --- Tokenization ---

def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.

    Steps:
      1. Lowercase everything
      2. Split on word boundaries (regex \\w+)
      3. Drop very short tokens (< 2 chars) - mostly noise
      4. Drop pure numbers? - NO. We keep numbers because clinical
         thresholds (7%, 126 mg/dL) matter in medical search.

    Returns: a list of lowercase token strings.
    """
    # Convert to lowercase
    text = text.lower()

    # Find all sequences of word characters
    # \w+ matches letters, digits, underscores
    tokens = re.findall(r"\w+", text)

    # Filter very short tokens (1-char tokens add noise without info)
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens


# --- BM25 Retriever class ---

class BM25Retriever:
    """
    Loads chunk metadata once, tokenizes all chunks, and builds
    an in-memory BM25 index. Reuses the index across queries.
    """

    def __init__(self, metadata_file: Path = METADATA_FILE):
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"   Run the FAISS indexer first: python -m src.retrieval.indexer"
            )

        # Load chunk metadata (same file used by FAISS retriever)
        print("  -> Loading chunk metadata...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.chunk_count = len(self.metadata["texts"])
        print(f"  -> Tokenizing {self.chunk_count} chunks...")

        # Tokenize every chunk's text
        # This is one-time work; ~5-10 seconds for 1000+ chunks
        self.tokenized_corpus = [
            tokenize(text) for text in self.metadata["texts"]
        ]

        # Build the BM25 index
        # BM25Okapi is the most common variant - same one Lucene uses
        print("  -> Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  -> BM25 index ready ({self.chunk_count} chunks).")

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Return the top-k chunks matching the query by BM25 score.

        Args:
            query: the user's question
            top_k: number of chunks to return
            source_filter: optional - "WHO", "PubMed", or "ICMR"

        Returns: a list of dicts with chunk_id, text, score, and metadata.
        """
        # Tokenize the query the same way we tokenized chunks
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores for ALL chunks (returns numpy array)
        scores = self.bm25.get_scores(query_tokens)

        # Sort indices by score descending
        # We sort all chunks then filter, because filtering first
        # would drop chunks before scoring
        sorted_indices = scores.argsort()[::-1]

        results = []
        for idx in sorted_indices:
            # Stop if score is 0 (no overlap with query)
            if scores[idx] <= 0:
                break

            source = self.metadata["sources"][idx]
            if source_filter and source != source_filter:
                continue

            results.append({
                "chunk_id": self.metadata["chunk_ids"][idx],
                "text": self.metadata["texts"][idx],
                "score": float(scores[idx]),
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


# --- CLI for quick testing ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.bm25_retriever \"your query here\"")
        print("Example: python -m src.retrieval.bm25_retriever \"HbA1c metformin\"")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 5

    print("=" * 60)
    print(f"Query: {query}")
    print("BM25 (keyword search)")
    print("=" * 60)

    retriever = BM25Retriever()
    results = retriever.search(query, top_k=top_k)

    if not results:
        print("\n[!] No matching chunks found (no keyword overlap).")
        return

    print(f"\nTop {len(results)} retrieved chunks:\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] BM25 Score: {r['score']:.3f} | Source: {r['metadata']['source']}")
        print(f"    Title: {r['metadata']['parent_title']}")
        print(f"    Excerpt: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()