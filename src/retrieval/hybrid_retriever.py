"""
Hybrid Retriever (Semantic + BM25 via Reciprocal Rank Fusion)
==============================================================
Combines the SemanticRetriever (FAISS) and BM25Retriever using
Reciprocal Rank Fusion (RRF) to produce a single ranked list
that benefits from both semantic understanding and exact keyword
matching.

This is the production retrieval interface. The generation layer
in Step 11 will call only this class.

Usage from another module:
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.search("HbA1c target elderly", top_k=5)

Usage as CLI:
    python -m src.retrieval.hybrid_retriever "your query here"
"""

import sys
from collections import defaultdict

from src.retrieval.retriever import SemanticRetriever
from src.retrieval.bm25_retriever import BM25Retriever


# --- Configuration ---

# RRF constant - 60 is the empirical sweet spot from the original paper
RRF_K = 60

# How many results to fetch from each retriever before fusing
# Larger = better recall but slower. 30 is a balanced choice for medical RAG.
PER_RETRIEVER_FETCH_SIZE = 30


# --- Hybrid Retriever class ---

class HybridRetriever:
    """
    Combines two retrievers via Reciprocal Rank Fusion.
    Loads both underlying retrievers once at init time.
    """

    def __init__(
        self,
        rrf_k: int = RRF_K,
        per_retriever_fetch_size: int = PER_RETRIEVER_FETCH_SIZE,
    ):
        self.rrf_k = rrf_k
        self.per_retriever_fetch_size = per_retriever_fetch_size

        print("[HybridRetriever] Initializing semantic retriever...")
        self.semantic = SemanticRetriever()

        print("[HybridRetriever] Initializing BM25 retriever...")
        self.bm25 = BM25Retriever()

        print("[HybridRetriever] Ready.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Run both retrievers, combine their results with RRF, and
        return the top-k unique chunks by RRF score.

        Args:
            query: the user's question
            top_k: number of final chunks to return
            source_filter: optional - "WHO", "PubMed", or "ICMR"

        Returns: list of dicts with chunk_id, text, scores, and metadata.
                 Each result also has individual retriever scores for debugging.
        """
        # Step 1: get ranked results from both retrievers
        semantic_results = self.semantic.search(
            query,
            top_k=self.per_retriever_fetch_size,
            source_filter=source_filter,
        )
        bm25_results = self.bm25.search(
            query,
            top_k=self.per_retriever_fetch_size,
            source_filter=source_filter,
        )

        # Step 2: build a mapping of chunk_id -> RRF score
        # We also track which retrievers found each chunk (useful for debugging)
        rrf_scores: dict[str, float] = defaultdict(float)
        retriever_ranks: dict[str, dict[str, int]] = defaultdict(dict)
        chunk_data: dict[str, dict] = {}

        # Add contributions from semantic retriever
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            retriever_ranks[chunk_id]["semantic"] = rank
            chunk_data[chunk_id] = result  # remember the chunk's metadata

        # Add contributions from BM25 retriever
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            retriever_ranks[chunk_id]["bm25"] = rank
            # Only set chunk_data if we don't already have it
            # (semantic and BM25 have the same metadata, so either is fine)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

        # Step 3: sort chunks by RRF score (descending)
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        # Step 4: build the final result list
        final_results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            data = chunk_data[chunk_id]
            final_results.append({
                "chunk_id": chunk_id,
                "text": data["text"],
                "score": rrf_scores[chunk_id],  # RRF score is the new "score"
                "metadata": data["metadata"],
                # Extra info for transparency / debugging
                "retriever_ranks": dict(retriever_ranks[chunk_id]),
            })

        return final_results


# --- CLI for quick testing ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.hybrid_retriever \"your query here\"")
        print("Example: python -m src.retrieval.hybrid_retriever \"HbA1c target elderly\"")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 5

    print("=" * 70)
    print(f"Query: {query}")
    print("Hybrid (Semantic + BM25 via RRF)")
    print("=" * 70)

    retriever = HybridRetriever()
    results = retriever.search(query, top_k=top_k)

    if not results:
        print("\n[!] No results returned.")
        return

    print(f"\nTop {len(results)} retrieved chunks (fused):\n")
    for i, r in enumerate(results, start=1):
        ranks = r["retriever_ranks"]
        rank_str_parts = []
        if "semantic" in ranks:
            rank_str_parts.append(f"sem #{ranks['semantic']}")
        if "bm25" in ranks:
            rank_str_parts.append(f"bm25 #{ranks['bm25']}")
        rank_summary = ", ".join(rank_str_parts) if rank_str_parts else "no individual ranks"

        print(f"[{i}] RRF Score: {r['score']:.4f} | {rank_summary}")
        print(f"    Source: {r['metadata']['source']}")
        print(f"    Title: {r['metadata']['parent_title']}")
        print(f"    URL: {r['metadata']['parent_url']}")
        print(f"    Excerpt: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()