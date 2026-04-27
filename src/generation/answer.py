"""
End-to-End RAG Pipeline
=======================
Combines retrieval (HybridRetriever) with generation (GroqGenerator)
into a single answer-the-question function.

Usage from another module:
    from src.generation.answer import RAGPipeline
    rag = RAGPipeline()
    result = rag.ask("what is HbA1c?")
    print(result["answer"])

Usage as CLI:
    python -m src.generation.answer "your question here"
"""

import sys
import time

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import GroqGenerator


# --- Configuration ---

# How many chunks to retrieve and pass to the LLM.
# More chunks = more context but more tokens (slower, more $$).
# 5 is the standard production choice.
TOP_K = 5

# Minimum RRF score for a chunk to be considered "good enough".
# Below this, the pipeline triggers a refusal without calling the LLM.
# Tuning this is part of evaluation in Step 12.
MIN_RELEVANCE_SCORE = 0.01


# --- Pipeline class ---

class RAGPipeline:
    """
    The full retrieval-augmented generation pipeline for HealthRAG-IN.
    Loads the retriever and generator once at init time.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        print("[RAGPipeline] Loading retrieval layer...")
        self.retriever = HybridRetriever()

        print("[RAGPipeline] Loading generation layer...")
        self.generator = GroqGenerator()

        self.top_k = top_k
        self.min_relevance_score = min_relevance_score

        print("[RAGPipeline] Ready.\n")

    def ask(
        self,
        question: str,
        source_filter: str | None = None,
    ) -> dict:
        """
        Ask a medical question and get back a grounded, cited answer.

        Args:
            question: the user's question
            source_filter: optional - "WHO", "PubMed", or "ICMR"

        Returns a dict with:
          - question: the original question
          - answer: the LLM's response text
          - retrieved_chunks: the chunks passed to the LLM
          - is_refusal: True if the system refused
          - is_emergency: True if emergency warning was issued
          - cited_source_numbers: which source numbers the LLM cited
          - retrieval_time_sec, generation_time_sec, total_time_sec
        """
        # Step 1: retrieve
        retrieval_start = time.time()
        chunks = self.retriever.search(
            question,
            top_k=self.top_k,
            source_filter=source_filter,
        )
        retrieval_time = time.time() - retrieval_start

        # Step 2: relevance gate - if no chunks scored above threshold,
        # trigger a refusal without calling the LLM (saves time + API quota)
        if not chunks or chunks[0]["score"] < self.min_relevance_score:
            return {
                "question": question,
                "answer": (
                    "I don't have grounded sources to answer this question reliably. "
                    "Please consult a qualified medical professional or refer to "
                    "authoritative resources like ICMR guidelines or your physician."
                ),
                "retrieved_chunks": chunks,
                "is_refusal": True,
                "is_emergency": False,
                "cited_source_numbers": set(),
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": 0.0,
                "total_time_sec": retrieval_time,
                "skipped_llm": True,
            }

        # Step 3: generate
        generation_start = time.time()
        gen_result = self.generator.generate(question, chunks)
        generation_time = time.time() - generation_start

        return {
            "question": question,
            "answer": gen_result["text"],
            "retrieved_chunks": chunks,
            "is_refusal": gen_result["is_refusal"],
            "is_emergency": gen_result["is_emergency"],
            "cited_source_numbers": gen_result["cited_source_numbers"],
            "retrieval_time_sec": retrieval_time,
            "generation_time_sec": generation_time,
            "total_time_sec": retrieval_time + generation_time,
            "skipped_llm": False,
            "model_name": gen_result["model_name"],
            "prompt_version": gen_result["prompt_version"],
        }


# --- CLI ---

def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.generation.answer "your question here"')
        print('Example: python -m src.generation.answer "what is HbA1c"')
        sys.exit(1)

    question = sys.argv[1]

    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    rag = RAGPipeline()
    result = rag.ask(question)

    # Print retrieved sources
    print("\n" + "=" * 70)
    print(f"RETRIEVED SOURCES (top {len(result['retrieved_chunks'])})")
    print("=" * 70)
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        meta = chunk["metadata"]
        print(f"[{i}] ({meta['source']}) {meta['parent_title']}")
        if meta.get('parent_url'):
            print(f"    URL: {meta['parent_url']}")

    # Print the answer
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    # Print metadata
    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    print(f"  Is refusal: {result['is_refusal']}")
    print(f"  Is emergency: {result['is_emergency']}")
    print(f"  Cited sources: {sorted(result['cited_source_numbers'])}")
    print(f"  Skipped LLM (low relevance): {result.get('skipped_llm', False)}")
    print(f"  Retrieval time: {result['retrieval_time_sec']:.2f}s")
    print(f"  Generation time: {result['generation_time_sec']:.2f}s")
    print(f"  Total time: {result['total_time_sec']:.2f}s")


if __name__ == "__main__":
    main()