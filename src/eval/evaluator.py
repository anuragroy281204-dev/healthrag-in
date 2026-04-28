"""
RAG Evaluation Runner
=====================
Runs the full test set through the RAG pipeline, scores every answer
with the LLM-as-judge module, and saves raw results to disk.

The output is a single JSON file containing per-question results.
A separate module (reporter.py) consumes this file to generate
human-readable Markdown reports and charts.

Usage:
    # Full evaluation run (~7-10 min for 30 questions)
    python -m src.eval.evaluator

    # Quick smoke test (3 questions only)
    python -m src.eval.evaluator --smoke

    # Resume from a partial run
    python -m src.eval.evaluator --resume

Output:
    data/eval_results/run_<timestamp>.json
    data/eval_results/latest.json   (symlink-style copy of most recent)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.eval.test_set import TEST_SET
from src.eval.judges import run_all_judges
from src.generation.answer import RAGPipeline


# --- Configuration ---

RESULTS_DIR = Path("data/eval_results")

# Sleep between questions to respect API rate limits.
# Each question uses ~4 LLM calls (1 generation + 3 judges)
# At free tier 30 req/min ceiling, ~10 sec between questions is safe.
INTER_QUESTION_DELAY_SECONDS = 4.0


# --- Helper functions ---

def estimate_eta(questions_done: int, total: int, elapsed_sec: float) -> str:
    """Estimate time remaining based on average per-question duration."""
    if questions_done == 0:
        return "calculating..."
    avg = elapsed_sec / questions_done
    remaining_sec = avg * (total - questions_done)
    mins = int(remaining_sec // 60)
    secs = int(remaining_sec % 60)
    return f"~{mins}m {secs}s"


def evaluate_one(rag: RAGPipeline, test_question: dict) -> dict:
    """
    Run a single test question through the full pipeline.

    Returns a dict with question, RAG output, and all judge scores.
    """
    qid = test_question["id"]
    category = test_question["category"]
    question = test_question["question"]
    expected = test_question["expected_behavior"]

    # Step 1: run RAG pipeline
    start = time.time()
    rag_result = rag.ask(question)
    rag_time = time.time() - start

    # Step 2: run all judges
    judge_start = time.time()
    judge_results = run_all_judges(
        question=question,
        expected_behavior=expected,
        answer=rag_result["answer"],
        retrieved_chunks=rag_result["retrieved_chunks"],
    )
    judge_time = time.time() - judge_start

    # Build a clean record
    return {
        "id": qid,
        "category": category,
        "question": question,
        "expected_behavior": expected,
        "notes": test_question.get("notes", ""),

        # RAG output
        "rag_answer": rag_result["answer"],
        "rag_is_refusal": rag_result["is_refusal"],
        "rag_is_emergency": rag_result["is_emergency"],
        "rag_skipped_llm": rag_result.get("skipped_llm", False),
        "rag_cited_sources": sorted(rag_result["cited_source_numbers"]),
        "rag_retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "source": c["metadata"]["source"],
                "title": c["metadata"]["parent_title"],
                "url": c["metadata"]["parent_url"],
                "text": c["text"][:400],  # truncate to keep results file small
            }
            for c in rag_result["retrieved_chunks"]
        ],
        "rag_time_seconds": round(rag_time, 2),

        # Judge scores
        "judges": judge_results,
        "judge_time_seconds": round(judge_time, 2),
        "total_time_seconds": round(rag_time + judge_time, 2),
    }


def print_question_summary(idx: int, total: int, result: dict, eta: str) -> None:
    """Print a one-paragraph summary of one question's evaluation."""
    qid = result["id"]
    cat = result["category"]
    expected = result["expected_behavior"]
    refusal = result["judges"]["refusal_correctness"]
    actual = refusal.get("actual_behavior", "?")
    correct = "✓" if refusal["correct"] else "✗"

    f_score = result["judges"]["faithfulness"].get("score", "—")
    r_score = result["judges"]["answer_relevance"].get("score", "—")
    p_score = result["judges"]["context_precision"].get("score", "—")
    c_score = result["judges"]["citation_accuracy"].get("score", "—")

    print(
        f"  [{idx}/{total}] {qid} ({cat}) | {correct} expected={expected}, got={actual}"
    )
    print(
        f"      faith={f_score} | rel={r_score} | prec={p_score} | cite={c_score}"
        f" | t={result['total_time_seconds']:.1f}s | ETA: {eta}"
    )


def save_run(results: list, metadata: dict, results_dir: Path) -> Path:
    """Save the run to a timestamped JSON file. Also write 'latest.json'."""
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"run_{timestamp}.json"

    payload = {
        "metadata": metadata,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Also write a 'latest.json' for easy access
    latest_file = results_dir / "latest.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_file


def load_existing_results(results_dir: Path) -> dict:
    """Load existing latest.json if present (for --resume)."""
    latest = results_dir / "latest.json"
    if not latest.exists():
        return {}
    try:
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["id"]: r for r in data.get("results", [])}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run only first 3 questions (quick test).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest.json, skipping completed IDs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N questions.")
    args = parser.parse_args()

    # Pick test subset
    questions = TEST_SET
    if args.smoke:
        questions = TEST_SET[:3]
    elif args.limit:
        questions = TEST_SET[:args.limit]

    # Resume support
    existing = {}
    if args.resume:
        existing = load_existing_results(RESULTS_DIR)
        if existing:
            print(f"  -> Resuming. Found {len(existing)} previously-completed questions.")

    print("=" * 70)
    print(f"HealthRAG-IN Evaluation Run")
    print("=" * 70)
    print(f"  Questions to evaluate: {len(questions)}")
    if existing:
        print(f"  Skipping (already done): {sum(1 for q in questions if q['id'] in existing)}")
    print(f"  Estimated time: ~{len(questions) * 18 // 60} minutes")
    print()

    # Initialize RAG pipeline ONCE
    print("Initializing pipeline...")
    rag = RAGPipeline()
    print()

    # Run evaluations
    print("Running evaluations:")
    print("-" * 70)
    results = list(existing.values())  # start with previously completed
    completed_ids = set(existing.keys())
    run_start = time.time()

    for i, q in enumerate(questions, start=1):
        if q["id"] in completed_ids:
            continue

        try:
            result = evaluate_one(rag, q)
            results.append(result)
            completed_ids.add(q["id"])

            elapsed = time.time() - run_start
            remaining = [x for x in questions if x["id"] not in completed_ids]
            eta = estimate_eta(
                questions_done=len(completed_ids) - len(existing),
                total=len(questions) - len(existing),
                elapsed_sec=elapsed,
            )
            print_question_summary(i, len(questions), result, eta)

            # Save incrementally after every question
            save_run(
                results,
                metadata={
                    "started_at": datetime.now().isoformat(),
                    "questions_total": len(questions),
                    "questions_completed": len(results),
                    "smoke_mode": args.smoke,
                },
                results_dir=RESULTS_DIR,
            )

            # Polite delay before next question
            if i < len(questions):
                time.sleep(INTER_QUESTION_DELAY_SECONDS)

        except KeyboardInterrupt:
            print("\n\n[Interrupted] Saving partial results before exit...")
            save_run(
                results,
                metadata={"interrupted_at": datetime.now().isoformat()},
                results_dir=RESULTS_DIR,
            )
            print(f"  Partial results saved.")
            return

        except Exception as e:
            print(f"  [!] Question {q['id']} failed: {e}")
            print(f"      Continuing with next question.")
            continue

    # Final save with completion metadata
    output_file = save_run(
        results,
        metadata={
            "started_at": (datetime.now() - timedelta_seconds(time.time() - run_start)).isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - run_start, 1),
            "questions_total": len(questions),
            "questions_completed": len(results),
            "smoke_mode": args.smoke,
        },
        results_dir=RESULTS_DIR,
    )

    # Done
    total_time = time.time() - run_start
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    print("-" * 70)
    print(f"\n[Done] Evaluated {len(results)} questions in {mins}m {secs}s")
    print(f"  Saved to: {output_file}")
    print(f"  Latest: {RESULTS_DIR / 'latest.json'}")
    print(f"\n  Next: run `python -m src.eval.reporter` to generate the report.")


def timedelta_seconds(seconds):
    """Helper to subtract seconds from a datetime."""
    from datetime import timedelta
    return timedelta(seconds=seconds)


if __name__ == "__main__":
    main()