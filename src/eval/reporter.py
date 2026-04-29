"""
Evaluation Report Generator
===========================
Reads the raw evaluation JSON produced by evaluator.py and generates
a polished Markdown report with aggregate metrics, per-category
breakdowns, charts, and per-question details.

Usage:
    # Generate report from latest evaluation run
    python -m src.eval.reporter

    # Generate report from a specific run file
    python -m src.eval.reporter --input data/eval_results/run_20260428_143022.json

Output:
    data/eval_results/report_<timestamp>.md
    data/eval_results/charts_<timestamp>/  (PNG charts)
    data/eval_results/latest_report.md     (always points to most recent)
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt


# --- Configuration ---

RESULTS_DIR = Path("data/eval_results")


# --- Helpers ---

def safe_score(judge_dict, default=None):
    """Extract a score from a judge result, handling missing/error cases."""
    if not isinstance(judge_dict, dict):
        return default
    score = judge_dict.get("score")
    if score is None or isinstance(score, bool):
        return default
    try:
        return float(score)
    except (TypeError, ValueError):
        return default


def aggregate_metric(results, judge_key):
    """Compute mean/median/min/max for one metric across all questions."""
    scores = []
    for r in results:
        score = safe_score(r["judges"].get(judge_key, {}))
        if score is not None:
            scores.append(score)

    if not scores:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}

    return {
        "mean":   round(mean(scores), 3),
        "median": round(median(scores), 3),
        "min":    round(min(scores), 3),
        "max":    round(max(scores), 3),
        "n":      len(scores),
    }


def aggregate_by_category(results, judge_key):
    """Compute mean score per category for one metric."""
    by_cat = defaultdict(list)
    for r in results:
        score = safe_score(r["judges"].get(judge_key, {}))
        if score is not None:
            by_cat[r["category"]].append(score)

    return {
        cat: round(mean(scores), 3) if scores else None
        for cat, scores in by_cat.items()
    }


def category_pass_rate(results):
    """For each category, compute % of questions where refusal_correctness was 1.0."""
    by_cat = defaultdict(list)
    for r in results:
        score = safe_score(r["judges"].get("refusal_correctness", {}))
        if score is not None:
            by_cat[r["category"]].append(1 if score >= 1.0 else 0)

    return {
        cat: round(100 * sum(scores) / len(scores), 1) if scores else None
        for cat, scores in by_cat.items()
    }


# --- Chart generation ---

def chart_metric_by_category(results, charts_dir):
    """Bar chart: mean score per metric, grouped by category."""
    metrics = ["faithfulness", "answer_relevance", "context_precision",
               "citation_accuracy", "refusal_correctness"]
    metric_labels = ["Faithfulness", "Relevance", "Precision",
                     "Citation Acc.", "Refusal Acc."]

    categories = sorted({r["category"] for r in results})
    cat_data = {m: aggregate_by_category(results, m) for m in metrics}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.18
    colors = {"factual": "#1f77b4", "comparative": "#2ca02c",
              "out_of_scope": "#ff7f0e", "adversarial": "#d62728"}

    for i, cat in enumerate(categories):
        values = [cat_data[m].get(cat) or 0 for m in metrics]
        offsets = [pos + (i - len(categories) / 2) * width + width / 2 for pos in x]
        ax.bar(offsets, values, width,
               label=cat.replace("_", " ").title(),
               color=colors.get(cat, "#888"))

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean score (0.0 – 1.0)")
    ax.set_title("Evaluation Metrics by Category")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = charts_dir / "metrics_by_category.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def chart_overall_scores(results, charts_dir):
    """Bar chart: overall mean score for each metric."""
    metrics = [
        ("faithfulness",        "Faithfulness"),
        ("answer_relevance",    "Answer Relevance"),
        ("context_precision",   "Context Precision"),
        ("citation_accuracy",   "Citation Accuracy"),
        ("refusal_correctness", "Refusal Correctness"),
    ]

    means = []
    labels = []
    for key, label in metrics:
        agg = aggregate_metric(results, key)
        if agg["mean"] is not None:
            means.append(agg["mean"])
            labels.append(label)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, color="#1f77b4", edgecolor="#0a4275")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean score (0.0 – 1.0)")
    ax.set_title("HealthRAG-IN — Overall Evaluation Scores")
    ax.grid(axis="y", alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = charts_dir / "overall_scores.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def chart_latency(results, charts_dir):
    """Histogram: latency distribution per question."""
    latencies = [r.get("rag_time_seconds", 0) for r in results
                 if r.get("rag_time_seconds") is not None]
    if not latencies:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(latencies, bins=12, color="#2ca02c", edgecolor="#1a5d1a")
    ax.set_xlabel("RAG response time (seconds)")
    ax.set_ylabel("Number of questions")
    ax.set_title(
        f"RAG Latency — mean {mean(latencies):.2f}s, "
        f"median {median(latencies):.2f}s, max {max(latencies):.2f}s"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = charts_dir / "latency_histogram.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


# --- Markdown report ---

def render_score_emoji(score):
    """Return an emoji indicator for a score."""
    if score is None:
        return "—"
    if score >= 0.85:
        return "🟢"
    if score >= 0.70:
        return "🟡"
    return "🔴"


def render_aggregate_table(results):
    """Render the aggregate metrics table as Markdown."""
    metrics = [
        ("faithfulness",        "Faithfulness",        "Are claims supported by sources?"),
        ("answer_relevance",    "Answer Relevance",    "Does the answer address the question?"),
        ("context_precision",   "Context Precision",   "Were retrieved chunks relevant?"),
        ("citation_accuracy",   "Citation Accuracy",   "Do [N] markers point to real chunks?"),
        ("refusal_correctness", "Refusal Correctness", "Refused out-of-scope correctly?"),
    ]

    lines = [
        "| Metric | Description | Mean | Median | Min | Max | N |",
        "|--------|-------------|------|--------|-----|-----|---|",
    ]
    for key, label, desc in metrics:
        agg = aggregate_metric(results, key)
        if agg["n"] == 0:
            lines.append(f"| {label} | {desc} | — | — | — | — | 0 |")
            continue
        emoji = render_score_emoji(agg["mean"])
        lines.append(
            f"| {emoji} {label} | {desc} | "
            f"{agg['mean']:.3f} | {agg['median']:.3f} | "
            f"{agg['min']:.3f} | {agg['max']:.3f} | {agg['n']} |"
        )
    return "\n".join(lines)


def render_category_table(results):
    """Render the by-category breakdown."""
    metrics = [
        ("faithfulness",        "Faith"),
        ("answer_relevance",    "Rel"),
        ("context_precision",   "Prec"),
        ("citation_accuracy",   "Cite"),
        ("refusal_correctness", "Refuse"),
    ]

    categories = sorted({r["category"] for r in results})
    cat_counts = defaultdict(int)
    for r in results:
        cat_counts[r["category"]] += 1

    headers = ["Category", "N"] + [label for _, label in metrics]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for cat in categories:
        row = [cat.replace("_", " ").title(), str(cat_counts[cat])]
        for key, _ in metrics:
            agg = aggregate_by_category(results, key).get(cat)
            row.append(f"{agg:.2f}" if agg is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_failure_modes(results, top_n=5):
    """Show the worst-scoring questions for debugging."""
    scored = []
    for r in results:
        f_score = safe_score(r["judges"].get("faithfulness", {}))
        if f_score is not None and not r["judges"]["faithfulness"].get("skipped"):
            scored.append((f_score, r))

    scored.sort(key=lambda x: x[0])
    worst = scored[:top_n]
    if not worst:
        return "_No factual answers to analyse (all questions were refusals)._\n"

    lines = []
    for score, r in worst:
        unsupported = r["judges"]["faithfulness"].get("unsupported_claims", [])
        reasoning = r["judges"]["faithfulness"].get("reasoning", "")
        lines.append(f"### {r['id']} ({r['category']}) — Faithfulness {score:.2f}\n")
        lines.append(f"**Question:** {r['question']}\n")
        lines.append(f"**Judge reasoning:** {reasoning}\n")
        if unsupported:
            lines.append("**Unsupported claims:**")
            for claim in unsupported[:3]:
                lines.append(f"- {claim}")
            lines.append("")
        lines.append("---\n")

    return "\n".join(lines)


def render_per_question_table(results):
    """Render the full per-question results table."""
    lines = [
        "| ID | Category | Faith | Rel | Prec | Cite | Refuse | Provider | Time (s) |",
        "|----|----------|-------|-----|------|------|--------|----------|----------|",
    ]
    for r in sorted(results, key=lambda x: x["id"]):
        f = safe_score(r["judges"].get("faithfulness", {}))
        rel = safe_score(r["judges"].get("answer_relevance", {}))
        p = safe_score(r["judges"].get("context_precision", {}))
        c = safe_score(r["judges"].get("citation_accuracy", {}))
        ref = safe_score(r["judges"].get("refusal_correctness", {}))

        provider = "—"
        chunks = r.get("rag_retrieved_chunks", [])
        if chunks and isinstance(r.get("rag_answer"), str):
            provider = "groq/gemini"

        time_s = r.get("rag_time_seconds", 0)

        def fmt(x):
            return f"{x:.2f}" if x is not None else "—"

        lines.append(
            f"| {r['id']} | {r['category']} | {fmt(f)} | {fmt(rel)} | "
            f"{fmt(p)} | {fmt(c)} | {fmt(ref)} | {provider} | {time_s:.1f} |"
        )

    return "\n".join(lines)


def build_report(payload, charts_dir, charts_relative_path):
    """Build the full Markdown report string."""
    metadata = payload.get("metadata", {})
    results = payload.get("results", [])

    if not results:
        return "# HealthRAG-IN Evaluation Report\n\n_No results to report._\n"

    started = metadata.get("started_at", "unknown")
    completed = metadata.get("completed_at", metadata.get("interrupted_at", "unknown"))
    duration = metadata.get("duration_seconds")
    smoke = metadata.get("smoke_mode", False)

    pass_rates = category_pass_rate(results)

    md = []
    md.append("# HealthRAG-IN — Evaluation Report")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    md.append("## Run Summary\n")
    md.append(f"- **Total questions evaluated:** {len(results)}")
    md.append(f"- **Mode:** {'Smoke (subset)' if smoke else 'Full test set'}")
    md.append(f"- **Started:** {started}")
    md.append(f"- **Completed:** {completed}")
    if duration:
        m, s = divmod(int(duration), 60)
        md.append(f"- **Duration:** {m}m {s}s")
    md.append("")

    md.append("## Aggregate Metrics\n")
    md.append("Higher is better for all metrics. 🟢 ≥ 0.85, 🟡 0.70-0.85, 🔴 < 0.70.\n")
    md.append(render_aggregate_table(results))
    md.append("")

    md.append("## By Category\n")
    md.append(render_category_table(results))
    md.append("")

    md.append("### Refusal-correctness pass rate by category\n")
    for cat, rate in pass_rates.items():
        md.append(f"- **{cat.replace('_', ' ').title()}:** {rate}%")
    md.append("")

    md.append("## Charts\n")
    md.append(f"![Overall scores]({charts_relative_path}/overall_scores.png)\n")
    md.append(f"![By category]({charts_relative_path}/metrics_by_category.png)\n")
    md.append(f"![Latency]({charts_relative_path}/latency_histogram.png)\n")

    md.append("## Failure Modes — Worst Faithfulness Scores\n")
    md.append(render_failure_modes(results))

    md.append("## Full Per-Question Results\n")
    md.append(render_per_question_table(results))
    md.append("")

    md.append("---\n")
    md.append(f"_HealthRAG-IN — automated evaluation report — {len(results)} questions_")

    return "\n".join(md)


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to evaluation JSON. Defaults to latest.json.")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = RESULTS_DIR / "latest.json"

    if not input_path.exists():
        print(f"[!] Evaluation file not found: {input_path}")
        print("    Run the evaluator first: python -m src.eval.evaluator")
        return

    print(f"  -> Loading evaluation results from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    print(f"  -> Loaded {len(results)} question results")

    if not results:
        print("[!] No results found in file.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = RESULTS_DIR / f"charts_{timestamp}"
    charts_dir.mkdir(parents=True, exist_ok=True)

    print(f"  -> Generating charts in {charts_dir}")
    chart_overall_scores(results, charts_dir)
    chart_metric_by_category(results, charts_dir)
    chart_latency(results, charts_dir)

    charts_relative = f"charts_{timestamp}"
    print(f"  -> Building Markdown report")
    md = build_report(payload, charts_dir, charts_relative)

    report_file = RESULTS_DIR / f"report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md)

    latest_report = RESULTS_DIR / "latest_report.md"
    with open(latest_report, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[Done]")
    print(f"  Report: {report_file}")
    print(f"  Latest: {latest_report}")
    print(f"  Charts: {charts_dir}")


if __name__ == "__main__":
    main()