"""Evaluate the current CRM retriever on the golden retrieval set.

This is the baseline measurement required before introducing a reranker. It
does not evaluate structured SQL tools; count/list questions belong in the
agent/tool benchmark because contacts, leads, and aggregates are returned by
tools rather than the document retriever.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from common.crm_retrieval import hybrid_search
from common.retrieval_reranker import rerank

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "retrieval_eval.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "retrieval_baseline.json"


def _load_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def _ndcg(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    gains = [relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(k_values: tuple[int, ...] = (5, 10), use_reranker: bool = False) -> dict:
    rows = _load_rows()
    per_query = []
    for row in rows:
        relevance = {item["document_id"]: int(item["relevance"]) for item in row["relevant_documents"]}
        hits = hybrid_search(row["question"], account_id=row.get("account_id"), limit=max(k_values, default=10) * 3)
        if use_reranker:
            hits = rerank(row["question"], hits, account_id=row.get("account_id"), limit=max(k_values))
        else:
            hits = hits[:max(k_values)]
        ranked_ids = [hit["document_id"] for hit in hits]
        first_relevant_rank = next((index + 1 for index, doc_id in enumerate(ranked_ids) if doc_id in relevance), None)
        per_query.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "ranked_documents": ranked_ids,
            "first_relevant_rank": first_relevant_rank,
            "recall_at": {str(k): int(any(doc_id in relevance for doc_id in ranked_ids[:k])) for k in k_values},
            "ndcg_at": {str(k): round(_ndcg(ranked_ids, relevance, k), 6) for k in k_values},
        })

    count = max(1, len(per_query))
    metrics = {}
    for k in k_values:
        metrics[f"recall_at_{k}"] = round(sum(item["recall_at"][str(k)] for item in per_query) / count, 6)
        metrics[f"ndcg_at_{k}"] = round(sum(item["ndcg_at"][str(k)] for item in per_query) / count, 6)
    metrics["mrr"] = round(
        sum(1 / item["first_relevant_rank"] if item["first_relevant_rank"] else 0 for item in per_query) / count,
        6,
    )
    return {
        "retriever": "hybrid_search+deterministic_reranker" if use_reranker else "current_hybrid_search",
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "queries": len(per_query),
        "metrics": metrics,
        "per_query": per_query,
        "notes": [
            "This is a reranked comparison." if use_reranker else "This is a baseline before reranking.",
            "Structured-only count/list questions are evaluated in the tool benchmark, not here.",
            "delta-MRR is undefined until a second retriever or reranker is measured.",
        ],
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranked", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        baseline = evaluate()
        reranked = evaluate(use_reranker=True)
        comparison = {
            "baseline": baseline,
            "reranked": reranked,
            "delta_metrics": {
                key: round(reranked["metrics"].get(key, 0) - baseline["metrics"].get(key, 0), 6)
                for key in baseline["metrics"]
            },
            "decision": "use_reranker" if (
                reranked["metrics"]["mrr"] > baseline["metrics"]["mrr"]
                or reranked["metrics"]["ndcg_at_5"] > baseline["metrics"]["ndcg_at_5"]
            ) else "keep_baseline",
        }
        output_path = PROJECT_ROOT / "results" / "retrieval_comparison.json"
        output_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"baseline": baseline["metrics"], "reranked": reranked["metrics"], "delta": comparison["delta_metrics"], "decision": comparison["decision"]}, indent=2))
        print(f"Wrote {output_path}")
        return
    result = evaluate(use_reranker=args.reranked)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_path = PROJECT_ROOT / "results" / ("retrieval_reranked.json" if args.reranked else "retrieval_baseline.json")
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
