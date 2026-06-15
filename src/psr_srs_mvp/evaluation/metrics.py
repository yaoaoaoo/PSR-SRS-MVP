"""Offline retrieval evaluation metrics — Precision, Recall, MRR, NDCG.

All implementations are self-contained (no ``scikit-learn`` or ``numpy``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from psr_srs_mvp.retrieval.bm25 import SearchResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Per-query metric values for a set of K cutoffs."""

    query_id: str
    query_text: str
    ks: list[int] = field(default_factory=list)
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query_id": self.query_id, "query_text": self.query_text}
        for k in self.ks:
            d[f"precision_at_{k}"] = self.precision.get(k, 0.0)
            d[f"recall_at_{k}"] = self.recall.get(k, 0.0)
            d[f"mrr_at_{k}"] = self.mrr.get(k, 0.0)
            d[f"ndcg_at_{k}"] = self.ndcg.get(k, 0.0)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dcg(rels: list[int]) -> float:
    """Discounted Cumulative Gain with log base 2 discount (rank 1-based)."""
    return sum(
        (2.0 ** rel - 1.0) / math.log2(i + 2)
        for i, rel in enumerate(rels)
    )


def _ideal_rels(
    qrels_for_query: dict[str, int],
    top_k: int,
) -> list[int]:
    """Sorted descending list of the top *top_k* relevance grades for a query."""
    sorted_grades = sorted(qrels_for_query.values(), reverse=True)
    return sorted_grades[:top_k]


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    results: Sequence[SearchResult],
    query_id: str,
    query_text: str,
    qrels: dict[str, dict[str, int]],
    ks: Sequence[int],
    relevance_threshold: int = 1,
) -> MetricResult:
    """Evaluate one query's results at multiple K cutoffs.

    Args:
        results: Ranked search results for this query.
        query_id: Query identifier.
        query_text: Raw query string (for output).
        qrels: Full qrels dict ``{qid: {iid: grade}}``.
        ks: List of K cutoffs (e.g. [5, 10, 20]).
        relevance_threshold: Minimum grade to consider an item "relevant".

    Returns:
        A ``MetricResult`` with precision, recall, MRR, NDCG for each K.
    """
    qrels_for_query = qrels.get(query_id, {})
    total_relevant = sum(
        1 for g in qrels_for_query.values() if g >= relevance_threshold
    )

    metric = MetricResult(query_id=query_id, query_text=query_text, ks=list(ks))

    for k in ks:
        top_k = results[:k]

        # ---- Precision@K ----
        rel_in_top = sum(
            1 for r in top_k
            if qrels_for_query.get(r.item_id, 0) >= relevance_threshold
        )
        metric.precision[k] = rel_in_top / k if k > 0 else 0.0

        # ---- Recall@K ----
        metric.recall[k] = rel_in_top / total_relevant if total_relevant > 0 else 0.0

        # ---- MRR@K ----
        mrr = 0.0
        for rank, r in enumerate(top_k, start=1):
            if qrels_for_query.get(r.item_id, 0) >= relevance_threshold:
                mrr = 1.0 / rank
                break
        metric.mrr[k] = mrr

        # ---- NDCG@K ----
        rels = [
            qrels_for_query.get(r.item_id, 0)
            for r in top_k
        ]
        dcg = _dcg(rels)
        ideal = _ideal_rels(qrels_for_query, k)
        idcg = _dcg(ideal)
        metric.ndcg[k] = dcg / idcg if idcg > 0 else 0.0

    return metric


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    all_results: dict[str, list[SearchResult]],
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, int]],
    ks: Sequence[int],
    relevance_threshold: int = 1,
) -> list[MetricResult]:
    """Evaluate all queries.

    Args:
        all_results: ``{query_id: [SearchResult, ...]}``.
        queries: List of query dicts from ``load_queries``.
        qrels: Qrels dict from ``load_qrels``.
        ks: K cutoffs.
        relevance_threshold: Minimum grade for relevance.

    Returns:
        One ``MetricResult`` per query that has search results.
    """
    query_texts = {q["query_id"]: q["query_text"] for q in queries}
    metrics: list[MetricResult] = []

    for qid in sorted(all_results.keys()):
        results = all_results[qid]
        text = query_texts.get(qid, qid)
        m = evaluate_query(
            results, qid, text, qrels, ks, relevance_threshold,
        )
        metrics.append(m)

    return metrics


def macro_average(metrics: list[MetricResult]) -> dict[str, dict[int, float]]:
    """Compute macro-average (mean) of each metric across all queries.

    Returns:
        ``{"precision": {5: float, 10: float, ...}, "recall": {...}, ...}``
    """
    if not metrics:
        return {}

    ks = metrics[0].ks
    sums: dict[str, dict[int, float]] = {
        "precision": {},
        "recall": {},
        "mrr": {},
        "ndcg": {},
    }
    counts = len(metrics)

    for m in metrics:
        for k in ks:
            sums["precision"][k] = sums["precision"].get(k, 0.0) + m.precision.get(k, 0.0)
            sums["recall"][k] = sums["recall"].get(k, 0.0) + m.recall.get(k, 0.0)
            sums["mrr"][k] = sums["mrr"].get(k, 0.0) + m.mrr.get(k, 0.0)
            sums["ndcg"][k] = sums["ndcg"].get(k, 0.0) + m.ndcg.get(k, 0.0)

    return {
        metric_name: {k: v / counts for k, v in kvs.items()}
        for metric_name, kvs in sums.items()
    }
