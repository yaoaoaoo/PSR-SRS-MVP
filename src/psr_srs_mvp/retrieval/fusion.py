"""Hybrid retrieval fusion — RRF and weighted linear score combination.

Combines BM25 keyword results with LSA semantic results.
Neither qrels nor user-behaviour data are used in fusion.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from psr_srs_mvp.retrieval.bm25 import SearchResult as BM25Result
from psr_srs_mvp.retrieval.semantic import SemanticSearchResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_NORMALIZATIONS = ("min_max",)


@dataclass
class FusionConfig:
    """Typed configuration for hybrid retrieval fusion."""

    candidate_k: int = 100
    top_k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    relevance_threshold: int = 1
    rrf_k: int = 60
    bm25_weight: float = 0.5
    semantic_weight: float = 0.5
    score_normalization: str = "min_max"

    @property
    def max_k(self) -> int:
        return max(self.top_k_values) if self.top_k_values else 20

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.candidate_k < 1:
            errors.append("candidate_k must be >= 1")
        if self.candidate_k < self.max_k:
            errors.append(f"candidate_k ({self.candidate_k}) < max K ({self.max_k})")
        if not self.top_k_values:
            errors.append("top_k_values must not be empty")
        for k in self.top_k_values:
            if k <= 0 or not isinstance(k, int):
                errors.append(f"top_k_values must be positive ints, got {k}")
        if self.rrf_k <= 0:
            errors.append("rrf_k must be > 0")
        if self.bm25_weight < 0 or not math.isfinite(self.bm25_weight):
            errors.append("bm25_weight must be non-negative finite")
        if self.semantic_weight < 0 or not math.isfinite(self.semantic_weight):
            errors.append("semantic_weight must be non-negative finite")
        if self.bm25_weight + self.semantic_weight <= 0:
            errors.append("at least one weight must be > 0")
        if self.score_normalization not in _VALID_NORMALIZATIONS:
            errors.append(f"score_normalization must be one of {_VALID_NORMALIZATIONS}")
        if self.relevance_threshold not in (1, 2, 3):
            errors.append("relevance_threshold must be 1, 2, or 3")
        return errors

    @classmethod
    def from_json(cls, path: str | Path) -> "FusionConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(**{k: v for k, v in raw.items()
                     if k in cls.__dataclass_fields__})
        errs = cfg.validate()
        if errs:
            raise ValueError("\n".join(errs))
        return cfg


# ---------------------------------------------------------------------------
# Fused search result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusedSearchResult:
    """A single result after fusion of BM25 + LSA."""

    item_id: str
    rank: int
    fusion_score: float
    bm25_rank: int | None
    semantic_rank: int | None
    bm25_score: float | None
    semantic_score: float | None
    bm25_normalized_score: float | None = None
    semantic_normalized_score: float | None = None
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "rank": str(self.rank),
            "fusion_score": f"{self.fusion_score:.6f}",
            "bm25_rank": str(self.bm25_rank) if self.bm25_rank is not None else "",
            "semantic_rank": str(self.semantic_rank) if self.semantic_rank is not None else "",
            "bm25_score": f"{self.bm25_score:.6f}" if self.bm25_score is not None else "",
            "semantic_score": f"{self.semantic_score:.6f}" if self.semantic_score is not None else "",
            "bm25_normalized_score": f"{self.bm25_normalized_score:.6f}" if self.bm25_normalized_score is not None else "",
            "semantic_normalized_score": f"{self.semantic_normalized_score:.6f}" if self.semantic_normalized_score is not None else "",
            "sources": ",".join(self.sources),
            "fusion_method": "",
        }


# ---------------------------------------------------------------------------
# Candidate building
# ---------------------------------------------------------------------------

def build_candidates(
    bm25_results: list[BM25Result],
    semantic_results: list[SemanticSearchResult],
) -> dict[str, dict[str, Any]]:
    """Build a unified candidate dict from two result lists.

    Returns:
        ``{item_id: {"bm25_rank": int|None, "bm25_score": float|None,
                     "semantic_rank": int|None, "semantic_score": float|None,
                     "sources": tuple}}``
    """
    candidates: dict[str, dict[str, Any]] = {}

    for r in bm25_results:
        candidates[r.item_id] = {
            "bm25_rank": r.rank,
            "bm25_score": r.score,
            "semantic_rank": None,
            "semantic_score": None,
            "sources": ("bm25",),
        }

    for r in semantic_results:
        if r.item_id in candidates:
            entry = candidates[r.item_id]
            entry["semantic_rank"] = r.rank
            entry["semantic_score"] = r.score
            entry["sources"] = ("bm25", "semantic")
        else:
            candidates[r.item_id] = {
                "bm25_rank": None,
                "bm25_score": None,
                "semantic_rank": r.rank,
                "semantic_score": r.score,
                "sources": ("semantic",),
            }

    return candidates


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def fuse_rrf(
    candidates: dict[str, dict[str, Any]],
    rrf_k: int,
    top_k: int = 20,
) -> list[FusedSearchResult]:
    """Reciprocal Rank Fusion.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        candidates: Output of ``build_candidates``.
        rrf_k: RRF smoothing constant (k=60 is standard).
        top_k: Max results to return.

    Returns:
        Ranked fused results.
    """
    scored: list[tuple[float, str, dict]] = []
    for item_id, info in candidates.items():
        score = 0.0
        if info["bm25_rank"] is not None:
            score += 1.0 / (rrf_k + info["bm25_rank"])
        if info["semantic_rank"] is not None:
            score += 1.0 / (rrf_k + info["semantic_rank"])
        scored.append((score, item_id, info))

    # Score descending, item_id ascending for ties
    scored.sort(key=lambda x: (-x[0], x[1]))

    results: list[FusedSearchResult] = []
    for rank, (score, item_id, info) in enumerate(scored[:top_k], start=1):
        results.append(FusedSearchResult(
            item_id=item_id,
            rank=rank,
            fusion_score=score,
            bm25_rank=info["bm25_rank"],
            semantic_rank=info["semantic_rank"],
            bm25_score=info["bm25_score"],
            semantic_score=info["semantic_score"],
            sources=info["sources"],
        ))
    return results


# ---------------------------------------------------------------------------
# Linear fusion (min-max normalised)
# ---------------------------------------------------------------------------

def _normalize_scores(
    scores: list[float],
) -> list[float]:
    """Min-max normalise a list of scores to [0, 1].

    If all scores are identical (or only one score), returns all 1.0.
    Missing/invalid scores are handled by the caller returning 0.0.
    """
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    denom = mx - mn
    return [(s - mn) / denom for s in scores]


def fuse_linear(
    candidates: dict[str, dict[str, Any]],
    bm25_weight: float,
    semantic_weight: float,
    top_k: int = 20,
) -> list[FusedSearchResult]:
    """Weighted linear score fusion with per-query min-max normalisation.

    linear_score(d) = bm25_weight × norm_bm25(d) + semantic_weight × norm_sem(d)

    Missing in a source → that source's contribution is 0.
    """
    # Normalise weights to sum to 1
    total_w = bm25_weight + semantic_weight
    w_bm25 = bm25_weight / total_w if total_w > 0 else 0.5
    w_sem = semantic_weight / total_w if total_w > 0 else 0.5

    # Collect BM25 scores and semantic scores for normalisation
    items = list(candidates.items())
    bm25_scores_raw = [info["bm25_score"] for _, info in items
                       if info["bm25_score"] is not None and math.isfinite(info["bm25_score"])]
    sem_scores_raw = [info["semantic_score"] for _, info in items
                      if info["semantic_score"] is not None and math.isfinite(info["semantic_score"])]

    # Normalise per source
    bm25_norm = _normalize_scores(bm25_scores_raw)
    sem_norm = _normalize_scores(sem_scores_raw)

    # Map back to item_ids
    bm25_norm_map: dict[str, float] = {}
    idx = 0
    for _, info in items:
        if info["bm25_score"] is not None and math.isfinite(info["bm25_score"]):
            bm25_norm_map[info.get("_id_placeholder", "") or ""] = bm25_norm[idx]
            idx += 1
    # Rebuild map using item_id from items list
    bm25_norm_map = {}
    idx = 0
    for item_id, info in items:
        if info["bm25_score"] is not None and math.isfinite(info["bm25_score"]):
            bm25_norm_map[item_id] = bm25_norm[idx]
            idx += 1

    sem_norm_map: dict[str, float] = {}
    idx = 0
    for item_id, info in items:
        if info["semantic_score"] is not None and math.isfinite(info["semantic_score"]):
            sem_norm_map[item_id] = sem_norm[idx]
            idx += 1

    # Score and sort
    scored: list[tuple[float, str, dict, float, float]] = []
    for item_id, info in items:
        bm25_n = bm25_norm_map.get(item_id, 0.0)
        sem_n = sem_norm_map.get(item_id, 0.0)
        linear = w_bm25 * bm25_n + w_sem * sem_n
        scored.append((linear, item_id, info, bm25_n, sem_n))

    scored.sort(key=lambda x: (-x[0], x[1]))

    results: list[FusedSearchResult] = []
    for rank, (score, item_id, info, bm25_n, sem_n) in enumerate(scored[:top_k], start=1):
        results.append(FusedSearchResult(
            item_id=item_id,
            rank=rank,
            fusion_score=score,
            bm25_rank=info["bm25_rank"],
            semantic_rank=info["semantic_rank"],
            bm25_score=info["bm25_score"],
            semantic_score=info["semantic_score"],
            bm25_normalized_score=bm25_n if info["bm25_rank"] is not None else None,
            semantic_normalized_score=sem_n if info["semantic_rank"] is not None else None,
            sources=info["sources"],
        ))
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class FusionDiagnostics:
    """Per-query diagnostic counters for fusion analysis."""

    bm25_candidate_count: int = 0
    semantic_candidate_count: int = 0
    union_count: int = 0
    intersection_count: int = 0

    bm25_empty: bool = False
    semantic_empty: bool = False


def compute_diagnostics(
    all_candidates: dict[str, dict[str, dict[str, Any]]],
    all_rrf_results: dict[str, list[FusedSearchResult]],
    all_linear_results: dict[str, list[FusedSearchResult]],
    all_bm25: dict[str, list[BM25Result]],
    all_semantic: dict[str, list[SemanticSearchResult]],
    bm25_query_metrics: dict[str, float],
    semantic_query_metrics: dict[str, float],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compute aggregate fusion diagnostics across all queries."""
    query_ids = sorted(all_candidates.keys())

    bm25_counts: list[int] = []
    sem_counts: list[int] = []
    union_counts: list[int] = []
    inter_counts: list[int] = []
    jaccards: list[float] = []

    bm25_empty = 0
    sem_empty = 0
    both_empty = 0
    rescued_by_sem = 0
    rescued_by_bm25 = 0

    top20_from_both = 0
    top20_bm25_only = 0
    top20_sem_only = 0

    rrf_better_bm25 = 0
    rrf_equal_bm25 = 0
    rrf_worse_bm25 = 0
    rrf_better_sem = 0
    rrf_equal_sem = 0
    rrf_worse_sem = 0

    lin_better_bm25 = 0
    lin_equal_bm25 = 0
    lin_worse_bm25 = 0
    lin_better_sem = 0
    lin_equal_sem = 0
    lin_worse_sem = 0

    for qid in query_ids:
        cand = all_candidates.get(qid, {})
        bm25_c = sum(1 for info in cand.values() if info["bm25_rank"] is not None)
        sem_c = sum(1 for info in cand.values() if info["semantic_rank"] is not None)
        union_c = len(cand)
        inter_c = sum(1 for info in cand.values() if info["bm25_rank"] is not None and info["semantic_rank"] is not None)

        bm25_counts.append(bm25_c)
        sem_counts.append(sem_c)
        union_counts.append(union_c)
        inter_counts.append(inter_c)
        jac = inter_c / union_c if union_c > 0 else 0.0
        jaccards.append(jac)

        bm25_empty_q = bm25_c == 0
        sem_empty_q = sem_c == 0
        if bm25_empty_q:
            bm25_empty += 1
        if sem_empty_q:
            sem_empty += 1
        if bm25_empty_q and sem_empty_q:
            both_empty += 1
        if bm25_empty_q and not sem_empty_q:
            rescued_by_sem += 1
        if not bm25_empty_q and sem_empty_q:
            rescued_by_bm25 += 1

        # Top-20 source analysis from RRF results
        rrf_res = all_rrf_results.get(qid, [])
        for r in rrf_res[:20]:
            if len(r.sources) == 2:
                top20_from_both += 1
            elif r.sources == ("bm25",):
                top20_bm25_only += 1
            elif r.sources == ("semantic",):
                top20_sem_only += 1

        # NDCG@10 comparisons
        bm25_ndcg = bm25_query_metrics.get(qid, 0.0)
        sem_ndcg = semantic_query_metrics.get(qid, 0.0)

        # For counts, we need to compute per-query NDCG from the fused results
        # We'll compute these during evaluation in the CLI; here we just set up structure

    return {
        "average_bm25_candidates": round(sum(bm25_counts) / len(bm25_counts), 2) if bm25_counts else 0,
        "average_semantic_candidates": round(sum(sem_counts) / len(sem_counts), 2) if sem_counts else 0,
        "average_union_candidates": round(sum(union_counts) / len(union_counts), 2) if union_counts else 0,
        "average_intersection_candidates": round(sum(inter_counts) / len(inter_counts), 2) if inter_counts else 0,
        "average_candidate_jaccard": round(sum(jaccards) / len(jaccards), 4) if jaccards else 0,
        "top20_from_both_count": top20_from_both,
        "top20_bm25_only_count": top20_bm25_only,
        "top20_semantic_only_count": top20_sem_only,
        "queries_bm25_empty": bm25_empty,
        "queries_semantic_empty": sem_empty,
        "queries_both_empty": both_empty,
        "queries_rescued_by_semantic": rescued_by_sem,
        "queries_rescued_by_bm25": rescued_by_bm25,
    }


def _compute_per_query_ndcg(
    results: list[FusedSearchResult],
    qrels_for_query: dict[str, int],
    k: int,
) -> float:
    """Compute NDCG@k for a fused result list."""
    rels = [qrels_for_query.get(r.item_id, 0) for r in results[:k]]
    dcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def add_comparison_counts(
    diagnostics: dict[str, Any],
    all_rrf: dict[str, list[FusedSearchResult]],
    all_linear: dict[str, list[FusedSearchResult]],
    all_bm25: dict[str, list[BM25Result]],
    all_semantic: dict[str, list[SemanticSearchResult]],
    qrels: dict[str, dict[str, int]],
    metric_name: str = "ndcg",
) -> dict[str, Any]:
    """Add per-query NDCG comparison counts to diagnostics."""
    from psr_srs_mvp.evaluation.metrics import evaluate_query, MetricResult
    # Re-compute using shared evaluation
    from psr_srs_mvp.evaluation.metrics import _dcg, _ideal_rels

    query_ids = sorted(all_rrf.keys())
    diag = dict(diagnostics)

    counts = {
        "rrf_queries_better_than_bm25_ndcg10": 0,
        "rrf_queries_equal_to_bm25_ndcg10": 0,
        "rrf_queries_worse_than_bm25_ndcg10": 0,
        "rrf_queries_better_than_semantic_ndcg10": 0,
        "rrf_queries_equal_to_semantic_ndcg10": 0,
        "rrf_queries_worse_than_semantic_ndcg10": 0,
        "linear_queries_better_than_bm25_ndcg10": 0,
        "linear_queries_equal_to_bm25_ndcg10": 0,
        "linear_queries_worse_than_bm25_ndcg10": 0,
        "linear_queries_better_than_semantic_ndcg10": 0,
        "linear_queries_equal_to_semantic_ndcg10": 0,
        "linear_queries_worse_than_semantic_ndcg10": 0,
    }

    for qid in query_ids:
        qrels_q = qrels.get(qid, {})

        bm25_ndcg = _compute_per_query_ndcg_q(
            [BM25Result(score=r.score, item_id=r.item_id, rank=r.rank)
             for r in all_bm25.get(qid, [])], qrels_q, 10)
        sem_ndcg = _compute_per_query_ndcg_q(
            [BM25Result(score=1.0, item_id=r.item_id, rank=r.rank)
             for r in all_semantic.get(qid, [])], qrels_q, 10)
        rrf_ndcg = _compute_per_query_ndcg_q(
            [BM25Result(score=1.0, item_id=r.item_id, rank=r.rank)
             for r in all_rrf.get(qid, [])], qrels_q, 10)
        lin_ndcg = _compute_per_query_ndcg_q(
            [BM25Result(score=1.0, item_id=r.item_id, rank=r.rank)
             for r in all_linear.get(qid, [])], qrels_q, 10)

        for prefix, ndcg in [("rrf", rrf_ndcg), ("linear", lin_ndcg)]:
            for baseline, base_ndcg in [("bm25", bm25_ndcg), ("semantic", sem_ndcg)]:
                key = f"{prefix}_queries_"
                if ndcg > base_ndcg:
                    key += f"better_than_{baseline}_ndcg10"
                elif abs(ndcg - base_ndcg) < 1e-9:
                    key += f"equal_to_{baseline}_ndcg10"
                else:
                    key += f"worse_than_{baseline}_ndcg10"
                counts[key] = counts.get(key, 0) + 1

    diag.update(counts)
    return diag


def _compute_per_query_ndcg_q(
    results: list[BM25Result],
    qrels_q: dict[str, int],
    k: int,
) -> float:
    rels = [qrels_q.get(r.item_id, 0) for r in results[:k]]
    if not rels:
        return 0.0
    dcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
