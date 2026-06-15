"""Personalization evaluation — behavior metrics + qrels protection + diagnostics."""

from __future__ import annotations

import math
from typing import Any, Sequence

from psr_srs_mvp.personalization.reranker import RankedItem


def behavior_grade(item_id: str, grades: dict[str, int]) -> int:
    return grades.get(item_id, 0)


def _dcg(grades: list[int]) -> float:
    return sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(grades))


def _ideal_dcg(grades: Sequence[int], k: int) -> float:
    return _dcg(sorted(grades, reverse=True)[:k])


def compute_behavior_metrics(
    results: list[RankedItem],
    behavior_grades: dict[str, int],
    positive_items: set[str],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Compute behavior-based HitRate, MRR, NDCG, Positive Recall per K."""
    metrics: dict[str, Any] = {}

    for k in ks:
        top = results[:k]

        # HitRate
        hit = any(r.item_id in positive_items for r in top)
        metrics[f"hit_rate_at_{k}"] = 1.0 if hit else 0.0

        # MRR
        mrr = 0.0
        for i, r in enumerate(top, start=1):
            if r.item_id in positive_items:
                mrr = 1.0 / i
                break
        metrics[f"mrr_at_{k}"] = mrr

        # NDCG
        grades = [behavior_grade(r.item_id, behavior_grades) for r in top]
        dcg = _dcg(grades)
        idcg = _ideal_dcg(list(behavior_grades.values()), k)
        metrics[f"ndcg_at_{k}"] = dcg / idcg if idcg > 0 else 0.0

        # Positive Recall
        pos_in_top = sum(1 for r in top if r.item_id in positive_items)
        metrics[f"positive_recall_at_{k}"] = pos_in_top / len(positive_items) if positive_items else 0.0

    return metrics


def compute_qrels_metrics(
    results: list[RankedItem],
    qrels: dict[str, int],
    ks: Sequence[int],
    relevance_threshold: int = 1,
) -> dict[str, Any]:
    """Compute standard qrels-based Precision, Recall, MRR, NDCG."""
    metrics: dict[str, Any] = {}
    total_relevant = sum(1 for g in qrels.values() if g >= relevance_threshold)

    for k in ks:
        top = results[:k]
        rel = sum(1 for r in top if qrels.get(r.item_id, 0) >= relevance_threshold)

        metrics[f"precision_at_{k}"] = rel / k if k > 0 else 0.0
        metrics[f"recall_at_{k}"] = rel / total_relevant if total_relevant > 0 else 0.0

        mrr = 0.0
        for i, r in enumerate(top, start=1):
            if qrels.get(r.item_id, 0) >= relevance_threshold:
                mrr = 1.0 / i
                break
        metrics[f"mrr_at_{k}"] = mrr

        grades = [qrels.get(r.item_id, 0) for r in top]
        dcg = _dcg(grades)
        idcg = _dcg(sorted(qrels.values(), reverse=True)[:k])
        metrics[f"ndcg_at_{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def macro_average_dict(metrics_list: list[dict], keys: list[str]) -> dict[str, float]:
    """Macro-average a list of per-request metric dicts."""
    result: dict[str, float] = {}
    for key in keys:
        vals = [m.get(key, 0.0) for m in metrics_list if key in m]
        result[key] = sum(vals) / len(vals) if vals else 0.0
    return result


def compute_candidate_coverage(
    test_requests: dict[str, dict],
    candidates_by_qid: dict[str, list[dict]],
    eligible_rids: list[str],
) -> dict[str, Any]:
    """Compute candidate positive coverage at request and item level."""
    covered_req = 0
    uncovered_req = 0
    total_pos_items = 0
    covered_pos_items = 0

    warm_covered = 0
    warm_total = 0
    cold_covered = 0
    cold_total = 0

    for rid in eligible_rids:
        info = test_requests[rid]
        qid = info["query_id"]
        candidates = candidates_by_qid.get(qid, [])
        candidate_ids = {c["item_id"] for c in candidates}
        pos_items = {iid for iid, g in info["items"].items() if g > 0}
        total_pos_items += len(pos_items)
        covered_pos = pos_items & candidate_ids
        covered_pos_items += len(covered_pos)

        if covered_pos:
            covered_req += 1
        else:
            uncovered_req += 1

        profile_status = info.get("profile_status", "unknown")
        if profile_status == "warm":
            warm_total += 1
            if covered_pos:
                warm_covered += 1
        else:
            cold_total += 1
            if covered_pos:
                cold_covered += 1

    return {
        "eligible_positive_request_count": len(eligible_rids),
        "covered_positive_request_count": covered_req,
        "uncovered_positive_request_count": uncovered_req,
        "request_level_candidate_positive_coverage": round(covered_req / len(eligible_rids), 6) if eligible_rids else 0,
        "total_positive_item_count": total_pos_items,
        "covered_positive_item_count": covered_pos_items,
        "uncovered_positive_item_count": total_pos_items - covered_pos_items,
        "item_level_candidate_positive_recall": round(covered_pos_items / total_pos_items, 6) if total_pos_items else 0,
        "warm_request_coverage": round(warm_covered / warm_total, 6) if warm_total else 0,
        "fallback_request_coverage": round(cold_covered / cold_total, 6) if cold_total else 0,
    }
