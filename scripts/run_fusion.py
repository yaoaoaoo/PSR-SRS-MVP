#!/usr/bin/env python3
"""Hybrid BM25 + LSA fusion retrieval for PSR-SRS MVP.

Usage::

    .venv/Scripts/python.exe scripts/run_fusion.py \\
        --items data/sample/items.csv \\
        --queries data/sample/queries.csv \\
        --qrels data/sample/qrels.csv \\
        --bm25-config configs/bm25.json \\
        --semantic-config configs/semantic.json \\
        --fusion-config configs/fusion.json \\
        --bm25-metrics outputs/bm25/metrics.json \\
        --semantic-metrics outputs/semantic/metrics.json \\
        --output outputs/hybrid \\
        --comparison-output outputs/comparison/retrieval_methods.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.evaluation import evaluate_all, macro_average, evaluate_query
from psr_srs_mvp.retrieval import (
    BM25Config, BM25Index, Document,
    SemanticConfig, SemanticIndex,
    build_item_text, tokenize,
    load_items, load_qrels, load_queries,
)
from psr_srs_mvp.retrieval.fusion import (
    FusionConfig,
    FusedSearchResult,
    build_candidates,
    fuse_rrf,
    fuse_linear,
    compute_diagnostics,
    add_comparison_counts,
)


def _bm25_result_to_dict(r, qid, qtext, qrels_q):
    return {
        "item_id": r.item_id, "score": r.score, "rank": r.rank,
    }


def _to_eval_result(r: FusedSearchResult):
    """Convert FusedSearchResult to a SearchResult-like for evaluation."""
    from psr_srs_mvp.retrieval.bm25 import SearchResult
    return SearchResult(score=r.fusion_score, item_id=r.item_id, rank=r.rank)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BM25 + LSA fusion retrieval")
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--bm25-config", required=True, type=Path)
    parser.add_argument("--semantic-config", required=True, type=Path)
    parser.add_argument("--fusion-config", required=True, type=Path)
    parser.add_argument("--bm25-metrics", required=True, type=Path)
    parser.add_argument("--semantic-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-output", required=True, type=Path)
    args = parser.parse_args()

    # 1. Load configs
    print("[1/9] Loading configs …")
    bm25_cfg = BM25Config.from_json(args.bm25_config)
    sem_cfg = SemanticConfig.from_json(args.semantic_config)
    fus_cfg = FusionConfig.from_json(args.fusion_config)
    print(f"  candidate_k={fus_cfg.candidate_k}, rrf_k={fus_cfg.rrf_k}, "
          f"weights=({fus_cfg.bm25_weight},{fus_cfg.semantic_weight})")

    # 2. Load items
    print(f"[2/9] Loading items: {args.items}")
    item_rows = load_items(args.items)
    item_ids = [r["item_id"] for r in item_rows]

    # Build BM25 texts (weighted) and LSA texts (raw)
    bm25_texts = [build_item_text(r["title"], r["description"], r["category"],
                                   r["subcategory"], r["brand"],
                                   weights=bm25_cfg.field_weights) for r in item_rows]
    sem_texts = [" ".join([r["title"], r["description"], r["category"],
                           r["subcategory"], r["brand"]]) for r in item_rows]

    # 3. Build indices
    print("[3/9] Building BM25 index …")
    bm25_docs = [Document(item_id=item_ids[i],
                          tokens=tokenize(bm25_texts[i], remove_stopwords=bm25_cfg.use_stopwords),
                          length=len(tokenize(bm25_texts[i], remove_stopwords=bm25_cfg.use_stopwords)))
                 for i in range(len(item_rows))]
    bm25_idx = BM25Index.build(bm25_docs, k1=bm25_cfg.k1, b=bm25_cfg.b)
    print(f"  BM25: {bm25_idx.document_count} docs, {bm25_idx.vocabulary_size} terms")

    print("[4/9] Building LSA index …")
    sem_idx = SemanticIndex.build(sem_texts, item_ids, sem_cfg)
    print(f"  LSA: {sem_idx.document_count} docs, {sem_idx.vector_dim}-d")

    # 5. Load queries
    print(f"[5/9] Loading queries: {args.queries}")
    query_rows = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    print(f"  {len(query_rows)} queries")

    # 6. Search + fuse
    print(f"[6/9] Searching & fusing (candidate_k={fus_cfg.candidate_k}) …")
    all_candidates: dict[str, dict] = {}
    all_rrf: dict[str, list[FusedSearchResult]] = {}
    all_linear: dict[str, list[FusedSearchResult]] = {}
    all_bm25: dict[str, list] = {}
    all_semantic: dict[str, list] = {}

    for q in query_rows:
        qid = q["query_id"]
        qtext = q["query_text"]

        bm25_res = bm25_idx.search(qtext, top_k=fus_cfg.candidate_k)
        sem_res = sem_idx.search(qtext, top_k=fus_cfg.candidate_k)

        all_bm25[qid] = bm25_res
        all_semantic[qid] = sem_res

        cand = build_candidates(bm25_res, sem_res)
        all_candidates[qid] = cand

        rrf_res = fuse_rrf(cand, fus_cfg.rrf_k, top_k=fus_cfg.max_k)
        lin_res = fuse_linear(cand, fus_cfg.bm25_weight, fus_cfg.semantic_weight,
                              top_k=fus_cfg.max_k)
        all_rrf[qid] = rrf_res
        all_linear[qid] = lin_res

    # 7. Evaluate
    print("[7/9] Evaluating …")
    ks = fus_cfg.top_k_values
    thr = fus_cfg.relevance_threshold

    rrf_eval = {qid: [_to_eval_result(r) for r in results] for qid, results in all_rrf.items()}
    lin_eval = {qid: [_to_eval_result(r) for r in results] for qid, results in all_linear.items()}
    bm25_eval = {qid: results for qid, results in all_bm25.items()}
    sem_eval = {qid: [type('SR', (), {'score': r.score, 'item_id': r.item_id, 'rank': r.rank})()
                      for r in results] for qid, results in all_semantic.items()}

    rrf_metrics = evaluate_all(rrf_eval, query_rows, qrels, ks=ks, relevance_threshold=thr)
    lin_metrics = evaluate_all(lin_eval, query_rows, qrels, ks=ks, relevance_threshold=thr)
    bm25_metrics_full = evaluate_all(bm25_eval, query_rows, qrels, ks=ks, relevance_threshold=thr)
    sem_metrics_full = evaluate_all(sem_eval, query_rows, qrels, ks=ks, relevance_threshold=thr)

    rrf_avg = macro_average(rrf_metrics)
    lin_avg = macro_average(lin_metrics)
    bm25_avg = macro_average(bm25_metrics_full)
    sem_avg = macro_average(sem_metrics_full)

    # 8. Diagnostics
    print("[8/9] Computing diagnostics …")
    bm25_qm = {m.query_id: m.ndcg.get(10, 0.0) for m in bm25_metrics_full}
    sem_qm = {m.query_id: m.ndcg.get(10, 0.0) for m in sem_metrics_full}

    diag_base = compute_diagnostics(
        all_candidates, all_rrf, all_linear,
        all_bm25, all_semantic,
        bm25_qm, sem_qm, qrels,
    )
    # Convert semantic results for comparison counts
    all_sem_for_comp = {}
    for qid, results in all_semantic.items():
        all_sem_for_comp[qid] = [type('SR', (), {'score': r.score, 'item_id': r.item_id, 'rank': r.rank})()
                                  for r in results]
    diag = add_comparison_counts(
        diag_base, all_rrf, all_linear,
        all_bm25, all_sem_for_comp, qrels,
    )

    # 9. Export
    print(f"[9/9] Exporting to: {args.output}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Coverage stats
    bm25_cov = sum(1 for q in query_rows if all_bm25.get(q["query_id"])) / len(query_rows)
    sem_cov = sum(1 for q in query_rows if all_semantic.get(q["query_id"])) / len(query_rows)
    rrf_cov = sum(1 for q in query_rows if all_rrf.get(q["query_id"])) / len(query_rows)
    lin_cov = sum(1 for q in query_rows if all_linear.get(q["query_id"])) / len(query_rows)

    def _write_hybrid_output(subdir: str, method: str, metrics_obj, avg, candidates_map):
        d = out / subdir
        d.mkdir(parents=True, exist_ok=True)

        # metrics.json
        mj: dict = {
            "algorithm": method,
            "candidate_k": fus_cfg.candidate_k,
            "document_count": len(item_rows),
            "query_count": len(metrics_obj),
            "query_coverage": round(rrf_cov if "RRF" in method else lin_cov, 6),
            "no_result_query_count": sum(1 for q in query_rows
                                         if not candidates_map.get(q["query_id"])),
        }
        if "RRF" in method:
            mj["rrf_k"] = fus_cfg.rrf_k
        else:
            mj["bm25_weight"] = fus_cfg.bm25_weight
            mj["semantic_weight"] = fus_cfg.semantic_weight
            mj["score_normalization"] = fus_cfg.score_normalization

        diag_copy = dict(diag)
        mj.update(diag_copy)

        for mn, kvs in avg.items():
            for k, v in kvs.items():
                mj[f"{mn}_at_{k}"] = round(v, 6)

        (d / "metrics.json").write_text(json.dumps(mj, indent=2, ensure_ascii=False), encoding="utf-8")

        # query_metrics.csv
        qm_fields = ["query_id", "query_text",
                     "bm25_result_count", "semantic_result_count",
                     "candidate_union_count", "candidate_intersection_count",
                     "result_count"]
        for k in ks:
            for mn in ("precision", "recall", "mrr", "ndcg"):
                qm_fields.append(f"{mn}_at_{k}")

        with (d / "query_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=qm_fields, extrasaction="ignore")
            w.writeheader()
            for m_obj in sorted(metrics_obj, key=lambda m: m.query_id):
                row = m_obj.to_flat_dict()
                qid = m_obj.query_id
                cand = all_candidates.get(qid, {})
                bm25_c = sum(1 for info in cand.values() if info["bm25_rank"] is not None)
                sem_c = sum(1 for info in cand.values() if info["semantic_rank"] is not None)
                union_c = len(cand)
                inter_c = sum(1 for info in cand.values()
                             if info["bm25_rank"] is not None and info["semantic_rank"] is not None)
                row["bm25_result_count"] = str(bm25_c)
                row["semantic_result_count"] = str(sem_c)
                row["candidate_union_count"] = str(union_c)
                row["candidate_intersection_count"] = str(inter_c)
                row["result_count"] = str(len(candidates_map.get(qid, [])))
                w.writerow(row)

        # search_results.csv
        sr_fields = ["query_id", "query_text", "rank", "item_id",
                     "fusion_method", "fusion_score",
                     "bm25_rank", "semantic_rank",
                     "bm25_score", "semantic_score",
                     "bm25_normalized_score", "semantic_normalized_score",
                     "sources", "relevance_grade"]
        with (d / "search_results.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sr_fields, extrasaction="ignore")
            w.writeheader()
            for q in query_rows:
                qid = q["query_id"]
                qrels_q = qrels.get(qid, {})
                results = candidates_map.get(qid, [])
                for r in results[:fus_cfg.max_k]:
                    rd = r.to_dict()
                    rd.update({
                        "query_id": qid,
                        "query_text": q["query_text"],
                        "fusion_method": method,
                        "relevance_grade": str(qrels_q.get(r.item_id, 0)),
                    })
                    w.writerow(rd)

    _write_hybrid_output("rrf", "RRF (k=60)", rrf_metrics, rrf_avg, all_rrf)
    _write_hybrid_output("linear", "Linear (0.5/0.5, min-max)", lin_metrics, lin_avg, all_linear)

    # diagnostics.json
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")

    # comparison/retrieval_methods.json
    comp_path = Path(args.comparison_output)
    comp_path.parent.mkdir(parents=True, exist_ok=True)

    def _metric_dict(avg):
        d = {}
        for mn, kvs in avg.items():
            for k, v in kvs.items():
                d[f"{mn}_at_{k}"] = round(v, 6)
        return d

    bm25_md = _metric_dict(bm25_avg)
    sem_md = _metric_dict(sem_avg)
    rrf_md = _metric_dict(rrf_avg)
    lin_md = _metric_dict(lin_avg)

    coverage_info = {
        "bm25": round(bm25_cov, 6), "semantic_lsa": round(sem_cov, 6),
        "hybrid_rrf": round(rrf_cov, 6), "hybrid_linear": round(lin_cov, 6),
    }

    comparison = {
        "bm25": {"query_coverage": coverage_info["bm25"], **bm25_md},
        "semantic_lsa": {"query_coverage": coverage_info["semantic_lsa"], **sem_md},
        "hybrid_rrf": {"query_coverage": coverage_info["hybrid_rrf"], **rrf_md},
        "hybrid_linear": {"query_coverage": coverage_info["hybrid_linear"], **lin_md},
    }
    for prefix, md in [("rrf", rrf_md), ("linear", lin_md)]:
        for baseline, base_md in [("bm25", bm25_md), ("semantic", sem_md)]:
            key = f"{prefix}_minus_{baseline}"
            comparison[key] = {}
            for mk in md:
                comparison[key][mk] = round(md[mk] - base_md.get(mk, 0), 6)

    comp_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print()
    print(f"  coverage: BM25={bm25_cov:.1%} LSA={sem_cov:.1%} RRF={rrf_cov:.1%} Linear={lin_cov:.1%}")
    print(f"  NDCG@10:  BM25={bm25_avg['ndcg'].get(10,0):.4f} LSA={sem_avg['ndcg'].get(10,0):.4f} "
          f"RRF={rrf_avg['ndcg'].get(10,0):.4f} Linear={lin_avg['ndcg'].get(10,0):.4f}")
    print(f"  Recall@20: BM25={bm25_avg['recall'].get(20,0):.4f} LSA={sem_avg['recall'].get(20,0):.4f} "
          f"RRF={rrf_avg['recall'].get(20,0):.4f} Linear={lin_avg['recall'].get(20,0):.4f}")
    print(f"  avg candidate overlap (Jaccard): {diag.get('average_candidate_jaccard', 'N/A')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
