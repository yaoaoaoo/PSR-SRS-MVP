#!/usr/bin/env python3
"""LSA semantic retrieval baseline for PSR-SRS MVP.

Usage::

    .venv/Scripts/python.exe scripts/run_semantic.py \\
        --items data/sample/items.csv \\
        --queries data/sample/queries.csv \\
        --qrels data/sample/qrels.csv \\
        --config configs/semantic.json \\
        --bm25-metrics outputs/bm25/metrics.json \\
        --output outputs/semantic \\
        --comparison-output outputs/comparison/bm25_vs_semantic.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.evaluation import evaluate_all, macro_average, MetricResult
from psr_srs_mvp.retrieval import (
    SemanticConfig,
    SemanticIndex,
    SemanticSearchResult,
    build_item_text,
    is_zero_vector,
    load_items,
    load_qrels,
    load_queries,
)


def _build_item_text_raw(row: dict[str, str]) -> str:
    """Build a raw item text without field repetition (unlike BM25 weighting)."""
    return " ".join([
        row["title"], row["description"], row["category"],
        row["subcategory"], row["brand"],
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="LSA semantic retrieval baseline")
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bm25-metrics", required=True, type=Path,
                        help="Path to BM25 metrics.json for comparison")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-output", required=True, type=Path)
    args = parser.parse_args()

    # 1. Load config
    print(f"[1/8] Loading config: {args.config}")
    cfg = SemanticConfig.from_json(args.config)

    # 2. Load items
    print(f"[2/8] Loading items: {args.items}")
    item_rows = load_items(args.items)
    print(f"  {len(item_rows)} items loaded")
    item_texts = [_build_item_text_raw(r) for r in item_rows]
    item_ids = [r["item_id"] for r in item_rows]

    # 3. Build SemanticIndex
    print("[3/8] Building LSA index (inductive) …")
    index = SemanticIndex.build(item_texts, item_ids, cfg)
    vec = index.vectorizer
    print(f"  word features:    {vec.word_feature_count}")
    print(f"  char features:    {vec.char_feature_count}")
    print(f"  combined:         {vec.combined_feature_count}")
    print(f"  svd: requested={cfg.svd_components}, actual={vec.svd_components_actual}")
    print(f"  explained var:    {vec.explained_variance_ratio_sum:.4f}")

    # 4. Load queries
    print(f"[4/8] Loading queries: {args.queries}")
    query_rows = load_queries(args.queries)
    print(f"  {len(query_rows)} queries loaded")

    # 5. Search all queries
    print(f"[5/8] Searching (max K={cfg.max_k}) …")
    all_results: dict[str, list[SemanticSearchResult]] = {}
    zero_vec_count = 0
    empty_count = 0
    nonzero_count = 0

    for q in query_rows:
        qid = q["query_id"]
        qtext = q["query_text"]
        results = index.search(qtext, top_k=cfg.max_k)
        all_results[qid] = results

        if results:
            nonzero_count += 1
        else:
            empty_count += 1
            # Check if zero-vector caused the empty result
            if vec is not None:
                qv = vec.transform([qtext])[0]
                if is_zero_vector(qv):
                    zero_vec_count += 1

    query_coverage = nonzero_count / len(query_rows) if query_rows else 0
    print(f"  nonzero queries:  {nonzero_count}")
    print(f"  zero-vector:      {zero_vec_count}")
    print(f"  no-result:        {empty_count}")
    print(f"  coverage:         {query_coverage:.1%}")

    # 6. Evaluate
    print("[6/8] Evaluating …")
    qrels = load_qrels(args.qrels)
    metrics = evaluate_all(
        {qid: [SemanticSearchResult(score=r.score, item_id=r.item_id, rank=r.rank)
               for r in results]
         for qid, results in all_results.items()},
        query_rows, qrels,
        ks=cfg.top_k_values, relevance_threshold=cfg.relevance_threshold,
    )
    averages = macro_average(metrics)

    # 7. Export
    print(f"[7/8] Exporting to: {args.output}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    metrics_json: dict = {
        "algorithm": "LSA (TF-IDF + TruncatedSVD)",
        "evaluation_setting": "inductive",
        "document_count": index.document_count,
        "query_count": len(metrics),
        "word_ngram_range": cfg.word_ngram_range,
        "char_ngram_range": cfg.char_ngram_range,
        "word_weight": cfg.word_weight,
        "char_weight": cfg.char_weight,
        "word_feature_count": vec.word_feature_count if vec else 0,
        "char_feature_count": vec.char_feature_count if vec else 0,
        "combined_feature_count": vec.combined_feature_count if vec else 0,
        "svd_components_requested": cfg.svd_components,
        "svd_components_actual": vec.svd_components_actual if vec else 0,
        "explained_variance_ratio_sum": round(vec.explained_variance_ratio_sum if vec else 0, 6),
        "random_state": cfg.random_state,
        "zero_vector_query_count": zero_vec_count,
        "nonzero_query_count": nonzero_count,
        "no_result_query_count": empty_count,
        "query_coverage": round(query_coverage, 6),
        "relevance_threshold": cfg.relevance_threshold,
    }
    for metric_name, kvs in averages.items():
        for k, v in kvs.items():
            metrics_json[f"{metric_name}_at_{k}"] = round(v, 6)

    (out / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [OK] metrics.json")

    # --- query_metrics.csv ---
    qm_fields = ["query_id", "query_text", "is_zero_vector", "result_count"]
    for k in cfg.top_k_values:
        for m in ("precision", "recall", "mrr", "ndcg"):
            qm_fields.append(f"{m}_at_{k}")

    with (out / "query_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=qm_fields, extrasaction="ignore")
        writer.writeheader()
        for m_obj in sorted(metrics, key=lambda m: m.query_id):
            row = m_obj.to_flat_dict()
            qid = m_obj.query_id
            results = all_results.get(qid, [])
            qtext = ""
            for qr in query_rows:
                if qr["query_id"] == qid:
                    qtext = qr["query_text"]
                    break
            qv = vec.transform([qtext])[0] if vec else None
            row["is_zero_vector"] = str(bool(qv is not None and is_zero_vector(qv))).lower()
            row["result_count"] = str(len(results))
            writer.writerow(row)
    print(f"  [OK] query_metrics.csv ({len(metrics)} rows)")

    # --- search_results.csv ---
    sr_fields = ["query_id", "query_text", "rank", "item_id", "semantic_score", "relevance_grade"]
    with (out / "search_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sr_fields, extrasaction="ignore")
        writer.writeheader()
        for q in query_rows:
            qid = q["query_id"]
            qrels_for_q = qrels.get(qid, {})
            results = all_results.get(qid, [])
            for r in results[:cfg.max_k]:
                writer.writerow({
                    "query_id": qid,
                    "query_text": q["query_text"],
                    "rank": str(r.rank),
                    "item_id": r.item_id,
                    "semantic_score": f"{r.score:.6f}",
                    "relevance_grade": str(qrels_for_q.get(r.item_id, 0)),
                })
    with (out / "search_results.csv").open("r", encoding="utf-8") as f:
        sr_count = sum(1 for _ in f)
    print(f"  [OK] search_results.csv ({sr_count - 1} rows)")

    # 8. Comparison with BM25
    print(f"[8/8] Comparison → {args.comparison_output}")
    bm25 = json.loads(Path(args.bm25_metrics).read_text(encoding="utf-8")) if Path(args.bm25_metrics).exists() else {}

    comparison = {"bm25": {}, "semantic": {}, "semantic_minus_bm25": {}}
    for key in ("query_coverage",):
        comparison["bm25"][key] = bm25.get(key)
        comparison["semantic"][key] = metrics_json.get(key)
        comparison["semantic_minus_bm25"][key] = (
            round(metrics_json.get(key, 0) - bm25.get(key, 0), 6)
            if bm25.get(key) is not None else None
        )
    for key in ("no_result_query_count",):
        comparison["bm25"][key] = bm25.get(key)
        comparison["semantic"][key] = metrics_json.get(key)
        comparison["semantic_minus_bm25"][key] = (
            metrics_json.get(key, 0) - bm25.get(key, 0)
            if bm25.get(key) is not None else None
        )
    for k in cfg.top_k_values:
        for m in ("precision", "recall", "mrr", "ndcg"):
            key = f"{m}_at_{k}"
            bm25_val = bm25.get(key)
            sem_val = metrics_json.get(key)
            comparison["bm25"][key] = bm25_val
            comparison["semantic"][key] = sem_val
            if bm25_val is not None and sem_val is not None:
                comparison["semantic_minus_bm25"][key] = round(sem_val - bm25_val, 6)
            else:
                comparison["semantic_minus_bm25"][key] = None

    Path(args.comparison_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.comparison_output).write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [OK] {args.comparison_output.name}")

    # Final summary
    print()
    for metric_name, kvs in averages.items():
        for k, v in kvs.items():
            diff = comparison["semantic_minus_bm25"].get(f"{metric_name}_at_{k}", None)
            diff_str = f" ({diff:+.4f})" if diff is not None else ""
            print(f"  {metric_name}@{k:2d} = {v:.4f}{diff_str}")

    print(f"\n  query coverage: {query_coverage:.1%}"
          f" (BM25: {bm25.get('query_coverage', 'N/A')})")
    print(f"  zero-vector queries: {zero_vec_count}")
    print("Done.")


if __name__ == "__main__":
    main()
