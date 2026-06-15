#!/usr/bin/env python3
"""BM25 keyword retrieval baseline for PSR-SRS MVP.

Usage::

    .venv/Scripts/python.exe scripts/run_bm25.py \\
        --items data/sample/items.csv \\
        --queries data/sample/queries.csv \\
        --qrels data/sample/qrels.csv \\
        --config configs/bm25.json \\
        --output outputs/bm25
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

from psr_srs_mvp.evaluation import evaluate_all, macro_average, MetricResult
from psr_srs_mvp.retrieval import (
    BM25Config,
    BM25Index,
    Document,
    SearchResult,
    build_item_text,
    load_items,
    load_qrels,
    load_queries,
    tokenize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 keyword retrieval baseline")
    parser.add_argument("--items", required=True, type=Path, help="Path to items.csv")
    parser.add_argument("--queries", required=True, type=Path, help="Path to queries.csv")
    parser.add_argument("--qrels", required=True, type=Path, help="Path to qrels.csv")
    parser.add_argument("--config", required=True, type=Path, help="Path to bm25 config JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    # 1. Load config
    print(f"[1/7] Loading config: {args.config}")
    cfg = BM25Config.from_json(args.config)
    print(f"  k1={cfg.k1}, b={cfg.b}, top_k={cfg.top_k_values}, rel_threshold={cfg.relevance_threshold}")

    # 2. Load items
    print(f"[2/7] Loading items: {args.items}")
    item_rows = load_items(args.items)
    print(f"  {len(item_rows)} items loaded")

    # 3. Build BM25 index
    print("[3/7] Building BM25 index …")
    documents: list[Document] = []
    for r in item_rows:
        text = build_item_text(
            title=r["title"],
            description=r["description"],
            category=r["category"],
            subcategory=r["subcategory"],
            brand=r["brand"],
            weights=cfg.field_weights,
        )
        tokens = tokenize(text, remove_stopwords=cfg.use_stopwords)
        documents.append(Document(item_id=r["item_id"], tokens=tokens, length=len(tokens)))

    index = BM25Index.build(documents, k1=cfg.k1, b=cfg.b)
    print(f"  {index.document_count} docs, {index.vocabulary_size} unique terms, avgdl={index.avgdl:.1f}")

    # 4. Load queries
    print(f"[4/7] Loading queries: {args.queries}")
    query_rows = load_queries(args.queries)
    print(f"  {len(query_rows)} queries loaded")

    # 5. Search all queries
    print(f"[5/7] Searching (max K={cfg.max_k}) …")
    all_results: dict[str, list[SearchResult]] = {}
    queries_with_results = 0
    queries_empty = 0
    for q in query_rows:
        qid = q["query_id"]
        results = index.search(q["query_text"], top_k=cfg.max_k)
        all_results[qid] = results
        if results:
            queries_with_results += 1
        else:
            queries_empty += 1
    print(f"  {queries_with_results} queries returned results, {queries_empty} empty")

    # 6. Evaluate
    print("[6/7] Evaluating …")
    qrels = load_qrels(args.qrels)
    metrics = evaluate_all(
        all_results, query_rows, qrels,
        ks=cfg.top_k_values, relevance_threshold=cfg.relevance_threshold,
    )
    averages = macro_average(metrics)
    for metric_name, kvs in averages.items():
        for k, v in kvs.items():
            print(f"  {metric_name}@{k:2d} = {v:.4f}")

    # 7. Export
    print(f"[7/7] Exporting to: {args.output}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    metrics_json = {
        "algorithm": "BM25",
        "k1": cfg.k1,
        "b": cfg.b,
        "document_count": index.document_count,
        "query_count": len(metrics),
        "tokenizer": "regex [a-z0-9]+ with built-in stopwords",
        "field_weights": cfg.field_weights,
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
    qm_fields = ["query_id", "query_text"]
    for k in cfg.top_k_values:
        for m in ("precision", "recall", "mrr", "ndcg"):
            qm_fields.append(f"{m}_at_{k}")

    with (out / "query_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=qm_fields, extrasaction="ignore")
        writer.writeheader()
        for m in sorted(metrics, key=lambda m: m.query_id):
            writer.writerow(m.to_flat_dict())
    print(f"  [OK] query_metrics.csv ({len(metrics)} rows)")

    # --- search_results.csv ---
    sr_fields = ["query_id", "query_text", "rank", "item_id", "bm25_score", "relevance_grade"]
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
                    "bm25_score": f"{r.score:.6f}",
                    "relevance_grade": str(qrels_for_q.get(r.item_id, 0)),
                })
    with (out / "search_results.csv").open("r", encoding="utf-8") as f:
        sr_count = sum(1 for _ in f)
    print(f"  [OK] search_results.csv ({sr_count - 1} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
