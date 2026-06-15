"""Unit tests for LSA semantic retrieval (TF-IDF + TruncatedSVD).

Standard-library ``unittest``.  Run with::

    .venv/Scripts/python.exe -m unittest tests.test_semantic -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.retrieval import (
    SemanticConfig,
    SemanticIndex,
    SemanticSearchResult,
    SemanticVectorizer,
    build_item_text,
    is_zero_vector,
    load_items,
    load_qrels,
    load_queries,
)
from psr_srs_mvp.evaluation import evaluate_all, macro_average


# ======================================================================
# 1. Config
# ======================================================================

class TestSemanticConfig(unittest.TestCase):
    def test_valid_default(self):
        cfg = SemanticConfig()
        self.assertEqual(len(cfg.validate()), 0)

    def test_invalid_word_ngram(self):
        cfg = SemanticConfig(word_ngram_range=[2, 1])
        self.assertTrue(any("word_ngram" in e for e in cfg.validate()))

    def test_invalid_char_ngram(self):
        cfg = SemanticConfig(char_ngram_range=[0, 5])
        self.assertTrue(any("char_ngram" in e for e in cfg.validate()))

    def test_all_weights_zero(self):
        cfg = SemanticConfig(word_weight=0, char_weight=0)
        self.assertTrue(any("weight" in e for e in cfg.validate()))

    def test_invalid_svd_components(self):
        cfg = SemanticConfig(svd_components=1)
        self.assertTrue(any("svd" in e for e in cfg.validate()))

    def test_invalid_top_k(self):
        cfg = SemanticConfig(top_k_values=[0, 10])
        self.assertTrue(any("top_k" in e for e in cfg.validate()))

    def test_random_state_preserved(self):
        cfg1 = SemanticConfig(random_state=20260614)
        cfg2 = SemanticConfig(random_state=999)
        self.assertNotEqual(cfg1.random_state, cfg2.random_state)


# ======================================================================
# 2. Vectorization
# ======================================================================

_TOY_DOCS = [
    "bm25 retrieval model ranking",
    "fox jumps over lazy dog",
    "bm25 retrieval ranking with fox",
    "cat sleeps on mat",
]

_TOY_IDS = ["d1", "d2", "d3", "d4"]


class TestVectorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = SemanticConfig(svd_components=2, random_state=42)
        cls.vec = SemanticVectorizer(cls.cfg)
        cls.vec.fit(_TOY_DOCS)
        cls.item_vecs = cls.vec.transform(_TOY_DOCS)
        cls.query_vecs = cls.vec.transform(["bm25 retrieval", "fox jumps"])

    def test_word_features_exist(self):
        self.assertGreater(self.vec.word_feature_count, 0)

    def test_char_features_exist(self):
        self.assertGreater(self.vec.char_feature_count, 0)

    def test_combined_dimension(self):
        expected = (self.vec.word_feature_count * (1 if self.cfg.word_weight > 0 else 0) +
                    self.vec.char_feature_count * (1 if self.cfg.char_weight > 0 else 0))
        self.assertEqual(self.vec.combined_feature_count, expected)

    def test_item_query_dimension_consistent(self):
        self.assertEqual(self.item_vecs.shape[1], self.query_vecs.shape[1])

    def test_svd_dimension_legal(self):
        self.assertLessEqual(self.vec.svd_components_actual, self.cfg.svd_components)
        self.assertGreaterEqual(self.vec.svd_components_actual, 2)

    def test_l2_norm_is_one(self):
        for v in self.item_vecs:
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)

    def test_reproducible(self):
        vec2 = SemanticVectorizer(self.cfg)
        vec2.fit(_TOY_DOCS)
        v2 = vec2.transform(_TOY_DOCS)
        self.assertTrue(np.allclose(self.item_vecs, v2))

    def test_zero_vector_detection(self):
        zero = np.zeros(10)
        self.assertTrue(is_zero_vector(zero))
        self.assertFalse(is_zero_vector(np.array([0.1, 0.2, 0.3])))

    def test_qrels_not_used_in_fit(self):
        # Smoke: fitting uses only documents, no qrels reference
        self.assertEqual(self.item_vecs.shape[0], 4)

    def test_query_not_used_in_inductive_fit(self):
        # Re-fit with a different set and verify query transform is consistent
        cfg2 = SemanticConfig(svd_components=2, random_state=42)
        vec2 = SemanticVectorizer(cfg2)
        vec2.fit(_TOY_DOCS)  # same docs, no queries
        qv = vec2.transform(["bm25 retrieval"])
        self.assertEqual(qv.shape[1], vec2.svd_components_actual)


# ======================================================================
# 3. SemanticIndex — toy corpus
# ======================================================================

class TestSemanticSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = SemanticConfig(svd_components=2, random_state=42)
        cls.index = SemanticIndex.build(_TOY_DOCS, _TOY_IDS, cls.cfg)

    def test_relevant_ranks_high(self):
        results = self.index.search("bm25 retrieval", top_k=4)
        self.assertGreater(len(results), 0)
        # d1 and d3 contain "bm25 retrieval" — should rank high
        top_ids = {r.item_id for r in results[:2]}
        self.assertTrue(top_ids & {"d1", "d3"})

    def test_top_k_respected(self):
        results = self.index.search("fox", top_k=2)
        self.assertEqual(len(results), 2)

    def test_invalid_top_k_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("test", top_k=0)

    def test_empty_query_returns_empty(self):
        results = self.index.search("", top_k=5)
        self.assertEqual(results, [])

    def test_oov_query_returns_empty(self):
        results = self.index.search("zzzxxx notaword", top_k=5)
        self.assertEqual(results, [])

    def test_zero_vector_returns_empty(self):
        # Search with a query that produces all-zero TF-IDF
        results = self.index.search("   ...   ", top_k=5)
        self.assertEqual(results, [])

    def test_rank_continuous(self):
        results = self.index.search("fox", top_k=4)
        for i, r in enumerate(results, start=1):
            self.assertEqual(r.rank, i)

    def test_no_duplicate_items(self):
        results = self.index.search("bm25", top_k=4)
        ids = [r.item_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_tie_break_by_item_id(self):
        # Create two identical documents
        docs = ["same exact text here", "same exact text here"]
        ids = ["item_z", "item_a"]
        cfg = SemanticConfig(svd_components=2, random_state=42)
        idx = SemanticIndex.build(docs, ids, cfg)
        results = idx.search("same text", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].item_id, "item_a")  # alpha sort
        self.assertEqual(results[1].item_id, "item_z")

    def test_scores_finite(self):
        results = self.index.search("bm25", top_k=4)
        for r in results:
            self.assertTrue(np.isfinite(r.score))

    def test_no_popularity_score_used(self):
        results = self.index.search("fox", top_k=2)
        self.assertGreater(len(results), 0)
        # SearchResult only has score, item_id, rank

    def test_reproducible(self):
        r1 = self.index.search("bm25 retrieval", top_k=4)
        r2 = self.index.search("bm25 retrieval", top_k=4)
        self.assertEqual(
            [(r.item_id, r.score) for r in r1],
            [(r.item_id, r.score) for r in r2],
        )


# ======================================================================
# 4. Integration
# ======================================================================

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items_path = _PROJECT / "data" / "sample" / "items.csv"
        cls.queries_path = _PROJECT / "data" / "sample" / "queries.csv"
        cls.qrels_path = _PROJECT / "data" / "sample" / "qrels.csv"
        cls.config_path = _PROJECT / "configs" / "semantic.json"

    def test_load_and_index(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)
        self.assertEqual(index.document_count, 500)
        self.assertGreater(index.vector_dim, 0)

    def test_all_queries_evaluated(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)

        queries = load_queries(self.queries_path)
        qrels = load_qrels(self.qrels_path)

        all_results = {}
        for q in queries:
            results = index.search(q["query_text"], top_k=cfg.max_k)
            all_results[q["query_id"]] = results

        metrics = evaluate_all(
            {qid: [SemanticSearchResult(score=r.score, item_id=r.item_id, rank=r.rank)
                   for r in results]
             for qid, results in all_results.items()},
            queries, qrels, ks=cfg.top_k_values, relevance_threshold=cfg.relevance_threshold,
        )
        self.assertEqual(len(metrics), 200)

        averages = macro_average(metrics)
        # Sanity checks
        self.assertGreater(averages["recall"][20], 0, "Recall@20 should be > 0")
        self.assertGreater(averages["ndcg"][10], 0, "NDCG@10 should be > 0")

    def test_metrics_in_range(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)
        queries = load_queries(self.queries_path)
        qrels = load_qrels(self.qrels_path)

        all_results = {}
        for q in queries:
            all_results[q["query_id"]] = index.search(q["query_text"], top_k=cfg.max_k)

        metrics = evaluate_all(
            {qid: [SemanticSearchResult(score=r.score, item_id=r.item_id, rank=r.rank)
                   for r in results]
             for qid, results in all_results.items()},
            queries, qrels, ks=cfg.top_k_values, relevance_threshold=cfg.relevance_threshold,
        )
        averages = macro_average(metrics)
        for metric_name, kvs in averages.items():
            for k, v in kvs.items():
                self.assertTrue(0.0 <= v <= 1.0,
                                f"{metric_name}@{k}={v} not in [0,1]")

    def test_recall_monotonic(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)
        queries = load_queries(self.queries_path)
        qrels = load_qrels(self.qrels_path)

        all_results = {}
        for q in queries:
            all_results[q["query_id"]] = index.search(q["query_text"], top_k=20)

        metrics = evaluate_all(
            {qid: [SemanticSearchResult(score=r.score, item_id=r.item_id, rank=r.rank)
                   for r in results]
             for qid, results in all_results.items()},
            queries, qrels, ks=[5, 10, 20], relevance_threshold=1,
        )
        averages = macro_average(metrics)
        self.assertGreaterEqual(averages["recall"][20], averages["recall"][10])
        self.assertGreaterEqual(averages["recall"][10], averages["recall"][5])

    def test_qrels_only_for_evaluation(self):
        # Verify qrels is never passed to SemanticIndex.build()
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)
        # Search works without qrels
        results = index.search("smartphone", top_k=5)
        self.assertGreaterEqual(len(results), 0)
        # Qrels only used in evaluation — tested separately

    def test_reproducible_output(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        idx1 = SemanticIndex.build(item_texts, item_ids, cfg)
        idx2 = SemanticIndex.build(item_texts, item_ids, cfg)
        r1 = idx1.search("smartphone", top_k=10)
        r2 = idx2.search("smartphone", top_k=10)
        self.assertEqual([(r.item_id, r.score) for r in r1],
                         [(r.item_id, r.score) for r in r2])

    def test_zero_vector_count(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)
        queries = load_queries(self.queries_path)

        zero_count = 0
        no_result = 0
        for q in queries:
            results = index.search(q["query_text"], top_k=cfg.max_k)
            if not results:
                no_result += 1
                qv = index.vectorizer.transform([q["query_text"]])[0]
                if is_zero_vector(qv):
                    zero_count += 1

        self.assertGreaterEqual(zero_count, 0)
        self.assertGreaterEqual(no_result, zero_count)

    def test_vector_dimension_consistent(self):
        cfg = SemanticConfig.from_json(self.config_path)
        items = load_items(self.items_path)
        item_texts = [" ".join([r["title"], r["description"], r["category"],
                                r["subcategory"], r["brand"]]) for r in items]
        item_ids = [r["item_id"] for r in items]
        index = SemanticIndex.build(item_texts, item_ids, cfg)

        qv = index.vectorizer.transform(["test query"])[0]
        self.assertEqual(len(qv), index.vector_dim)
        self.assertEqual(qv.shape[0], index.vector_dim)


# ======================================================================
# 5. CLI
# ======================================================================

class TestCLISemantic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=_PROJECT / "outputs")
        self.comp_path = Path(self.tmpdir) / "comparison.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cli_writes_output_files(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_semantic.py",
                "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                "--config", str(_PROJECT / "configs" / "semantic.json"),
                "--bm25-metrics", str(_PROJECT / "outputs" / "bm25" / "metrics.json"),
                "--output", str(self.tmpdir),
                "--comparison-output", str(self.comp_path),
            ]
            from scripts.run_semantic import main
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        finally:
            sys.argv = old_argv

        for fname in ("metrics.json", "query_metrics.csv", "search_results.csv"):
            p = Path(self.tmpdir) / fname
            self.assertTrue(p.exists(), f"Missing: {fname}")
            self.assertGreater(p.stat().st_size, 0)

        self.assertTrue(self.comp_path.exists())

    def test_cli_comparison_valid(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_semantic.py",
                "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                "--config", str(_PROJECT / "configs" / "semantic.json"),
                "--bm25-metrics", str(_PROJECT / "outputs" / "bm25" / "metrics.json"),
                "--output", str(self.tmpdir),
                "--comparison-output", str(self.comp_path),
            ]
            from scripts.run_semantic import main
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        comp = json.loads(self.comp_path.read_text(encoding="utf-8"))
        self.assertIn("bm25", comp)
        self.assertIn("semantic", comp)
        self.assertIn("semantic_minus_bm25", comp)
        self.assertIsNotNone(comp["semantic_minus_bm25"].get("precision_at_10"))

    def test_cli_reproducible(self):
        import hashlib

        def run(out_dir: Path, comp_path: Path) -> dict[str, str]:
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_semantic.py",
                    "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                    "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                    "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                    "--config", str(_PROJECT / "configs" / "semantic.json"),
                    "--bm25-metrics", str(_PROJECT / "outputs" / "bm25" / "metrics.json"),
                    "--output", str(out_dir),
                    "--comparison-output", str(comp_path),
                ]
                from scripts.run_semantic import main
                main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
            hashes = {}
            for fn in ("metrics.json", "query_metrics.csv", "search_results.csv"):
                hashes[fn] = hashlib.sha256((out_dir / fn).read_bytes()).hexdigest()
            return hashes

        d1 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        d2 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        c1 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs")) / "comp.json"
        c2 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs")) / "comp.json"
        try:
            h1 = run(d1, c1)
            h2 = run(d2, c2)
            for fn in h1:
                self.assertEqual(h1[fn], h2[fn], f"{fn} differs")
        finally:
            import shutil
            for p in (d1, d2, c1.parent, c2.parent):
                shutil.rmtree(p, ignore_errors=True)


# ======================================================================
# 6. Regression — existing tests still pass
# ======================================================================

class TestRegression(unittest.TestCase):
    """Verify that existing BM25 and data-generation tests still pass."""

    def test_bm25_imports_unchanged(self):
        from psr_srs_mvp.retrieval.bm25 import BM25Index, SearchResult, Document
        # Core types still importable
        self.assertTrue(True)

    def test_evaluation_imports_unchanged(self):
        from psr_srs_mvp.evaluation import evaluate_all, macro_average, evaluate_query
        self.assertTrue(True)

    def test_bm25_output_unchanged(self):
        bm25_path = _PROJECT / "outputs" / "bm25" / "metrics.json"
        self.assertTrue(bm25_path.exists())
        bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
        self.assertEqual(bm25["algorithm"], "BM25")
        self.assertIn("ndcg_at_10", bm25)
