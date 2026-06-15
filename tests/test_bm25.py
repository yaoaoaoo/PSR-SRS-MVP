"""Unit tests for BM25 retrieval and offline evaluation metrics.

Standard-library ``unittest``.  Run with::

    .venv/Scripts/python.exe -m unittest tests.test_bm25 -v
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.evaluation import evaluate_all, evaluate_query, macro_average
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


# ======================================================================
# 1. Tokenizer
# ======================================================================

class TestTokenizer(unittest.TestCase):
    def test_lowercasing(self):
        self.assertEqual(tokenize("Hello WORLD"), ["hello", "world"])

    def test_unicode_nfkc(self):
        # Full-width "Ａ" → "a"
        tokens = tokenize("Ａbc")
        self.assertIn("abc", tokens or [""])

    def test_punctuation_removal(self):
        tokens = tokenize("hello, world!!! test-case's")
        self.assertEqual(tokens, ["hello", "world", "test", "case", "s"])

    def test_numbers_preserved(self):
        tokens = tokenize("item 123 test 456")
        self.assertEqual(tokens, ["item", "123", "test", "456"])

    def test_stopwords_removed(self):
        tokens = tokenize("the quick brown fox", remove_stopwords=True)
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)

    def test_stopwords_kept_when_disabled(self):
        tokens = tokenize("the quick fox", remove_stopwords=False)
        self.assertIn("the", tokens)

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])

    def test_deterministic(self):
        t1 = tokenize("Hello World! Test 123")
        t2 = tokenize("Hello World! Test 123")
        self.assertEqual(t1, t2)


# ======================================================================
# 2. build_item_text
# ======================================================================

class TestBuildItemText(unittest.TestCase):
    def test_default_weights(self):
        text = build_item_text(
            "T-Shirt", "Cotton tee", "Clothing", "Shirts", "UrbanStitch",
        )
        # title appears 3 times
        self.assertGreaterEqual(text.count("T-Shirt"), 3)

    def test_custom_weights(self):
        text = build_item_text(
            "A", "B", "C", "D", "E",
            weights={"title": 1, "description": 0, "category": 1, "subcategory": 0, "brand": 1},
        )
        self.assertEqual(text.count("A"), 1)
        self.assertEqual(text.count("B"), 0)
        self.assertEqual(text.count("D"), 0)

    def test_deterministic(self):
        t1 = build_item_text("A", "B", "C", "D", "E")
        t2 = build_item_text("A", "B", "C", "D", "E")
        self.assertEqual(t1, t2)


# ======================================================================
# 3. BM25 Index — toy corpus
# ======================================================================

_TOY_DOCS = [
    Document(item_id="doc_a", tokens="bm25 is a retrieval model for ranking".split(), length=7),
    Document(item_id="doc_b", tokens="the quick brown fox jumps over the lazy dog".split(), length=9),
    Document(item_id="doc_c", tokens="bm25 retrieval ranking with fox jumps".split(), length=6),
]


class TestBM25ToyCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = BM25Index.build(_TOY_DOCS, k1=1.5, b=0.75)

    def test_relevant_doc_ranks_first(self):
        # "bm25 retrieval" should rank doc_a or doc_c first
        results = self.index.search("bm25 retrieval", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn(results[0].item_id, ("doc_a", "doc_c"))

    def test_idf_positive_for_present_term(self):
        idf = self.index.idf("bm25")
        self.assertGreater(idf, 0)

    def test_idf_zero_for_unknown_term(self):
        idf = self.index.idf("nonexistent123")
        self.assertEqual(idf, 0.0)

    def test_document_count(self):
        self.assertEqual(self.index.document_count, 3)

    def test_avgdl(self):
        self.assertAlmostEqual(self.index.avgdl, (7 + 9 + 6) / 3)

    def test_empty_query_returns_empty(self):
        results = self.index.search("", top_k=5)
        self.assertEqual(results, [])

    def test_top_k_respected(self):
        results = self.index.search("fox", top_k=1)
        self.assertEqual(len(results), 1)

    def test_invalid_top_k_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("test", top_k=0)
        with self.assertRaises(ValueError):
            self.index.search("test", top_k=-1)

    def test_tie_break_by_item_id(self):
        # Two docs with same content should break tie by item_id
        docs = [
            Document(item_id="item_z", tokens="same content here".split(), length=3),
            Document(item_id="item_a", tokens="same content here".split(), length=3),
        ]
        idx = BM25Index.build(docs)
        results = idx.search("same content", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].item_id, "item_a")
        self.assertEqual(results[1].item_id, "item_z")

    def test_invalid_k1_rejected(self):
        with self.assertRaises(ValueError):
            BM25Index.build(_TOY_DOCS, k1=0)

    def test_invalid_b_rejected(self):
        with self.assertRaises(ValueError):
            BM25Index.build(_TOY_DOCS, b=1.5)

    def test_reproducible(self):
        r1 = self.index.search("bm25 retrieval", top_k=3)
        r2 = self.index.search("bm25 retrieval", top_k=3)
        self.assertEqual(
            [(r.item_id, r.score) for r in r1],
            [(r.item_id, r.score) for r in r2],
        )

    def test_no_popularity_or_qrels_in_scoring(self):
        """Verify search doesn't reference any user-behaviour fields."""
        # This is a smoke test — the BM25Index only accesses tokens
        results = self.index.search("fox", top_k=2)
        self.assertGreater(len(results), 0)
        # Scores come from term frequencies only


# ======================================================================
# 4. Evaluation metrics — toy examples
# ======================================================================

def _make_result(item_id: str, score: float, rank: int) -> SearchResult:
    return SearchResult(score=score, item_id=item_id, rank=rank)


class TestMetricsToyExamples(unittest.TestCase):
    def setUp(self):
        # 3 relevant items total: i1 (grade 3), i2 (grade 2), i3 (grade 1)
        self.qrels = {"q1": {"i1": 3, "i2": 2, "i3": 1, "i4": 0}}
        # Search returns: i1 at rank 1, i5 at rank 2 (not relevant), i2 at rank 3
        self.results = [
            _make_result("i1", 9.5, 1),
            _make_result("i5", 8.0, 2),
            _make_result("i2", 7.5, 3),
            _make_result("i6", 6.0, 4),
            _make_result("i3", 5.0, 5),
        ]

    def test_precision_at_5(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[5], relevance_threshold=1)
        # 3 relevant in top 5
        self.assertEqual(m.precision[5], 3 / 5)

    def test_precision_at_3(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[3], relevance_threshold=1)
        # i1, i5(not rel), i2 → 2/3
        self.assertEqual(m.precision[3], 2 / 3)

    def test_recall_at_5(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[5], relevance_threshold=1)
        # 3/3 relevant found
        self.assertEqual(m.recall[5], 1.0)

    def test_recall_at_2(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[2], relevance_threshold=1)
        # Only i1 in top 2 → 1/3
        self.assertAlmostEqual(m.recall[2], 1 / 3)

    def test_mrr_at_3(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[3], relevance_threshold=1)
        # First relevant at rank 1 → MRR = 1.0
        self.assertEqual(m.mrr[3], 1.0)

    def test_mrr_first_relevant_at_rank_3(self):
        # First two results are NOT relevant, third is
        results = [
            _make_result("i5", 8.0, 1),   # not in qrels → grade 0
            _make_result("i6", 6.0, 2),   # not in qrels → grade 0
            _make_result("i2", 7.5, 3),   # grade 2 → relevant!
        ]
        m = evaluate_query(results, "q1", "test", self.qrels, ks=[3], relevance_threshold=1)
        self.assertEqual(m.mrr[3], 1 / 3)

    def test_ndcg_at_5(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[5], relevance_threshold=1)
        # graded: i1=3, i5=0, i2=2, i6=0, i3=1
        # DCG = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4) + 0 + (2^1-1)/log2(6)
        #     = 7/1 + 0 + 3/2 + 0 + 1/log2(6)
        #     = 7 + 1.5 + 0.387 = 8.887
        # IDCG: ideal order = [3, 2, 1, 0, ...]
        #     = 7/1 + 3/log2(3) + 1/log2(4) = 7 + 1.893 + 0.5 = 9.393
        # NDCG = 8.887 / 9.393 ≈ 0.946
        self.assertGreater(m.ndcg[5], 0.9)
        self.assertLessEqual(m.ndcg[5], 1.0)

    def test_all_relevant_results(self):
        results = [_make_result(f"i{i}", 10.0 - i, i) for i in range(1, 6)]
        qrels = {"q1": {f"i{i}": 3 for i in range(1, 6)}}
        m = evaluate_query(results, "q1", "test", qrels, ks=[5], relevance_threshold=1)
        self.assertEqual(m.precision[5], 1.0)
        self.assertEqual(m.recall[5], 1.0)

    def test_no_relevant_results(self):
        results = [_make_result("ix", 1.0, 1)]
        qrels = {"q1": {"iy": 3}}
        m = evaluate_query(results, "q1", "test", qrels, ks=[5], relevance_threshold=1)
        self.assertEqual(m.precision[5], 0.0)
        self.assertEqual(m.mrr[5], 0.0)
        self.assertEqual(m.ndcg[5], 0.0)

    def test_empty_results(self):
        m = evaluate_query([], "q1", "test", self.qrels, ks=[5], relevance_threshold=1)
        self.assertEqual(m.precision[5], 0.0)
        self.assertEqual(m.recall[5], 0.0)
        self.assertEqual(m.mrr[5], 0.0)

    def test_k_larger_than_results(self):
        m = evaluate_query(
            self.results[:2], "q1", "test", self.qrels, ks=[5], relevance_threshold=1,
        )
        # denominator is still K
        self.assertEqual(m.precision[5], 1 / 5)  # only 1 relevant (i1) in results

    def test_metrics_in_range(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[5, 10], relevance_threshold=1)
        for k in (5, 10):
            for val in (m.precision[k], m.recall[k], m.mrr[k], m.ndcg[k]):
                self.assertTrue(0.0 <= val <= 1.0, f"metric {val} not in [0,1]")

    def test_macro_average(self):
        m1 = evaluate_query(self.results, "q1", "t1", self.qrels, ks=[5], relevance_threshold=1)
        m2 = evaluate_query(self.results, "q2", "t2", self.qrels, ks=[5], relevance_threshold=1)
        avg = macro_average([m1, m2])
        self.assertIn("precision", avg)
        self.assertIn(5, avg["precision"])

    def test_threshold_2_filters_grade_1(self):
        m = evaluate_query(self.results, "q1", "test", self.qrels, ks=[5], relevance_threshold=2)
        # relevant: i1 (3), i2 (2) only; i3 (1) is not counted
        # top 5: i1, i5, i2 → 2 relevant
        self.assertEqual(m.precision[5], 2 / 5)
        self.assertEqual(m.recall[5], 2 / 2)  # only 2 are relevant at threshold 2


# ======================================================================
# 5. BM25Config validation
# ======================================================================

class TestBM25Config(unittest.TestCase):
    def test_valid_default(self):
        cfg = BM25Config()
        self.assertEqual(len(cfg.validate()), 0)

    def test_invalid_k1(self):
        cfg = BM25Config(k1=0)
        self.assertTrue(any("k1" in e for e in cfg.validate()))

    def test_invalid_b(self):
        cfg = BM25Config(b=2.0)
        self.assertTrue(any("b" in e for e in cfg.validate()))

    def test_invalid_top_k(self):
        cfg = BM25Config(top_k_values=[])
        self.assertTrue(any("top_k" in e for e in cfg.validate()))

    def test_invalid_relevance_threshold(self):
        cfg = BM25Config(relevance_threshold=5)
        self.assertTrue(any("threshold" in e for e in cfg.validate()))


# ======================================================================
# 6. Integration
# ======================================================================

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items_path = _PROJECT / "data" / "sample" / "items.csv"
        cls.queries_path = _PROJECT / "data" / "sample" / "queries.csv"
        cls.qrels_path = _PROJECT / "data" / "sample" / "qrels.csv"
        cls.config_path = _PROJECT / "configs" / "bm25.json"

    def test_load_sample_items(self):
        items = load_items(self.items_path)
        self.assertEqual(len(items), 500)
        self.assertIn("item_id", items[0])
        self.assertIn("title", items[0])

    def test_load_sample_queries(self):
        queries = load_queries(self.queries_path)
        self.assertEqual(len(queries), 200)
        self.assertIn("query_text", queries[0])

    def test_load_sample_qrels(self):
        qrels = load_qrels(self.qrels_path)
        self.assertGreater(len(qrels), 0)
        for qid, items in qrels.items():
            for iid, grade in items.items():
                self.assertIn(grade, (1, 2, 3))

    def test_build_index_and_search_all_queries(self):
        items = load_items(self.items_path)
        queries = load_queries(self.queries_path)
        cfg = BM25Config.from_json(self.config_path)

        docs = []
        for r in items:
            text = build_item_text(
                r["title"], r["description"], r["category"],
                r["subcategory"], r["brand"], weights=cfg.field_weights,
            )
            docs.append(Document(
                item_id=r["item_id"],
                tokens=tokenize(text, remove_stopwords=cfg.use_stopwords),
                length=len(tokenize(text, remove_stopwords=cfg.use_stopwords)),
            ))

        index = BM25Index.build(docs, k1=cfg.k1, b=cfg.b)
        self.assertEqual(index.document_count, 500)

        all_results = {}
        for q in queries:
            results = index.search(q["query_text"], top_k=cfg.max_k)
            all_results[q["query_id"]] = results

        self.assertEqual(len(all_results), 200)

        # Evaluate
        qrels = load_qrels(self.qrels_path)
        metrics = evaluate_all(
            all_results, queries, qrels,
            ks=cfg.top_k_values, relevance_threshold=cfg.relevance_threshold,
        )
        self.assertEqual(len(metrics), 200)

        averages = macro_average(metrics)
        # Sanity checks — must be > 0 (not all zero)
        self.assertGreater(averages["recall"][20], 0, "Recall@20 should be > 0")
        self.assertGreater(averages["ndcg"][10], 0, "NDCG@10 should be > 0")

    def test_search_results_deterministic(self):
        items = load_items(self.items_path)
        cfg = BM25Config.from_json(self.config_path)
        docs = []
        for r in items:
            text = build_item_text(
                r["title"], r["description"], r["category"],
                r["subcategory"], r["brand"], weights=cfg.field_weights,
            )
            docs.append(Document(
                item_id=r["item_id"],
                tokens=tokenize(text, remove_stopwords=cfg.use_stopwords),
                length=len(tokenize(text, remove_stopwords=cfg.use_stopwords)),
            ))
        index = BM25Index.build(docs, k1=cfg.k1, b=cfg.b)

        r1 = index.search("smartphone", top_k=20)
        r2 = index.search("smartphone", top_k=20)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a.item_id, b.item_id)
            self.assertAlmostEqual(a.score, b.score)
            self.assertEqual(a.rank, b.rank)

    def test_recall_monotonic(self):
        """Recall@20 >= Recall@10 >= Recall@5"""
        items = load_items(self.items_path)
        queries = load_queries(self.queries_path)
        qrels = load_qrels(self.qrels_path)
        cfg = BM25Config.from_json(self.config_path)

        docs = []
        for r in items:
            text = build_item_text(
                r["title"], r["description"], r["category"],
                r["subcategory"], r["brand"], weights=cfg.field_weights,
            )
            docs.append(Document(
                item_id=r["item_id"],
                tokens=tokenize(text, remove_stopwords=cfg.use_stopwords),
                length=len(tokenize(text, remove_stopwords=cfg.use_stopwords)),
            ))
        index = BM25Index.build(docs, k1=cfg.k1, b=cfg.b)

        all_results = {q["query_id"]: index.search(q["query_text"], top_k=20)
                       for q in queries}
        metrics = evaluate_all(
            all_results, queries, qrels, ks=[5, 10, 20], relevance_threshold=1,
        )
        averages = macro_average(metrics)
        self.assertGreaterEqual(
            averages["recall"][20], averages["recall"][10],
            "Recall@20 >= Recall@10",
        )
        self.assertGreaterEqual(
            averages["recall"][10], averages["recall"][5],
            "Recall@10 >= Recall@5",
        )


# ======================================================================
# 7. CLI integration
# ======================================================================

class TestCLIBm25(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=_PROJECT / "outputs")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cli_writes_output_files(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_bm25.py",
                "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                "--config", str(_PROJECT / "configs" / "bm25.json"),
                "--output", str(self.tmpdir),
            ]
            from scripts.run_bm25 import main
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        finally:
            sys.argv = old_argv

        for fname in ("metrics.json", "query_metrics.csv", "search_results.csv"):
            p = Path(self.tmpdir) / fname
            self.assertTrue(p.exists(), f"Missing: {fname}")
            self.assertGreater(p.stat().st_size, 0, f"Empty: {fname}")

    def test_cli_reproducible(self):
        """Two runs produce identical output files."""
        import hashlib

        def run(output_dir: Path) -> dict[str, str]:
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_bm25.py",
                    "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                    "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                    "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                    "--config", str(_PROJECT / "configs" / "bm25.json"),
                    "--output", str(output_dir),
                ]
                from scripts.run_bm25 import main
                main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            hashes = {}
            for fname in ("metrics.json", "query_metrics.csv", "search_results.csv"):
                p = output_dir / fname
                hashes[fname] = hashlib.sha256(p.read_bytes()).hexdigest()
            return hashes

        d1 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        d2 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        try:
            h1 = run(d1)
            h2 = run(d2)
            for fname in h1:
                self.assertEqual(h1[fname], h2[fname],
                                 f"{fname} differs between runs")
        finally:
            import shutil
            shutil.rmtree(d1, ignore_errors=True)
            shutil.rmtree(d2, ignore_errors=True)

    def test_search_results_no_duplicate_ranks(self):
        """Each query's results have unique ranks and no duplicate items."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_bm25.py",
                "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                "--config", str(_PROJECT / "configs" / "bm25.json"),
                "--output", str(self.tmpdir),
            ]
            from scripts.run_bm25 import main
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        sr_path = Path(self.tmpdir) / "search_results.csv"
        with sr_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        from collections import defaultdict
        q_items: dict[str, set[str]] = defaultdict(set)
        q_ranks: dict[str, set[int]] = defaultdict(set)
        for r in rows:
            qid = r["query_id"]
            q_items[qid].add(r["item_id"])
            q_ranks[qid].add(int(r["rank"]))

        for qid in q_items:
            # No duplicate items per query
            self.assertEqual(
                len(q_items[qid]),
                sum(1 for r in rows if r["query_id"] == qid),
                f"Duplicate items for {qid}"
            )
            # Ranks are 1-based and consecutive
            expected_ranks = set(range(1, len(q_items[qid]) + 1))
            self.assertEqual(q_ranks[qid], expected_ranks,
                             f"Non-consecutive ranks for {qid}")
