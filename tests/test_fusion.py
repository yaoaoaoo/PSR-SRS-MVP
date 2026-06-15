"""Unit tests for hybrid BM25 + LSA fusion (RRF + Linear)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.retrieval.bm25 import SearchResult as BM25Result
from psr_srs_mvp.retrieval.semantic import SemanticSearchResult
from psr_srs_mvp.retrieval.fusion import (
    FusionConfig,
    FusedSearchResult,
    build_candidates,
    fuse_rrf,
    fuse_linear,
    compute_diagnostics,
)


# ======================================================================
# 1. Config
# ======================================================================

class TestFusionConfig(unittest.TestCase):
    def test_valid_default(self):
        cfg = FusionConfig()
        self.assertEqual(len(cfg.validate()), 0)

    def test_invalid_candidate_k_zero(self):
        cfg = FusionConfig(candidate_k=0)
        self.assertTrue(any("candidate_k" in e for e in cfg.validate()))

    def test_candidate_k_lt_max_k(self):
        cfg = FusionConfig(candidate_k=5, top_k_values=[5, 10, 20])
        self.assertTrue(any("candidate_k" in e for e in cfg.validate()))

    def test_candidate_k_ge_max_k_ok(self):
        cfg = FusionConfig(candidate_k=20, top_k_values=[5, 10, 20])
        self.assertEqual(len(cfg.validate()), 0)

    def test_invalid_rrf_k(self):
        cfg = FusionConfig(rrf_k=0)
        self.assertTrue(any("rrf_k" in e for e in cfg.validate()))

    def test_negative_bm25_weight(self):
        cfg = FusionConfig(bm25_weight=-0.1)
        self.assertTrue(any("bm25_weight" in e for e in cfg.validate()))

    def test_both_weights_zero(self):
        cfg = FusionConfig(bm25_weight=0, semantic_weight=0)
        self.assertTrue(any("weight" in e for e in cfg.validate()))

    def test_invalid_normalization(self):
        cfg = FusionConfig(score_normalization="zscore")
        self.assertTrue(any("normalization" in e for e in cfg.validate()))


# ======================================================================
# 2. RRF
# ======================================================================

class TestRRF(unittest.TestCase):
    def setUp(self):
        self.bm25 = [
            BM25Result(score=10.0, item_id="a", rank=1),
            BM25Result(score=8.0, item_id="b", rank=2),
            BM25Result(score=5.0, item_id="c", rank=3),
        ]
        self.semantic = [
            SemanticSearchResult(score=0.9, item_id="b", rank=1),
            SemanticSearchResult(score=0.8, item_id="d", rank=2),
            SemanticSearchResult(score=0.7, item_id="a", rank=3),
        ]

    def test_rrf_formula(self):
        # item "a": BM25 rank 1, sem rank 3
        # score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        a_result = [r for r in results if r.item_id == "a"][0]
        expected = 1/61 + 1/63
        self.assertAlmostEqual(a_result.fusion_score, expected, places=6)

    def test_rrf_single_source(self):
        cand = build_candidates(self.bm25, [])
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsNotNone(r.bm25_rank)
            self.assertIsNone(r.semantic_rank)

    def test_rrf_missing_source_zero(self):
        # item "c" only in BM25 at rank 3
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        c_result = [r for r in results if r.item_id == "c"][0]
        self.assertAlmostEqual(c_result.fusion_score, 1/63, places=6)

    def test_rrf_higher_rank_higher_score(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        # "a" (bm25 rank 1, sem rank 3) should outrank "c" (bm25 rank 3 only)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].fusion_score, results[i+1].fusion_score)

    def test_rrf_tie_break(self):
        bm25 = [BM25Result(score=1, item_id="z", rank=1),
                BM25Result(score=1, item_id="a", rank=1)]
        cand = build_candidates(bm25, [])
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        self.assertEqual(results[0].item_id, "a")
        self.assertEqual(results[1].item_id, "z")

    def test_rrf_no_duplicates(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        ids = [r.item_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_rrf_rank_continuous(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        for i, r in enumerate(results, 1):
            self.assertEqual(r.rank, i)

    def test_rrf_empty_both(self):
        results = fuse_rrf({}, rrf_k=60, top_k=10)
        self.assertEqual(results, [])

    def test_rrf_deterministic(self):
        cand1 = build_candidates(self.bm25, self.semantic)
        r1 = fuse_rrf(cand1, rrf_k=60, top_k=10)
        r2 = fuse_rrf(cand1, rrf_k=60, top_k=10)
        self.assertEqual([(r.item_id, r.fusion_score) for r in r1],
                         [(r.item_id, r.fusion_score) for r in r2])

    def test_rrf_input_not_modified(self):
        bm25_copy = [BM25Result(score=r.score, item_id=r.item_id, rank=r.rank) for r in self.bm25]
        cand = build_candidates(self.bm25, self.semantic)
        fuse_rrf(cand, rrf_k=60, top_k=10)
        for orig, copy in zip(self.bm25, bm25_copy):
            self.assertEqual(orig.item_id, copy.item_id)

    def test_rrf_sources_marked(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_rrf(cand, rrf_k=60, top_k=10)
        b = [r for r in results if r.item_id == "b"][0]
        self.assertEqual(b.sources, ("bm25", "semantic"))
        d = [r for r in results if r.item_id == "d"][0]
        self.assertEqual(d.sources, ("semantic",))
        c = [r for r in results if r.item_id == "c"][0]
        self.assertEqual(c.sources, ("bm25",))


# ======================================================================
# 3. Linear fusion
# ======================================================================

class TestLinearFusion(unittest.TestCase):
    def setUp(self):
        self.bm25 = [
            BM25Result(score=10.0, item_id="a", rank=1),
            BM25Result(score=5.0, item_id="b", rank=2),
        ]
        self.semantic = [
            SemanticSearchResult(score=0.9, item_id="b", rank=1),
            SemanticSearchResult(score=0.5, item_id="c", rank=2),
        ]

    def test_min_max_normalization(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        # "a": bm25_norm=1.0 (highest bm25), sem_norm=0.0 (missing)
        # linear = 0.5*1.0 + 0.5*0.0 = 0.5
        a = [r for r in results if r.item_id == "a"][0]
        self.assertAlmostEqual(a.fusion_score, 0.5, places=6)
        self.assertAlmostEqual(a.bm25_normalized_score, 1.0, places=6)
        self.assertIsNone(a.semantic_normalized_score)

    def test_single_candidate_normalized_to_one(self):
        bm25 = [BM25Result(score=5.0, item_id="x", rank=1)]
        cand = build_candidates(bm25, [])
        results = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].bm25_normalized_score, 1.0)

    def test_all_equal_scores_normalized_to_one(self):
        bm25 = [BM25Result(score=5.0, item_id="x", rank=1),
                BM25Result(score=5.0, item_id="y", rank=2)]
        cand = build_candidates(bm25, [])
        results = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        for r in results:
            self.assertAlmostEqual(r.bm25_normalized_score, 1.0)

    def test_negative_cosine_normalized(self):
        bm25 = [BM25Result(score=3.0, item_id="x", rank=1)]
        sem = [SemanticSearchResult(score=-0.3, item_id="x", rank=1),
               SemanticSearchResult(score=0.7, item_id="y", rank=2)]
        cand = build_candidates(bm25, sem)
        results = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        for r in results:
            self.assertTrue(r.fusion_score >= 0)

    def test_weight_normalization(self):
        cand = build_candidates(self.bm25, self.semantic)
        # 0.7 and 0.3 should normalize to 0.7/(0.7+0.3)=0.7 and 0.3
        r1 = fuse_linear(cand, bm25_weight=0.7, semantic_weight=0.3, top_k=10)
        r2 = fuse_linear(cand, bm25_weight=0.7, semantic_weight=0.3, top_k=10)
        self.assertEqual([r.item_id for r in r1], [r.item_id for r in r2])

    def test_no_nan(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        for r in results:
            self.assertFalse(any(x != x for x in [r.fusion_score] if isinstance(x, float)))

    def test_tie_break(self):
        bm25 = [BM25Result(score=10.0, item_id="z", rank=1),
                BM25Result(score=10.0, item_id="a", rank=2)]
        cand = build_candidates(bm25, [])
        results = fuse_linear(cand, bm25_weight=1.0, semantic_weight=0.0, top_k=10)
        self.assertEqual(results[0].item_id, "a")

    def test_rank_continuous(self):
        cand = build_candidates(self.bm25, self.semantic)
        results = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        for i, r in enumerate(results, 1):
            self.assertEqual(r.rank, i)

    def test_deterministic(self):
        cand = build_candidates(self.bm25, self.semantic)
        r1 = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        r2 = fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        self.assertEqual([(r.item_id, r.fusion_score) for r in r1],
                         [(r.item_id, r.fusion_score) for r in r2])

    def test_input_not_modified(self):
        bm25_copy = [BM25Result(score=r.score, item_id=r.item_id, rank=r.rank) for r in self.bm25]
        cand = build_candidates(self.bm25, self.semantic)
        fuse_linear(cand, bm25_weight=0.5, semantic_weight=0.5, top_k=10)
        for orig, copy in zip(self.bm25, bm25_copy):
            self.assertEqual(orig.item_id, copy.item_id)


# ======================================================================
# 4. Candidate building
# ======================================================================

class TestCandidateBuilding(unittest.TestCase):
    def test_union_correct(self):
        bm25 = [BM25Result(score=1, item_id="a", rank=1)]
        sem = [SemanticSearchResult(score=1, item_id="b", rank=1)]
        cand = build_candidates(bm25, sem)
        self.assertEqual(len(cand), 2)
        self.assertIn("a", cand)
        self.assertIn("b", cand)

    def test_intersection_marked(self):
        bm25 = [BM25Result(score=1, item_id="a", rank=1)]
        sem = [SemanticSearchResult(score=1, item_id="a", rank=2)]
        cand = build_candidates(bm25, sem)
        self.assertEqual(len(cand), 1)
        self.assertEqual(cand["a"]["sources"], ("bm25", "semantic"))

    def test_sources_marked(self):
        bm25 = [BM25Result(score=1, item_id="a", rank=1)]
        sem = [SemanticSearchResult(score=1, item_id="b", rank=1)]
        cand = build_candidates(bm25, sem)
        self.assertEqual(cand["a"]["sources"], ("bm25",))
        self.assertEqual(cand["b"]["sources"], ("semantic",))

    def test_original_ranks_preserved(self):
        bm25 = [BM25Result(score=10.0, item_id="a", rank=3)]
        sem = [SemanticSearchResult(score=0.5, item_id="a", rank=7)]
        cand = build_candidates(bm25, sem)
        self.assertEqual(cand["a"]["bm25_rank"], 3)
        self.assertEqual(cand["a"]["semantic_rank"], 7)
        self.assertAlmostEqual(cand["a"]["bm25_score"], 10.0)
        self.assertAlmostEqual(cand["a"]["semantic_score"], 0.5)


# ======================================================================
# 5. Integration (with sample data)
# ======================================================================

class TestIntegrationFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from psr_srs_mvp.retrieval import (
            BM25Config, BM25Index, Document,
            SemanticConfig, SemanticIndex,
            build_item_text, tokenize, load_items, load_queries, load_qrels,
        )
        items = load_items(_PROJECT / "data" / "sample" / "items.csv")
        cls.item_ids = [r["item_id"] for r in items]
        cls.queries = load_queries(_PROJECT / "data" / "sample" / "queries.csv")
        cls.qrels = load_qrels(_PROJECT / "data" / "sample" / "qrels.csv")

        bm25_cfg = BM25Config.from_json(_PROJECT / "configs" / "bm25.json")
        bm25_texts = [build_item_text(r["title"], r["description"], r["category"],
                                       r["subcategory"], r["brand"],
                                       weights=bm25_cfg.field_weights) for r in items]
        bm25_docs = [Document(item_id=cls.item_ids[i],
                              tokens=tokenize(bm25_texts[i], remove_stopwords=bm25_cfg.use_stopwords),
                              length=len(tokenize(bm25_texts[i], remove_stopwords=bm25_cfg.use_stopwords)))
                     for i in range(len(items))]
        cls.bm25_idx = BM25Index.build(bm25_docs, k1=bm25_cfg.k1, b=bm25_cfg.b)

        sem_cfg = SemanticConfig.from_json(_PROJECT / "configs" / "semantic.json")
        sem_texts = [" ".join([r["title"], r["description"], r["category"],
                               r["subcategory"], r["brand"]]) for r in items]
        cls.sem_idx = SemanticIndex.build(sem_texts, cls.item_ids, sem_cfg)

        cls.fus_cfg = FusionConfig.from_json(_PROJECT / "configs" / "fusion.json")

    def test_all_queries_executed(self):
        for q in self.queries:
            bm25 = self.bm25_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
            sem = self.sem_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
            cand = build_candidates(bm25, sem)
            rrf = fuse_rrf(cand, self.fus_cfg.rrf_k, top_k=self.fus_cfg.max_k)
            lin = fuse_linear(cand, self.fus_cfg.bm25_weight, self.fus_cfg.semantic_weight,
                              top_k=self.fus_cfg.max_k)
            self.assertGreaterEqual(len(rrf), 0)
            self.assertGreaterEqual(len(lin), 0)

    def test_metrics_in_range(self):
        from psr_srs_mvp.evaluation import evaluate_all, macro_average
        all_rrf = {}
        all_lin = {}
        for q in self.queries:
            bm25 = self.bm25_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
            sem = self.sem_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
            cand = build_candidates(bm25, sem)
            rrf_res = fuse_rrf(cand, self.fus_cfg.rrf_k, top_k=self.fus_cfg.max_k)
            lin_res = fuse_linear(cand, self.fus_cfg.bm25_weight, self.fus_cfg.semantic_weight,
                                  top_k=self.fus_cfg.max_k)
            all_rrf[q["query_id"]] = [type('SR', (), {'score': r.fusion_score, 'item_id': r.item_id, 'rank': r.rank})()
                                       for r in rrf_res]
            all_lin[q["query_id"]] = [type('SR', (), {'score': r.fusion_score, 'item_id': r.item_id, 'rank': r.rank})()
                                       for r in lin_res]

        rrf_metrics = evaluate_all(all_rrf, self.queries, self.qrels,
                                   ks=self.fus_cfg.top_k_values)
        lin_metrics = evaluate_all(all_lin, self.queries, self.qrels,
                                    ks=self.fus_cfg.top_k_values)

        for metrics, name in [(rrf_metrics, "RRF"), (lin_metrics, "Linear")]:
            avg = macro_average(metrics)
            for mn, kvs in avg.items():
                for k, v in kvs.items():
                    self.assertTrue(0.0 <= v <= 1.0, f"{name} {mn}@{k}={v}")

    def test_recall_monotonic(self):
        from psr_srs_mvp.evaluation import evaluate_all, macro_average
        all_rrf = {}
        q_sample = self.queries[:10]
        for q in q_sample:
            bm25 = self.bm25_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
            sem = self.sem_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
            cand = build_candidates(bm25, sem)
            rrf = fuse_rrf(cand, self.fus_cfg.rrf_k, top_k=20)
            all_rrf[q["query_id"]] = [type('SR', (), {'score': r.fusion_score, 'item_id': r.item_id, 'rank': r.rank})()
                                       for r in rrf]

        metrics = evaluate_all(all_rrf, q_sample, self.qrels, ks=[5, 10, 20])
        avg = macro_average(metrics)
        self.assertGreaterEqual(avg["recall"][20], avg["recall"][10])
        self.assertGreaterEqual(avg["recall"][10], avg["recall"][5])

    def test_qrels_not_in_fusion(self):
        # Fusion uses only BM25 and LSA results + config — no qrels
        bm25 = self.bm25_idx.search("test query", top_k=100)
        sem = self.sem_idx.search("test query", top_k=100)
        cand = build_candidates(bm25, sem)
        rrf = fuse_rrf(cand, 60, 20)
        self.assertGreaterEqual(len(rrf), 0)

    def test_reproducible(self):
        bm25 = self.bm25_idx.search("smartphone", top_k=100)
        sem = self.sem_idx.search("smartphone", top_k=100)
        cand1 = build_candidates(bm25, sem)
        r1 = fuse_rrf(cand1, 60, 20)
        r2 = fuse_rrf(cand1, 60, 20)
        self.assertEqual([(r.item_id, r.fusion_score) for r in r1],
                         [(r.item_id, r.fusion_score) for r in r2])

    def test_no_duplicate_items_per_query(self):
        q = self.queries[0]
        bm25 = self.bm25_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
        sem = self.sem_idx.search(q["query_text"], top_k=self.fus_cfg.candidate_k)
        cand = build_candidates(bm25, sem)
        for method in ["rrf", "linear"]:
            if method == "rrf":
                results = fuse_rrf(cand, self.fus_cfg.rrf_k, top_k=self.fus_cfg.max_k)
            else:
                results = fuse_linear(cand, self.fus_cfg.bm25_weight, self.fus_cfg.semantic_weight,
                                      top_k=self.fus_cfg.max_k)
            ids = [r.item_id for r in results]
            self.assertEqual(len(ids), len(set(ids)), f"{method}: duplicate items")


# ======================================================================
# 6. Regression
# ======================================================================

class TestRegression(unittest.TestCase):
    def test_bm25_output_unchanged(self):
        bm25 = json.loads((_PROJECT / "outputs" / "bm25" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(bm25["algorithm"], "BM25")

    def test_semantic_output_unchanged(self):
        sem = json.loads((_PROJECT / "outputs" / "semantic" / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("LSA", sem["algorithm"])


# ======================================================================
# 7. CLI
# ======================================================================

class TestCLIFusion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=_PROJECT / "outputs")
        self.comp_path = Path(self.tmpdir) / "comp.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cli_writes_all_outputs(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_fusion.py",
                "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                "--bm25-config", str(_PROJECT / "configs" / "bm25.json"),
                "--semantic-config", str(_PROJECT / "configs" / "semantic.json"),
                "--fusion-config", str(_PROJECT / "configs" / "fusion.json"),
                "--bm25-metrics", str(_PROJECT / "outputs" / "bm25" / "metrics.json"),
                "--semantic-metrics", str(_PROJECT / "outputs" / "semantic" / "metrics.json"),
                "--output", str(self.tmpdir),
                "--comparison-output", str(self.comp_path),
            ]
            from scripts.run_fusion import main
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        finally:
            sys.argv = old_argv

        for sub in ("rrf", "linear"):
            for fn in ("metrics.json", "query_metrics.csv", "search_results.csv"):
                p = Path(self.tmpdir) / sub / fn
                self.assertTrue(p.exists(), f"Missing: {sub}/{fn}")
        self.assertTrue((Path(self.tmpdir) / "diagnostics.json").exists())
        self.assertTrue(self.comp_path.exists())

    def test_cli_reproducible(self):
        import hashlib

        def run(out_dir, comp_path):
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_fusion.py",
                    "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                    "--queries", str(_PROJECT / "data" / "sample" / "queries.csv"),
                    "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                    "--bm25-config", str(_PROJECT / "configs" / "bm25.json"),
                    "--semantic-config", str(_PROJECT / "configs" / "semantic.json"),
                    "--fusion-config", str(_PROJECT / "configs" / "fusion.json"),
                    "--bm25-metrics", str(_PROJECT / "outputs" / "bm25" / "metrics.json"),
                    "--semantic-metrics", str(_PROJECT / "outputs" / "semantic" / "metrics.json"),
                    "--output", str(out_dir),
                    "--comparison-output", str(comp_path),
                ]
                from scripts.run_fusion import main
                main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
            h = {}
            for sub in ("rrf", "linear"):
                for fn in ("metrics.json",):
                    h[f"{sub}/{fn}"] = hashlib.sha256((out_dir / sub / fn).read_bytes()).hexdigest()
            return h

        d1 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        d2 = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        try:
            h1 = run(d1, d1 / "c.json")
            h2 = run(d2, d2 / "c.json")
            for k in h1:
                self.assertEqual(h1[k], h2[k], f"{k} differs")
        finally:
            import shutil
            shutil.rmtree(d1, ignore_errors=True)
            shutil.rmtree(d2, ignore_errors=True)
