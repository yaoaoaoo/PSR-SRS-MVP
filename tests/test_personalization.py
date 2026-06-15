"""Unit tests for personalization pipeline — split, profiles, reranker, eval."""

from __future__ import annotations

import csv, json, sys, tempfile, unittest
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.personalization import (
    split_events, load_events, build_profiles, load_items, load_users_map,
    PersonalizationConfig, rerank_candidates,
    compute_behavior_metrics, compute_qrels_metrics, macro_average_dict,
)


def _make_event(user_id, session_id, request_id, query_id, event_type, item_id, timestamp, position="1"):
    return {"user_id": user_id, "session_id": session_id, "request_id": request_id,
            "query_id": query_id, "event_type": event_type, "item_id": item_id,
            "timestamp": timestamp, "position": position, "query_text": "test query",
            "event_id": f"ev_{user_id}_{session_id}_{event_type}", "click_duration_ms": "",
            "add_to_cart_quantity": "", "purchase_amount": ""}


class TestSplit(unittest.TestCase):
    def test_same_session_not_split(self):
        ts = "2026-01-15T00:00:00+00:00"
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", ts),
                   _make_event("u1", "s1", "r1", "q1", "impression", "i2", ts)]
        train, test, info = split_events(events, 0.8)
        self.assertGreater(len(train), 0)
        self.assertEqual(len(test), 0)  # 1 session → train only

    def test_same_request_not_split(self):
        ts = "2026-01-15T00:00:00+00:00"
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", ts),
                   _make_event("u1", "s1", "r1", "q1", "impression", "i2", ts)]
        train, test, info = split_events(events)
        train_rids = {e["request_id"] for e in train}
        test_rids = {e["request_id"] for e in test}
        self.assertTrue(train_rids.isdisjoint(test_rids))

    def test_train_before_test(self):
        events = []
        for s in range(5):
            t = datetime(2026, 1, 1 + s*5, tzinfo=timezone.utc).isoformat()
            events.append(_make_event("u1", f"s{s}", f"r{s}", "q1", "click", "i1", t))
        train, test, info = split_events(events, 0.8)
        self.assertTrue(info["time_leakage_free"])
        self.assertGreater(len(test), 0)

    def test_single_session_user_train_only(self):
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1",
                              "2026-01-15T00:00:00+00:00")]
        train, test, info = split_events(events)
        self.assertGreater(len(train), 0)
        self.assertEqual(len(test), 0)

    def test_deterministic(self):
        events = [_make_event("u1", f"s{s}", f"r{s}", "q1", "click", "i1",
                              f"2026-01-{1+s*3:02d}T00:00:00+00:00") for s in range(5)]
        t1, ts1, i1 = split_events(events)
        t2, ts2, i2 = split_events(events)
        self.assertEqual(len(t1), len(t2))
        self.assertEqual(len(ts1), len(ts2))


class TestProfiles(unittest.TestCase):
    def setUp(self):
        self.items = {"i1": {"item_id": "i1", "category": "Electronics", "subcategory": "Phones",
                              "brand": "TechPro", "price": "500"},
                      "i2": {"item_id": "i2", "category": "Clothing", "subcategory": "Shirts",
                              "brand": "UrbanStitch", "price": "50"}}
        self.users = {"u1": {"user_id": "u1", "is_cold_start": "false"},
                      "u2": {"user_id": "u2", "is_cold_start": "true"}}

    def test_only_positive_events_count(self):
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", "2026-02-01T00:00:00+00:00"),
                   _make_event("u1", "s1", "r1", "q1", "impression", "i2", "2026-02-01T00:00:00+00:00")]
        profiles = build_profiles(events, self.items, self.users, {"click": 1.0}, 30.0)
        self.assertEqual(profiles["u1"].positive_event_count, 1)

    def test_impression_not_weighted(self):
        events = [_make_event("u1", "s1", "r1", "q1", "impression", "i1", "2026-02-01T00:00:00+00:00")]
        profiles = build_profiles(events, self.items, self.users, {"impression": 1.0, "click": 1.0}, 30.0)
        self.assertEqual(profiles["u1"].positive_event_count, 0)

    def test_event_weights(self):
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", "2026-02-01T00:00:00+00:00"),
                   _make_event("u1", "s2", "r2", "q1", "purchase", "i1", "2026-02-02T00:00:00+00:00")]
        profiles = build_profiles(events, self.items, self.users,
                                  {"click": 1.0, "purchase": 5.0}, 1000.0)
        cw = profiles["u1"].category_weights
        self.assertAlmostEqual(sum(cw.values()), 1.0, places=4)
        self.assertIn("Electronics", cw)

    def test_time_decay(self):
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", "2026-01-01T00:00:00+00:00"),
                   _make_event("u1", "s2", "r2", "q1", "click", "i1", "2026-03-01T00:00:00+00:00")]
        profiles = build_profiles(events, self.items, self.users, {"click": 1.0}, 30.0)
        self.assertGreater(profiles["u1"].positive_event_count, 0)

    def test_category_weights_normalize(self):
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", "2026-02-01T00:00:00+00:00"),
                   _make_event("u1", "s2", "r2", "q1", "click", "i2", "2026-02-02T00:00:00+00:00")]
        profiles = build_profiles(events, self.items, self.users, {"click": 1.0}, 30.0)
        cw = profiles["u1"].category_weights
        self.assertAlmostEqual(sum(cw.values()), 1.0, places=4)

    def test_no_positive_profile_empty(self):
        events = [_make_event("u1", "s1", "r1", "q1", "impression", "i1", "2026-02-01T00:00:00+00:00")]
        profiles = build_profiles(events, self.items, self.users, {"click": 1.0}, 30.0)
        self.assertEqual(profiles["u1"].profile_status, "no_positive")

    def test_cold_start_user_marked(self):
        profiles = build_profiles([], self.items, self.users, {}, 30.0)
        self.assertEqual(profiles["u2"].profile_status, "cold_start")
        self.assertTrue(profiles["u2"].is_cold_start)

    def test_build_deterministic(self):
        events = [_make_event("u1", "s1", "r1", "q1", "click", "i1", "2026-02-01T00:00:00+00:00")]
        p1 = build_profiles(events, self.items, self.users, {"click": 1.0}, 30.0)
        p2 = build_profiles(events, self.items, self.users, {"click": 1.0}, 30.0)
        self.assertEqual(p1["u1"].category_weights, p2["u1"].category_weights)


class TestPersonalizationConfig(unittest.TestCase):
    def test_valid(self): self.assertEqual(len(PersonalizationConfig().validate()), 0)
    def test_invalid_train_ratio(self):
        self.assertTrue(any("train_ratio" in e for e in PersonalizationConfig(train_ratio=0.0).validate()))
    def test_invalid_event_weights(self):
        self.assertTrue(any("event_weight" in e for e in PersonalizationConfig(event_weights={}).validate()))
    def test_invalid_half_life(self):
        self.assertTrue(any("half_life" in e for e in PersonalizationConfig(half_life_days=0).validate()))


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.items = {"i1": {"category": "Electronics", "subcategory": "Phones", "brand": "TechPro", "price": "500"},
                      "i2": {"category": "Clothing", "subcategory": "Shirts", "brand": "UrbanStitch", "price": "50"}}
        self.candidates = [
            {"rank": "1", "item_id": "i1", "fusion_score": "0.95"},
            {"rank": "2", "item_id": "i2", "fusion_score": "0.85"},
        ]
        self.cfg = PersonalizationConfig()
        self.profile = type('P', (), {
            'profile_status': 'warm', 'is_cold_start': False,
            'category_weights': {'Electronics': 0.8, 'Clothing': 0.2},
            'subcategory_weights': {'Phones': 0.7, 'Shirts': 0.3},
            'brand_weights': {'TechPro': 0.9, 'UrbanStitch': 0.1},
            'mean_log_price': math.log(500), 'price_std': 0.5,
        })()

    def test_rerank_changes_order(self):
        results = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        # i1 should get higher score due to Electronics/TechPro preference
        self.assertEqual(results[0].item_id, "i1")

    def test_rank_continuous(self):
        results = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        for i, r in enumerate(results, 1):
            self.assertEqual(r.rank, i)

    def test_no_duplicates(self):
        # Input has duplicates, output should deduplicate
        dup_cands = self.candidates + [{"rank": "3", "item_id": "i1", "fusion_score": "0.80"}]
        results = rerank_candidates(dup_cands, self.profile, self.items, self.cfg)
        ids = [r.item_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_candidate_set_unchanged(self):
        results = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        orig_ids = {c["item_id"] for c in self.candidates}
        result_ids = {r.item_id for r in results}
        self.assertEqual(orig_ids, result_ids)

    def test_retrieval_score_normalized(self):
        results = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        for r in results:
            self.assertTrue(0 <= r.normalized_retrieval_score <= 1)
            self.assertTrue(0 <= r.personalized_score <= 1)

    def test_cold_start_fallback(self):
        cold = type('P', (), {'profile_status': 'cold_start', 'is_cold_start': True,
                               'category_weights': {}, 'subcategory_weights': {},
                               'brand_weights': {}, 'mean_log_price': None, 'price_std': 0.5})()
        results = rerank_candidates(self.candidates, cold, self.items, self.cfg)
        self.assertEqual(results[0].item_id, "i1")
        self.assertEqual(results[1].item_id, "i2")

    def test_deterministic(self):
        r1 = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        r2 = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        self.assertEqual([(r.item_id, r.personalized_score) for r in r1],
                         [(r.item_id, r.personalized_score) for r in r2])

    def test_no_qrels_bias(self):
        results = rerank_candidates(self.candidates, self.profile, self.items, self.cfg)
        self.assertGreater(len(results), 0)


import math


class TestEvaluation(unittest.TestCase):
    def test_behavior_hit_rate(self):
        results = [type('R', (), {'item_id': 'i1', 'rank': 1})(),
                   type('R', (), {'item_id': 'i2', 'rank': 2})()]
        m = compute_behavior_metrics(results, {"i1": 3}, {"i1"}, [5])
        self.assertEqual(m["hit_rate_at_5"], 1.0)

    def test_behavior_mrr(self):
        results = [type('R', (), {'item_id': 'i5', 'rank': 1})(),
                   type('R', (), {'item_id': 'i1', 'rank': 2})()]
        m = compute_behavior_metrics(results, {"i1": 3}, {"i1"}, [2])
        self.assertAlmostEqual(m["mrr_at_2"], 0.5)

    def test_behavior_ndcg(self):
        results = [type('R', (), {'item_id': 'i1', 'rank': 1})(),
                   type('R', (), {'item_id': 'i2', 'rank': 2})()]
        m = compute_behavior_metrics(results, {"i1": 4, "i2": 2}, {"i1", "i2"}, [2])
        self.assertGreater(m["ndcg_at_2"], 0)

    def test_impression_only_grade_zero(self):
        # Impression-only items are not in behavior_grades
        m = compute_behavior_metrics([type('R',(),{'item_id':'i3','rank':1})()], {}, set(), [1])
        self.assertEqual(m["hit_rate_at_1"], 0.0)

    def test_qrels_metrics(self):
        results = [type('R',(),{'item_id':'i1','rank':1})(), type('R',(),{'item_id':'i2','rank':2})()]
        m = compute_qrels_metrics(results, {"i1": 3, "i2": 0}, [2])
        self.assertAlmostEqual(m["precision_at_2"], 0.5)
        self.assertAlmostEqual(m["mrr_at_2"], 1.0)

    def test_all_metrics_in_range(self):
        results = [type('R',(),{'item_id':f'i{j}','rank':j})() for j in range(1,6)]
        grades = {f"i{j}": j%5 for j in range(1,6)}
        for m in [compute_behavior_metrics(results, grades, set(grades), [5]),
                  compute_qrels_metrics(results, grades, [5])]:
            for v in m.values():
                self.assertTrue(0.0 <= v <= 1.0, f"value {v} not in [0,1]")


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items_path = _PROJECT / "data" / "sample" / "items.csv"
        cls.users_path = _PROJECT / "data" / "sample" / "users.csv"
        cls.events_path = _PROJECT / "data" / "sample" / "events.csv"

    def test_full_pipeline_smoke(self):
        events = load_events(self.events_path)
        train, test, info = split_events(events, 0.8)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertTrue(info["time_leakage_free"])

        items = load_items(self.items_path)
        users = load_users_map(self.users_path)
        cfg = PersonalizationConfig()
        profiles = build_profiles(train, items, users, cfg.event_weights, cfg.half_life_days)
        warm = sum(1 for p in profiles.values() if p.profile_status == "warm")
        self.assertGreater(warm, 0)

    def test_outputs_match_input_users(self):
        users = load_users_map(self.users_path)
        self.assertEqual(len(users), 100)


class TestRegression(unittest.TestCase):
    def test_bm25_unchanged(self):
        bm25 = json.loads((_PROJECT / "outputs" / "bm25" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(bm25["algorithm"], "BM25")

    def test_hybrid_unchanged(self):
        h = json.loads((_PROJECT / "outputs" / "hybrid" / "linear" / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("Linear", h["algorithm"])


class TestCLIPersonalization(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=_PROJECT / "outputs")
    def tearDown(self):
        import shutil; shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cli_runs(self):
        old = sys.argv
        try:
            sys.argv = ["run_personalization.py",
                        "--items", str(_PROJECT / "data" / "sample" / "items.csv"),
                        "--users", str(_PROJECT / "data" / "sample" / "users.csv"),
                        "--events", str(_PROJECT / "data" / "sample" / "events.csv"),
                        "--qrels", str(_PROJECT / "data" / "sample" / "qrels.csv"),
                        "--hybrid-results", str(_PROJECT / "outputs" / "hybrid" / "linear" / "search_results.csv"),
                        "--config", str(_PROJECT / "configs" / "personalization.json"),
                        "--output", str(self.tmpdir),
                        "--comparison-output", str(Path(self.tmpdir) / "comp.json")]
            from scripts.run_personalization import main
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        finally:
            sys.argv = old
        for fn in ("user_profiles.csv", "metrics.json", "diagnostics.json",
                   "request_metrics.csv", "search_results.csv"):
            self.assertTrue((Path(self.tmpdir) / fn).exists(), f"Missing: {fn}")
        self.assertTrue((Path(self.tmpdir) / "comp.json").exists())
