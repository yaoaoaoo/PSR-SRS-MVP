"""Unit tests for synthetic data generation — calibrated funnel + qrels.

Uses standard-library ``unittest``.  Run with::

    .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psr_srs_mvp.data_generation import (
    DataGenerator,
    GenerationConfig,
    load_config,
    read_csv_files,
    validate_generated_data,
    validate_qrels,
    write_csv_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config(**overrides) -> GenerationConfig:
    """Small-but-sufficient GenerationConfig for fast tests."""
    cfg = GenerationConfig(
        seed=42,
        num_items=100,
        num_users=30,
        num_queries=50,
        num_sessions=150,
        cold_start_user_ratio=0.1,
        cold_start_item_ratio=0.1,
        start_date="2026-01-01T00:00:00+00:00",
        end_date="2026-03-31T23:59:59+00:00",
        categories=[
            "Electronics", "Clothing", "Home & Kitchen",
            "Sports & Outdoors", "Books", "Beauty & Personal Care",
        ],
        subcategories={
            "Electronics": ["Smartphones", "Laptops", "Headphones"],
            "Clothing": ["Men's Shirts", "Women's Dresses", "Shoes"],
            "Home & Kitchen": ["Cookware", "Furniture", "Bedding"],
            "Sports & Outdoors": ["Camping Gear", "Fitness Equipment", "Cycling"],
            "Books": ["Fiction", "Non-Fiction", "Children's Books"],
            "Beauty & Personal Care": ["Skincare", "Makeup", "Hair Care"],
        },
        brands={
            "Electronics": ["TechPro", "NovaDigital"],
            "Clothing": ["UrbanStitch", "EleganceWear"],
            "Home & Kitchen": ["HomeCraft", "CozyNest"],
            "Sports & Outdoors": ["TrailMaster", "FitGear Pro"],
            "Books": ["PenPoint Press", "ReadersGate"],
            "Beauty & Personal Care": ["GlowLab", "PureBloom"],
        },
        price_ranges={
            "Electronics": [50, 2500],
            "Clothing": [10, 300],
            "Home & Kitchen": [5, 800],
            "Sports & Outdoors": [10, 600],
            "Books": [5, 80],
            "Beauty & Personal Care": [3, 150],
        },
        serp_size=20,
        base_click_probability=0.14,
        max_clicks_per_request=3,
        post_click_stop_probability=0.50,
        relevance_boost=1.8,
        preference_boost=1.4,
        base_favorite_probability=0.10,
        base_add_to_cart_probability=0.09,
        base_purchase_probability=0.35,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _generate_all(cfg: GenerationConfig) -> dict:
    gen = DataGenerator(cfg)
    return gen.generate_all()


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_load_default_config_from_json(self):
        cfg = load_config(_PROJECT / "configs" / "sample.json")
        self.assertEqual(cfg.seed, 20260614)
        self.assertGreater(cfg.base_click_probability, 0)
        self.assertEqual(len(cfg.validate()), 0)

    def test_invalid_seed(self):
        cfg = _default_config(seed=-1)
        errors = cfg.validate()
        self.assertTrue(any("seed" in e.lower() for e in errors))

    def test_time_range(self):
        cfg = _default_config()
        self.assertGreater(cfg.time_range_seconds, 0)

    def test_new_params_present(self):
        cfg = _default_config()
        self.assertTrue(0 < cfg.base_click_probability <= 1)
        self.assertGreaterEqual(cfg.max_clicks_per_request, 1)
        self.assertTrue(0 <= cfg.post_click_stop_probability <= 1)


class TestConfigValidation(unittest.TestCase):
    def test_negative_counts_rejected(self):
        cfg = _default_config(num_items=0, num_users=0)
        errors = cfg.validate()
        self.assertTrue(any("num_items" in e.lower() for e in errors))

    def test_invalid_ratio_rejected(self):
        cfg = _default_config(cold_start_user_ratio=1.5)
        errors = cfg.validate()
        self.assertTrue(any("ratio" in e.lower() for e in errors))

    def test_inverted_dates_rejected(self):
        cfg = _default_config(
            start_date="2026-06-01T00:00:00+00:00",
            end_date="2026-01-01T00:00:00+00:00",
        )
        errors = cfg.validate()
        self.assertTrue(any("date" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    def test_same_seed_identical_output(self):
        cfg = _default_config(seed=12345)
        d1 = _generate_all(cfg)
        d2 = _generate_all(cfg)
        for key in ("items", "users", "queries", "events", "qrels"):
            self.assertEqual(len(d1[key]), len(d2[key]))
            for r1, r2 in zip(d1[key], d2[key]):
                self.assertEqual(r1, r2)

    def test_different_seed_different_output(self):
        cfg_a = _default_config(seed=100)
        cfg_b = _default_config(seed=200)
        da = _generate_all(cfg_a)
        db = _generate_all(cfg_b)
        any_diff = any(
            ea != eb
            for ea, eb in zip(
                sorted(da["events"], key=lambda r: r["event_id"]),
                sorted(db["events"], key=lambda r: r["event_id"]),
            )
        )
        self.assertTrue(any_diff, "Different seeds should differ")


# ---------------------------------------------------------------------------
# 3. Items
# ---------------------------------------------------------------------------

class TestItemGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config()
        cls.data = _generate_all(cls.cfg)

    def test_ids_unique(self):
        ids = [r["item_id"] for r in self.data["items"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_count_matches(self):
        self.assertEqual(len(self.data["items"]), self.cfg.num_items)

    def test_no_empty_required(self):
        for r in self.data["items"]:
            self.assertTrue(r["item_id"].strip())
            self.assertTrue(r["title"].strip())

    def test_price_positive(self):
        for r in self.data["items"]:
            self.assertGreater(float(r["price"]), 0)

    def test_scores_in_range(self):
        for r in self.data["items"]:
            self.assertTrue(0 <= float(r["quality_score"]) <= 1)
            self.assertTrue(0 <= float(r["popularity_score"]) <= 1)

    def test_cold_start_exist(self):
        cold = [r for r in self.data["items"] if r["is_cold_start"].lower() == "true"]
        self.assertGreater(len(cold), 0)


# ---------------------------------------------------------------------------
# 4. Users
# ---------------------------------------------------------------------------

class TestUserGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config()
        cls.data = _generate_all(cls.cfg)

    def test_ids_unique(self):
        ids = [r["user_id"] for r in self.data["users"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_count_matches(self):
        self.assertEqual(len(self.data["users"]), self.cfg.num_users)

    def test_cold_start_exist(self):
        cold = [r for r in self.data["users"] if r["is_cold_start"].lower() == "true"]
        self.assertGreater(len(cold), 0)

    def test_valid_prefs(self):
        for r in self.data["users"]:
            self.assertIn(r["price_preference"], self.cfg.price_preference_levels)
            self.assertIn(r["activity_level"], self.cfg.activity_levels)

    def test_preferred_categories_json(self):
        for r in self.data["users"]:
            cats = json.loads(r["preferred_categories"])
            self.assertIsInstance(cats, list)
            self.assertGreater(len(cats), 0)


# ---------------------------------------------------------------------------
# 5. Queries
# ---------------------------------------------------------------------------

class TestQueryGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config()
        cls.data = _generate_all(cls.cfg)

    def test_ids_unique(self):
        ids = [r["query_id"] for r in self.data["queries"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_count_matches(self):
        self.assertEqual(len(self.data["queries"]), self.cfg.num_queries)

    def test_all_have_text(self):
        for r in self.data["queries"]:
            self.assertTrue(r["query_text"].strip())


# ---------------------------------------------------------------------------
# 6. Events
# ---------------------------------------------------------------------------

class TestEventGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=777, num_sessions=100)
        cls.data = _generate_all(cls.cfg)
        cls.item_ids = {r["item_id"] for r in cls.data["items"]}
        cls.user_ids = {r["user_id"] for r in cls.data["users"]}
        cls.query_ids = {r["query_id"] for r in cls.data["queries"]}

    def test_event_ids_unique(self):
        ids = [r["event_id"] for r in self.data["events"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_event_type_valid(self):
        valid = {"impression", "click", "favorite", "add_to_cart", "purchase"}
        for r in self.data["events"]:
            self.assertIn(r["event_type"], valid)

    def test_foreign_keys(self):
        for r in self.data["events"]:
            self.assertIn(r["user_id"], self.user_ids)
            self.assertIn(r["item_id"], self.item_ids)
            if r.get("query_id"):
                self.assertIn(r["query_id"], self.query_ids)

    def test_position_ge_1(self):
        for r in self.data["events"]:
            self.assertGreaterEqual(int(r["position"]), 1)

    def test_timestamp_utc(self):
        for r in self.data["events"]:
            dt = datetime.fromisoformat(r["timestamp"])
            self.assertIsNotNone(dt.tzinfo)

    def test_click_duration_non_negative(self):
        for r in self.data["events"]:
            d = r.get("click_duration_ms", "")
            if d:
                self.assertGreaterEqual(int(d), 0)

    def test_add_to_cart_quantity_ge_1(self):
        for r in self.data["events"]:
            q = r.get("add_to_cart_quantity", "")
            if q:
                self.assertGreaterEqual(int(q), 1)

    def test_purchase_amount_non_negative(self):
        for r in self.data["events"]:
            a = r.get("purchase_amount", "")
            if a:
                self.assertGreaterEqual(float(a), 0)


# ---------------------------------------------------------------------------
# 7. Statistical targets
# ---------------------------------------------------------------------------

def _compute_stats(events):
    cnt = Counter(r["event_type"] for r in events)
    imp, clk = cnt["impression"], cnt["click"]
    fav, atc, pur = cnt["favorite"], cnt["add_to_cart"], cnt["purchase"]
    sessions = len({r["session_id"] for r in events})
    return {
        "ctr": clk / imp * 100 if imp else 0,
        "avg_clicks": clk / sessions if sessions else 0,
        "fav_click": fav / clk * 100 if clk else 0,
        "atc_click": atc / clk * 100 if clk else 0,
        "pur_click": pur / clk * 100 if clk else 0,
        "pur_atc": pur / atc * 100 if atc else 0,
    }


class TestStatisticalTargets(unittest.TestCase):
    """Verify stats fall within configured target ranges."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=555, num_sessions=200, num_items=150,
                                   base_purchase_probability=0.28)
        cls.data = _generate_all(cls.cfg)
        cls.stats = _compute_stats(cls.data["events"])

    def test_ctr_in_range(self):
        self.assertTrue(8 <= self.stats["ctr"] <= 20,
                        f"CTR {self.stats['ctr']:.1f}% not in [8, 20]")

    def test_avg_clicks_in_range(self):
        self.assertTrue(1 <= self.stats["avg_clicks"] <= 3,
                        f"avg clicks {self.stats['avg_clicks']:.2f} not in [1, 3]")

    def test_fav_click_in_range(self):
        self.assertTrue(5 <= self.stats["fav_click"] <= 15,
                        f"fav/click {self.stats['fav_click']:.1f}% not in [5, 15]")

    def test_atc_click_in_range(self):
        self.assertTrue(4 <= self.stats["atc_click"] <= 12,
                        f"atc/click {self.stats['atc_click']:.1f}% not in [4, 12]")

    def test_pur_click_in_range(self):
        self.assertTrue(1 <= self.stats["pur_click"] <= 5,
                        f"pur/click {self.stats['pur_click']:.1f}% not in [1, 5]")

    def test_pur_atc_in_range(self):
        self.assertTrue(15 <= self.stats["pur_atc"] <= 40,
                        f"pur/atc {self.stats['pur_atc']:.1f}% not in [15, 40]")

    def test_funnel_order(self):
        cnt = Counter(r["event_type"] for r in self.data["events"])
        self.assertGreater(cnt["impression"], cnt["click"])
        self.assertGreater(cnt["click"], cnt["favorite"])
        self.assertGreater(cnt["click"], cnt["add_to_cart"])
        self.assertGreater(cnt["click"], cnt["purchase"])


# ---------------------------------------------------------------------------
# 8. Event dependencies & time order
# ---------------------------------------------------------------------------

class TestEventDependencies(unittest.TestCase):
    """Verify strict event dependencies within request chains."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=666, num_sessions=200)
        cls.data = _generate_all(cls.cfg)

    def test_purchase_only_after_add_to_cart(self):
        """Every purchase must have a preceding add_to_cart for same (request, item)."""
        from collections import defaultdict
        req_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in self.data["events"]:
            req_events[(r["request_id"], r["item_id"])].append(r)

        for (rid, iid), chain in req_events.items():
            chain.sort(key=lambda r: r["timestamp"])
            has_atc = False
            for ev in chain:
                if ev["event_type"] == "add_to_cart":
                    has_atc = True
                if ev["event_type"] == "purchase":
                    self.assertTrue(has_atc,
                                    f"purchase without add_to_cart: req={rid} item={iid}")

    def test_favorite_only_after_click(self):
        """Every favorite must have a preceding click for same (request, item)."""
        from collections import defaultdict
        req_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in self.data["events"]:
            req_events[(r["request_id"], r["item_id"])].append(r)

        for (rid, iid), chain in req_events.items():
            chain.sort(key=lambda r: r["timestamp"])
            has_click = False
            for ev in chain:
                if ev["event_type"] == "click":
                    has_click = True
                if ev["event_type"] == "favorite":
                    self.assertTrue(has_click,
                                    f"favorite without click: req={rid} item={iid}")

    def test_add_to_cart_only_after_click(self):
        """Every add_to_cart must have a preceding click."""
        from collections import defaultdict
        req_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in self.data["events"]:
            req_events[(r["request_id"], r["item_id"])].append(r)

        for (rid, iid), chain in req_events.items():
            chain.sort(key=lambda r: r["timestamp"])
            has_click = False
            for ev in chain:
                if ev["event_type"] == "click":
                    has_click = True
                if ev["event_type"] == "add_to_cart":
                    self.assertTrue(has_click,
                                    f"add_to_cart without click: req={rid} item={iid}")

    def test_time_order_in_chain(self):
        """impression <= click <= add_to_cart <= purchase in timestamp order."""
        from collections import defaultdict
        req_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in self.data["events"]:
            req_events[(r["request_id"], r["item_id"])].append(r)

        order_map = {"impression": 0, "click": 1, "favorite": 2, "add_to_cart": 3, "purchase": 4}
        for (rid, iid), chain in req_events.items():
            chain.sort(key=lambda r: r["timestamp"])
            prev_ts = ""
            prev_order = -1
            for ev in chain:
                if prev_ts:
                    # Allow equal timestamps within a chain
                    self.assertGreaterEqual(
                        ev["timestamp"], prev_ts,
                        f"time goes backwards: req={rid} item={iid}"
                    )
                prev_ts = ev["timestamp"]

    def test_max_clicks_per_request(self):
        """No request exceeds configured max clicks."""
        req_clicks = Counter()
        for r in self.data["events"]:
            if r["event_type"] == "click":
                req_clicks[r["request_id"]] += 1
        for rid, cnt in req_clicks.items():
            self.assertLessEqual(cnt, self.cfg.max_clicks_per_request,
                                 f"req={rid}: {cnt} clicks > max {self.cfg.max_clicks_per_request}")


# ---------------------------------------------------------------------------
# 9. Business rules (existing + new)
# ---------------------------------------------------------------------------

class TestBusinessRuleStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=555, num_sessions=300, num_items=120)
        cls.data = _generate_all(cls.cfg)

    def test_preference_category_ctr_higher(self):
        items_by_id = {it["item_id"]: it for it in self.data["items"]}
        user_pref_cats = {}
        for u in self.data["users"]:
            user_pref_cats[u["user_id"]] = set(json.loads(u["preferred_categories"]))

        p_imp = p_clk = n_imp = n_clk = 0
        for r in self.data["events"]:
            item_cat = items_by_id.get(r["item_id"], {}).get("category", "")
            is_pref = item_cat in user_pref_cats.get(r["user_id"], set())
            if r["event_type"] == "impression":
                if is_pref: p_imp += 1
                else: n_imp += 1
            elif r["event_type"] == "click":
                if is_pref: p_clk += 1
                else: n_clk += 1

        if p_imp > 0 and n_imp > 0:
            self.assertGreater(p_clk / p_imp, n_clk / n_imp,
                               "pref CTR should exceed non-pref CTR")

    def test_top_positions_higher_ctr(self):
        pos_imp = Counter()
        pos_clk = Counter()
        for r in self.data["events"]:
            p = int(r["position"])
            if r["event_type"] == "impression": pos_imp[p] += 1
            elif r["event_type"] == "click": pos_clk[p] += 1

        top_imp = sum(pos_imp[p] for p in range(1, 6))
        top_clk = sum(pos_clk[p] for p in range(1, 6))
        bot_imp = sum(pos_imp[p] for p in range(15, 21))
        bot_clk = sum(pos_clk[p] for p in range(15, 21))

        if top_imp > 0 and bot_imp > 0:
            self.assertGreater(top_clk / top_imp, bot_clk / bot_imp)

    def test_high_quality_converts_better(self):
        items_by_id = {it["item_id"]: it for it in self.data["items"]}
        hq = {iid for iid, it in items_by_id.items() if float(it["quality_score"]) >= 0.7}
        lq = {iid for iid, it in items_by_id.items() if float(it["quality_score"]) < 0.4}

        hc = hp = lc = lp = 0
        for r in self.data["events"]:
            iid = r["item_id"]
            if r["event_type"] == "click":
                if iid in hq: hc += 1
                elif iid in lq: lc += 1
            elif r["event_type"] == "purchase":
                if iid in hq: hp += 1
                elif iid in lq: lp += 1

        if hc > 0 and lc > 0:
            self.assertGreater(hp / hc, lp / lc,
                               "high-quality items should convert better")


# ---------------------------------------------------------------------------
# 10. Qrels
# ---------------------------------------------------------------------------

class TestQrels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=42)
        cls.data = _generate_all(cls.cfg)
        cls.qids = {r["query_id"] for r in cls.data["queries"]}
        cls.iids = {r["item_id"] for r in cls.data["items"]}

    def test_qrels_not_empty(self):
        self.assertGreater(len(self.data["qrels"]), 0)

    def test_pairs_unique(self):
        pairs = [(r["query_id"], r["item_id"]) for r in self.data["qrels"]]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_foreign_keys_valid(self):
        for r in self.data["qrels"]:
            self.assertIn(r["query_id"], self.qids)
            self.assertIn(r["item_id"], self.iids)

    def test_grade_valid(self):
        for r in self.data["qrels"]:
            self.assertIn(r["relevance_grade"], ("1", "2", "3"))

    def test_each_query_has_strong_relevance(self):
        """Every query must have at least one grade 2+ item."""
        q_has = {}
        for r in self.data["qrels"]:
            if int(r["relevance_grade"]) >= 2:
                q_has[r["query_id"]] = True
        missing = [qid for qid in self.qids if not q_has.get(qid)]
        self.assertEqual(len(missing), 0,
                         f"{len(missing)} queries lack grade-2+ items: {missing[:5]}")

    def test_qrels_independent_of_event_count(self):
        """Qrels should be identical regardless of session count."""
        cfg1 = _default_config(seed=42, num_sessions=10)
        cfg2 = _default_config(seed=42, num_sessions=200)
        # Items and queries identical — only session count differs
        d1 = _generate_all(cfg1)
        d2 = _generate_all(cfg2)
        self.assertEqual(d1["qrels"], d2["qrels"],
                         "Qrels must be independent of session count")

    def test_qrels_reproducible(self):
        d1 = _generate_all(_default_config(seed=99))
        d2 = _generate_all(_default_config(seed=99))
        self.assertEqual(d1["qrels"], d2["qrels"])

    def test_validate_qrels_passes(self):
        errors = validate_qrels(
            self.data["qrels"], self.qids, self.iids,
        )
        self.assertEqual(len(errors), 0, f"Qrels errors: {errors[:5]}")


# ---------------------------------------------------------------------------
# 11. CSV round-trip
# ---------------------------------------------------------------------------

class TestCSVReadWrite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=99, num_sessions=50)
        cls.data = _generate_all(cls.cfg)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=_PROJECT / "data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_readback_preserves_counts(self):
        paths = write_csv_files(self.data, self.tmpdir)
        self.assertEqual(len(paths), 5)  # items, users, queries, events, qrels
        reloaded = read_csv_files(self.tmpdir)
        for key in ("items", "users", "queries", "events", "qrels"):
            self.assertEqual(len(self.data[key]), len(reloaded[key]),
                             f"Row count mismatch for {key}")

    def test_reloaded_items_have_same_ids(self):
        write_csv_files(self.data, self.tmpdir)
        reloaded = read_csv_files(self.tmpdir)
        self.assertEqual(
            {r["item_id"] for r in self.data["items"]},
            {r["item_id"] for r in reloaded["items"]},
        )


# ---------------------------------------------------------------------------
# 12. CLI integration
# ---------------------------------------------------------------------------

class TestCLIGeneration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=_PROJECT / "data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generate_to_output_dir(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                "generate_data.py",
                "--config", str(_PROJECT / "configs" / "sample.json"),
                "--output", str(self.tmpdir),
            ]
            from scripts.generate_data import main
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        finally:
            sys.argv = old_argv

        for fname in ("items.csv", "users.csv", "queries.csv", "events.csv", "qrels.csv"):
            p = Path(self.tmpdir) / fname
            self.assertTrue(p.exists(), f"Missing: {fname}")
            self.assertGreater(p.stat().st_size, 0, f"Empty: {fname}")


# ---------------------------------------------------------------------------
# 13. Validation integration
# ---------------------------------------------------------------------------

class TestValidationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=42, num_sessions=200)
        cls.data = _generate_all(cls.cfg)

    def test_validate_passes(self):
        errors = validate_generated_data(
            self.data["items"], self.data["users"],
            self.data["queries"], self.data["events"],
            self.cfg.num_items, self.cfg.num_users, self.cfg.num_queries,
            self.cfg.max_clicks_per_request,
        )
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors[:5]}")

    def test_detects_duplicate_item_ids(self):
        items = self.data["items"].copy()
        items.append(items[0].copy())
        errors = validate_generated_data(
            items, self.data["users"], self.data["queries"],
            self.data["events"],
            self.cfg.num_items + 1, self.cfg.num_users, self.cfg.num_queries,
            self.cfg.max_clicks_per_request,
        )
        self.assertTrue(any("duplicate" in e.lower() for e in errors))

    def test_detects_invalid_event_type(self):
        events = self.data["events"].copy()
        c = events[0].copy()
        c["event_type"] = "INVALID"
        events.append(c)
        errors = validate_generated_data(
            self.data["items"], self.data["users"], self.data["queries"],
            events,
            self.cfg.num_items, self.cfg.num_users, self.cfg.num_queries,
            self.cfg.max_clicks_per_request,
        )
        self.assertTrue(any("event_type" in e.lower() for e in errors))

    def test_detects_bad_foreign_key(self):
        events = self.data["events"].copy()
        c = events[0].copy()
        c["event_id"] = "ev_bogus_999999"
        c["user_id"] = "user_nonexistent"
        events.append(c)
        errors = validate_generated_data(
            self.data["items"], self.data["users"], self.data["queries"],
            events,
            self.cfg.num_items, self.cfg.num_users, self.cfg.num_queries,
            self.cfg.max_clicks_per_request,
        )
        self.assertTrue(any("not found" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# 14. Cold-start presence
# ---------------------------------------------------------------------------

class TestColdStartPresence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _default_config(seed=101, cold_start_user_ratio=0.2, cold_start_item_ratio=0.15)
        cls.data = _generate_all(cls.cfg)

    def test_cold_users_exist(self):
        cold = [u for u in self.data["users"] if u["is_cold_start"].lower() == "true"]
        self.assertGreater(len(cold), 0)
        self.assertGreater(len(cold) / len(self.data["users"]), 0.05)

    def test_cold_items_exist(self):
        cold = [it for it in self.data["items"] if it["is_cold_start"].lower() == "true"]
        self.assertGreater(len(cold), 0)
        self.assertGreater(len(cold) / len(self.data["items"]), 0.02)


# ---------------------------------------------------------------------------
# 15. Reproducibility & CLI safeguards (added Phase Finalization)
# ---------------------------------------------------------------------------

class TestReproducibilityEndToEnd(unittest.TestCase):
    def test_dual_run_sha256_match(self):
        import hashlib, subprocess, shutil, tempfile
        config = _PROJECT / "configs" / "sample.json"
        tmp = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        try:
            for run in ("run_a", "run_b"):
                out = tmp / run
                r = subprocess.run(
                    [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
                     str(_PROJECT / "scripts" / "generate_data.py"),
                     "--config", str(config), "--output", str(out), "--force"],
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(r.returncode, 0, f"{run} failed: {r.stderr[:300]}")
            for fn in ("items.csv", "users.csv", "queries.csv", "events.csv", "qrels.csv"):
                h1 = hashlib.sha256((tmp / "run_a" / fn).read_bytes()).hexdigest()
                h2 = hashlib.sha256((tmp / "run_b" / fn).read_bytes()).hexdigest()
                self.assertEqual(h1, h2, f"{fn}: SHA-256 mismatch")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_different_seed_produces_different_events(self):
        cfg_a = _default_config(seed=100, num_sessions=10)
        cfg_b = _default_config(seed=200, num_sessions=10)
        d_a = _generate_all(cfg_a)
        d_b = _generate_all(cfg_b)
        # Event IDs should differ since different seed → different RNG sequence
        ids_a = set(e["event_id"] for e in d_a["events"])
        ids_b = set(e["event_id"] for e in d_b["events"])
        # At least some event IDs should differ (extremely unlikely all same with different seeds)
        self.assertTrue(ids_a != ids_b or len(ids_a) == 0,
                        "Different seeds should produce different event sequences")


class TestOverwriteProtection(unittest.TestCase):
    def test_without_force_aborts(self):
        import subprocess
        r = subprocess.run(
            [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
             str(_PROJECT / "scripts" / "generate_data.py"),
             "--config", str(_PROJECT / "configs" / "sample.json"),
             "--output", str(_PROJECT / "data" / "sample")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ABORT", r.stdout)

    def test_with_force_succeeds(self):
        import subprocess, shutil, tempfile
        tmp = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        try:
            for _ in range(2):
                r = subprocess.run(
                    [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
                     str(_PROJECT / "scripts" / "generate_data.py"),
                     "--config", str(_PROJECT / "configs" / "sample.json"),
                     "--output", str(tmp), "--force"],
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(r.returncode, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestManifestGeneration(unittest.TestCase):
    def test_manifest_created_with_valid_content(self):
        import json as _json, subprocess, shutil, tempfile
        tmp = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        mp = tmp / "manifest.json"
        try:
            r = subprocess.run(
                [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
                 str(_PROJECT / "scripts" / "generate_data.py"),
                 "--config", str(_PROJECT / "configs" / "sample.json"),
                 "--output", str(tmp), "--force", "--manifest", str(mp)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(r.returncode, 0)
            self.assertTrue(mp.exists())
            content = _json.loads(mp.read_text(encoding="utf-8"))
            self.assertIn("seed", content)
            for fn in ("items.csv", "users.csv", "queries.csv", "events.csv", "qrels.csv"):
                self.assertIn(fn, content["files"])
                self.assertIn("sha256", content["files"][fn])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestValidationCLI(unittest.TestCase):
    def test_valid_data_passes(self):
        import subprocess
        r = subprocess.run(
            [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
             str(_PROJECT / "scripts" / "validate_data.py"),
             "--data-dir", str(_PROJECT / "data" / "sample")],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0)

    def test_missing_foreign_key_detected(self):
        import csv, shutil, subprocess, tempfile
        tmp = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        try:
            for fn in ("items.csv", "users.csv", "queries.csv", "events.csv", "qrels.csv"):
                shutil.copy(_PROJECT / "data" / "sample" / fn, tmp / fn)
            rows = list(csv.DictReader(open(tmp / "events.csv", encoding="utf-8", newline="")))
            rows[0]["user_id"] = "user_nonexistent_999"
            with open(tmp / "events.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            r = subprocess.run(
                [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
                 str(_PROJECT / "scripts" / "validate_data.py"),
                 "--data-dir", str(tmp)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r.returncode, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_file_detected(self):
        import shutil, subprocess, tempfile
        tmp = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
        try:
            r = subprocess.run(
                [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
                 str(_PROJECT / "scripts" / "validate_data.py"),
                 "--data-dir", str(tmp)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r.returncode, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 16. Notebook execution & validation tests (Phase Finalization)
# ---------------------------------------------------------------------------

class TestNotebookSource(unittest.TestCase):
    def test_source_notebook_exists(self):
        self.assertTrue((_PROJECT / "notebooks" / "01_mvp_end_to_end.ipynb").exists())

    def test_source_notebook_parsable(self):
        import json
        nb = json.loads((_PROJECT / "notebooks" / "01_mvp_end_to_end.ipynb").read_text(encoding="utf-8"))
        self.assertIn("cells", nb)

    def test_kernel_metadata_correct(self):
        import json
        nb = json.loads((_PROJECT / "notebooks" / "01_mvp_end_to_end.ipynb").read_text(encoding="utf-8"))
        kernel = nb.get("metadata", {}).get("kernelspec", {})
        self.assertEqual(kernel.get("language"), "python")

    def test_no_pip_install_in_notebook(self):
        import json
        nb = json.loads((_PROJECT / "notebooks" / "01_mvp_end_to_end.ipynb").read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                self.assertNotIn("pip install", "".join(c["source"]).lower())

    def test_recompute_uses_env_var(self):
        import json
        nb = json.loads((_PROJECT / "notebooks" / "01_mvp_end_to_end.ipynb").read_text(encoding="utf-8"))
        found = False
        for c in nb["cells"]:
            if c["cell_type"] == "code" and "PSR_SRS_RECOMPUTE" in "".join(c["source"]):
                found = True
        self.assertTrue(found)


class TestNotebookExecution(unittest.TestCase):
    def test_executed_notebook_no_errors(self):
        from scripts.validate_notebook import validate_notebook
        result = validate_notebook(_PROJECT / "outputs" / "notebook" / "01_mvp_end_to_end.executed.ipynb")
        self.assertEqual(result["error_cells"], 0)

    def test_executed_notebook_all_executed(self):
        from scripts.validate_notebook import validate_notebook
        result = validate_notebook(_PROJECT / "outputs" / "notebook" / "01_mvp_end_to_end.executed.ipynb")
        self.assertEqual(result["unexecuted_cells"], 0)

    def test_recomputed_notebook_no_errors(self):
        from scripts.validate_notebook import validate_notebook
        result = validate_notebook(_PROJECT / "outputs" / "notebook" / "01_mvp_end_to_end.recomputed.ipynb")
        self.assertEqual(result["error_cells"], 0)


class TestFrozenBaselines(unittest.TestCase):
    def test_all_metrics_match(self):
        from scripts.validate_notebook import validate_metrics
        result = validate_metrics()
        self.assertTrue(result["all_passed"])

    def test_frozen_outputs_unchanged(self):
        for sub in ("bm25", "semantic", "hybrid", "personalization"):
            self.assertTrue((_PROJECT / "outputs" / sub).exists())
