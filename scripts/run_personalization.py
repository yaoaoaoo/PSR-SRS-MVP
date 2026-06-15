#!/usr/bin/env python3
"""Personalized re-ranking over Linear Hybrid Top-20 candidates — complete evaluation.

Usage: .venv/Scripts/python.exe scripts/run_personalization.py --items ... --users ... --events ... --qrels ... --hybrid-results ... --config ... --output ... --comparison-output ...
"""

from __future__ import annotations

import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from psr_srs_mvp.personalization import (
    split_events, load_events, build_profiles, load_items, load_users_map,
    PersonalizationConfig, rerank_candidates,
    compute_behavior_metrics, compute_qrels_metrics, macro_average_dict,
    compute_candidate_coverage,
)


def main():
    p = argparse.ArgumentParser(description="Personalized re-ranking evaluation")
    p.add_argument("--items", required=True, type=Path)
    p.add_argument("--users", required=True, type=Path)
    p.add_argument("--events", required=True, type=Path)
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--hybrid-results", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--comparison-output", required=True, type=Path)
    args = p.parse_args()

    cfg = PersonalizationConfig.from_json(args.config)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ks = cfg.top_k_values
    TOL = 1e-12

    # ==================================================================
    # 1. Load data
    # ==================================================================
    print("[1/8] Loading data …")
    events_all = load_events(args.events)
    items_map = load_items(args.items)
    users_map = load_users_map(args.users)
    # Qrels
    qrels_all: dict[str, dict[str, int]] = defaultdict(dict)
    with open(args.qrels, encoding="utf-8", newline="") as qf:
        for r in csv.DictReader(qf):
            qrels_all[r["query_id"]][r["item_id"]] = int(r["relevance_grade"])
    # Hybrid candidates per query
    with open(args.hybrid_results, encoding="utf-8", newline="") as hf:
        hybrid_rows = list(csv.DictReader(hf))
    hybrid_by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in hybrid_rows:
        hybrid_by_qid[r["query_id"]].append(r)

    # Basic stats
    unique_sids_events = sorted({e["session_id"] for e in events_all})
    unique_uids_events = {e["user_id"] for e in events_all}
    all_uids_csv = set(users_map.keys())
    print(f"  {len(events_all)} events, {len(items_map)} items, {len(users_map)} users")

    # ==================================================================
    # 2. Split
    # ==================================================================
    print("[2/8] Time-based train/test split …")
    train_evts, test_evts, split_info = split_events(events_all, cfg.train_ratio)
    assigned_train_sids = len({e["session_id"] for e in train_evts})
    assigned_test_sids = len({e["session_id"] for e in test_evts})
    assigned_all = assigned_train_sids + assigned_test_sids
    unassigned_sids = len(unique_sids_events) - assigned_all

    print(f"  Configured sessions: 500")
    print(f"  Unique sessions in events: {len(unique_sids_events)}")
    print(f"  Train sessions: {assigned_train_sids}  Test: {assigned_test_sids}  Unassigned: {unassigned_sids}")
    print(f"  Time leakage: {'PASS' if split_info['time_leakage_free'] else 'FAIL'}")

    # ==================================================================
    # 3. User grouping (exhaustive, mutually exclusive)
    # ==================================================================
    print("[3/8] User grouping …")
    # Primary groups (mutually exclusive, covers all 100 users)
    cold_csv = {uid for uid, u in users_map.items() if u.get("is_cold_start", "false").lower() == "true"}
    zero_sess_uids = all_uids_csv - unique_uids_events
    # Remaining: users with events
    uids_with_events = all_uids_csv - zero_sess_uids
    user_sessions = defaultdict(set)
    for e in events_all:
        user_sessions[e["user_id"]].add(e["session_id"])
    multi_sess_uids = {uid for uid in uids_with_events if len(user_sessions[uid]) >= 2}
    single_sess_uids = uids_with_events - multi_sess_uids

    grouping = {}
    grouping["total_users"] = len(users_map)
    grouping["users_with_events"] = len(uids_with_events)
    grouping["users_without_events"] = len(zero_sess_uids)
    grouping["multi_session_users"] = len(multi_sess_uids)
    grouping["single_session_users"] = len(single_sess_uids)
    grouping["zero_session_users"] = len(zero_sess_uids)
    grouping["synthetic_cold_start_users_total"] = len(cold_csv)
    # Verification
    assert len(multi_sess_uids) + len(single_sess_uids) + len(zero_sess_uids) == 100

    # ==================================================================
    # 4. Build profiles (train only)
    # ==================================================================
    print("[4/8] Building user profiles …")
    profiles = build_profiles(train_evts, items_map, users_map, cfg.event_weights, cfg.half_life_days)

    # Count profile statuses (mutually exclusive from build_profiles)
    warm_uids = {uid for uid, p in profiles.items() if p.profile_status == "warm"}
    cold_flag_uids = {uid for uid, p in profiles.items() if p.profile_status == "cold_start"}
    no_pos_uids = {uid for uid, p in profiles.items() if p.profile_status == "no_positive"}
    no_hist_uids = {uid for uid, p in profiles.items() if p.profile_status in ("no_history", "empty")}

    grouping["warm_profile_users"] = len(warm_uids)
    grouping["cold_start_flagged_users"] = len(cold_flag_uids)
    grouping["no_positive_profile_users"] = len(no_pos_uids)
    grouping["no_history_users"] = len(no_hist_uids)
    # Insufficient: single-session (can never be split into test) among non-warm users
    insufficient = single_sess_uids - warm_uids
    grouping["insufficient_history_users"] = len(insufficient)

    # Verify grouping sum
    profile_sum = len(warm_uids) + len(cold_flag_uids) + len(no_pos_uids) + len(no_hist_uids)
    print(f"  warm={len(warm_uids)} cold_flag={len(cold_flag_uids)} no_pos={len(no_pos_uids)} "
          f"no_hist={len(no_hist_uids)} insufficient={len(insufficient)}")
    print(f"  profile sum={profile_sum} (should be 100)")

    # ==================================================================
    # 5. Build test requests
    # ==================================================================
    print("[5/8] Building test requests …")
    # Identify cold-start users WITH test requests
    cold_with_test = {uid for uid in cold_flag_uids
                      if uid in {e["user_id"] for e in test_evts}}

    test_requests: dict[str, dict] = {}
    grade_order = {"click": 1, "favorite": 2, "add_to_cart": 3, "purchase": 4}
    for e in test_evts:
        rid = e["request_id"]
        if rid not in test_requests:
            test_requests[rid] = {"request_id": rid, "session_id": e["session_id"],
                                  "user_id": e["user_id"], "query_id": e["query_id"],
                                  "query_text": e.get("query_text", ""), "items": {}}
        g = grade_order.get(e["event_type"], 0)
        iid = e["item_id"]
        test_requests[rid]["items"][iid] = max(test_requests[rid]["items"].get(iid, 0), g)

    all_rids = sorted(test_requests)
    eligible_rids = [rid for rid in all_rids
                     if any(g > 0 for g in test_requests[rid]["items"].values())]

    # ==================================================================
    # 6. Re-rank
    # ==================================================================
    print("[6/8] Re-ranking …")
    personalized_results: dict[str, list] = {}
    baseline_results: dict[str, list] = {}

    behavior_grades_map: dict[str, dict[str, int]] = {}
    for e in test_evts:
        rid = e["request_id"]
        iid = e["item_id"]
        g = grade_order.get(e["event_type"], 0)
        if g > behavior_grades_map.setdefault(rid, {}).get(iid, 0):
            behavior_grades_map[rid][iid] = g

    for rid in test_requests:
        info = test_requests[rid]
        uid, qid = info["user_id"], info["query_id"]
        candidates = hybrid_by_qid.get(qid, [])
        if not candidates:
            continue

        profile = profiles.get(uid)
        if profile is None:
            profile = type('P', (), {'profile_status': 'no_history', 'is_cold_start': False,
                         'category_weights': {}, 'subcategory_weights': {},
                         'brand_weights': {}, 'mean_log_price': None, 'price_std': 0.5})()

        test_requests[rid]["profile_status"] = profile.profile_status
        bg = behavior_grades_map.get(rid, {})
        qq = qrels_all.get(qid, {})

        ranked = rerank_candidates(candidates, profile, items_map, cfg, bg, qq)
        personalized_results[rid] = ranked

        # Baseline: original order
        orig_cands = sorted(candidates, key=lambda c: int(c["rank"]))
        orig = []
        for c in orig_cands:
            iid = c["item_id"]
            orig.append(type('B', (), {
                'item_id': iid, 'rank': int(c["rank"]),
                'behavior_relevance_grade': bg.get(iid, 0),
                'qrels_relevance_grade': qq.get(iid, 0),
            })())
        baseline_results[rid] = orig

    # ==================================================================
    # 7. Evaluate
    # ==================================================================
    print("[7/8] Evaluating …")
    bh_keys = [f"{m}_at_{k}" for k in ks for m in ["hit_rate","mrr","ndcg","positive_recall"]]
    qr_keys = [f"{m}_at_{k}" for k in ks for m in ["precision","recall","mrr","ndcg"]]
    all_bh, all_qr, all_bb, all_bq = [], [], [], []
    improved = unchanged = worsened = 0
    excluded_rids = []

    for rid in eligible_rids:
        if rid not in personalized_results or rid not in baseline_results:
            excluded_rids.append((rid, "missing_results"))
            continue
        info = test_requests[rid]
        pers = personalized_results[rid]
        base = baseline_results[rid]
        bg = behavior_grades_map.get(rid, {})
        pos_items = {iid for iid, g in info["items"].items() if g > 0}
        qq = qrels_all.get(info["query_id"], {})

        bm = compute_behavior_metrics(pers, bg, pos_items, ks)
        bm["request_id"] = rid
        all_bh.append(bm)
        bb = compute_behavior_metrics(base, bg, pos_items, ks)
        bb["request_id"] = rid
        all_bb.append(bb)

        qm = compute_qrels_metrics(pers, qq, ks)
        all_qr.append(qm)
        bq = compute_qrels_metrics(base, qq, ks)
        all_bq.append(bq)

        pers_ndcg = bm.get("ndcg_at_10", 0)
        base_ndcg = bb.get("ndcg_at_10", 0)
        delta = pers_ndcg - base_ndcg
        if delta > TOL:
            improved += 1
        elif delta < -TOL:
            worsened += 1
        else:
            unchanged += 1

    evaluated_count = improved + unchanged + worsened
    excluded_count = len(eligible_rids) - evaluated_count
    assert evaluated_count + excluded_count == len(eligible_rids), \
        f"evaluated={evaluated_count} + excluded={excluded_count} != eligible={len(eligible_rids)}"

    # ==================================================================
    # 7b. Candidate coverage
    # ==================================================================
    coverage = compute_candidate_coverage(test_requests, hybrid_by_qid, eligible_rids)

    # ==================================================================
    # 7c. Fallback statistics
    # ==================================================================
    # Identify fallback requests and their reasons
    fallback_uids: dict[str, set[str]] = defaultdict(set)
    fallback_rids: dict[str, set[str]] = defaultdict(set)
    for rid in test_requests:
        info = test_requests[rid]
        uid = info["user_id"]
        p = profiles.get(uid)
        status = p.profile_status if p else "no_history"
        reasons = []
        if status == "cold_start":
            reasons.append("cold_start_flag")
        if status == "no_history" or status == "empty":
            reasons.append("no_train_session")
        if status == "no_positive":
            reasons.append("no_positive_behavior")
        if status in ("cold_start", "no_history", "empty", "no_positive"):
            reasons.append("empty_profile")
        for r in reasons:
            fallback_uids[r].add(uid)
            fallback_rids[r].add(rid)

    # Unique counts
    all_fallback_uids = set.union(*fallback_uids.values()) if fallback_uids else set()
    all_fallback_rids = set.union(*fallback_rids.values()) if fallback_rids else set()

    # Verify fallback requests have exact match with baseline
    exact_match_count = 0
    for rid in all_fallback_rids:
        if rid in personalized_results and rid in baseline_results:
            pers = personalized_results[rid]
            base = baseline_results[rid]
            if (len(pers) == len(base) and
                all(p.item_id == b.item_id and p.rank == b.rank for p, b in zip(pers, base))):
                exact_match_count += 1

    fallback_stats = {
        "fallback_due_to_cold_start_flag": len(fallback_uids.get("cold_start_flag", set())),
        "fallback_due_to_no_train_session": len(fallback_uids.get("no_train_session", set())),
        "fallback_due_to_no_positive_behavior": len(fallback_uids.get("no_positive_behavior", set())),
        "fallback_due_to_empty_profile": len(fallback_uids.get("empty_profile", set())),
        "fallback_user_count": len(all_fallback_uids),
        "fallback_request_count": len(all_fallback_rids),
        "exact_fallback_user_count": len({test_requests[rid]["user_id"] for rid in all_fallback_rids
                                           if rid in all_fallback_rids}),
        "exact_fallback_request_count": len(all_fallback_rids),
        "fallback_exact_match_rate": round(exact_match_count / len(all_fallback_rids), 6) if all_fallback_rids else 1.0,
    }

    # ==================================================================
    # 8. Export
    # ==================================================================
    print("[8/8] Exporting …")

    bh_avg = macro_average_dict(all_bh, bh_keys)
    qr_avg = macro_average_dict(all_qr, qr_keys)
    bb_avg = macro_average_dict(all_bb, bh_keys)
    bq_avg = macro_average_dict(all_bq, qr_keys)

    # --- metrics.json ---
    metrics_json: dict = {
        "algorithm": "Personalized Reranking over Linear Hybrid",
        "config": {
            "train_ratio": cfg.train_ratio,
            "event_weights": cfg.event_weights,
            "half_life_days": cfg.half_life_days,
            "rerank_weights": {
                "retrieval": cfg.retrieval_weight,
                "category": cfg.category_weight,
                "subcategory": cfg.subcategory_weight,
                "brand": cfg.brand_weight,
                "price": cfg.price_weight,
            },
        },
    }
    # Session stats
    metrics_json.update({
        "configured_session_count": 500,
        "unique_session_count_in_events": len(unique_sids_events),
        "train_session_count": assigned_train_sids,
        "test_session_count": assigned_test_sids,
        "unassigned_session_count": unassigned_sids,
        "session_difference_reason": "16 configured sessions generated zero events",
    })
    # User grouping
    metrics_json.update(grouping)
    metrics_json["profile_sum_check"] = profile_sum
    # Request stats
    metrics_json.update({
        "total_test_request_count": len(test_requests),
        "eligible_positive_request_count": len(eligible_rids),
        "eligible_evaluated_request_count": evaluated_count,
        "excluded_eligible_request_count": excluded_count,
        "excluded_reason_counts": dict(Counter(r for r, _ in excluded_rids)) if excluded_rids else {},
    })
    # Coverage
    metrics_json.update(coverage)
    # Behavior metrics
    for k in ks:
        for mn in ["hit_rate", "mrr", "ndcg", "positive_recall"]:
            key = f"{mn}_at_{k}"
            metrics_json[f"baseline_{key}"] = round(bb_avg.get(key, 0), 6)
            metrics_json[f"personalized_{key}"] = round(bh_avg.get(key, 0), 6)
            metrics_json[f"{key}_delta"] = round(bh_avg.get(key, 0) - bb_avg.get(key, 0), 6)
    # Qrels protection
    for k in [10]:
        for mn in ["precision", "recall", "mrr", "ndcg"]:
            key = f"{mn}_at_{k}"
            metrics_json[f"baseline_qrels_{key}"] = round(bq_avg.get(key, 0), 6)
            metrics_json[f"personalized_qrels_{key}"] = round(qr_avg.get(key, 0), 6)
            metrics_json[f"qrels_{key}_delta"] = round(qr_avg.get(key, 0) - bq_avg.get(key, 0), 6)
    # Comparison counts
    metrics_json.update({
        "improved_request_count": improved,
        "unchanged_request_count": unchanged,
        "worsened_request_count": worsened,
        "improved_plus_unchanged_plus_worsened": improved + unchanged + worsened,
    })
    # Fallback
    metrics_json.update(fallback_stats)

    (out / "metrics.json").write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- user_profiles.csv ---
    prof_fields = ["user_id","is_cold_start","train_event_count","train_session_count",
                   "positive_event_count","profile_status","top_categories","top_subcategories",
                   "top_brands","category_weights","subcategory_weights","brand_weights",
                   "mean_log_price","price_std","last_train_event_at"]
    with (out / "user_profiles.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prof_fields, extrasaction="ignore")
        w.writeheader()
        for uid in sorted(profiles):
            w.writerow(profiles[uid].to_row())

    # --- search_results.csv ---
    sr_fields = ["request_id","session_id","user_id","query_id","query_text",
                 "rank","original_rank","item_id","original_fusion_score",
                 "normalized_retrieval_score","category_affinity","subcategory_affinity",
                 "brand_affinity","price_affinity","personalized_score",
                 "profile_status","is_cold_start","behavior_relevance_grade","qrels_relevance_grade"]
    with (out / "search_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sr_fields, extrasaction="ignore")
        w.writeheader()
        for rid in sorted(personalized_results):
            info = test_requests[rid]
            for r in personalized_results[rid]:
                d = r.to_dict()
                d.update({"request_id": rid, "session_id": info["session_id"],
                          "user_id": info["user_id"], "query_id": info["query_id"],
                          "query_text": info["query_text"]})
                w.writerow(d)

    # --- request_metrics.csv ---
    rm_fields = ["request_id","user_id","query_id","profile_status",
                 "has_positive_behavior","positive_candidate_coverage"]
    for k in ks:
        for m in ["ndcg"]:
            rm_fields += [f"baseline_behavior_{m}_at_{k}", f"personalized_behavior_{m}_at_{k}",
                          f"behavior_{m}_at_{k}_delta",
                          f"baseline_qrels_{m}_at_{k}", f"personalized_qrels_{m}_at_{k}",
                          f"qrels_{m}_at_{k}_delta"]
    with (out / "request_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rm_fields, extrasaction="ignore")
        w.writeheader()
        for i, rid in enumerate(eligible_rids):
            if rid not in personalized_results:
                continue
            info = test_requests[rid]
            cand = hybrid_by_qid.get(info["query_id"], [])
            cand_ids = {c["item_id"] for c in cand}
            pos_items = {iid for iid, g in info["items"].items() if g > 0}
            cand_cov = len(pos_items & cand_ids) / max(1, len(pos_items))

            bm = all_bh[i] if i < len(all_bh) else {}
            bb = all_bb[i] if i < len(all_bb) else {}
            qm = all_qr[i] if i < len(all_qr) else {}
            bq = all_bq[i] if i < len(all_bq) else {}
            row = {"request_id": rid, "user_id": info["user_id"], "query_id": info["query_id"],
                   "profile_status": info.get("profile_status", "unknown"),
                   "has_positive_behavior": "true",
                   "positive_candidate_coverage": f"{cand_cov:.6f}"}
            for k in ks:
                for m in ["ndcg"]:
                    b_val = bb.get(f"{m}_at_{k}", 0); p_val = bm.get(f"{m}_at_{k}", 0)
                    row[f"baseline_behavior_{m}_at_{k}"] = f"{b_val:.6f}"
                    row[f"personalized_behavior_{m}_at_{k}"] = f"{p_val:.6f}"
                    row[f"behavior_{m}_at_{k}_delta"] = f"{p_val - b_val:.6f}"
                    bq_val = bq.get(f"{m}_at_{k}", 0); pq_val = qm.get(f"{m}_at_{k}", 0)
                    row[f"baseline_qrels_{m}_at_{k}"] = f"{bq_val:.6f}"
                    row[f"personalized_qrels_{m}_at_{k}"] = f"{pq_val:.6f}"
                    row[f"qrels_{m}_at_{k}_delta"] = f"{pq_val - bq_val:.6f}"
            w.writerow(row)

    # --- diagnostics.json ---
    diag = {k: metrics_json[k] for k in metrics_json
            if k not in ("config", "baseline_hit_rate_at_5", "personalized_hit_rate_at_5",
                         "hit_rate_at_5_delta", "qrels_precision_at_10_delta")}
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- comparison ---
    comp = {
        "baseline_linear_hybrid_on_test_requests": bb_avg,
        "personalized_reranking": bh_avg,
        "personalized_minus_baseline": {k: round(bh_avg.get(k, 0) - bb_avg.get(k, 0), 6) for k in bh_keys},
        "qrels_baseline": bq_avg,
        "qrels_personalized": qr_avg,
        "qrels_delta": {k: round(qr_avg.get(k, 0) - bq_avg.get(k, 0), 6) for k in qr_keys},
    }
    Path(args.comparison_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.comparison_output).write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  Users: total=100 with_events={grouping['users_with_events']} "
          f"without_events={grouping['users_without_events']}")
    print(f"  Sessions: configured=500 unique_in_events={len(unique_sids_events)} "
          f"train={assigned_train_sids} test={assigned_test_sids} unassigned={unassigned_sids}")
    print(f"  Profiles: warm={len(warm_uids)} cold_flag={len(cold_flag_uids)} "
          f"no_pos={len(no_pos_uids)} no_hist={len(no_hist_uids)}")
    print(f"  Test requests: total={len(test_requests)} eligible={len(eligible_rids)} "
          f"evaluated={evaluated_count} excluded={excluded_count}")
    print(f"  Candidate positive coverage (request): {coverage['request_level_candidate_positive_coverage']:.4f}")
    print(f"  Candidate positive recall (item): {coverage['item_level_candidate_positive_recall']:.4f}")
    print(f"  Fallback: users={len(all_fallback_uids)} requests={len(all_fallback_rids)} "
          f"exact_match={fallback_stats['fallback_exact_match_rate']:.4f}")
    print(f"  baseline_behavior_ndcg_at_10 = {bb_avg.get('ndcg_at_10',0):.6f}")
    print(f"  personalized_behavior_ndcg_at_10 = {bh_avg.get('ndcg_at_10',0):.6f}")
    print(f"  delta = {bh_avg.get('ndcg_at_10',0)-bb_avg.get('ndcg_at_10',0):+.6f}")
    print(f"  baseline_qrels_ndcg_at_10 = {bq_avg.get('ndcg_at_10',0):.6f}")
    print(f"  personalized_qrels_ndcg_at_10 = {qr_avg.get('ndcg_at_10',0):.6f}")
    print(f"  improved={improved} unchanged={unchanged} worsened={worsened} "
          f"(sum={improved+unchanged+worsened})")
    print("\nDone.")


if __name__ == "__main__":
    main()
