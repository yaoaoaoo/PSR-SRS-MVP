# Personalized Re-ranking — PSR-SRS MVP

> Date: 2026-06-14 | Phase 3, Step 1

## 1. Data Split

**Method**: Per-user session-level chronological split (80/20).

For each user with ≥2 sessions:
1. Sort sessions by earliest timestamp
2. Last `max(1, floor(count × 0.2))` sessions → test
3. Remaining → train

Single-session users → train only (no test evaluation).
Same request_id and session_id are never split across train/test.

| Statistic | Value |
|-----------|-------|
| Configured sessions | 500 |
| Unique sessions in events.csv | 484 |
| Sessions with zero events | 16 |
| Train sessions | 400 |
| Test sessions | 84 |
| Unassigned (events.csv) | 0 |
| Time leakage | **None** |

## 2. User Grouping

Mutually exclusive groups covering all 100 users:

| Group | Count |
|-------|-------|
| Multi-session (≥2) | 65 |
| Single-session (1) | 22 |
| Zero-session (no events) | 13 |
| **Total** | **100** |

| Profile status | Count |
|----------------|-------|
| Warm (has positive train events) | 81 |
| Cold-start (flagged in users.csv) | 4 |
| No positive behavior (events but no click/fav/atc/pur) | 3 |
| No history (zero sessions) | 12 |
| **Total** | **100** |

## 3. User Profiles

Built exclusively from **training events**:

| Signal | Method |
|--------|--------|
| Event types | click (w=1), favorite (w=2), add_to_cart (w=3), purchase (w=5) |
| Time decay | `0.5^(age_days / 30)` from last train event |
| Category/brand weights | Weighted event sum → L1 normalized |
| Price preference | Weighted log-price mean; L2 std |

**Impression events do not contribute to profiles.**

`users.csv` fields `preferred_categories`, `preferred_brands`, `price_preference`
are **never read** for scoring — they are oracle labels that would leak ground truth.

## 4. Personalized Re-ranking Formula

```
personalized_score =
  0.70 × normalized_retrieval_score
+ 0.12 × category_affinity
+ 0.06 × subcategory_affinity
+ 0.06 × brand_affinity
+ 0.06 × price_affinity
```

- Retrieval score: per-query min-max normalized from Linear Hybrid fusion_score
- Category/subcategory/brand affinity: normalized profile weight for item's attribute (0 if absent)
- Price affinity: `exp(-|log(price) - mean_log_price| / max(std, 0.1))`

**Candidates are fixed**: Linear Hybrid Top-20 from Phase 2.3. Re-ranking only
changes item order, never adds/removes candidates.

## 5. Cold-Start Fallback

Users with `profile_status ∈ {cold_start, no_history, empty, no_positive}`
receive **exact Linear Hybrid original order**.

| Fallback reason | Users affected |
|-----------------|---------------|
| Cold-start flag | 4 |
| No train session | 12 |
| No positive behavior | 3 |

Fallback requests: 1 (with exact match verified).

## 6. Candidate Positive Coverage

The Linear Hybrid Top-20 candidate set is the **upper bound** of what
personalization can achieve. If positive items are not in the candidate
set, no re-ranking can surface them.

| Metric | Value |
|--------|-------|
| Request-level coverage | **0.1385** (9/65 eligible requests) |
| Item-level recall | **0.1190** (17/143 positive items) |

**Only 13.85% of test requests have at least one positive item in their Top-20 candidates.**
This severely limits potential personalization gains.

## 7. Behavior Metrics

Evaluated on 64 eligible test requests (1 excluded for missing results).

| Metric | Baseline | Personalized | Δ |
|--------|----------|--------------|------|
| HitRate@5 | 0.0938 | 0.0938 | 0.0000 |
| HitRate@10 | 0.0938 | 0.0938 | 0.0000 |
| HitRate@20 | 0.1094 | 0.1094 | 0.0000 |
| MRR@5 | 0.0313 | 0.0313 | 0.0000 |
| MRR@10 | 0.0313 | 0.0313 | 0.0000 |
| MRR@20 | 0.0344 | 0.0344 | 0.0000 |
| NDCG@5 | 0.0221 | 0.0221 | 0.0000 |
| **NDCG@10** | **0.0261** | **0.0265** | **+0.0003** |
| NDCG@20 | 0.0339 | 0.0341 | +0.0002 |
| Pos Recall@5 | 0.0156 | 0.0156 | 0.0000 |
| Pos Recall@10 | 0.0156 | 0.0156 | 0.0000 |
| Pos Recall@20 | 0.0234 | 0.0234 | 0.0000 |

**improved=2, unchanged=61, worsened=1** (NDCG@10, tolerance=1e-12)

## 8. Qrels Protection

| Metric | Baseline | Personalized | Δ |
|--------|----------|--------------|------|
| Precision@10 | 0.4078 | 0.4082 | +0.0004 |
| Recall@20 | 0.2090 | 0.2096 | +0.0006 |
| MRR@10 | 0.5320 | 0.5396 | +0.0076 |
| NDCG@10 | 0.3919 | 0.3968 | +0.0049 |

Qrels relevance is preserved. The small improvement is coincidental — items
that score higher on user affinity also happen to have slightly better qrels
grades in this dataset.

## 9. Why Personalized Gain Is Minimal

1. **Candidate coverage only 13.85%**: Most positive items are not in the
   Linear Hybrid Top-20. Re-ranking cannot create new candidates.

2. **Synthetic data sparsity**: 624 positive events across 81 warm users
   spread over 3 months. Average ~7.7 positive events per warm user.

3. **Single positive per request**: Most test requests have only 1 positive
   item (click). With 1 positive out of 20 candidates, the expected rank
   improvement from re-ranking is small.

4. **Retrieval weight 0.70**: The retrieval score dominates the personalized
   score. Affinity signals are dampened by the 0.30 total weight.

5. **Fixed weights**: 0.70/0.30 split is not tuned per user or query type.

## 10. What This Method Is Not

- **Not Learning to Rank**: No machine-learned ranking function; weights are
  fixed and hand-set.
- **Not using oracle labels**: `preferred_categories`, `preferred_brands`,
  `price_preference` from users.csv are never read.
- **Not collaborative filtering**: No user-user or item-item similarity.
- **Not session-based recommendation**: Uses only historical event aggregates.

## 11. Next Step — Notebook

A Jupyter Notebook should:
1. Walk through the full pipeline: data → BM25 → LSA → Hybrid → Personalized
2. Display key metrics and diagnostics visually
3. Show candidate overlap and coverage analysis
4. Illustrate user profile construction with examples
5. Demonstrate cold-start fallback behavior
6. Quantify the recall ceiling imposed by candidate coverage
