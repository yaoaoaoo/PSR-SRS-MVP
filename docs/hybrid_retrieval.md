# Hybrid Retrieval (BM25 + LSA) — PSR-SRS MVP

> Date: 2026-06-14 | Phase 2, Step 3

## 1. Motivation

BM25 and LSA capture fundamentally different relevance signals:

- **BM25**: Exact keyword matching via TF-IDF statistics. Strong precision on
  known tokens; zero recall when query terms are absent from the corpus.
- **LSA**: Latent semantic matching via TF-IDF → TruncatedSVD. Handles
  vocabulary mismatch through co-occurrence patterns; weaker at exact-match
  precision.

Combining both through score fusion aims to leverage their complementarity.

## 2. Candidate Generation

For each query, both BM25 and LSA retrieve `candidate_k=100` results:

```
BM25 top-100  \
                → union → deduplicate by item_id → fused candidates
LSA top-100   /
```

Each candidate records its source(s), original rank, and original score from
both retrieval channels.

## 3. Reciprocal Rank Fusion (RRF)

```
RRF(d) = Σ 1 / (k + rank_r(d))
```

Where `k = 60` (standard smoothing constant).

For a two-channel system:

```
RRF(d) = (1/(60+bm25_rank) if in BM25 else 0)
       + (1/(60+semantic_rank) if in LSA else 0)
```

**Why RRF doesn't need score normalisation**: RRF operates on ranks, not raw
scores. Ranks are scale-free, so BM25's unbounded positive scores and LSA's
[-1, 1] cosine scores don't need harmonisation.

## 4. Weighted Linear Fusion

```
linear(d) = w_bm25 × norm_bm25(d) + w_sem × norm_sem(d)
```

Where `w_bm25 = 0.5`, `w_sem = 0.5` (fixed, no tuning).

### Min-Max Normalisation

Per-query, per-channel:

```
norm(d) = (score(d) - min) / (max - min)     if max > min
norm(d) = 1.0                                 if all equal or single candidate
norm(d) = 0.0                                 if missing from channel
```

## 5. Results

| Metric | BM25 | LSA | RRF | Linear |
|--------|------|-----|-----|--------|
| Precision@10 | 0.3800 | 0.4980 | 0.4634 | 0.4783 |
| Recall@20 | 0.1506 | 0.1938 | 0.2003 | 0.2005 |
| MRR@10 | 0.4399 | 0.5163 | 0.5202 | 0.5347 |
| NDCG@10 | 0.2980 | 0.3733 | 0.3847 | 0.3923 |
| Query coverage | 54.0% | 99.5% | 99.5% | 99.5% |

### Deltas vs Baselines

| Metric | RRF−BM25 | RRF−LSA | Linear−BM25 | Linear−LSA |
|--------|----------|---------|-------------|------------|
| NDCG@10 | +0.0867 | +0.0114 | +0.0943 | +0.0190 |
| Recall@20 | +0.0497 | +0.0065 | +0.0499 | +0.0067 |

## 6. Candidate Diagnostics

| Statistic | Value |
|-----------|-------|
| Avg BM25 candidates (of 100) | 39.5 |
| Avg LSA candidates (of 100) | 99.5 |
| Avg union | 104.5 |
| Avg intersection | 34.5 |
| Avg Jaccard overlap | 0.317 |
| Queries BM25-empty | 92 |
| Queries rescued by LSA | 91 |

**Key insight**: LSA dominates the candidate pool (99.5 avg candidates).
The low Jaccard (0.317) confirms BM25 and LSA retrieve substantially
different item sets, but LSA's near-universal coverage means fusion
adds limited novel candidates beyond what LSA already provides.

## 7. Query-Level Analysis

| Comparison | RRF | Linear |
|------------|-----|--------|
| Better than BM25 | 47 queries | 41 queries |
| Equal to BM25 | 138 | 139 |
| Worse than BM25 | 15 | 20 |
| Better than LSA | 19 | 22 |
| Equal to LSA | 160 | 163 |
| Worse than LSA | 21 | 15 |

Most queries show **equal** NDCG@10 across methods (ties). This reflects
the small vocabulary and limited result-set diversity in the synthetic
data. Where fusion helps, it's modest; where it hurts, it's also modest.

## 8. Why Not Parameter Tuning

This phase deliberately uses **fixed parameters** (rrf_k=60, weights=0.5/0.5)
without grid search, Bayesian optimisation, or qrels-driven tuning:

1. **Avoids overfitting**: With 200 queries and synthetic data, tuning would
   find parameters that exploit random seed artifacts.
2. **Honest baseline**: Fixed weights establish a reproducible lower-bound
   for what fusion can achieve.
3. **Future Learning to Rank**: Parameterised fusion is better handled by
   LTR models trained on the event data.

## 9. Limitations

1. **LSA-dominant**: With 99.5% LSA coverage, fusion primarily perturbs LSA
   rankings. BM25 contributes candidates for only 54% of queries.
2. **Synthetic vocabulary**: The 76-term BM25 vocabulary limits recall even
   at candidate_k=100.
3. **Fixed weights**: 0.5/0.5 is blind to query-level strength variations.
4. **No user personalisation**: All queries use the same fusion parameters
   regardless of user intent or history.

## 10. Next Steps

**User profiling and personalised re-ranking**: The next phase should:
1. Build user profiles from historical event data
2. Incorporate user category/brand preferences into ranking
3. Handle cold-start users with default or popularity-based fallback
4. Evaluate personalisation uplift against the unpersonalised hybrid baseline
