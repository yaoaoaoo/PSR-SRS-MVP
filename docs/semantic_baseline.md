# LSA Semantic Retrieval Baseline — PSR-SRS MVP

> Date: 2026-06-14 | Phase 2, Step 2

## 1. Overview

LSA (Latent Semantic Analysis) provides a **dense vector** alternative to
BM25 keyword matching. By projecting documents and queries into a shared
low-dimensional latent space, LSA captures **implicit semantic relationships**
that surface-level token overlap cannot.

## 2. Method

### Pipeline

```
Item text
  → word TF-IDF (unigram + bigram)
  → char TF-IDF (3–5 gram, word-boundary)
  → weighted concatenation (word × 1.0 + char × 0.5)
  → TruncatedSVD (64 components)
  → L2 normalisation
  → item vectors

Query text → same TF-IDF/SVD pipeline → query vector

Cosine similarity(query, item) → Top-K results
```

### Inductive Setting

- TF-IDF vocabulary: **fit only on item documents**
- TruncatedSVD: **fit only on item matrix**
- Queries: **transform only** (no re-fitting)
- Qrels: **not used** in any fitting step
- Evaluation: **post-retrieval**, using existing BM25 eval module

This is a strict inductive unsupervised setting. No query text participates
in vocabulary or SVD fitting.

## 3. Configuration

| Parameter | Value |
|-----------|-------|
| word n-gram | [1, 2] |
| char n-gram | [3, 5] |
| word weight | 1.0 |
| char weight | 0.5 |
| sublinear TF | true |
| SVD components (req) | 64 |
| SVD components (actual) | 64 |
| random_state | 20260614 |

## 4. Vectorization Statistics

| Statistic | Value |
|-----------|-------|
| Word features | 603 |
| Char features | 1,386 |
| Combined features | 1,989 |
| SVD actual dim | 64 |
| Explained variance | 93.31% |
| Item vectors | 500 × 64 |
| Query vectors | 200 × 64 |

## 5. Item Text

LSA uses **raw field concatenation** (each field once):

```
title + description + category + subcategory + brand
```

This differs from BM25, which uses weighted field repetition
(title ×3, category ×2, etc.) for TF emphasis. The raw text
is a fairer input for TF-IDF, which has its own IDF weighting.

## 6. Results

| Metric | @5 | @10 | @20 |
|--------|-----|------|------|
| Precision | 0.5080 | 0.4980 | 0.4870 |
| Recall | 0.0505 | 0.0991 | 0.1938 |
| MRR | 0.5108 | 0.5163 | 0.5189 |
| NDCG | 0.3624 | 0.3733 | 0.3828 |

| Statistic | Value |
|-----------|-------|
| Query coverage | **99.5%** (199/200) |
| Zero-vector queries | 1 |
| No-result queries | 1 |

## 7. Comparison with BM25

| Metric | BM25 | LSA | Δ |
|--------|------|-----|---|
| Precision@10 | 0.3800 | 0.4980 | **+0.1180** |
| Recall@20 | 0.1506 | 0.1938 | **+0.0432** |
| MRR@10 | 0.4399 | 0.5163 | **+0.0763** |
| NDCG@10 | 0.2980 | 0.3733 | **+0.0753** |
| Query coverage | 54.0% | 99.5% | **+45.5pp** |
| No-result queries | 92 | 1 | **−91** |

LSA outperforms BM25 on **every metric** while reducing empty queries
from 92 to 1. The character n-gram TF-IDF is particularly effective
at handling the synthetic product vocabulary.

## 8. Advantages of LSA

1. **Query coverage**: 99.5% vs BM25's 54% — character n-grams match
   partial brand/category tokens that BM25's word-level regex cannot

2. **Soft matching**: Even when exact tokens differ, LSA places
   semantically similar items nearby in latent space

3. **No vocabulary bottleneck**: 1,989 combined features → 64 latent
   dimensions capture multi-token patterns

4. **Deterministic**: Fixed random_state ensures reproducibility

## 9. Limitations

1. **Not a neural embedding**: LSA is a linear decomposition of TF-IDF
   co-occurrence. It cannot capture deep semantic relationships that
   transformer models (BERT, Sentence-BERT) would.

2. **Cold vocabulary**: New brands or categories not seen during
   fitting produce near-zero vectors. The 1 zero-vector query
   contains only unknown terms.

3. **Linear only**: SVD captures linear co-occurrence patterns.
   Non-linear relationships (e.g., analogies) are not modeled.

4. **No cross-lingual support**: TF-IDF operates on surface forms only.

5. **Fixed dimensionality**: 64 components may be suboptimal; no
   hyperparameter search was performed (by design — to avoid
   overfitting to qrels).

## 10. Fairness of Comparison

| Aspect | BM25 | LSA |
|--------|------|-----|
| Item fields | title, desc, cat, subcat, brand | Same, but raw (no weight repetition) |
| Query fields | query_text | query_text |
| Qrels in fit? | No | No |
| Evaluation module | Same | Same |
| relevance_threshold | 1 | 1 |
| top_k | 5, 10, 20 | 5, 10, 20 |

The comparison is fair: both methods use identical item/query text sources,
identical qrels, and identical evaluation code. BM25 uses weighted field
repetition (its algorithmic advantage); LSA uses TF-IDF + SVD (its
algorithmic advantage).

## 11. Next Steps

**Hybrid fusion** (BM25 + LSA) is the logical next step. BM25 provides
exact-match precision on known tokens; LSA provides recall on semantically
related but lexically distant items. Combining both via score fusion (e.g.,
RRF or linear interpolation) should exceed either baseline alone.

## 12. Reproducibility

- `random_state=20260614` for TruncatedSVD
- TF-IDF is fully deterministic (no random component)
- Same items + queries + config → bit-identical output files
- 43 unit tests verify pipeline determinism
