# BM25 Baseline — PSR-SRS MVP

> Date: 2026-06-14 | Phase 2, Step 1

## 1. Overview

This document describes the BM25 keyword retrieval baseline for the PSR-SRS
MVP. BM25 is a classic probabilistic retrieval model that serves as a
**non-semantic baseline** before introducing embedding-based retrieval.

## 2. BM25 Algorithm

### Formula

Standard Okapi BM25:

```
score(D,Q) = Σ idf(t) × tf(t,D) × (k1+1) / (tf(t,D) + k1 × (1 - b + b × |D|/avgdl))
```

### IDF

```
idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))
```

Non-negative for all terms (including those absent from the corpus).

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| k1 | 1.5 | Term frequency saturation |
| b | 0.75 | Document length normalisation |
| top_k | 5, 10, 20 | Result cutoffs |

## 3. Tokenizer

Deterministic tokenizer using only Python standard library:

1. **Unicode NFKC normalisation** — handles full-width and compatibility forms
2. **Lowercase** — case-insensitive matching
3. **Regex extraction** — `[a-z0-9]+` (letters and digits only)
4. **Stop-word removal** — 105 built-in English stop words (no network access)

### Stop Words

The stop-word list is moderate (105 words) and preserves:
- Brand names: "TechPro", "NovaDigital"
- Categories: "Electronics", "Clothing"
- Colours, sizes, attributes (not stripped as they're mixed-case proper nouns)

## 4. Item Document Construction

Weighted field repetition (not true BM25F):

| Field | Weight | Rationale |
|-------|--------|-----------|
| title | 3× | Most discriminative for relevance |
| category | 2× | Strong relevance signal |
| subcategory | 2× | Granularity signal |
| brand | 2× | Brand-matching signal |
| description | 1× | Supplementary signal |

Example: a "TechPro Smartphone" item produces text like:
```
"TechPro Smartphone TechPro Smartphone TechPro Smartphone ... Electronics Electronics ... Smartphones Smartphones ... TechPro TechPro ... <description>"
```

This is **not** true BM25F — it's a simple concatenation heuristic. BM25F would
apply per-field saturation parameters.

## 5. Qrels Usage

Qrels are used **exclusively for post-retrieval evaluation**. They are never
used for:
- Candidate filtering
- Score boosting
- Query expansion
- Parameter tuning per query
- Item text construction

## 6. Evaluation Metrics

All metrics are self-implemented (no `scikit-learn`).

### Precision@K

```
P@K = |{relevant in top-K}| / K
```

### Recall@K

```
R@K = |{relevant in top-K}| / |{all relevant for query}|
```

### MRR@K

```
MRR@K = 1 / rank_of_first_relevant
```

### NDCG@K

```
DCG@K  = Σ (2^rel_i - 1) / log2(i + 1)
IDCG@K = DCG@K for ideal ranking
NDCG@K = DCG / IDCG
```

### Macro Averaging

All per-query metrics are averaged (macro-average) across the 200 queries.

## 7. Results

**Run command:**

```bash
.venv/Scripts/python.exe scripts/run_bm25.py \
  --items data/sample/items.csv \
  --queries data/sample/queries.csv \
  --qrels data/sample/qrels.csv \
  --config configs/bm25.json \
  --output outputs/bm25
```

### Aggregate Metrics

| Metric | @5 | @10 | @20 |
|--------|-----|------|------|
| Precision | 0.3890 | 0.3800 | 0.3777 |
| Recall | 0.0388 | 0.0758 | 0.1506 |
| MRR | 0.4358 | 0.4399 | 0.4407 |
| NDCG | 0.3026 | 0.2980 | 0.3022 |

### Index Statistics

| Statistic | Value |
|-----------|-------|
| Documents | 500 |
| Unique terms | 76 |
| Avg doc length | 27.8 tokens |
| Queries with results | 108 / 200 (54%) |
| Queries empty | 92 / 200 (46%) |

## 8. Limitations

### Vocabulary Mismatch (Primary Issue)

The tokenizer produces only **76 unique terms** across 500 items. This is
because:

1. **Regex tokenization** (`[a-z0-9]+`) splits multi-word brand names like
   "FitGear Pro" into `["fitgear", "pro"]`, but "Pro" is a stop word →
   `["fitgear"]`. Queries containing "FitGear Pro" will not match.

2. **Ampersand symbols** like "Home & Kitchen" become `["home", "kitchen"]`
   (no "&"), losing the compound category signal.

3. **No stemming/lemmatisation** — "running" ≠ "run", "shoes" ≠ "shoe".

4. **No query expansion** — exact keyword match only.

### Why Semantic Retrieval Is Necessary

The 46% empty-query rate and low Recall@20 (15%) demonstrate that pure
keyword matching is insufficient for e-commerce search where:

- Queries use natural language ("affordable skincare for everyday use")
- Brand names contain special characters or spaces
- Categories span compound terms
- Users describe attributes not present in exact product text

Semantic vector retrieval (e.g., sentence-transformers) addresses these by
encoding both queries and items into a shared embedding space where
**meaning**, not exact token overlap, determines relevance.

## 9. Next Steps

1. Implement dense vector retrieval with local embeddings
2. Combine BM25 + vector into a hybrid/fusion ranker
3. Evaluate against the same qrels
4. Compare recall gains from semantic matching

## 10. Reproducibility

Same items + queries + config → bit-identical output files. All randomness in
the pipeline was removed in Phase 1 (data generation); BM25 is fully
deterministic.
