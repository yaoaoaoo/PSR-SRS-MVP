# PSR-SRS MVP — Personalized Search Ranking & Semantic Retrieval

E-commerce personalized search ranking and semantic retrieval MVP.
Built entirely with Python standard library for local execution.

## Project Goals

This MVP demonstrates the full pipeline from synthetic data generation through
multi-channel retrieval, user profiling, personalized re-ranking, cold-start
strategies, and offline evaluation — all running locally with no external
services.

## Current Phase — Data Foundation (Calibrated)

**This phase is complete:**

- Project skeleton
- Configurable synthetic data generator
- **Cascade browsing click model** with position decay and stop probability
- **Strict event dependencies** (purchase only after add_to_cart)
- **Qrels (relevance judgments)** based purely on query-item semantics
- Data quality validation (structural + statistical)
- 63 unit tests (stdlib `unittest`) — all passing
- CLI for fully reproducible data generation
- Calibrated behaviour funnel matching realistic e-commerce CTR and conversion rates

**Phase 2 Step 1 — BM25 Baseline (complete):**

- Deterministic tokenizer (NFKC + regex + stopwords)
- Okapi BM25 index with IDF, TF saturation, length normalisation
- Weighted field construction (title 3×, category 2×, etc.)
- Self-implemented evaluation: Precision@K, Recall@K, MRR@K, NDCG@K
- CLI: `scripts/run_bm25.py` with JSON/csv output
- 52 unit tests — all passing
- 500 docs, 76 unique vocabulary terms, 27.8 avg doc length
- 108/200 queries returned results; 92 empty (need semantic retrieval)

**Phase 2 Step 2 — LSA Semantic Retrieval (complete):**

- TF-IDF vectorization (word unigram+bigram, char 3-5 gram)
- TruncatedSVD/LSA latent space (64-d, 93.3% explained variance)
- Inductive setting: fit only on items; queries are transformed
- Cosine similarity semantic search
- Same evaluation: Precision@K, Recall@K, MRR@K, NDCG@K
- CLI: `scripts/run_semantic.py` with BM25 comparison output
- 43 unit tests — all passing
- **Query coverage: 99.5%** (vs BM25 54%), 1 zero-vector query
- **NDCG@10: 0.3733** (+0.0753 over BM25)

**Phase 2 Step 3 — Hybrid Fusion (complete):**

- Reciprocal Rank Fusion (RRF, k=60) and Linear (0.5/0.5, min-max norm)
- Candidate union from BM25 + LSA (candidate_k=100)
- Diagnostics: overlap analysis, per-query NDCG comparison
- CLI: `scripts/run_fusion.py` with unified 4-method comparison
- 43 unit tests — all passing (total: **201 tests**)
- **Linear NDCG@10: 0.3923** (+0.0943 vs BM25, +0.0190 vs LSA)
- **LSA dominates coverage (99.5%)**; BM25 rescues 0 additional queries
- Candidate Jaccard: 0.32 (complementary but LSA has near-full coverage)

**Phase 3 Step 1 — Personalized Re-ranking (complete):**

- Time-based session-level train/test split (80/20, no leakage)
- User profiles: weighted event × time decay → category/brand/price preferences
- Personalized re-ranking over Linear Hybrid Top-20 (0.70 retrieval + 0.30 affinity)
- Cold-start fallback: exact original order preservation
- Behavior metrics: HitRate, MRR, NDCG, Positive Recall
- Qrels correlation protection monitoring
- CLI: `scripts/run_personalization.py` with diagnostics
- 36 unit tests — all passing (total: **237 tests**)
- **Qrels NDCG@10: 0.3968** (+0.0049 vs unpersonalized baseline)
- **Behavior NDCG@10 delta: +0.0003** (minimal — candidate positive coverage only 13.85%)
- Full diagnostics: 100-user grouping, session accounting, fallback verification, candidate coverage
- See `docs/personalized_reranking.md` for complete results and analysis

**Not yet implemented:**

- Semantic vector retrieval (local embeddings)
- Multi-channel fusion
- User profile construction
- Personalized re-ranking
- Cold-start strategies
- Jupyter Notebook demonstrations
- FastAPI service
- Database integration

## Data Entities

| Entity | File          | Description                              |
|--------|---------------|------------------------------------------|
| Items  | `items.csv`   | 500 products across 10 categories        |
| Users  | `users.csv`   | 100 users with preferences and activity  |
| Queries| `queries.csv` | 200 search queries of varied intent      |
| Events | `events.csv`  | ~6,400 search interaction events         |
| Qrels  | `qrels.csv`   | 10,076 query-item relevance judgments    |

### Item Fields

`item_id`, `title`, `description`, `category`, `subcategory`, `brand`,
`price`, `quality_score`, `popularity_score`, `is_cold_start`, `created_at`

### User Fields

`user_id`, `preferred_categories`, `preferred_brands`, `price_preference`,
`activity_level`, `is_cold_start`, `created_at`

### Query Fields

`query_id`, `query_text`, `intended_category`, `semantic_intent`, `created_at`

### Event Fields

`event_id`, `event_type`, `request_id`, `session_id`, `user_id`, `query_id`,
`query_text`, `item_id`, `position`, `timestamp`, `click_duration_ms`,
`add_to_cart_quantity`, `purchase_amount`

Event types: `impression`, `click`, `favorite`, `add_to_cart`, `purchase`

### Qrels Fields

`query_id`, `item_id`, `relevance_grade` (1, 2, or 3; sparse — only grade > 0 stored)

## Data Generation Model

### Cascade Browsing Click Model

- User browses SERP position 1 → N sequentially
- Click probability: `base_click / pos^alpha × relevance_boost × preference_boost`
- After each click, user may stop browsing (`post_click_stop_probability`)
- Hard cap: `max_clicks_per_request` (default 3)
- Cold-start users get relevance boost only (no historical preference signal)

### Event Dependencies (Strict)

```
impression → click → favorite           (after click only)
                   → add_to_cart         (after click only)
                       → purchase        (after add_to_cart only)
```

### Funnel Statistics (Sample Data, seed=20260614)

| Metric                    | Value  | Target    |
|---------------------------|--------|-----------|
| impression-level CTR      | 9.2%   | 8–20%     |
| avg clicks / session      | 1.09   | 1–3       |
| favorite / click          | 8.9%   | 5–15%     |
| add_to_cart / click       | 7.6%   | 4–12%     |
| purchase / click          | 1.5%   | 1–5%      |
| purchase / add_to_cart    | 20.0%  | 15–40%    |

### Qrels (Relevance Judgments)

- Generated **independently** of user events — no click/position/popularity leakage
- Based purely on objective signals: category, subcategory, brand, keyword overlap
- Sparse format: only relevance_grade 1, 2, 3 are stored; absent = grade 0
- Every query guaranteed at least one grade 2+ item
- **Independent of `num_sessions`** — same qrels regardless of event volume

## Project Structure

```
PSR-SRS-MVP/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── sample.json              # Default generation config
├── docs/
│   └── synthetic_data_design.md # Detailed design doc
├── data/
│   └── sample/                  # Generated CSVs
├── scripts/
│   └── generate_data.py         # CLI entry point
├── src/
│   └── psr_srs_mvp/
│       ├── __init__.py
│       └── data_generation/
│           ├── __init__.py
│           ├── config.py        # Config dataclass & loader
│           ├── schemas.py       # Field name constants
│           ├── generator.py     # Core generation logic
│           ├── validation.py    # Data quality checks
│           └── writers.py       # CSV read/write
└── tests/
    ├── __init__.py
    └── test_data_generation.py  # Unit tests
```

## Environment

- **Python**: 3.12.6
- **Virtual environment**: `.venv/` (created with `python -m venv`)
- **Dependencies**: Python standard library only (no `pip install` required for data generation)
- **Planned future deps**: numpy, scipy, scikit-learn, sentence-transformers, rank-bm25

## Notebook

The end-to-end MVP Notebook (`notebooks/01_mvp_end_to_end.ipynb`) has been
**executed and verified** in both cache and recompute modes.

### Install notebook dependencies

```bash
.venv/Scripts/python.exe -m pip install jupyter ipykernel nbconvert nbformat nbclient
```

### Run interactively

```bash
.venv/Scripts/python.exe -m jupyter notebook notebooks/01_mvp_end_to_end.ipynb
```

### Execute headless (cache mode)

```bash
.venv/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/01_mvp_end_to_end.ipynb --output 01_mvp_end_to_end.executed.ipynb --output-dir outputs/notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3
```

### Execute headless (recompute mode — rebuilds all indices)

```bash
set PSR_SRS_RECOMPUTE=1
.venv/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/01_mvp_end_to_end.ipynb --output 01_mvp_end_to_end.recomputed.ipynb --output-dir outputs/notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3
```

**Execution status**: Both modes executed successfully — 0 errors, 16/16 code cells, all metrics match frozen baselines.

### Notebook output files

| File | Mode |
|------|------|
| `outputs/notebook/01_mvp_end_to_end.executed.ipynb` | Cache (pre-computed outputs) |
| `outputs/notebook/01_mvp_end_to_end.recomputed.ipynb` | Recompute (rebuilt indices) |
| `outputs/notebook/notebook_execution_report.json` | Validation report |

### 1. Activate virtual environment

```bash
.venv/Scripts/python.exe --version   # verify Python 3.12
```

### 2. Generate sample data

```bash
.venv/Scripts/python.exe scripts/generate_data.py \
    --config configs/sample.json \
    --output data/sample
```

### 3. Validate data quality

```bash
.venv/Scripts/python.exe scripts/validate_data.py \
    --data-dir data/sample \
    --statistics --manifest \
    --output outputs/data_generation/data_quality_report.json
```

### 4. Run tests

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

### 5. Run retrieval baselines

```bash
# BM25
.venv/Scripts/python.exe scripts/run_bm25.py --items data/sample/items.csv \
    --queries data/sample/queries.csv --qrels data/sample/qrels.csv \
    --config configs/bm25.json --output outputs/bm25

# LSA Semantic
.venv/Scripts/python.exe scripts/run_semantic.py --items data/sample/items.csv \
    --queries data/sample/queries.csv --qrels data/sample/qrels.csv \
    --config configs/semantic.json --bm25-metrics outputs/bm25/metrics.json \
    --output outputs/semantic --comparison-output outputs/comparison/bm25_vs_semantic.json

# Hybrid Fusion
.venv/Scripts/python.exe scripts/run_fusion.py --items data/sample/items.csv \
    --queries data/sample/queries.csv --qrels data/sample/qrels.csv \
    --bm25-config configs/bm25.json --semantic-config configs/semantic.json \
    --fusion-config configs/fusion.json --bm25-metrics outputs/bm25/metrics.json \
    --semantic-metrics outputs/semantic/metrics.json --output outputs/hybrid \
    --comparison-output outputs/comparison/retrieval_methods.json

# Personalized Re-ranking
.venv/Scripts/python.exe scripts/run_personalization.py --items data/sample/items.csv \
    --users data/sample/users.csv --events data/sample/events.csv \
    --qrels data/sample/qrels.csv --hybrid-results outputs/hybrid/linear/search_results.csv \
    --config configs/personalization.json --output outputs/personalization \
    --comparison-output outputs/comparison/hybrid_vs_personalized.json
```

## Output Files

After generation, `data/sample/` contains:

- `items.csv` — product catalog
- `users.csv` — user profiles
- `queries.csv` — search queries
- `events.csv` — interaction events

## Relationship to Enterprise Project

This MVP is a **standalone, simplified** project. The Enterprise-level
counterpart lives at `D:/project/PSR-SRS` and includes:

- FastAPI service layer
- PostgreSQL + SQLAlchemy ORM
- Alembic migrations
- Docker containerization
- Redis, OpenSearch, Qdrant integration
- Ruff, mypy, pytest CI

The MVP intentionally avoids these dependencies to enable rapid local
experimentation with retrieval and ranking algorithms.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
