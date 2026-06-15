# Changelog

All notable changes to the PSR-SRS MVP project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-06-15

### Added

- Synthetic data generator with configurable entities: items (500), users (100),
  queries (200), events (~6,400), and qrels (10,076)
- Cascade browsing event model with strict click → add_to_cart → purchase dependencies
- Configurable cold-start user/item ratios and behavioral probability parameters
- Data quality validator: 66 automated checks (PK uniqueness, FK integrity, enums,
  numeric ranges, timestamps, business rules)
- Dual-run reproducibility verification: 5/5 CSV SHA-256 match
- BM25 keyword retrieval with Okapi weighting (k1=1.5, b=0.75)
- LSA semantic retrieval via TF-IDF + TruncatedSVD (64-d latent space, 93.3%
  explained variance)
- Linear Hybrid fusion: 0.5/0.5 weighted combination with min-max per-query normalization
- Reciprocal Rank Fusion (RRF, k=60)
- Personalized re-ranking: user profiles from weighted behavior + time decay,
  category/brand/price affinity scoring
- Cold-start fallback verification with 100% exact match rate
- Offline evaluation: Precision@K, Recall@K, MRR@K, NDCG@K (qrels-based)
- Behavior evaluation: HitRate@K, Behavioral MRR@K, NDCG@K, Positive Recall@K
- Candidate positive coverage diagnostics (request-level and item-level)
- End-to-end Notebook (44 cells, 18 chapters) with cache and recompute modes
- Comprehensive test suite: 255 tests covering data generation, retrieval,
  fusion, personalization, and notebook execution
- Reproducibility evidence: data quality report, reproducibility report,
  data manifest, execution report

### Changed

- MIT License applied (copyright: wangwenyao)

### Fixed

- N/A (initial release)

### Validation

- All 255 tests pass
- Data quality: 66/66 checks passed
- Reproducibility: 5/5 CSV SHA-256 matched
- Notebook cache mode: 16/16 cells executed, 0 errors
- Notebook recompute mode: 16/16 cells executed, 0 errors
- Frozen baselines: 10/10 metrics verified

[0.1.0]: https://github.com/owner/psr-srs-mvp/releases/tag/v0.1.0
