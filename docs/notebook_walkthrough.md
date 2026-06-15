# Notebook Walkthrough — 01_mvp_end_to_end.ipynb

The main Notebook has **44 cells** across **18 chapters** (numbered 0–17).

## Chapter Index

| # | Chapter | What It Shows |
|---|---------|---------------|
| 0 | Project Introduction | Pipeline overview, synthetic data notice |
| 1 | Environment & Configuration | Python version, seed, RECOMPUTE switch |
| 2 | Data Loading | All 5 CSV tables via official loaders |
| 3 | Data Quality Summary | 66 checks, reproducibility, SHA-256 manifest |
| 4 | Data Distribution | Users, sessions, events, categories |
| 5 | Train/Test Split | Time-based split, leakage check, example user |
| 6 | BM25 Retrieval | Cached results, example query, empty queries |
| 7 | LSA Semantic Retrieval | Coverage, BM25-failure rescue case |
| 8 | BM25 vs LSA Comparison | 4-metric comparison table |
| 9 | Linear Hybrid Fusion | Formula, 3-way benchmark comparison |
| 10 | User Profiles | 81 warm / 4 cold / 3 no_pos / 12 no_hist |
| 11 | Personalized Re-ranking | Improved case with before/after table |
| 12 | Behavior Metrics | HitRate, MRR, NDCG, Positive Recall |
| 13 | Qrels Protection | Verified no relevance degradation |
| 14 | Candidate Coverage | **Key diagnostic**: 9/65 request coverage |
| 15 | Fallback Verification | 100% exact match confirmed |
| 16 | Full Model Comparison | 4-stage NDCG summary table |
| 17 | Conclusions & Limitations | 7 findings + 10 limitations |

## Execution Modes

- **Cache mode** (`RECOMPUTE=False`, default): Reads pre-computed outputs. Fast.
- **Recompute mode** (`PSR_SRS_RECOMPUTE=1`): Rebuilds all indices via official modules.

## Key Outputs

- `outputs/notebook/01_mvp_end_to_end.executed.ipynb`
- `outputs/notebook/01_mvp_end_to_end.recomputed.ipynb`
- `outputs/notebook/notebook_execution_report.json`
