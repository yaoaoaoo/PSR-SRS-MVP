# Reproducibility Guide — PSR-SRS MVP

## Quick Summary

| Check | Result |
|-------|--------|
| Data generation (dual-run) | 5/5 SHA-256 matched |
| Data quality | 66/66 checks passed |
| Notebook cache mode | 16/16 cells, 0 errors |
| Notebook recompute mode | 16/16 cells, 0 errors |
| Frozen metrics | 10/10 verified |
| Tests | 255 passed |

## Environment

- Python: 3.12.6
- Platform: Windows 11
- Seed: 20260614

## Steps to Reproduce

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python -m pip install -e ".[notebook,release]"
```

### 2. Generate Data

```bash
python scripts/generate_data.py --config configs/sample.json --output data/sample --force
```

### 3. Validate Data

```bash
python scripts/validate_data.py --data-dir data/sample
# Expected: 66/66 passed, 0 errors, 0 warnings
```

### 4. Verify Reproducibility

```bash
python scripts/reproducibility_check.py
# Expected: 5/5 files SHA-256 matched
```

### 5. Run Tests

```bash
python -m pytest -q
# Expected: 255 passed
```

### 6. Run Full Pipeline

See README for BM25, LSA, Hybrid, and Personalization commands.

### 7. Execute Notebooks

```bash
# Cache mode
python -m nbconvert --to notebook --execute notebooks/01_mvp_end_to_end.ipynb --output 01_mvp_end_to_end.executed.ipynb --output-dir outputs/notebook --ExecutePreprocessor.timeout=600

# Recompute mode
set PSR_SRS_RECOMPUTE=1
python -m nbconvert --to notebook --execute notebooks/01_mvp_end_to_end.ipynb --output 01_mvp_end_to_end.recomputed.ipynb --output-dir outputs/notebook --ExecutePreprocessor.timeout=600
```

## Expected Metrics

| Metric | Value |
|--------|-------|
| BM25 NDCG@10 | 0.297994 |
| LSA NDCG@10 | 0.373320 |
| Linear Hybrid NDCG@10 | 0.392327 |
| Personalized qrels NDCG@10 | 0.396789 |

## Troubleshooting

- **Different seed?** Check `configs/sample.json` → `seed` must be `20260614`
- **Different platform?** Python `random.Random(seed)` is cross-platform deterministic
- **Package not found?** Run `pip install -e .` from the project root
