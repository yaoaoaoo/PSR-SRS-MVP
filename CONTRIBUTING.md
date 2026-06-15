# Contributing to PSR-SRS MVP

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev,notebook]"
```

## Running Tests

```bash
python -m pytest -q
```

All 255 tests must pass before submitting changes.

## Frozen Baselines

The following must **not** be changed without explicit discussion and
downstream re-validation:

- `data/sample/*.csv` — sample data
- `configs/*.json` — algorithm configurations
- BM25, LSA, Hybrid, or Personalization algorithms
- Ranking weights and parameters
- Qrels or evaluation metric definitions
- Candidate set generation logic

## Before Submitting

- [ ] `python -m pytest -q` passes
- [ ] `python -m compileall src scripts tests` passes
- [ ] `python -m pip check` passes
- [ ] `python scripts/validate_data.py --data-dir data/sample` passes (66/66)
- [ ] Sample data SHA-256 hashes are unchanged
- [ ] No new dependencies added without updating `pyproject.toml`

## Scope

This MVP is a **local-only, research/demonstration project**. New features
that require:
- Online services (APIs, databases, message queues)
- Cloud deployment
- Multi-user support
- Real-time inference

should be developed in the Enterprise-level project instead.

## Code Style

- Python 3.12 type annotations
- Clear docstrings for public functions
- `pathlib.Path` for all file paths
- No hard-coded absolute paths
- No network access in tests or data generation
