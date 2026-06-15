# Release Checklist — v0.1.0

## Pre-Release Verification

- [x] All 255 tests pass
- [x] Data quality: 66/66 checks passed
- [x] Reproducibility: 5/5 SHA-256 matched
- [x] Notebook cache mode: 0 errors
- [x] Notebook recompute mode: 0 errors
- [x] Frozen metrics: 10/10 verified
- [x] compileall: OK
- [x] pip check: no broken requirements
- [x] Wheel build: successful
- [x] sdist build: successful
- [x] twine check: passed
- [x] Clean install smoke test: passed
- [x] **LICENSE file** — MIT License with copyright holder
- [ ] **Repository URL** — update pyproject.toml project.urls
- [x] **Author metadata** — wangwenyao
- [ ] **Git tag** — `git tag v0.1.0` (requires git init)
- [ ] **GitHub release** — create with release notes (requires git init)

## Release Assets

| Asset | Status |
|-------|--------|
| `.editorconfig` | ✓ |
| `.gitattributes` | ✓ |
| `.gitignore` | ✓ |
| `pyproject.toml` | ✓ (license placeholder) |
| `README.md` | ✓ |
| `CHANGELOG.md` | ✓ |
| `CONTRIBUTING.md` | ✓ |
| `SECURITY.md` | ✓ |
| `requirements-lock.txt` | ✓ |
| `docs/` (9 files) | ✓ |
| `notebooks/` (1 file) | ✓ |
| `LICENSE` | ✗ — release blocker |

## Blocking Items

1. **LICENSE** — Must be selected and added before public release.
2. **Repository URL** — Update `pyproject.toml` `project.urls.Repository`.
3. **Author metadata** — Verify `pyproject.toml` `authors` field.
