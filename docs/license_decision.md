# License Decision — PSR-SRS MVP

## Final Decision

**MIT License** — selected on 2026-06-15.

## Rationale

- **Simple and permissive**: Allows use, modification, distribution, and
  commercial use with minimal restrictions.
- **Widely adopted**: The most common open-source license; well understood
  by contributors and users.
- **Compatible**: Works with the project's dependencies (scikit-learn,
  numpy, scipy — all BSD/MIT-compatible).

## Requirements

Users must retain the copyright notice and permission statement in all
copies or substantial portions of the software.

## Third-Party Dependencies

The MIT License applies only to the PSR-SRS MVP project code. Third-party
packages (scikit-learn, numpy, scipy, joblib, etc.) are governed by their
own licenses. Users are responsible for compliance with all applicable
third-party licenses.

## Copyright

```
Copyright (c) 2026 wangwenyao
```

## Verification

- `LICENSE` file at repository root: ✓
- `pyproject.toml` SPDX identifier: `MIT`
- `license-files = ["LICENSE"]`: ✓
- Wheel metadata `License-Expression: MIT`: ✓
- `README.md` License section with link: ✓
- `release_check.py` validation: ✓
