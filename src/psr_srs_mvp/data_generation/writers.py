"""CSV read / write helpers for synthetic data.

Uses ``csv.DictWriter`` with UTF-8 encoding and consistent quoting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from psr_srs_mvp.data_generation.schemas import (
    EVENT_FIELDS,
    ITEM_FIELDS,
    QRELS_FIELDS,
    QUERY_FIELDS,
    USER_FIELDS,
)


def write_csv_files(data: dict[str, list[dict[str, str]]], output_dir: str | Path) -> list[Path]:
    """Write entity records to CSV files in *output_dir*.

    Args:
        data: Dict with keys ``"items"``, ``"users"``, ``"queries"``, ``"events"``,
              each value a list of dict rows.
        output_dir: Directory to write to (created if needed).

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, list[str]]] = [
        ("items.csv", ITEM_FIELDS),
        ("users.csv", USER_FIELDS),
        ("queries.csv", QUERY_FIELDS),
        ("events.csv", EVENT_FIELDS),
        ("qrels.csv", QRELS_FIELDS),
    ]

    written: list[Path] = []
    for filename, fieldnames in specs:
        key = filename.replace(".csv", "")
        rows = data.get(key, [])
        path = out / filename
        _write_csv(path, fieldnames, rows)
        written.append(path)

    return written


def read_csv_files(data_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    """Read back generated CSVs into the same dict shape used by generators.

    Returns:
        ``{"items": [...], "users": [...], "queries": [...], "events": [...]}``
    """
    base = Path(data_dir)
    result: dict[str, list[dict[str, str]]] = {}
    for key in ("items", "users", "queries", "events", "qrels"):
        path = base / f"{key}.csv"
        if path.exists():
            result[key] = _read_csv(path)
        else:
            result[key] = []
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
