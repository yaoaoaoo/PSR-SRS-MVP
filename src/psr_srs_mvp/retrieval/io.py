"""CSV loaders for items, queries, and qrels.

Reads the sample data CSVs produced by the data generation module.
All functions return plain dicts/lists — no ORM or database dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_items(path: str | Path) -> list[dict[str, str]]:
    """Load items.csv → list of dicts with all columns as strings."""
    return _read_csv(path)


def load_queries(path: str | Path) -> list[dict[str, str]]:
    """Load queries.csv → list of dicts with all columns as strings."""
    return _read_csv(path)


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load qrels.csv → ``{query_id: {item_id: relevance_grade}}``.

    Relevance grades are parsed as ints (1, 2, or 3).
    Rows with invalid grades raise ``ValueError``.
    Duplicate ``(query_id, item_id)`` pairs raise ``ValueError``.
    """
    rows = _read_csv(path)
    qrels: dict[str, dict[str, int]] = {}
    for r in rows:
        qid = r["query_id"].strip()
        iid = r["item_id"].strip()
        try:
            grade = int(r["relevance_grade"])
        except (ValueError, KeyError):
            raise ValueError(
                f"Invalid relevance_grade in qrels row: {r}"
            ) from None
        if grade not in (1, 2, 3):
            raise ValueError(f"relevance_grade must be 1, 2, or 3, got {grade}")
        if qid not in qrels:
            qrels[qid] = {}
        if iid in qrels[qid]:
            raise ValueError(f"Duplicate qrels entry: query={qid} item={iid}")
        qrels[qid][iid] = grade
    return qrels


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
