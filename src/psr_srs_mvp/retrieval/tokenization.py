"""Deterministic text tokenization for BM25 retrieval.

Uses only Python standard library — no NLTK, spaCy, or network access.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Built-in English stop-word list (moderate, preserves brand/category terms)
# ---------------------------------------------------------------------------
STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "as", "into", "up", "out", "about", "over", "under", "also",
    "very", "just", "each", "all", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "well", "rather",
    "between", "during", "before", "after", "above", "below",
}

# Pre-compiled token pattern: lowercase letters + digits
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize *text* into a list of lower-case alpha-numeric tokens.

    Steps:
    1. Unicode NFKC normalisation
    2. Lower-case
    3. Regex extraction of ``[a-z0-9]+``
    4. Optional stop-word removal

    Args:
        text: Raw input string.
        remove_stopwords: If True, filter out tokens in ``STOP_WORDS``.

    Returns:
        Deterministic token list (may be empty).
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text)
    lowered = normalized.lower()
    tokens = _TOKEN_RE.findall(lowered)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def build_item_text(
    title: str,
    description: str,
    category: str,
    subcategory: str,
    brand: str,
    weights: dict[str, int] | None = None,
) -> str:
    """Construct a weighted text representation of an item for BM25 indexing.

    *weights* is a dict mapping field names to repeat counts.  Default::

        {"title": 3, "description": 1, "category": 2, "subcategory": 2, "brand": 2}

    Returns a single string where weighted fields are repeated.
    """
    if weights is None:
        weights = {"title": 3, "description": 1, "category": 2, "subcategory": 2, "brand": 2}

    parts: list[str] = []
    for field, w in weights.items():
        if w <= 0:
            continue
        value: str = {
            "title": title,
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
        }.get(field, "")
        if value:
            parts.append(" ".join([value] * w))
    return " ".join(parts)
