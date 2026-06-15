"""LSA semantic retrieval index with cosine-similarity search.

Uses the ``SemanticVectorizer`` (TF-IDF + TruncatedSVD) pipeline.
Strictly **inductive**: fit on item documents only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from psr_srs_mvp.retrieval.vectorization import (
    SemanticConfig,
    SemanticVectorizer,
    is_zero_vector,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticSearchResult:
    """A single semantic search result."""

    score: float
    item_id: str
    rank: int = 0


# ---------------------------------------------------------------------------
# Semantic index
# ---------------------------------------------------------------------------

class SemanticIndex:
    """LSA-based semantic retrieval index.

    Builds an index from item texts using TF-IDF + TruncatedSVD, then
    searches by encoding queries into the same latent space and computing
    cosine similarity against all item vectors.
    """

    def __init__(self):
        self._item_ids: list[str] = []
        self._item_vectors: np.ndarray | None = None
        self._vectorizer: SemanticVectorizer | None = None
        self._config: SemanticConfig | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        documents: Sequence[str],
        item_ids: Sequence[str],
        config: SemanticConfig,
    ) -> "SemanticIndex":
        """Fit the vectorizer on *documents* and encode all items.

        Args:
            documents: Raw item text strings (same order as *item_ids*).
            item_ids: Unique item identifiers.
            config: Semantic configuration.

        Returns:
            A fitted ``SemanticIndex`` ready for search.
        """
        vec = SemanticVectorizer(config)
        vec.fit(list(documents))
        vectors = vec.transform(list(documents))

        idx = cls()
        idx._item_ids = list(item_ids)
        idx._item_vectors = vectors
        idx._vectorizer = vec
        idx._config = config
        return idx

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SemanticSearchResult]:
        """Encode *query* and return top-K results by cosine similarity.

        Args:
            query: Raw query string.
            top_k: Maximum results.  Must be >= 1.

        Returns:
            Ranked results (score descending, item_id ascending for ties).

        Raises:
            ValueError: If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if self._item_vectors is None or self._vectorizer is None:
            raise RuntimeError("Index not built. Call SemanticIndex.build() first.")

        # Encode query
        q_vec = self._vectorizer.transform([query])[0]

        # Zero-vector → no results
        if is_zero_vector(q_vec):
            return []

        # Cosine similarity: dot product (vectors are L2-normalised)
        scores = self._item_vectors @ q_vec  # (N,) array

        # Filter NaN / Inf
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return []

        # Get top-k indices
        valid_indices = np.where(finite_mask)[0]
        valid_scores = scores[valid_indices]

        if len(valid_indices) == 0:
            return []

        # Sort by score descending, then item_id ascending for ties
        order = np.lexsort(
            (np.array([self._item_ids[i] for i in valid_indices]),
             -valid_scores)
        )
        top_indices = valid_indices[order[:top_k]]

        results = []
        for rank, i in enumerate(top_indices, start=1):
            results.append(SemanticSearchResult(
                score=float(scores[i]),
                item_id=self._item_ids[i],
                rank=rank,
            ))
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return len(self._item_ids)

    @property
    def vector_dim(self) -> int:
        if self._item_vectors is None:
            return 0
        return self._item_vectors.shape[1]

    @property
    def config(self) -> SemanticConfig | None:
        return self._config

    @property
    def vectorizer(self) -> SemanticVectorizer | None:
        return self._vectorizer
