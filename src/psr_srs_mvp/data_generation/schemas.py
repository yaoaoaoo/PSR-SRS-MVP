"""Field name constants and schema definitions for synthetic data CSVs."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------
EVENT_TYPES = ("impression", "click", "favorite", "add_to_cart", "purchase")

# ---------------------------------------------------------------------------
# Item CSV fields
# ---------------------------------------------------------------------------
ITEM_FIELDS: list[str] = [
    "item_id",
    "title",
    "description",
    "category",
    "subcategory",
    "brand",
    "price",
    "quality_score",
    "popularity_score",
    "is_cold_start",
    "created_at",
]

# ---------------------------------------------------------------------------
# User CSV fields
# ---------------------------------------------------------------------------
USER_FIELDS: list[str] = [
    "user_id",
    "preferred_categories",
    "preferred_brands",
    "price_preference",
    "activity_level",
    "is_cold_start",
    "created_at",
]

# ---------------------------------------------------------------------------
# Query CSV fields
# ---------------------------------------------------------------------------
QUERY_FIELDS: list[str] = [
    "query_id",
    "query_text",
    "intended_category",
    "semantic_intent",
    "created_at",
]

# ---------------------------------------------------------------------------
# Event CSV fields
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Qrels CSV fields
# ---------------------------------------------------------------------------
QRELS_FIELDS: list[str] = [
    "query_id",
    "item_id",
    "relevance_grade",
]

# ---------------------------------------------------------------------------
# Event CSV fields
# ---------------------------------------------------------------------------
EVENT_FIELDS: list[str] = [
    "event_id",
    "event_type",
    "request_id",
    "session_id",
    "user_id",
    "query_id",
    "query_text",
    "item_id",
    "position",
    "timestamp",
    "click_duration_ms",
    "add_to_cart_quantity",
    "purchase_amount",
]
