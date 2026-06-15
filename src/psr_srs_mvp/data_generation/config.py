"""Configuration dataclass and JSON loader for synthetic data generation.

Uses only Python standard library — no Pydantic or PyYAML dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class GenerationConfig:
    """Typed configuration for synthetic data generation."""

    # Random seed
    seed: int = 20260614

    # Entity counts
    num_items: int = 500
    num_users: int = 100
    num_queries: int = 200
    num_sessions: int = 500
    target_event_count: int = 8000

    # Cold-start ratios
    cold_start_user_ratio: float = 0.1
    cold_start_item_ratio: float = 0.1

    # Time range (ISO 8601 strings)
    start_date: str = "2026-01-01T00:00:00+00:00"
    end_date: str = "2026-03-31T23:59:59+00:00"

    # Category hierarchy
    categories: list[str] = field(default_factory=list)
    subcategories: dict[str, list[str]] = field(default_factory=dict)
    brands: dict[str, list[str]] = field(default_factory=dict)
    price_ranges: dict[str, list[float]] = field(default_factory=dict)

    # User attribute enums
    price_preference_levels: list[str] = field(
        default_factory=lambda: ["budget", "mid_range", "premium"]
    )
    activity_levels: list[str] = field(
        default_factory=lambda: ["low", "medium", "high"]
    )

    # Behavioural tuning parameters
    serp_size: int = 20
    position_decay_alpha: float = 0.6

    # Click model (cascade browsing)
    base_click_probability: float = 0.14
    max_clicks_per_request: int = 3
    post_click_stop_probability: float = 0.50
    relevance_boost: float = 1.8
    preference_boost: float = 1.4

    # Post-click event base probabilities
    base_favorite_probability: float = 0.10
    base_add_to_cart_probability: float = 0.09
    base_purchase_probability: float = 0.35

    @property
    def event_types(self) -> list[str]:
        """Valid event types ordered by funnel stage."""
        return ["impression", "click", "favorite", "add_to_cart", "purchase"]

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromisoformat(self.start_date)

    @property
    def end_datetime(self) -> datetime:
        return datetime.fromisoformat(self.end_date)

    @property
    def time_range_seconds(self) -> float:
        return (self.end_datetime - self.start_datetime).total_seconds()

    def validate(self) -> list[str]:
        """Validate configuration values. Returns list of error messages."""
        errors: list[str] = []
        if self.seed < 0:
            errors.append("seed must be non-negative")
        if self.num_items < 1:
            errors.append("num_items must be >= 1")
        if self.num_users < 1:
            errors.append("num_users must be >= 1")
        if self.num_queries < 1:
            errors.append("num_queries must be >= 1")
        if self.num_sessions < 1:
            errors.append("num_sessions must be >= 1")
        if not (0 <= self.cold_start_user_ratio <= 1):
            errors.append("cold_start_user_ratio must be in [0, 1]")
        if not (0 <= self.cold_start_item_ratio <= 1):
            errors.append("cold_start_item_ratio must be in [0, 1]")
        if self.start_datetime >= self.end_datetime:
            errors.append("start_date must be before end_date")
        if self.serp_size < 1:
            errors.append("serp_size must be >= 1")
        if self.position_decay_alpha <= 0:
            errors.append("position_decay_alpha must be > 0")
        if not (0 < self.base_click_probability <= 1):
            errors.append("base_click_probability must be in (0, 1]")
        if self.max_clicks_per_request < 1:
            errors.append("max_clicks_per_request must be >= 1")
        if not (0 <= self.post_click_stop_probability <= 1):
            errors.append("post_click_stop_probability must be in [0, 1]")
        if not (0 <= self.base_favorite_probability <= 1):
            errors.append("base_favorite_probability must be in [0, 1]")
        if not (0 <= self.base_add_to_cart_probability <= 1):
            errors.append("base_add_to_cart_probability must be in [0, 1]")
        if not (0 <= self.base_purchase_probability <= 1):
            errors.append("base_purchase_probability must be in [0, 1]")
        if not self.categories:
            errors.append("categories must not be empty")
        if not self.price_preference_levels:
            errors.append("price_preference_levels must not be empty")
        if not self.activity_levels:
            errors.append("activity_levels must not be empty")
        return errors


def load_config(path: str | Path) -> GenerationConfig:
    """Load generation configuration from a JSON file.

    Args:
        path: Path to a JSON configuration file.

    Returns:
        A validated GenerationConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If configuration values are invalid.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = GenerationConfig(**{k: v for k, v in raw.items() if k in GenerationConfig.__dataclass_fields__})
    errors = config.validate()
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    return config
