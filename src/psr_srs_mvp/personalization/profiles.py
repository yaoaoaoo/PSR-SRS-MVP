"""User profile construction from historical behaviour events."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class UserProfile:
    """Aggregated user preferences from training events."""

    def __init__(self, user_id: str, is_cold_start: bool = False):
        self.user_id = user_id
        self.is_cold_start = is_cold_start
        self.train_event_count = 0
        self.train_session_count: set[str] = set()
        self.positive_event_count = 0
        self.profile_status = "empty"

        # Weighted accumulators
        self._cat_scores: dict[str, float] = defaultdict(float)
        self._subcat_scores: dict[str, float] = defaultdict(float)
        self._brand_scores: dict[str, float] = defaultdict(float)
        self._price_log_sum = 0.0
        self._price_weight_sum = 0.0
        self._last_ts: datetime | None = None

    @property
    def category_weights(self) -> dict[str, float]:
        return _normalize(self._cat_scores)

    @property
    def subcategory_weights(self) -> dict[str, float]:
        return _normalize(self._subcat_scores)

    @property
    def brand_weights(self) -> dict[str, float]:
        return _normalize(self._brand_scores)

    @property
    def mean_log_price(self) -> float | None:
        if self._price_weight_sum > 0:
            return self._price_log_sum / self._price_weight_sum
        return None

    @property
    def price_std(self) -> float:
        # Simplified: return a fixed minimum scale
        # Full weighted std would need a second pass
        return 0.5

    def add_event(self, event: dict, item: dict, weight: float, decay: float):
        """Add a single positive event contribution."""
        eff = weight * decay
        if eff <= 0:
            return
        cat = item.get("category", "")
        subcat = item.get("subcategory", "")
        brand = item.get("brand", "")
        if cat:
            self._cat_scores[cat] += eff
        if subcat:
            self._subcat_scores[subcat] += eff
        if brand:
            self._brand_scores[brand] += eff

        try:
            price = float(item.get("price", 0))
        except (ValueError, TypeError):
            price = 1.0
        if price > 0:
            self._price_log_sum += math.log(price) * eff
            self._price_weight_sum += eff

        self.positive_event_count += 1
        ts = _parse_ts(event["timestamp"])
        if self._last_ts is None or ts > self._last_ts:
            self._last_ts = ts

    def finalize(self):
        """Mark profile as ready and compute status."""
        if self.positive_event_count > 0:
            self.profile_status = "warm"
        elif self.is_cold_start:
            self.profile_status = "cold_start"
        elif self.train_event_count > 0:
            self.profile_status = "no_positive"
        else:
            self.profile_status = "no_history"

    def last_train_event_at(self) -> str:
        if self._last_ts:
            return self._last_ts.isoformat()
        return ""

    def to_row(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "is_cold_start": str(self.is_cold_start).lower(),
            "train_event_count": str(self.train_event_count),
            "train_session_count": str(len(self.train_session_count)),
            "positive_event_count": str(self.positive_event_count),
            "profile_status": self.profile_status,
            "top_categories": json.dumps(
                sorted(self.category_weights, key=self.category_weights.get, reverse=True)[:5],
                ensure_ascii=False),
            "top_subcategories": json.dumps(
                sorted(self.subcategory_weights, key=self.subcategory_weights.get, reverse=True)[:5],
                ensure_ascii=False),
            "top_brands": json.dumps(
                sorted(self.brand_weights, key=self.brand_weights.get, reverse=True)[:5],
                ensure_ascii=False),
            "category_weights": json.dumps(self.category_weights, ensure_ascii=False),
            "subcategory_weights": json.dumps(self.subcategory_weights, ensure_ascii=False),
            "brand_weights": json.dumps(self.brand_weights, ensure_ascii=False),
            "mean_log_price": f"{self.mean_log_price:.4f}" if self.mean_log_price is not None else "",
            "price_std": f"{self.price_std:.4f}",
            "last_train_event_at": self.last_train_event_at(),
        }


def _normalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in d.items()}


def load_items(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def load_users_map(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {r["user_id"]: r for r in csv.DictReader(f)}


def build_profiles(
    train_events: list[dict],
    items: dict[str, dict],
    users_map: dict[str, dict],
    event_weights: dict[str, float],
    half_life_days: float,
) -> dict[str, UserProfile]:
    """Build user profiles from training events only."""
    profiles: dict[str, UserProfile] = {}

    # Initialize all known users
    for uid, urow in users_map.items():
        is_cold = urow.get("is_cold_start", "false").lower() == "true"
        profiles[uid] = UserProfile(uid, is_cold)

    # Find the latest training timestamp for decay reference
    train_ts = [_parse_ts(e["timestamp"]) for e in train_events]
    ref_ts = max(train_ts) if train_ts else datetime(2026, 3, 31, tzinfo=timezone.utc)

    positive_types = {"click", "favorite", "add_to_cart", "purchase"}

    for e in train_events:
        uid = e["user_id"]
        if uid not in profiles:
            profiles[uid] = UserProfile(uid)
        profile = profiles[uid]
        profile.train_event_count += 1
        profile.train_session_count.add(e.get("session_id", ""))

        etype = e["event_type"]
        if etype not in positive_types:
            continue

        w = event_weights.get(etype, 0.0)
        if w <= 0:
            continue

        # Time decay
        ts = _parse_ts(e["timestamp"])
        age_days = (ref_ts - ts).total_seconds() / 86400.0
        decay = 0.5 ** (age_days / half_life_days) if half_life_days > 0 else 1.0

        iid = e.get("item_id", "")
        item = items.get(iid)
        if item:
            profile.add_event(e, item, w, decay)

    for p in profiles.values():
        p.finalize()

    return profiles
