"""Deterministic synthetic data generator for e-commerce search & ranking MVP.

All randomness goes through a single ``random.Random(seed)`` instance so results
are bit-for-bit reproducible.  IDs are zero-padded counters, *not* UUID4.

Session-based generation ensures realistic event funnels:
    impression  →  click  →  favorite / add_to_cart  →  purchase
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from psr_srs_mvp.data_generation.config import GenerationConfig
from psr_srs_mvp.data_generation.schemas import (
    EVENT_FIELDS,
    EVENT_TYPES,
    ITEM_FIELDS,
    QUERY_FIELDS,
    USER_FIELDS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str, index: int, width: int = 6) -> str:
    """Deterministic ID: ``prefix`` + zero-padded ``index``."""
    return f"{prefix}_{index:0{width}d}"


def _to_json(obj: Any) -> str:
    """Stable JSON serialization for list columns in CSV."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _random_timestamp(
    rng: random.Random, start_ts: float, end_ts: float
) -> str:
    """Return an ISO-8601 UTC timestamp between *start_ts* and *end_ts* (POSIX)."""
    ts = rng.uniform(start_ts, end_ts)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class _Item:
    item_id: str
    title: str
    description: str
    category: str
    subcategory: str
    brand: str
    price: float
    quality_score: float
    popularity_score: float
    is_cold_start: bool
    created_at: str

    def to_row(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "price": f"{self.price:.2f}",
            "quality_score": f"{self.quality_score:.4f}",
            "popularity_score": f"{self.popularity_score:.4f}",
            "is_cold_start": str(self.is_cold_start).lower(),
            "created_at": self.created_at,
        }


@dataclass
class _User:
    user_id: str
    preferred_categories: list[str]
    preferred_brands: list[str]
    price_preference: str
    activity_level: str
    is_cold_start: bool
    created_at: str

    def to_row(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "preferred_categories": _to_json(self.preferred_categories),
            "preferred_brands": _to_json(self.preferred_brands),
            "price_preference": self.price_preference,
            "activity_level": self.activity_level,
            "is_cold_start": str(self.is_cold_start).lower(),
            "created_at": self.created_at,
        }


@dataclass
class _Query:
    query_id: str
    query_text: str
    intended_category: str
    semantic_intent: str
    created_at: str

    def to_row(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "intended_category": self.intended_category,
            "semantic_intent": self.semantic_intent,
            "created_at": self.created_at,
        }


@dataclass
class _Event:
    event_id: str
    event_type: str
    request_id: str
    session_id: str
    user_id: str
    query_id: str
    query_text: str
    item_id: str
    position: int
    timestamp: str
    click_duration_ms: str
    add_to_cart_quantity: str
    purchase_amount: str

    def to_row(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "item_id": self.item_id,
            "position": str(self.position),
            "timestamp": self.timestamp,
            "click_duration_ms": self.click_duration_ms,
            "add_to_cart_quantity": self.add_to_cart_quantity,
            "purchase_amount": self.purchase_amount,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DataGenerator:
    """Deterministic generator for all synthetic entities."""

    def __init__(self, config: GenerationConfig):
        self.cfg = config
        self.rng = random.Random(config.seed)

        # POSIX timestamps for the configured time range
        self._start_ts = config.start_datetime.timestamp()
        self._end_ts = config.end_datetime.timestamp()

        # Lookup tables built during generation
        self.items: dict[str, _Item] = {}
        self.users: dict[str, _User] = {}
        self.queries: dict[str, _Query] = {}
        self.events: list[_Event] = []
        self._event_counter = 0

        # Indexes for fast lookup during session generation
        self._items_by_category: dict[str, list[_Item]] = defaultdict(list)
        self._items_sorted_by_pop: list[_Item] = []

        # Query text templates
        self._query_templates: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Items
    # ------------------------------------------------------------------

    def _generate_title(self, subcat: str, brand: str) -> str:
        adjectives = ["Premium", "Classic", "Modern", "Essential", "Deluxe",
                      "Compact", "Professional", "Everyday", "Ultra", "Eco"]
        adj = self.rng.choice(adjectives)
        return f"{brand} {adj} {subcat.rstrip('s')}"

    def _generate_description(self, title: str, category: str, subcategory: str) -> str:
        templates = [
            f"{title} — high-quality {subcategory.lower()} for {category.lower()} enthusiasts.",
            f"Top-rated {subcategory.lower()} product from {title.split()[0]}. Ideal for daily use.",
            f"Experience excellence with {title}. Premium {category.lower()} choice.",
            f"{title} delivers outstanding performance in {subcategory.lower()}.",
            f"Discover {title}, the perfect {subcategory.lower()} solution.",
        ]
        return self.rng.choice(templates)

    def generate_items(self) -> list[_Item]:
        """Generate all items deterministically."""
        items: list[_Item] = []
        categories = self.cfg.categories.copy()
        cold_count = max(1, int(self.cfg.num_items * self.cfg.cold_start_item_ratio))

        # Pre-assign categories round-robin, then shuffle deterministically
        assignments: list[tuple[str, str, str]] = []  # (category, subcategory, brand)
        for i in range(self.cfg.num_items):
            cat = categories[i % len(categories)]
            subs = self.cfg.subcategories.get(cat, [cat])
            sub = subs[i % len(subs)]
            brs = self.cfg.brands.get(cat, ["GenericBrand"])
            brand = brs[i % len(brs)]
            assignments.append((cat, sub, brand))
        self.rng.shuffle(assignments)

        for idx in range(self.cfg.num_items):
            cat, sub, brand = assignments[idx]
            title = self._generate_title(sub, brand)
            desc = self._generate_description(title, cat, sub)

            # Price in configured range
            lo, hi = self.cfg.price_ranges.get(cat, [1, 100])
            price = round(self.rng.uniform(lo, hi), 2)

            # Quality: beta-like via mixture, mostly 0.4-0.95
            quality = round(max(0.0, min(1.0, self.rng.gauss(0.7, 0.15))), 4)

            # Popularity: power-law (Pareto-like) — a few items get most attention
            popularity = round(self.rng.betavariate(1.5, 5.0), 4)

            is_cold = idx < cold_count
            created = _random_timestamp(
                self.rng,
                self._end_ts - 86400 * 30 if is_cold else self._start_ts,
                self._end_ts if is_cold else self._end_ts - 86400 * 30,
            )

            item = _Item(
                item_id=_make_id("item", idx + 1),
                title=title,
                description=desc,
                category=cat,
                subcategory=sub,
                brand=brand,
                price=price,
                quality_score=quality,
                popularity_score=popularity,
                is_cold_start=is_cold,
                created_at=created,
            )
            items.append(item)
            self.items[item.item_id] = item
            self._items_by_category[cat].append(item)

        self._items_sorted_by_pop = sorted(items, key=lambda x: x.popularity_score, reverse=True)
        return items

    # ------------------------------------------------------------------
    # 2. Users
    # ------------------------------------------------------------------

    def generate_users(self) -> list[_User]:
        """Generate all users deterministically."""
        users: list[_User] = []
        cats = self.cfg.categories
        cold_count = max(1, int(self.cfg.num_users * self.cfg.cold_start_user_ratio))

        for idx in range(self.cfg.num_users):
            is_cold = idx < cold_count

            # Preferred categories: 1-3 from all categories
            n_pref = self.rng.randint(1, 3)
            pref_cats = self.rng.sample(cats, min(n_pref, len(cats)))

            # Preferred brands from those categories
            pref_brands: list[str] = []
            for cat in pref_cats:
                brs = self.cfg.brands.get(cat, [])
                if brs:
                    pref_brands.append(self.rng.choice(brs))
            if not pref_brands:
                all_brands = [b for blist in self.cfg.brands.values() for b in blist]
                pref_brands = self.rng.sample(all_brands, min(2, len(all_brands)))

            price_pref = self.rng.choice(self.cfg.price_preference_levels)
            activity = self.rng.choice(self.cfg.activity_levels)

            # Cold-start users were created recently; others span the full range
            created = _random_timestamp(
                self.rng,
                self._end_ts - 86400 * 14 if is_cold else self._start_ts,
                self._end_ts if is_cold else self._end_ts - 86400 * 14,
            )

            user = _User(
                user_id=_make_id("user", idx + 1),
                preferred_categories=pref_cats,
                preferred_brands=pref_brands,
                price_preference=price_pref,
                activity_level=activity,
                is_cold_start=is_cold,
                created_at=created,
            )
            users.append(user)
            self.users[user.user_id] = user

        return users

    # ------------------------------------------------------------------
    # 3. Queries
    # ------------------------------------------------------------------

    def _build_query_templates(self) -> None:
        """Build a pool of query text templates keyed by category."""
        for cat in self.cfg.categories:
            subs = self.cfg.subcategories.get(cat, [cat])
            brands = self.cfg.brands.get(cat, ["popular brand"])
            for sub in subs:
                for brand in brands:
                    # Exact keyword queries
                    self._query_templates.append({
                        "text": f"{brand} {sub}",
                        "category": cat,
                        "intent": f"Exact search for {sub} by {brand} in {cat}",
                        "kind": "exact",
                    })
                    # Category browse queries
                    self._query_templates.append({
                        "text": f"best {sub.lower()}",
                        "category": cat,
                        "intent": f"Browse top-rated {sub} in {cat}",
                        "kind": "category",
                    })
                    # Natural-language queries
                    self._query_templates.append({
                        "text": f"affordable {sub.lower()} for everyday use",
                        "category": cat,
                        "intent": f"I need budget-friendly {sub} from {cat}",
                        "kind": "nl",
                    })
                    self._query_templates.append({
                        "text": f"{brand} {sub.lower()} review",
                        "category": cat,
                        "intent": f"Looking for reviews of {brand} {sub}",
                        "kind": "brand",
                    })
        self.rng.shuffle(self._query_templates)

    def generate_queries(self) -> list[_Query]:
        """Generate all queries deterministically."""
        if not self._query_templates:
            self._build_query_templates()

        queries: list[_Query] = []
        # Cycle through templates, shuffling periodically for variety
        pool = self._query_templates.copy()
        self.rng.shuffle(pool)

        for idx in range(self.cfg.num_queries):
            tmpl = pool[idx % len(pool)]
            created = _random_timestamp(self.rng, self._start_ts, self._end_ts)
            q = _Query(
                query_id=_make_id("query", idx + 1),
                query_text=tmpl["text"],
                intended_category=tmpl["category"],
                semantic_intent=tmpl["intent"],
                created_at=created,
            )
            queries.append(q)
            self.queries[q.query_id] = q

        return queries

    # ------------------------------------------------------------------
    # 4. Events (session-based cascade browsing)
    # ------------------------------------------------------------------

    def _relevance_score(self, item: _Item, query: _Query) -> float:
        """Score how relevant *item* is to *query* (0-1)."""
        score = 0.0
        if item.category == query.intended_category:
            score += 0.6
        if item.subcategory.lower() in query.query_text.lower():
            score += 0.3
        if item.brand.lower() in query.query_text.lower():
            score += 0.1
        return min(score, 1.0)

    def _click_probability(
        self,
        position: int,
        relevance: float,
        user: _User,
        item: _Item,
    ) -> float:
        """Cascade-browsing click probability at a SERP position.

        base_click × position_decay × relevance_boost × preference_boost
        Cold-start users get only relevance boost (no historical preferences).
        """
        base = self.cfg.base_click_probability / (position ** self.cfg.position_decay_alpha)
        rel_factor = 1.0 + (self.cfg.relevance_boost - 1.0) * relevance

        # Cold-start users lack preference history — use only relevance
        if user.is_cold_start:
            pref = 1.0
        else:
            pref = 1.0
            if item.category in user.preferred_categories:
                pref *= self.cfg.preference_boost
            if item.brand in user.preferred_brands:
                pref *= self.cfg.preference_boost

        prob = base * rel_factor * pref
        return min(prob, 0.85)

    def _purchase_probability(self, item: _Item, user: _User) -> float:
        """Post-add_to_cart purchase probability based on quality, price, and category."""
        base = self.cfg.base_purchase_probability
        quality_factor = 0.5 + 0.5 * item.quality_score

        price_map = {"budget": 0.0, "mid_range": 0.5, "premium": 1.0}
        user_price_idx = price_map.get(user.price_preference, 0.5)
        cat_range = self.cfg.price_ranges.get(item.category, [1, 100])
        price_span = cat_range[1] - cat_range[0]
        price_norm = (item.price - cat_range[0]) / price_span if price_span > 0 else 0.5
        price_affinity = 1.0 - abs(price_norm - user_price_idx)

        # Category preference gives slight boost even for purchase
        cat_boost = 1.2 if item.category in user.preferred_categories else 1.0

        prob = base * quality_factor * (0.4 + 0.6 * price_affinity) * cat_boost
        return min(prob, 0.70)

    def _build_serp(
        self, query: _Query, user: _User, exclude_ids: set[str]
    ) -> list[tuple[_Item, float]]:
        """Build a search engine results page (SERP) for a query.

        Mixes relevant and popular items, ordered by a ranking score.
        """
        cat_items = self._items_by_category.get(query.intended_category, [])
        all_items = list(self.items.values())

        scored: list[tuple[_Item, float]] = []

        # Relevant items from the query's category
        relevant_pool = [it for it in cat_items if it.item_id not in exclude_ids]
        self.rng.shuffle(relevant_pool)
        for item in relevant_pool[: self.cfg.serp_size // 2]:
            rel = self._relevance_score(item, query)
            rank_score = rel * 0.7 + item.popularity_score * 0.3
            scored.append((item, rank_score))

        # Popular items from other categories (diversity / noise)
        diverse_pool = [
            it for it in all_items
            if it.category != query.intended_category and it.item_id not in exclude_ids
        ]
        self.rng.shuffle(diverse_pool)
        if diverse_pool:
            weights = [it.popularity_score + 0.01 for it in diverse_pool]
            selected = self.rng.choices(diverse_pool, weights=weights,
                                        k=min(self.cfg.serp_size // 2, len(diverse_pool)))
            for item in selected:
                rel = self._relevance_score(item, query)
                rank_score = rel * 0.3 + item.popularity_score * 0.7
                scored.append((item, rank_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.cfg.serp_size]

    def generate_events(self) -> list[_Event]:
        """Generate all events via cascade-browsing session simulation.

        Cascade model:
        - User browses SERP from position 1 downward
        - At each position: always impression; click decided by cascade model
        - After a click, user may stop browsing (post_click_stop_probability)
        - Maximum max_clicks_per_request clicks per request
        - Event dependencies: purchase ONLY after add_to_cart (not after click alone)
        """
        if not self.items or not self.users or not self.queries:
            raise RuntimeError(
                "Items, users, and queries must be generated before events. "
                "Call generate_items(), generate_users(), generate_queries() first."
            )

        activity_weight = {"low": 1.0, "medium": 3.0, "high": 8.0}
        user_list = list(self.users.values())
        user_weights = [activity_weight.get(u.activity_level, 1.0) for u in user_list]
        query_list = list(self.queries.values())

        for sess_idx in range(self.cfg.num_sessions):
            request_id = _make_id("req", sess_idx + 1)
            session_id = _make_id("sess", sess_idx + 1)

            # Pick user (weighted by activity)
            user = self.rng.choices(user_list, weights=user_weights, k=1)[0]
            if user.is_cold_start and self.rng.random() < 0.3:
                continue

            # Pick query
            if user.preferred_categories and self.rng.random() < 0.7:
                pref_qs = [q for q in query_list
                           if q.intended_category in user.preferred_categories]
                pool = pref_qs if pref_qs else query_list
            else:
                pool = query_list
            query = self.rng.choice(pool)

            session_base = self.rng.uniform(self._start_ts, self._end_ts)
            t = session_base
            clicks_in_request = 0

            # Build SERP
            serp = self._build_serp(query, user, exclude_ids=set())

            for pos_zero, (item, _rank_score) in enumerate(serp):
                position = pos_zero + 1
                t = self._time_offset(t, 2.0)

                # --- impression (always) ---
                self.events.append(_Event(
                    event_id=self._next_event_id(),
                    event_type="impression",
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user.user_id,
                    query_id=query.query_id,
                    query_text=query.query_text,
                    item_id=item.item_id,
                    position=position,
                    timestamp=self._ts_str(t),
                    click_duration_ms="",
                    add_to_cart_quantity="",
                    purchase_amount="",
                ))

                # --- click (cascade model) ---
                if clicks_in_request >= self.cfg.max_clicks_per_request:
                    continue  # hit the cap — no more clicks in this request

                relevance = self._relevance_score(item, query)
                cprob = self._click_probability(position, relevance, user, item)
                if self.rng.random() >= cprob:
                    continue  # no click — keep scanning

                # Click occurred
                clicks_in_request += 1
                t = self._time_offset(t, 30.0)
                click_dur = max(0, int(self.rng.gauss(6000, 4000)))
                self.events.append(_Event(
                    event_id=self._next_event_id(),
                    event_type="click",
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user.user_id,
                    query_id=query.query_id,
                    query_text=query.query_text,
                    item_id=item.item_id,
                    position=position,
                    timestamp=self._ts_str(t),
                    click_duration_ms=str(click_dur),
                    add_to_cart_quantity="",
                    purchase_amount="",
                ))

                # --- favorite (only after click) ---
                fav_prob = self.cfg.base_favorite_probability * (0.5 + 0.5 * item.quality_score)
                if self.rng.random() < fav_prob:
                    t = self._time_offset(t, 10.0)
                    self.events.append(_Event(
                        event_id=self._next_event_id(),
                        event_type="favorite",
                        request_id=request_id,
                        session_id=session_id,
                        user_id=user.user_id,
                        query_id=query.query_id,
                        query_text=query.query_text,
                        item_id=item.item_id,
                        position=position,
                        timestamp=self._ts_str(t),
                        click_duration_ms="",
                        add_to_cart_quantity="",
                        purchase_amount="",
                    ))

                # --- add_to_cart (only after click) ---
                atc_prob = self.cfg.base_add_to_cart_probability * (0.5 + 0.5 * item.quality_score)
                added_to_cart = False
                if self.rng.random() < atc_prob:
                    t = self._time_offset(t, 15.0)
                    qty = self.rng.randint(1, 5)
                    self.events.append(_Event(
                        event_id=self._next_event_id(),
                        event_type="add_to_cart",
                        request_id=request_id,
                        session_id=session_id,
                        user_id=user.user_id,
                        query_id=query.query_id,
                        query_text=query.query_text,
                        item_id=item.item_id,
                        position=position,
                        timestamp=self._ts_str(t),
                        click_duration_ms="",
                        add_to_cart_quantity=str(qty),
                        purchase_amount="",
                    ))
                    added_to_cart = True

                # --- purchase (ONLY after add_to_cart — strict dependency) ---
                if added_to_cart:
                    pur_prob = self._purchase_probability(item, user)
                    if self.rng.random() < pur_prob:
                        t = self._time_offset(t, 120.0)
                        amount = round(item.price * self.rng.uniform(0.8, 1.2), 2)
                        self.events.append(_Event(
                            event_id=self._next_event_id(),
                            event_type="purchase",
                            request_id=request_id,
                            session_id=session_id,
                            user_id=user.user_id,
                            query_id=query.query_id,
                            query_text=query.query_text,
                            item_id=item.item_id,
                            position=position,
                            timestamp=self._ts_str(t),
                            click_duration_ms="",
                            add_to_cart_quantity="",
                            purchase_amount=f"{amount:.2f}",
                        ))

                # Post-click stop — user may leave after a click
                if self.rng.random() < self.cfg.post_click_stop_probability:
                    break

        return self.events

    # ------------------------------------------------------------------
    # 5. Qrels (query relevance judgments) — independent of events
    # ------------------------------------------------------------------

    def generate_qrels(self) -> list[dict[str, str]]:
        """Generate relevance judgments (qrels) purely from query-item semantics.

        Relevance is based ONLY on objective attributes:
        - intended_category match
        - subcategory overlap
        - brand mention in query text
        - keyword overlap in title/description

        This is INDEPENDENT of clicks, position, popularity, and user behavior.
        """
        if not self.items or not self.queries:
            raise RuntimeError(
                "Items and queries must be generated before qrels. "
                "Call generate_items(), generate_queries() first."
            )

        qrels: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for query in self.queries.values():
            query_tokens = set(query.query_text.lower().split())
            query_intent_tokens = set(query.semantic_intent.lower().split())

            for item in self.items.values():
                # --- Compute relevance score from objective signals ---
                score = 0

                # Category match (major signal)
                if item.category == query.intended_category:
                    score += 1

                # Subcategory match
                if item.subcategory.lower() in query.query_text.lower():
                    score += 1
                elif item.subcategory.lower() in query.semantic_intent.lower():
                    score += 0  # already in category match

                # Brand match in query
                if item.brand.lower() in query.query_text.lower():
                    score += 1

                # Title keyword overlap
                title_tokens = set(item.title.lower().split())
                kw_overlap = len(query_tokens & title_tokens)
                if kw_overlap >= 2:
                    score += 1
                elif kw_overlap == 1:
                    score += 0  # minimal signal

                # Description keyword overlap
                desc_tokens = set(item.description.lower().split())
                desc_overlap = len(query_intent_tokens & desc_tokens)
                if desc_overlap >= 3:
                    score += 1

                # --- Map score to relevance grade ---
                if score >= 3:
                    grade = 3  # highly relevant
                elif score >= 2:
                    grade = 2  # relevant
                elif score >= 1:
                    grade = 1  # weakly relevant
                else:
                    continue  # not relevant — sparse storage

                pair = (query.query_id, item.item_id)
                if pair in seen:
                    continue
                seen.add(pair)

                qrels.append({
                    "query_id": query.query_id,
                    "item_id": item.item_id,
                    "relevance_grade": str(grade),
                })

        # --- Fallback: guarantee every query has at least one grade-2+ item ---
        grades_by_query: dict[str, set[int]] = defaultdict(set)
        for r in qrels:
            grades_by_query[r["query_id"]].add(int(r["relevance_grade"]))

        for query in self.queries.values():
            existing_grades = grades_by_query.get(query.query_id, set())
            if max(existing_grades, default=0) >= 2:
                continue  # already covered

            # Force a grade-2 item from the query's category
            cat_items = self._items_by_category.get(query.intended_category, [])
            if not cat_items:
                continue
            # Sort by keyword overlap
            query_tokens = set(query.query_text.lower().split())
            cat_items_sorted = sorted(
                cat_items,
                key=lambda it: len(query_tokens & set(it.title.lower().split())),
                reverse=True,
            )
            # Pick the best item; if all are already paired, still add a grade-2 entry
            added = False
            for item in cat_items_sorted:
                pair = (query.query_id, item.item_id)
                if pair not in seen:
                    seen.add(pair)
                    qrels.append({
                        "query_id": query.query_id,
                        "item_id": item.item_id,
                        "relevance_grade": "2",
                    })
                    added = True
                    break
            # Cross-category safety net: pick ANY item from the whole catalog
            if not added:
                all_sorted = sorted(
                    self.items.values(),
                    key=lambda it: len(query_tokens & set(it.title.lower().split())),
                    reverse=True,
                )
                for item in all_sorted:
                    pair = (query.query_id, item.item_id)
                    if pair not in seen:
                        seen.add(pair)
                        qrels.append({
                            "query_id": query.query_id,
                            "item_id": item.item_id,
                            "relevance_grade": "2",
                        })
                        break

        # Sort for deterministic output
        qrels.sort(key=lambda r: (r["query_id"], r["item_id"]))
        return qrels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_offset(self, base: float, max_delta: float) -> float:
        return min(base + self.rng.uniform(0, max_delta), self._end_ts)

    def _ts_str(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event_{self._event_counter:08d}"

    # ------------------------------------------------------------------
    # Master entry-point
    # ------------------------------------------------------------------

    def generate_all(self) -> dict[str, list[dict[str, str]]]:
        """Run the full generation pipeline and return rows keyed by entity.

        Returns:
            ``{"items": [...], "users": [...], "queries": [...], "events": [...], "qrels": [...]}``
        """
        items = self.generate_items()
        users = self.generate_users()
        queries = self.generate_queries()
        events = self.generate_events()
        qrels = self.generate_qrels()

        return {
            "items": [it.to_row() for it in items],
            "users": [u.to_row() for u in users],
            "queries": [q.to_row() for q in queries],
            "events": [e.to_row() for e in events],
            "qrels": qrels,
        }

    def summary(self) -> dict[str, Any]:
        """Return a short summary dict suitable for printing."""
        etype_counts: dict[str, int] = defaultdict(int)
        cold_users = 0
        cold_items = 0
        for e in self.events:
            etype_counts[e.event_type] += 1
        for u in self.users.values():
            if u.is_cold_start:
                cold_users += 1
        for it in self.items.values():
            if it.is_cold_start:
                cold_items += 1
        return {
            "num_items": len(self.items),
            "num_users": len(self.users),
            "num_queries": len(self.queries),
            "num_events": len(self.events),
            "event_type_counts": dict(etype_counts),
            "cold_start_users": cold_users,
            "cold_start_items": cold_items,
            "seed": self.cfg.seed,
        }
