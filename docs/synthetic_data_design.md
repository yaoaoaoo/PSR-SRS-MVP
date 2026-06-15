# Synthetic Data Design — PSR-SRS MVP

> Version: 0.1.0 | Date: 2026-06-14

## 1. Overview

The synthetic data generator produces a realistic, self-consistent dataset for
e-commerce personalized search ranking experiments. All data is generated
locally with a fixed random seed for perfect reproducibility.

## 2. Entity Relationships

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Items    │     │  Users   │     │ Queries  │
│ (500)     │     │ (100)    │     │ (200)    │
└─────┬─────┘     └─────┬────┘     └────┬─────┘
      │                 │               │
      │    ┌────────────┼───────────────┘
      │    │            │
      ▼    ▼            ▼
   ┌────────────────────────┐
   │        Events          │
   │      (~8,000)          │
   │                        │
   │  session_id ──────────► groups events into user sessions
   │  request_id ──────────► groups events into SERP views
   └────────────────────────┘
```

- One **session** = one user issuing one query and interacting with the SERP
- One **request** = one SERP page view (may span multiple events)
- Events within a session share `session_id`
- Impression events within the same SERP share `request_id`

## 3. Generation Algorithm

### 3.0 Cascade Browsing Model

Events are generated through a **session-based cascade browsing model**:

1. **User & Query selection**: User weighted by activity level; query biased toward user preferences
2. **SERP construction**: Mix of query-relevant items (50%, sorted by relevance × 0.7 + popularity × 0.3) and diverse popular items (50%, sorted by relevance × 0.3 + popularity × 0.7)
3. **Cascade browsing**: User scans positions 1 → N. At each position:
   - Always: impression
   - Click: `base_click / pos^alpha × relevance_boost × preference_boost` (cold-start users: relevance only)
   - After click: may stop browsing (`post_click_stop_probability`)
   - Hard cap: `max_clicks_per_request` clicks per SERP
4. **Post-click events** (only after click on same `(request, item)`):
   - Favorite: `base_favorite × (0.5 + 0.5 × quality)`
   - Add to cart: `base_atc × (0.5 + 0.5 × quality)`
   - Purchase: **only after add_to_cart** — strict dependency

### 3.1 Items

1. Assign each item a `(category, subcategory, brand)` tuple via shuffled
   round-robin assignment — ensures even distribution with realistic variety.
2. Generate titles from templates: `"{Brand} {Adjective} {Subcategory}"`
3. Sample `quality_score` from a Gaussian(0.7, 0.15) clipped to [0, 1]
4. Sample `popularity_score` from Beta(1.5, 5.0) → power-law distribution
5. Price drawn uniformly from the category's configured price range
6. Cold-start items (~10%) have `created_at` within the last 30 days of the
   time range; others span the full range

### 3.2 Users

1. Each user selects 1–3 preferred categories uniformly
2. Each user selects 1–2 preferred brands from their preferred categories
3. `price_preference` ∈ {`budget`, `mid_range`, `premium`}
4. `activity_level` ∈ {`low`, `medium`, `high`}
5. Cold-start users (~10%) created in the last 14 days

### 3.3 Queries

Queries are drawn from a template pool built from the cartesian product of
categories × subcategories × brands, with four query types:

| Type      | Example                                     |
|-----------|---------------------------------------------|
| Exact     | `"TechPro Premium Laptops"`                 |
| Category  | `"best headphones"`                         |
| Brand     | `"NovaDigital camera review"`               |
| NL Intent | `"affordable skincare for everyday use"`    |

The pool is shuffled and cycled to produce the configured number of queries.
Each query records its `intended_category` and a human-readable
`semantic_intent`.

### 3.4 Events (Session-Based)

This is the core of the data generator. Each session simulates a complete
search interaction:

#### Step 1: Select User & Query

- User selected with probability proportional to `activity_level`
- Cold-start users have a 30% chance of being skipped (fewer sessions)
- Query selected with 70% bias toward the user's preferred categories

#### Step 2: Build SERP

- **Relevant pool**: items from the query's intended category (~half the SERP)
- **Diverse pool**: popular items from other categories (~half the SERP)
- Items scored by: `relevance × 0.7 + popularity × 0.3`
- Sorted descending → SERP order

#### Step 3: Generate Funnel Events

For each position `p` in the SERP:

1. **Impression** — always created

2. **Click** — probability = base_CTR × relevance_boost × preference_boost

   ```
   base_CTR(p)     = 1 / p^alpha          (alpha = 0.5)
   relevance_boost = 1 + (boost - 1) × relevance_score
   preference_boost = pref_cat_boost × pref_brand_boost
                    (1.5× if item category in user preferences,
                     1.5× if item brand in user preferences)
   ```

3. **Favorite** — probability = quality_score × 0.3 (only after click)

4. **Add to Cart** — probability = quality_score × 0.25 (only after click)

5. **Purchase** — probability based on quality and price alignment:
   ```
   p_purchase = quality × 0.6 × (0.4 + 0.6 × price_affinity) × 0.5
   ```
   where `price_affinity` measures alignment between item price percentile
   and user price preference level.

## 4. Probability Model Summary

| Factor | Effect |
|--------|--------|
| Position decay | Power-law `1/pos^α`. Higher position = lower CTR |
| Relevance | Items matching query category get up to 2× CTR boost |
| User category pref | Items in preferred categories get 1.5× CTR boost |
| User brand pref | Items from preferred brands get 1.5× CTR boost |
| Popularity | More popular items appear more often in diverse SERP slots |
| Quality → conversion | Higher quality → more favorites, add-to-carts, purchases |
| Price alignment | Better price-preference match → higher purchase probability |
| Activity level | High-activity users appear in more sessions |

## 5. Reproducibility

- **Seed**: Fixed integer (default: `20260614`)
- **RNG**: `random.Random(seed)` — fully isolated from global state
- **IDs**: Zero-padded deterministic counters (e.g., `item_000042`)
- **JSON fields**: `json.dumps(..., sort_keys=True, ensure_ascii=False)`
- **Timestamps**: Determined by RNG; ISO 8601 with UTC offset

Same config + same seed → bit-identical CSV output on any platform.

## 6. Configuration

See `configs/sample.json` for the full default configuration.

Key tunable parameters:

```json
{
  "seed": 20260614,
  "num_items": 500,
  "num_users": 100,
  "num_queries": 200,
  "num_sessions": 500,
  "cold_start_user_ratio": 0.1,
  "cold_start_item_ratio": 0.1,
  "start_date": "2026-01-01T00:00:00+00:00",
  "end_date": "2026-03-31T23:59:59+00:00",
  "serp_size": 20,
  "position_decay_alpha": 0.5,
  "relevance_boost": 2.0,
  "preference_boost": 1.5
}
```

## 7. Data Formats

### CSV Encoding

- **Encoding**: UTF-8
- **Delimiter**: comma
- **Quoting**: `csv.QUOTE_MINIMAL` (via `csv.DictWriter`)
- **Header row**: present in all files

### Nested Fields (JSON Strings)

`preferred_categories` and `preferred_brands` in `users.csv` are JSON-encoded
string arrays. Parse with:

```python
import json
cats = json.loads(row["preferred_categories"])  # → list[str]
```

### Missing Values

Numeric fields irrelevant to an event type are left as empty strings (not
`null` or `NaN`). Example: `click_duration_ms` is empty for impression events.

| Field | Present for event types |
|-------|------------------------|
| `click_duration_ms` | `click` |
| `add_to_cart_quantity` | `add_to_cart` |
| `purchase_amount` | `purchase` |

## 8. Qrels (Relevance Judgments)

### Purpose

Qrels provide ground-truth relevance labels for offline retrieval evaluation.
They are **completely independent** of user behaviour events — no click data,
position bias, or popularity leakage.

### Generation Method

For each (query, item) pair, compute a relevance score from objective signals:

| Signal | Points |
|--------|--------|
| `item.category == query.intended_category` | +1 |
| `item.subcategory` appears in query text | +1 |
| `item.brand` appears in query text | +1 |
| ≥2 keyword overlap between query text and item title | +1 |
| ≥3 keyword overlap between semantic_intent and description | +1 |

Score → grade mapping:
- ≥3 → grade 3 (highly relevant)
- ≥2 → grade 2 (relevant)
- ≥1 → grade 1 (weakly relevant)
- =0 → not stored (sparse format)

### Coverage Guarantee

After scoring, a **fallback pass** ensures every query has at least one grade-2+
item by selecting the best keyword-matching item from the query's intended
category (or any category as a last resort).

### Qrels vs Events

| Property | Qrels | Events |
|----------|-------|--------|
| Based on | Query-item text semantics | User browsing simulation |
| Depends on sessions? | No | Yes |
| Depends on clicks? | No | Yes |
| Seed-reproducible? | Yes | Yes |
| Purpose | Retrieval evaluation | Personalization training |

## 9. Quality Assurance

The data quality validator (`validation.py`) checks:

1. **Uniqueness**: all primary keys are unique
2. **Referential integrity**: all event foreign keys resolve
3. **Domain constraints**: scores in [0,1], price > 0, position ≥ 1
4. **Enum validity**: event_type in allowed set
5. **Timestamp format**: ISO 8601 with UTC timezone
6. **Cold-start presence**: cold-start users and items exist
7. **Funnel ordering**: impression > click > favorite/add_to_cart > purchase
8. **Business rules**: preference CTR > non-preference CTR, position CTR decay,
   quality-conversion correlation

## 9. Extending the Generator

To add new data patterns:

1. **New fields**: Add to `schemas.py` field lists and `generator.py` entity classes
2. **New business rules**: Add probability functions in `generator.py`, validate in `validation.py`
3. **New event types**: Add to `schemas.EVENT_TYPES` and handle in the session loop
4. **Scale changes**: Edit `configs/sample.json` — all entity counts are configurable
