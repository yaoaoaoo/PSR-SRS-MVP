"""Data-quality validation for generated synthetic CSVs.

All checks return lists of error strings.  An empty list means the data passed.
Designed to be called both from the CLI and from unit tests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from psr_srs_mvp.data_generation.schemas import EVENT_TYPES


# ---------------------------------------------------------------------------
# Low-level field checks
# ---------------------------------------------------------------------------

def _check_unique(rows: list[dict[str, str]], key: str, label: str) -> list[str]:
    ids = [r[key] for r in rows]
    dupes = [id_ for id_, cnt in Counter(ids).items() if cnt > 1]
    if dupes:
        return [f"{label}.{key}: duplicate values found: {dupes[:5]}"]
    return []


def _check_no_empty(rows: list[dict[str, str]], keys: list[str], label: str) -> list[str]:
    errors: list[str] = []
    for k in keys:
        if any(not r.get(k, "").strip() for r in rows):
            errors.append(f"{label}.{k}: contains empty values")
    return errors


def _check_float_range(
    rows: list[dict[str, str]], key: str, lo: float, hi: float, label: str
) -> list[str]:
    errors: list[str] = []
    for i, r in enumerate(rows):
        try:
            v = float(r[key])
        except (ValueError, KeyError):
            errors.append(f"{label}[{i}].{key}: not a valid float: {r.get(key)!r}")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{label}[{i}].{key}: {v} not in [{lo}, {hi}]")
            if len(errors) >= 10:
                break
    return errors


def _check_timestamp_iso(
    rows: list[dict[str, str]], key: str, label: str
) -> list[str]:
    errors: list[str] = []
    for i, r in enumerate(rows):
        val = r.get(key, "")
        if not val:
            continue
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                errors.append(f"{label}[{i}].{key}: missing timezone: {val}")
            elif dt.utcoffset().total_seconds() != 0:
                errors.append(f"{label}[{i}].{key}: not UTC: {val}")
        except (ValueError, TypeError):
            errors.append(f"{label}[{i}].{key}: invalid ISO timestamp: {val!r}")
        if len(errors) >= 10:
            break
    return errors


# ---------------------------------------------------------------------------
# Entity-level checks
# ---------------------------------------------------------------------------

def validate_items(
    rows: list[dict[str, str]], config_item_count: int
) -> list[str]:
    errors: list[str] = []
    label = "items"

    errors += _check_unique(rows, "item_id", label)
    errors += _check_no_empty(rows, ["item_id", "title", "category"], label)
    errors += _check_float_range(rows, "price", 0.01, 1_000_000, label)
    errors += _check_float_range(rows, "quality_score", 0.0, 1.0, label)
    errors += _check_float_range(rows, "popularity_score", 0.0, 1.0, label)
    errors += _check_timestamp_iso(rows, "created_at", label)

    # Cold-start items exist
    cold = [r for r in rows if r.get("is_cold_start", "").lower() == "true"]
    if not cold:
        errors.append(f"{label}: no cold-start items found")

    # Count matches config
    if len(rows) != config_item_count:
        errors.append(f"{label}: expected {config_item_count} rows, got {len(rows)}")

    return errors


def validate_users(
    rows: list[dict[str, str]], config_user_count: int
) -> list[str]:
    errors: list[str] = []
    label = "users"

    errors += _check_unique(rows, "user_id", label)
    errors += _check_no_empty(rows, ["user_id"], label)
    errors += _check_timestamp_iso(rows, "created_at", label)

    # Cold-start users exist
    cold = [r for r in rows if r.get("is_cold_start", "").lower() == "true"]
    if not cold:
        errors.append(f"{label}: no cold-start users found")

    if len(rows) != config_user_count:
        errors.append(f"{label}: expected {config_user_count} rows, got {len(rows)}")

    return errors


def validate_queries(
    rows: list[dict[str, str]], config_query_count: int
) -> list[str]:
    errors: list[str] = []
    label = "queries"

    errors += _check_unique(rows, "query_id", label)
    errors += _check_no_empty(rows, ["query_id", "query_text"], label)
    errors += _check_timestamp_iso(rows, "created_at", label)

    if len(rows) != config_query_count:
        errors.append(f"{label}: expected {config_query_count} rows, got {len(rows)}")

    return errors


def validate_events(
    rows: list[dict[str, str]],
    item_ids: set[str],
    user_ids: set[str],
    query_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    label = "events"

    errors += _check_unique(rows, "event_id", label)
    errors += _check_no_empty(
        rows, ["event_id", "event_type", "user_id", "item_id"], label
    )

    # event_type in allowed set
    for i, r in enumerate(rows):
        if r["event_type"] not in EVENT_TYPES:
            errors.append(f"{label}[{i}].event_type: invalid value {r['event_type']!r}")
            if len(errors) >= 10:
                break

    # Foreign key checks
    for i, r in enumerate(rows):
        if r["user_id"] not in user_ids:
            errors.append(f"{label}[{i}]: user_id {r['user_id']} not found in users")
        if r["item_id"] not in item_ids:
            errors.append(f"{label}[{i}]: item_id {r['item_id']} not found in items")
        qid = r.get("query_id", "")
        if qid and qid not in query_ids:
            errors.append(f"{label}[{i}]: query_id {qid} not found in queries")
        if len(errors) >= 10:
            break

    # position >= 1
    for i, r in enumerate(rows):
        try:
            pos = int(r["position"])
            if pos < 1:
                errors.append(f"{label}[{i}].position: {pos} < 1")
        except (ValueError, KeyError):
            errors.append(f"{label}[{i}].position: invalid: {r.get('position')!r}")
        if len(errors) >= 10:
            break

    # Numeric field bounds (only when present)
    for i, r in enumerate(rows):
        dur = r.get("click_duration_ms", "")
        if dur:
            try:
                if int(dur) < 0:
                    errors.append(f"{label}[{i}].click_duration_ms: {dur} < 0")
            except ValueError:
                errors.append(f"{label}[{i}].click_duration_ms: not int: {dur!r}")

        qty = r.get("add_to_cart_quantity", "")
        if qty:
            try:
                if int(qty) < 1:
                    errors.append(f"{label}[{i}].add_to_cart_quantity: {qty} < 1")
            except ValueError:
                errors.append(f"{label}[{i}].add_to_cart_quantity: not int: {qty!r}")

        amt = r.get("purchase_amount", "")
        if amt:
            try:
                if float(amt) < 0:
                    errors.append(f"{label}[{i}].purchase_amount: {amt} < 0")
            except ValueError:
                errors.append(f"{label}[{i}].purchase_amount: not float: {amt!r}")

        if len(errors) >= 10:
            break

    # Timestamp validity
    errors += _check_timestamp_iso(rows, "timestamp", label)

    return errors


# ---------------------------------------------------------------------------
# Statistical / business-rule checks (with target ranges)
# ---------------------------------------------------------------------------

# Target ranges for behaviour stats
_STAT_TARGETS = {
    "ctr": (8.0, 20.0),          # impression-level CTR %
    "avg_clicks": (1.0, 3.0),     # avg clicks per session
    "fav_click": (5.0, 15.0),     # favorite / click %
    "atc_click": (4.0, 12.0),     # add_to_cart / click %
    "pur_click": (1.0, 5.0),      # purchase / click %
    "pur_atc": (15.0, 40.0),      # purchase / add_to_cart %
}


def _compute_stats(event_rows: list[dict[str, str]]) -> dict[str, float]:
    """Compute behaviour statistics from event rows."""
    etype_counts = Counter(r["event_type"] for r in event_rows)
    imp = etype_counts.get("impression", 0)
    clk = etype_counts.get("click", 0)
    fav = etype_counts.get("favorite", 0)
    atc = etype_counts.get("add_to_cart", 0)
    pur = etype_counts.get("purchase", 0)
    sessions = len({r["session_id"] for r in event_rows})

    return {
        "ctr": clk / imp * 100 if imp > 0 else 0,
        "avg_clicks": clk / sessions if sessions > 0 else 0,
        "fav_click": fav / clk * 100 if clk > 0 else 0,
        "atc_click": atc / clk * 100 if clk > 0 else 0,
        "pur_click": pur / clk * 100 if clk > 0 else 0,
        "pur_atc": pur / atc * 100 if atc > 0 else 0,
    }


def validate_statistical_targets(
    event_rows: list[dict[str, str]],
) -> list[str]:
    """Check that behavioural stats fall in target ranges."""
    errors: list[str] = []
    stats = _compute_stats(event_rows)
    for key, (lo, hi) in _STAT_TARGETS.items():
        val = stats[key]
        if not (lo <= val <= hi):
            errors.append(f"{key}: {val:.2f} not in target range [{lo}, {hi}]")
    return errors


def validate_event_dependencies(
    event_rows: list[dict[str, str]],
) -> list[str]:
    """Verify event dependency chains within each request.

    - purchase must have a preceding add_to_cart for the same (request, item)
    - favorite must have a preceding click
    - add_to_cart must have a preceding click
    - Timestamps within a chain must be non-decreasing
    """
    errors: list[str] = []

    # Index events by (request_id, item_id)
    from collections import defaultdict
    req_item_events: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for r in event_rows:
        key = (r["request_id"], r["item_id"])
        req_item_events[key].append(r)

    # Check each chain
    for (rid, iid), chain in req_item_events.items():
        chain.sort(key=lambda r: (r["timestamp"], _event_order(r["event_type"])))

        has_click = False
        has_atc = False
        prev_ts = ""

        for ev in chain:
            etype = ev["event_type"]

            # Timestamp monotonicity within chain
            if prev_ts and ev["timestamp"] < prev_ts:
                errors.append(
                    f"request={rid} item={iid}: {etype} timestamp "
                    f"({ev['timestamp']}) before previous ({prev_ts})"
                )
                if len(errors) >= 10:
                    return errors
            prev_ts = ev["timestamp"]

            if etype == "click":
                has_click = True
            elif etype == "favorite":
                if not has_click:
                    errors.append(
                        f"request={rid} item={iid}: favorite without preceding click"
                    )
            elif etype == "add_to_cart":
                if not has_click:
                    errors.append(
                        f"request={rid} item={iid}: add_to_cart without preceding click"
                    )
                has_atc = True
            elif etype == "purchase":
                if not has_atc:
                    errors.append(
                        f"request={rid} item={iid}: purchase without preceding add_to_cart"
                    )

            if len(errors) >= 10:
                return errors

    return errors


def validate_max_clicks_per_request(
    event_rows: list[dict[str, str]], max_clicks: int
) -> list[str]:
    """No request may exceed the configured max clicks."""
    errors: list[str] = []
    req_clicks: dict[str, int] = Counter()
    for r in event_rows:
        if r["event_type"] == "click":
            req_clicks[r["request_id"]] += 1
    for rid, cnt in req_clicks.items():
        if cnt > max_clicks:
            errors.append(f"request={rid}: {cnt} clicks exceeds max {max_clicks}")
            if len(errors) >= 10:
                break
    return errors


def validate_business_rules(
    event_rows: list[dict[str, str]],
    item_rows: list[dict[str, str]],
    user_rows: list[dict[str, str]],
    max_clicks_per_request: int = 3,
) -> list[str]:
    """All business-rule checks combined."""
    errors: list[str] = []

    # Funnel ordering (impression > click > all post-click events)
    etype_counts = Counter(r["event_type"] for r in event_rows)
    imp = etype_counts.get("impression", 0)
    clk = etype_counts.get("click", 0)
    if imp <= clk:
        errors.append(f"impressions ({imp}) <= clicks ({clk})")
    for post in ("favorite", "add_to_cart", "purchase"):
        if clk <= etype_counts.get(post, 0):
            errors.append(f"clicks ({clk}) <= {post} ({etype_counts.get(post, 0)})")

    # Statistical targets
    errors += validate_statistical_targets(event_rows)

    # Event dependencies
    errors += validate_event_dependencies(event_rows)

    # Max clicks per request
    errors += validate_max_clicks_per_request(event_rows, max_clicks_per_request)

    # Category-preference CTR
    import json
    items_by_id = {it["item_id"]: it for it in item_rows}
    user_pref_cats: dict[str, set[str]] = {}
    for u in user_rows:
        try:
            user_pref_cats[u["user_id"]] = set(json.loads(u["preferred_categories"]))
        except (json.JSONDecodeError, KeyError):
            user_pref_cats[u["user_id"]] = set()

    pref_imp = pref_clk = nonpref_imp = nonpref_clk = 0
    for r in event_rows:
        item_cat = items_by_id.get(r["item_id"], {}).get("category", "")
        is_pref = item_cat in user_pref_cats.get(r["user_id"], set())
        if r["event_type"] == "impression":
            if is_pref:
                pref_imp += 1
            else:
                nonpref_imp += 1
        elif r["event_type"] == "click":
            if is_pref:
                pref_clk += 1
            else:
                nonpref_clk += 1

    if pref_imp > 0 and nonpref_imp > 0:
        pref_ctr = pref_clk / pref_imp
        nonpref_ctr = nonpref_clk / nonpref_imp
        if pref_ctr <= nonpref_ctr:
            errors.append(f"preference CTR ({pref_ctr:.4f}) <= non-preference CTR ({nonpref_ctr:.4f})")

    # Position CTR decay
    pos_imp: dict[int, int] = defaultdict(int)
    pos_clk: dict[int, int] = defaultdict(int)
    for r in event_rows:
        p = int(r["position"])
        if r["event_type"] == "impression":
            pos_imp[p] += 1
        elif r["event_type"] == "click":
            pos_clk[p] += 1

    top_imp = sum(pos_imp.get(p, 0) for p in range(1, 6))
    top_clk = sum(pos_clk.get(p, 0) for p in range(1, 6))
    bot_imp = sum(pos_imp.get(p, 0) for p in range(15, 21))
    bot_clk = sum(pos_clk.get(p, 0) for p in range(15, 21))
    if top_imp > 0 and bot_imp > 0:
        if top_clk / top_imp <= bot_clk / bot_imp:
            errors.append("top-5 CTR <= bottom-5 CTR")

    return errors


# ---------------------------------------------------------------------------
# Qrels validation
# ---------------------------------------------------------------------------

def validate_qrels(
    qrels: list[dict[str, str]],
    query_ids: set[str],
    item_ids: set[str],
) -> list[str]:
    """Validate qrels data quality."""
    errors: list[str] = []
    label = "qrels"

    if not qrels:
        errors.append(f"{label}: empty — no relevance judgments generated")
        return errors

    # Unique (query_id, item_id) pairs
    pairs = [(r["query_id"], r["item_id"]) for r in qrels]
    seen: set[tuple[str, str]] = set()
    dupes = []
    for p in pairs:
        if p in seen:
            dupes.append(str(p))
        seen.add(p)
        if len(dupes) >= 5:
            break
    if dupes:
        errors.append(f"{label}: duplicate (query_id, item_id) pairs: {dupes}")

    # Foreign keys
    for i, r in enumerate(qrels):
        if r["query_id"] not in query_ids:
            errors.append(f"{label}[{i}]: unknown query_id {r['query_id']}")
        if r["item_id"] not in item_ids:
            errors.append(f"{label}[{i}]: unknown item_id {r['item_id']}")
        if len(errors) >= 10:
            break

    # Valid grades
    for i, r in enumerate(qrels):
        g = r.get("relevance_grade", "")
        if g not in ("1", "2", "3"):
            errors.append(f"{label}[{i}]: invalid relevance_grade {g!r}")
            if len(errors) >= 10:
                break

    # Every query must have at least one grade 2+ item
    query_has_grade2: dict[str, bool] = defaultdict(bool)
    for r in qrels:
        if int(r["relevance_grade"]) >= 2:
            query_has_grade2[r["query_id"]] = True

    missing = [qid for qid in query_ids if not query_has_grade2.get(qid)]
    if missing:
        errors.append(
            f"{label}: {len(missing)} queries have no grade-2+ item: {missing[:5]}"
        )

    return errors


def _event_order(etype: str) -> int:
    """Deterministic ordering for event types within a chain."""
    order_map = {"impression": 0, "click": 1, "favorite": 2, "add_to_cart": 3, "purchase": 4}
    return order_map.get(etype, 99)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def validate_generated_data(
    items: list[dict[str, str]],
    users: list[dict[str, str]],
    queries: list[dict[str, str]],
    events: list[dict[str, str]],
    config_item_count: int,
    config_user_count: int,
    config_query_count: int,
    max_clicks_per_request: int = 3,
) -> list[str]:
    """Run all data-quality checks and return a combined error list."""
    errors: list[str] = []
    errors += validate_items(items, config_item_count)
    errors += validate_users(users, config_user_count)
    errors += validate_queries(queries, config_query_count)

    item_ids = {r["item_id"] for r in items}
    user_ids = {r["user_id"] for r in users}
    query_ids = {r["query_id"] for r in queries}
    errors += validate_events(events, item_ids, user_ids, query_ids)

    errors += validate_business_rules(events, items, users, max_clicks_per_request)
    return errors
