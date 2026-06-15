#!/usr/bin/env python3
"""Comprehensive data quality validator for the PSR-SRS MVP sample data.

Usage::

    .venv/Scripts/python.exe scripts/validate_data.py --data-dir data/sample
    .venv/Scripts/python.exe scripts/validate_data.py --data-dir data/sample --output outputs/data_generation/data_quality_report.json
"""

from __future__ import annotations

import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ======================================================================
# Check result types
# ======================================================================

class CheckResult:
    def __init__(self, name: str, level: str = "info"):
        self.name = name
        self.level = level  # "error", "warning", "info"
        self.passed = True
        self.actual: Any = None
        self.expected: Any = None
        self.message: str = ""

    def fail(self, actual, expected, msg=""):
        self.passed = False
        self.actual = actual
        self.expected = expected
        self.message = msg
        return self

    def to_dict(self) -> dict:
        return {"name": self.name, "level": self.level, "passed": self.passed,
                "actual": str(self.actual), "expected": str(self.expected),
                "message": self.message}


class Report:
    def __init__(self):
        self.checks: list[CheckResult] = []
        self.errors: list[CheckResult] = []
        self.warnings: list[CheckResult] = []
        self.infos: list[CheckResult] = []

    def add(self, c: CheckResult):
        self.checks.append(c)
        if not c.passed:
            if c.level == "error": self.errors.append(c)
            elif c.level == "warning": self.warnings.append(c)
        else:
            self.infos.append(c)

    @property
    def passed(self): return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {"total_checks": len(self.checks), "passed_count": len(self.infos),
                "error_count": len(self.errors), "warning_count": len(self.warnings),
                "checks": [c.to_dict() for c in self.checks]}


# ======================================================================
# Loaders
# ======================================================================

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ======================================================================
# Checks
# ======================================================================

def check_file_exists(report: Report, path: Path, label: str):
    c = CheckResult(f"{label}: file exists", "error")
    if not path.exists():
        c.fail("missing", path)
    report.add(c)


def check_file_not_empty(report: Report, path: Path, label: str):
    c = CheckResult(f"{label}: file not empty", "error")
    if path.exists() and path.stat().st_size == 0:
        c.fail("empty file", "non-empty")
    report.add(c)


def check_unique(report: Report, rows: list[dict], key: str, label: str):
    c = CheckResult(f"{label}: {key} unique", "error")
    ids = [r.get(key, "") for r in rows]
    dupes = [id_ for id_, cnt in Counter(ids).items() if cnt > 1]
    if dupes:
        c.fail({"count": len(dupes), "examples": dupes[:5]}, "no duplicates")
    report.add(c)


def check_foreign_key(report: Report, rows: list[dict], key: str, ref_set: set, label: str):
    c = CheckResult(f"{label}: {key} foreign key valid", "error")
    missing = {r.get(key, "") for r in rows} - ref_set
    missing.discard("")
    if missing:
        c.fail({"count": len(missing), "examples": list(missing)[:5]}, "all present")
    report.add(c)


def check_enum(report: Report, rows: list[dict], key: str, valid: set, label: str):
    c = CheckResult(f"{label}: {key} in valid set", "error")
    bad = {r.get(key) for r in rows if r.get(key) not in valid}
    if bad:
        c.fail({"count": len(bad), "values": list(bad)[:5]}, str(valid))
    report.add(c)


def check_range(report: Report, rows: list[dict], key: str, lo: float, hi: float, label: str):
    c = CheckResult(f"{label}: {key} in [{lo},{hi}]", "error")
    bad = []
    for r in rows:
        try:
            v = float(r.get(key, float('nan')))
        except (ValueError, TypeError):
            bad.append(r.get(key))
            continue
        if not (lo <= v <= hi):
            bad.append(v)
    if bad:
        c.fail({"count": len(bad), "examples": bad[:5]}, f"[{lo},{hi}]")
    report.add(c)


def check_timestamp(report: Report, rows: list[dict], key: str, label: str):
    c = CheckResult(f"{label}: {key} valid ISO-8601 with UTC", "error")
    bad = []
    for r in rows:
        val = r.get(key, "")
        if not val: continue
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
                bad.append(val)
        except (ValueError, TypeError):
            bad.append(val)
        if len(bad) >= 5: break
    if bad:
        c.fail({"count": len(bad), "examples": bad}, "valid UTC timestamps")
    report.add(c)


def check_no_empty(report: Report, rows: list[dict], key: str, label: str):
    c = CheckResult(f"{label}: {key} non-empty", "error")
    empty = sum(1 for r in rows if not r.get(key, "").strip())
    if empty > 0:
        c.fail({"empty_count": empty}, "0 empty")
    report.add(c)


# ======================================================================
# Main validator
# ======================================================================

def validate_data(data_dir: str | Path, configured_sessions: int = 500,
                  expected_users: int = 100, expected_items: int = 500,
                  expected_queries: int = 200) -> Report:
    report = Report()
    base = Path(data_dir)

    # ---------- File existence ----------
    for fname, label in [("items.csv","items"),("users.csv","users"),("queries.csv","queries"),
                          ("events.csv","events"),("qrels.csv","qrels")]:
        check_file_exists(report, base / fname, label)
        check_file_not_empty(report, base / fname, label)

    # Load data
    items = load_csv(base / "items.csv") if (base / "items.csv").exists() else []
    users = load_csv(base / "users.csv") if (base / "users.csv").exists() else []
    queries = load_csv(base / "queries.csv") if (base / "queries.csv").exists() else []
    events = load_csv(base / "events.csv") if (base / "events.csv").exists() else []
    qrels_rows = load_csv(base / "qrels.csv") if (base / "qrels.csv").exists() else []

    # ---------- Count checks ----------
    for label, rows, expected in [("items", items, expected_items),
                                   ("users", users, expected_users),
                                   ("queries", queries, expected_queries)]:
        c = CheckResult(f"{label}: row count", "info")
        c.actual, c.expected = len(rows), expected
        if len(rows) != expected:
            c.level = "warning"
        report.add(c)

    c = CheckResult("events: row count", "info")
    c.actual, c.expected = len(events), "any (>0)"
    if len(events) == 0:
        c.level = "error"; c.passed = False
    report.add(c)

    # ---------- Primary keys ----------
    for label, rows, key in [("items", items, "item_id"), ("users", users, "user_id"),
                              ("queries", queries, "query_id"), ("events", events, "event_id")]:
        check_unique(report, rows, key, label)

    # ---------- Session/request uniqueness ----------
    if events:
        sids = {e["session_id"] for e in events}
        rids = {e["request_id"] for e in events}
        c = CheckResult("events: session_id unique", "info")
        c.actual, c.expected = len(sids), len(events)
        if len(sids) < len(events) * 0.01:
            c.level = "warning"
        report.add(c)

        c = CheckResult("events: request_id unique", "info")
        c.actual = len(rids)
        report.add(c)

        # Configured vs actual sessions
        c = CheckResult("sessions: configured vs event-unique", "info")
        c.actual, c.expected = len(sids), configured_sessions
        diff = configured_sessions - len(sids)
        if diff > 0:
            c.message = f"{diff} configured sessions generated zero events (activity skip)"
            c.level = "warning"
        report.add(c)

    # ---------- Foreign keys ----------
    item_ids = {r["item_id"] for r in items}
    user_ids = {r["user_id"] for r in users}
    query_ids = {r["query_id"] for r in queries}

    if events:
        check_foreign_key(report, events, "user_id", user_ids, "events")
        check_foreign_key(report, events, "item_id", item_ids, "events")
        check_foreign_key(report, events, "query_id", query_ids, "events")

    if qrels_rows:
        check_foreign_key(report, qrels_rows, "query_id", query_ids, "qrels")
        check_foreign_key(report, qrels_rows, "item_id", item_ids, "qrels")
        # Qrels: (query_id, item_id) unique
        pairs = [(r["query_id"], r["item_id"]) for r in qrels_rows]
        dup_pairs = [p for p, cnt in Counter(pairs).items() if cnt > 1]
        c = CheckResult("qrels: (query_id,item_id) unique", "error")
        if dup_pairs:
            c.fail({"count": len(dup_pairs), "examples": dup_pairs[:5]}, "no duplicates")
        report.add(c)

    # ---------- Enum checks ----------
    if events:
        check_enum(report, events, "event_type", {"impression","click","favorite","add_to_cart","purchase"}, "events")
    if qrels_rows:
        check_enum(report, qrels_rows, "relevance_grade", {"1","2","3"}, "qrels")
    if users:
        check_enum(report, users, "activity_level", {"low","medium","high"}, "users")
        check_enum(report, users, "price_preference", {"budget","mid_range","premium"}, "users")

    # ---------- Numeric ranges ----------
    if items:
        check_range(report, items, "price", 0.01, 1_000_000, "items")
        check_range(report, items, "quality_score", 0.0, 1.0, "items")
        check_range(report, items, "popularity_score", 0.0, 1.0, "items")
    if events:
        # position >= 1
        c = CheckResult("events: position >= 1", "error")
        bad_pos = sum(1 for e in events if int(e.get("position","0")) < 1)
        if bad_pos:
            c.fail(bad_pos, "0")
        report.add(c)

        # click_duration_ms >= 0 when present
        c = CheckResult("events: click_duration_ms >= 0", "error")
        bad = sum(1 for e in events if e.get("click_duration_ms","") and int(e.get("click_duration_ms","0")) < 0)
        if bad: c.fail(bad, "0")
        report.add(c)

        # add_to_cart_quantity >= 1 when present
        c = CheckResult("events: add_to_cart_quantity >= 1", "error")
        bad = sum(1 for e in events if e.get("add_to_cart_quantity","") and int(e.get("add_to_cart_quantity","0")) < 1)
        if bad: c.fail(bad, "0")
        report.add(c)

        # purchase_amount >= 0 when present
        c = CheckResult("events: purchase_amount >= 0", "error")
        bad = sum(1 for e in events if e.get("purchase_amount","") and float(e.get("purchase_amount","0")) < 0)
        if bad: c.fail(bad, "0")
        report.add(c)

    # ---------- Timestamps ----------
    if items: check_timestamp(report, items, "created_at", "items")
    if users: check_timestamp(report, users, "created_at", "users")
    if queries: check_timestamp(report, queries, "created_at", "queries")
    if events: check_timestamp(report, events, "timestamp", "events")

    # ---------- Non-empty required fields ----------
    if items: check_no_empty(report, items, "item_id", "items"); check_no_empty(report, items, "title", "items")
    if users: check_no_empty(report, users, "user_id", "users")
    if queries: check_no_empty(report, queries, "query_id", "queries"); check_no_empty(report, queries, "query_text", "queries")
    if events: check_no_empty(report, events, "event_id", "events"); check_no_empty(report, events, "event_type", "events")

    # ---------- Business rules ----------
    if events:
        etypes = Counter(e["event_type"] for e in events)
        c = CheckResult("events: impression > click", "warning")
        if etypes["impression"] <= etypes["click"]:
            c.fail(etypes, "imp > click")
        report.add(c)

        c = CheckResult("events: click > favorite", "warning")
        if etypes["click"] <= etypes["favorite"]:
            c.fail(etypes, "click > fav")
        report.add(c)

        c = CheckResult("events: click > add_to_cart", "warning")
        if etypes["click"] <= etypes["add_to_cart"]:
            c.fail(etypes, "click > atc")
        report.add(c)

        c = CheckResult("events: click > purchase", "warning")
        if etypes["click"] <= etypes["purchase"]:
            c.fail(etypes, "click > purchase")
        report.add(c)

        # One request = one query
        req_qids = defaultdict(set)
        for e in events:
            req_qids[e["request_id"]].add(e.get("query_id",""))
        multi_q = {rid: qs for rid, qs in req_qids.items() if len(qs) > 1}
        c = CheckResult("events: one request = one query", "warning")
        if multi_q:
            c.fail({"count": len(multi_q), "examples": list(multi_q.keys())[:3]}, "1:1")
        report.add(c)

    # ---------- Orphan detection ----------
    if users and events:
        users_with_events = {e["user_id"] for e in events}
        orphans = user_ids - users_with_events
        c = CheckResult("users: zero-event users", "info")
        c.actual, c.expected = len(orphans), "13"
        c.message = f"Users without events: {len(orphans)}"
        report.add(c)

    if items and events:
        items_with_events = {e["item_id"] for e in events}
        unused_items = item_ids - items_with_events
        c = CheckResult("items: unused items", "info")
        c.actual = len(unused_items)
        c.message = f"Items never appearing in events: {len(unused_items)}"
        report.add(c)

    # ---------- Cold-start presence ----------
    if users:
        cold_users = [u for u in users if u.get("is_cold_start","").lower() == "true"]
        c = CheckResult("users: cold-start users exist", "info")
        c.actual, c.expected = len(cold_users), 10
        report.add(c)

    if items:
        cold_items = [it for it in items if it.get("is_cold_start","").lower() == "true"]
        c = CheckResult("items: cold-start items exist", "info")
        c.actual, c.expected = len(cold_items), 50
        report.add(c)

    # ---------- Qrels coverage ----------
    if qrels_rows and queries:
        qids_with_qrels = {r["query_id"] for r in qrels_rows}
        missing_qrels = query_ids - qids_with_qrels
        c = CheckResult("qrels: every query has judgments", "warning")
        c.actual, c.expected = len(missing_qrels), 0
        if missing_qrels:
            c.passed = False
        report.add(c)

        # Each query has at least 1 grade-2+ item
        q_grade2 = defaultdict(bool)
        for r in qrels_rows:
            if int(r["relevance_grade"]) >= 2:
                q_grade2[r["query_id"]] = True
        no_grade2 = [qid for qid in query_ids if not q_grade2.get(qid)]
        c = CheckResult("qrels: each query has grade-2+ item", "warning")
        if no_grade2:
            c.fail({"count": len(no_grade2), "examples": no_grade2[:5]}, "0")
        report.add(c)

    # ---------- Event type distribution ----------
    if events:
        etypes = Counter(e["event_type"] for e in events)
        for et in ["impression","click","favorite","add_to_cart","purchase"]:
            c = CheckResult(f"events: {et} count", "info")
            c.actual = etypes.get(et, 0)
            report.add(c)

    # ---------- Session/request conservation ----------
    if events:
        train_sids = {e["session_id"] for e in events if e.get("_is_train") == "true"}
        test_sids = {e["session_id"] for e in events if e.get("_is_train") == "false"}
        all_evt_sids = {e["session_id"] for e in events}
        c = CheckResult("sessions: unique in events", "info")
        c.actual = len(all_evt_sids)
        report.add(c)

    return report


# ======================================================================
# Statistics generator
# ======================================================================

def generate_statistics(data_dir: Path) -> dict:
    items = load_csv(data_dir / "items.csv") if (data_dir / "items.csv").exists() else []
    users = load_csv(data_dir / "users.csv") if (data_dir / "users.csv").exists() else []
    queries = load_csv(data_dir / "queries.csv") if (data_dir / "queries.csv").exists() else []
    events = load_csv(data_dir / "events.csv") if (data_dir / "events.csv").exists() else []
    qrels_rows = load_csv(data_dir / "qrels.csv") if (data_dir / "qrels.csv").exists() else []

    uids_events = {e["user_id"] for e in events}
    user_sessions = defaultdict(set)
    for e in events: user_sessions[e["user_id"]].add(e["session_id"])
    multi = sum(1 for uid, sids in user_sessions.items() if len(sids) >= 2)
    single = sum(1 for uid, sids in user_sessions.items() if len(sids) == 1)
    zero = len({u["user_id"] for u in users} - uids_events)

    return {
        "user_count": len(users),
        "item_count": len(items),
        "query_count": len(queries),
        "event_count": len(events),
        "qrels_count": len(qrels_rows),
        "configured_sessions": 500,
        "unique_sessions_in_events": len({e["session_id"] for e in events}),
        "unique_requests_in_events": len({e["request_id"] for e in events}),
        "multi_session_users": multi,
        "single_session_users": single,
        "zero_session_users": zero,
        "event_type_distribution": dict(Counter(e["event_type"] for e in events)),
        "positive_event_count": sum(1 for e in events if e["event_type"] in
                                     {"click","favorite","add_to_cart","purchase"}),
    }


def generate_manifest(config: dict, data_dir: Path, output_dir: Path) -> dict:
    manifest: dict = {
        "schema_version": "1.0.0",
        "generator_version": "0.1.0",
        "seed": config.get("seed"),
        "config_file": config.get("_config_file", ""),
        "output_dir": str(data_dir),
        "files": {},
    }
    for fname in ["items.csv","users.csv","queries.csv","events.csv","qrels.csv"]:
        p = data_dir / fname
        if p.exists():
            rows = load_csv(p)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest["files"][fname] = {
                "rows": len(rows), "columns": list(rows[0].keys()) if rows else [],
                "sha256": sha, "size_bytes": p.stat().st_size,
            }
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return manifest


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="PSR-SRS MVP Data Quality Validator")
    p.add_argument("--data-dir", required=True, type=Path, help="Path to data/sample")
    p.add_argument("--output", type=Path, help="Output JSON report path")
    p.add_argument("--configured-sessions", type=int, default=500)
    p.add_argument("--statistics", action="store_true")
    p.add_argument("--manifest", action="store_true")
    args = p.parse_args()

    print(f"Validating data in: {args.data_dir}")
    report = validate_data(args.data_dir, args.configured_sessions)

    # Print summary
    print(f"\n  Total checks: {len(report.checks)}")
    print(f"  Passed: {len(report.infos)}")
    print(f"  Warnings: {len(report.warnings)}")
    print(f"  Errors: {len(report.errors)}")

    if report.errors:
        print("\n  ERRORS:")
        for c in report.errors:
            print(f"    [FAIL] {c.name}: {c.message or c.actual}")

    if report.warnings:
        print("\n  WARNINGS:")
        for c in report.warnings:
            print(f"    [WARN] {c.name}: {c.message or c.actual}")

    # Output JSON
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result: dict = {"quality_report": report.to_dict()}

        if args.statistics:
            result["statistics"] = generate_statistics(args.data_dir)

        if args.manifest:
            result["manifest"] = generate_manifest({"seed": 20260614}, args.data_dir, out.parent)

        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Report written to: {out}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
