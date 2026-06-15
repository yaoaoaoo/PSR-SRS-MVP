#!/usr/bin/env python3
"""Generate synthetic e-commerce search & ranking data for the PSR-SRS MVP.

Usage::

    .venv/Scripts/python.exe scripts/generate_data.py \\
        --config configs/sample.json \\
        --output data/sample
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the *project* src/ is on sys.path so imports work from any CWD.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from collections import Counter

from psr_srs_mvp.data_generation import (
    DataGenerator,
    load_config,
    validate_generated_data,
    validate_qrels,
    write_csv_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic e-commerce search & ranking data"
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow overwriting existing data files"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run comprehensive data quality validation after generation"
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Write data manifest JSON to this path"
    )
    args = parser.parse_args()

    # Check overwrite protection
    out_dir = Path(args.output)
    existing = list(out_dir.glob("*.csv")) if out_dir.exists() else []
    if existing and not args.force:
        print(f"[ABORT] Output directory {out_dir} already contains {len(existing)} CSV file(s).")
        print(f"  Use --force to overwrite.")
        sys.exit(1)

    # 1. Load config
    print(f"[1/6] Loading config: {args.config}")
    cfg = load_config(args.config)

    # 2. Generate data
    print(f"[2/6] Generating data (seed={cfg.seed})…")
    gen = DataGenerator(cfg)
    data = gen.generate_all()

    # 3. Validate events
    print("[3/6] Validating events…")
    errors = validate_generated_data(
        data["items"], data["users"], data["queries"], data["events"],
        cfg.num_items, cfg.num_users, cfg.num_queries,
        cfg.max_clicks_per_request,
    )
    if errors:
        print(f"\n  [FAIL] {len(errors)} validation error(s):")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
        sys.exit(1)
    print("  [PASS] Event validations passed")

    # 4. Validate qrels
    print("[4/6] Validating qrels…")
    qrels_errors = validate_qrels(
        data["qrels"],
        {r["query_id"] for r in data["queries"]},
        {r["item_id"] for r in data["items"]},
    )
    if qrels_errors:
        print(f"\n  [FAIL] {len(qrels_errors)} qrels validation error(s):")
        for e in qrels_errors[:20]:
            print(f"    - {e}")
        sys.exit(1)
    print("  [PASS] Qrels validations passed")

    # 5. Write CSV
    print(f"[5/6] Writing CSV to: {out_dir}")
    paths = write_csv_files(data, out_dir)
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {p.name} ({size_kb:.1f} KB)")

    # 6. Summary
    print("[6/6] Summary")
    s = gen.summary()

    # Compute derived statistics
    events_list = data["events"]
    imp_count = sum(1 for e in events_list if e["event_type"] == "impression")
    clk_count = sum(1 for e in events_list if e["event_type"] == "click")
    fav_count = sum(1 for e in events_list if e["event_type"] == "favorite")
    atc_count = sum(1 for e in events_list if e["event_type"] == "add_to_cart")
    pur_count = sum(1 for e in events_list if e["event_type"] == "purchase")

    # Count sessions that actually produced events
    sessions_with_events = len({e["session_id"] for e in events_list})

    ctr = clk_count / imp_count * 100 if imp_count > 0 else 0
    avg_clicks = clk_count / sessions_with_events if sessions_with_events > 0 else 0
    fav_ratio = fav_count / clk_count * 100 if clk_count > 0 else 0
    atc_ratio = atc_count / clk_count * 100 if clk_count > 0 else 0
    pur_click_ratio = pur_count / clk_count * 100 if clk_count > 0 else 0
    pur_atc_ratio = pur_count / atc_count * 100 if atc_count > 0 else 0

    # Qrels grade distribution
    qrels_data = data["qrels"]
    grade_counts = Counter(r["relevance_grade"] for r in qrels_data)
    queries_with_qrels = len({r["query_id"] for r in qrels_data})

    print(f"  Items:              {s['num_items']}")
    print(f"  Users:              {s['num_users']}")
    print(f"  Queries:            {s['num_queries']}")
    print(f"  Sessions w/ events: {sessions_with_events}")
    print(f"  Events:             {s['num_events']}")
    print(f"  Qrels rows:         {len(qrels_data)}")
    print(f"  Queries w/ qrels:   {queries_with_qrels}")
    print(f"  Cold-start users:   {s['cold_start_users']}")
    print(f"  Cold-start items:   {s['cold_start_items']}")
    print()
    print(f"  Event counts:")
    print(f"    impression        {imp_count:6d}")
    print(f"    click             {clk_count:6d}")
    print(f"    favorite          {fav_count:6d}")
    print(f"    add_to_cart       {atc_count:6d}")
    print(f"    purchase          {pur_count:6d}")
    print()
    print(f"  Behaviour stats:")
    print(f"    impression-level CTR:    {ctr:.1f}%")
    print(f"    avg clicks / session:    {avg_clicks:.2f}")
    print(f"    favorite / click:        {fav_ratio:.1f}%")
    print(f"    add_to_cart / click:     {atc_ratio:.1f}%")
    print(f"    purchase / click:        {pur_click_ratio:.1f}%")
    print(f"    purchase / add_to_cart:  {pur_atc_ratio:.1f}%")
    print()
    print(f"  Qrels grade distribution:")
    for g in ("3", "2", "1"):
        print(f"    grade {g}:              {grade_counts.get(g, 0):6d}")

    # Manifest
    if args.manifest:
        import hashlib
        manifest: dict = {
            "schema_version": "1.0.0", "generator_version": "0.1.0",
            "seed": cfg.seed, "config_file": str(args.config),
            "output_dir": str(out_dir), "files": {},
        }
        for p in paths:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest["files"][p.name] = {"rows": len(data.get(p.name.replace(".csv",""), [])),
                                          "sha256": sha, "size_bytes": p.stat().st_size}
        manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  [OK] Manifest: {manifest_path}")

    # Optional comprehensive validation
    if args.validate:
        from scripts.validate_data import validate_data
        report = validate_data(out_dir, cfg.num_sessions)
        print(f"\n  [Validate] Checks={len(report.checks)} "
              f"passed={len(report.infos)} warnings={len(report.warnings)} "
              f"errors={len(report.errors)}")
        if not report.passed:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
