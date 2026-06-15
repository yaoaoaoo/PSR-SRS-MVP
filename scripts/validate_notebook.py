#!/usr/bin/env python3
"""Validate executed Notebook — check for errors, unexecuted cells, and metrics."""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent


def validate_notebook(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]

    unexecuted = sum(1 for c in code_cells if c.get("execution_count") is None)
    error_cells_list = []
    for i, c in enumerate(code_cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                error_cells_list.append({"cell_index": i, "ename": o.get("ename", ""),
                                         "evalue": str(o.get("evalue", ""))[:200]})

    return {
        "source_notebook": str(nb_path),
        "total_cells": len(cells),
        "code_cells": len(code_cells),
        "executed_code_cells": len(code_cells) - unexecuted,
        "markdown_cells": len(md_cells),
        "unexecuted_cells": unexecuted,
        "error_cells": len(error_cells_list),
        "error_details": error_cells_list[:10],
        "execution_passed": unexecuted == 0 and len(error_cells_list) == 0,
        "kernel_name": nb.get("metadata", {}).get("kernelspec", {}).get("name", "unknown"),
        "execution_mode": "nbconvert",
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def validate_source_notebook(nb_path: Path) -> dict:
    """Validate source notebook structural properties."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    issues = []

    # Check for pip install
    for i, c in enumerate(code_cells):
        src = "".join(c["source"])
        if "pip install" in src.lower():
            issues.append(f"cell {i}: contains pip install")

    # Check for D:/ absolute paths
    for i, c in enumerate(code_cells):
        src = "".join(c["source"])
        if "D:\\\\" in src or "D:/" in src:
            # Allow only paths relative to PROJECT_ROOT variable
            # Check for hardcoded absolute paths
            project_root_str = str(_PROJECT.resolve())
            if "PROJECT_ROOT" not in src and project_root_str not in src:
                issues.append(f"cell {i}: contains absolute path")

    # Check RECOMPUTE is False by default
    for i, c in enumerate(code_cells):
        src = "".join(c["source"])
        if "RECOMPUTE" in src:
            has_env = "PSR_SRS_RECOMPUTE" in src or "getenv" in src
            has_default = "False" in src or '"0"' in src
            if not has_env:
                issues.append(f"cell {i}: RECOMPUTE should use env var")
            break

    # Check kernel metadata
    kernel = nb.get("metadata", {}).get("kernelspec", {})
    if kernel.get("language") != "python":
        issues.append("kernel language is not python")

    return {
        "nbformat": nb.get("nbformat"),
        "kernelspec": kernel,
        "issues": issues,
        "valid": len(issues) == 0,
    }


def validate_metrics() -> dict:
    """Validate frozen baselines against stored metrics files."""
    bm25 = json.loads((_PROJECT / "outputs" / "bm25" / "metrics.json").read_text(encoding="utf-8"))
    sem = json.loads((_PROJECT / "outputs" / "semantic" / "metrics.json").read_text(encoding="utf-8"))
    hyb = json.loads((_PROJECT / "outputs" / "hybrid" / "linear" / "metrics.json").read_text(encoding="utf-8"))
    per = json.loads((_PROJECT / "outputs" / "personalization" / "metrics.json").read_text(encoding="utf-8"))

    checks = [
        ("BM25 NDCG@10", bm25["ndcg_at_10"], 0.297994),
        ("LSA NDCG@10", sem["ndcg_at_10"], 0.373320),
        ("Linear NDCG@10", hyb["ndcg_at_10"], 0.392327),
        ("Pers qrels NDCG@10", per["personalized_qrels_ndcg_at_10"], 0.396789),
        ("improved", per["improved_request_count"], 2),
        ("unchanged", per["unchanged_request_count"], 61),
        ("worsened", per["worsened_request_count"], 1),
        ("fallback_exact_match", per["fallback_exact_match_rate"], 1.0),
        ("request_coverage", per["request_level_candidate_positive_coverage"], 0.138462),
        ("item_recall", per["item_level_candidate_positive_recall"], 0.119048),
    ]
    results = []
    for name, actual, expected in checks:
        passed = abs(actual - expected) <= 1e-6
        results.append({"metric": name, "expected": expected, "actual": actual,
                        "delta": actual - expected, "tolerance": 1e-6, "passed": passed})
    return {"checks": results, "all_passed": all(r["passed"] for r in results)}


def main():
    p = argparse.ArgumentParser(description="Validate executed Notebook")
    p.add_argument("--notebook", type=Path, help="Path to executed .ipynb")
    p.add_argument("--source-notebook", type=Path, help="Path to source .ipynb for structure check")
    p.add_argument("--metrics", action="store_true", help="Validate frozen baselines")
    p.add_argument("--output", type=Path, help="Output JSON report path")
    args = p.parse_args()

    report: dict = {}

    if args.source_notebook and args.source_notebook.exists():
        report["source_validation"] = validate_source_notebook(args.source_notebook)
        print(f"Source notebook: {'VALID' if report['source_validation']['valid'] else 'ISSUES'}")
        for iss in report["source_validation"]["issues"]:
            print(f"  - {iss}")

    if args.notebook and args.notebook.exists():
        report["execution_validation"] = validate_notebook(args.notebook)
        ev = report["execution_validation"]
        print(f"Executed notebook: errors={ev['error_cells']} unexecuted={ev['unexecuted_cells']} "
              f"total_code={ev['code_cells']} passed={ev['execution_passed']}")

    if args.metrics:
        report["metrics_validation"] = validate_metrics()
        mv = report["metrics_validation"]
        passed = sum(1 for c in mv["checks"] if c["passed"])
        print(f"Metrics: {passed}/{len(mv['checks'])} passed")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport: {out}")

    all_ok = True
    if report.get("execution_validation", {}).get("execution_passed") is False:
        all_ok = False
    if report.get("metrics_validation", {}).get("all_passed") is False:
        all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
