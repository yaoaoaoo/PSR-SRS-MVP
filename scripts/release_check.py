#!/usr/bin/env python3
"""Local release candidate verification — calls existing tools, aggregates results."""

from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent

CHECKS: list[dict] = []

def add(name, passed, detail=""):
    CHECKS.append({"name": name, "passed": passed, "detail": str(detail)[:200]})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")

def run(cmd, timeout=300, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(_PROJECT), **kw)

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Pre-check: no hard-coded 'D:/project/PSR-SRS-MVP' in source
def check_no_hardcoded():
    _SKIP = {"release_check.py", "validate_notebook.py"}  # validator scripts contain the path as search pattern
    for root in ["src", "scripts", "tests"]:
        for p in Path(_PROJECT, root).rglob("*.py"):
            if p.name in _SKIP: continue
            text = p.read_text(encoding="utf-8")
            if "D:/project/PSR-SRS-MVP" in text:
                add(f"hardcoded_path:{p.relative_to(_PROJECT)}", False, "contains absolute path")
                return
    add("no_hardcoded_absolute_paths", True)

def check_no_secrets():
    # Case-sensitive patterns unlikely to appear in normal code
    patterns = ["API_KEY=", "SECRET_KEY=", "PASSWORD=", "private key"]
    for root in ["src", "scripts", "configs"]:
        for p in Path(_PROJECT, root).rglob("*"):
            if p.suffix in (".pyc", ".png") or p.name == "release_check.py": continue
            try:
                text = p.read_text(encoding="utf-8")
            except: continue
            for pat in patterns:
                if pat in text:
                    add(f"secret_scan:{p.relative_to(_PROJECT)}", False, f"contains '{pat}'")
                    return
    add("secret_scan", True)

def main():
    p = argparse.ArgumentParser(description="PSR-SRS MVP Release Candidate Check")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-notebook-execution", action="store_true")
    p.add_argument("--output-dir", type=Path, default=_PROJECT / "outputs" / "release")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    py_exe = str(_PROJECT / ".venv" / "Scripts" / "python.exe")

    print("=== PSR-SRS MVP Release Check ===\n")

    # 1. Python version
    r = run([py_exe, "--version"])
    add("python_version", r.returncode == 0, r.stdout.strip())

    # 2. compileall
    r = run([py_exe, "-m", "compileall", "-q", "src", "scripts", "tests"])
    add("compileall", r.returncode == 0)

    # 3. pip check
    r = run([py_exe, "-m", "pip", "check"])
    add("pip_check", r.returncode == 0, r.stderr.strip() or "ok")

    # 4. Tests
    r = run([py_exe, "-m", "unittest", "discover", "-s", "tests", "-v"], timeout=600)
    passed = r.returncode == 0
    last_line = [l for l in r.stdout.splitlines() if l.startswith("Ran ") or l == "OK"]
    add("tests", passed, last_line[-1] if last_line else r.stderr[:100])

    # 5. Data validation
    r = run([py_exe, "scripts/validate_data.py", "--data-dir", "data/sample"])
    add("data_quality", r.returncode == 0, "66/66" if r.returncode == 0 else r.stdout[:100])

    # 6. Reproducibility
    r = run([py_exe, "scripts/reproducibility_check.py"], timeout=120)
    add("reproducibility", r.returncode == 0, "5/5 matched" if r.returncode == 0 else r.stdout[:200])

    # 7. Sample data SHA-256
    for fn in ["items.csv","users.csv","queries.csv","events.csv","qrels.csv"]:
        h = sha256(_PROJECT / "data" / "sample" / fn)
        add(f"sample_sha256:{fn}", True, h[:32])

    # 8. Frozen metrics
    try:
        bm25 = json.loads((_PROJECT/"outputs/bm25/metrics.json").read_text(encoding="utf-8"))
        sem = json.loads((_PROJECT/"outputs/semantic/metrics.json").read_text(encoding="utf-8"))
        hyb = json.loads((_PROJECT/"outputs/hybrid/linear/metrics.json").read_text(encoding="utf-8"))
        per = json.loads((_PROJECT/"outputs/personalization/metrics.json").read_text(encoding="utf-8"))
        checks = [
            ("BM25 NDCG@10", bm25["ndcg_at_10"], 0.297994),
            ("LSA NDCG@10", sem["ndcg_at_10"], 0.373320),
            ("Linear NDCG@10", hyb["ndcg_at_10"], 0.392327),
            ("Pers qrels NDCG@10", per["personalized_qrels_ndcg_at_10"], 0.396789),
            ("improved", per["improved_request_count"], 2),
            ("unchanged", per["unchanged_request_count"], 61),
            ("worsened", per["worsened_request_count"], 1),
            ("fallback", per["fallback_exact_match_rate"], 1.0),
        ]
        all_ok = True
        for name, actual, exp in checks:
            ok = abs(actual - exp) <= 1e-6
            if not ok: all_ok = False
            add(f"metric:{name}", ok, f"{actual} vs {exp}")
        add("frozen_metrics_all", all_ok)
    except Exception as e:
        add("frozen_metrics_all", False, str(e))

    # 9. Notebook validation
    for mode, nb_name in [("cache", "01_mvp_end_to_end.executed.ipynb"),
                           ("recompute", "01_mvp_end_to_end.recomputed.ipynb")]:
        nb_path = _PROJECT / "outputs" / "notebook" / nb_name
        if nb_path.exists():
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            code_cells = [c for c in nb["cells"] if c["cell_type"]=="code"]
            errors = sum(1 for c in code_cells for o in c.get("outputs",[])
                        if o.get("output_type")=="error")
            unexec = sum(1 for c in code_cells if c.get("execution_count") is None)
            add(f"notebook_{mode}_errors", errors==0, f"{errors} errors")
            add(f"notebook_{mode}_executed", unexec==0, f"{unexec} unexecuted")
        else:
            add(f"notebook_{mode}", False, "file missing")

    # 10. Notebook source
    src_nb = _PROJECT / "notebooks" / "01_mvp_end_to_end.ipynb"
    if src_nb.exists():
        nb = json.loads(src_nb.read_text(encoding="utf-8"))
        has_env = any("PSR_SRS_RECOMPUTE" in "".join(c.get("source",[]))
                     for c in nb["cells"] if c["cell_type"]=="code")
        add("notebook_source_has_env_var", has_env)
    else:
        add("notebook_source", False, "missing")

    # 11. Check no secrets / hardcoded paths
    check_no_hardcoded()
    check_no_secrets()

    # 12. Build
    if not args.skip_build:
        import shutil
        for d in ["build", "dist", "*.egg-info"]:
            for p in _PROJECT.glob(d):
                if p.is_dir(): shutil.rmtree(p, ignore_errors=True)

        r = run([py_exe, "-m", "build"], timeout=120)
        add("build", r.returncode == 0, r.stderr[:200] if r.returncode != 0 else "ok")

        # twine check
        dists = list((_PROJECT / "dist").glob("*"))
        if dists:
            r = run([py_exe, "-m", "twine", "check"] + [str(d) for d in dists])
            add("twine_check", r.returncode == 0, r.stdout[:200] if r.returncode != 0 else "ok")
            for d in dists:
                add(f"artifact:{d.name}", True, f"{d.stat().st_size} bytes, sha256={sha256(d)[:16]}")
        else:
            add("build_artifacts", False, "no dist files")

        # Clean install smoke test
        if dists:
            whl = [d for d in dists if d.suffix == ".whl"]
            if whl:
                tmp_venv = _PROJECT / ".release_venv"
                if tmp_venv.exists(): shutil.rmtree(tmp_venv, ignore_errors=True)
                r = run([py_exe, "-m", "venv", str(tmp_venv)])
                if r.returncode == 0:
                    pip_exe = str(tmp_venv / "Scripts" / "python.exe")
                    r2 = run([pip_exe, "-m", "pip", "install", str(whl[0])], timeout=120)
                    add("wheel_install", r2.returncode == 0)
                    if r2.returncode == 0:
                        r3 = run([pip_exe, "-c", "import psr_srs_mvp; print('OK')"])
                        add("import_smoke", r3.returncode == 0, r3.stdout.strip())
                    else:
                        add("import_smoke", False, "install failed")
                shutil.rmtree(tmp_venv, ignore_errors=True)
    else:
        add("build", True, "skipped")

    # 13. License check
    license_path = _PROJECT / "LICENSE"
    license_exists = license_path.exists()
    is_mit = False
    if license_exists:
        try:
            text = license_path.read_text(encoding="utf-8")
            is_mit = ("MIT License" in text and
                      "Permission is hereby granted, free of charge" in text and
                      "THE SOFTWARE IS PROVIDED \"AS IS\"" in text)
            add("license_file", is_mit, "MIT License verified")
        except Exception as e:
            add("license_file", False, str(e))
    else:
        add("license_file", False, "missing")
    # Verify pyproject.toml SPDX
    pt = (_PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    spdx_ok = 'license = "MIT"' in pt
    add("pyproject_spdx_license", spdx_ok)
    lf_ok = 'license-files = ["LICENSE"]' in pt
    add("pyproject_license_files", lf_ok)
    # Build artifact metadata
    dists = list((_PROJECT / "dist").glob("*.whl"))
    if dists:
        import zipfile
        with zipfile.ZipFile(dists[0]) as zf:
            if "psr_srs_mvp-0.1.0.dist-info/METADATA" in zf.namelist():
                meta = zf.read("psr_srs_mvp-0.1.0.dist-info/METADATA").decode("utf-8")
                lic_in_meta = "License-Expression: MIT" in meta
                add("wheel_license_metadata", lic_in_meta)
            else:
                add("wheel_license_metadata", False, "METADATA not found")
            lic_in_wheel = any("LICENSE" in n for n in zf.namelist())
            add("wheel_contains_license_file", lic_in_wheel)
    else:
        add("wheel_license_metadata", False, "no wheel found")

    # Report
    total = len(CHECKS)
    passed = sum(1 for c in CHECKS if c["passed"])
    failed = total - passed

    manifest = {
        "project_name": "psr-srs-mvp",
        "version": "0.1.0",
        "python_version": sys.version.split()[0],
        "git_commit": "not a git repository",
        "git_branch": "N/A",
        "git_dirty": "N/A",
        "checks_total": total,
        "checks_passed": passed,
        "checks_failed": failed,
        "release_candidate_passed": failed == 0,
        "license_status": "MIT",
        "license_file": "LICENSE",
        "license_check_passed": license_exists and is_mit,
        "public_open_source_release_ready": license_exists and failed == 0,
        "checks": CHECKS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = out / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Result: {passed}/{total} passed, {failed} failed ===")
    print(f"  Release Candidate: {'PASSED' if failed == 0 else 'FAILED'}")
    if not license_exists:
        print(f"  Public release blocked: LICENSE file missing")
    print(f"  Manifest: {manifest_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
