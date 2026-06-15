#!/usr/bin/env python3
"""Dual-run reproducibility check — same seed, separate temp dirs, SHA-256 comparison."""

from __future__ import annotations

import hashlib, json, shutil, sys, tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = _PROJECT / "configs" / "sample.json"
    tmp_base = Path(tempfile.mkdtemp(dir=_PROJECT / "outputs"))
    dir_a = tmp_base / "run_a"
    dir_b = tmp_base / "run_b"

    # Run generation twice
    for label, out_dir in [("first_run", dir_a), ("second_run", dir_b)]:
        import subprocess
        result = subprocess.run(
            [str(_PROJECT / ".venv" / "Scripts" / "python.exe"),
             str(_PROJECT / "scripts" / "generate_data.py"),
             "--config", str(config), "--output", str(out_dir), "--force"],
            capture_output=True, text=True,
            env={**__import__("os").environ,
                 "TMP": "D:/project/.tmp", "TEMP": "D:/project/.tmp",
                 "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTHONPYCACHEPREFIX": "D:/project/.cache/pycache-mvp"},
        )
        if result.returncode != 0:
            print(f"[FAIL] {label} failed: {result.stderr[:500]}")
            shutil.rmtree(tmp_base, ignore_errors=True)
            sys.exit(1)
        print(f"[OK] {label}: {out_dir}")

    # Compare
    files = ["items.csv", "users.csv", "queries.csv", "events.csv", "qrels.csv"]
    report = {"seed": 20260614, "config": str(config),
              "first_run": str(dir_a), "second_run": str(dir_b),
              "files": {}, "overall_passed": True}

    for fname in files:
        pa, pb = dir_a / fname, dir_b / fname
        ha, hb = sha256(pa), sha256(pb)
        matched = ha == hb
        ra, rb = len(pa.read_text(encoding="utf-8").splitlines()), len(pb.read_text(encoding="utf-8").splitlines())
        report["files"][fname] = {
            "sha256_run_a": ha, "sha256_run_b": hb,
            "matched": matched, "lines_a": ra, "lines_b": rb,
        }
        if not matched:
            report["overall_passed"] = False
        print(f"  {fname}: {'MATCH' if matched else 'DIFFER'}  sha256={ha[:16]}...  lines={ra}")

    out_path = _PROJECT / "outputs" / "data_generation" / "reproducibility_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Overall: {'PASSED' if report['overall_passed'] else 'FAILED'}")
    print(f"  Report: {out_path}")

    # Cleanup
    shutil.rmtree(tmp_base, ignore_errors=True)

    sys.exit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
