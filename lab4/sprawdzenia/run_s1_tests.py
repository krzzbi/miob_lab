"""S1: Uruchomienie 20 testow unittest z lab3/project (z PYTHONDONTWRITEBYTECODE=1 i -B).

Nie modyfikuje zadnych plikow w lab3/. Wynik zapisuje w lab4/wyniki/s1_tests.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT_DIR = REPO_ROOT / "lab3" / "project"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s1_tests.json"


def parse_unittest_output(stderr_text: str) -> Dict[str, object]:
    """Parsuje wyjscie `unittest -v` (ktore unittest wypisuje na stderr)."""
    tests: List[Dict[str, str]] = []
    current_test: str | None = None
    for line in stderr_text.splitlines():
        m_same_line = re.match(r"^(test_\S+\s+\([^)]+\))\s+\.\.\.\s+(ok|FAIL|ERROR|skipped.*)$", line)
        if m_same_line:
            tests.append({"name": m_same_line.group(1), "status": m_same_line.group(2)})
            current_test = None
            continue
        m_start = re.match(r"^(test_\S+\s+\([^)]+\))\s*$", line)
        if m_start:
            current_test = m_start.group(1)
            continue
        if current_test is not None:
            m_end = re.search(r"\.\.\.\s+(ok|FAIL|ERROR|skipped.*)$", line)
            if m_end:
                tests.append({"name": current_test, "status": m_end.group(1)})
                current_test = None

    m_ran = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+([0-9.]+)s", stderr_text)
    ran_count = int(m_ran.group(1)) if m_ran else len(tests)
    unittest_reported_s = float(m_ran.group(2)) if m_ran else None

    n_ok = sum(1 for t in tests if t["status"] == "ok")
    n_fail = sum(1 for t in tests if t["status"] == "FAIL")
    n_error = sum(1 for t in tests if t["status"] == "ERROR")
    n_skipped = sum(1 for t in tests if t["status"].startswith("skipped"))

    return {
        "ran_count": ran_count,
        "unittest_reported_seconds": unittest_reported_s,
        "ok": n_ok,
        "fail": n_fail,
        "error": n_error,
        "skipped": n_skipped,
        "tests": tests,
    }


def main() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    cmd = [
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-t",
        ".",
        "-v",
    ]

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(LAB3_PROJECT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_s = time.monotonic() - t0

    parsed = parse_unittest_output(proc.stderr)
    passed = (
        proc.returncode == 0
        and parsed["ran_count"] == 20
        and parsed["ok"] == 20
        and parsed["fail"] == 0
        and parsed["error"] == 0
        and parsed["skipped"] == 0
    )

    payload: Dict[str, object] = {
        "check": "S1",
        "description": "Uruchomienie 20 testow unittest z lab3/project z PYTHONDONTWRITEBYTECODE=1 i -B",
        "expected": {
            "returncode": 0,
            "ran_count": 20,
            "ok": 20,
            "fail": 0,
            "error": 0,
            "skipped": 0,
        },
        "command": cmd,
        "cwd": "lab3/project",
        "python_version": sys.version,
        "returncode": proc.returncode,
        "wall_seconds": round(elapsed_s, 4),
        "parsed": parsed,
        "passed": passed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S1 zakonczone: passed={passed} | ran={parsed['ran_count']} | "
        f"OK={parsed['ok']} FAIL={parsed['fail']} ERROR={parsed['error']} SKIP={parsed['skipped']} | "
        f"czas={elapsed_s:.3f}s (unittest: {parsed['unittest_reported_seconds']}s) | "
        f"returncode={proc.returncode}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
