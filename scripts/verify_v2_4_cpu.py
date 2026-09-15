from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run_logged(command: list[str], log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("x") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="artifacts/benchmark_v2_4",
    )
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "cpu_verification.json"
    if manifest_path.exists():
        raise FileExistsError(f"verification already exists: {manifest_path}")
    pytest_code, pytest_seconds = run_logged(
        [sys.executable, "-m", "pytest", "-q"],
        output / "full_pytest.log",
    )
    compile_code, compile_seconds = run_logged(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "benchmarks",
            "eval",
            "generators",
            "scripts",
            "tests",
        ],
        output / "compileall.log",
    )
    payload = {
        "schema_version": "benchmark-v2.4-cpu-verification",
        "source_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "python_executable": sys.executable,
        "pytest_command": f"{sys.executable} -m pytest -q",
        "pytest_exit_code": pytest_code,
        "pytest_elapsed_seconds": pytest_seconds,
        "pytest_status": "PASS" if pytest_code == 0 else "FAIL",
        "pytest_log": str(output / "full_pytest.log"),
        "compileall_command": (
            f"{sys.executable} -m compileall -q "
            "benchmarks eval generators scripts tests"
        ),
        "compileall_exit_code": compile_code,
        "compileall_elapsed_seconds": compile_seconds,
        "compileall_status": "PASS" if compile_code == 0 else "FAIL",
        "compileall_log": str(output / "compileall.log"),
    }
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if pytest_code or compile_code:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
