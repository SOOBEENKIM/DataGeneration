"""Source-only plan/dry-run CLI for ``cof_ccmtpp_v1``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eval.cof_ccmtpp_v1_contract import CCMTPPContractError
from experiments.cof_ccmtpp_v1_preparation import (
    build_ccmtpp_plan,
    ccmtpp_plan_report,
    dry_run_ccmtpp,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only cof_ccmtpp_v1 plan/dry-run. Execute, authorization, "
            "device, GPU, CUDA, fit, sample, evaluation, selection, and test "
            "modes are intentionally unavailable."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml"),
    )
    parser.add_argument("--mode", choices=("plan", "dry-run"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.repository_root.resolve()
    config = args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    plan = build_ccmtpp_plan(config)
    report = ccmtpp_plan_report(plan) if args.mode == "plan" else dry_run_ccmtpp(plan)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except CCMTPPContractError as error:
        raise SystemExit(f"cof_ccmtpp_v1 source preparation rejected: {error}")
