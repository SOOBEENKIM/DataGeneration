"""Source-only CoF-SeqGen v3 plan and dry-run CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eval.cof_seqgen_v3_contract import V3ContractError
from experiments.cof_seqgen_v3_preparation import (
    build_v3_plan,
    dry_run_v3,
    v3_plan_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only CoF-SeqGen v3 source plan/dry-run. "
            "There is no authorization, execute, device, GPU, CUDA, "
            "fit, checkpoint, sample, validation, selection, or test mode."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/benchmark_v2/"
            "cof_seqgen_v3_source_preparation.yaml"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.repository_root.resolve()
    config = (
        args.config.resolve()
        if args.config.is_absolute()
        else (root / args.config).resolve()
    )
    plan = build_v3_plan(config)
    report = (
        v3_plan_report(plan)
        if args.mode == "plan"
        else dry_run_v3(plan)
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except V3ContractError as error:
        raise SystemExit(f"CoF-SeqGen v3 source preparation rejected: {error}")
