from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eval.candidate_preparation_v2_7 import (
    V27PreparationContractError,
)
from experiments.candidate_preparation_runner_v2_7 import (
    build_preparation_execution_plan,
    dry_run_preparation,
    preparation_plan_report,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only plan/dry-run for the v2.7 source preparation. "
            "This command has no authorization, execute, device, GPU, "
            "training, sampling, selection, or test mode."
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
            "selection_v2_7_source_preparation.yaml"
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
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (root / args.config).resolve()
    )
    plan = build_preparation_execution_plan(config_path)
    report = (
        preparation_plan_report(plan)
        if args.mode == "plan"
        else dry_run_preparation(plan)
    )
    print(
        json.dumps(
            report,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except V27PreparationContractError as error:
        raise SystemExit(
            f"v2.7 source preparation rejected: {error}"
        )
