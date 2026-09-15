from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eval.candidate_preparation_v2_8 import (
    V28PreparationContractError,
)
from experiments.candidate_preparation_runner_v2_8 import (
    build_v28_plan,
    dry_run_v28,
    v28_plan_report,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only v2.8 source-amendment plan/dry-run. "
            "There is no authorization, execute, device, GPU, CUDA, "
            "training, checkpoint update, sampling, evaluation, "
            "selection, or test mode."
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
            "selection_v2_8_source_amendment.yaml"
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
    plan = build_v28_plan(config_path)
    report = (
        v28_plan_report(plan)
        if args.mode == "plan"
        else dry_run_v28(plan)
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
    except V28PreparationContractError as error:
        raise SystemExit(
            f"v2.8 source amendment rejected: {error}"
        )
