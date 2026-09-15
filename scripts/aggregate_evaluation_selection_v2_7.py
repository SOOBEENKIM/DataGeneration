from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the nine immutable v2.7 validation-only "
            "candidate artifacts. This command never queries a GPU, "
            "trains, samples, or accesses a test split."
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
            "configs/benchmark_v2/evaluation_only_v2_7.yaml"
        ),
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/benchmark_v2_7/candidate_selection/"
            "aggregate_attempt_001"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "execute"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from experiments.evaluation_aggregate_v2_7 import (
        EvaluationAggregateContractError,
        execute_v27_validation_aggregate,
    )

    args = parse_args(argv)
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return (
            path.resolve()
            if path.is_absolute()
            else (root / path).resolve()
        )

    try:
        result = execute_v27_validation_aggregate(
            repository_root=root,
            config_path=resolve(args.config),
            authorization_path=resolve(args.authorization),
            output_root=resolve(args.output_root),
            source_commit=args.source_commit,
            mode=args.mode,
        )
    except EvaluationAggregateContractError as error:
        raise SystemExit(
            f"v2.7 aggregate-only runner rejected: {error}"
        ) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
