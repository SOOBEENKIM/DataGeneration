from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Sequence


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, validate, or execute the v2.8 stored-evidence-only "
            "validation aggregate. This command never queries a GPU, "
            "trains, samples, recalculates guards, or accesses test data."
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
            "configs/benchmark_v2/evaluation_aggregate_v2_8.yaml"
        ),
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/benchmark_v2_8/candidate_selection/"
            "aggregate_attempt_001"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "execute"),
        required=True,
    )
    args = parser.parse_args(argv)
    if args.mode == "execute" and args.authorization is None:
        parser.error("--authorization is required for execute")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    from experiments.evaluation_aggregate_v2_8 import (
        EvaluationAggregateV28ContractError,
        execute_v28_validation_aggregate,
    )

    args = parse_args(argv)
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return (
            path.resolve()
            if path.is_absolute()
            else (root / path).resolve()
        )

    source_commit = args.source_commit
    if source_commit is None:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    try:
        result = execute_v28_validation_aggregate(
            repository_root=root,
            config_path=resolve(args.config),
            authorization_path=(
                resolve(args.authorization)
                if args.authorization is not None
                else None
            ),
            output_root=resolve(args.output_root),
            source_commit=source_commit,
            mode=args.mode,
        )
    except EvaluationAggregateV28ContractError as error:
        raise SystemExit(
            f"v2.8 aggregate-only runner rejected: {error}"
        ) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
