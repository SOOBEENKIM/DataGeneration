from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.single_factor_aggregate_v2_6 import (
    SingleFactorAggregateContractError,
    execute_single_factor_validation_aggregate,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read and aggregate the nine immutable v2.6 single-factor "
            "validation candidates. This command never trains or samples "
            "a model and rejects test/fresh-test execution."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/benchmark_v2/"
            "selection_v2_6_single_factor_amendment.yaml"
        ),
    )
    parser.add_argument(
        "--base-selection-config",
        type=Path,
        default=Path("configs/benchmark_v2/selection_v2_6.yaml"),
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/benchmark_v2_6/selection_single_factor/"
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
    args = parse_args(argv)
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return (
            path.resolve()
            if path.is_absolute()
            else (root / path).resolve()
        )

    try:
        result = execute_single_factor_validation_aggregate(
            repository_root=root,
            config_path=resolve(args.config),
            base_selection_config_path=resolve(
                args.base_selection_config
            ),
            authorization_path=resolve(args.authorization),
            output_root=resolve(args.output_root),
            source_commit=args.source_commit,
            mode=args.mode,
        )
    except SingleFactorAggregateContractError as error:
        raise SystemExit(
            f"single-factor aggregate rejected: {error}"
        ) from error
    if args.mode == "plan":
        print(
            "single-factor aggregate plan PASS; workers=3; "
            "candidates=9; artifacts_written=false"
        )
    elif args.mode == "dry-run":
        print(
            "single-factor aggregate dry-run PASS; "
            "primary_c2_selection_ready="
            f"{str(result['report']['primary_c2_selection_ready']).lower()}; "
            "artifacts_written=false"
        )
    else:
        print(
            "single-factor aggregate COMPLETE; "
            "primary_c2_selection_ready="
            f"{str(result['report']['primary_c2_selection_ready']).lower()}; "
            "fresh-test/full execution remain unauthorized"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
