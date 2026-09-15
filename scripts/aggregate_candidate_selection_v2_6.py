from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.candidate_runner_v2_6 import (
    execute_authorized_validation_aggregate,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate authorized v2.6 validation-only candidates after "
            "the required model workers have immutable COMPLETE terminals. "
            "This command never trains a model and rejects all test paths."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_v2/selection_v2_6.yaml"),
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path(
            "configs/benchmark_v2/development_data_v2_6.yaml"
        ),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/benchmark_v2_6/selection/candidates"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/benchmark_v2_6/selection/"
            "aggregate_attempt_001"
        ),
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "execute"),
        default="execute",
        help=(
            "plan validates frozen scope without candidate evaluation; "
            "dry-run evaluates validation candidates without writes; "
            "execute creates the append-only selection bundle"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return (
            path.resolve()
            if path.is_absolute()
            else (root / path).resolve()
        )

    result = execute_authorized_validation_aggregate(
        repository_root=root,
        config_path=resolve(args.config),
        development_manifest_path=resolve(args.development_manifest),
        candidate_root=resolve(args.candidate_root),
        output_root=resolve(args.output_root),
        authorization_path=resolve(args.authorization),
        source_commit=args.source_commit,
        mode=args.mode,
    )
    if args.mode == "plan":
        print(
            "selection plan PASS; workers="
            f"{result['readiness']['worker_count']}; no artifacts written"
        )
    elif args.mode == "dry-run":
        print(
            "selection dry-run PASS; primary_c2_selection_ready="
            f"{result['report']['primary_c2_selection_ready']}; no "
            "artifacts written"
        )
    else:
        print(
            "aggregate complete; primary_c2_selection_ready="
            f"{result['report']['primary_c2_selection_ready']}; fresh test "
            "and five-seed/full execution remain unauthorized"
        )


if __name__ == "__main__":
    main()
