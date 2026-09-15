from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from eval.validation_selection_v2_6 import MODEL_IDS
from experiments.candidate_runner_v2_6 import (
    execute_authorized_candidate_continuation,
    execute_authorized_candidate_plan,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded, append-only v2.6 train-only candidate "
            "trajectories. A matching explicit authorization manifest is "
            "mandatory; this command never reads validation arrays or test "
            "data."
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
        default=Path(
            "artifacts/benchmark_v2_6/selection/candidates"
        ),
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--continuation",
        action="store_true",
        help=(
            "Run the authorized append-only CTGAN/TVAE attempt_002 "
            "continuation. Native c00/c01 checkpoints are evaluation-only; "
            "only c02/c03 trajectories train."
        ),
    )
    parser.add_argument(
        "--model",
        choices=MODEL_IDS,
        required=True,
        help=(
            "Run exactly one preregistered model worker. Omission and "
            "unknown model IDs fail closed."
        ),
    )
    parsed = parser.parse_args(argv)
    if parsed.continuation and parsed.model not in {
        "ctgan_separate_class",
        "tvae_separate_class",
    }:
        parser.error(
            "--continuation is limited to ctgan_separate_class and "
            "tvae_separate_class"
        )
    return parsed


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    execute = (
        execute_authorized_candidate_continuation
        if args.continuation
        else execute_authorized_candidate_plan
    )
    results = execute(
        repository_root=root,
        config_path=resolve(args.config),
        development_manifest_path=resolve(args.development_manifest),
        candidate_root=resolve(args.candidate_root),
        authorization_path=resolve(args.authorization),
        source_commit=args.source_commit,
        device=args.device,
        model_id=args.model,
    )
    if not results or any(result["status"] != "COMPLETE" for result in results):
        raise SystemExit(
            "candidate runner stopped fail-closed; inspect append-only marker"
        )
    print(
        "all preregistered candidate trajectories completed; validation "
        "selection and all test/full execution remain separately unauthorized"
    )


if __name__ == "__main__":
    main()
