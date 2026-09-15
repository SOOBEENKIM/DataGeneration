from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.evaluation_only_runner_v2_8 import (
    MODEL_IDS,
    EvaluationOnlyContractError,
    build_evaluation_execution_plan,
    dry_run_evaluation,
    evaluation_plan_report,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, dry-run, or separately authorized execution of the "
            "v2.8 validation-only candidates. Plan/dry-run never query "
            "GPU/CUDA or call restore/sample/evaluation."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/benchmark_v2/evaluation_only_v2_8.yaml"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "execute"),
        required=True,
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--model", choices=MODEL_IDS)
    parser.add_argument("--device")
    parsed = parser.parse_args(argv)
    if parsed.mode == "execute":
        if (
            parsed.authorization is None
            or parsed.model is None
            or parsed.device is None
            or not parsed.device.startswith("cuda:")
        ):
            parser.error(
                "execute requires --authorization, --model, and "
                "--device cuda:<logical-id>"
            )
    elif any(
        value is not None
        for value in (parsed.authorization, parsed.model, parsed.device)
    ):
        parser.error(
            "authorization, model, and device are forbidden in plan/dry-run"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (root / args.config).resolve()
    )
    plan = build_evaluation_execution_plan(config_path)
    if plan.repository_root != root:
        raise EvaluationOnlyContractError(
            "repository root differs from runner config root"
        )
    if args.mode == "plan":
        report = evaluation_plan_report(plan)
    elif args.mode == "dry-run":
        report = dry_run_evaluation(plan)
    else:
        from experiments.evaluation_only_runner_v2_8 import (
            execute_model_worker,
        )

        authorization = (
            args.authorization.resolve()
            if args.authorization.is_absolute()
            else (root / args.authorization).resolve()
        )
        report = execute_model_worker(
            plan=plan,
            authorization_path=authorization,
            model_id=args.model,
            device=args.device,
        )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except EvaluationOnlyContractError as error:
        raise SystemExit(
            f"v2.8 evaluation-only runner rejected: {error}"
        )
