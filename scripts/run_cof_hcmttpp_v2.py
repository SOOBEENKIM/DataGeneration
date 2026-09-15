"""Plan, dry-run, or execute one dataset-scoped CoF-HCMTTPP-v2 H1 cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.cof_hcmttpp_v2_execution_runner import (
    H1RunnerContractError,
    build_execution_plan,
    dry_run_execution,
    execution_plan_report,
    execute_authorized,
)


def _default_execution_dependencies():
    from generators.cof_hcmttpp_v2_execution_backend import (
        build_default_execution_dependencies,
    )

    return build_default_execution_dependencies()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_v2/cof_hcmttpp_v2_execution_runner.yaml"),
    )
    parser.add_argument("--mode", choices=("plan", "dry-run", "execute"), required=True)
    parser.add_argument("--dataset", choices=("amlsim", "sparkov"), required=True)
    parser.add_argument("--candidate", choices=("H1",), required=True)
    parser.add_argument("--authorization", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repository_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    plan = build_execution_plan(
        config, dataset=args.dataset, candidate_id=args.candidate
    )
    if args.mode == "execute":
        if args.authorization is None:
            raise H1RunnerContractError("execute requires authorization")
        report = execute_authorized(
            plan=plan,
            authorization_path=args.authorization,
            dependencies=_default_execution_dependencies,
        )
    else:
        if args.authorization is not None:
            raise H1RunnerContractError("plan/dry-run cannot accept authorization")
        report = (
            execution_plan_report(plan)
            if args.mode == "plan"
            else dry_run_execution(plan)
        )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except H1RunnerContractError as error:
        raise SystemExit(f"cof_hcmttpp_v2 runner rejected: {error}")
