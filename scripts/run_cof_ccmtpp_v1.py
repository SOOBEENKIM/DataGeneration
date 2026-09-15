"""Plan, dry-run, or execute the authorization-bound CCMTPP runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eval.cof_ccmtpp_v1_contract import CCMTPPContractError
from experiments.cof_ccmtpp_v1_execution_runner import (
    build_execution_plan,
    dry_run_execution,
    execute_authorized,
    execution_plan_report,
)


def _default_execution_dependencies():
    # This import is deliberately downstream of authorization validation in
    # execute_authorized. Plan/dry-run and unauthorized execute never import
    # the execution backend or model code.
    from generators.cof_ccmtpp_v1_execution_backend import (
        build_default_execution_dependencies,
    )

    return build_default_execution_dependencies()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/benchmark_v2/cof_ccmtpp_v1_execution_runner.yaml"
        ),
    )
    parser.add_argument("--mode", choices=("plan", "dry-run", "execute"), required=True)
    parser.add_argument("--candidate", choices=("C1", "C2", "C3", "C4"), required=True)
    parser.add_argument("--dataset", choices=("amlsim", "sparkov"), required=True)
    parser.add_argument("--authorization", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repository_root.resolve()
    config = args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    plan = build_execution_plan(
        config, candidate_id=args.candidate, dataset=args.dataset
    )
    if args.mode == "execute":
        if args.authorization is None:
            raise CCMTPPContractError("execute requires authorization")
        result = execute_authorized(
            plan=plan,
            authorization_path=args.authorization,
            dependencies=_default_execution_dependencies,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if args.authorization is not None:
        raise CCMTPPContractError("plan/dry-run cannot accept authorization")
    report = execution_plan_report(plan) if args.mode == "plan" else dry_run_execution(plan)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CCMTPPContractError as error:
        raise SystemExit(f"cof_ccmtpp_v1 runner rejected: {error}")
