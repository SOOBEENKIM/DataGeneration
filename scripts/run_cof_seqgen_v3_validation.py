"""CLI for the separately authorized CoF-SeqGen v3 validation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.cof_seqgen_v3_execution_runner import (
    build_v3_execution_plan,
    dry_run_v3_execution,
    run_authorized_v3_candidate,
    v3_execution_plan_report,
)


DEFAULT_CONFIG = Path(
    "configs/benchmark_v2/cof_seqgen_v3_validation_runner.yaml"
)
_CANDIDATES = (
    "cof_v3_c01_direct_joint",
    "cof_v3_c02_factorized_joint",
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoF-SeqGen v3 validation candidate runner",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("dry-run")
    execute = subparsers.add_parser("execute")
    execute.add_argument("--candidate", choices=_CANDIDATES, required=True)
    execute.add_argument("--device", required=True)
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--ownership-id", required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    plan = build_v3_execution_plan(args.config)
    if args.mode == "plan":
        result = v3_execution_plan_report(plan)
    elif args.mode == "dry-run":
        result = dry_run_v3_execution(plan)
    else:
        result = run_authorized_v3_candidate(
            plan=plan,
            candidate_id=args.candidate,
            device=args.device,
            authorization_path=args.authorization,
            ownership_id=args.ownership_id,
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
