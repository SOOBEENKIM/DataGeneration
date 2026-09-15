from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eval.single_factor_amendment_v2_6 import sha256_file
from experiments.single_factor_runner_v2_6 import (
    SINGLE_FACTOR_MODEL_IDS,
    SingleFactorRunnerContractError,
    build_single_factor_execution_plan,
    dry_run_single_factor_worker,
    execute_single_factor_worker,
    plan_report,
    preserved_tree_record_sha256,
    read_authorization,
    single_factor_relevant_source_sha256,
    validate_single_factor_authorization,
    write_preflight_record,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, dry-run, or execute the append-only v2.6 "
            "validation-only single-factor candidates. Plan/dry-run never "
            "query a GPU or call CUDA, model fit, sample, DGP, selection, "
            "fresh-test, or test paths."
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
            "selection_v2_6_single_factor_amendment.yaml"
        ),
    )
    parser.add_argument(
        "--base-selection-config",
        type=Path,
        default=Path(
            "configs/benchmark_v2/selection_v2_6.yaml"
        ),
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "execute"),
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=SINGLE_FACTOR_MODEL_IDS,
    )
    parser.add_argument("--device")
    parsed = parser.parse_args(argv)
    if parsed.mode == "execute":
        if parsed.model is None:
            parser.error("--model is required in execute mode")
        if (
            parsed.device is None
            or not parsed.device.startswith("cuda")
        ):
            parser.error(
                "--device cuda:<logical-id> is required in execute mode"
            )
    elif parsed.device is not None:
        parser.error(
            "--device is forbidden in plan/dry-run mode"
        )
    return parsed


def _resolve(root: Path, value: Path) -> Path:
    return (
        value.resolve()
        if value.is_absolute()
        else (root / value).resolve()
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.repository_root.resolve()
    config = _resolve(root, args.config)
    base_config = _resolve(root, args.base_selection_config)
    authorization_path = _resolve(root, args.authorization)
    plan = build_single_factor_execution_plan(
        config_path=config,
        base_selection_config_path=base_config,
    )
    authorization = read_authorization(authorization_path)
    tree_hash, _, _ = preserved_tree_record_sha256(
        root / "artifacts/benchmark_v2_6/selection"
    )
    source_hash = single_factor_relevant_source_sha256(root)
    validate_single_factor_authorization(
        authorization,
        plan=plan,
        source_commit=args.source_commit,
        relevant_source_sha256=source_hash,
        prior_candidate_tree_sha256=tree_hash,
    )
    if args.mode == "plan":
        report = {
            **dict(plan_report(plan)),
            "source_commit": args.source_commit,
            "relevant_source_sha256": source_hash,
            "authorization_path": str(
                authorization_path.relative_to(root)
            ),
            "authorization_sha256": sha256_file(
                authorization_path
            ),
            "prior_v2_6_candidate_tree_sha256": tree_hash,
        }
        output = write_preflight_record(
            repository_root=root,
            kind="plan",
            value=report,
        )
        print(
            json.dumps(
                {
                    **report,
                    "record_path": str(output.relative_to(root)),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.mode == "dry-run":
        model_ids = (
            (args.model,)
            if args.model is not None
            else SINGLE_FACTOR_MODEL_IDS
        )
        results = [
            dry_run_single_factor_worker(
                repository_root=root,
                plan=plan,
                authorization=authorization,
                source_commit=args.source_commit,
                prior_candidate_tree_sha256=tree_hash,
                model_id=model_id,
                verify_files=True,
            )
            for model_id in model_ids
        ]
        report = {
            "schema_version": (
                "benchmark-v2.6-single-factor-dry-run-suite-v1"
            ),
            "status": "DRY_RUN_PASS",
            "source_commit": args.source_commit,
            "relevant_source_sha256": source_hash,
            "authorization_path": str(
                authorization_path.relative_to(root)
            ),
            "authorization_sha256": sha256_file(
                authorization_path
            ),
            "models": list(model_ids),
            "results": results,
            "test_split_accesses": 0,
            "fresh_test_accesses": 0,
            "gpu_queries": 0,
            "cuda_calls": 0,
            "fit_calls": 0,
            "sample_calls": 0,
            "dgp_calls": 0,
            "selection_calls": 0,
        }
        output = write_preflight_record(
            repository_root=root,
            kind="dry_run",
            value=report,
        )
        print(
            json.dumps(
                {
                    **report,
                    "record_path": str(output.relative_to(root)),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    result = execute_single_factor_worker(
        repository_root=root,
        config_path=config,
        base_selection_config_path=base_config,
        authorization_path=authorization_path,
        source_commit=args.source_commit,
        model_id=args.model,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result.get("status") != "COMPLETE":
        raise SystemExit(
            "single-factor worker stopped fail-closed; inspect its "
            "append-only terminal artifact"
        )


if __name__ == "__main__":
    try:
        main()
    except SingleFactorRunnerContractError as error:
        raise SystemExit(f"single-factor runner rejected: {error}")
