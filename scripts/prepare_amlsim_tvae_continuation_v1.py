"""Prepare one append-only AMLSim TVAE attempt_002 authorization.

Plan and dry-run are read-only. Create writes only the explicitly requested
authorization document; it never queries a GPU or runs a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import yaml

from scripts.run_external_validation_v1 import (
    ExternalValidationError,
    build_amlsim_tvae_continuation_authorization,
    build_amlsim_tvae_continuation_plan,
    validate_external_validation_authorization,
    write_external_validation_authorization,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run", "create"), required=True)
    parser.add_argument("--approval-text", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    if args.mode in {"plan", "dry-run"}:
        if args.output is not None:
            raise ExternalValidationError(
                "AMLSim TVAE plan/dry-run cannot write an authorization"
            )
        result = build_amlsim_tvae_continuation_plan(
            repo_root=repo_root,
            config_path=config_path,
            approval_text=args.approval_text,
            mode=args.mode,
        )
    else:
        if args.output is None:
            raise ExternalValidationError(
                "AMLSim TVAE authorization create requires --output"
            )
        authorization = build_amlsim_tvae_continuation_authorization(
            repo_root=repo_root,
            config_path=config_path,
            approval_text=args.approval_text,
        )
        validated = validate_external_validation_authorization(
            repo_root=repo_root,
            config_path=config_path,
            authorization=authorization,
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_root = (
            repo_root
            / config["corrective_continuation"]["authorization_root"]
        ).resolve()
        output = args.output.resolve()
        if output.parent != expected_root:
            raise ExternalValidationError(
                "AMLSim TVAE authorization output is outside append-only root"
            )
        write_external_validation_authorization(output, authorization)
        observed = _sha256(output)
        expected = validated["authorization_sha256"]
        # The validator hashes canonical compact JSON while the file uses the
        # append-only indented document representation. Report both explicitly.
        result = {
            "schema_version": "amlsim-tvae-memory-safe-authorization-created-v1",
            "status": "PASS",
            "path": str(output),
            "file_sha256": observed,
            "canonical_authorization_sha256": expected,
            "job_count": 1,
            "target": {
                "dataset": "amlsim",
                "model": "tvae_separate_class",
                "attempt": "attempt_002",
            },
            "execution_counts": {
                "gpu_inventory_queries": 0,
                "cuda_calls": 0,
                "model_import_calls": 0,
                "model_fit_calls": 0,
                "model_sample_calls": 0,
                "validation_metric_calls": 0,
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
