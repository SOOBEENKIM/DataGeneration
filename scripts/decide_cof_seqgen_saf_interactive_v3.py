#!/usr/bin/env python3
"""Apply the preregistered interactive-v3 dependency-routing decision rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(summary: Mapping[str, Any], required_positive_seeds: int) -> dict[str, Any]:
    mean_positive = float(summary["mean"]) > 0.0
    enough_positive_seeds = int(summary["positive_seed_count"]) >= required_positive_seeds
    return {
        "mean": float(summary["mean"]),
        "ci95_low": float(summary["ci95_low"]),
        "ci95_high": float(summary["ci95_high"]),
        "positive_seed_count": int(summary["positive_seed_count"]),
        "required_positive_seeds": required_positive_seeds,
        "mean_positive": mean_positive,
        "enough_positive_seeds": enough_positive_seeds,
        "pass": mean_positive and enough_positive_seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    analysis_path = Path(config["analysis_path"])
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    required = int(config["required_positive_seeds"])
    endpoints = tuple(config["primary_endpoints"])

    branch_results: dict[str, Any] = {}
    failures: list[str] = []
    for branch, contrast_name in config["route_contrasts"].items():
        interaction_name = config["kappa_interactions"][branch]
        kappa_1_checks = {
            endpoint: _check(
                analysis["paired_improvement_contrasts"]["kappa_1"][
                    contrast_name
                ][endpoint],
                required,
            )
            for endpoint in endpoints
        }
        interaction_checks = {
            endpoint: _check(
                analysis["kappa_interactions"][interaction_name][endpoint],
                required,
            )
            for endpoint in endpoints
        }
        branch_pass = all(
            result["pass"]
            for result in (*kappa_1_checks.values(), *interaction_checks.values())
        )
        branch_results[branch] = {
            "kappa_1_routing_gain": kappa_1_checks,
            "kappa_1_minus_kappa_0_interaction": interaction_checks,
            "pass": branch_pass,
        }
        for family, checks in (
            ("kappa_1_routing_gain", kappa_1_checks),
            ("kappa_1_minus_kappa_0_interaction", interaction_checks),
        ):
            failures.extend(
                f"{branch}:{family}:{endpoint}"
                for endpoint, result in checks.items()
                if not result["pass"]
            )

    overall_pass = all(result["pass"] for result in branch_results.values())
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": config["schema_version"],
        "model_implementation_version": analysis["model_implementation_version"],
        "decision": "PASS" if overall_pass else "FAIL",
        "decision_rule": config["decision_rule"],
        "branch_results": branch_results,
        "failed_checks": failures,
        "analysis_path": str(analysis_path),
        "analysis_sha256": _sha256(analysis_path),
        "decision_config_sha256": _sha256(args.config),
        "test_accessed": bool(analysis["test_accessed"]),
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": artifact["decision"],
                "failed_checks": failures,
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "test_accessed": artifact["test_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
