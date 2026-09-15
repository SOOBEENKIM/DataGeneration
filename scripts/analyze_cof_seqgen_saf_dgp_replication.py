#!/usr/bin/env python3
"""Aggregate paired five-seed κ=0/κ=1 SAF mechanism contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t
import yaml


DEFAULT_PRIMARY = (
    "gap_conditioned_mark_tv",
    "short_gap_repeat_curve_l1",
    "gap_repeat_mi_error",
)
GAP = ("gap_ks", "gap_scaled_w1", "gap_zero_rate_error")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Sequence[float]) -> Dict[str, float | int | bool]:
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if n > 1 else 0.0
    half_width = (
        float(t.ppf(0.975, n - 1) * standard_deviation / np.sqrt(n))
        if n > 1
        else float("nan")
    )
    return {
        "n": n,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "positive_seed_count": int(np.sum(array > 0)),
        "all_seeds_positive": bool(np.all(array > 0)),
    }


def _load_reports(paths: Mapping[int, str]) -> tuple[Dict[int, Any], Dict[int, str]]:
    reports = {}
    hashes = {}
    for seed, raw_path in paths.items():
        path = Path(raw_path)
        reports[int(seed)] = json.loads(path.read_text(encoding="utf-8"))
        hashes[int(seed)] = _sha256(path)
    return reports, hashes


def _metric(report: Mapping[str, Any], candidate: str, endpoint: str) -> float:
    return float(report["results"][candidate]["validation_fidelity"][endpoint])


def _paired_improvement(
    reports: Mapping[int, Any],
    control: str,
    treatment: str,
    endpoints: Iterable[str],
) -> Dict[str, Any]:
    # Every fidelity endpoint is an error, so positive control-treatment means
    # the treatment lowered error.
    return {
        endpoint: {
            **_summary([
                _metric(report, control, endpoint)
                - _metric(report, treatment, endpoint)
                for _, report in sorted(reports.items())
            ]),
            "direction": "positive_means_treatment_lower_error",
        }
        for endpoint in endpoints
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_implementation = config.get("model_implementation_version")
    primary = tuple(config.get("primary_endpoints", DEFAULT_PRIMARY))
    cells = {}
    report_hashes = {}
    for cell, paths in config["reports"].items():
        cells[cell], report_hashes[cell] = _load_reports(
            {int(seed): path for seed, path in paths.items()}
        )
    expected_seeds = set(config["seeds"])
    if any(set(reports) != expected_seeds for reports in cells.values()):
        raise RuntimeError("every κ cell must contain the same preregistered seeds")
    if expected_implementation is not None and any(
        report.get("model_implementation_version") != expected_implementation
        for reports in cells.values()
        for report in reports.values()
    ):
        raise RuntimeError("validation report implementation version mismatch")

    candidates = tuple(config["candidates"])
    per_candidate = {
        cell: {
            candidate: {
                endpoint: _summary([
                    _metric(report, candidate, endpoint)
                    for _, report in sorted(reports.items())
                ])
                for endpoint in (*primary, *GAP)
            }
            for candidate in candidates
        }
        for cell, reports in cells.items()
    }
    contrasts = {
        cell: {
            "H1_support_alignment_C0_to_U0": _paired_improvement(
                reports, "SAF-C0", "SAF-U0", GAP
            ),
            "H2_order_without_route_U0_to_O0": _paired_improvement(
                reports, "SAF-U0", "SAF-O0", (*primary, *GAP)
            ),
            "H2_order_with_route_U1_to_O1": _paired_improvement(
                reports, "SAF-U1", "SAF-O1", (*primary, *GAP)
            ),
            "H3_route_unordered_U0_to_U1": _paired_improvement(
                reports, "SAF-U0", "SAF-U1", primary
            ),
            "H3_route_ordered_O0_to_O1": _paired_improvement(
                reports, "SAF-O0", "SAF-O1", primary
            ),
        }
        for cell, reports in cells.items()
    }
    interactions = {}
    for name, control, treatment in (
        ("H3_route_unordered_kappa_interaction", "SAF-U0", "SAF-U1"),
        ("H3_route_ordered_kappa_interaction", "SAF-O0", "SAF-O1"),
    ):
        interactions[name] = {}
        for endpoint in primary:
            values = []
            for seed in sorted(expected_seeds):
                k1 = _metric(cells["kappa_1"][seed], control, endpoint) - _metric(
                    cells["kappa_1"][seed], treatment, endpoint
                )
                k0 = _metric(cells["kappa_0"][seed], control, endpoint) - _metric(
                    cells["kappa_0"][seed], treatment, endpoint
                )
                values.append(k1 - k0)
            interactions[name][endpoint] = {
                **_summary(values),
                "direction": "positive_means_larger_routing_gain_at_kappa_1",
            }

    empirical_calibration_gate = {
        cell: all(
            report["results"]["empirical_sequence_sampler"][
                "all_calibrated_marginal_structural_endpoints_pass"
            ]
            for report in reports.values()
        )
        for cell, reports in cells.items()
    }
    artifact = {
        "schema_version": "cof-seqgen-saf-dgp-five-seed-mechanism-v1",
        "model_implementation_version": expected_implementation,
        "seeds": sorted(expected_seeds),
        "candidate_ids": candidates,
        "primary_endpoints": primary,
        "lower_is_better_for_all_aggregated_endpoints": True,
        "report_sha256": report_hashes,
        "per_candidate": per_candidate,
        "paired_improvement_contrasts": contrasts,
        "kappa_interactions": interactions,
        "empirical_noninferiority_calibration_gate": empirical_calibration_gate,
        "noninferiority_interpretation_allowed": bool(
            all(empirical_calibration_gate.values())
        ),
        "inference_status": (
            "PAIRED_FIVE_SEED_DEVELOPMENT_ESTIMATES; CI_IS_MODEL_SEED_VARIATION; "
            "NO_TEST_ACCESS; NO_MULTIPLICITY_CORRECTED_CONFIRMATORY_CLAIM"
        ),
        "test_accessed": False,
    }
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "noninferiority_interpretation_allowed": artifact[
                    "noninferiority_interpretation_allowed"
                ],
                "test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
