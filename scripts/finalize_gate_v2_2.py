from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csvs(paths: list[Path], name: str) -> list[dict]:
    rows = []
    for path in paths:
        with (path / name).open() as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def means(rows: list[dict], kappa: float, generator: str) -> float:
    selected = [
        float(row["association_recovery_error"])
        for row in rows
        if float(row["kappa"]) == kappa
        and row["generator"] == generator
        and row["channel"] == "joint_alignment"
    ]
    if not selected:
        raise ValueError(f"missing {kappa=} {generator=}")
    return float(np.mean(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/benchmark_v2/v2_2_gate")
    args = parser.parse_args()
    root = Path(args.root)
    cells = sorted(path for path in root.glob("gate_b_k*") if path.is_dir())
    if len(cells) != 5:
        raise RuntimeError(f"expected five Gate B cell directories, got {cells}")
    association = read_csvs(cells, "continuous_association.csv")
    scores = read_csvs(cells, "endpoint_comparison.csv")
    audit = json.loads((root / "full_audit_manifest.json").read_text())
    receiver = json.loads(
        Path(
            "artifacts/benchmark_v2/receiver_diagnosis/v2_2_candidate/"
            "calibration_manifest.json"
        ).read_text()
    )
    position = json.loads(
        Path(
            "artifacts/benchmark_v2/positionwise_v2_2/"
            "positionwise_manifest.json"
        ).read_text()
    )
    tolerance = 0.005
    gate_b_checks = []
    gate_c_checks = []
    for kappa in (0.0, 0.3, 0.5, 0.7, 1.0):
        c0 = means(association, kappa, "c0_sampling_floor")
        c1 = means(association, kappa, "c1")
        iid = means(association, kappa, "iid")
        oracle = means(association, kappa, "oracle")
        block = [means(association, kappa, f"block_{value}") for value in (1, 2, 4, 8, "full")]
        gate_c_checks.append(oracle <= c0 + tolerance)
        gate_b_checks.extend(
            [
                abs(iid - c1) <= tolerance,
                abs(block[0] - c1) <= tolerance,
            ]
        )
        if kappa > 0:
            gate_b_checks.append(block[1] < block[0])
            gate_b_checks.extend(
                right <= left + tolerance
                for left, right in zip(block[1:], block[2:])
            )
    # Reference/oracle validity is a benchmark property. Bad-generator
    # invalidity is retained as an invalid model score, not a benchmark fail.
    gate_e_reference_checks = []
    bad_generator_invalid_cells = 0
    for row in scores:
        bins = int(row["n_bins"])
        if bins not in (4, 8):
            continue
        generator = row["generator"]
        invalid = int(row["invalid_bin_count"])
        if generator in {"c1", "oracle", "c0"}:
            gate_e_reference_checks.append(invalid == 0)
        elif generator in {"iid", "block_1"} and invalid:
            bad_generator_invalid_cells += 1
    full_association = list(
        csv.DictReader((root / "real_continuous_association.csv").open())
    )
    negative = [
        abs(float(row["standardized_delta"]))
        for row in full_association
        if row["scenario"] == "joint_semimarkov_v2b"
        and row["channel"] in {"velocity", "gap", "fanout", "amount"}
    ]
    # Fail closed: fanout reaches a medium standardized effect (about 0.34),
    # which cannot be described as approximately zero. No post-hoc threshold
    # is introduced to turn this into a pass.
    negative_control = max(negative) < 0.10
    status = {
        "gate_a_single_row_auroc": audit["single_row_auroc_equivalence"],
        "gate_a_signed_receiver_frequency": audit["signed_receiver_frequency"],
        "gate_a_receiver_100_seed_calibration": receiver["status"],
        "gate_a_positionwise_and_segments": position["status"],
        "gate_b_continuous_iid_and_block_ordering": (
            "PASS" if all(gate_b_checks) else "FAIL"
        ),
        "gate_c_parametric_oracle": (
            "PASS" if all(gate_c_checks) else "FAIL"
        ),
        "gate_d_real_joint_association": audit["real_joint_association"],
        "gate_d_channel_negative_controls": (
            "PASS" if negative_control else "FAIL"
        ),
        "gate_e_reference_and_oracle_bin_validity": (
            "PASS" if all(gate_e_reference_checks) else "FAIL"
        ),
        "gate_e_bad_generator_invalid_score_routing": (
            "PASS" if bad_generator_invalid_cells > 0 else "FAIL"
        ),
    }
    overall = "PASS" if all(value == "PASS" for value in status.values()) else "FAIL"
    report = {
        "schema_version": "benchmark_gate_v2.2-candidate",
        "overall_status": overall,
        **status,
        "diagnostics": {
            "gate_b_checks": len(gate_b_checks),
            "gate_b_failed_checks": sum(not value for value in gate_b_checks),
            "gate_c_checks": len(gate_c_checks),
            "gate_c_failed_checks": sum(not value for value in gate_c_checks),
            "bad_generator_invalid_cells": bad_generator_invalid_cells,
            "maximum_abs_negative_control_standardized_delta": max(negative),
            "learned_smoke_authorized": overall == "PASS",
        },
    }
    (root / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Benchmark v2.2 candidate gate report",
        "",
        f"Overall: **{overall}**",
        "",
        *(f"- {key}: {value}" for key, value in status.items()),
        "",
        "Diagnostics:",
        "",
        f"- Gate B failed checks: {report['diagnostics']['gate_b_failed_checks']} / {report['diagnostics']['gate_b_checks']}",
        f"- Gate C failed checks: {report['diagnostics']['gate_c_failed_checks']} / {report['diagnostics']['gate_c_checks']}",
        f"- bad-generator invalid support cells routed to invalid model score: {bad_generator_invalid_cells}",
        f"- maximum absolute v2b negative-control standardized delta: {max(negative):.6f}",
        f"- learned smoke authorized: {overall == 'PASS'}",
    ]
    (root / "gate_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
