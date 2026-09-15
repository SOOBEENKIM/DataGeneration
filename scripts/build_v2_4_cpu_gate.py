from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_report(
    *,
    prior: dict,
    fanout: dict,
    channel: dict,
    auroc: dict,
    verification: dict,
    evidence_paths: dict[str, str],
) -> dict:
    retained = {
        "continuous_association_primary": prior[
            "gate_b_continuous_iid_and_block_ordering"
        ],
        "iid_block1_c1_equivalence_and_block_ladder": prior[
            "gate_b_continuous_iid_and_block_ordering"
        ],
        "oracle_c0_equivalence": prior["gate_c_parametric_oracle"],
        "receiver_signed_frequency": prior[
            "gate_a_signed_receiver_frequency"
        ],
        "position_and_segment_stationarity": prior[
            "gate_a_positionwise_and_segments"
        ],
        "invalid_support_routing": prior[
            "gate_e_bad_generator_invalid_score_routing"
        ],
        "reference_and_oracle_bin_validity": prior[
            "gate_e_reference_and_oracle_bin_validity"
        ],
    }
    revised = {
        "fanout_secondary_joint_positive_control": fanout["status"],
        "true_channel_only_negative_controls": channel["status"],
        "calibrated_global_max_single_row_auroc": auroc["status"],
        "full_pytest": verification["pytest_status"],
        "compileall": verification["compileall_status"],
    }
    all_statuses = [*retained.values(), *revised.values()]
    authorized = all(status == "PASS" for status in all_statuses)
    return {
        "schema_version": "benchmark_gate_v2.4",
        "overall_status": "PASS" if authorized else "FAIL",
        "learned_smoke_authorized": authorized,
        "full_experiment_authorized": False,
        "retained_dgp_and_endpoint": True,
        "learned_results_used_for_rule_selection": False,
        "retained_checks": retained,
        "revised_checks": revised,
        "auroc_gate": {
            "statistic": "max_over_4_cells_times_2_classifiers",
            "threshold": auroc["clean"]["threshold"],
            "calibration_order_statistic_rank": auroc["clean"][
                "calibration_order_statistic_rank"
            ],
            "calibration_seeds": auroc["clean"]["calibration_seeds"],
            "validation_seeds": auroc["clean"]["validation_seeds"],
            "validation_false_failures": auroc["clean"]["false_failures"],
            "validation_false_fail_exact_upper": auroc["clean"][
                "false_fail_ci_high"
            ],
            "minimum_power_exact_lower": min(
                row["detection_power_ci_low"] for row in auroc["power"]
            ),
            "power_cells": len(auroc["power"]),
        },
        "evidence": evidence_paths,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Benchmark v2.4 CPU gate",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
        "No learned result was used. DGP parameters and the continuous primary "
        "endpoint are unchanged.",
        "",
        "## Retained checks",
        "",
        *[
            f"- {name}: {status}"
            for name, status in report["retained_checks"].items()
        ],
        "",
        "## Revised checks",
        "",
        *[
            f"- {name}: {status}"
            for name, status in report["revised_checks"].items()
        ],
        "",
        "## AUROC calibration",
        "",
        f"- threshold: `{report['auroc_gate']['threshold']}`",
        "- calibration rank: "
        f"`{report['auroc_gate']['calibration_order_statistic_rank']}/"
        f"{report['auroc_gate']['calibration_seeds']}`",
        "- validation false-fail exact upper: "
        f"`{report['auroc_gate']['validation_false_fail_exact_upper']}`",
        "- minimum leakage-power exact lower: "
        f"`{report['auroc_gate']['minimum_power_exact_lower']}`",
        "",
        f"Learned smoke authorized: `{report['learned_smoke_authorized']}`",
        "FULL_EXPERIMENT authorized: `false`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-2-report",
        default="artifacts/benchmark_v2/v2_2_gate/gate_report.json",
    )
    parser.add_argument(
        "--fanout-manifest",
        default=(
            "artifacts/benchmark_v2_3/fanout_semantic_audit/"
            "fanout_semantic_manifest.json"
        ),
    )
    parser.add_argument(
        "--channel-manifest",
        default=(
            "artifacts/benchmark_v2_3/channel_controls/"
            "channel_control_manifest.json"
        ),
    )
    parser.add_argument(
        "--auroc-manifest",
        default=(
            "artifacts/benchmark_v2_4/auroc_calibration/"
            "calibration_manifest.json"
        ),
    )
    parser.add_argument(
        "--verification-manifest",
        default="artifacts/benchmark_v2_4/cpu_verification.json",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/benchmark_v2_4/gates",
    )
    args = parser.parse_args()
    paths = {
        "v2_2_cpu_gate": args.v2_2_report,
        "fanout_semantic_audit": args.fanout_manifest,
        "channel_control_audit": args.channel_manifest,
        "auroc_calibration": args.auroc_manifest,
        "repository_verification": args.verification_manifest,
    }
    report = build_report(
        prior=load(Path(args.v2_2_report)),
        fanout=load(Path(args.fanout_manifest)),
        channel=load(Path(args.channel_manifest)),
        auroc=load(Path(args.auroc_manifest)),
        verification=load(Path(args.verification_manifest)),
        evidence_paths=paths,
    )
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (output / "gate_report.md").write_text(render_markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
