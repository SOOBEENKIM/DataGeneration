from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


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
            "artifacts/benchmark_v2_3/auroc_calibration/"
            "calibration_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root", default="artifacts/benchmark_v2_3/gates"
    )
    args = parser.parse_args()

    prior_path = Path(args.v2_2_report)
    fanout_path = Path(args.fanout_manifest)
    channel_path = Path(args.channel_manifest)
    auroc_path = Path(args.auroc_manifest)
    prior = load(prior_path)
    fanout = load(fanout_path)
    channel = load(channel_path)
    auroc = load(auroc_path)

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
        "calibrated_single_row_auroc": auroc["status"],
    }
    all_statuses = [*retained.values(), *revised.values()]
    report = {
        "schema_version": "benchmark_gate_v2.3-candidate",
        "overall_status": (
            "PASS" if all(status == "PASS" for status in all_statuses)
            else "FAIL"
        ),
        "learned_smoke_authorized": all(
            status == "PASS" for status in all_statuses
        ),
        "retained_dgp_and_endpoint": True,
        "learned_results_used_for_rule_selection": False,
        "retained_checks": retained,
        "revised_checks": revised,
        "selected_auroc_method": auroc.get("selected_method"),
        "evidence": {
            "v2_2_cpu_gate": str(prior_path),
            "fanout_semantic_audit": str(fanout_path),
            "channel_control_audit": str(channel_path),
            "auroc_calibration": str(auroc_path),
        },
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    lines = [
        "# Benchmark v2.3 candidate CPU gate",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
        "No learned result was used. The v2.2 DGP parameters and continuous "
        "primary endpoint are unchanged.",
        "",
        "## Retained checks",
        "",
        *[f"- {name}: {status}" for name, status in retained.items()],
        "",
        "## Revised checks",
        "",
        *[f"- {name}: {status}" for name, status in revised.items()],
        "",
        "Selected AUROC method: `null`",
        f"Learned smoke authorized: `{report['learned_smoke_authorized']}`",
        "",
    ]
    (output / "gate_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
