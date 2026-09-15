from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", default="artifacts/benchmark_v2/gates")
    args = parser.parse_args()
    root = Path(args.gate_root)
    report_path = root / "gate_report.json"
    report = json.loads(report_path.read_text())
    completion = json.loads((root / "cpu_gate_completion.json").read_text())
    simultaneous = json.loads(
        (root / "positionwise_stationarity_simultaneous.json").read_text()
    )
    keys = (
        "gate_a_positionwise_stationarity",
        "gate_b_iid_equivalence_to_c1",
        "gate_c_dgp_oracle_near_c0",
        "gate_e_bin_robustness",
    )
    for key in keys:
        report[key] = completion[key]
    # The simultaneous band is mandatory evidence for the positionwise result.
    if simultaneous["status"] != "PASS":
        report["gate_a_positionwise_stationarity"] = "FAIL"
    report["overall_status"] = (
        "PASS"
        if all(value == "PASS" for key, value in report.items() if key != "schema_version")
        else "FAIL"
    )
    report["completion_evidence"] = {
        "cpu_gate_completion": "cpu_gate_completion.json",
        "positionwise_simultaneous": "positionwise_stationarity_simultaneous.json",
        "gate_b": ["iid_bootstrap_4bin.csv", "iid_bootstrap_8bin.csv"],
        "gate_c": "dgp_oracle.csv",
        "gate_e": "bin_robustness.csv",
        "context_ladder": "context_ladder/context_length_curve.csv",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Benchmark v2 gate report",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
    ]
    lines.extend(
        f"- {key}: {value}"
        for key, value in report.items()
        if key not in {"schema_version", "overall_status", "completion_evidence"}
    )
    lines.extend(
        [
            "",
            "Completion evidence:",
            "",
            "- entity-cluster positionwise simultaneous band: "
            f"{simultaneous['status']}",
            f"- CPU Gate B/C/E completion: {completion['status']}",
            "- missing or non-computable evidence remains fail-closed.",
        ]
    )
    (root / "gate_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
