from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.types import SequenceBatch
from benchmarks.validation import context_auc_curve, gate_status, marginal_metrics, row_classifier_auc
from eval.behavior_summaries_v2 import compute_behavior_summaries_v2, fit_short_gap_threshold
from eval.sequence_metrics import class_contrast


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(**{key: values[key] for key in SequenceBatch.__dataclass_fields__})


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data/benchmark_v2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/gates")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    data_root, out = Path(args.data_root), Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    marginal_rows, classifier_rows, context_rows, sequential_rows = [], [], [], []
    statuses: dict[str, str] = {}
    for scenario in raw["scenarios"]:
        scenario_ranges = []
        for kappa in raw["coupling"]["kappas"]:
            folder = data_root / scenario / f"kappa_{float(kappa):.2f}"
            train, test = load_batch(folder / "train.npz"), load_batch(folder / "test.npz")
            with np.load(folder / "test_audit_latent.npz") as latent:
                gaps = latent["raw_gap"]
                states = latent["gap_state"]
            meta = json.loads((folder / "meta.json").read_text())
            tau = np.asarray(meta["tau"])
            marginal = marginal_metrics(test, gaps)
            marginal_rows.append({"scenario": scenario, "kappa": kappa, **marginal})
            auc = row_classifier_auc(train, test)
            classifier_rows.append({"scenario": scenario, "kappa": kappa, **auc})
            cutoff = fit_short_gap_threshold(train, tau=tau)
            curve = context_auc_curve(train, test, float(np.median(train.dt_bin[train.valid_mask])))
            for m, value in curve.items():
                context_rows.append({"scenario": scenario, "kappa": kappa, "context_length": m, "auroc": value})
            summaries = compute_behavior_summaries_v2(
                test, tau=tau, short_gap_threshold=cutoff,
                window_width=float(raw["data"]["window_width"]),
            )
            primary = (
                np.mean([
                    abs(class_contrast(summaries[c], test.y_entity))
                    for c in ("velocity", "gap", "fanout")
                ])
                if scenario.endswith("v2a")
                else abs(class_contrast(summaries["joint_alignment"], test.y_entity))
            )
            scenario_ranges.append(primary)
            sequential_rows.append({
                "scenario": scenario, "kappa": kappa,
                "primary_class_contrast": primary,
                "occupancy_y0": float(states[(test.y_entity == 0)[:, None] & test.valid_mask].mean()),
                "occupancy_y1": float(states[(test.y_entity == 1)[:, None] & test.valid_mask].mean()),
            })
            if float(kappa) in (0.0, 1.0):
                current = gate_status(marginal, auc, curve, primary)
                suffix = "kappa0" if float(kappa) == 0 else "kappa1"
                for key, value in current.items():
                    statuses[f"{scenario}_{suffix}_{key}"] = value
        correlation = float(np.corrcoef(raw["coupling"]["kappas"], scenario_ranges)[0, 1])
        statuses[f"{scenario}_gate_d_monotonic_coupling"] = "PASS" if correlation >= 0.90 else "FAIL"
    write_csv(out / "row_marginals.csv", marginal_rows)
    write_csv(out / "single_row_classifier.csv", classifier_rows)
    write_csv(out / "context_classifier_curve.csv", context_rows)
    write_csv(out / "sequential_statistics.csv", sequential_rows)
    # Fail closed: pending equivalence/oracle/bin gates are not silently declared passing.
    statuses.update({
        "gate_a_positionwise_stationarity": "PENDING",
        "gate_b_iid_equivalence_to_c1": "PENDING",
        "gate_c_dgp_oracle_near_c0": "PENDING",
        "gate_e_bin_robustness": "PENDING",
    })
    overall = "PASS" if statuses and all(value == "PASS" for value in statuses.values()) else "FAIL"
    report = {"schema_version": "benchmark_gate_v2.0", "overall_status": overall, **statuses}
    (out / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Benchmark v2 gate report", "", f"Overall: **{overall}**", ""]
    lines.extend(f"- {key}: {value}" for key, value in statuses.items())
    (out / "gate_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
