from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from benchmarks.types import SequenceBatch
from benchmarks.validation import row_matrix, stratified_entity_bootstrap_auc
from eval.behavior_summaries_v2 import (
    compute_behavior_summaries_v2,
    fit_short_gap_threshold,
)
from eval.joint_association_v2 import association_statistics
from scripts.calibrate_receiver_gate_v2_1 import simultaneous_interval
from scripts.calibrate_receiver_gate_v2_2 import category_counts


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(
            **{key: values[key] for key in SequenceBatch.__dataclass_fields__}
        )


def one_row_per_entity(
    batch: SequenceBatch, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = np.asarray(
        [rng.integers(0, int(length)) for length in batch.lengths]
    )
    rows = np.arange(len(batch.lengths))
    features = np.column_stack(
        [
            batch.x_num[rows, positions, 0],
            batch.dt_bin[rows, positions],
            batch.x_cat[rows, positions, 0],
        ]
    )
    return features, batch.y_entity


def classifier_rows(
    train: SequenceBatch,
    test: SequenceBatch,
    *,
    scenario: str,
    kappa: float,
    resamples: int,
) -> list[dict]:
    x_train, y_train = row_matrix(train)
    x_test_all, y_test_all = row_matrix(test)
    x_entity, y_entity = one_row_per_entity(test, seed=73)
    transformer = ColumnTransformer(
        [
            ("continuous", StandardScaler(), [0]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                [1, 2],
            ),
        ]
    )
    models = {
        "logistic": make_pipeline(
            transformer,
            LogisticRegression(max_iter=300, class_weight="balanced"),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, random_state=42
        ),
    }
    rows = []
    for index, (name, model) in enumerate(models.items()):
        model.fit(x_train, y_train)
        all_scores = model.predict_proba(x_test_all)[:, 1]
        entity_scores = model.predict_proba(x_entity)[:, 1]
        point, low, high = stratified_entity_bootstrap_auc(
            entity_scores,
            y_entity,
            resamples=resamples,
            seed=9000 + index,
        )
        rows.append(
            {
                "scenario": scenario,
                "kappa": kappa,
                "classifier": name,
                "all_rows_point_auroc": float(
                    roc_auc_score(y_test_all, all_scores)
                ),
                "one_row_per_entity_point_auroc": point,
                "entity_bootstrap_ci_low": low,
                "entity_bootstrap_ci_high": high,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_2_candidate.yaml")
    parser.add_argument("--data-root", default="data/benchmark_v2_2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/v2_2_gate")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    data_root, output = Path(args.data_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    classifier, receiver, associations = [], [], []
    interval = raw["receiver_gate"]["primary"]["single_row_auroc_equivalence_interval"]
    receiver_margin = float(
        raw["receiver_gate"]["primary"]["category_practical_margin"]
    )
    for scenario in raw["scenarios"]:
        for kappa in raw["coupling"]["kappas"]:
            folder = data_root / scenario / f"kappa_{float(kappa):.2f}"
            train, test = load_batch(folder / "train.npz"), load_batch(folder / "test.npz")
            meta = json.loads((folder / "meta.json").read_text())
            tau = np.asarray(meta["tau"])
            cutoff = fit_short_gap_threshold(train, tau=tau)
            summaries = compute_behavior_summaries_v2(
                test,
                tau=tau,
                short_gap_threshold=cutoff,
                window_width=float(raw["data"]["window_width"]),
            )
            for channel, values in summaries.items():
                associations.append(
                    {
                        "scenario": scenario,
                        "kappa": float(kappa),
                        "channel": channel,
                        **association_statistics(values, test.y_entity),
                    }
                )
            counts = category_counts(
                test.x_cat[..., 0],
                test.lengths,
                int(raw["data"]["n_receiver_categories"]),
            )
            frequencies = np.divide(
                counts,
                counts.sum(1, keepdims=True),
                out=np.zeros_like(counts, dtype=float),
                where=counts.sum(1, keepdims=True) > 0,
            )
            difference, low, high = simultaneous_interval(
                frequencies, test.y_entity
            )
            receiver.append(
                {
                    "scenario": scenario,
                    "kappa": float(kappa),
                    "max_abs_difference": float(np.max(np.abs(difference))),
                    "max_abs_simultaneous_bound": float(
                        np.max(np.maximum(np.abs(low), np.abs(high)))
                    ),
                    "pass": bool(
                        np.all(low >= -receiver_margin)
                        and np.all(high <= receiver_margin)
                    ),
                }
            )
            if float(kappa) in (0.0, 1.0):
                classifier.extend(
                    classifier_rows(
                        train,
                        test,
                        scenario=scenario,
                        kappa=float(kappa),
                        resamples=int(
                            raw["receiver_gate"]["primary"][
                                "entity_bootstrap_resamples"
                            ]
                        ),
                    )
                )
    write_csv(output / "single_row_auroc_entity_bootstrap.csv", classifier)
    write_csv(output / "receiver_signed_frequency.csv", receiver)
    write_csv(output / "real_continuous_association.csv", associations)
    auc_pass = all(
        row["entity_bootstrap_ci_low"] >= float(interval[0])
        and row["entity_bootstrap_ci_high"] <= float(interval[1])
        for row in classifier
    )
    receiver_pass = all(row["pass"] for row in receiver)
    joint = [
        row for row in associations
        if row["scenario"] == "joint_semimarkov_v2b"
        and row["channel"] == "joint_alignment"
    ]
    deltas = np.asarray([row["delta_joint"] for row in joint])
    association_pass = bool(
        abs(deltas[0])
        <= float(raw["evaluation"]["association"]["kappa0_abs_delta_max"])
        and np.all(np.diff(deltas) >= -float(
            raw["evaluation"]["association"]["monotonic_tolerance"]
        ))
        and deltas[-1] > deltas[0]
    )
    (output / "full_audit_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "benchmark_v2.2-candidate-full-audit",
                "status": (
                    "PASS"
                    if auc_pass and receiver_pass and association_pass
                    else "FAIL"
                ),
                "single_row_auroc_equivalence": "PASS" if auc_pass else "FAIL",
                "signed_receiver_frequency": "PASS" if receiver_pass else "FAIL",
                "real_joint_association": "PASS" if association_pass else "FAIL",
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
