"""Evaluate every registered free draw; no detector, refit, or output repair."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from run_argn_state_first import ROOT, OUT, SOURCE, CFG, digest, write, check_manifest
from evaluate_sparkov_argn_control import extended, numeric_metrics, additional_relations, conditions, ratio_ci
from benchmarks.argn_fraud_audit import summaries, position_curves, tv, AMOUNT, MERCHANT
from benchmarks.argn_state_evaluation import (
    episode_features, scalar_summary, hazard_table, age_amount_table, log_w1,
    bootstrap_rates, histogram_by_customer, bootstrap_w1_hist, interval,
)

DOCS = ROOT / "docs/argn_state_first_v1/evaluation"
ART = OUT / "evaluation"
REPEATS = 500
BOOT_SEED = 20260927


def reference_variability(real, runs, ids, weights):
    scalar = scalar_summary(real, runs)
    rates = bootstrap_rates(real, ids, weights)
    rows = [dict(metric=k, reference_value=scalar[k], **interval(v), kind="customer-bootstrap scalar") for k, v in rates.items()]
    rates_fixed = {k: np.full(REPEATS, scalar[k]) for k in rates}
    for k, v in rates.items():
        rows.append(dict(metric=k + "_absolute_error", reference_value=0, **interval(np.abs(v-rates_fixed[k])), kind="real-real resample vs fixed reference"))
    for metric, values, entities in [
        ("run_length_log_w1", runs.length.to_numpy(), runs.entity_id.to_numpy()),
        ("fraud_ratio_log_w1", real.loc[real.fraud.eq(1) & real.amount_history_ratio_raw.notna(), "amount_history_ratio_raw"].to_numpy(),
         real.loc[real.fraud.eq(1) & real.amount_history_ratio_raw.notna(), "entity_id"].to_numpy()),
    ]:
        valid = np.isfinite(values) & (values >= 0)
        values, entities = values[valid], entities[valid]
        support = np.unique(values)
        counts = histogram_by_customer(values, entities, ids, support)
        sampled = weights @ counts
        fixed_cdf = np.cumsum(counts.sum(axis=0)) / counts.sum()
        sampled_cdf = np.cumsum(sampled, axis=1) / sampled.sum(axis=1)[:, None]
        distance = np.sum(np.abs(sampled_cdf[:, :-1] - fixed_cdf[:-1]) * np.diff(np.log1p(support)), axis=1)
        rows.append(dict(metric=metric, reference_value=0, **interval(distance), kind="real-real resample vs fixed reference"))
    return rows, rates


def fit_summaries():
    rows = []
    for arm in CFG["arms"]:
        for fit_seed in CFG["fit_seeds"]:
            folder = OUT / "runs" / f"{arm}_{fit_seed}"
            data = pd.read_csv(folder / "workspace/ModelStore/model-data/progress-messages.csv")
            valid = data[data.val_loss.notna()]
            best = valid.loc[valid.val_loss.idxmin()]
            fit = json.loads((folder / "FIT.json").read_text())
            rows.append(dict(arm=arm, fit_seed=fit_seed, stopped_epoch=data.epoch.max(),
                             best_epoch=best.epoch, best_val_loss=best.val_loss, seconds=fit["seconds"],
                             weights_sha256=fit["weights_sha256"], parameters=fit["total_parameters"]))
    return rows


def main():
    check_manifest()
    ART.mkdir(parents=True, exist_ok=False)
    DOCS.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), test_events_read=False,
                    bootstrap_repeats=REPEATS, bootstrap_seed=BOOT_SEED,
                    uncertainty_scope="customer resampling conditional on each fit/draw; 2 independent training seeds",
                    inputs={}, code={})
    for p in [Path(__file__), ROOT / "benchmarks/argn_state_evaluation.py",
              ROOT / "benchmarks/argn_fraud_audit.py", ROOT / "scripts/evaluate_sparkov_argn_control.py",
              ROOT / "docs/argn_state_first_v1/EVALUATION_PROTOCOL.md"]:
        manifest["code"][str(p.relative_to(ROOT))] = digest(p)
    write(ART / "START.json", manifest)
    state_path = SOURCE / "prepared/metric_state.json"
    state = json.loads(state_path.read_text())
    train_path, real_path = [SOURCE / "prepared" / f"{s}.parquet" for s in ["train", "validation"]]
    for p in [state_path, train_path, real_path]:
        manifest["inputs"][str(p)] = digest(p)
    raw_train = pd.read_parquet(train_path)
    raw_real = pd.read_parquet(real_path)
    real, real_runs = episode_features(extended(raw_real, state))
    ids = np.sort(real.entity_id.unique())
    assert len(ids) == 147
    rng = np.random.default_rng(BOOT_SEED)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=REPEATS)
    reference, real_boot_rates = reference_variability(real, real_runs, ids, weights)
    pd.DataFrame(reference).to_csv(DOCS / "real_only_variability.csv", index=False)
    print("REAL_ONLY_REFERENCE_COMPLETE", flush=True)

    metrics, numeric, hazards, age_amounts, conditional, positions = [], [], [], [], [], []
    boot_errors = {}
    paths = [("real_validation", None, None, None, real_path), ("real_train", None, None, None, train_path)]
    for arm in CFG["arms"]:
        assert json.loads((OUT / f"queue_{arm}.json").read_text())["status"] == "complete"
        for fs in CFG["fit_seeds"]:
            folder = OUT / "runs" / f"{arm}_{fs}"
            fit = json.loads((folder / "FIT.json").read_text())
            weight_path = folder / "workspace/ModelStore/model-data/model-weights.pt"
            # Ask the pinned workspace for the authoritative checkpoint filename.
            from mostlyai.engine._workspace import Workspace
            weight_path = Workspace(folder / "workspace").model_tabular_weights_path
            assert digest(weight_path) == fit["weights_sha256"]
            for gs in CFG["generation_seeds"]:
                path = folder / f"generated_validation_{gs}.parquet"
                meta = json.loads((folder / f"generation_validation_{gs}.json").read_text())
                assert digest(path) == meta["sha256"] and meta["weights_sha256"] == fit["weights_sha256"]
                paths.append((f"{arm}_{fs}_{gs}", arm, fs, gs, path))
    for name, arm, fs, gs, path in paths:
        manifest["inputs"][str(path)] = digest(path)
        raw = pd.read_parquet(path)
        if arm:
            assert set(raw.entity_id) == set(ids)
        d, runs = episode_features(extended(raw, state))
        # Invalid output is a failure to investigate, not an silently repaired span.
        assert np.isfinite(d[AMOUNT]).all() and d[AMOUNT].ge(0).all()
        gaps = d.loc[d.event_index.gt(0), "gap"]
        assert np.isfinite(gaps).all() and gaps.ge(0).all()
        meta = dict(run=name, arm=arm or name, fit_seed=fs, generation_seed=gs)
        result = dict(**meta, **summaries(raw_real, raw, state), **scalar_summary(d, runs))
        result.update(additional_relations(real, d))
        result["previous_current_fraud_tv"] = tv(real[real.event_index.gt(0)], d[d.event_index.gt(0)], ["previous_fraud", "fraud"])
        result["run_length_log_w1"] = log_w1(real_runs.length, runs.length)
        result["run_span_log_w1"] = log_w1(real_runs.span_seconds, runs.span_seconds)
        rr = real_runs[real_runs.completed_known_start]
        ss = runs[runs.completed_known_start]
        result["completed_run_length_log_w1"] = log_w1(rr.length, ss.length)
        result["completed_run_span_log_w1"] = log_w1(rr.span_seconds, ss.span_seconds)
        bands, amount_result = age_amount_table(real, d, name)
        result.update(amount_result)
        for key in ["fraud_rate", "onset_rate", "continuation_rate"]:
            result[key + "_absolute_error"] = abs(result[key] - scalar_summary(real, real_runs)[key])
        length_r, length_s = [x.groupby("entity_id").size() for x in [real, d]]
        result["customer_length_log_w1"] = log_w1(length_r, length_s)
        numeric_rows = numeric_metrics(real, d, name)
        for row in numeric_rows:
            result[f"{row['label']}_{row['field']}_log_w1"] = row["wasserstein_log1p"]
        metrics.append(result); numeric.extend(numeric_rows); age_amounts.extend(bands)
        hazards.extend(hazard_table(d, name))
        positions.extend([dict(run=name, **v) for v in position_curves(raw_real, raw, raw_train, state)])
        for condition, mask in conditions(d).items():
            conditional.append(dict(run=name, condition=condition, **ratio_ci(d.loc[mask], d.entity_id.unique(), REPEATS)))
        # Per-customer episode tables stay local: public results contain aggregates.
        runs.to_parquet(ART / f"episodes_{name}.parquet", index=False)
        if arm:
            rates = bootstrap_rates(d, ids, weights)
            errors = {k + "_absolute_error": np.abs(rates[k] - real_boot_rates[k]) for k in rates}
            support = np.arange(1, int(max(real_runs.length.max(), runs.length.max())) + 1)
            rc = histogram_by_customer(real_runs.length, real_runs.entity_id, ids, support)
            sc = histogram_by_customer(runs.length, runs.entity_id, ids, support)
            errors["run_length_log_w1"] = bootstrap_w1_hist(rc, sc, weights, support)
            boot_errors[arm, fs, gs] = errors
        print("EVALUATED", name, flush=True)
    table = pd.DataFrame(metrics)
    table.to_csv(DOCS / "metrics.csv", index=False)
    for filename, rows in [("numeric.csv", numeric), ("hazards.csv", hazards), ("age_amounts.csv", age_amounts),
                           ("conditional_rates.csv", conditional), ("position_curves.csv", positions),
                           ("fit_summary.csv", fit_summaries())]:
        pd.DataFrame(rows).to_csv(DOCS / filename, index=False)
    error_cols = [k for k in table if k.endswith("_log_w1") or k.endswith("_tv") or k.endswith("_absolute_error")]
    pairs, boot_rows = [], []
    for fs in CFG["fit_seeds"]:
        for gs in CFG["generation_seeds"]:
            b = table[(table.arm == "B") & (table.fit_seed == fs) & (table.generation_seed == gs)].iloc[0]
            s = table[(table.arm == "B_S") & (table.fit_seed == fs) & (table.generation_seed == gs)].iloc[0]
            for metric in error_cols:
                delta = s[metric] - b[metric]
                pairs.append(dict(fit_seed=fs, generation_seed=gs, metric=metric, B=b[metric], B_S=s[metric], delta_S_minus_B=delta))
            for metric, arr in boot_errors["B", fs, gs].items():
                delta = boot_errors["B_S", fs, gs][metric] - arr
                boot_rows.append(dict(fit_seed=fs, generation_seed=gs, metric=metric,
                                      delta_S_minus_B=s[metric]-b[metric], **interval(delta)))
    paired = pd.DataFrame(pairs)
    paired.to_csv(DOCS / "paired_draw_differences.csv", index=False)
    paired.groupby(["fit_seed", "metric"])[["B", "B_S", "delta_S_minus_B"]].mean().reset_index().to_csv(DOCS / "paired_fit_means.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(DOCS / "paired_customer_bootstrap.csv", index=False)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"] = {p.name: digest(p) for p in DOCS.glob("*.csv")}
    manifest["no_output_or_weight_mutation"] = True
    write(DOCS / "manifest.json", manifest)
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
