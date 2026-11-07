"""Stage 1 of rare_state_persistence_v1: evaluate every existing development generation
against the real development table, and compute the real-vs-real noise floor.

No model is trained or sampled here. The sealed test split is never read.
Run on the server inside the research runtime:
    ./run python scripts/run_rare_state_persistence.py --floor-draws 200
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.rare_state_persistence import all_metrics, DIFF_COLUMNS  # noqa: E402

WS = ROOT.parent
SF = ROOT / "artifacts"
FA = WS / "research-argn-fraud-audit/artifacts/sparkov_argn_control_v2"
OUT = ROOT / "docs/rare_state_persistence_v1"
REAL_DEV = FA / "prepared/validation.parquet"
REAL_TRAIN = FA / "prepared/train.parquet"


def registry():
    """family, model, fit, draw -> path. Only generations of the 147 development customers."""
    rows = []
    def add(family, model, fit, draw, path):
        rows.append(dict(family=family, model=model, fit_seed=fit, generation_seed=draw, path=str(path)))
    for fs in [20260928, 20260929]:
        for gs in [20261011, 20261012]:
            add("native_argn", "ARGN_native", fs, gs, FA / f"runs/native_{fs}/generated_validation_{gs}.parquet")
            add("native_argn_config", "ARGN_numeric_digit_relaxed", fs, gs, FA / f"runs/numeric_digit_relaxed_{fs}/generated_validation_{gs}.parquet")
            add("native_argn_config", "ARGN_order_category_numeric_digit_relaxed", fs, gs, FA / f"runs/order_category_numeric_digit_relaxed_{fs}/generated_validation_{gs}.parquet")
            add("other_generator", "CPAR_128_CPU_supported143", fs if fs == 20260928 else None, gs, FA / f"runs/cpar_20260928/generated_supported_validation_{gs}.parquet")
    for fs in [20260930, 20261001]:
        for gs in [20261011, 20261012]:
            for arm in ["B", "B_S"]:
                add("state_first", arm, fs, gs, SF / f"argn_state_first_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
            add("state_first", "B_event_weighted", fs, gs, SF / f"argn_event_weight_control_v1/runs/B_event_weighted_{fs}/generated_validation_{gs}.parquet")
            for arm in ["B_label_first", "B_event_label_first"]:
                add("state_first", arm, fs, gs, SF / f"argn_label_first_control_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
            for arm in ["natural_shared", "balanced_shared", "balanced_mixture", "balanced_routed"]:
                add("output_heads", f"amount_{arm}", fs, gs, SF / f"argn_amount_learning_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
            for arm in ["category_only", "category_and_amount"]:
                add("output_heads", arm, fs, gs, SF / f"argn_category_amount_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
            for arm in ["gap_only", "episode_only", "transition_only", "gap_and_transition"]:
                add("episode_heads", arm, fs, gs, SF / f"argn_gap_episode_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
    for gs in [20261011, 20261012]:
        add("time_calibration", "init_only_20260930", 20260930, gs, SF / f"argn_clock_initial_coupling_v1/runs/init_only_20260930/generated_validation_{gs}.parquet")
        add("time_calibration", "history_only_20260930", 20260930, gs, SF / f"argn_time_density_v1/runs/history_only_20260930/generated_validation_{gs}.parquet")
        add("time_calibration", "clock_prefix", 20260930, gs, SF / f"argn_clock_location_v1/runs/clock_prefix_20260930/generated_validation_{gs}.parquet")
        add("time_calibration", "personal_six", 20260930, gs, SF / f"argn_clock_stratified_loss_v1/runs/personal_six_20260930/generated_validation_{gs}.parquet")
        add("time_calibration", "mixture_marginal", 20260930, gs, SF / f"argn_joint_clock_density_v1/runs/mixture_marginal_20260930/generated_validation_{gs}.parquet")
        add("other_generator", "LBDTPP_official_scale", None, gs, SF / f"lbdtpp_sparkov_v1/official_scale/generated_validation_{gs}.parquet")
        add("other_generator", "LBDTPP_median_scale", None, gs, SF / f"lbdtpp_sparkov_v1/median_scale/generated_validation_{gs}.parquet")
    for gs in [20261011, 20261012]:  # registered 2026-11-07, realtabformer_sparkov_v1 (60-row windows, real length given)
        add("other_generator", "REaLTabFormer_w60", 20261107, gs, SF / f"realtabformer_sparkov_v1/runs/generated_development_{gs}.parquet")
    for fs in [20261021, 20261022, 20261023]:
        for gs in [20261031, 20261032]:
            for arm in ["history_only", "history_position6", "history_calibration_only"]:
                add("time_calibration", arm, fs, gs, SF / f"argn_whole_validation_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                # native_validation_*.parquet of these runs lacks entity_id/event_index (raw engine order);
                # it is not evaluated here rather than guessing the event order.
            add("time_calibration", "init_only", fs, gs, SF / f"argn_matched_init_comparison_v1/runs/init_only_{fs}/generated_validation_{gs}.parquet")
            add("other_generator", "CPAR_length", fs, gs, SF / f"comparison_completion_v1/cpar_length/{fs}/generated_development_{gs}.parquet")
            if fs != 20261023:
                add("other_generator", "MeLD_500", fs, gs, SF / f"meld_500_pilot_v1/runs/{fs}/generated_development_{gs}.parquet")
    return pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-draws", type=int, default=200)
    ap.add_argument("--floor-seed", type=int, default=20261107)
    ap.add_argument("--skip-floor", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    real = pd.read_parquet(REAL_DEV)
    assert real.entity_id.nunique() == 147, real.entity_id.nunique()
    reg = registry()
    results, hazards, skipped, hashes = [], [], [], {}
    for r in reg.itertuples():
        p = Path(r.path)
        if not p.exists():
            skipped.append(dict(model=r.model, path=r.path, reason="missing")); continue
        try:
            g = pd.read_parquet(p)
            if "entity_id" in g and "customer_id" in g and g.entity_id.isna().all():
                g["entity_id"] = g.customer_id
            row, hz = all_metrics(real, g)
        except Exception as e:  # recorded, never silently dropped
            skipped.append(dict(model=r.model, path=r.path, reason=f"{type(e).__name__}: {e}")); continue
        hashes[r.path] = sha(p)
        row.update(family=r.family, model=r.model, fit_seed=r.fit_seed, generation_seed=r.generation_seed, path=r.path)
        results.append(row)
        hz.insert(0, "model", r.model); hz.insert(1, "fit_seed", r.fit_seed); hz.insert(2, "generation_seed", r.generation_seed)
        hazards.append(hz)
        print(f"{r.family:20s} {r.model:45s} fit={r.fit_seed} draw={r.generation_seed} "
              f"cont={row['R_continuation_rate_gen']:.3f} run={row['R_mean_run_length_gen']:.2f} "
              f"gapW1={row['S_gap_logw1']:.3f} catTV={row['S_category_tv']:.3f}", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(OUT / "ALL_GENERATIONS.csv", index=False)
    pd.concat(hazards).to_csv(OUT / "HAZARD_BY_AGE_BAND.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "SKIPPED.csv", index=False)

    floor = pd.DataFrame()
    if not args.skip_floor:
        train = pd.read_parquet(REAL_TRAIN)
        ids = np.sort(train.entity_id.unique())
        assert len(ids) == 688 and not set(ids) & set(real.entity_id)
        rng = np.random.default_rng(args.floor_seed)
        rows = []
        for b in range(args.floor_draws):
            pick = rng.choice(ids, size=147, replace=False)
            sub = train[train.entity_id.isin(pick)]
            row, _ = all_metrics(real, sub)
            row.update(draw=b); rows.append(row)
            if b % 20 == 0:
                print(f"floor draw {b}: cont={row['R_continuation_rate_gen']:.3f} gapW1={row['S_gap_logw1']:.3f}", flush=True)
        floor = pd.DataFrame(rows)
        floor.to_csv(OUT / "FLOOR_DRAWS.csv", index=False)
        summary = floor[DIFF_COLUMNS].agg(["mean", "std", "median", lambda s: s.quantile(.95), "max"]).T
        summary.columns = ["mean", "std", "median", "p95", "max"]
        summary.to_csv(OUT / "FLOOR_SUMMARY.csv", index_label="metric")
        # Degradation ratio: generated difference divided by the real-vs-real median difference.
        ratio = res[["family", "model", "fit_seed", "generation_seed"]].copy()
        for c in DIFF_COLUMNS:
            med = summary.loc[c, "median"]
            ratio[c] = res[c] / med if med and np.isfinite(med) and med > 0 else np.nan
        ratio.to_csv(OUT / "DEGRADATION_RATIO.csv", index=False)
        ratio.groupby(["family", "model"])[DIFF_COLUMNS].mean().to_csv(OUT / "DEGRADATION_RATIO_MEANS.csv")
    res.groupby(["family", "model"])[DIFF_COLUMNS + ["R_continuation_rate_gen", "R_mean_run_length_gen",
        "R_customers_with_fraud_gen", "S_fraud_rate_gen"]].mean().to_csv(OUT / "MEANS.csv")
    json.dump(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(time.time() - t0, 1),
                   real_development=str(REAL_DEV), real_development_sha256=sha(REAL_DEV),
                   real_train=str(REAL_TRAIN), real_train_sha256=sha(REAL_TRAIN) if not args.skip_floor else None,
                   floor_draws=0 if args.skip_floor else args.floor_draws, floor_seed=args.floor_seed,
                   generations_evaluated=len(res), generations_skipped=len(skipped), input_hashes=hashes,
                   new_fits=0, new_generations=0, test_events_read=False),
              open(OUT / "MANIFEST.json", "w"), indent=1)
    print("done", len(res), "generations,", len(skipped), "skipped,", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
