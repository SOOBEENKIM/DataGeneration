"""fraud_structure_v1 (docs/fraud_structure_v1/PROTOCOL.md): four-layer fraud structure of the real data and its
reproduction by the backbones and the current model. Diagnosis only: no training, no sampling, CPU.
    ./run python scripts/run_fraud_structure.py --dataset ibm
    ./run python scripts/run_fraud_structure.py --dataset sparkov
The sealed test split is never read (IBM raw timestamps are loaded for train/validation customers only).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.fraud_structure import structure_stats, distances, add_anchored_timestamp, ARRAY_DISTANCES, SCALAR_DISTANCES  # noqa: E402

WS = ROOT.parent
SF = ROOT / "artifacts"
FA = WS / "research-argn-fraud-audit/artifacts/sparkov_argn_control_v2"
IBM = SF / "ibm_tabformer_v1"
SPK = Path("/home/ssd-990/soobeenkim/DataGeneration/data/sparkov_prepared")  # server path; ROOT resolves through a symlink
OUT = ROOT / "docs/fraud_structure_v1"
FLOOR_SEED, FLOOR_DRAWS = 20261107, 200


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ibm_timestamps(prepared, roles, splits=("train", "validation")):
    """Absolute timestamps of the 2013 window for train/validation customers, joined on (customer, event_index)."""
    keep = roles[roles.split.isin(list(splits))]
    raw = pd.read_parquet(IBM / "raw_events.parquet", columns=["user", "year", "timestamp"])
    raw = raw[raw.year.eq(2013) & raw.user.isin(keep.user)].sort_values(["user", "timestamp"], kind="stable")
    raw["event_index"] = raw.groupby("user", sort=False).cumcount()
    raw = raw.merge(keep[["user", "customer_id"]], on="user").rename(columns={"customer_id": "entity_id"})
    m = prepared.merge(raw[["entity_id", "event_index", "timestamp"]], on=["entity_id", "event_index"], how="left", validate="one_to_one")
    assert m.timestamp.notna().all(), "timestamp join incomplete"
    # consistency: the recorded gap must equal the timestamp difference
    chk = m.sort_values(["entity_id", "event_index"]); dt = chk.groupby("entity_id").timestamp.diff()
    assert np.nanmax(np.abs(dt.to_numpy() - chk.gap.to_numpy())) < 1, "gap/timestamp mismatch"
    return m


def dataset(name):
    if name == "ibm":
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles)
        dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")])
        numeric = ["cards_before", "events_per_year_before", "first_year"]
        gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model="ARGN_native", role="backbone", fit=fs, draw=gs, path=IBM / f"runs/native_{fs}/generated_validation_{gs}.parquet"),
                         dict(model="ARGN_model", role="current_model", fit=fs, draw=gs, path=SF / f"ibm_routed_heads_v1/runs/routed_subst_table_{fs}/generated_validation_{gs}.parquet"),
                         dict(model="RTF_native", role="backbone", fit=fs, draw=gs, path=SF / f"ibm_realtabformer_v1/runs/rtf_native_{fs}/generated_validation_{gs}.parquet"),
                         dict(model="RTF_model", role="current_model", fit=fs, draw=gs, path=SF / f"ibm_realtabformer_v1/runs/rtf_table_{fs}/generated_validation_{gs}.parquet")]
                gens += [dict(model=f"ARGN_mm_{arm}", role="merchant_memory", fit=fs, draw=gs, path=SF / f"ibm_merchant_memory_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["repeat_blind", "gate_only", "full"]]  # ibm_merchant_memory_v1 (2026-11-09)
                gens += [dict(model=f"ARGN_gap_{arm}", role="run_gap", fit=fs, draw=gs, path=SF / f"ibm_run_gap_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["table_pooled", "table_fraud", "table_kind", "ratio_kind"]]  # ibm_run_gap_v1 (2026-11-10)
                gens += [dict(model=f"ARGN_{arm}", role="merchant_attr", fit=fs, draw=gs, path=SF / f"ibm_merchant_attr_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["attr_cat", "attr_chan", "attr_normal", "attr_all"]]  # ibm_merchant_attr_v1 (2026-11-10)
                gens += [dict(model=f"ARGN_{arm}", role="merchant_marginal_v2", fit=fs, draw=gs, path=SF / f"ibm_merchant_marginal_v2/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["rarefix_named", "visitrate"]]  # ibm_merchant_marginal_v2 (2026-11-10)
                gens += [dict(model=f"ARGN_{arm}", role="merchant_marginal", fit=fs, draw=gs, path=SF / f"ibm_merchant_marginal_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["novelcat", "novelcat_pos", "novelcat_pos_rarefix"]]  # ibm_merchant_marginal_v1 (2026-11-10)
                gens += [dict(model=f"RTF_mm_{arm}", role="merchant_memory", fit=fs, draw=gs, path=SF / f"ibm_realtabformer_v1/runs/rtf_{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["full", "fraud_only", "fraud_only_chan"]]  # H5 (2026-11-10); fraud_only = amendment 2
        return dict(train=train, dev=dev, ctx=ctx, numeric=numeric, gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_impl_a1":  # ibm_normal_impl_v1 amendment 1 (2026-11-11): 2013 dev, all 24 independent generations per arm
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for fs, seeds in {20261108: [*range(20261100, 20261104), *range(20261200, 20261208)], 20261109: [*range(20261110, 20261114), *range(20261300, 20261308)]}.items():
            for gs in seeds:
                for arm in ("B", "R", "BA", "RA"):
                    gens.append(dict(model=arm, role="normal_impl", fit=fs, draw=gs, path=SF / f"ibm_normal_impl_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet"))
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_impl":  # ibm_normal_impl_v1 (2026-11-11): 2013 dev, 8 independent generations per arm
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for i, fs in enumerate([20261108, 20261109]):
            for gs in [20261100 + 10 * i + j for j in range(4)]:  # distinct seeds per fit (ibm_eval_frame_v2 amendment 1)
                for arm in ("B", "R", "BA", "RA"):
                    gens.append(dict(model=arm, role="normal_impl", fit=fs, draw=gs, path=SF / f"ibm_normal_impl_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet"))
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_normal4":  # ibm_normal_rows_v1 amendment 4 (2026-11-11): 2013 dev
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model=arm, role="normal_rows_v4", fit=fs, draw=gs, path=SF / folder / f"generated_validation_{gs}.parquet") for arm, folder in [("base", f"ibm_merchant_marginal_v2/runs/visitrate_{fs}"), ("ALL3", f"ibm_normal_rows_v1/sim3/ALL3_{fs}"), ("ALL4", f"ibm_normal_rows_v1/sim4/ALL4_{fs}")]]
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_normal3":  # ibm_normal_rows_v1 amendment 3 (2026-11-11): 2013 dev
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model=arm, role="normal_rows_v3", fit=fs, draw=gs, path=SF / folder / f"generated_validation_{gs}.parquet") for arm, folder in [("base", f"ibm_merchant_marginal_v2/runs/visitrate_{fs}"), ("ALL2", f"ibm_normal_rows_v1/sim2/ALL2_{fs}")] + [(a, f"ibm_normal_rows_v1/sim3/{a}_{fs}") for a in ("AMT3", "ALL3")]]
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_normal2":  # ibm_normal_rows_v1 amendment 2 (2026-11-11): 2013 dev
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model=arm, role="normal_rows_v2", fit=fs, draw=gs, path=SF / folder / f"generated_validation_{gs}.parquet") for arm, folder in [("base", f"ibm_merchant_marginal_v2/runs/visitrate_{fs}")] + [(a, f"ibm_normal_rows_v1/sim2/{a}_{fs}") for a in ("REC2", "AMT0", "AMT2", "REC2_AMT0", "ALL2")]]
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_normal":  # ibm_normal_rows_v1 (2026-11-11): 2013 dev
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model=arm, role="normal_rows", fit=fs, draw=gs, path=SF / folder / f"generated_validation_{gs}.parquet") for arm, folder in [("base", f"ibm_merchant_marginal_v2/runs/visitrate_{fs}")] + [(a, f"ibm_normal_rows_v1/sim/{a}_{fs}") for a in ("AMT", "REC", "GAP", "AMT_REC", "ALL")]]
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_episode":  # ibm_episode_state_v1 (2026-11-11): 2013 dev
        roles = pd.read_parquet(IBM / "prepared/roles.parquet")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/validation.parquet"), roles)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/validation_context.parquet")]); gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model=arm, role="episode_state", fit=fs, draw=gs, path=SF / folder / f"generated_validation_{gs}.parquet") for arm, folder in [("episode", f"ibm_episode_state_v1/runs/episode_{fs}"), ("argn_final", f"ibm_merchant_marginal_v2/runs/visitrate_{fs}"), ("rtf_table", f"ibm_realtabformer_v1/runs/rtf_table_{fs}")]]
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=240, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/validation.parquet")))
    if name == "ibm_test":  # ibm_sealed_test_v1 (2026-11-10): the sealed test, opened once
        roles = pd.read_parquet(IBM / "prepared/roles.parquet"); both = ("train", "test")
        train = ibm_timestamps(pd.read_parquet(IBM / "prepared/train.parquet"), roles, both); dev = ibm_timestamps(pd.read_parquet(IBM / "prepared/test.parquet"), roles, both)
        ctx = pd.concat([pd.read_parquet(IBM / "prepared/train_context.parquet"), pd.read_parquet(IBM / "prepared/test_context.parquet")])
        gens = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                gens += [dict(model=arm, role="sealed_test", fit=fs, draw=gs, path=SF / f"ibm_sealed_test_v1/runs/{arm}_{fs}/generated_test_{gs}.parquet") for arm in ["argn_native", "argn_minimal", "argn_merchant", "argn_gap", "argn_final"]]
                gens += [dict(model=arm, role="sealed_test", fit=fs, draw=gs, path=SF / f"ibm_realtabformer_v1/runs/{arm}_{fs}/generated_test_{gs}.parquet") for arm in ["rtf_native", "rtf_table"]]
        return dict(train=train, dev=dev, ctx=ctx, numeric=["cards_before", "events_per_year_before", "first_year"], gens=gens, n_dev=239, n_train=1000,
                    sources=dict(train=str(IBM / "prepared/train.parquet"), dev=str(IBM / "prepared/test.parquet")))
    if name == "sparkov":
        train = pd.read_parquet(SPK / "train.parquet"); dev = pd.read_parquet(SPK / "validation.parquet")
        ctx = pd.concat([pd.read_parquet(SPK / "train_context.parquet"), pd.read_parquet(SPK / "validation_context.parquet")])
        numeric = ["birth_year", "city_population"]
        gens = []
        for fs in [20260928, 20260929]:
            for gs in [20261011, 20261012]:
                gens.append(dict(model="ARGN_native", role="backbone", fit=fs, draw=gs, path=FA / f"runs/native_{fs}/generated_validation_{gs}.parquet"))
        for fs in [20260930, 20261001]:
            for gs in [20261011, 20261012]:
                gens.append(dict(model="ARGN_gap_transition", role="current_model", fit=fs, draw=gs, path=SF / f"argn_gap_episode_v1/runs/gap_and_transition_{fs}/generated_validation_{gs}.parquet"))
        for fs in [20260928, 20260929]:  # sparkov_transfer_v1 (2026-11-10)
            for gs in [20261011, 20261012]:
                gens += [dict(model=f"ARGN_{arm}", role="transfer", fit=fs, draw=gs, path=SF / f"sparkov_transfer_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet")
                         for arm in ["sf_base", "sf_gap", "sf_attr", "sf_attr_all", "sf_merchant", "sf_full"]]
        for gs in [20261011, 20261012]:
            gens.append(dict(model="RTF_native", role="backbone", fit=20261107, draw=gs, path=SF / f"realtabformer_sparkov_v1/runs/generated_development_{gs}.parquet"))
        return dict(train=train, dev=dev, ctx=ctx, numeric=numeric, gens=gens, n_dev=147, n_train=688,
                    sources=dict(train=str(SPK / "train.parquet"), dev=str(SPK / "validation.parquet")))
    raise ValueError(name)


def scalar_part(s):
    return {k: v for k, v in s.items() if not k.startswith("_") and not isinstance(v, list)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["ibm", "sparkov", "ibm_test", "ibm_episode", "ibm_normal", "ibm_normal2", "ibm_normal3", "ibm_normal4", "ibm_impl", "ibm_impl_a1"]); ap.add_argument("--floor-draws", type=int, default=FLOOR_DRAWS)
    a = ap.parse_args(); ds = dataset(a.dataset); out = ROOT / "docs/ibm_sealed_test_v1/structure" if a.dataset == "ibm_test" else ROOT / "docs/ibm_episode_state_v1/structure" if a.dataset == "ibm_episode" else ROOT / "docs/ibm_normal_rows_v1/structure" if a.dataset == "ibm_normal" else ROOT / "docs/ibm_normal_rows_v1/v2/structure" if a.dataset == "ibm_normal2" else ROOT / "docs/ibm_normal_rows_v1/v3/structure" if a.dataset == "ibm_normal3" else ROOT / "docs/ibm_normal_rows_v1/v4/structure" if a.dataset == "ibm_normal4" else ROOT / "docs/ibm_normal_impl_v1/structure" if a.dataset == "ibm_impl" else ROOT / "docs/ibm_normal_impl_v1/a1/structure" if a.dataset == "ibm_impl_a1" else OUT / a.dataset; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    train, dev = ds["train"], ds["dev"]
    assert not set(train.entity_id) & set(dev.entity_id)
    s_train = structure_stats(train, ds["ctx"], ds["numeric"]); s_dev = structure_stats(dev, ds["ctx"], ds["numeric"])
    real = pd.DataFrame({"train": scalar_part(s_train), "dev": scalar_part(s_dev)})
    real.to_csv(out / "REAL_STRUCTURE.csv", index_label="statistic")
    json.dump({k: v for k, v in s_train.items() if isinstance(v, list)}, open(out / "REAL_TRAIN_LISTS.json", "w"), indent=1)
    np.savez(out / "REAL_TRAIN_ARRAYS.npz", **{k: np.asarray(v, dtype=float) for k, v in s_train["_arrays"].items()})
    print(real.to_string(), flush=True)
    # anchor for generated timestamps: real dev first-event time per customer
    dev_sorted = dev.sort_values(["entity_id", "event_index"]); first_ts = dev_sorted.groupby("entity_id").timestamp.first()
    rows, skipped = [], []
    for g in ds["gens"]:
        if not Path(g["path"]).exists():
            skipped.append(dict(g, path=str(g["path"]))); continue
        gen = pd.read_parquet(g["path"]).drop(columns=["timestamp"], errors="ignore")
        gen = add_anchored_timestamp(gen, first_ts)
        sg = structure_stats(gen, ds["ctx"], ds["numeric"])
        row = dict(model=g["model"], role=g["role"], fit_seed=g["fit"], generation_seed=g["draw"], path=str(g["path"]), sha256=sha(g["path"]))
        row.update(distances(s_dev, sg)); row.update({f"gen_{k}": v for k, v in scalar_part(sg).items()}); rows.append(row)
        print(f"{g['model']} {g['fit']} {g['draw']}: frauds {sg['n_fraud']} customers {sg['n_fraud_customers']} episodes {sg['n_episodes24h']}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(out / "GENERATIONS.csv", index=False); pd.DataFrame(skipped).to_csv(out / "SKIPPED.csv", index=False)
    # floor: train subsamples of n_dev customers vs dev
    ids = np.sort(train.entity_id.unique()); assert len(ids) == ds["n_train"]
    rng = np.random.default_rng(FLOOR_SEED); frows = []
    for b in range(a.floor_draws):
        pick = rng.choice(ids, size=ds["n_dev"], replace=False)
        sub = train[train.entity_id.isin(pick)]
        d = distances(s_dev, structure_stats(sub, ds["ctx"], ds["numeric"])); d["draw"] = b; frows.append(d)
        if b % 25 == 0:
            print(f"floor draw {b} ({time.time() - t0:.0f}s)", flush=True)
    floor = pd.DataFrame(frows); floor.to_csv(out / "FLOOR_DRAWS.csv", index=False)
    dist_cols = [c for c in floor.columns if c != "draw"]
    summary = floor[dist_cols].agg(["mean", "std", "median", lambda s: s.quantile(.95), "max"]).T; summary.columns = ["mean", "std", "median", "p95", "max"]
    summary.to_csv(out / "FLOOR_SUMMARY.csv", index_label="metric")
    ratio = res[["model", "role", "fit_seed", "generation_seed"]].copy()
    for c in dist_cols:
        med = summary.loc[c, "median"]
        ratio[c] = res[c] / med if np.isfinite(med) and med > 0 else np.nan
    ratio.to_csv(out / "RATIOS.csv", index=False)
    mean_ratio = ratio.drop(columns=["fit_seed", "generation_seed"]).groupby(["model", "role"]).mean(numeric_only=True).T
    mean_ratio.to_csv(out / "RATIOS_BY_MODEL.csv", index_label="metric")
    print(mean_ratio.round(2).to_string(), flush=True)
    json.dump(dict(dataset=a.dataset, sources=ds["sources"], source_sha256={k: sha(v) for k, v in ds["sources"].items()}, floor_seed=FLOOR_SEED,
                   floor_draws=a.floor_draws, generations=len(res), skipped=len(skipped), seconds=round(time.time() - t0),
                   protocol_sha256=sha(OUT / "PROTOCOL.md"), test_read=bool(a.dataset == "ibm_test")), open(out / "RUN.json", "w"), indent=1)


if __name__ == "__main__":
    main()
