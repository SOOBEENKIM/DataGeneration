"""ibm_normal_impl_v1 (docs/ibm_normal_impl_v1/PROTOCOL.md): the adopted stack with the gap-conditional stay inside the generator (arm R)
next to the adopted stack itself (arm B), 8 independent dev generations each (distinct seeds per fit), then the amount decoding stage.
    CUDA_VISIBLE_DEVICES= ./run python scripts/run_ibm_normal_impl.py generate --fit 20261108      # arms B then R, 4 seeds each
    ./run python scripts/run_ibm_normal_impl.py decode                                              # BA / RA: history-referencing amount for normal rows
    ./run python scripts/run_ibm_normal_impl.py measure                                             # normal-row structure, discriminator, fraud-like share
Amendment 1 (seed extension, registered before running): `--set a1` generates / decodes the 16 new seeds per arm and measures all 24;
`summary` computes the paired utility contrasts. The first 8 generations and their outputs stay as they are.
No training; backbone weights, heads and every existing table are the frozen ones. Never reads a test file.
"""
import argparse, json, logging, os, sys, time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
for k, v in {"LOKY_MAX_CPU_COUNT": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch
from run_ibm_merchant_attr import NATIVE, TABLE, ROUTED_HEADS, MM_PARAMS, CHAN_TABLE, GAP_DIR, CORE, digest, write, seed, tables_path  # noqa: E402
from run_ibm_merchant_marginal_v2 import V1, DOCS as V2_DOCS, name_rare, type_rates  # noqa: E402

OUT = ROOT / "artifacts/ibm_normal_impl_v1"; DOCS = ROOT / "docs/ibm_normal_impl_v1"; NR = ROOT / "artifacts/ibm_normal_rows_v1"
FITS = [20261108, 20261109]
SEEDS = {fs: [20261100 + 10 * i + j for j in range(4)] for i, fs in enumerate(FITS)}      # distinct seeds per fit (ibm_eval_frame_v2 amendment 1)
SEEDS_A1 = {fs: [20261200 + 100 * i + j for j in range(8)] for i, fs in enumerate(FITS)}  # amendment 1: 16 more per arm (the old formula would collide across fits for j >= 4)
ENGINE_ARMS = ("B", "R"); DECODED = {"BA": "B", "RA": "R"}; ALL_ARMS = ("B", "R", "BA", "RA")
assert len({g for fs in FITS for g in SEEDS[fs] + SEEDS_A1[fs]}) == 24


def new_seeds(a, fs):          # what generate / decode produce for the chosen set
    return SEEDS_A1[fs] if a.set == "a1" else SEEDS[fs]


def all_seeds(a, fs):          # what measure / summary read for the chosen set
    return SEEDS[fs] + SEEDS_A1[fs] if a.set == "a1" else SEEDS[fs]


def path(arm, fs, gs):
    return OUT / "runs" / f"{arm}_{fs}" / f"generated_validation_{gs}.parquet"


def cmd_generate(a):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    from benchmarks.argn_merchant_attr import attr_module
    from benchmarks.argn_stay import stay_module
    fs = a.fit; table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); gap_params = json.loads((GAP_DIR / f"gap_tables_{fs}.json").read_text()); stay = json.loads((NR / "tables_v2.json").read_text())["stay"]
    cal = json.loads((V1 / "calibration.json").read_text()); tables = dict(np.load(tables_path(fs))); state_table = json.loads(CHAN_TABLE.read_text())["table"]
    sel = json.loads((V2_DOCS / "SIM_SELECTED.json").read_text())["selected"]; kappa, lam = float(sel["kappa"]), float(sel["lam"]); assert (kappa, lam) == (8.0, 1.0), "frozen kernel"
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    codes = stats["columns"]["receiver_or_mark"]["codes"]; rho = type_rates(cal["categories"], set(map(str, codes.keys())))
    calib = dict(cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel_pos"], rare_no_repeat=True, prior=(kappa * rho[np.asarray(cal["cat_of_m"])]).tolist(), lam=lam)
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    head_path = ROUTED_HEADS / f"routed_subst_{fs}" / "heads.pt"; head_state = torch.load(head_path, map_location="cpu")
    for arm in (a.arms or ENGINE_ARMS):
        for gs in new_seeds(a, fs):
            dest = path(arm, fs, gs); dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            seed(gs); t0 = time.monotonic(); infos = {}
            with ExitStack() as st:
                infos["base"] = st.enter_context(generation_module(stats, table, label_first=True, head_state=head_state, attach=lambda m: attach_routed_heads(m, stats)))
                infos["merchant"] = st.enter_context(merchant_module(stats, mm, arm="full", calibration=calib))
                if arm == "R":
                    infos["stay"] = st.enter_context(stay_module(stats, stay))
                infos["gap"] = st.enter_context(gap_module(stats, gap_params, "table_kind"))
                infos["attr"] = st.enter_context(attr_module(stats, tables, state_table, "attr_all"))
                engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); d = raw.rename(columns={"customer_id": "entity_id"}).copy(); d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d, named = name_rare(d); d.to_parquet(dest, index=False); assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); g = d.groupby("entity_id", sort=False); m = d.receiver_or_mark.astype(str); gapv = pd.to_numeric(d.gap, errors="coerce").astype(float)
            nn = lab.eq(0) & lab.groupby(d.entity_id, sort=False).shift().eq(0); same = m.eq(m.groupby(d.entity_id, sort=False).shift())
            facts = dict(arm=arm, fit_seed=fs, generation_seed=gs, device=device, seconds=round(time.monotonic() - t0, 1), events=int(len(d)), customers=int(d.entity_id.nunique()), frauds=int(lab.sum()), fraud_customers=int(d[lab.eq(1)].entity_id.nunique()),
                         merchants_per_customer_median=float(g.receiver_or_mark.nunique().median()), novel_merchant_normal=float((~d.duplicated(["entity_id", "receiver_or_mark"]))[lab.eq(0)].mean()),
                         same_merchant_gap_lt_10min=float(same[nn & gapv.lt(600)].mean()), same_merchant_gap_ge_1h=float(same[nn & gapv.ge(3600)].mean()), same_merchant_as_previous=float(same[nn].mean()),
                         rare_rows_named=named, base_weights_sha256=weights, heads_sha256=digest(head_path), modules=infos, sha256=digest(dest))
            write(dest.parent / f"generation_validation_{gs}.json", facts)
            print(f"GENERATED {arm} {fs} {gs}: events={facts['events']} frauds={facts['frauds']} merchants={facts['merchants_per_customer_median']} same<10m={facts['same_merchant_gap_lt_10min']:.3f} same>=1h={facts['same_merchant_gap_ge_1h']:.3f} "
                  f"stay={infos.get('stay', {}).get('counter')} {facts['seconds']}s", flush=True)


def cmd_decode(a):
    from sim_normal_rows import base_frame, finish
    from sim_normal_rows_v2 import apply_amt3, TABLES, TABLES3
    T = json.loads(TABLES.read_text()); T["map"] = json.loads(TABLES3.read_text()); log = {}
    for new, src in DECODED.items():
        for fs in FITS:
            for gs in new_seeds(a, fs):
                raw = pd.read_parquet(path(src, fs, gs)); ref = raw.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True); d, used = apply_amt3(base_frame(raw), T, np.random.default_rng([20261115, fs, gs]))
                out = finish(d, ref); fr = pd.to_numeric(ref.event_is_fraud, errors="coerce").eq(1).to_numpy()
                for col in ("receiver_or_mark", "category", "use_chip", "event_is_fraud"):
                    assert (out[col].astype(str).to_numpy() == ref[col].astype(str).to_numpy()).all(), f"{col} changed"
                assert np.allclose(pd.to_numeric(out.gap, errors="coerce").to_numpy(dtype=float), pd.to_numeric(ref.gap, errors="coerce").to_numpy(dtype=float), equal_nan=True), "gap changed"
                assert np.allclose(pd.to_numeric(out.loc[fr, "amount_or_numeric_value"]).to_numpy(dtype=float), pd.to_numeric(ref.loc[fr, "amount_or_numeric_value"]).to_numpy(dtype=float), equal_nan=True), "fraud amounts changed"
                dest = path(new, fs, gs); dest.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(dest, index=False); log[f"{new}_{fs}_{gs}"] = used; print("DECODED", new, fs, gs, used, flush=True)
    write(DOCS / ("DECODE_A1.json" if a.set == "a1" else "DECODE.json"), log)


def cmd_measure(a):
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)
    import benchmarks.fraud_utility as fu
    import diag_utility_gap as dg
    from diag_utility_gap_b import structure
    from sim_normal_rows_v2 import extra
    from run_ibm_merchant_attr import consistency, merchant_attributes
    t0 = time.time(); prep = NATIVE / "prepared"; dev = pd.read_parquet(prep / "validation.parquet"); fu.configure(dev); test = fu.features(dev); train = pd.read_parquet(prep / "train.parquet"); ftrain = fu.features(train); cols = dg.cols_all(); att = merchant_attributes()
    ids = np.sort(train.entity_id.unique()); dets = []
    for i in range(4):
        pick = np.random.default_rng(dg.SEED + i).choice(ids, size=dg.N_DEV, replace=False); clf = fu.fit(fu.features(train[train.entity_id.isin(pick)]), cols, detector="stable"); thr = float(np.quantile(clf.predict_proba(test[cols].to_numpy())[:, 1], 0.99))
        held = np.random.default_rng(dg.SEED + 100 + i).choice(np.setdiff1d(ids, pick), size=dg.N_DEV, replace=False); h = fu.features(train[train.entity_id.isin(held)]); hn = h[h.y.eq(0)]
        dets.append((clf, thr, float((clf.predict_proba(hn[cols].to_numpy())[:, 1] >= thr).mean())))
    rng = np.random.default_rng(0); rn = ftrain[ftrain.y.eq(0)]; rn = rn.iloc[rng.choice(len(rn), 50000, replace=False)]
    res = dict(real_train=dict(structure=structure(train), extra=extra(train), fraud_amount_vs_med20=float(ftrain[ftrain.y.eq(1)].amount_vs_med20.mean()), fraudlike_normal_share=float(np.mean([x[2] for x in dets]))))
    for arm in ALL_ARMS:
        frames = [pd.read_parquet(path(arm, fs, gs)) for fs in FITS for gs in all_seeds(a, fs)]; feats = [fu.features(f).assign(src=f"{arm}{i}_") for i, f in enumerate(frames)]; st = [structure(f) for f in frames]; ex = [extra(f) for f in frames]
        s = pd.concat(feats, ignore_index=True); sn = s[s.y.eq(0)]; sn = sn.iloc[rng.choice(len(sn), 50000, replace=False)]; cons = [consistency(f, att) for f in frames]
        fl = [float(np.mean([(c.predict_proba(f[f.y.eq(0)][cols].to_numpy())[:, 1] >= thr).mean() for c, thr, _ in dets])) for f in feats]
        res[arm] = dict(structure={k: float(np.mean([x[k] for x in st])) for k in st[0]}, extra={k: float(np.mean([x[k] for x in ex])) for k in ex[0]}, normal_discriminator=dg.discriminate(rn, sn, cols),
                        fraud_amount_vs_med20=float(s[s.y.eq(1)].amount_vs_med20.mean()), fraudlike_normal_share=float(np.mean(fl)), normal_category_matches_merchant=float(np.mean([x["normal"]["category_matches_merchant"] for x in cons])),
                        normal_channel_matches_merchant=float(np.mean([x["normal"]["channel_matches_merchant"] for x in cons])), events=float(np.mean([len(f) for f in frames])), frauds=[int(f.y.sum()) for f in feats])
        print(arm, "fraud-like", round(res[arm]["fraudlike_normal_share"], 4), "disc AUC", round(res[arm]["normal_discriminator"]["auc"], 3), {k: round(v, 3) for k, v in res[arm]["normal_discriminator"]["permutation_auc_drop"].items()},
              "fraud amt", round(res[arm]["fraud_amount_vs_med20"], 3), "cons", round(res[arm]["normal_category_matches_merchant"], 4), round(res[arm]["normal_channel_matches_merchant"], 4), "frauds", res[arm]["frauds"], flush=True)
    res["seconds"] = round(time.time() - t0, 1); res["test_read"] = False; res["generations_per_arm"] = sum(len(all_seeds(a, fs)) for fs in FITS); write(DOCS / ("MEASURE_A1.json" if a.set == "a1" else "MEASURE.json"), res)
    keys = ["amount_sd", "amount_sd_within_category", "amount_sd_within_customer_category", "abs_dlog_amount_vs_previous_visit_same_merchant_median", "same_merchant_as_previous", "same_merchant_within_5", "share_rows_top3_merchants",
            "same_category_as_previous", "distinct_merchants_per_customer_median"]
    pd.set_option("display.width", 250); arms = ["real_train", *ALL_ARMS]; print(pd.concat([pd.DataFrame({k: res[k]["structure"] for k in arms}).loc[keys], pd.DataFrame({k: res[k]["extra"] for k in arms})]).round(3).to_string())


def cmd_summary(a):
    """Amendment 1: paired contrasts over the 24 generations (same seed = same lengths and labels in every arm), the new 16 alone, and the
    check that the first 8 reproduce the values stored before the extension."""
    from scipy import stats as ss
    old = {fs: set(SEEDS[fs]) for fs in FITS}; res = {}

    def contrast(d):
        d = np.asarray(d, float); n = len(d); m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)); h = float(ss.t.ppf(0.975, n - 1) * se)
        return dict(n=n, mean=m, se=se, ci95=[m - h, m + h], positive=int((d > 0).sum()), sd=float(d.std(ddof=1)))

    for det in ("stable", "logistic"):
        u = pd.read_csv(DOCS / "a1" / f"utility_detector_{det}" / "UTILITY_ALL.csv"); g = u[u.kind.eq("generation")].copy(); g["fit_seed"] = g.fit_seed.astype(int); g["generation_seed"] = g.generation_seed.astype(int)
        assert "error" not in g or g.error.isna().all(), "utility errors"
        w = g.pivot(index=["fit_seed", "generation_seed"], columns="model_name", values="history_auprc"); assert w.shape == (24, 4) and w.notna().all().all(), w.shape
        is_old = np.array([gs in old[fs] for fs, gs in w.index]); r = dict(per_generation={f"{fs}_{gs}": {k: float(v) for k, v in row.items()} for (fs, gs), row in w.iterrows()},
                                                                           arms={k: dict(mean=float(w[k].mean()), sd=float(w[k].std(ddof=1)), mean_first8=float(w[k][is_old].mean()), mean_new16=float(w[k][~is_old].mean())) for k in ALL_ARMS},
                                                                           row_auprc={k: float(v) for k, v in g.groupby("model_name").row_auprc.mean().items()})
        for x, y in (("RA", "B"), ("RA", "R"), ("BA", "B"), ("R", "B")):
            d = (w[x] - w[y]).to_numpy(); r[f"{x}-{y}"] = dict(all24=contrast(d), first8=contrast(d[is_old]), new16=contrast(d[~is_old]), by_fit={str(fs): contrast(d[w.index.get_level_values(0) == fs]) for fs in FITS})
        prev = pd.read_csv(DOCS / f"utility_detector_{det}" / "UTILITY_ALL.csv"); prev = prev[prev.kind.eq("generation")]
        pw = prev.assign(fit_seed=prev.fit_seed.astype(int), generation_seed=prev.generation_seed.astype(int)).pivot(index=["fit_seed", "generation_seed"], columns="model_name", values="history_auprc")
        r["first8_max_abs_change_vs_stored"] = float((w.loc[pw.index, pw.columns] - pw).abs().to_numpy().max())
        real = u[u.kind.eq("real_subset")].history_auprc; r["ceiling"] = dict(mean=float(real.mean()), sd=float(real.std(ddof=1)), n=int(len(real))); res[det] = r
        print(det, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in r["arms"].items()}, "ceiling", round(r["ceiling"]["mean"], 4), "first8 change", r["first8_max_abs_change_vs_stored"])
        for k in ("RA-B", "RA-R", "BA-B", "R-B"):
            c = r[k]; print(f"  {k}: all24 {c['all24']['mean']:+.4f} se {c['all24']['se']:.4f} ci [{c['all24']['ci95'][0]:+.4f}, {c['all24']['ci95'][1]:+.4f}] pos {c['all24']['positive']}/24 | "
                            f"new16 {c['new16']['mean']:+.4f} se {c['new16']['se']:.4f} pos {c['new16']['positive']}/16 | first8 {c['first8']['mean']:+.4f}")
    c = res["stable"]["RA-B"]; m, lo, new = c["all24"]["mean"], c["all24"]["ci95"][0], c["new16"]["mean"]
    verdict = "미재현" if m <= 0 else "재현" if (m >= 0.02 and lo > 0 and new > 0) else "부분 재현"
    res["I2"] = dict(verdict=verdict, mean=m, ci95_low=lo, new16_mean=new, rule="재현: 평균 >= +.02, 95% 하한 > 0, 새 16개 평균 > 0; 미재현: 평균 <= 0; 그 밖은 부분 재현"); print("I2:", verdict)
    write(DOCS / "A1_SUMMARY.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["generate", "decode", "measure", "summary"]); ap.add_argument("--fit", type=int, default=FITS[0]); ap.add_argument("--arms", nargs="*")
    ap.add_argument("--set", choices=["v1", "a1"], default="v1"); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"); {"generate": cmd_generate, "decode": cmd_decode, "measure": cmd_measure, "summary": cmd_summary}[a.command](a)
