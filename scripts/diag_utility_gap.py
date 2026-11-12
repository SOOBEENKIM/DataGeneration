"""ibm_gap_diag_v1 D2 (docs/ibm_gap_diag_v1/PROTOCOL.md): where does the real-vs-synthetic utility gap come from?
    ./run python scripts/diag_utility_gap.py [--jobs 12]
(1) row swap (real fraud rows + synthetic normal rows, and the reverse), (2) feature-group ablation on both sides,
(3) real-vs-synthetic discriminators on fraud rows / normal rows with grouped permutation importance.
CPU only; train + dev + existing dev generations; never the sealed test; nothing is generated or trained except detectors.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import benchmarks.fraud_utility as fu  # noqa: E402

ART = ROOT / "artifacts"; PREP = ART / "ibm_tabformer_v1/prepared"; OUT = ROOT / "docs/ibm_gap_diag_v1"
FITS = [(fs, gs) for fs in (20261108, 20261109) for gs in (20261011, 20261012)]
SOURCES = {"argn_final": lambda fs, gs: ART / f"ibm_merchant_marginal_v2/runs/visitrate_{fs}/generated_validation_{gs}.parquet",
           "rtf_table": lambda fs, gs: ART / f"ibm_realtabformer_v1/runs/rtf_table_{fs}/generated_validation_{gs}.parquet"}
N_DEV, SEED = 240, 20261108
F = {}       # name -> feature frame (filled before the pool forks)
TEST = None


def groups(cols):
    g = dict(amount=[c for c in cols if "amount" in c], gap=[c for c in cols if "gap" in c], merchant=[c for c in cols if c.startswith("seen_merchant")],
             category=[c for c in cols if c.startswith("cat_") or c == "same_category_5"], channel=[c for c in cols if c.startswith("chan_")], position=["log_position"])
    assert sorted(sum(g.values(), [])) == sorted(cols), set(cols) ^ set(sum(g.values(), []))
    return g


def cols_all():
    return list(fu.ROW) + list(fu.HISTORY)


def _fit_eval(job):
    from threadpoolctl import threadpool_limits
    name, spec, drop = job; t0 = time.time()
    with threadpool_limits(4):
        parts = [F[src][F[src].y.eq(lab)] for src, lab in spec]; tr = pd.concat(parts, ignore_index=True)
        cols = [c for c in cols_all() if c not in set(drop)]
        clf = fu.fit(tr, cols, detector="stable"); score = clf.predict_proba(TEST[cols].to_numpy())[:, 1]
        ep = fu.episode_metrics(TEST, score)
    return dict(name=name, history_auprc=float(average_precision_score(TEST.y, score)), early=ep["early_detection_top1"], train_rows=len(tr), train_fraud=int(tr.y.sum()), seconds=round(time.time() - t0, 1))


def discriminate(real, synth, cols, seed=0, n_splits=5):
    """Grouped-CV AUC of real-vs-synthetic rows and the AUC drop when one feature group is permuted in the held-out fold."""
    X = pd.concat([real[cols], synth[cols]], ignore_index=True).fillna(0).to_numpy(dtype=float); z = np.r_[np.zeros(len(real)), np.ones(len(synth))].astype(int)
    grp = np.r_["R" + real.entity_id.astype(str).to_numpy(), synth.src.astype(str).to_numpy() + synth.entity_id.astype(str).to_numpy()]
    G = groups(cols); idx = {k: [cols.index(c) for c in v] for k, v in G.items()}; rng = np.random.default_rng(seed); aucs = []; drops = {k: [] for k in G}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, z, grp):
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200, learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=20).fit(X[tr], z[tr])
        base = roc_auc_score(z[te], clf.predict_proba(X[te])[:, 1]); aucs.append(base)
        for k, ii in idx.items():
            Xp = X[te].copy(); Xp[:, ii] = Xp[rng.permutation(len(te))][:, ii]; drops[k].append(base - roc_auc_score(z[te], clf.predict_proba(Xp)[:, 1]))
    return dict(auc=float(np.mean(aucs)), n_real=int(len(real)), n_synth=int(len(synth)), permutation_auc_drop={k: float(np.mean(v)) for k, v in drops.items()})


def summary(d):
    on = [c for c in d.columns if c.startswith("chan_")]
    out = dict(rows=int(len(d)), seen_merchant=float(d.seen_merchant.mean()), seen_merchants_5=float(d.seen_merchants_5.mean()), log_amount_mean=float(d.log_amount.mean()), log_amount_sd=float(d.log_amount.std()),
               amount_vs_med20_mean=float(d.amount_vs_med20.mean()), amount_vs_med20_sd=float(d.amount_vs_med20.std()), log_gap_median=float(d.log_gap.median()), gap_vs_med20_mean=float(d.gap_vs_med20.mean()),
               same_category_5=float(d.same_category_5.fillna(0).mean()), log_position_mean=float(d.log_position.mean()), relative_position_mean=float((d.event_index / d.n_events).mean()))
    out.update({c: float(d[c].mean()) for c in on}); return out


def with_run_position(f):
    f = f.copy(); prev = f.groupby("entity_id", sort=False).y.shift().fillna(0); f["onset"] = (f.y.eq(1) & prev.eq(0)).astype(int); f["n_events"] = f.groupby("entity_id", sort=False).y.transform("size"); return f


def main():
    global TEST
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_parquet(PREP / "validation.parquet"); vocab = fu.configure(dev); TEST = fu.features(dev)
    train = pd.read_parquet(PREP / "train.parquet"); ids = np.sort(train.entity_id.unique())
    for i in range(4):
        pick = np.random.default_rng(SEED + i).choice(ids, size=N_DEV, replace=False); F[f"real{i}"] = fu.features(train[train.entity_id.isin(pick)])
    for m, path in SOURCES.items():
        for i, (fs, gs) in enumerate(FITS):
            F[f"{m}{i}"] = fu.features(pd.read_parquet(path(fs, gs)))
    print("features ready", round(time.time() - t0), "s", {k: (len(v), int(v.y.sum())) for k, v in F.items()}, flush=True)
    cols = cols_all(); G = groups(cols); jobs = []
    for i in range(4):
        jobs.append((f"real|RR|{i}|all", [(f"real{i}", 1), (f"real{i}", 0)], []))
        for g, drop in G.items():
            jobs.append((f"real|RR|{i}|no_{g}", [(f"real{i}", 1), (f"real{i}", 0)], drop))
        for m in SOURCES:
            jobs.append((f"{m}|SS|{i}|all", [(f"{m}{i}", 1), (f"{m}{i}", 0)], []))
            jobs.append((f"{m}|RfSn|{i}|all", [(f"real{i}", 1), (f"{m}{i}", 0)], []))
            jobs.append((f"{m}|SfRn|{i}|all", [(f"{m}{i}", 1), (f"real{i}", 0)], []))
            for g, drop in G.items():
                jobs.append((f"{m}|SS|{i}|no_{g}", [(f"{m}{i}", 1), (f"{m}{i}", 0)], drop))
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(_fit_eval, jobs):
            src, kind, i, fs = r["name"].split("|"); r.update(source=src, kind=kind, draw=int(i), features=fs); rows.append(r)
            print(f"{r['name']:38s} auprc={r['history_auprc']:.4f} early={r['early']:.3f} fraud={r['train_fraud']} {r['seconds']}s", flush=True)
    res = pd.DataFrame(rows); res.to_csv(OUT / "UTILITY_GAP_FITS.csv", index=False)
    mean = res.groupby(["source", "kind", "features"]).history_auprc.agg(["mean", "std", "count"]).round(4); mean.to_csv(OUT / "UTILITY_GAP_MEANS.csv"); print(mean.to_string(), flush=True)
    # discriminators: all real train fraud rows vs all synthetic fraud rows of the 4 generations; normal rows 50k each
    ftrain = with_run_position(fu.features(train)); rng = np.random.default_rng(0); disc = {}; summ = dict(real_train=dict(fraud=summary(ftrain[ftrain.y.eq(1)]), onset=summary(ftrain[ftrain.onset.eq(1)]),
                                                                                                               continuation=summary(ftrain[ftrain.y.eq(1) & ftrain.onset.eq(0)]), normal=summary(ftrain[ftrain.y.eq(0)])))
    for m in SOURCES:
        s = pd.concat([with_run_position(F[f"{m}{i}"]).assign(src=f"{m}{i}_") for i in range(4)], ignore_index=True)
        summ[m] = dict(fraud=summary(s[s.y.eq(1)]), onset=summary(s[s.onset.eq(1)]), continuation=summary(s[s.y.eq(1) & s.onset.eq(0)]), normal=summary(s[s.y.eq(0)]))
        rn, sn = ftrain[ftrain.y.eq(0)], s[s.y.eq(0)]
        disc[m] = dict(fraud=discriminate(ftrain[ftrain.y.eq(1)], s[s.y.eq(1)], cols), onset=discriminate(ftrain[ftrain.onset.eq(1)], s[s.onset.eq(1)], cols),
                       continuation=discriminate(ftrain[ftrain.y.eq(1) & ftrain.onset.eq(0)], s[s.y.eq(1) & s.onset.eq(0)], cols),
                       normal=discriminate(rn.iloc[rng.choice(len(rn), 50000, replace=False)], sn.iloc[rng.choice(len(sn), 50000, replace=False)], cols))
        print(m, json.dumps({k: dict(auc=round(v["auc"], 3), drop={g: round(x, 3) for g, x in v["permutation_auc_drop"].items()}) for k, v in disc[m].items()}), flush=True)
    # reference: real train split in two halves of customers (what AUC does "same distribution" give with these sample sizes?)
    half = ftrain.entity_id.isin(ids[::2]); a_, b_ = ftrain[half], ftrain[~half].assign(src="B_")
    disc["real_vs_real"] = dict(fraud=discriminate(a_[a_.y.eq(1)], b_[b_.y.eq(1)], cols), onset=discriminate(a_[a_.onset.eq(1)], b_[b_.onset.eq(1)], cols))
    print("real_vs_real", json.dumps({k: round(v["auc"], 3) for k, v in disc["real_vs_real"].items()}), flush=True)
    (OUT / "UTILITY_GAP.json").write_text(json.dumps(dict(vocabulary=vocab, groups=G, discriminators=disc, feature_summary=summ, seconds=round(time.time() - t0, 1), test_read=False, detector="stable"), indent=1, default=float) + "\n")
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
