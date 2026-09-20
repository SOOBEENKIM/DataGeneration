"""Frozen train/validation relationship audit; no neural training or test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cs_saf_external_relations_v1.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def clean(s):
    s = s.astype("string").str.strip()
    return s.mask(s.isin(["", "nan", "<NA>"]))


def verify_inputs(data_root, name):
    c = data_root / "canonical" / name
    a = data_root / "manifests/acquisition" / f"{name}.json"
    records = []
    for record in json.loads(a.read_text())["files"]:
        p = data_root / "raw" / name / record["name"]
        actual = digest(p)
        assert actual == record["sha256"], f"raw checksum mismatch: {p}"
        records.append(dict(path=str(p), sha256=actual, kind="raw_bytes_only"))
    m = c / "canonical_manifest.json"
    for filename, record in json.loads(m.read_text())["files"].items():
        p = c / filename
        actual = digest(p)
        assert actual == record["sha256"], f"canonical checksum mismatch: {p}"
        records.append(dict(path=str(p), sha256=actual, kind="canonical_bytes_only"))
    records += [dict(path=str(p), sha256=digest(p), kind="manifest") for p in (a, m)]
    split = pd.read_parquet(c / "entity_splits.parquet")
    assert not split.entity_id.duplicated().any()
    assert set(split.split) == {"train", "validation", "test"}
    return c, split, records


def read_development(c, split, name):
    ids = split.loc[split.split.isin(["train", "validation"]), "entity_id"].tolist()
    cols = ["entity_id", "event_id", "event_index", "timestamp", "gap",
            "receiver_or_mark", "amount_or_numeric_value"]
    cols += ["balance", "transaction_type"] if name == "berka" else ["category", "event_is_fraud"]
    df = pd.read_parquet(c / "events.parquet", columns=cols, filters=[("entity_id", "in", ids)])
    df = df.rename(columns={"receiver_or_mark": "mark", "amount_or_numeric_value": "amount"})
    df["split"] = df.entity_id.map(split.set_index("entity_id").split)
    assert set(df.split) == {"train", "validation"}
    assert not df.entity_id.isin(split.loc[split.split.eq("test"), "entity_id"]).any()
    assert not df[["entity_id", "event_id"]].duplicated().any()
    df = df.sort_values(["entity_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    g = df.groupby("entity_id", sort=False)
    np.testing.assert_array_equal(df.event_index, g.cumcount())
    np.testing.assert_allclose(df.gap, g.timestamp.diff(), rtol=0, atol=0, equal_nan=True)
    assert df.timestamp.notna().all() and df.amount.notna().all() and (df.amount >= 0).all()
    assert (df.gap.dropna() >= 0).all()
    assert df.mark.notna().all()
    return df


def audit_raw(data_root, name, df):
    """Independent raw reconstruction restricted to development identities."""
    ids = set(df.entity_id)
    if name == "berka":
        p = data_root / "raw/berka/trans.asc"
        cols = ["trans_id", "account_id", "date", "type", "operation", "amount", "balance"]
        chunks = pd.read_csv(p, sep=";", usecols=cols, chunksize=100000)
        raw = pd.concat([x.loc[x.account_id.isin(ids)] for x in chunks], ignore_index=True)
        raw["entity_id"], raw["event_id"] = raw.account_id, raw.trans_id
        dt = pd.to_datetime("19" + raw.date.astype(str).str.zfill(6), format="%Y%m%d")
        raw["timestamp"] = dt.astype("int64") / (86400 * 1e9)
        raw["mark"] = clean(raw.operation).fillna(clean(raw.type)).fillna("<MISSING>")
        raw["transaction_type"] = clean(raw.type)
        compare = ["timestamp", "mark", "amount", "balance", "transaction_type"]
        extra = {}
    else:
        p = data_root / "raw/sparkov/fraudTrain.csv"
        cols = ["cc_num", "trans_num", "trans_date_trans_time", "unix_time", "merchant", "category", "amt", "is_fraud"]
        chunks = pd.read_csv(p, usecols=cols, chunksize=100000)
        raw = pd.concat([x.loc[x.cc_num.isin(ids)] for x in chunks], ignore_index=True)
        raw["entity_id"], raw["event_id"] = raw.cc_num, raw.trans_num
        dt = pd.to_datetime(raw.trans_date_trans_time, format="%Y-%m-%d %H:%M:%S")
        raw["timestamp"] = dt.astype("int64") / 1e9
        raw["mark"], raw["amount"] = clean(raw.merchant), raw.amt
        raw["category"], raw["event_is_fraud"] = clean(raw.category), raw.is_fraud.astype(str)
        offset = raw.timestamp - raw.unix_time
        extra = dict(provided_unix_offset_min_seconds=float(offset.min()),
                     provided_unix_offset_max_seconds=float(offset.max()),
                     provided_unix_offset_unique=int(offset.nunique()))
        compare = ["timestamp", "mark", "amount", "category", "event_is_fraud"]
    raw = raw.sort_values(["entity_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    assert len(raw) == len(df)
    for col in ["entity_id", "event_id"] + compare:
        if pd.api.types.is_numeric_dtype(raw[col]):
            np.testing.assert_allclose(raw[col], df[col], rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(raw[col].astype(str), df[col].astype(str))
    return dict(development_rows_verified=len(raw), compared_columns=compare,
                exact_equality=True, test_outcome_analysis=False, **extra)


def partition(train_ids, name, seed, fraction):
    ranked = sorted(train_ids, key=lambda x: hashlib.sha256(f"{name}:{x}:{seed}".encode()).digest())
    n = int(len(ranked) * fraction)
    return set(ranked[:n]), set(ranked[n:])


def make_edges(gap, quantiles):
    positive = np.asarray(gap)[np.isfinite(gap) & (np.asarray(gap) > 0)]
    assert len(positive)
    return np.unique(np.quantile(positive, quantiles))


def gap_codes(gap, edges):
    a = np.asarray(gap)
    return np.where(~np.isfinite(a), -1, np.where(a == 0, 0, 1 + np.searchsorted(edges, a, side="left")))


def sequence_features(df, edges, order="canonical"):
    x = df.copy()
    if order == "reverse":
        x = x.sort_values(["entity_id", "timestamp", "event_id"], ascending=[True, True, False], kind="stable")
    elif order.startswith("shuffle_"):
        x["_tie"] = np.random.default_rng(int(order.split("_")[1])).random(len(x))
        x = x.sort_values(["entity_id", "timestamp", "_tie"], kind="stable").drop(columns="_tie")
    else:
        assert order == "canonical"
    x = x.reset_index(drop=True)
    same_entity = x.entity_id.eq(x.entity_id.shift())
    tie_size = x.groupby(["entity_id", "timestamp"], sort=False).entity_id.transform("size")
    x["has_previous"] = same_entity
    x["tied_event"] = tie_size.gt(1)
    x["unambiguous"] = same_entity & tie_size.eq(1) & tie_size.shift().eq(1)
    x["gap"] = x.timestamp.diff().where(same_entity)
    x["gap_bin"] = gap_codes(x.gap, edges)
    x["previous_mark"] = x.mark.shift().where(same_entity)
    x["repeat"] = x.mark.eq(x.previous_mark).astype(float)
    secondary = "transaction_type" if "transaction_type" in x else "category"
    x["secondary"] = x[secondary].fillna("<MISSING>").astype(str)
    x["secondary_repeat"] = (x.secondary.eq(x.secondary.shift()) & same_entity).astype(float)
    x["log_amount"] = np.log1p(x.amount)
    # Reset run chains at every ambiguous link; then shift to strictly past state.
    continuation = x.repeat.eq(1) & x.unambiguous
    run = x.groupby((~continuation).cumsum(), sort=False).cumcount() + 1
    x["prior_run"] = run.shift().where(same_entity)
    x["run_bin"] = pd.cut(x.prior_run, [0, 1, 3, 7, np.inf], labels=["1", "2-3", "4-7", "8+"]).astype("string")
    if "balance" in x:
        x["balance_delta"] = x.balance.diff().where(same_entity)
    return x


class ClusterSummary:
    def __init__(self, entity_ids, repetitions, seed):
        self.ids = pd.Index(sorted(set(entity_ids)))
        n = len(self.ids)
        self.weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=repetitions).astype(float)

    def mean(self, ids, values):
        v = np.asarray(values, float)
        i = self.ids.get_indexer(np.asarray(ids))
        assert (i >= 0).all() and np.isfinite(v).all() and len(v)
        count = np.bincount(i, minlength=len(self.ids)).astype(float)
        sums = np.bincount(i, weights=v, minlength=len(self.ids))
        den = self.weights @ count
        boot = np.divide(self.weights @ sums, den, out=np.full(len(den), np.nan), where=den > 0)
        q = np.nanquantile(boot, [.025, .975])
        present = count > 0
        return dict(value=float(v.mean()), ci_low=float(q[0]), ci_high=float(q[1]),
                    entity_mean=float((sums[present]/count[present]).mean()),
                    rows=int(len(v)), entities=int(present.sum()))


def grouped_stats(x, keys, variables, cluster, cfg, prefix):
    rows = []
    for key, group in x.groupby(keys, observed=True, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for var in variables:
            ok = group[var].notna()
            if not ok.any():
                continue
            s = cluster.mean(group.loc[ok, "entity_id"], group.loc[ok, var])
            rows.append(dict(**prefix, **dict(zip(keys, key)), metric=var, **s,
                             coverage=s["rows"] >= cfg["min_transitions"] and s["entities"] >= cfg["min_entities"]))
    return rows


class EmpiricalDiagnostics:
    """Hierarchical observed-transition and log-amount tables; never consume fraud labels."""
    def __init__(self, vocabulary, n_bins, alpha=.5, shrink=20):
        self.vocabulary = {v: i+1 for i, v in enumerate(vocabulary)}
        self.k, self.b, self.alpha, self.shrink = len(vocabulary)+1, n_bins, alpha, shrink

    def codes(self, x):
        return (x.mark.map(self.vocabulary).fillna(0).to_numpy(int),
                x.previous_mark.map(self.vocabulary).fillna(0).to_numpy(int),
                x.gap_bin.to_numpy(int))

    def fit(self, x):
        m, p, g = self.codes(x)
        assert len(x) and (g >= 0).all() and (g < self.b).all()
        k, b, a = self.k, self.b, self.shrink
        counts = np.bincount(m, minlength=k)
        self.p0 = (counts+self.alpha)/(len(m)+k*self.alpha)
        c1 = np.bincount(p*k+m, minlength=k*k).reshape(k,k)
        self.p1 = (c1+a*self.p0)/(c1.sum(-1, keepdims=True)+a)
        c2 = np.bincount((p*b+g)*k+m, minlength=k*b*k).reshape(k,b,k)
        self.p2 = (c2+a*self.p1[:,None,:])/(c2.sum(-1, keepdims=True)+a)
        y = x.log_amount.to_numpy()
        self.a0 = float(y.mean())
        n1 = np.bincount(m, minlength=k)
        self.a1 = (np.bincount(m, weights=y, minlength=k)+a*self.a0)/(n1+a)
        n2 = np.bincount(m*b+g, minlength=k*b).reshape(k,b)
        self.a2 = (np.bincount(m*b+g, weights=y, minlength=k*b).reshape(k,b)+a*self.a1[:,None])/(n2+a)
        for prob in (self.p0, self.p1, self.p2):
            np.testing.assert_allclose(prob.sum(-1), 1., atol=1e-12)
        return self

    def losses(self, x):
        m, p, g = self.codes(x)
        y = x.log_amount.to_numpy()
        return {"M0_mark_nll": -np.log(self.p0[m]), "M1_mark_nll": -np.log(self.p1[p,m]),
                "M2_mark_nll": -np.log(self.p2[p,g,m]), "A0_amount_mae": np.abs(y-self.a0),
                "A1_amount_mae": np.abs(y-self.a1[m]), "A2_amount_mae": np.abs(y-self.a2[m,g])}


def simple_scores(train, evaluation, vocab, b, cluster, cfg, prefix):
    model = EmpiricalDiagnostics(vocab, b, cfg["marginal_pseudocount"], cfg["conditional_pseudocount"]).fit(train)
    losses = model.losses(evaluation)
    rows = [dict(**prefix, comparison=k, **cluster.mean(evaluation.entity_id, v)) for k,v in losses.items()]
    for small, large in [("M0_mark_nll", "M1_mark_nll"), ("M1_mark_nll", "M2_mark_nll"),
                         ("A0_amount_mae", "A1_amount_mae"), ("A1_amount_mae", "A2_amount_mae")]:
        diff = losses[large]-losses[small]
        rows.append(dict(**prefix, comparison=f"{large}_minus_{small}",
                         relative_gain=float(-diff.mean()/losses[small].mean()),
                         **cluster.mean(evaluation.entity_id, diff)))
    return rows


def dataset_run(data_root, name, out, cfg):
    print(f"{name}: verifying files", flush=True)
    c, splits, inputs = verify_inputs(data_root, name)
    df = read_development(c, splits, name)
    raw_check = audit_raw(data_root, name, df)
    train_ids = splits.loc[splits.split.eq("train"), "entity_id"].tolist()
    fit_ids, check_ids = partition(train_ids, name, cfg["partition_seed"], cfg["fit_fraction"])
    fit = df.loc[df.entity_id.isin(fit_ids)]
    edges = make_edges(fit.gap, cfg["positive_gap_quantiles"])
    vocab = sorted(fit.mark.unique().tolist())
    threshold = float(fit.amount.quantile(cfg["amount_high_quantile"]))
    x = sequence_features(df, edges)
    summaries, profiles, fraud, balance, scores, sensitivity = [], [], [], [], [], []
    for split_name in ["train", "validation"]:
        s = x.loc[x.split.eq(split_name)].copy()
        primary = s.loc[s.unambiguous]
        lengths = s.groupby("entity_id").size()
        summaries.append(dict(dataset=name, split=split_name, entities=int(s.entity_id.nunique()), events=len(s),
            transitions=int(s.has_previous.sum()), unambiguous_transitions=len(primary),
            unambiguous_fraction=float(s.unambiguous.sum()/s.has_previous.sum()),
            tied_event_fraction=float(s.tied_event.mean()), zero_gap_fraction=float(s.gap.eq(0).sum()/s.has_previous.sum()),
            length_median=float(lengths.median()), length_p95=float(lengths.quantile(.95)),
            length_max=int(lengths.max()), fraction_entities_over_32=float((lengths>32).mean()),
            vocabulary_size=int(s.mark.nunique()), fit_vocabulary_oov_fraction=float((~s.mark.isin(vocab)).mean()),
            missing_first_gap=int(s.gap.isna().sum())))
        cluster = ClusterSummary(s.entity_id, cfg["bootstrap_repetitions"], cfg["bootstrap_seed"])
        prefix = dict(dataset=name, split=split_name, view="unambiguous")
        print(f"{name}/{split_name}: profiles", flush=True)
        for keys in [["gap_bin"], ["run_bin"], ["gap_bin", "run_bin"]]:
            profiles.extend(grouped_stats(primary, keys, ["repeat", "secondary_repeat", "log_amount"], cluster, cfg,
                                          dict(**prefix, grouping="x".join(keys))))
        if name == "sparkov":
            s["fraud"] = s.event_is_fraud.astype(int)
            s["high_amount"] = np.where(s.amount > threshold, "above_fit_p95", "at_or_below_fit_p95")
            s["all"] = "all"
            for keys in [["all"], ["gap_bin"], ["secondary"], ["high_amount"]]:
                fraud.extend(grouped_stats(s, keys, ["fraud"], cluster, cfg,
                                           dict(dataset=name, split=split_name, grouping="x".join(keys))))
            summaries[-1].update(fraud_events=int(s.fraud.sum()), fraud_entities=int(s.loc[s.fraud.eq(1)].entity_id.nunique()))
        else:
            balance.extend(grouped_stats(primary, ["secondary"], ["amount", "balance", "balance_delta"], cluster, cfg, prefix))
    fit_x = x.loc[x.entity_id.isin(fit_ids) & x.unambiguous]
    check_x = x.loc[x.entity_id.isin(check_ids) & x.unambiguous]
    tr = x.loc[x.split.eq("train") & x.unambiguous]
    va = x.loc[x.split.eq("validation") & x.unambiguous]
    for label, training, evaluation in [("internal_check", fit_x, check_x), ("validation", tr, va)]:
        cluster = ClusterSummary(evaluation.entity_id, cfg["bootstrap_repetitions"], cfg["bootstrap_seed"])
        scores.extend(simple_scores(training, evaluation, vocab, len(edges)+2, cluster, cfg,
                                    dict(dataset=name, evaluation=label, view="unambiguous")))
    # Ordering sensitivity is descriptive; no CI or pseudo-replicate treatment.
    for order in ["canonical", "reverse"] + [f"shuffle_{i}" for i in cfg["tie_seeds"]]:
        print(f"{name}: tie sensitivity {order}", flush=True)
        y = x if order == "canonical" else sequence_features(df, edges, order)
        for split_name in ["train", "validation"]:
            z = y.loc[y.split.eq(split_name) & y.has_previous]
            for key, subset in [("all", z)] + [(f"gap_{g}", q) for g,q in z.groupby("gap_bin")]:
                sensitivity.append(dict(dataset=name, split=split_name, order=order, cell=key,
                    rows=len(subset), entities=int(subset.entity_id.nunique()), repeat=float(subset.repeat.mean()),
                    secondary_repeat=float(subset.secondary_repeat.mean())))
        # Unknown tied order must not affect unambiguous adjacency measurements.
        old = x.loc[x.unambiguous, ["entity_id", "event_id", "gap", "previous_mark", "repeat"]]
        new = y.loc[y.unambiguous, old.columns]
        pd.testing.assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True))
    frames = dict(coverage=summaries, profiles=profiles, empirical_scores=scores, tie_sensitivity=sensitivity,
                  fraud_profiles=fraud, balance_profiles=balance)
    for filename, rows in frames.items():
        if rows:
            pd.DataFrame(rows).to_csv(out / f"{name}_{filename}.csv", index=False)
    state = dict(dataset=name, gap_edges=edges.tolist(), gap_bin_count=len(edges)+2,
        gap_units="days" if name == "berka" else "seconds", amount_fit_p95=threshold,
        fit_entities=len(fit_ids), internal_check_entities=len(check_ids), fit_mark_vocabulary_size=len(vocab),
        fit_ids_sha256=hashlib.sha256(json.dumps(sorted(fit_ids)).encode()).hexdigest(),
        vocabulary_sha256=hashlib.sha256(json.dumps(vocab).encode()).hexdigest(),
        raw_verification=raw_check, order_invariance_checks=5, inputs=inputs)
    write_json(out / f"{name}_audit.json", state)
    return state


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, name in zip(axes, ["berka", "sparkov"]):
        p = pd.read_csv(out / f"{name}_profiles.csv")
        for split, color in [("train", "#3074b4"), ("validation", "#d95f02")]:
            s = p.loc[p.grouping.eq("gap_bin") & p.metric.eq("repeat") & p.split.eq(split)].sort_values("gap_bin")
            ax.errorbar(s.gap_bin, s.value*100, yerr=np.array([s.value-s.ci_low, s.ci_high-s.value])*100,
                        marker="o", capsize=3, label=split, color=color)
        ax.set(title=f"{name.title()}: unambiguous adjacent transactions", xlabel="Gap bin (positive train quantiles)", ylabel="Same mark probability (%)")
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Observed-data diagnostic; entity bootstrap 95% intervals", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "gap_repeat.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,3.7))
    for ax,name in zip(axes,["berka","sparkov"]):
        p=pd.read_csv(out/f"{name}_empirical_scores.csv")
        for label,color in [("internal_check","#3074b4"),("validation","#d95f02")]:
            q=p.set_index(["evaluation","comparison"])
            v=[q.loc[(label,f"M{i}_mark_nll"),"value"] for i in range(3)]
            ax.plot([0,1,2],v,marker="o",label=label,color=color)
        ax.set_xticks([0,1,2],["Marginal","Previous mark","Previous + gap"])
        ax.set(title=name.title(),ylabel="Conditional mark NLL (lower better)")
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle("Empirical prediction tables, not sequence generators",fontsize=11)
    fig.tight_layout();fig.savefig(out/"empirical_nll.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(), "do not overwrite an existing scientific run"
    args.output.mkdir(parents=True)
    cfg=json.loads(CONFIG.read_text())
    started=time.time()
    states=[dataset_run(args.data_root,n,args.output,cfg) for n in cfg["datasets"]]
    plot(args.output)
    write_json(args.output/"run_manifest.json",dict(
        version=cfg["version"],source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_sha256=digest(__file__),config_sha256=digest(CONFIG),host=platform.node(),
        elapsed_seconds=time.time()-started,datasets=[s["dataset"] for s in states],
        neural_fits=0,new_generations=0,test_outcome_analysis=False,
        files={p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(f"COMPLETE: {args.output}",flush=True)


if __name__=="__main__":
    main()
