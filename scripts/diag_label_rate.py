"""ibm_gap_diag_v1 D1 (docs/ibm_gap_diag_v1/PROTOCOL.md): why does the frozen transition table emit ~1.6x the train fraud rate?
    ./run python scripts/diag_label_rate.py
CPU only. Reads train, dev and existing dev generations; never the sealed test. No model change: variant tables are simulated only.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from benchmarks.argn_generation_module import fit_table, MAX_AGE, SMOOTH  # noqa: E402

ART = ROOT / "artifacts"; PREP = ART / "ibm_tabformer_v1/prepared"; OUT = ROOT / "docs/ibm_gap_diag_v1"
TABLE = ART / "ibm_generation_module_v1/transition_table.json"
GENS = [ART / f"ibm_merchant_marginal_v2/runs/visitrate_{fs}/generated_validation_{gs}.parquet" for fs in (20261108, 20261109) for gs in (20261011, 20261012)]
REPS = 100


def states(d):
    """Per-row (table row index, previous-run age, ever) exactly as fit_table / the generation module define them."""
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int).eq(1).astype(int); g = d.entity_id
    prev = y.groupby(g).shift(); boundary = g.ne(g.shift()) | y.ne(y.shift()); age = d.groupby(boundary.cumsum()).cumcount() + 1
    prev_age = age.groupby(g).shift().fillna(0).clip(upper=MAX_AGE).astype(int); ever = (y.groupby(g).cumsum() - y).gt(0).astype(int)
    row = np.where(prev.isna(), 0, prev.fillna(0).astype(int) + 1)
    return d, y.to_numpy(), row, prev_age.to_numpy(), ever.to_numpy()


def describe(d, y, ever):
    g = d.entity_id.to_numpy(); f = pd.DataFrame(dict(g=g, y=y, ever=ever)); n = f.groupby("g", sort=False).y.size()
    per = f.groupby("g", sort=False).y.sum(); fc = per[per.gt(0)]
    onset = (f.y.eq(1) & ~(f.y.shift().eq(1) & f.g.eq(f.g.shift()))).sum()
    pos = f.assign(i=f.groupby("g", sort=False).cumcount()); first = pos[pos.y.eq(1)].groupby("g", sort=False).i.min()
    rel = (first / n.reindex(first.index)).to_numpy()
    return dict(events=int(len(f)), frauds=int(y.sum()), rate=float(y.mean()), runs=int(onset), fraud_customers=int(len(fc)), customers=int(len(n)),
                ever1_events=int(ever.sum()), ever1_share=float(ever.mean()), frauds_per_fraud_customer_mean=float(fc.mean()) if len(fc) else None,
                frauds_per_fraud_customer_quantiles={q: float(fc.quantile(q)) for q in (.25, .5, .75, .9)} if len(fc) else None, frauds_per_fraud_customer_max=int(fc.max()) if len(fc) else None,
                top5_customer_share=float(fc.sort_values().tail(5).sum() / max(1, fc.sum())),
                first_fraud_relpos_quantiles={q: float(np.quantile(rel, q)) for q in (.1, .25, .5, .75, .9)} if len(rel) else None)


def forced(table, row, age, ever):
    return float(np.asarray(table)[row, age, ever].sum())


def simulate(table, lengths, reps, seed):
    """Free run of the label chain, vectorised over reps x customers, chains sorted by length so the active set shrinks."""
    rng = np.random.default_rng(seed); T = np.asarray(table); L = np.sort(np.tile(np.asarray(lengths), reps))[::-1]; n = len(L)
    prev = np.zeros(n, dtype=np.int64); age = np.zeros(n, dtype=np.int64); ever = np.zeros(n, dtype=np.int64)
    frauds = np.zeros(n, dtype=np.int64); runs = np.zeros(n, dtype=np.int64); first = np.full(n, -1, dtype=np.int64); ever_events = np.zeros(n, dtype=np.int64)
    k = n
    for t in range(int(L[0])):
        while k > 0 and L[k - 1] <= t:
            k -= 1
        s = slice(0, k)
        p = np.full(k, T[0, 0, 0]) if t == 0 else T[prev[s] + 1, np.minimum(age[s], MAX_AGE), ever[s]]
        ever_events[s] += ever[s]
        lab = (rng.random(k) < p).astype(np.int64)
        runs[s] += lab & ((prev[s] == 0) | (t == 0)); frauds[s] += lab
        first[s] = np.where((first[s] < 0) & (lab == 1), t, first[s])
        same = (prev[s] == lab) & (t > 0); age[s] = np.where(same, age[s] + 1, 1); prev[s] = lab; ever[s] = np.maximum(ever[s], lab)
    per_rep = n // reps; order = np.argsort(-np.tile(np.asarray(lengths), reps), kind="stable")  # chains are exchangeable across reps; aggregate over all
    fc = frauds > 0; rel = first[fc] / L[fc]
    return dict(reps=reps, frauds_per_rep=float(frauds.sum() / reps), rate=float(frauds.sum() / L.sum()), runs_per_rep=float(runs.sum() / reps), fraud_customers_per_rep=float(fc.sum() / reps),
                ever1_events_per_rep=float(ever_events.sum() / reps), ever1_share=float(ever_events.sum() / L.sum()),
                frauds_per_fraud_customer_mean=float(frauds[fc].mean()), frauds_per_fraud_customer_quantiles={q: float(np.quantile(frauds[fc], q)) for q in (.25, .5, .75, .9)},
                frauds_per_fraud_customer_max=int(frauds.max()), first_fraud_relpos_quantiles={q: float(np.quantile(rel, q)) for q in (.1, .25, .5, .75, .9)}, customers_per_rep=per_rep)


def variant_tables(train):
    """Simulation-only variants of the table (never used by the generator)."""
    d, y, row, age, ever = states(train); ok = row > 0; p = row[ok] - 1; a = age[ok]; e = ever[ok]; yy = y[ok]; out = {}
    first = float(y[~ok].mean())
    def build(key_fn, smooth):
        T = np.zeros((3, MAX_AGE + 1, 2)); T[0] = (y[~ok].sum() + smooth) / ((~ok).sum() + 2 * smooth) if smooth else first
        for pp in (0, 1):
            base_sel = p == pp; base = (yy[base_sel].sum() + smooth) / (base_sel.sum() + 2 * smooth)
            for aa in range(1, MAX_AGE + 1):
                for ee in (0, 1):
                    sel = base_sel & key_fn(a, e, aa, ee); n = int(sel.sum())
                    T[pp + 1, aa, ee] = (yy[sel].sum() + smooth) / (n + 2 * smooth) if n > 0 else base
        return T
    out["no_smoothing"] = build(lambda a, e, aa, ee: (a == aa) & (e == ee), 0.0)
    out["no_ever"] = build(lambda a, e, aa, ee: a == aa, SMOOTH)
    out["no_age"] = build(lambda a, e, aa, ee: e == ee, SMOOTH)
    out["markov_only"] = build(lambda a, e, aa, ee: np.ones(len(a), dtype=bool), SMOOTH)
    return out


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(TABLE.read_text()); T = np.asarray(frozen["table"])
    train = pd.read_parquet(PREP / "train.parquet"); dev = pd.read_parquet(PREP / "validation.parquet")
    refit = np.asarray(fit_table(train)["table"]); res = dict(frozen_equals_refit=bool(np.allclose(T, refit)), smoothing=SMOOTH, reps=REPS)
    d, y, row, age, ever = states(train); lengths = d.groupby("entity_id", sort=False).size().to_numpy()
    res["train_real"] = describe(d, y, ever)
    res["train_forced_expected_frauds"] = forced(T, row, age, ever)
    res["train_free_sim"] = simulate(T, lengths, REPS, 1)
    # where the table's mass sits: exposure and expected frauds by cell family, real paths vs the smoothing floor
    cells = pd.DataFrame(frozen["counts"]); cells["raw"] = cells.frauds / cells.events.clip(lower=1); cells["excess"] = (cells.probability - cells.raw) * cells.events
    res["smoothing_excess_by_family"] = {f"prev{p}_ever{e}": dict(events=int(c.events.sum()), frauds=int(c.frauds.sum()), smoothing_excess=float(c.excess.sum()), cells_with_zero_frauds=int(c.frauds.eq(0).sum()))
                                         for (p, e), c in cells.groupby(["previous", "ever"])}
    res["cells_prev0"] = cells[cells.previous.eq(0)][["age", "ever", "events", "frauds", "probability"]].to_dict("records")
    res["variants"] = {}
    for name, V in variant_tables(train).items():
        res["variants"][name] = dict(forced_expected_frauds=forced(V, row, age, ever), free_sim=simulate(V, lengths, REPS, 2))
        print(name, round(res["variants"][name]["forced_expected_frauds"], 1), round(res["variants"][name]["free_sim"]["frauds_per_rep"], 1), flush=True)
    # dev: real, and the adopted model's dev generations (does the generator follow the table? does length matter?)
    dd, dy, drow, dage, dever = states(dev); dl = dd.groupby("entity_id", sort=False).size().to_numpy()
    res["dev_real"] = describe(dd, dy, dever); res["dev_free_sim_real_lengths"] = simulate(T, dl, REPS, 3); res["dev_forced_expected_frauds"] = forced(T, drow, dage, dever)
    gens = []
    for p in GENS:
        g = pd.read_parquet(p); gd, gy, grow, gage, gever = states(g); gl = gd.groupby("entity_id", sort=False).size().to_numpy()
        gens.append(dict(path=str(p), generated=describe(gd, gy, gever), forced_expected_frauds=forced(T, grow, gage, gever), free_sim_generated_lengths=simulate(T, gl, 50, 4)))
        print("gen", gens[-1]["generated"]["frauds"], round(gens[-1]["forced_expected_frauds"], 1), round(gens[-1]["free_sim_generated_lengths"]["frauds_per_rep"], 1), flush=True)
    res["dev_generations"] = gens; res["seconds"] = round(time.time() - t0, 1); res["test_read"] = False
    (OUT / "LABEL_RATE.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    tr, fs = res["train_real"], res["train_free_sim"]
    print(json.dumps(dict(frozen_equals_refit=res["frozen_equals_refit"], train_real=dict(frauds=tr["frauds"], runs=tr["runs"], fraud_customers=tr["fraud_customers"], ever1_share=tr["ever1_share"], relpos=tr["first_fraud_relpos_quantiles"], per=tr["frauds_per_fraud_customer_quantiles"], top5=tr["top5_customer_share"]),
                          forced=res["train_forced_expected_frauds"], free=dict(frauds=fs["frauds_per_rep"], runs=fs["runs_per_rep"], fraud_customers=fs["fraud_customers_per_rep"], ever1_share=fs["ever1_share"], relpos=fs["first_fraud_relpos_quantiles"], per=fs["frauds_per_fraud_customer_quantiles"]),
                          smoothing=res["smoothing_excess_by_family"], dev_real=dict(frauds=res["dev_real"]["frauds"], fraud_customers=res["dev_real"]["fraud_customers"]), dev_sim=res["dev_free_sim_real_lengths"]["frauds_per_rep"], seconds=res["seconds"]), indent=1, default=float))


if __name__ == "__main__":
    main()
