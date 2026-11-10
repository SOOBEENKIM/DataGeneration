"""ibm_merchant_marginal_v2: CPU simulation of the merchant choice process alone (docs/ibm_merchant_marginal_v2/PROTOCOL.md).
    ./run python scripts/sim_merchant_process.py
Train + development only (sequence lengths and reference statistics); the sealed test is never read. No GPU, no backbone.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SF = ROOT / "artifacts"; IBM = SF / "ibm_tabformer_v1/prepared"
OUT = ROOT / "docs/ibm_merchant_marginal_v2"
EDGES = np.array([1, 5, 20, 100, 500]); HI = ["5499", "5541", "4829", "4784"]
TOP = ["5499", "5541", "4829", "4784", "5812", "5411", "5912", "4900", "5311", "7538"]
G = {}


def setup():
    mm = json.loads((SF / "ibm_merchant_memory_v1/merchant_memory.json").read_text()); cal = json.loads((SF / "ibm_merchant_marginal_v1/calibration.json").read_text())
    cats = cal["categories"]; K = len(cats); rare = K - 1
    from mostlyai.engine._workspace import Workspace
    codes = set(map(str, Workspace(SF / "ibm_tabformer_v1/runs/native_20261108/workspace").tgt_stats.read()["columns"]["receiver_or_mark"]["codes"].keys()))
    tr = pd.read_parquet(IBM / "train.parquet"); tr["m"] = tr.receiver_or_mark.astype(str); tr["c"] = tr.category.astype(str)
    y = pd.to_numeric(tr.event_is_fraud, errors="coerce").fillna(0).astype(int); trn = tr[y.eq(0)]
    modal = trn.groupby("m").c.agg(lambda s: s.value_counts().index[0]); is_rare = ~trn.m.isin(codes)
    # relative visit rate by merchant type (known merchants), and the actual-category mix of rare-merchant rows
    per = trn[~is_rare].groupby(["entity_id", "m"]).size().rename("v").reset_index(); per["c"] = per.m.map(modal)
    rho = per.groupby("c").v.mean() / per.v.mean(); rho_vec = np.array([float(rho.get(c, 1.0)) for c in cats]); rho_vec[rare] = 1.0
    cidx = {c: i for i, c in enumerate(cats)}
    rare_mix = np.bincount(trn[is_rare].c.map(cidx).dropna().astype(int), minlength=K).astype(float); rare_mix[rare] = 0; rare_mix /= rare_mix.sum()
    G.update(gate=np.array(mm["gate"][0]), lam=float(mm["lam"]), cats=cats, K=K, rare=rare, rho=rho_vec, rare_mix=rare_mix, q_flat=np.array(cal["p_cat_novel"]), q_pos=np.array(cal["p_cat_novel_pos"]),
             hi=np.array([cidx[c] for c in HI]), top=np.array([cidx[c] for c in TOP]), cidx=cidx)
    return tr, codes


def reference(frame, codes):
    """Real statistics of normal rows in the simulator's terms (rare merchants keep distinct ids)."""
    d = frame.sort_values(["entity_id", "event_index"], kind="stable"); y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); d = d[y.eq(0)].copy()
    d["m"] = d.receiver_or_mark.astype(str); d["c"] = d.category.astype(str).map(G["cidx"]); d = d[d.c.notna()]; d["c"] = d.c.astype(int)
    cat = np.bincount(d.c, minlength=G["K"]).astype(float); cat /= cat.sum()
    per = d.groupby(["entity_id", "m"]).agg(v=("c", "size"), c=("c", "first")).reset_index(); tot = per.groupby("entity_id").v.sum()
    t1 = per.sort_values("v", ascending=False).drop_duplicates("entity_id")
    return dict(cat=cat, merchants=float(per.groupby("entity_id").size().median()), top1=float((per.groupby("entity_id").v.max() / tot).median()), hi_top1=float(t1.c.isin(G["hi"]).mean()),
                visits={c: float(per[per.c.eq(G["cidx"][c])].v.mean()) for c in TOP}, rare_share=float((~d.m.isin(codes)).mean()), novel=float((~d.duplicated(["entity_id", "m"])).mean()),
                lengths=d.groupby("entity_id").size().to_numpy())


def simulate(job):
    lengths, cfg, seed = job
    rng = np.random.default_rng(seed); K, rare, rho, gate = G["K"], G["rare"], G["rho"], G["gate"]
    Q = G["q_pos"] if cfg["table"] == "pos" else np.tile(G["q_flat"], (6, 1)); Qc = np.cumsum(Q, 1)
    lam, beta, kappa, rare_repeat = cfg.get("lam", G["lam"]), cfg.get("beta", 0.0), cfg.get("kappa", 0.0), cfg.get("rare_repeat", False)
    cat_rows = np.zeros(K); n_rows = 0; rare_rows = 0; novel_rows = 0; merchants = []; top1 = []; hi_top1 = 0
    vis_sum = np.zeros(K); vis_n = np.zeros(K); rare_cum = np.cumsum(G["rare_mix"])
    for T in lengths:
        cnt = np.zeros(256); last = np.zeros(256); typ = np.zeros(256, dtype=int); w = np.zeros(256); M = 0; rare_slot = -1; distinct = 0; rare_count = 0
        bins = np.searchsorted(EDGES, np.arange(T), side="right"); u = rng.random((T, 3))
        for n in range(T):
            g = 1.0 if n == 0 else gate[bins[n]]
            if u[n, 0] < g or M == 0 or (M == 1 and rare_slot == 0 and not rare_repeat):
                c = int(np.searchsorted(Qc[bins[n]], u[n, 1] * Qc[bins[n], -1]))
                if c == rare:
                    rare_count += 1; distinct += 1  # each rare pick = a distinct rare merchant (naming rule); novel by definition
                    novel_rows += 1 if not rare_repeat else (1 if rare_slot < 0 else 0)
                    if rare_repeat:   # v1 behaviour: one code that accumulates and can be repeated
                        if rare_slot < 0:
                            if M == len(cnt):
                                cnt, last, typ, w = (np.concatenate([a, np.zeros_like(a)]) for a in (cnt, last, typ, w))
                            rare_slot = M; typ[M] = rare; w[M] = 1.0; M += 1
                        cnt[rare_slot] += 1; last[rare_slot] = n
                    cat_rows[int(np.searchsorted(rare_cum, u[n, 2]))] += 1
                    continue
                if M == len(cnt):
                    cnt, last, typ, w = (np.concatenate([a, np.zeros_like(a)]) for a in (cnt, last, typ, w))
                typ[M] = c; cnt[M] = 1; last[M] = n; w[M] = rho[c] ** beta if beta else 1.0; M += 1; distinct += 1; novel_rows += 1; cat_rows[c] += 1
            else:
                f = cnt[:M] + (kappa * rho[typ[:M]] if kappa else 0.0); f = f / f.sum(); r = 1.0 / (n - last[:M]); r = r / r.sum()
                p = (lam * f + (1 - lam) * r) * w[:M]
                if not rare_repeat and rare_slot >= 0:
                    p[rare_slot] = 0.0
                pc = np.cumsum(p); j = int(np.searchsorted(pc, u[n, 1] * pc[-1])); j = min(j, M - 1)
                cnt[j] += 1; last[j] = n
                if typ[j] == rare:
                    rare_count += 1; cat_rows[int(np.searchsorted(rare_cum, u[n, 2]))] += 1
                else:
                    cat_rows[typ[j]] += 1
        n_rows += T; rare_rows += rare_count
        known = typ[:M] != rare; v = cnt[:M][known]; t = typ[:M][known]
        np.add.at(vis_sum, t, v); np.add.at(vis_n, t, 1)
        string_distinct = int(known.sum()) + (rare_count if not rare_repeat else (1 if rare_slot >= 0 else 0))
        merchants.append(string_distinct)
        best = cnt[:M].max() if M else 0; top1.append(best / T); hi_top1 += int(M > 0 and typ[int(np.argmax(cnt[:M]))] in G["hi"])
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = vis_sum / vis_n
    return dict(cat=cat_rows / cat_rows.sum(), merchants=float(np.median(merchants)), top1=float(np.median(top1)), hi_top1=hi_top1 / len(lengths), rare_share=rare_rows / n_rows, novel=novel_rows / n_rows,
                visits={c: float(vis[G["cidx"][c]]) for c in TOP})


def summarise(runs, ref):
    tv = float(np.mean([0.5 * np.abs(r["cat"] - ref["cat"]).sum() for r in runs])); m = float(np.mean([r["merchants"] for r in runs]))
    vis = {c: float(np.nanmean([r["visits"][c] for r in runs])) for c in TOP}
    sp = float(pd.Series(vis).corr(pd.Series(ref["visits"]), method="spearman"))
    return dict(std_cat_tv=tv, merchants=m, merchants_reldiff=abs(m - ref["merchants"]) / ref["merchants"], top1=float(np.mean([r["top1"] for r in runs])), hi_top1=float(np.mean([r["hi_top1"] for r in runs])),
                rare_share=float(np.mean([r["rare_share"] for r in runs])), novel=float(np.mean([r["novel"] for r in runs])), visits_spearman=sp,
                visits_log_error=float(np.mean([abs(np.log(vis[c] / ref["visits"][c])) for c in TOP])), visits=vis)


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); tr, codes = setup()
    dev = pd.read_parquet(IBM / "validation.parquet"); ref_dev = reference(dev, codes)
    ids = np.sort(tr.entity_id.unique()); pick = np.random.default_rng(20261110).choice(ids, 240, replace=False); ref_tr = reference(tr[tr.entity_id.isin(pick)], codes)
    seeds = [1, 2, 3]
    validation = {"novelcat": dict(table="flat", rare_repeat=True), "novelcat_pos": dict(table="pos", rare_repeat=True), "novelcat_pos_rarefix": dict(table="pos", rare_repeat=False)}
    cands = {}
    for lam in (0.75, 1.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            cands[f"mult_b{b}_l{lam}"] = dict(table="pos", rare_repeat=False, beta=b, lam=lam)
        for k in (0.5, 1.0, 2.0, 4.0, 8.0):
            cands[f"add_k{k}_l{lam}"] = dict(table="pos", rare_repeat=False, kappa=k, lam=lam)
    jobs = [(ref_dev["lengths"], c, s) for c in validation.values() for s in seeds] + [(ref_tr["lengths"], c, s) for c in cands.values() for s in seeds]
    with ProcessPoolExecutor(max_workers=16, initializer=setup) as ex:
        res = list(ex.map(simulate, jobs))
    it = iter(res); val = {k: summarise([next(it) for _ in seeds], ref_dev) for k in validation}
    measured = {"novelcat": (0.369, 0.143), "novelcat_pos": (0.272, 0.089), "novelcat_pos_rarefix": (0.261, 0.016)}
    for k, (tv, rs) in measured.items():
        val[k].update(measured_std_cat_tv=tv, measured_rare_share=rs, tv_ok=bool(abs(val[k]["std_cat_tv"] - tv) <= 0.04), rare_ok=bool(abs(val[k]["rare_share"] - rs) <= 0.02))
    order_ok = val["novelcat"]["std_cat_tv"] > val["novelcat_pos"]["std_cat_tv"] >= val["novelcat_pos_rarefix"]["std_cat_tv"] - 0.02
    passed = bool(all(v["tv_ok"] and v["rare_ok"] for v in val.values()) and order_ok)
    refs = {k: {a: (b if not isinstance(b, np.ndarray) else None) for a, b in r.items() if a not in ("cat", "lengths")} for k, r in (("dev", ref_dev), ("train240", ref_tr))}
    json.dump(dict(validation=val, order_ok=bool(order_ok), passed=passed, reference=refs), open(OUT / "SIM_VALIDATION.json", "w"), indent=1)
    print("VALIDATION passed =", passed); print(pd.DataFrame(val).T[["std_cat_tv", "measured_std_cat_tv", "rare_share", "measured_rare_share", "merchants", "top1", "hi_top1", "visits_spearman"]].round(3).to_string())
    print("reference", json.dumps(refs)[:900])
    rows = []
    for k, c in cands.items():
        s = summarise([next(it) for _ in seeds], ref_tr); rows.append(dict(config=k, **c, **{a: b for a, b in s.items() if a != "visits"}, **{f"visits_{x}": s["visits"][x] for x in TOP}))
    df = pd.DataFrame(rows); df["eligible"] = df.merchants_reldiff.le(0.12) & df.top1.between(0.5 * ref_tr["top1"], 1.5 * ref_tr["top1"]); df.to_csv(OUT / "SIM_CANDIDATES.csv", index=False)
    print(df[["config", "std_cat_tv", "merchants", "merchants_reldiff", "top1", "hi_top1", "rare_share", "visits_spearman", "visits_log_error", "eligible"]].round(3).to_string())
    sel = None
    if passed and df.eligible.any():
        best = df[df.eligible].sort_values("std_cat_tv").iloc[0]; cfg = cands[best.config]
        with ProcessPoolExecutor(max_workers=8, initializer=setup) as ex:
            devruns = list(ex.map(simulate, [(ref_dev["lengths"], cfg, s) for s in (11, 12, 13)]))
        d = summarise(devruns, ref_dev); gate = bool(d["std_cat_tv"] <= 0.159 and d["merchants_reldiff"] <= 0.12)
        sel = dict(config=best.config, **cfg, train=best.drop(labels=["config"]).to_dict(), dev=d, gate_passed=gate, gate=dict(std_cat_tv_max=0.159, merchants_reldiff_max=0.12))
        print("SELECTED", best.config, "dev:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "visits"}, "GATE", gate); print("dev visits", {k: round(v, 1) for k, v in d["visits"].items()})
    json.dump(dict(selected=sel, validation_passed=passed, seconds=round(time.time() - t0)), open(OUT / "SIM_SELECTED.json", "w"), indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    print("seconds", round(time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
