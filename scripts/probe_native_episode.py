"""native_argn_episode_probe_v1: frozen native ARGN engine, one forward pass in teacher mode
over (a) real development sequences and (b) the engine's own free generations.

Reports the label-head fraud probability grouped by the state of the previous event
(normal / fraud run age 1, 2-5, 6-10, 11-20, 21+) next to the realised next-label rate.
Weights are never changed; nothing is rolled out; the sealed test split is not read.
Run on the server:  ./run python scripts/probe_native_episode.py native_20260928
"""
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_sequence_length_stats, get_ctx_sequence_length, get_argn_name, SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.argn import SequentialModel
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _resolve_gen_column_order
from mostlyai.engine._tabular.encoding import encode_df, flatten_frame, _enrich_slen_sidx_sdec
from mostlyai.engine._tabular.training import BatchCollator

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
CONTROL = WS / "research-argn-fraud-audit/artifacts/sparkov_argn_control_v2"
OUT = ROOT / "docs/native_argn_episode_probe_v1"
MERCHANT, AMOUNT, LABEL = "receiver_or_mark", "amount_or_numeric_value", "event_is_fraud"
CORE = ["gap", MERCHANT, AMOUNT, "category", LABEL]
N_CUSTOMERS = 147
# 2026-11-08: the same probe on the IBM/TabFormer 2013 model (ibm_tabformer_v1); selected with --dataset ibm.
DATASETS = {"sparkov": dict(control=CONTROL, out=OUT, core=CORE, n=147),
            "ibm": dict(control=ROOT / "artifacts/ibm_tabformer_v1", out=ROOT / "docs/ibm_tabformer_v1/probe",
                        core=["gap", MERCHANT, AMOUNT, "category", "use_chip", LABEL], n=240)}
PREFIX = 512  # overridden by --prefix
BANDS = [(0, 0, "prev_normal"), (1, 1, "prev_fraud_age_1"), (2, 5, "prev_fraud_age_2_5"), (6, 10, "prev_fraud_age_6_10"),
         (11, 20, "prev_fraud_age_11_20"), (21, 10**9, "prev_fraud_age_21_plus")]


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed(n):
    torch.manual_seed(n); np.random.seed(n)


def load_model(folder):
    ws = Workspace(folder / "workspace"); ts = ws.tgt_stats.read(); cs = ws.ctx_stats.read(); cfg = ws.model_tabular_configs.read()
    cardinalities = get_cardinalities(ts); lengths = get_sequence_length_stats(ts)
    order = _resolve_gen_column_order(column_stats=ts["columns"], cardinalities=cardinalities, rebalancing=None, imputation=None, sample_seed=None, fairness=None)
    model = SequentialModel(tgt_cardinalities=cardinalities, tgt_seq_len_median=lengths["median"], tgt_seq_len_max=lengths["max"],
                            ctx_cardinalities=get_cardinalities(cs), ctxseq_len_median=get_ctx_sequence_length(cs, key="median"),
                            model_size=cfg["model_units"], column_order=order, device=torch.device("cpu"))
    load_model_weights(model=model, path=ws.model_tabular_weights_path, device=torch.device("cpu")); model.eval()
    columns = {c: get_argn_name(st["argn_processor"], st["argn_table"], st["argn_column"]) for c, st in ts["columns"].items()}
    # Digit-encoded numeric columns have several sub-columns; only the label head is read here.
    names = {c: [k for k in cardinalities if k.startswith(prefix + "__")] for c, prefix in columns.items()}
    assert len(names[LABEL]) == 1, names[LABEL]
    names = {c: v[0] for c, v in names.items() if len(v) == 1}
    return model, ws, ts, cs, lengths, order, names


def state_features(raw):
    """Previous-event state for every position, computed from the sequence itself (strict past)."""
    d = raw.sort_values(["customer_id", "event_index"], kind="stable").reset_index(drop=True)
    d = d[d.event_index.lt(PREFIX)]
    f = pd.to_numeric(d[LABEL], errors="coerce").fillna(0).astype(int).eq(1)
    boundary = d.customer_id.ne(d.customer_id.shift()) | f.ne(f.shift())
    run_age = d.groupby(boundary.cumsum(), sort=False).cumcount() + 1
    age = run_age.where(f, 0)
    prev_age = age.groupby(d.customer_id, sort=False).shift().fillna(-1).astype(int)  # -1 = first event
    d["prev_age"] = prev_age.to_numpy(); d["fraud"] = f.astype(int).to_numpy()
    return d


def encode_sequences(raw, ts, cs, lengths):
    raw = raw.rename(columns={"entity_id": "customer_id"}).copy(); raw["gap"] = raw.gap.fillna(0)
    encoded, _, key = encode_df(raw[["customer_id", *CORE]], ts, tgt_context_key="customer_id", n_jobs=1)
    encoded = flatten_frame(_enrich_slen_sidx_sdec(encoded, key, lengths["max"]), key)
    for col in encoded:
        if col.startswith("tgt:"):
            encoded[col] = encoded[col].map(lambda x: x[:PREFIX])
    ctx = pd.read_parquet(CONTROL / "prepared/validation_context.parquet")
    context, pk, _ = encode_df(ctx, cs, ctx_primary_key="customer_id", n_jobs=1)
    frame = encoded.merge(context, left_on=key, right_on=pk, validate="one_to_one")
    ids = frame[key].to_numpy(); frame = frame.drop(columns=[key, pk])
    return frame, ids, raw


@torch.no_grad()
def probe_source(model, ts, names, frame, ids, feats, source):
    collate = BatchCollator(True, None, torch.device("cpu"))
    fraud_code = ts["columns"][LABEL]["codes"]["1"]
    totals = defaultdict(lambda: [0, 0.0, 0, 0.0])  # events, sum P(fraud), observed fraud, sum nll
    by_customer = {cid: g for cid, g in feats.groupby("customer_id", sort=False)}
    for start in range(0, len(frame), 3):
        rows = frame.iloc[start:start + 3]; batch = collate(rows.to_dict("records"))
        output, _ = model(batch, mode="trn")
        y = batch[names[LABEL]].squeeze(-1); logits = output[names[LABEL]]
        mask = torch.zeros_like(y, dtype=torch.bool)
        for key in batch:
            if key.startswith(SLEN_SUB_COLUMN_PREFIX):
                mask |= batch[key].squeeze(-1).ne(0)
        p_fraud = logits.softmax(-1)[..., fraud_code]
        nll = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").reshape_as(y)
        prev = torch.full_like(y, -2)
        for i, cid in enumerate(ids[start:start + 3]):
            g = by_customer[int(cid)]; n = min(len(g), mask.shape[1])
            assert int(mask[i].sum()) == n, (cid, int(mask[i].sum()), n)
            prev[i, :n] = torch.as_tensor(g.prev_age.to_numpy()[:n])
            assert torch.equal(y[i, :n].eq(fraud_code), torch.as_tensor(g.fraud.to_numpy()[:n]).eq(1)), cid
        position = torch.arange(mask.shape[1])[None, :].expand_as(mask)
        for lo, hi, name in BANDS + [(-1, -1, "first_event"), (1, 1, "prev_fraud_age_1_pos_lt_512"), (1, 1, "prev_fraud_age_1_pos_ge_512"), (0, 0, "prev_normal_pos_ge_512")]:
            use = mask & prev.ge(lo) & prev.le(hi)
            if name.endswith("pos_lt_512"): use &= position.lt(512)
            if name.endswith("pos_ge_512"): use &= position.ge(512)
            a = totals[name]; a[0] += int(use.sum()); a[1] += float(p_fraud[use].double().sum())
            a[2] += int(y[use].eq(fraud_code).sum()); a[3] += float(nll[use].double().sum())
    rows = [dict(source=source, group=k, events=v[0], mean_fraud_probability=v[1] / v[0] if v[0] else None,
                 observed_next_fraud_rate=v[2] / v[0] if v[0] else None, label_nll=v[3] / v[0] if v[0] else None)
            for k, v in totals.items()]
    return rows


def onset_fields(feats, source):
    """Plain data comparison input: fields of fraud events by run age, for real vs generated."""
    d = feats.copy(); d["age"] = d.groupby("customer_id", sort=False).fraud.transform(lambda x: x.groupby((x.ne(x.shift())).cumsum()).cumcount() + 1).where(d.fraud.eq(1), 0)
    d["seen_merchant"] = d.duplicated(["customer_id", MERCHANT]).astype(int)
    out = []
    for lo, hi, name in [(1, 1, "onset_age_1"), (2, 5, "age_2_5"), (6, 10**9, "age_6_plus")]:
        s = d[d.fraud.eq(1) & d.age.between(lo, hi)]
        out.append(dict(source=source, group=name, events=len(s), customers=s.customer_id.nunique(),
                        amount_log_mean=float(np.log1p(pd.to_numeric(s[AMOUNT], errors="coerce")).mean()) if len(s) else None,
                        gap_log_mean=float(np.log1p(pd.to_numeric(s.gap, errors="coerce").clip(lower=0)).mean()) if len(s) else None,
                        seen_merchant_rate=float(s.seen_merchant.mean()) if len(s) else None,
                        top_category=s.category.mode().iloc[0] if len(s) else None,
                        top_category_share=float(s.category.value_counts(normalize=True).iloc[0]) if len(s) else None,
                        online_share=float(s.use_chip.astype(str).eq("Online Transaction").mean()) if len(s) and "use_chip" in s else None))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--draws", nargs="*", default=["20261011", "20261012"]); ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--dataset", choices=list(DATASETS), default="sparkov")
    args = ap.parse_args(); seed(20261107); t0 = time.time()
    global PREFIX, CONTROL, OUT, CORE, N_CUSTOMERS
    ds = DATASETS[args.dataset]; CONTROL, OUT, CORE, N_CUSTOMERS = ds["control"], ds["out"], ds["core"], ds["n"]
    OUT.mkdir(parents=True, exist_ok=True)
    PREFIX = args.prefix; tag = f"{args.run}_prefix{PREFIX}"
    folder = CONTROL / "runs" / args.run; assert (folder / "FIT.json").exists(), folder
    model, ws, ts, cs, lengths, order, names = load_model(folder); before = digest(ws.model_tabular_weights_path)
    sources = {"real_development": CONTROL / "prepared/validation.parquet"}
    for gs in args.draws:
        sources[f"self_generated_{gs}"] = folder / f"generated_validation_{gs}.parquet"
    rows, fields, hashes = [], [], {}
    for source, path in sources.items():
        raw = pd.read_parquet(path); hashes[str(path)] = digest(path)
        frame, ids, raw = encode_sequences(raw, ts, cs, lengths); feats = state_features(raw)
        assert len(frame) == N_CUSTOMERS == feats.customer_id.nunique()
        rows += probe_source(model, ts, names, frame, ids, feats, source); fields += onset_fields(feats, source)
        print(source, "done", round(time.time() - t0), "s", flush=True)
    assert digest(ws.model_tabular_weights_path) == before
    pd.DataFrame(rows).to_csv(OUT / f"{tag}_state_probe.csv", index=False)
    pd.DataFrame(fields).to_csv(OUT / f"{tag}_fraud_fields_by_age.csv", index=False)
    json.dump(dict(run=args.run, dataset=args.dataset, weights_sha256=before, weights_unchanged=True, prefix_limit=PREFIX, customers=N_CUSTOMERS,
                   column_order=list(order), label_predicted_after=[c for c in order if c != names[LABEL]],
                   inputs=hashes, mode="teacher forward only; no rollout; no new fit; test not read",
                   seconds=round(time.time() - t0, 1), script_sha256=digest(__file__)),
              open(OUT / f"{tag}_MANIFEST.json", "w"), indent=1)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
