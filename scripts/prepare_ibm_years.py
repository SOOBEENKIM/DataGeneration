"""ibm_eval_frame_v2 (docs/ibm_eval_frame_v2/PROTOCOL.md): prepare and SEAL the confirmation years.
    ./run python scripts/prepare_ibm_years.py            # 2015 2016 2018 2019
Same task definition as the 2013 window (scripts/run_ibm_tabformer_argn.py prepare), except that the customer partition is
the 2013 one (prepared/roles.parquet) for every year, so a test customer is never a training customer in any year.
The test files are written and hashed, never read back; no fraud statistic of a test partition is computed here.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from run_ibm_tabformer_argn import RAW, CFG, CORE, STATIC, CATEGORICAL, MERCHANT, AMOUNT, LABEL, OUT as OUT13, digest, write, seed  # noqa: E402

YEARS = [2015, 2016, 2018, 2019]
OUT = ROOT / "artifacts/ibm_years_v2"; DOCS = ROOT / "docs/ibm_eval_frame_v2"


def prepare(year, raw, roles):
    from mostlyai.engine import split, analyze, encode
    from mostlyai.engine._workspace import Workspace
    base = OUT / str(year) / "prepared"; base.mkdir(parents=True, exist_ok=False)
    before = raw[raw.year.lt(year)]; win = raw[raw.year.eq(year)].copy()
    n_win = win.groupby("user").size(); enough = n_win[n_win.ge(CFG["min_events_in_window"])].index
    eligible = np.sort(np.intersect1d(enough.to_numpy(), roles.user.to_numpy()))            # 30+ events in the year AND a 2013 role
    pd.DataFrame(dict(user=np.setdiff1d(enough.to_numpy(), roles.user.to_numpy()))).assign(reason="no_2013_role").to_csv(base / "excluded_customers.csv", index=False)
    r = roles[roles.user.isin(eligible)].copy()                                             # keeps the 2013 customer_id
    win = win[win.user.isin(eligible)].sort_values(["user", "timestamp"], kind="stable")
    win["event_index"] = win.groupby("user", sort=False).cumcount(); win["gap"] = win.groupby("user", sort=False).timestamp.diff()
    ev = pd.DataFrame(dict(user=win.user.to_numpy(), event_index=win.event_index.to_numpy(), gap=win.gap.to_numpy()))
    ev[MERCHANT] = win.merchant.astype(str).to_numpy(); ev[AMOUNT] = win.amount.astype(float).to_numpy()
    ev["category"] = win.mcc.astype(int).astype(str).to_numpy(); ev["use_chip"] = win.use_chip.astype(str).to_numpy(); ev[LABEL] = win.is_fraud.astype(int).astype(str).to_numpy()
    ev = ev.merge(r, on="user", validate="many_to_one").sort_values(["customer_id", "event_index"], kind="stable")
    g = before[before.user.isin(eligible)].groupby("user")
    ctx = pd.DataFrame(dict(first_year=g.year.min(), cards_before=g.card.nunique(), events_before=g.size(),
                            home_state=g.state.agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "NONE"))).reindex(eligible)
    ctx["first_year"] = ctx.first_year.fillna(year).astype(int); ctx["cards_before"] = ctx.cards_before.fillna(0).astype(int)
    ctx["events_per_year_before"] = (ctx.events_before.fillna(0) / (year - ctx.first_year).clip(lower=1)).astype(float); ctx["home_state"] = ctx.home_state.fillna("NONE").astype(str)
    ctx = ctx.drop(columns=["events_before"]).reset_index().rename(columns={"index": "user"}).merge(r, on="user", validate="one_to_one")
    r.to_parquet(base / "roles.parquet", index=False); sizes = {}
    for role in ["train", "validation", "test"]:
        part = ev[ev.split.eq(role)]; sizes[role] = dict(customers=int(part.customer_id.nunique()), events=int(len(part)))
        part.drop(columns=["user", "split"]).rename(columns={"customer_id": "entity_id"})[["entity_id", "event_index", *CORE]].to_parquet(base / f"{role}.parquet", index=False)
        ctx[ctx.split.eq(role)][["customer_id", *STATIC]].to_parquet(base / f"{role}_context.parquet", index=False)
    del ev, win, before
    train = pd.read_parquet(base / "train.parquet"); val = pd.read_parquet(base / "validation.parquet"); parent = pd.read_parquet(base / "train_context.parquet")
    target = train.rename(columns={"entity_id": "customer_id"})[["customer_id", *CORE]].copy(); target.gap = target.gap.fillna(0)
    types = {c: "TABULAR_CATEGORICAL" if c in CATEGORICAL else "TABULAR_NUMERIC_AUTO" for c in CORE}; ctypes = {c: "TABULAR_CATEGORICAL" if c == "home_state" else "TABULAR_NUMERIC_AUTO" for c in STATIC}
    seed(CFG["preparation_seed"])
    split(tgt_data=parent, tgt_primary_key="customer_id", tgt_encoding_types=ctypes, workspace_dir=base / "parent"); analyze(workspace_dir=base / "parent"); encode(workspace_dir=base / "parent")
    split(tgt_data=target, ctx_data=parent, tgt_context_key="customer_id", ctx_primary_key="customer_id", tgt_encoding_types=types, ctx_encoding_types=ctypes, workspace_dir=base / "child")
    analyze(workspace_dir=base / "child"); encode(workspace_dir=base / "child")
    stats = Workspace(base / "child").tgt_stats.read(); tf = train[LABEL].eq("1"); vf = val[LABEL].eq("1")
    info = dict(year=year, protocol_sha256=digest(DOCS / "PROTOCOL.md"), roles_2013_sha256=digest(OUT13 / "prepared/roles.parquet"), eligible_customers=int(len(eligible)), excluded_no_2013_role=int(len(np.setdiff1d(enough.to_numpy(), roles.user.to_numpy()))),
                sizes=sizes, test_events_read=False, test_fraud_statistics_computed=False,
                train_frauds=int(tf.sum()), train_fraud_customers=int(train[tf].entity_id.nunique()), validation_frauds=int(vf.sum()), validation_fraud_customers=int(val[vf].entity_id.nunique()),
                target_columns={c: dict(encoding=stats["columns"][c]["encoding_type"], cardinality=len(stats["columns"][c].get("codes", []))) for c in CORE},
                files={p.name: digest(p) for p in sorted(base.glob("*.parquet"))})
    write(DOCS / f"preparation_{year}.json", info); print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=1), flush=True)


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or YEARS
    raw = pd.read_parquet(RAW, columns=["user", "card", "year", "timestamp", "amount", "use_chip", "merchant", "state", "mcc", "is_fraud"])
    roles = pd.read_parquet(OUT13 / "prepared/roles.parquet")
    for y in years:
        prepare(y, raw, roles)
