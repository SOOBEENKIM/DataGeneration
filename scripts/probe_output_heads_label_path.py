"""ibm_output_heads_v1 probe: how much the retrained use_chip head relies on the label input vs. the real row/history
context. Teacher-forced on the internal validation partition (100 customers); weights untouched.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/probe_output_heads_label_path.py
"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first")
from benchmarks.argn_output_heads import *
from benchmarks.argn_output_heads import _records
A="/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first/artifacts"
for fs in [20261108, 20261109]:
    ws = Workspace(f"{A}/ibm_tabformer_v1/runs/native_{fs}/workspace")
    ts = ws.tgt_stats.read(); card = get_cardinalities(ts); order = label_first_order(ts, card)
    model, ts, card = build_model(ws, "cuda", order)
    label_key = argn_column(ts, LABEL) + "__cat"; code1 = int(ts["columns"][LABEL]["codes"]["1"]); code0 = int(ts["columns"][LABEL]["codes"]["0"])
    uc = [k for k in card if k.startswith(argn_column(ts, "use_chip") + "__")][0]; online = int(ts["columns"]["use_chip"]["codes"]["Online Transaction"])
    model.load_state_dict({k: v.cuda() for k, v in torch.load(f"{A}/ibm_output_heads_v1/heads/balanced_{fs}/heads.pt").items()}, strict=False); model.eval()
    recs = _records(list(ws.encoded_data_val.fetch_all())[0])
    loader = DataLoader(recs, shuffle=False, batch_size=4, collate_fn=BatchCollator(True, None, torch.device("cuda")))
    res = {}
    with torch.no_grad():
        for variant in ["fraud_rows_label_real", "fraud_rows_label_set_0", "normal_rows_label_real", "normal_rows_label_set_1"]:
            ps = []
            for data in loader:
                d = dict(data); fraud = d[label_key].squeeze(-1).eq(code1); normal = d[label_key].squeeze(-1).eq(code0)
                if variant.endswith("set_0"): d[label_key] = torch.where(fraud.unsqueeze(-1), torch.full_like(d[label_key], code0), d[label_key])
                if variant.endswith("set_1"): d[label_key] = torch.where(normal.unsqueeze(-1), torch.full_like(d[label_key], code1), d[label_key])
                out, _ = model(d, mode="trn"); p = out[uc].softmax(-1)[..., online]
                ps.append(p[fraud if variant.startswith("fraud") else normal].cpu())
            res[variant] = round(float(torch.cat(ps).mean()), 3)
    print(fs, "balanced heads, mean P(online):", res)
