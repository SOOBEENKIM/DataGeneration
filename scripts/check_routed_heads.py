"""ibm_routed_heads_v1 routing check (run before training): with untrained fraud copies the routed model must
reproduce the plain label-first model exactly; after perturbing the fraud copies only fraud rows may change.
    CUDA_VISIBLE_DEVICES=0 ./run python scripts/check_routed_heads.py --fit 20261108
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities
from mostlyai.engine._tabular.training import BatchCollator
from benchmarks.argn_output_heads import build_model, label_first_order, argn_column, LABEL, _records
from benchmarks.argn_routed_heads import routed_model, fraud_head_names, _valid_mask

ap = argparse.ArgumentParser(); ap.add_argument("--fit", type=int, default=20261108); a = ap.parse_args()
ws = Workspace(ROOT / "artifacts/ibm_tabformer_v1/runs" / f"native_{a.fit}" / "workspace")
ts = ws.tgt_stats.read(); card = get_cardinalities(ts); order = label_first_order(ts, card)
plain, _, _ = build_model(ws, "cuda", order); plain.eval()
routed, _, _, _ = routed_model(ws, "cuda"); routed.eval()
label_key = argn_column(ts, LABEL) + "__cat"; c1 = int(ts["columns"][LABEL]["codes"]["1"])
val = _records(list(ws.encoded_data_val.fetch_all())[0])
loader = DataLoader(val, shuffle=False, batch_size=4, collate_fn=BatchCollator(True, None, torch.device("cuda")))
batch = next(iter(loader))
with torch.no_grad():
    o1, _ = plain(batch, mode="trn"); o2, _ = routed(batch, mode="trn")
    identical = {k: float((o1[k] - o2[k]).abs().max()) for k in o1}
    print("max |plain - routed| with untrained copies:", max(identical.values()))
    assert max(identical.values()) < 1e-5, identical
    for n, p in routed.named_parameters():
        if n.startswith("fraud_predictors.") and n.endswith(".bias"):
            p.add_(1.0)
    o3, _ = routed(batch, mode="trn")
    fraud = batch[label_key].squeeze(-1).eq(c1) & _valid_mask(batch); normal = _valid_mask(batch) & ~fraud
    report = {}
    for k in o1:
        d = (o1[k] - o3[k]).abs().amax(-1)
        report[k] = dict(routed=k in routed.routed_sub_columns, normal_max=float(d[normal].max()), fraud_max=float(d[fraud].max()) if fraud.any() else None)
    print(json.dumps(report, indent=1))
    assert all(v["normal_max"] < 1e-5 for v in report.values()), "normal rows changed"
    assert all((v["fraud_max"] or 0) > 0.5 for v in report.values() if v["routed"]) and fraud.any(), "fraud rows not routed"
    assert all((v["fraud_max"] or 0) < 1e-5 for v in report.values() if not v["routed"]), "unrouted column changed"
print("fraud head parameters:", sum(p.numel() for n, p in routed.named_parameters() if n in set(fraud_head_names(routed))))
print("ROUTING CHECK PASSED: fraud rows in batch =", int(fraud.sum()))
