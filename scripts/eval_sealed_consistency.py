"""ibm_sealed_test_v1 S4: merchant-attribute consistency of the sealed-test generations (same measure as ibm_merchant_attr_v1 §1).
    ./run python scripts/eval_sealed_consistency.py
CPU only, no generation. Reads the frozen test generations and (as the reference row) the sealed test events, which this
registered one-shot evaluation already opened. Run after the pipeline because the pipeline script omitted this S4 item.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd
from run_ibm_merchant_attr import NATIVE, GEN_SEEDS, consistency, merchant_attributes, write  # noqa: E402

ART = ROOT / "artifacts"
ARGN = ["argn_native", "argn_minimal", "argn_merchant", "argn_gap", "argn_final"]
att = merchant_attributes(); rows = {"real_test": consistency(pd.read_parquet(NATIVE / "prepared/test.parquet"), att)}
for arm in ARGN + ["rtf_native", "rtf_table"]:
    base = ART / ("ibm_sealed_test_v1" if arm in ARGN else "ibm_realtabformer_v1") / "runs"
    per = []
    for fs in (20261108, 20261109):
        for gs in GEN_SEEDS:
            c = consistency(pd.read_parquet(base / f"{arm}_{fs}" / f"generated_test_{gs}.parquet"), att); c.update(fit=fs, seed=gs); per.append(c)
    mean = {nm: {k: float(sum(p[nm][k] for p in per) / len(per)) for k in per[0][nm]} for nm in ("normal", "fraud")}
    rows[arm] = dict(mean=mean, generations=per)
    print(arm, json.dumps({nm: {k: round(v, 4) for k, v in mean[nm].items()} for nm in mean}), flush=True)
print("real_test", json.dumps(rows["real_test"]))
write(ROOT / "docs/ibm_sealed_test_v1/CONSISTENCY.json", rows)
