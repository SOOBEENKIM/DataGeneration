import json, glob
from pathlib import Path
import pandas as pd
rows=[]
for d in sorted(glob.glob("artifacts/ibm_baselines_v1/runs/*")):
    n=Path(d).name; fit=json.load(open(d+"/FIT.json")) if Path(d+"/FIT.json").exists() else {}
    st=json.load(open(d+"/START.json")) if Path(d+"/START.json").exists() else {}
    for g in sorted(glob.glob(d+"/generation_validation_*.json")):
        j=json.load(open(g)); rows.append(dict(run=n, generation_seed=Path(g).stem[-8:], events=j["events"], frauds=j["frauds"], fraud_customers=j["fraud_customers"], generation_seconds=round(j["seconds"],1), generation_sha256=j["sha256"],
            fit_seconds=round(fit.get("seconds",float("nan")),1), epochs_run=fit.get("epochs_run"), best_checkpoint_epoch=fit.get("best_checkpoint_epoch"), best_val_loss=fit.get("best_val_loss"), trainable_parameters=fit.get("trainable_parameters"), weights_sha256=fit.get("weights_sha256") or j.get("base_weights_sha256"), data=json.dumps(st.get("data",{}))))
pd.DataFrame(rows).to_csv("docs/ibm_baselines_v1/FITS_AND_GENERATIONS.csv", index=False); print(len(rows))
