"""Read-only train-mean repetition prediction sanity control.

Input ordering and provenance checks live in verify_cs_saf_external_artifacts.py.
This constant prediction is not a sequence generator or an additional neural fit.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import OUT,write


def main():
    report=[]
    for k in [0,1]:
        inp=OUT/f'kappa_{k}/input'
        train=pd.read_parquet(inp/'train_events.parquet')
        parent=pd.read_parquet(inp/'train_context.parquet')
        vp=pd.read_parquet(inp/'validation_context.parquet')
        mask=train.groupby('entity_id').cumcount()>0
        actual=train.receiver_or_mark.eq(train.groupby('entity_id').receiver_or_mark.shift()).fillna(False)
        for folder in sorted((OUT/f'kappa_{k}').glob('seed_*')):
            file=folder/'replay_validation.parquet'
            if not file.exists():continue
            validation=pd.read_parquet(file)
            for group in ['pooled','context_0','context_1']:
                def ids(df):return set(df.entity_id if group=='pooled' else df.loc[df.entity_label.astype(str)==group[-1],'entity_id'])
                rate=float(actual[mask & train.entity_id.isin(ids(parent))].mean())
                v=validation[(validation.event_index>0)&validation.entity_id.isin(ids(vp))]
                y=v.actual_repeat.to_numpy();p=v.predicted_repeat.to_numpy()
                report.append(dict(kappa=k,seed=int(folder.name.split('_')[1]),group=group,
                    train_repeat_mean=rate,validation_repeat_mean=float(y.mean()),model_repeat_mean=float(p.mean()),
                    model_repeat_brier=float(np.mean((y-p)**2)),train_mean_control_brier=float(np.mean((y-rate)**2)),
                    model_observed_mark_nll=float(-np.log(v.observed_mark_probability.clip(lower=1e-12)).mean()),
                    uniform_64_mark_nll=float(np.log(64)),
                    role='posthoc_prediction_sanity_control_not_a_sequence_generator_or_new_fit'))
    write(ROOT/'docs/cs_saf/external_audit_v1/repeat_prediction_sanity.json',report)
    print(pd.DataFrame(report).to_string(index=False))


if __name__=='__main__':main()
