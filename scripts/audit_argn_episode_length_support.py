"""Read-only support audit: generated lengths and initial-state inference."""
import json
import numpy as np
import pandas as pd
from run_argn_gap_episode import inputs,CFG,OUT,DOCS,foundation,write,digest,verify
from benchmarks.argn_episode_control import first_probability


def main():
    verify();_,_,_,metas=inputs();params=json.loads((OUT/'episode_parameters.json').read_text())
    train=metas['optimization'].groupby('record').agg(length=('label','size'),initial=('label','first'))
    fraud_max=int(train.loc[train.initial.eq(1),'length'].max());normal_min=int(train.loc[train.initial.eq(0),'length'].min())
    rows=[]
    for split,m in metas.items():
        c=m.groupby('record').agg(length=('label','size'),initial=('label','first'))
        rows.append(dict(source=split,customers=len(c),short_le100=int(c.length.le(100).sum()),
            length_le_train_fraud_max=int(c.length.le(fraud_max).sum()),
            between_train_initial_groups=int((c.length.gt(fraud_max)&c.length.lt(normal_min)).sum()),
            predicted_first_fraud_sum=first_probability(c.length.to_numpy(),params['first_tree']).sum(),
            actual_first_fraud=int(c.initial.sum())))
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            c=pd.read_parquet(foundation(fs)/f'generated_validation_{gs}.parquet',columns=['entity_id']).groupby('entity_id').size()
            rows.append(dict(source=f'generated_lengths_{fs}_{gs}',customers=len(c),short_le100=int(c.le(100).sum()),
                length_le_train_fraud_max=int(c.le(fraud_max).sum()),
                between_train_initial_groups=int((c.gt(fraud_max)&c.lt(normal_min)).sum()),
                predicted_first_fraud_sum=first_probability(c.to_numpy(),params['first_tree']).sum()))
    pd.DataFrame(rows).to_csv(DOCS/'length_support.csv',index=False)
    write(DOCS/'length_support_boundaries.json',dict(training_initial_fraud_max_length=fraud_max,
        training_initial_normal_min_length=normal_min,source_sha256=digest(__file__),
        note='post-registration read-only audit, not a new threshold imposed on generation'))
    print(pd.DataFrame(rows).to_string(index=False));print(fraud_max,normal_min)


if __name__=='__main__':main()
