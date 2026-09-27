"""Compare observed customer onset rates and implied accumulation by length."""
import json
import numpy as np
import pandas as pd
from run_argn_joint_preservation import inputs,DOCS,OUT,PRIOR
from benchmarks.argn_episode_control import episode_probabilities
from benchmarks.argn_joint_preservation import horizon_probability


def main():
    _,_,_,metas=inputs()
    episode=json.loads((PRIOR/'episode_parameters.json').read_text())
    hazard=json.loads((OUT/'hazard_parameters.json').read_text());rows=[]
    for split,m in metas.items():
        c=m.groupby('record').agg(length=('label','size'),first=('label','first'),frauds=('label','sum'))
        normal=c[c['first'].eq(0)]
        for label,g in normal.groupby(pd.cut(normal.length,[0,750,1250,1750,2250,np.inf]),observed=True):
            rows.append(dict(split=split,length_band=str(label),customers=len(g),customers_with_onset=int(g.frauds.gt(0).sum()),
                observed_customer_onset_rate=g.frauds.gt(0).mean(),mean_length=g.length.mean()))
    pd.DataFrame(rows).to_csv(DOCS/'observed_customer_onset_by_length.csv',index=False)
    curves=[]
    for n in [500,1000,1500,2000,2500,3000]:
        m=pd.DataFrame(dict(record=np.zeros(n,dtype=int),event_index=np.arange(n),label=np.zeros(n,dtype=int),
            previous_label=np.r_[-1,np.zeros(n-1,dtype=int)],prior_age=np.arange(n)))
        current=episode_probabilities(m,episode,True)[1:]
        new=horizon_probability(np.arange(1,n),np.full(n-1,n),hazard['rates'])
        curves.append(dict(length=n,current_onset_probability_given_normal_first=1-np.exp(np.log1p(-current).sum()),
            horizon_onset_probability_given_normal_first=1-np.exp(np.log1p(-new).sum())))
    pd.DataFrame(curves).to_csv(DOCS/'implied_customer_onset_by_length.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(curves).to_string(index=False))


if __name__=='__main__':main()
