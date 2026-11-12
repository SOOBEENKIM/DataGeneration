"""Label-only Monte Carlo under fixed, MODEL-SAMPLED native lengths.

Not additional neural fits or full transaction datasets. No validation truth
is supplied to the label generator, and no parameters are selected here.
"""
import json
import numpy as np
import pandas as pd
from run_argn_gap_episode import OUT,DOCS,CFG,OLD,foundation,verify,digest,write
from benchmarks.argn_transition_reference import reference_table
from benchmarks.argn_episode_control import first_probability


def simulate(lengths,parameters,old,mode,repeats=64,seed=20261201):
    rng=np.random.default_rng(seed);n=len(lengths)
    lengths=np.tile(np.asarray(lengths,int),repeats)
    previous=np.zeros(len(lengths),int);age=np.zeros(len(lengths),int)
    ever=np.zeros(len(lengths),int);initial=np.zeros(len(lengths),int)
    frauds=np.zeros(len(lengths),int);episodes=np.zeros(len(lengths),int)
    table=reference_table(old,'duration');ep=np.asarray(parameters['table'])
    initial_probs=first_probability(lengths,parameters['first_tree'])
    for t in range(lengths.max()):
        row=np.zeros_like(previous) if t==0 else previous+1
        if mode=='duration':p=table[row,np.minimum(age,21)]
        else:p=ep[row,np.minimum(age,21),ever,initial]
        if t==0 and mode=='joint':p=initial_probs
        label=(rng.random(len(lengths))<p).astype(int);active=t<lengths
        frauds+=label*active;episodes+=((label==1)&((previous==0)|(t==0))&active)
        age=np.where(label==previous,age+1,1) if t else np.ones_like(age)
        ever=np.maximum(ever,label)
        if t==0:initial=label.copy()
        previous=label
    f=frauds.reshape(repeats,n);e=episodes.reshape(repeats,n);lens=lengths.reshape(repeats,n)
    return pd.DataFrame(dict(repeat=np.arange(repeats),customers_with_fraud=(f>0).sum(1),
        all_fraud_customers=(f==lens).sum(1),repeated_episode_customers=(e>1).sum(1),
        fraud_events=f.sum(1),fraud_rate=f.sum(1)/lens.sum(1),total_episodes=e.sum(1),
        short_customers=(lens<=100).sum(1)))


def main():
    verify();parameters=json.loads((OUT/'episode_parameters.json').read_text());old=json.loads(OLD.read_text());rows=[]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            p=foundation(fs)/f'generated_validation_{gs}.parquet'
            lengths=pd.read_parquet(p,columns=['entity_id']).groupby('entity_id').size().to_numpy()
            for mode in ['duration','episode','joint']:
                d=simulate(lengths,parameters,old,mode)
                d['fit_seed']=fs;d['length_draw']=gs;d['mode']=mode;rows.append(d)
                print('LABEL_ONLY_MC',fs,gs,mode,d.customers_with_fraud.mean(),d.all_fraud_customers.mean(),flush=True)
    data=pd.concat(rows,ignore_index=True);data.to_csv(DOCS/'label_only_rollouts.csv',index=False)
    data.groupby('mode').agg(['mean','std']).to_csv(DOCS/'label_only_rollout_summary.csv')
    write(DOCS/'LABEL_ROLLOUT_COMPLETE.json',dict(label_only_draws=len(data),full_transaction_generations=0,
          source_sha256=digest(__file__),parameter_sha256=digest(OUT/'episode_parameters.json'),test_events_read=False))


if __name__=='__main__':main()
