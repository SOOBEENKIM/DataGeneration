"""Post-generation composition diagnostics; never changes generation or fit selection."""
import json
import numpy as np
import pandas as pd
from diagnose_argn_joint_preservation import augment,w1
from run_argn_joint_preservation import ROOT,OUT,DOCS,CFG,foundation,verify,write,digest
from run_argn_state_first import SOURCE


def main():
    verify();state=json.loads((SOURCE/'prepared/metric_state.json').read_text())
    real,_=augment(pd.read_parquet(SOURCE/'prepared/validation.parquet'),state)
    rf=real[real.fraud.eq(1)&real.event_index.ge(5)];rows=[];std=[];identity=[]
    for fs in CFG['fit_seeds']:
        for arm in ['current']+list(CFG['generation_arms']):
            run=foundation(fs) if arm=='current' else OUT/f'runs/{arm}_{fs}'
            for gs in CFG['generation_seeds']:
                path=run/f'generated_validation_{gs}.parquet';raw=pd.read_parquet(path);syn,runs=augment(raw,state)
                sf=syn[syn.fraud.eq(1)&syn.event_index.ge(5)]
                identity.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,events=len(syn),fraud_events=int(syn.fraud.eq(1).sum()),
                    runs=len(runs),mean_run_length=runs.length.mean(),fraud_rate=syn.fraud.eq(1).mean(),
                    first_fraud_customers=int(syn.event_index.eq(0).mul(syn.fraud.eq(1)).sum()),
                    first_fraud_mean_length=syn[syn.entity_id.isin(syn.loc[syn.event_index.eq(0)&syn.fraud.eq(1),'entity_id'])].groupby('entity_id').size().mean()))
                assert int(runs.length.sum())==int(syn.fraud.eq(1).sum())
                for phase in ['all','left_boundary','onset','continuation']:
                    r=rf if phase=='all' else rf[rf.phase.eq(phase)];s=sf if phase=='all' else sf[sf.phase.eq(phase)]
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,phase=phase,real_events=len(r),generated_events=len(s),
                        ratio_w1=w1(r.amount_history_ratio_raw,s.amount_history_ratio_raw),
                        amount_w1=w1(r.amount_or_numeric_value,s.amount_or_numeric_value),
                        history_w1=w1(r.prior_median,s.prior_median)))
                for cols in [['phase'],['phase','age_band'],['phase','age_band','category']]:
                    weight=pd.Series(0.,index=sf.index);covered=0.;within=0.
                    sgroups={k:g for k,g in sf.groupby(cols,observed=True)}
                    for k,r in rf.groupby(cols,observed=True):
                        s=sgroups.get(k)
                        if s is None or not len(s):continue
                        share=len(r)/len(rf);weight.loc[s.index]=share/len(s);covered+=share
                        within+=share*w1(r.amount_history_ratio_raw,s.amount_history_ratio_raw)
                    std.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,conditioning='+'.join(cols),coverage=covered,
                        standardized_ratio_w1=w1(rf.amount_history_ratio_raw,sf.amount_history_ratio_raw,weight),
                        real_weighted_within_cell_ratio_w1=within/covered))
    pd.DataFrame(rows).to_csv(DOCS/'rollout_phase_diagnostics.csv',index=False)
    pd.DataFrame(std).to_csv(DOCS/'rollout_standardization.csv',index=False)
    pd.DataFrame(identity).to_csv(DOCS/'fraud_rate_decomposition.csv',index=False)
    write(DOCS/'POST_DIAGNOSIS_COMPLETE.json',dict(source_sha256=digest(__file__),generations=len(identity),test_events_read=False,
        purpose='post-generation explanation only; no checkpoint or probability selection'))
    print(pd.DataFrame(identity).groupby('arm').mean(numeric_only=True).to_string())
    print(pd.DataFrame(rows).groupby(['arm','phase']).mean(numeric_only=True).to_string())


if __name__=='__main__':main()
