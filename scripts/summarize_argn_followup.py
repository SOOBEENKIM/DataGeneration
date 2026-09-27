"""Descriptive D1 summaries; development diagnosis, not hypothesis confirmation."""
import json
import numpy as np
import pandas as pd
from run_argn_state_first import ROOT, digest, write
from diagnose_argn_followup_data import DEST,DOCS,ARMS,SEEDS


def summarize(frame,arm,fs,regime,case):
    groups=[('all','all',frame)]
    groups += [('transition',str(k),g) for k,g in frame.groupby('transition')]
    groups += [('cohort',str(k),g) for k,g in frame.groupby('cohort')]
    groups += [('previous_label',str(k),g) for k,g in frame.groupby('previous_label')]
    groups += [('cohort_previous_label','/'.join(map(str,k)),g) for k,g in frame.groupby(['cohort','previous_label'])]
    groups += [('cohort_transition','/'.join(k),g) for k,g in frame.groupby(['cohort','transition'])]
    f=frame[frame.previous_label==1].copy()
    f['age_bin']=pd.cut(f.prior_run_age,[0,1,3,6,10,15,np.inf],labels=['1','2-3','4-6','7-10','11-15','16+'])
    groups += [('previous_fraud_run_age',str(k),g) for k,g in f.groupby('age_bin',observed=True)]
    rows=[]
    for grouping,name,g in groups:
        if not len(g): continue
        p=g.fraud_probability.to_numpy(dtype=float)
        y=g.label.to_numpy(dtype=float)
        r=dict(arm=arm,fit_seed=fs,regime=regime,case=case,grouping=grouping,group=name,
               prefixes=len(g),customers=g.customer_id.nunique(),actual_fraud_rate=y.mean(),
               mean_fraud_probability=p.mean(),brier_vs_observed=np.mean((p-y)**2))
        if regime=='teacher':
            r.update(raw_label_nll=g.label_nll.mean(),amount_digit_nll=g.amount_digit_nll.mean())
        else:
            # Pure Monte Carlo error, not a customer/generalization confidence interval.
            r['mean_probability_mc_se']=np.sqrt(np.square(g.mc_se).sum())/len(g)
        rows.append(r)
    return rows


def main():
    manifest=json.loads((DEST/'D1_INPUTS.json').read_text())
    rows=[]; complete=[]
    for arm in ARMS:
        for fs in SEEDS:
            run=DEST/f'{arm}_{fs}'
            info=json.loads((run/'COMPLETE.json').read_text()); complete.append(info)
            teacher=pd.read_parquet(run/'teacher.parquet')
            rows += summarize(teacher,arm,fs,'teacher','all_development')
            rows += summarize(teacher[teacher.selected],arm,fs,'teacher','matched_prefixes')
            step=pd.read_parquet(run/'one_step.parquet')
            for case,g in step.groupby('case'):
                rows += summarize(g,arm,fs,'one_step',case)
    result=pd.DataFrame(rows)
    result.to_csv(DOCS/'conditional_summary.csv',index=False)
    for path,h in manifest['weights'].items(): assert digest(path)==h
    d0=json.loads((DEST/'D0_COMPLETE.json').read_text())
    for path,h in d0['generation_hashes'].items(): assert digest(path)==h
    write(DEST/'COMPLETE.json',dict(models=complete,weights_unchanged=True,existing_generations_unchanged=True,
          test_events_read=False,diagnostic_only=True,summary_script_sha256=digest(__file__),
          caveats=['Teacher/one-step use true lengths; existing free rollout does not.',
                   'One-step normal stays are sampled per customer; overall selected-prefix prevalence is not a population score.',
                   'One-step raw_label_nll/amount_digit_nll in raw per-prefix files are inherited teacher references, not sampled-field losses.',
                   'MC standard error measures field sampling only; no independent confirmation inference.']))
    print(result[(result.grouping=='transition') & result.group.isin(['continuation','termination','onset']) &
                 result['case'].isin(['matched_prefixes','sample_all','fix_amount_or_numeric_value'])]
          [['arm','fit_seed','case','group','prefixes','customers','mean_fraud_probability']].to_string(index=False))


if __name__=='__main__': main()
