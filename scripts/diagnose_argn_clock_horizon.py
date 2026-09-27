"""Post-hoc matched-position diagnostic across the calibration horizon."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from report_argn_clock_location import ROOT,OUT,DOCS,CFG,verify,digest,write,locate
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate,w1


def main():
    verify(); assert (DOCS/'COMPLETE.json').exists()
    real=decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet')); lengths=real.groupby('entity_id').size()
    rows=[];bounds=[6,50,200,500,1024,2048,np.inf]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            reference=pd.read_parquet(locate('clock_joint',fs,gs)[0]); synth_lengths=reference.groupby('entity_id').size()
            common=pd.concat([lengths.rename('real'),synth_lengths.rename('generated')],axis=1).dropna().min(axis=1)
            r=real[real.event_index.lt(real.entity_id.map(common))]
            for arm in ['clock_joint','history_only']+CFG['arms']:
                d=decorate(pd.read_parquet(locate(arm,fs,gs)[0]));d=d[d.event_index.lt(d.entity_id.map(common))]
                for low,high in zip(bounds[:-1],bounds[1:]):
                    a=r[r.event_index.ge(low)&r.event_index.lt(high)&r.label.eq(0)&r.past_gap.gt(0)]
                    b=d[d.event_index.ge(low)&d.event_index.lt(high)&d.label.eq(0)&d.past_gap.gt(0)]
                    if not len(a) or not len(b):continue
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=low,end=None if not np.isfinite(high) else int(high),
                        real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique(),
                        gap_w1=w1(a.gap,b.gap),personal_gap_w1=w1(a.gap/a.past_gap,b.gap/b.past_gap),clock_w1=w1(a.past_gap,b.past_gap),
                        real_gap_median=float(a.gap.median()),generated_gap_median=float(b.gap.median()),
                        real_clock_median=float(a.past_gap.median()),generated_clock_median=float(b.past_gap.median())))
    result=pd.DataFrame(rows);result.to_csv(DOCS/'horizon_diagnostic.csv',index=False)
    result.groupby(['arm','start'])[['real_events','generated_events','gap_w1','personal_gap_w1','clock_w1',
        'real_gap_median','generated_gap_median','real_clock_median','generated_clock_median']].mean().to_csv(DOCS/'horizon_means.csv')
    write(DOCS/'HORIZON_DIAGNOSTIC_COMPLETE.json',dict(posthoc_descriptive=True,new_fits=0,new_generations=0,test_events_read=False,
        source_sha256=digest(__file__),plan_sha256=digest(DOCS/'HORIZON_DIAGNOSTIC_PLAN.md'),completed_utc=datetime.now(timezone.utc).isoformat()))
    print(result.groupby(['arm','start'])[['gap_w1','personal_gap_w1','clock_w1']].mean().round(5).to_string())


if __name__=='__main__':main()
