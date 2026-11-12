"""Post-hoc support-size follow-up; no fit or generation."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from diagnose import OUT,CFG,SOURCE,verify,write,digest,times,locate


def main():
    verify();assert not (OUT/'cohort1024.csv').exists()
    write(OUT/'COHORT1024_START.json',dict(source_sha256=digest(__file__),
        plan_sha256=digest(OUT/'COHORT_AMENDMENT.md'),utc=datetime.now(timezone.utc).isoformat()))
    real=times(SOURCE/'prepared/validation.parquet');lengths=real.groupby('entity_id').size();rows=[]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            samples={a:times(locate(a,fs,gs)[0]) for a in ['clock_joint','clock_prefix','clock_full_prefix']}
            lens=samples['clock_joint'].groupby('entity_id').size()
            common=pd.concat([lengths,lens],axis=1).dropna().min(axis=1)
            keep=common.index[common.ge(1024)]
            r=real[real.entity_id.isin(keep)&real.event_index.lt(1024)]
            for arm,d in samples.items():
                d=d[d.entity_id.isin(keep)&d.event_index.lt(1024)]
                for lo,hi in [(6,50),(50,200),(200,500),(500,1024)]:
                    a=r[r.label.eq(0)&r.clock.gt(0)&r.event_index.ge(lo)&r.event_index.lt(hi)]
                    b=d[d.label.eq(0)&d.clock.gt(0)&d.event_index.ge(lo)&d.event_index.lt(hi)]
                    both=np.intersect1d(a.entity_id.unique(),b.entity_id.unique());a=a[a.entity_id.isin(both)];b=b[b.entity_id.isin(both)]
                    wa=1/a.groupby('entity_id').entity_id.transform('size').to_numpy();wb=1/b.groupby('entity_id').entity_id.transform('size').to_numpy()
                    x,y=np.log1p(a.gap),np.log1p(b.gap)
                    ca,cb=np.log1p(a.clock),np.log1p(b.clock)
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=lo,customers=len(both),
                        event_gap_w1=wasserstein_distance(x,y),customer_gap_w1=wasserstein_distance(x,y,u_weights=wa,v_weights=wb),
                        log_gap_bias=float(np.average(y,weights=wb)-np.average(x,weights=wa)),
                        log_clock_bias=float(np.average(cb,weights=wb)-np.average(ca,weights=wa))))
    d=pd.DataFrame(rows);d.to_csv(OUT/'cohort1024.csv',index=False)
    m=d.groupby(['arm','start'])[['customers','event_gap_w1','customer_gap_w1','log_gap_bias','log_clock_bias']].mean()
    m.to_csv(OUT/'cohort1024_means.csv');print(m.round(5).to_string(),flush=True)
    verify();write(OUT/'COHORT1024_COMPLETE.json',dict(test_events_read=False,new_fits=0,new_generations=0))


if __name__=='__main__':main()
