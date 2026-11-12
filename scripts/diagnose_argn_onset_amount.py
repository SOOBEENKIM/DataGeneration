"""Separate onset-specific conditional learning from changed rollout composition."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_argn_joint_preservation import ROOT,DOCS,CFG,inputs,write,digest,verify


def w1(a,b):
    a=np.asarray(a);b=np.asarray(b);a=a[np.isfinite(a)&(a>=0)];b=b[np.isfinite(b)&(b>=0)]
    return wasserstein_distance(np.log1p(a),np.log1p(b)) if len(a) and len(b) else np.nan


def main():
    verify();_,codec,frames,metas=inputs();rows=[];paths=[]
    for split in ['optimization','development']:
        parts=[]
        for ri,r in enumerate(frames[split]):
            amount=codec.numeric({k:np.asarray(v).reshape(-1) for k,v in r.items() if k.startswith(codec.prefixes['amount_or_numeric_value']+'__')},'amount_or_numeric_value')
            prior=pd.Series(amount).shift().rolling(20,min_periods=5).median()
            parts.append(pd.DataFrame(dict(record=ri,event_index=np.arange(len(amount)),prior=prior)))
        meta=metas[split].copy()
        left=meta.groupby('record').label.cummin().eq(1)
        meta['phase']=np.where(meta.label.eq(0),'normal',np.where(left,'left_boundary',np.where(meta.previous_label.eq(0),'onset','continuation')))
        meta=meta.merge(pd.concat(parts,ignore_index=True),on=['record','event_index'])
        for fs in CFG['fit_seeds']:
            path=ROOT/f'artifacts/argn_amount_learning_v1/runs/balanced_shared_{fs}/conditional_{split}.parquet';paths.append(path)
            d=pd.read_parquet(path).merge(meta[['record','event_index','phase','prior']],on=['record','event_index'])
            for phase,g in d.groupby('phase'):
                valid=g[g.event_index.ge(5)&g.prior.gt(0)]
                rows.append(dict(split=split,fit_seed=fs,phase=phase,positions=len(g)//4,draw_rows=len(g),
                    real_median=g.real_amount.median(),generated_median=g.generated_amount.median(),
                    amount_log_w1=w1(g.real_amount,g.generated_amount),
                    relative_log_w1=w1(valid.real_amount/valid.prior,valid.generated_amount/valid.prior)))
    pd.DataFrame(rows).to_csv(DOCS/'oracle_amount_by_phase.csv',index=False)
    write(DOCS/'ORACLE_PHASE_COMPLETE.json',dict(source_sha256=digest(__file__),inputs={str(p):digest(p) for p in paths},
        new_training=0,test_events_read=False,scope='actual past and actual current label/category/gap'))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
