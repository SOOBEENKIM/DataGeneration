"""Read-only composition/conditional-output separation on permitted splits."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_argn_gap_episode import ROOT,inputs,CFG,OUT as PRIOR,foundation,verify,write,digest
from run_argn_state_first import SOURCE
from evaluate_sparkov_argn_control import extended
from benchmarks.argn_state_evaluation import episode_features

OUT=ROOT/'artifacts/argn_joint_preservation_v1';DOCS=ROOT/'docs/argn_joint_preservation_v1'


def w1(a,b,weights=None):
    a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&(a>=0);other=np.isfinite(b)&(b>=0)
    if not ok.any() or not other.any():return np.nan
    return wasserstein_distance(np.log1p(a[ok]),np.log1p(b[other]),v_weights=None if weights is None else np.asarray(weights)[other])


def augment(raw,state):
    d,runs=episode_features(extended(raw,state))
    d['prior_median']=d.amount_or_numeric_value/d.amount_history_ratio_raw
    d['phase']=np.where(d.left_boundary_run,'left_boundary',np.where(d.run_age.eq(1),'onset','continuation'))
    return d,runs


def main():
    verify();ws,codec,frames,metas=inputs();rows=[];hazards=[]
    for split,m in metas.items():
        m=m.copy();m['length']=m.groupby('record').label.transform('size')
        m['ever']=m.groupby('record').label.cumsum().sub(m.label).gt(0)
        m['position_decile']=np.minimum(9,(10*m.event_index/m.length).astype(int))
        m['age_band']=np.searchsorted([20,100,250,500,750,1000,1500,2000,3000],m.prior_age,side='left')
        for key in ['position_decile','age_band']:
            for band,g in m[m.previous_label.eq(0)&~m.ever].groupby(key):
                hazards.append(dict(split=split,conditioning=key,band=int(band),events=len(g),customers=g.record.nunique(),
                    onsets=int(g.label.sum()),rate=g.label.mean()))
    pd.DataFrame(hazards).to_csv(DOCS/'onset_exposures.csv',index=False)
    state=json.loads((SOURCE/'prepared/metric_state.json').read_text())
    real,_=augment(pd.read_parquet(SOURCE/'prepared/validation.parquet'),state)
    paths={}
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            paths[f'old_{fs}_{gs}']=foundation(fs)/f'generated_validation_{gs}.parquet'
            paths[f'current_{fs}_{gs}']=PRIOR/f'runs/gap_and_transition_{fs}/generated_validation_{gs}.parquet'
    cells=[];standards=[]
    for name,path in paths.items():
        syn,_=augment(pd.read_parquet(path),state)
        rf=real[real.fraud.eq(1)&real.event_index.ge(5)];sf=syn[syn.fraud.eq(1)&syn.event_index.ge(5)]
        for phase in ['all','left_boundary','onset','continuation']:
            r=rf if phase=='all' else rf[rf.phase.eq(phase)];s=sf if phase=='all' else sf[sf.phase.eq(phase)]
            rows.append(dict(run=name,phase=phase,real_events=len(r),generated_events=len(s),
                amount_w1=w1(r.amount_or_numeric_value,s.amount_or_numeric_value),ratio_w1=w1(r.amount_history_ratio_raw,s.amount_history_ratio_raw),
                history_level_w1=w1(r.prior_median,s.prior_median),real_history_median=r.prior_median.median(),synthetic_history_median=s.prior_median.median(),
                real_ratio_median=r.amount_history_ratio_raw.median(),synthetic_ratio_median=s.amount_history_ratio_raw.median()))
        for cols in [['phase'],['phase','age_band'],['phase','age_band','category']]:
            weight=pd.Series(0.,index=sf.index);covered=0.;within=0.
            sgroups={k:g for k,g in sf.groupby(cols,observed=True)}
            for key,r in rf.groupby(cols,observed=True):
                s=sgroups.get(key)
                if s is None or not len(s):continue
                share=len(r)/len(rf);weight.loc[s.index]=share/len(s);covered+=share
                error=w1(r.amount_history_ratio_raw,s.amount_history_ratio_raw);within+=share*error
                cells.append(dict(run=name,conditioning='+'.join(cols),cell=str(key),real_events=len(r),generated_events=len(s),ratio_w1=error))
            standards.append(dict(run=name,conditioning='+'.join(cols),coverage=covered,
                standardized_ratio_w1=w1(rf.amount_history_ratio_raw,sf.amount_history_ratio_raw,weight),
                real_weighted_within_cell_ratio_w1=within/covered))
    pd.DataFrame(rows).to_csv(DOCS/'composition.csv',index=False)
    pd.DataFrame(cells).to_csv(DOCS/'conditional_cells.csv',index=False)
    pd.DataFrame(standards).to_csv(DOCS/'standardization.csv',index=False)
    oracle=[]
    for split in ['optimization','development']:
        parts=[]
        for ri,record in enumerate(frames[split]):
            tokens={k:np.asarray(v).reshape(-1) for k,v in record.items() if k.startswith(codec.prefixes['amount_or_numeric_value']+'__')}
            amount=codec.numeric(tokens,'amount_or_numeric_value')
            prior=pd.Series(amount).shift().rolling(20,min_periods=5).median()
            parts.append(pd.DataFrame(dict(record=ri,event_index=np.arange(len(amount)),prior=prior)))
        past=pd.concat(parts,ignore_index=True)
        for fs in CFG['fit_seeds']:
            path=ROOT/f'artifacts/argn_amount_learning_v1/runs/balanced_shared_{fs}/conditional_{split}.parquet'
            d=pd.read_parquet(path).merge(past,on=['record','event_index'])
            for label in [0,1]:
                g=d[d.label.eq(label)&d.event_index.ge(5)&d.prior.gt(0)]
                oracle.append(dict(split=split,fit_seed=fs,label=label,draw_rows=len(g),
                    amount_w1=w1(g.real_amount,g.generated_amount),
                    relative_w1=w1(g.real_amount/g.prior,g.generated_amount/g.prior)))
    pd.DataFrame(oracle).to_csv(DOCS/'oracle_relative_amount.csv',index=False)
    write(DOCS/'DIAGNOSIS_COMPLETE.json',dict(source_sha256=digest(__file__),previous_manifest_sha256=digest(PRIOR/'MANIFEST.json'),
        inspected_generations={k:digest(v) for k,v in paths.items()},test_events_read=False,neural_training=0))
    print(pd.DataFrame(rows).groupby([pd.DataFrame(rows).run.str.split('_').str[0],'phase']).mean(numeric_only=True).to_string())
    print(pd.DataFrame(standards).groupby([pd.DataFrame(standards).run.str.split('_').str[0],'conditioning']).mean(numeric_only=True).to_string())
    print(pd.DataFrame(oracle).to_string(index=False))


if __name__=='__main__':main()
