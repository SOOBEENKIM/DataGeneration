"""Train-only phase log-mixtures as a no-personal-history conditional control."""
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from diagnose_argn_residual import ROOT,OUT as SOURCE,verify,write,digest,w1,SAMPLES
from run_argn_amount_learning import Workspace,BASE

OUT=ROOT/'artifacts/argn_boundary_gap_v1/simple_control';DOCS=ROOT/'docs/argn_boundary_gap_v1/simple_control'


def main():
    verify();assert not (OUT/'COMPLETE.json').exists();OUT.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    ds={s:pd.read_parquet(SOURCE/f'metadata_{s}.parquet') for s in ['optimization','internal_validation','development']}
    stats=Workspace(BASE/'prepared/workspace').tgt_stats.read()['columns']['gap'];lo,hi=min(stats['min5']),max(stats['max5'])
    models={};selections=[];samples=[]
    for phase in ['onset','return_normal']:
        train=ds['optimization'].query('phase == @phase');val=ds['internal_validation'].query('phase == @phase')
        fits=[]
        for k in [1,2,3]:
            m=GaussianMixture(k,random_state=20261115,n_init=5,reg_covar=1e-4).fit(np.log1p(train.gap.to_numpy())[:,None])
            nll=-m.score(np.log1p(val.gap.to_numpy())[:,None]);selections.append(dict(phase=phase,k=k,internal_nll=nll,optimization_events=len(train),internal_events=len(val)));fits.append((nll,k,m))
        _,k,m=min(fits,key=lambda x:x[0]);models[phase]=dict(k=k,weights=m.weights_.tolist(),means=m.means_.reshape(-1).tolist(),variances=m.covariances_.reshape(-1).tolist())
        dev=ds['development'].query('phase == @phase')
        for gs in SAMPLES:
            rng=np.random.default_rng(gs+(1 if phase=='onset' else 2));c=rng.choice(k,len(dev),p=m.weights_)
            v=np.expm1(rng.normal(m.means_.reshape(-1)[c],np.sqrt(m.covariances_.reshape(-1)[c])))
            d=dev[['entity_id','event_index','phase','past_gap','past_gap_band']].copy();d['real_gap']=dev.gap.to_numpy();d['generated_gap']=np.clip(v,lo,hi)
            d['sampling_seed']=gs;d['clipped']=(v<lo)|(v>hi);samples.append(d)
    raw=pd.concat(samples,ignore_index=True);raw.to_parquet(OUT/'conditional_development.parquet',index=False)
    pd.DataFrame(selections).to_csv(DOCS/'mixture_selection.csv',index=False);write(OUT/'models.json',models)
    comparison=[]
    candidates=[('phase_mixture',-1,raw)]
    meta=ds['development'][['entity_id','event_index','phase','past_gap','past_gap_band']].rename(columns={'entity_id':'record'})
    for fs in [20260930,20261001]:
        for arm in ['frozen','all_label_fit','boundary_fit']:
            d=pd.read_parquet(ROOT/f'artifacts/argn_boundary_gap_v1/worker_{fs}/{arm}/conditional_development.parquet')
            d=d.merge(meta,on=['record','event_index'],validate='many_to_one').rename(columns={'record':'entity_id'})
            candidates.append((arm,fs,d))
    for arm,fs,d in candidates:
        for (phase,gs),g in d.groupby(['phase','sampling_seed']):
            for band,q in [('all',g)]+[(str(b),v) for b,v in g.groupby('past_gap_band')]:
                comparison.append(dict(arm=arm,fit_seed=fs,phase=phase,sampling_seed=gs,past_gap_band=band,events=len(q),customers=q.entity_id.nunique(),
                    sufficient_support=len(q)>=20 and q.entity_id.nunique()>=10,log_w1=w1(q.real_gap,q.generated_gap),
                    relative_gap_w1=w1(q.real_gap/q.past_gap,q.generated_gap/q.past_gap)))
    result=pd.DataFrame(comparison);result.to_csv(DOCS/'conditional_comparison.csv',index=False)
    write(OUT/'COMPLETE.json',dict(source_sha256=digest(__file__),protocol_sha256=digest(DOCS.parent/'SIMPLE_CONTROL_PROTOCOL.md'),
        mixture_models=models,clipped_samples=int(raw.clipped.sum()),clip_limits=[lo,hi],new_full_generations=0,test_events_read=False))
    print(result[result.past_gap_band.eq('all')].groupby(['arm','phase'])[['log_w1','relative_gap_w1']].mean().to_string())


if __name__=='__main__':main()
