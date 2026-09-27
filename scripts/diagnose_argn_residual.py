"""Frozen final candidate: phase/condition decomposition and full-prefix oracle."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from run_argn_amount_learning import (ROOT, inputs, load_parent, build_cache, AmountHeads,
    seed, digest, write, _sample, _translate_fixed_probs, _fix_rare_token_probs)
from run_argn_state_first import SOURCE

OUT=ROOT/'artifacts/argn_residual_v1'
DOCS=ROOT/'docs/argn_residual_v1'
REGISTRY=ROOT/'docs/argn_joint_preservation_v1/MODEL_REGISTRY.json'
FITS=[20260930,20261001]
DRAWS=[20261011,20261012,20261021,20261022]
SAMPLES=[20261111,20261112,20261113,20261114]
AMOUNT='amount_or_numeric_value'


def w1(a,b,weights=None):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    aa=np.isfinite(a)&(a>=0);bb=np.isfinite(b)&(b>=0)
    if not aa.any() or not bb.any():return np.nan
    ww=None if weights is None else np.asarray(weights)[bb]
    if ww is not None and ww.sum()<=0:return np.nan
    return float(wasserstein_distance(np.log1p(a[aa]),np.log1p(b[bb]),v_weights=ww))


def verify_registry():
    reg=json.loads(REGISTRY.read_text())
    for p,v in reg['shared'].items():assert digest(p)==v['sha256'],p
    for m in reg['models']:
        for v in m['components'].values():assert digest(v['path'])==v['sha256'],v['path']
    return reg


def verify():
    verify_registry();m=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in m['hashes'].items():assert digest(p)==h,p
    return m


def generation_path(arm,fs,gs):
    root=ROOT/'artifacts/argn_joint_preservation_v1';fresh=gs>=20261020
    if arm=='current':
        r=root/'confirmation/runs'/f'current_{fs}' if fresh else ROOT/f'artifacts/argn_gap_episode_v1/runs/gap_and_transition_{fs}'
    elif arm=='fixed_joint':
        r=root/('confirmation/runs' if fresh else 'runs')/f'count_hazard_no_context_{fs}'
    else:r=root/'onset_output'/('confirmation/runs' if fresh else 'runs')/f'onset_fit_{fs}'
    return r/f'generated_validation_{gs}.parquet'


def decorate(raw):
    d=raw.sort_values(['entity_id','event_index'],kind='stable').reset_index(drop=True).copy()
    assert not d.duplicated(['entity_id','event_index']).any()
    for c in [AMOUNT,'gap']:d[c]=pd.to_numeric(d[c]).astype(float)
    d['label']=pd.to_numeric(d.event_is_fraud).astype(int)
    d['previous']=d.groupby('entity_id',sort=False).label.shift().fillna(-1).astype(int)
    boundary=d.entity_id.ne(d.entity_id.shift())|d.label.ne(d.label.shift())
    run=boundary.cumsum();d['age']=d.groupby(run,sort=False).cumcount()+1
    left=d.groupby(run,sort=False).event_index.transform('min').eq(0)
    d['phase']=np.select([d.label.eq(1)&left,d.label.eq(1)&d.previous.eq(0),d.label.eq(1),
                          d.previous.eq(1),d.event_index.eq(0)],
                         ['left_fraud','onset','continuation','return_normal','first_normal'],default='normal_stay')
    d.loc[d.event_index.eq(0),'gap']=np.nan
    for col,name in [(AMOUNT,'past_amount'),('gap','past_gap')]:
        d[name]=d.groupby('entity_id',sort=False)[col].transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    normal_past=pd.Series(np.nan,index=d.index)
    for _,g in d.groupby('entity_id',sort=False):
        normal=g.loc[g.label.eq(0),AMOUNT].rolling(20,min_periods=5).median()
        normal_past.loc[g.index]=normal.reindex(g.index).ffill().shift(1)
    d['past_normal_amount']=normal_past
    d['relative']=d[AMOUNT]/d.past_amount.where(d.past_amount.gt(0))
    d['normal_relative']=d[AMOUNT]/normal_past.where(normal_past.gt(0))
    ever=d.groupby('entity_id',sort=False).label.cumsum().sub(d.label).gt(0)
    d['recovery20']=d.label.eq(0)&ever&d.age.le(20)
    return d


def decoded(records,codec):
    rows=[];cat=codec.prefixes['category']+'__cat'
    rev={int(v):k for k,v in codec.columns['category']['codes'].items()}
    lk=codec.prefixes['event_is_fraud']+'__cat'
    for ri,r in enumerate(records):
        n=len(r[lk]);d=dict(entity_id=np.repeat(ri,n),event_index=np.arange(n),
            event_is_fraud=(np.asarray(r[lk]).reshape(-1)==codec.codes['1']).astype(int),
            category=[rev[int(x)] for x in np.asarray(r[cat]).reshape(-1)])
        for col in ['gap',AMOUNT]:
            d[col]=codec.numeric({k:np.asarray(v).reshape(-1) for k,v in r.items() if k.startswith(codec.prefixes[col]+'__')},col)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows,ignore_index=True)


def bands(d,edges):
    for name,v in edges.items():
        d[name+'_band']=np.where(d[name].notna(),np.searchsorted(v,d[name],side='right'),-1)
    return d


def groups(d):
    yield 'all',d
    for label in [0,1]:yield f'label_{label}',d[d.label.eq(label)]
    for phase,g in d.groupby('phase',sort=True):yield phase,g
    yield 'recovery20',d[d.recovery20]


def summary(real,syn,info):
    rows=[];sg=dict(groups(syn))
    for group,r in groups(real):
        s=sg.get(group,syn.iloc[:0])
        for field in ['gap',AMOUNT,'relative','normal_relative','past_amount','past_gap','past_normal_amount']:
            a=r[field].dropna();b=s[field].dropna()
            rows.append(dict(**info,group=group,field=field,real_events=len(a),generated_events=len(b),
                real_customers=r.loc[r[field].notna(),'entity_id'].nunique(),
                generated_customers=s.loc[s[field].notna(),'entity_id'].nunique(),
                log_w1=w1(a,b),real_median=a.median(),generated_median=b.median()))
    return rows


def standardization(real,syn,info):
    cells=[];std=[]
    for label,field in [(1,'gap'),(0,AMOUNT),(1,'relative')]:
        r=real[real.label.eq(label)&real[field].notna()];s=syn[syn.label.eq(label)&syn[field].notna()]
        for cols in [['phase'],['phase','category'],['phase','past_amount_band'],['phase','past_gap_band']]:
            grouped={k:g for k,g in s.groupby(cols,observed=True)};weight=pd.Series(0.,index=s.index)
            ri=[];within=0.;covered=0.;supported=0.
            for key,a in r.groupby(cols,observed=True):
                b=grouped.get(key,s.iloc[:0]);share=len(a)/len(r);good=len(a)>=20 and len(b)>=20 and a.entity_id.nunique()>=10 and b.entity_id.nunique()>=10
                cells.append(dict(**info,label=label,field=field,conditioning='+'.join(cols),cell=str(key),real_events=len(a),
                    generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique(),
                    sufficient_support=good,log_w1=w1(a[field],b[field]),real_median=a[field].median(),generated_median=b[field].median()))
                if not len(b):continue
                covered+=share;ri.extend(a.index);weight.loc[b.index]=share/len(b)
                within+=share*w1(a[field],b[field]);supported+=share*good
            std.append(dict(**info,label=label,field=field,conditioning='+'.join(cols),real_coverage=covered,
                supported_real_mass=supported,raw_log_w1=w1(r[field],s[field]),
                common_support_standardized_w1=w1(r.loc[ri,field],s[field],weight),
                weighted_within_cell_w1=within/covered if covered else np.nan))
    return cells,std


def prepare():
    assert not (OUT/'MANIFEST.json').exists();verify_registry()
    OUT.mkdir(exist_ok=True,parents=True);DOCS.mkdir(exist_ok=True,parents=True)
    ws,codec,frames,metas=inputs();edges={}
    tr=decorate(decoded(frames['optimization'],codec))
    for name in ['past_amount','past_gap']:edges[name]=np.unique(np.quantile(tr[name].dropna(),[.25,.5,.75])).tolist()
    write(OUT/'EDGES.json',edges)
    for s in ['optimization','internal_validation','development']:
        d=bands(decorate(decoded(frames[s],codec)),edges);d.to_parquet(OUT/f'metadata_{s}.parquet',index=False)
        assert np.array_equal(d.label.to_numpy(),metas[s].label.to_numpy())
    paths=[Path(__file__),REGISTRY,DOCS/'PROTOCOL.md',OUT/'EDGES.json',SOURCE/'prepared/validation.parquet',
        ROOT/'scripts/run_argn_amount_learning.py',ROOT/'benchmarks/argn_amount_control.py',
        ROOT/'scripts/audit_argn_fit_generalization.py',ROOT/'benchmarks/argn_past_state.py',ROOT/'benchmarks/argn_label_first_control.py']
    paths += [generation_path(a,f,g) for a in ['current','fixed_joint','onset_fit'] for f in FITS for g in DRAWS]
    paths += [OUT/f'metadata_{s}.parquet' for s in frames]
    paths += [ROOT/f'artifacts/argn_amount_learning_v1/worker_{f}/development_cache.pt' for f in FITS]
    m=dict(created_utc=datetime.now(timezone.utc).isoformat(),hashes={str(p):digest(p) for p in paths},
        training=0,fit_seeds=FITS,generation_seeds=DRAWS,conditional_sample_seeds=SAMPLES,
        test_events_read=False,source_counts={s:dict(customers=len(frames[s]),events=len(metas[s])) for s in frames})
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('RESIDUAL_PREPARED',flush=True)


def rollout():
    verify();edges=json.loads((OUT/'EDGES.json').read_text())
    real=bands(decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet')),edges)
    real.to_parquet(OUT/'real_development.parquet',index=False)
    rows=[];cells=[];std=[]
    for fs in FITS:
        for arm in ['current','fixed_joint','onset_fit']:
            for gs in DRAWS:
                d=bands(decorate(pd.read_parquet(generation_path(arm,fs,gs))),edges)
                info=dict(arm=arm,fit_seed=fs,generation_seed=gs)
                rows+=summary(real,d,info);c,s=standardization(real,d,info);cells+=c;std+=s
                print('RESIDUAL_ROLLOUT',arm,fs,gs,flush=True)
    pd.DataFrame(rows).to_csv(DOCS/'rollout_phases.csv',index=False)
    pd.DataFrame(cells).to_csv(DOCS/'rollout_cells.csv',index=False)
    pd.DataFrame(std).to_csv(DOCS/'rollout_standardization.csv',index=False)
    write(DOCS/'ROLLOUT_COMPLETE.json',dict(compared_generations=24,new_training=0,test_events_read=False))


def get_head(model,keys,payload):
    head=AmountHeads(model,keys).to(model.device)
    head.load_state_dict({k:payload['state_dict'][k] for k in head.state_dict()})
    return head.eval().requires_grad_(False)


@torch.no_grad()
def worker(fs,device):
    verify();start=time.monotonic();device=torch.device(device);seed(fs)
    run=OUT/f'worker_{fs}';run.mkdir(exist_ok=False)
    write(run/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    _,codec,frames,metas=inputs();ws,model=load_parent(fs,device)
    records=frames['development'];meta=metas['development'];del frames,metas
    reg=verify_registry();comp=next(m['components'] for m in reg['models'] if m['fit_seed']==fs)
    payload=torch.load(comp['frozen_category_gap_amount']['path'],map_location='cpu',weights_only=True)
    onset=torch.load(comp['onset_amount_expert']['path'],map_location='cpu',weights_only=True)
    masks=_translate_fixed_probs(_fix_rare_token_probs(ws.tgt_stats.read()),ws.tgt_stats.read());checks={}
    for field in ['gap',AMOUNT]:
        keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes[field]+'__')]
        if field=='gap':cache,checks[field]=build_cache(model,records,meta,codec,keys,device)
        else:
            cache=torch.load(ROOT/f'artifacts/argn_amount_learning_v1/worker_{fs}/development_cache.pt',map_location='cpu',weights_only=True,mmap=True)
            checks[field]='reused hash-verified actual-prefix cache'
        assert len(cache['labels'])==len(meta) and np.array_equal(cache['labels'].numpy(),meta.label.to_numpy())
        head=get_head(model,keys,payload);expert=get_head(model,keys,onset) if field==AMOUNT else None
        selected_onset=meta.label.eq(1)&meta.previous_label.eq(0)
        parts_out=[];losses=[]
        for gs in SAMPLES:
            seed(gs)
            for lo in range(0,len(meta),4096):
                hi=min(lo+4096,len(meta));labels=cache['labels'][lo:hi].to(device)
                x=[cache['base'][lo:hi].to(device)];truth=cache['targets'][lo:hi].to(device);tokens={}
                mask=torch.as_tensor(selected_onset.iloc[lo:hi].to_numpy(),device=device)
                for j,key in enumerate(keys):
                    xx=torch.cat(x,-1);logits=head(xx,key,labels)
                    if expert is not None:logits=torch.where(mask[:,None],expert(xx,key,labels),logits)
                    value=_sample(logits.softmax(-1),fixed_probs=masks.get(key)).reshape(-1)
                    tokens[key]=value.cpu().numpy();x.append(model.embedders.get(key)(value))
                data=meta.iloc[lo:hi][['record','event_index','label']].copy()
                data['sampling_seed']=gs
                data['real_value']=codec.numeric({k:truth[:,j].cpu().numpy() for j,k in enumerate(keys)},field)
                data['generated_value']=codec.numeric(tokens,field);parts_out.append(data)
        data=pd.concat(parts_out,ignore_index=True);data.to_parquet(run/f'oracle_{field}.parquet',index=False)
        print('RESIDUAL_ORACLE',fs,field,len(data),flush=True)
        del cache,head,expert;torch.cuda.empty_cache()
    write(run/'COMPLETE.json',dict(seconds=time.monotonic()-start,checks=checks,test_events_read=False,new_training=0,
        conditional_positions=len(meta),samples_per_field=len(meta)*len(SAMPLES)))
    verify();print('RESIDUAL_WORKER_COMPLETE',fs,flush=True)


def oracle_report():
    verify();meta=pd.read_parquet(OUT/'metadata_development.parquet').rename(columns={'entity_id':'record'})
    rows=[]
    for fs in FITS:
        for field in ['gap',AMOUNT]:
            raw=pd.read_parquet(OUT/f'worker_{fs}/oracle_{field}.parquet')
            d=raw.merge(meta[['record','event_index','phase','recovery20','past_amount','past_normal_amount']],on=['record','event_index'],validate='many_to_one')
            d=d.rename(columns={'record':'entity_id'})
            if field=='gap':d=d[d.event_index.gt(0)]
            for gs,g in d.groupby('sampling_seed'):
                for group,q in groups(g):
                    rows.append(dict(fit_seed=fs,field=field,sampling_seed=gs,group=group,events=len(q),customers=q.entity_id.nunique(),
                        log_w1=w1(q.real_value,q.generated_value),real_median=q.real_value.median(),generated_median=q.generated_value.median(),
                        relative_w1=w1(q.real_value/q.past_amount,q.generated_value/q.past_amount) if field==AMOUNT else np.nan))
    pd.DataFrame(rows).to_csv(DOCS/'oracle_phases.csv',index=False)
    write(DOCS/'ORACLE_COMPLETE.json',dict(new_training=0,test_events_read=False,actual_prefix=True,current_predecessor_fields='actual'))
    print(pd.DataFrame(rows).groupby(['field','group']).mean(numeric_only=True).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','rollout','worker','report']);p.add_argument('--seed',type=int,choices=FITS);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='rollout':rollout()
    elif a.mode=='report':oracle_report()
    else:worker(a.seed,a.device)
