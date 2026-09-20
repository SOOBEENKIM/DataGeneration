"""Frozen U/G, ten-parameter train-only sequential calibration.

No oracle target or validation objective is exposed to the search functions.
Two tape losses are averaged (not their frequency tables before squaring).
"""
import copy
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

from experiments import cs_saf_structure_v1 as parent
from experiments.cs_saf_pilot import state_digest
from models.cs_saf_observed_repeat_control import assign_bins, apply_numpy, apply_torch, redistribute
from scripts.run_cs_saf_external_audit_v1 import digest, write

ROOT = parent.ROOT
CONFIG = ROOT/'configs/benchmark_v2/cs_saf_rollout_calibration_v1.json'
OUT = ROOT/'artifacts/cs_saf/rollout_calibration_v1'
DOC = ROOT/'docs/cs_saf/rollout_calibration_v1'
SOURCES = parent.SOURCES + ['experiments/cs_saf_rollout_calibration_v1.py',
    'scripts/run_cs_saf_rollout_calibration_v1.py',
    'configs/benchmark_v2/cs_saf_rollout_calibration_v1.json']


def config():
    return json.loads(CONFIG.read_text())


def sources():
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:
        assert (ROOT/name).read_bytes() == subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT), name
    return dict(commit=commit, hashes={name:digest(ROOT/name) for name in SOURCES})


def folder(k,t,name):
    return OUT/f'runs/kappa_{k}/trial_{t}/{name}'


def load_parent(k,t,name,device):
    old = parent.folder_for(k,t,name)
    expected = config()['parents'][f'{k}/{t}/{name}']
    for f,h in expected.items():
        assert digest(old/f) == h, str(old/f)
    payload, provenance = parent.payload_for(k)
    model = parent.make_model(payload,name,t,device)
    model.load_state_dict(torch.load(old/'checkpoint_best.pt',map_location=device)['model_state'],strict=True)
    model.eval().requires_grad_(False)
    a = json.loads((old/'evaluation/fit.json').read_text())['controls']['gap']
    features = pd.read_parquet(old/'evaluation/train_features.parquet')
    assert set(features.entity_id) == set(payload['train']['entity_ids'])
    assert len(features) == int((payload['train']['lengths']-1).sum())
    return model,payload,a,features,dict(parent_files=expected,data=provenance,state_sha256=state_digest(model))


def plans(train,k):
    c=config(); rng=np.random.default_rng(c['plan_seed_base']+k)
    fit=[]; select=[]; n=c['plan_entities_per_group']
    for g in (3,4):
        ids=np.flatnonzero(train['codes'].numpy()==g)
        ids=rng.permutation(ids); assert len(ids)>=2*n
        fit.extend(ids[:n]);select.extend(ids[n:2*n])
    fit=np.asarray(fit);select=np.asarray(select)
    assert not set(fit)&set(select)
    def plan(ids):
        return dict(positions=ids,lengths=train['lengths'][ids].clone(),codes=train['codes'][ids].clone())
    return plan(fit),plan(select)


def control_at(a,parameters):
    result=copy.deepcopy(a)
    result['parameters']=np.asarray(parameters).reshape(2,5).tolist()
    # Old optimizer diagnostics do not describe the new fit.
    result.pop('groups',None)
    result['target']='outer_train_observed_gap_bin_repeat_joint_frequency'
    return result


class TrainingTarget:
    def __init__(self,features,a):
        self.group=features.group.to_numpy(int)
        self.bin=assign_bins(features.gap_code.to_numpy(),self.group,a['edges'])
        self.y=features.y.to_numpy(float)
        self.z=logit(features.p.to_numpy(float).clip(1e-9,1-1e-9))
        self.edges=a['edges']
        self.target=np.stack([np.bincount(2*self.bin[self.group==g]+self.y[self.group==g].astype(int),
            minlength=10)/sum(self.group==g) for g in (0,1)])
        self.base=self.costs(np.asarray(a['parameters']))

    def costs(self,parameters):
        z=self.z+np.asarray(parameters).reshape(2,5)[self.group,self.bin]
        p=expit(z)
        return np.array([[np.mean((p[self.group==g]-self.y[self.group==g])**2),
            np.mean((np.logaddexp(0,z)-self.y*z)[self.group==g])] for g in (0,1)])

    def feasible(self,parameters):
        d=self.costs(parameters)-self.base;c=config()['prediction_guard']
        return bool((d[:,0]<=c['training_brier_increase_each_group']+1e-12).all() and
            (d[:,1]<=c['training_mark_nll_increase_each_group']+1e-12).all())

    def frequencies(self,sample,model):
        valid=sample['valid_mask'].numpy().copy();valid[:,0]=False
        groups=np.broadcast_to(sample['codes'].numpy()[:,None]-3,valid.shape)[valid]
        gap=sample['gap'].numpy()[valid]
        code=np.searchsorted(np.asarray(model.support.upper_bounds,dtype=np.float32),gap,side='left')
        bins=assign_bins(code,groups,self.edges)
        marks=sample['receiver'].numpy()
        repeat=(marks==np.roll(marks,1,axis=1))[valid].astype(int)
        return np.stack([np.bincount(2*bins[groups==g]+repeat[groups==g],minlength=10)/sum(groups==g) for g in (0,1)])


def initial_hidden(model,codes):
    e=model.encoder
    h=e.start[None].expand(len(codes),-1)+e.static_projection(e.static_categorical[0](codes))
    return h,h[None].repeat(e.config.num_layers,1,1).contiguous()


def stream_event(model,gap,mark,value,valid,state):
    e=model.encoder
    features=torch.stack((torch.log1p(torch.nan_to_num(gap).clamp_min(0)),torch.isfinite(gap).to(gap.dtype)),-1)
    event=torch.cat((e.history_gap(features),e.history_mark(mark.clamp(0,66)),torch.nan_to_num(value)[:,None]),-1)
    event=e.input_projection(event)*valid[:,None]
    h,state=e.gru(event[:,None],state)
    return h[:,0],state


def inverse_cdf(p,u):
    p=p.double();p=p/p.sum(-1,keepdim=True)
    # CUDA cumsum is disallowed by torch 2.1 deterministic mode.
    cdf=p.cpu().cumsum(-1).to(p.device);cdf[:,-1]=1.
    return (u[:,None]>=cdf).sum(-1)


def corrected_probabilities(model,h,codes,gap,previous,has_previous,control):
    context=model.context(h,(codes,))
    lp,_,_=model.mark_distribution(context,gap,previous,has_previous,static_codes=codes)
    p=lp.exp()
    if control is not None and has_previous.any():
        ix=has_previous
        repeat=p[ix].gather(1,previous[ix,None])[:,0]
        code=(model._support_code(gap[ix])-3).clamp_min(0)
        new=apply_torch(repeat,code,codes[ix]-3,control)
        p[ix]=redistribute(p[ix],previous[ix],new)
    return p


@torch.no_grad()
def generate(model,plan,seed,control,device,batch_size=256,check_prefix=False):
    n=len(plan['lengths']);steps=32
    streams=np.random.SeedSequence(seed).spawn(3)
    tapes=[np.random.default_rng(streams[i]).random((n,steps)) for i in (0,1)]
    tapes.append(np.random.default_rng(streams[2]).standard_normal((n,steps)))
    parts=[];max_hidden_error=0.;max_probability_error=0.
    for start in range(0,n,batch_size):
        sl=slice(start,min(n,start+batch_size));lengths=plan['lengths'][sl].to(device);codes=plan['codes'][sl].to(device)
        gu,mu,vn=[torch.from_numpy(v[sl]).to(device) for v in tapes]
        valid=torch.arange(steps,device=device)[None,:]<lengths[:,None]
        gap=torch.full(valid.shape,float('nan'),device=device);mark=torch.zeros_like(valid,dtype=torch.long)
        value=torch.zeros_like(gap);h,state=initial_hidden(model,codes)
        for t in range(steps):
            active=valid[:,t]
            if t:
                h,state=stream_event(model,gap[:,t-1],mark[:,t-1],value[:,t-1],valid[:,t-1],state)
                b=inverse_cdf(model.gap_decoder.logits(h).softmax(-1),gu[:,t])
                gap[active,t]=model.support.decode_tensor(b)[active]
            previous=mark[:,t-1] if t else torch.ones(len(codes),device=device,dtype=torch.long)
            p=corrected_probabilities(model,h,codes,gap[:,t],previous,active & (t>0),control)
            if check_prefix:
                local=mark[:,:t+1].clone();local[:,-1]=1
                full=model.encoder(gap[:,:t+1],local,value[:,:t+1],valid[:,:t+1],static_categorical=(codes,))[:,-1]
                max_hidden_error=max(max_hidden_error,float((full-h).abs().max()))
                fullp=corrected_probabilities(model,full,codes,gap[:,t],previous,active & (t>0),control)
                max_probability_error=max(max_probability_error,float((p-fullp).abs().max()))
            emitted=inverse_cdf(p,mu[:,t]);mark[active,t]=emitted[active]
            loc,scale=model._value_parameters(h,gap[:,t],emitted)
            numeric=loc+scale*vn[:,t].to(loc.dtype);value[active,t]=numeric[active]
        assert (mark[valid]>=3).all() and (mark[valid]<=66).all()
        assert torch.isfinite(value[valid]).all() and torch.isnan(gap[:,0]).all()
        parts.append({key:v.cpu() for key,v in dict(gap=gap,receiver=mark,numeric_value=value,
            valid_mask=valid,lengths=lengths,codes=codes).items()})
    result={key:torch.cat([part[key] for part in parts]) for key in parts[0]}
    if check_prefix:
        assert max_hidden_error<2e-5 and max_probability_error<2e-5,(max_hidden_error,max_probability_error)
        result['prefix_check']=dict(hidden_max_error=max_hidden_error,probability_max_error=max_probability_error)
    return result


def pattern_search(initial,objective,selection,feasible,trial,progress=None):
    c=config();current=np.asarray(initial,dtype=float).reshape(-1).copy()
    order=np.random.default_rng(c['coordinate_seed_base']+trial).permutation(10)
    value=objective(current);endpoints=[current.copy()];trace=[];calls=1
    for sweep,step in enumerate(c['step_sizes']):
        for coordinate in order:
            candidates=[]
            for sign in c['search_signs']:
                candidate=current.copy();candidate[coordinate]=np.clip(candidate[coordinate]+sign*step,-8,8)
                loss=objective(candidate);calls+=1;eligible=feasible(candidate)
                candidates.append((candidate,loss,eligible))
                trace.append(dict(sweep=sweep,coordinate=int(coordinate),sign=sign,parameters=candidate.tolist(),
                    objective=float(loss),eligible=eligible))
            best=current;bestvalue=value
            for candidate,loss,eligible in candidates:
                if eligible and loss<bestvalue-c['tie_improvement']:best,bestvalue=candidate,loss
            current,value=best,bestvalue
        endpoints.append(current.copy())
        if progress:progress(sweep,value)
    chosen=0;best=float('inf');selections=[]
    for i,candidate in enumerate(endpoints):
        loss=selection(candidate);eligible=feasible(candidate)
        selections.append(dict(endpoint=i,parameters=candidate.tolist(),objective=float(loss),eligible=eligible))
        if eligible and loss<best-c['tie_improvement']:best,chosen=loss,i
    assert calls==61 and len(trace)==60 and len(selections)==4
    return endpoints[chosen],dict(trace=trace,selections=selections,selected_endpoint=chosen,
        search_objective_calls=calls,selection_objective_calls=4,coordinate_order=order.tolist())


def fit(k,t,name,device):
    c=config();src=sources();assert str(device).startswith('cuda')
    for gate in ['cpu_gate.json','gpu_gate.json']:
        g=json.loads((DOC/gate).read_text());assert g['status']=='PASS' and g['source_hashes']==src['hashes']
    model,payload,a,features,provenance=load_parent(k,t,name,device)
    target=TrainingTarget(features,a);fitplan,selplan=plans(payload['train'],k)
    dest=folder(k,t,name);dest.mkdir(parents=True,exist_ok=True)
    for method in ['B','P']:
        done=dest/f'{method}_DONE.json'
        if done.exists():
            saved=json.loads(done.read_text());assert saved['source']['hashes']==src['hashes'];continue
        assert not (dest/f'{method}_START.json').exists(),'incomplete run needs technical review'
        write(dest/f'{method}_START.json',dict(source=src,provenance=provenance,kappa=k,trial=t,name=name,
            method=method,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),start_time=time.time()))
        calls=[0];started=time.monotonic()
        def objective(parameters,selection=False):
            plan=selplan if selection else fitplan
            seeds=c['selection_seeds' if selection else 'search_seeds'];losses=[]
            for seed in seeds:
                sample=generate(model,plan,seed+100*t,control_at(a,parameters),device,c['generation_batch_size'])
                frequencies=target.frequencies(sample,model)
                losses.append(float(np.mean(np.sum((frequencies-target.target)**2,axis=1))));calls[0]+=1
            return float(np.mean(losses))
        def progress(sweep,value):
            print(f'FIT {k}/{t}/{name}/{method} sweep={sweep+1}/3 seconds={time.monotonic()-started:.1f}',flush=True)
        try:
            parameters,trace=pattern_search(a['parameters'],objective,lambda x:objective(x,True),
                target.feasible if method=='P' else lambda x:True,t,progress)
            assert calls[0]==130 and state_digest(model)==provenance['state_sha256']
            result=dict(kappa=k,trial=t,name=name,method=method,source=src,provenance=provenance,
                control=control_at(a,parameters),initial_control=a,target=target.target.tolist(),
                training_cost_delta=(target.costs(parameters)-target.base).tolist(),
                training_guard_satisfied=target.feasible(parameters),search=trace,sample_calls=calls[0],
                fit_positions=fitplan['positions'].tolist(),selection_positions=selplan['positions'].tolist(),
                seconds=time.monotonic()-started,end_time=time.time(),weights_unchanged=True,
                validation_used=False,oracle_used=False,test_accessed=False,
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write(done,result)
            print(f'FIT_DONE {k}/{t}/{name}/{method} seconds={result["seconds"]:.1f}',flush=True)
        except Exception as exc:
            write(dest/f'{method}_FAILED.json',dict(error=repr(exc),source=src));raise


def all_fitted():
    c=config();records=[]
    for k in c['kappas']:
        for t in c['trials']:
            for name in c['models']:
                for method in ['B','P']:
                    path=folder(k,t,name)/f'{method}_DONE.json'
                    record=json.loads(path.read_text())
                    assert not record['validation_used'] and not record['oracle_used'] and record['sample_calls']==130
                    assert record['weights_unchanged']
                    for f,h in record['source']['hashes'].items():assert digest(ROOT/f)==h
                    records.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),end_time=record['end_time']))
    assert len(records)==24
    return records
