"""Frozen calibrated predictors on both native history sources; no fitting."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments import cs_saf_generation_repeats as parent
from experiments.cs_saf_rollout_audit import initial_hidden, stream_event
from experiments.cs_saf_pilot import ROOT, frozen_source, state_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONFIG = ROOT/'configs/benchmark_v2/cs_saf_calibrated_replay_v1.json'
CONFIG_SHA = '2ea453dd07898be16ac539335807a048c00b24a72706f7f94bb1699fd5713b41'
OUTPUT = ROOT/'artifacts/cs_saf/calibrated_replay_v1'
NAMES = ('Ucal', 'Ecal')
GROUPS = ('pooled', 'context_0', 'context_1')
FIELDS = ('gap', 'receiver', 'numeric_value', 'valid_mask', 'lengths', 'codes')
ATOL = 3e-6


def contract():
    if sha256(CONFIG) != CONFIG_SHA: raise ValueError('changed registration')
    c = json.loads(CONFIG.read_text())
    for name, digest in c['frozen_sources'].items():
        if sha256(ROOT/name) != digest: raise ValueError('changed frozen source: '+name)
    if sha256(parent.OUTPUT/'native_verification.json') != c['parent_native_verification_sha256']:
        raise ValueError('changed reference evidence')
    parent.contract()
    return c


def folder_for(pi, kappa, trial):
    return OUTPUT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}'


def finish(folder, source, **extra):
    if (folder/'COMPLETE.json').exists(): raise FileExistsError('immutable completed output')
    write_json(folder/'COMPLETE.json', dict(status='COMPLETE', source_commit=source,
        config_sha256=CONFIG_SHA, artifact_sha256={str(p.relative_to(folder)):sha256(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}, **extra))


def verify(folder):
    m = json.loads((folder/'COMPLETE.json').read_text())
    if m['status'] != 'COMPLETE' or m['config_sha256'] != CONFIG_SHA: raise ValueError('wrong manifest')
    for n,h in m['artifact_sha256'].items():
        if sha256(folder/n) != h: raise ValueError('changed artifact: '+str(folder/n))
    return m


def as_data(sample):
    return {k:sample['static_codes'] if k == 'codes' else sample[k] for k in FIELDS}


def selected(data, ids):
    return {k:v[ids].clone() for k,v in data.items()}


def transition_mask(data, group):
    mask = data['valid_mask'].numpy().copy(); mask[:,0] = False
    if group != 'pooled': mask &= (data['codes'].numpy() == int(group[-1])+3)[:,None]
    return mask


@torch.no_grad()
def probabilities(model, x):
    """Use the native generation categorical normalization, strict-past encoder."""
    h = model.encoder(x['gap'], x['receiver'], x['numeric_value'], x['valid_mask'],
                      static_categorical=(x['codes'],))
    ctx = model.context(h, (x['codes'],))
    mask = x['valid_mask'].clone(); mask[:,0] = False
    prev = x['receiver'].roll(1, 1)[mask]
    codes = x['codes'][:,None].expand_as(mask)[mask]
    parts=[]; max_formula_error=0.
    for start in range(0,len(prev),2048):
        sl=slice(start,start+2048); c=ctx[mask][sl]; g=x['gap'][mask][sl]; p=prev[sl]
        logp,logr,_=model.mark_distribution(c,g,p,torch.ones_like(p,dtype=torch.bool),static_codes=codes[sl])
        probs=logp.softmax(-1)
        if not torch.isfinite(probs).all() or (probs[:,:3]!=0).any(): raise ValueError('invalid categorical probabilities')
        if (probs.sum(-1)-1).abs().max()>ATOL: raise ValueError('unnormalized probabilities')
        repeat=probs.gather(1,p[:,None])[:,0]
        err=float((repeat-logr.exp()).abs().max());max_formula_error=max(err,max_formula_error)
        if err>ATOL: raise ValueError('copy/fresh observable repeat mismatch')
        parts.append(probs)
    return torch.cat(parts), mask, max_formula_error


@torch.no_grad()
def evaluate_pair(models, data, device, batch_size=128):
    n,t=data['gap'].shape
    repeats=np.zeros((2,n,t),np.float32); tv=np.zeros((n,t),np.float64);largest=0.
    for start in range(0,n,batch_size):
        x={k:v[start:start+batch_size].to(device) for k,v in data.items()}
        probs=[]
        for j,name in enumerate(NAMES):
            p,mask,error=probabilities(models[name],x);probs.append(p.double());largest=max(largest,error)
            r=p.gather(1,x['receiver'].roll(1,1)[mask][:,None])[:,0]
            view=repeats[j,start:start+len(mask)];view[mask.cpu().numpy()]=r.cpu().numpy()
        dist=.5*(probs[1]-probs[0]).abs().sum(-1)
        rdelta=(probs[1]-probs[0]).gather(1,x['receiver'].roll(1,1)[mask][:,None])[:,0].abs()
        if (rdelta-dist).max()>ATOL: raise ValueError('TV contraction failed')
        tv[start:start+len(mask)][mask.cpu().numpy()]=dist.cpu().numpy()
    return dict(repeat=repeats,mark_tv=tv,max_formula_error=largest)


@torch.no_grad()
def sequential_probabilities(model,data,device,*,prefix=False):
    x={k:v.to(device) for k,v in data.items()}; n,t=x['gap'].shape
    allp=torch.zeros((n,t,67),dtype=torch.float64,device=device)
    hidden,state=initial_hidden(model,x['codes'])
    for step in range(1,t):
        if prefix:
            h=model.encoder(x['gap'][:,:step+1],x['receiver'][:,:step+1],x['numeric_value'][:,:step+1],
                x['valid_mask'][:,:step+1],static_categorical=(x['codes'],))[:,-1]
        else:
            h,state=stream_event(model,x['gap'][:,step-1],x['receiver'][:,step-1],
                x['numeric_value'][:,step-1],x['valid_mask'][:,step-1],state)
        active=x['valid_mask'][:,step]
        c=model.context(h,(x['codes'],))
        logp,_,_=model.mark_distribution(c,x['gap'][:,step],x['receiver'][:,step-1],active,static_codes=x['codes'])
        allp[active,step]=logp.softmax(-1)[active].double()
    return allp.cpu().numpy()


def check_replay(models,data,device,*,prefix=False):
    ids=torch.cat([torch.where(data['codes']==s)[0][:4] for s in (3,4)])
    small=selected(data,ids);x={k:v.to(device) for k,v in small.items()}
    result=evaluate_pair(models,small,device);maxerror=0.
    mask=transition_mask(small,'pooled');seq=[]
    for j,name in enumerate(NAMES):
        p,_,_=probabilities(models[name],x)
        sequential=sequential_probabilities(models[name],small,device)
        maxerror=max(maxerror,float(np.max(abs(sequential[mask]-p.cpu().numpy()))))
        if prefix:
            direct=sequential_probabilities(models[name],small,device,prefix=True)
            maxerror=max(maxerror,float(np.max(abs(direct-sequential))))
        seq.append(sequential)
        rows,steps=np.where(mask);prev=small['receiver'].numpy()[rows,steps-1]
        maxerror=max(maxerror,float(np.max(abs(result['repeat'][j][mask]-sequential[rows,steps,prev]))))
    maxerror=max(maxerror,float(np.max(abs(result['mark_tv'][mask]-.5*np.abs(seq[1][mask]-seq[0][mask]).sum(-1)))))
    if maxerror>ATOL: raise ValueError(f'sequential replay disagreement {maxerror}')
    return result,maxerror,ids.tolist()


def bin_means(bins,values,n_bins):
    counts=np.bincount(bins,minlength=n_bins)
    totals=np.bincount(bins,weights=np.asarray(values,dtype=float),minlength=n_bins)
    return counts,np.divide(totals,counts,out=np.zeros(n_bins),where=counts>0)


def score_path(data, arrays, ref, edges):
    result={}
    for group in GROUPS:
        mask=transition_mask(data,group);rows,steps=np.where(mask)
        bins=np.searchsorted(np.asarray(edges[group]),data['gap'].numpy()[mask].astype(float),side='right')
        reference=np.array(ref[group]);counts_ref=reference.sum(1)
        target=np.divide(reference[:,1],counts_ref,out=np.zeros(len(reference)),where=counts_ref>0)
        weight=counts_ref/counts_ref.sum();marks=data['receiver'].numpy()
        count,emp=bin_means(bins,marks[rows,steps]==marks[rows,steps-1],len(target))
        predicted={name:bin_means(bins,arrays['repeat'][j][mask],len(target))[1] for j,name in enumerate(NAMES)}
        l1=lambda v:float(weight@abs(v-target))
        delta=arrays['repeat'][1][mask].astype(float)-arrays['repeat'][0][mask].astype(float)
        tv=arrays['mark_tv'][mask]
        entcounts=np.bincount(rows,minlength=len(data['codes']));enttv=np.bincount(rows,weights=tv,minlength=len(entcounts))
        stages={}
        for label,lo,hi in (('early',1,10),('middle',11,20),('late',21,31)):
            ix=(steps>=lo)&(steps<=hi)
            stages[label]=dict(transitions=int(ix.sum()),mark_TV=float(tv[ix].mean()) if ix.any() else None,
                signed_repeat_difference=float(delta[ix].mean()) if ix.any() else None,
                absolute_repeat_difference=float(abs(delta[ix]).mean()) if ix.any() else None)
        result[group]=dict(counts=count.tolist(),empty_bins=np.flatnonzero(count==0).tolist(),
            reference_counts=reference.tolist(),reference_repeat=target.tolist(),reference_weights=weight.tolist(),
            empirical_repeat=emp.tolist(),empirical_L1=l1(emp),
            predicted_repeat={n:v.tolist() for n,v in predicted.items()},
            expected_curve_L1={n:l1(v) for n,v in predicted.items()},
            signed_bias={n:(v-target).tolist() for n,v in predicted.items()},
            mark_TV_transition_mean=float(tv.mean()),mark_TV_entity_mean=float((enttv[entcounts>0]/entcounts[entcounts>0]).mean()),
            signed_repeat_difference=float(delta.mean()),absolute_repeat_difference=float(abs(delta).mean()),
            bin_mark_TV=bin_means(bins,tv,len(target))[1].tolist(),
            bin_signed_repeat_difference=bin_means(bins,delta,len(target))[1].tolist(),stages=stages)
    return result


def decomposition(source_scores):
    out={}
    for group in GROUPS:
        u=source_scores['Ucal'][group];e=source_scores['Ecal'][group]
        uu,eu=u['expected_curve_L1']['Ucal'],u['expected_curve_L1']['Ecal']
        ue,ee=e['expected_curve_L1']['Ucal'],e['expected_curve_L1']['Ecal']
        f=.5*(ee-ue+eu-uu);h=.5*(ee-eu+ue-uu)
        d=e['empirical_L1']-u['empirical_L1'];s=(e['empirical_L1']-ee)-(u['empirical_L1']-uu)
        if abs(d-f-h-s)>1e-12:raise ValueError('nonlinear L1 decomposition failed')
        a,b,c,z=(np.array(v) for v in (u['predicted_repeat']['Ucal'],u['predicted_repeat']['Ecal'],
                                       e['predicted_repeat']['Ucal'],e['predicted_repeat']['Ecal']))
        signed_f=.5*(z-c+b-a);signed_h=.5*(z-b+c-a)
        signed_d=np.array(e['empirical_repeat'])-u['empirical_repeat']
        signed_s=(np.array(e['empirical_repeat'])-z)-(np.array(u['empirical_repeat'])-a)
        np.testing.assert_allclose(signed_d,signed_f+signed_h+signed_s,atol=1e-12,rtol=0)
        out[group]=dict(matrix_target_by_source=[[uu,ue],[eu,ee]],predictor_F=f,source_H=h,
            empirical_remainder_S=s,expected_difference=ee-uu,native_difference=d,
            predictor_on_U=eu-uu,predictor_on_E=ee-ue,source_under_U=ue-uu,source_under_E=ee-eu,
            interaction=ee-ue-eu+uu,signed_bin=dict(F=signed_f.tolist(),H=signed_h.tolist(),S=signed_s.tolist(),native=signed_d.tolist()))
    return out


def references(pi,kappa):
    native=json.loads((parent.OUTPUT/'native_verification.json').read_text())
    records=[r for r in native['records'] if r['pi']==pi and r['kappa']==kappa]
    refs={}
    for r in records:
        if r['group'] in refs and refs[r['group']]!=r['reference_counts']:raise ValueError('inconsistent references')
        refs[r['group']]=r['reference_counts']
    if set(refs)!=set(GROUPS):raise ValueError('missing reference group')
    return refs


def load_models(pi,kappa,trial,device):
    payload=parent.parent.load_cache(parent.parent.CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt')
    models={};inputs={}
    for name in NAMES:
        models[name],inputs[name],_=parent.load_fixed(payload,pi,kappa,trial,name,device)
    return models,inputs


def load_path(pi,kappa,trial,name,repeat,inputs):
    folder=parent.folder_for(pi,kappa,trial)/name/f'repeat_{repeat}';parent.verify(folder)
    saved=torch.load(folder/'generated_sample.pt',map_location='cpu')
    old=json.loads((folder/'comparison.json').read_text());sample=saved['sample']
    plan=torch.load(parent.folder_for(pi,kappa,trial)/'plan.pt',map_location='cpu')
    seed=parent.contract()['generation_seeds'][str(trial)][repeat]
    if saved['model_state_sha256']!=inputs[name]['state_sha256'] or saved['sampling_seed']!=seed:raise ValueError('wrong source model or seed')
    for key,expected in (('static_codes',plan['contexts']),('lengths',plan['lengths']),('plan_train_indices',torch.as_tensor(plan['positions']))):
        torch.testing.assert_close(sample[key],expected,rtol=0,atol=0)
    return as_data(sample),old,dict(sample_sha256=sha256(folder/'generated_sample.pt'),manifest_sha256=sha256(folder/'COMPLETE.json'),sampling_seed=seed)


def run_cell(pi,kappa,trial,device):
    contract();source=frozen_source();out=folder_for(pi,kappa,trial);started=time.monotonic()
    if (out/'COMPLETE.json').exists():
        m=verify(out)
        if m['source_commit']!=source:raise ValueError('resume source changed')
        return
    out.mkdir(parents=True,exist_ok=False)
    try:
        parent.verify(parent.folder_for(pi,kappa,trial))
        models,inputs=load_models(pi,kappa,trial,device);ref=references(pi,kappa)
        paths={};decomposed={};max_native=0.;max_replay=0.;max_formula=0.;identities={}
        for repeat in range(5):
            scores={}
            for name in NAMES:
                data,old,identity=load_path(pi,kappa,trial,name,repeat,inputs)
                arrays=evaluate_pair(models,data,device);max_formula=max(max_formula,arrays['max_formula_error'])
                if repeat==0:
                    _,error,_=check_replay(models,data,device);max_replay=max(max_replay,error)
                edges={g:old['metrics'][g]['train_metric_state']['gap_bin_edges'] for g in GROUPS}
                scores[name]=score_path(data,arrays,ref,edges)
                for g in GROUPS:
                    error=abs(scores[name][g]['empirical_L1']-old['metrics'][g]['metrics']['short_gap_repeat_curve_l1'])
                    max_native=max(max_native,error)
                    if error>1e-12:raise ValueError('native parent L1 mismatch')
                key=f'{name}_repeat_{repeat}';paths[key]=dict(scores=scores[name],edges=edges,identity=identity)
                identities[key]=identity
                np.savez_compressed(out/(key+'.npz'),repeat=arrays['repeat'],mark_tv=arrays['mark_tv'])
            decomposed[str(repeat)]=decomposition(scores)
        for name in NAMES:
            if state_digest(models[name])!=inputs[name]['state_sha256']:raise ValueError('model mutated')
            if sha256(Path(inputs[name]['checkpoint_path']))!=inputs[name]['checkpoint_sha256']:raise ValueError('checkpoint changed')
        write_json(out/'replay.json',dict(pi=pi,kappa=kappa,trial=trial,paths=paths,decomposition=decomposed,
            inputs=inputs,parent_cell_manifest_sha256=sha256(parent.folder_for(pi,kappa,trial)/'COMPLETE.json'),
            max_native_error=max_native,max_sequential_error=max_replay,max_formula_error=max_formula,
            seconds=time.monotonic()-started,source_commit=source,config_sha256=CONFIG_SHA,
            device=str(device),test_accessed=False,new_fits=0,new_sequences=0))
        finish(out,source);print(f'COMPLETE pi={pi} kappa={kappa} trial={trial} seconds={time.monotonic()-started:.1f}',flush=True)
    except Exception as exc:
        write_json(out/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),source_commit=source));raise


def smoke(device,gate):
    contract();source=frozen_source();out=OUTPUT/gate;out.mkdir(parents=True,exist_ok=False)
    models,inputs=load_models(.05,1,0,device);records={};saved={}
    for name in NAMES:
        data,_,_=load_path(.05,1,0,name,0,inputs)
        result,error,ids=check_replay(models,data,device,prefix=True)
        records[name]=dict(max_sequential_and_prefix_error=error,ids=ids)
        for key in ('repeat','mark_tv'):saved[name+'_'+key]=result[key]
    for n in NAMES:
        if state_digest(models[n])!=inputs[n]['state_sha256']:raise ValueError('gate model mutated')
    if gate=='gpu_gate':
        previous=np.load(OUTPUT/'cpu_gate/probabilities.npz');err=max(float(np.max(abs(previous[k]-v))) for k,v in saved.items())
        if err>ATOL:raise ValueError('CPU GPU replay disagreement')
        records['max_CPU_GPU_error']=err
    np.savez_compressed(out/'probabilities.npz',**saved)
    write_json(out/'gate.json',dict(decision='PASS',source_commit=source,config_sha256=CONFIG_SHA,device=str(device),records=records))
    finish(out,source);print(json.dumps(records),flush=True)
