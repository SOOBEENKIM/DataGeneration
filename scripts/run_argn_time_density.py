"""Registered equal-capacity temporal density study; no sealed test reads."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import torch
from run_argn_amount_learning import (ROOT,BASE,inputs,load_parent,build_cache,seed,digest,write,
    Workspace,folder,get_cardinalities,get_ctx_sequence_length,load_model_weights,_translate_fixed_probs,_fix_rare_token_probs)
from run_argn_boundary_gap import generation_args
from diagnose_argn_residual import verify_registry,generation_path,w1
import run_argn_joint_preservation as study
from benchmarks.argn_time_density import TimeDensity,PastClock,clock_features,time_density_generation,ARMS
from benchmarks.argn_phase_gap import encode_values

OUT=ROOT/'artifacts/argn_time_density_v1';DOCS=ROOT/'docs/argn_time_density_v1'
META=ROOT/'artifacts/argn_residual_v1'
CFG=dict(fit_seeds=[20260930,20261001],generation_seeds=[20261011,20261012,20261021,20261022],
    sampling_seeds=[20261111,20261112,20261113,20261114],arms=list(ARMS),components=3,hidden=64,
    learning_rate=3e-4,weight_decay=1e-4,batch_size=512,max_steps=2000,validation_every=100,
    patience=6,min_delta=1e-5,gradient_clip=1.,allowed_gpu_uuids=[
    'GPU-91378a3a-8cbc-62ac-7636-36e0f3f7d933','GPU-4e575de7-5122-2c13-d12d-9ee15e4a1af9',
    'GPU-f1d4c556-ae70-3415-adec-c1d68f9bb637','GPU-5eae2fe0-7b63-fcc9-dd31-4e915ab63073'])


def verify():
    verify_registry();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    verify_registry();assert not (OUT/'MANIFEST.json').exists()
    m=pd.read_parquet(META/'metadata_optimization.parquet');valid=m.event_index.gt(0)
    fallback=float(m.loc[valid&m.gap.gt(0),'gap'].median());clock,_=clock_features(m,fallback)
    phase=2*m.previous.clip(lower=0).to_numpy()+m.label.to_numpy()
    params=dict(fallback=fallback,limits=[0.,1045691.],quantum=1.,priors={})
    for kind in ['absolute','relative']:
        params['priors'][kind]={}
        z=np.log1p(m.gap.to_numpy())-(clock if kind=='relative' else 0)
        for p in range(4):
            x=z[valid.to_numpy()&(phase==p)]
            if len(x)>50000:x=np.random.default_rng(20261115).choice(x,50000,replace=False)
            fit=GaussianMixture(3,random_state=20261115,n_init=3,reg_covar=1e-4).fit(x[:,None])
            scale=np.sqrt(fit.covariances_.ravel()).clip(.051,4.99)
            params['priors'][kind][str(p)]=dict(weights=fit.weights_.tolist(),means=fit.means_.ravel().tolist(),
                raw_scales=np.log(np.expm1(scale-.05)).tolist(),positions=int((valid&(phase==p)).sum()),fitted_positions=len(x))
    write(OUT/'parameters.json',params);write(DOCS/'parameters.json',params)
    paths=[Path(__file__),ROOT/'benchmarks/argn_time_density.py',ROOT/'tests/test_argn_time_density.py',DOCS/'PROTOCOL.md',
        OUT/'parameters.json',ROOT/'scripts/run_argn_boundary_gap.py',ROOT/'scripts/run_argn_amount_learning.py',
        ROOT/'benchmarks/argn_onset_output.py',ROOT/'benchmarks/argn_phase_gap.py',ROOT/'benchmarks/argn_past_state.py',
        ROOT/'docs/argn_joint_preservation_v1/MODEL_REGISTRY.json',BASE/'prepared/MANIFEST.json']
    paths += [META/f'metadata_{s}.parquet' for s in ['optimization','internal_validation','development']]
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',record);write(DOCS/'MANIFEST.json',record);print('DENSITY_PREPARED',flush=True)


def cached_inputs(cache,meta,metadata,codec,keys,params,device):
    assert np.array_equal(meta.record.to_numpy(),metadata.entity_id.to_numpy())
    assert np.array_equal(meta.event_index.to_numpy(),metadata.event_index.to_numpy())
    assert np.array_equal(meta.label.to_numpy(),metadata.label.to_numpy())
    y=codec.numeric({k:cache['targets'][:,j].numpy() for j,k in enumerate(keys)},'gap')
    keep=metadata.event_index.gt(0).to_numpy()
    np.testing.assert_allclose(y[keep],metadata.gap.to_numpy()[keep],rtol=0,atol=0)
    clock,valid=clock_features(metadata,params['fallback'])
    phase=2*metadata.previous.clip(lower=0).to_numpy()+metadata.label.to_numpy()
    data=dict(base=cache['base'][keep],phase=torch.as_tensor(phase[keep],dtype=torch.long),
        clock=torch.as_tensor(clock[keep],dtype=torch.float32),valid=torch.as_tensor(valid[keep]),
        gap=torch.as_tensor(y[keep],dtype=torch.float32))
    return {k:v.to(device) for k,v in data.items()},metadata.loc[keep].reset_index(drop=True)


def loss(head,data,ix,params):
    return head.nll(*(data[k][ix] for k in ['base','phase','clock','valid','gap']),params['limits'],params['quantum'])


@torch.no_grad()
def score(head,data,params):
    head.eval();sums=torch.zeros(4,device=data['base'].device,dtype=torch.float64);counts=torch.zeros_like(sums)
    for lo in range(0,len(data['gap']),8192):
        ix=slice(lo,lo+8192);ls=loss(head,data,ix,params);phase=data['phase'][ix]
        for p in range(4):sums[p]+=ls[phase.eq(p)].sum();counts[p]+=phase.eq(p).sum()
    values=(sums/counts).cpu().tolist();return float(np.mean(values)),values


def fit(data,keys,params,fs,arm,run):
    tr,val=data['optimization'],data['internal_validation'];device=tr['base'].device
    seed(fs+400);head=TimeDensity(tr['base'].shape[-1],arm,CFG['components'],CFG['hidden']).to(device)
    with torch.no_grad():
        head.center.copy_(tr['base'].mean(0));head.scale.copy_(tr['base'].std(0).clamp(min=.05))
        head.clock_center.copy_(tr['clock'].mean());head.clock_scale.copy_(tr['clock'].std().clamp(min=.05))
        for p,prior in params['priors']['relative' if arm=='history_relative' else 'absolute'].items():
            head.prior_logits[int(p)]=torch.tensor(prior['weights'],device=device).log()
            head.prior_mean[int(p)]=torch.tensor(prior['means'],device=device)
            head.prior_raw_scale[int(p)]=torch.tensor(prior['raw_scales'],device=device)
    pools=[torch.nonzero(tr['phase'].eq(p)).flatten() for p in range(4)]
    opt=torch.optim.AdamW(head.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    best,byphase=score(head,val,params);initial=best;best_state=deepcopy(head.state_dict());best_step=0;stale=0
    curve=[dict(step=0,macro_validation_nll=best,**{f'phase_{i}':v for i,v in enumerate(byphase)})]
    for step in range(1,CFG['max_steps']+1):
        head.train();ix=torch.cat([pool[torch.randint(len(pool),(CFG['batch_size']//4,),device=device)] for pool in pools])
        nll=loss(head,tr,ix,params).mean();assert torch.isfinite(nll)
        opt.zero_grad(set_to_none=True);nll.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),CFG['gradient_clip'],error_if_nonfinite=True);opt.step()
        if step%CFG['validation_every']:continue
        value,byphase=score(head,val,params);curve.append(dict(step=step,macro_validation_nll=value,**{f'phase_{i}':v for i,v in enumerate(byphase)}))
        if value<best-CFG['min_delta']:best=value;best_state=deepcopy(head.state_dict());best_step=step;stale=0
        else:stale+=1
        if stale>=CFG['patience']:break
    head.load_state_dict(best_state);head.eval().requires_grad_(False)
    payload=dict(dim=head.dim,arm=arm,components=head.components,hidden=head.hidden,
        state_dict={k:v.cpu() for k,v in head.state_dict().items()},fallback=params['fallback'],limits=params['limits'],quantum=params['quantum'],keys=keys)
    torch.save(payload,run/'time_head.pt');pd.DataFrame(curve).to_csv(run/'learning_curve.csv',index=False)
    record=dict(arm=arm,fit_seed=fs,selected_step=best_step,steps=step,initial_nll=initial,best_nll=best,
        train_positions=[len(x) for x in pools],validation_positions=[int(val['phase'].eq(p).sum()) for p in range(4)],
        parameters=sum(p.numel() for p in head.parameters()),sha256=digest(run/'time_head.pt'))
    write(run/'FIT_COMPLETE.json',record);write(DOCS/f'fits/{arm}_{fs}.json',record)
    print('DENSITY_FIT',fs,arm,best_step,initial,best,flush=True);return head


@torch.no_grad()
def conditional(head,data,meta,params,fs,arm,split):
    mean_nll,phase_nll=score(head,data,params);rows=[]
    for gs in CFG['sampling_seeds']:
        rng=np.random.default_rng(gs);values=[];clipped=0
        for lo in range(0,len(data['gap']),8192):
            ix=slice(lo,lo+8192);v,c=head.sample(*(data[k][ix] for k in ['base','phase','clock','valid']),rng,params['limits'],params['quantum']);values.append(v);clipped+=c
        y=np.concatenate(values);truth=meta.gap.to_numpy()
        for phase in ['onset','return_normal','normal_stay','continuation','left_fraud']:
            keep=meta.phase.eq(phase).to_numpy();rel=keep&meta.past_gap.gt(0).to_numpy();past=meta.past_gap.to_numpy()
            rows.append(dict(arm=arm,fit_seed=fs,split=split,sampling_seed=gs,phase=phase,events=int(keep.sum()),
                gap_w1=w1(truth[keep],y[keep]),personal_gap_w1=w1(truth[rel]/past[rel],y[rel]/past[rel]),
                real_median=float(np.median(truth[keep])),generated_median=float(np.median(y[keep])),
                macro_nll=mean_nll,clipped_fraction=clipped/len(y)))
    return rows


@torch.no_grad()
def integration(ws,native,records,metas,codec,fs,head_path,run):
    import mostlyai.engine._tabular.generation as generation
    from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX
    device=native.device;c=study.context_cache(native,records,metas)['x'][:4]
    args,kwargs=generation_args(ws,fs);ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read();lk=codec.prefixes['event_is_fraud']+'__cat'
    with time_density_generation(*args,**kwargs,time_payload=head_path):
        model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
            model_size=ws.model_tabular_configs.read()['model_units'],column_order=None,device=device)
        assert not any(k.startswith('time_density.') for k in model.state_dict())
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=device);model.to(device).eval()
        expected=model.clock_codec.empty(4);history=state=None;structural={};checks=0;first_checked=0;captured={};outputs={}
        def pre(module,a):
            if a[1] in model.time_keys:captured[a[1]]=torch.cat(a[0],-1)
        def post(module,a,out):
            if a[1] in model.time_keys:outputs[a[1]]=out
        h1=model.regressors.register_forward_pre_hook(pre);h2=model.predictors.register_forward_hook(post)
        for step in range(10):
            if step==7:
                ix=torch.tensor([3,0],device=device);c=c[ix];history=history[ix];state=tuple(v[:,ix] for v in state)
                structural={k:v[ix] for k,v in structural.items()};expected=expected[[3,0]]
            labels=(torch.arange(len(c),device=device)+step)%2;labels=labels[:,None]
            output,history,state=model(None,mode='gen',batch_size=len(c),context=([c],[],[]),history=history,history_state=state,
                fixed_values={**structural,lk:torch.where(labels.eq(1),codec.codes['1'],codec.codes['0'])},fixed_probs=_translate_fixed_probs(_fix_rare_token_probs(ts),ts))
            want=model.clock_codec.features(expected)
            for i in [0,1]:np.testing.assert_allclose(model.current_clock[i],want[i],rtol=0,atol=0)
            if step==0:
                for key,x in captured.items():
                    torch.testing.assert_close(outputs[key],model.amount_heads(x,key,labels),rtol=0,atol=0);first_checked+=len(c)
            else:
                for key in model.time_keys:
                    torch.testing.assert_close(output[key].reshape(-1).cpu(),torch.as_tensor(model.time_tokens[key]),rtol=0,atol=0);checks+=len(c)
            event={k:v.cpu().numpy() for k,v in output.items() if k.startswith(codec.prefixes['gap']+'__')}
            expected=model.clock_codec.advance(expected,codec.numeric(event,'gap'))
            np.testing.assert_allclose(state[4][0].cpu().numpy(),expected,rtol=0,atol=0,equal_nan=True)
            structural={k:v for k,v in output.items() if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
        h1.remove();h2.remove()
    write(run/'ROUTING_CHECK.json',dict(first_native_logits=first_checked,sampled_digit_positions=checks,steps=10,
        strict_past_clock_exact=True,reordered_compacted_batch=True,strict_native_checkpoint=True))


def generate_arm(ws,fs,arm,head_path,device):
    from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    before=digest(Workspace(run/'workspace').model_tabular_weights_path);context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    args,kwargs=generation_args(ws,fs)
    with time_density_generation(*args,**kwargs,time_payload=head_path) as audit:
        for gs in CFG['generation_seeds']:
            seed(gs);start=time.monotonic();prior=len(audit);generate(ctx_data=context,device=str(device),workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            d=raw.rename(columns={'customer_id':'entity_id'});d['event_index']=d.groupby('entity_id',sort=False).cumcount();d.loc[d.event_index.eq(0),'gap']=np.nan
            path=run/f'generated_validation_{gs}.parquet';d.to_parquet(path,index=False)
            base=pd.read_parquet(generation_path('onset_fit',fs,gs));cols=['entity_id','event_index','event_is_fraud'];assert d[cols].equals(base[cols]),'changed fixed label/length sequence'
            assert digest(Workspace(run/'workspace').model_tabular_weights_path)==before
            write(run/f'GENERATION_{gs}.json',dict(rows=len(d),seconds=time.monotonic()-start,sha256=digest(path),
                exact_labels_lengths=True,head_sha256=digest(head_path),sampling_audit=audit[prior:]))
            print('DENSITY_GENERATED',fs,arm,gs,len(d),flush=True)
    saved=evaluation.DOCS,evaluation.CONTROL;evaluation.DOCS=DOCS;evaluation.CONTROL={**saved[1],'generation_seeds':CFG['generation_seeds']}
    try:evaluation.evaluate(run,arm,fs)
    finally:evaluation.DOCS,evaluation.CONTROL=saved
    write(run/'COMPLETE.json',dict(generations=4,exact_labels_lengths=True,test_events_read=False))


def worker(fs,device):
    verify();start=time.monotonic();device=torch.device(device);assert device.type=='cuda' and torch.cuda.is_available()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in CFG['allowed_gpu_uuids']
    seed(fs);wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(device)))
    params=json.loads((OUT/'parameters.json').read_text());_,codec,frames,metas=inputs();ws,model=load_parent(fs,device)
    keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['gap']+'__')]
    s=ws.tgt_stats.read()['columns']['gap'];assert params['limits']==[min(s['min5']),max(s['max5'])] and params['quantum']==10.**s['min_decimal']
    data={};metadata={};checks={}
    for split in frames:
        cache,checks[split]=build_cache(model,frames[split],metas[split],codec,keys,device)
        data[split],metadata[split]=cached_inputs(cache,metas[split],pd.read_parquet(META/f'metadata_{split}.parquet'),codec,keys,params,device)
        print('DENSITY_CACHE',fs,split,len(data[split]['gap']),flush=True)
    del cache;write(wd/'CACHE_CHECK.json',checks);rows=[]
    for arm in CFG['arms']:
        run=wd/arm;run.mkdir();head=fit(data,keys,params,fs,arm,run)
        for split in data:rows+=conditional(head,data[split],metadata[split],params,fs,arm,split)
        integration(ws,model,frames['development'],metas['development'],codec,fs,run/'time_head.pt',run)
        pd.DataFrame(rows).to_csv(DOCS/f'conditional_{fs}.csv',index=False)
    del data,metadata,frames,metas,head,model;torch.cuda.empty_cache()
    for arm in CFG['arms']:generate_arm(ws,fs,arm,wd/arm/'time_head.pt',device)
    verify();write(wd/'COMPLETE.json',dict(fits=3,generations=12,seconds=time.monotonic()-start,test_events_read=False));print('DENSITY_COMPLETE',fs,flush=True)


def dispatch():
    from dispatch_argn_label_first_control import gpu_snapshot
    verify();assert not (OUT/'LAUNCH.json').exists();(OUT/'logs').mkdir(exist_ok=True)
    pending=list(CFG['fit_seeds']);running={};finished={};launched={};idle={}
    while pending or running:
        for fs,job in list(running.items()):
            code=job['process'].poll()
            if code is not None:finished[fs]=dict(pid=job['process'].pid,exit_code=code);del running[fs]
        for gpu in gpu_snapshot():
            uuid=gpu['uuid'];occupied=any(v['gpu']==uuid for v in running.values())
            free=uuid in CFG['allowed_gpu_uuids'] and gpu['memory_mib']<500 and gpu['utilization']<=5 and not gpu['has_compute_process'] and not occupied
            idle[uuid]=idle.get(uuid,0)+1 if free else 0
            if not pending or idle[uuid]<2:continue
            now=next(g for g in gpu_snapshot() if g['uuid']==uuid)
            if now['memory_mib']>=500 or now['utilization']>5 or now['has_compute_process']:continue
            fs=pending.pop(0);cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--seed',str(fs),'--device','cuda:0']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (OUT/'logs'/f'{fs}.log').open('x') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[fs]=dict(process=proc,gpu=uuid);launched[fs]=dict(pid=proc.pid,gpu=uuid,command=cmd,started_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'LAUNCH.json',launched);write(DOCS/'LAUNCH.json',launched);print('DENSITY_LAUNCHED',fs,proc.pid,gpu['index'],flush=True)
        write(OUT/'STATUS.json',dict(pending=pending,running={k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in running.items()},finished=finished))
        if pending or running:time.sleep(5)
    result=dict(success=len(finished)==2 and all(x['exit_code']==0 for x in finished.values()),jobs=finished,completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'DISPATCH_COMPLETE.json',result)
    assert result['success'],result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','dispatch']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    prepare() if a.mode=='prepare' else worker(a.seed,a.device) if a.mode=='worker' else dispatch()
