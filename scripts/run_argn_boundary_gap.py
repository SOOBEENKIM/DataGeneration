"""Equal-capacity gap learning control, boundary routing, and free generation."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from run_argn_amount_learning import (ROOT,BASE,inputs,load_parent,build_cache,AmountHeads,row_nll,seed,digest,write,
    Workspace,folder,get_cardinalities,get_ctx_sequence_length,load_model_weights,_sample,_translate_fixed_probs,_fix_rare_token_probs)
from diagnose_argn_residual import verify_registry,get_head,generation_path,w1
import run_argn_joint_preservation as study
from benchmarks.argn_boundary_gap import boundary_gap_generation,boundary_mask

OUT=ROOT/'artifacts/argn_boundary_gap_v1';DOCS=ROOT/'docs/argn_boundary_gap_v1'
CFG=dict(fit_seeds=[20260930,20261001],generation_seeds=[20261011,20261012],
    sampling_seeds=[20261111,20261112,20261113,20261114],arms=['all_label_fit','boundary_fit'],
    learning_rate=1e-4,weight_decay=.001,batch_size=256,max_steps=1000,validation_every=50,patience=5,
    min_delta=1e-5,gradient_clip=1.,allowed_gpu_uuids=['GPU-f1d4c556-ae70-3415-adec-c1d68f9bb637','GPU-5eae2fe0-7b63-fcc9-dd31-4e915ab63073'])


def verify():
    verify_registry();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    verify_registry();assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),ROOT/'benchmarks/argn_boundary_gap.py',ROOT/'tests/test_argn_boundary_gap.py',
        ROOT/'scripts/dispatch_argn_boundary_gap.py',DOCS/'PROTOCOL.md',ROOT/'docs/argn_residual_v1/oracle_phases.csv',
        ROOT/'docs/argn_residual_v1/rollout_phases.csv',ROOT/'docs/argn_joint_preservation_v1/MODEL_REGISTRY.json',
        ROOT/'benchmarks/argn_onset_output.py',ROOT/'scripts/run_argn_amount_learning.py',ROOT/'scripts/diagnose_argn_residual.py']
    m=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('BOUNDARY_PREPARED',flush=True)


@torch.no_grad()
def score(head,model,c,keys):
    head.eval();loss=row_nll(head,model,c,keys,torch.arange(len(c['labels']),device=model.device))
    return float(torch.stack([loss[c['labels'].eq(y)].mean() for y in [0,1]]).mean())


def fit(model,caches,metas,keys,payload,arm,fs,run):
    seed(fs+300);head=get_head(model,keys,payload).requires_grad_(True)
    tr=caches['optimization'];m=metas['optimization']
    mask=m.event_index.gt(0)
    if arm=='boundary_fit':mask&=m.previous_label.ne(m.label)
    pools=[torch.as_tensor(np.flatnonzero(mask&m.label.eq(y)),device=model.device) for y in [0,1]]
    vmask=metas['internal_validation'].event_index.gt(0)&metas['internal_validation'].previous_label.ne(metas['internal_validation'].label)
    vix=torch.as_tensor(np.flatnonzero(vmask),device=model.device)
    val={k:v[vix] for k,v in caches['internal_validation'].items()}
    opt=torch.optim.AdamW(head.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    best=score(head,model,val,keys);initial=best;best_state=deepcopy(head.state_dict());best_step=0;stale=0;curve=[dict(step=0,boundary_validation_nll=best)]
    for step in range(1,CFG['max_steps']+1):
        head.train();ix=torch.cat([pool[torch.randint(len(pool),(CFG['batch_size']//2,),device=model.device)] for pool in pools])
        loss=row_nll(head,model,tr,keys,ix).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),CFG['gradient_clip'],error_if_nonfinite=True);opt.step()
        if step%CFG['validation_every']:continue
        nll=score(head,model,val,keys);curve.append(dict(step=step,boundary_validation_nll=nll))
        if nll<best-CFG['min_delta']:best=nll;best_state=deepcopy(head.state_dict());best_step=step;stale=0
        else:stale+=1
        if stale>=CFG['patience']:break
    head.load_state_dict(best_state);head.eval().requires_grad_(False)
    torch.save(dict(keys=keys,kind='shared',state_dict={k:v.cpu() for k,v in head.state_dict().items()}),run/'boundary_head.pt')
    pd.DataFrame(curve).to_csv(run/'learning_curve.csv',index=False)
    write(run/'FIT_COMPLETE.json',dict(arm=arm,fit_seed=fs,selected_step=best_step,steps=step,initial_nll=initial,best_nll=best,
        train_positions=[len(x) for x in pools],validation_positions=len(vix),parameters=sum(p.numel() for p in head.parameters()),sha256=digest(run/'boundary_head.pt')))
    print('BOUNDARY_FIT',fs,arm,best_step,initial,best,flush=True);return head


@torch.no_grad()
def conditional(head,model,c,meta,codec,keys,fs,arm,split,run,masks):
    selected=np.flatnonzero(meta.event_index.gt(0)&meta.previous_label.ne(meta.label));rows=[]
    for gs in CFG['sampling_seeds']:
        seed(gs);ix=torch.as_tensor(selected,device=model.device);labels=c['labels'][ix];parts=[c['base'][ix]];tokens={}
        for key in keys:
            v=_sample(head(torch.cat(parts,-1),key,labels).softmax(-1),fixed_probs=masks.get(key)).reshape(-1)
            tokens[key]=v.cpu().numpy();parts.append(model.embedders.get(key)(v))
        d=meta.iloc[selected][['record','event_index','label','previous_label']].copy();d['sampling_seed']=gs
        d['real_gap']=codec.numeric({k:c['targets'][ix,j].cpu().numpy() for j,k in enumerate(keys)},'gap');d['generated_gap']=codec.numeric(tokens,'gap');rows.append(d)
    data=pd.concat(rows,ignore_index=True);data.to_parquet(run/f'conditional_{split}.parquet',index=False)
    return [dict(fit_seed=fs,arm=arm,split=split,transition='onset' if y else 'return_normal',sampling_seed=gs,
        events=len(g),real_median=g.real_gap.median(),generated_median=g.generated_gap.median(),log_w1=w1(g.real_gap,g.generated_gap))
        for (y,gs),g in data.groupby(['label','sampling_seed'])]


def generation_args(ws,fs):
    old=json.loads(study.OLD.read_text());ep=json.loads((study.PRIOR/'episode_parameters.json').read_text());hazard=json.loads((study.OUT/'hazard_parameters.json').read_text())
    reg=verify_registry();c=next(x['components'] for x in reg['models'] if x['fit_seed']==fs)
    return (ws.tgt_stats.read(),old,c['frozen_category_gap_amount']['path'],ep,hazard,c['joint_count_initial_state']['path'],True),dict(onset_payload=c['onset_amount_expert']['path'])


@torch.no_grad()
def integration(ws,native,records,metas,codec,fs,head_path,run):
    import mostlyai.engine._tabular.generation as generation
    from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX
    device=native.device;c=study.context_cache(native,records,metas)['x'][:4]
    args,kwargs=generation_args(ws,fs);ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read();lk=codec.prefixes['event_is_fraud']+'__cat'
    with boundary_gap_generation(*args,**kwargs,boundary_payload=head_path):
        model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
            model_size=ws.model_tabular_configs.read()['model_units'],column_order=None,device=device)
        assert not any(k.startswith('boundary_head.') for k in model.state_dict())
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=device);model.to(device).eval()
        captured={};logits={}
        def pre(module,a):
            if a[1] in model.boundary_head.keys:captured[a[1]]=torch.cat(a[0],-1)
        def post(module,a,out):
            if a[1] in model.boundary_head.keys:logits[a[1]]=out
        h1=model.regressors.register_forward_pre_hook(pre);h2=model.predictors.register_forward_hook(post)
        history=state=None;structural={};checks=0
        for step,values in enumerate([[0,0,1,1],[1,0,1,0],[1,1]]):
            if step==2:
                ix=torch.tensor([3,0],device=device);c=c[ix];history=history[ix];state=tuple(v[:,ix] for v in state);structural={k:v[ix] for k,v in structural.items()}
            labels=torch.tensor(values,device=device)[:,None]
            output,history,state=model(None,mode='gen',batch_size=len(c),context=([c],[],[]),history=history,history_state=state,
                fixed_values={**structural,lk:torch.where(labels.eq(1),codec.codes['1'],codec.codes['0'])},fixed_probs=_translate_fixed_probs(_fix_rare_token_probs(ts),ts))
            expected=torch.tensor([False]*4 if step==0 else [True,False,False,True] if step==1 else [True,False],device=device)[:,None,None]
            for k,x in captured.items():
                base=model.amount_heads(x,k,labels);adapted=model.boundary_head(x,k,labels)
                torch.testing.assert_close(logits[k],torch.where(expected,adapted,base),rtol=0,atol=0);checks+=len(c)
            structural={k:v for k,v in output.items() if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
        h1.remove();h2.remove()
    write(run/'ROUTING_CHECK.json',dict(exact_token_positions=checks,first_and_both_transitions=True,reordered_compacted_batch=True))


def generate_arm(ws,fs,arm,head_path,device):
    from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    before=digest(Workspace(run/'workspace').model_tabular_weights_path);context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    args,kwargs=generation_args(ws,fs)
    with boundary_gap_generation(*args,**kwargs,boundary_payload=head_path):
        for gs in CFG['generation_seeds']:
            seed(gs);started=time.monotonic();generate(ctx_data=context,device=str(device),workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            d=raw.rename(columns={'customer_id':'entity_id'});d['event_index']=d.groupby('entity_id',sort=False).cumcount();d.loc[d.event_index.eq(0),'gap']=np.nan
            path=run/f'generated_validation_{gs}.parquet';d.to_parquet(path,index=False)
            base=pd.read_parquet(generation_path('onset_fit',fs,gs));cols=['entity_id','event_index','event_is_fraud']
            assert d[cols].equals(base[cols]),'changed fixed label/length sequence'
            assert digest(Workspace(run/'workspace').model_tabular_weights_path)==before
            write(run/f'GENERATION_{gs}.json',dict(rows=len(d),seconds=time.monotonic()-started,sha256=digest(path),exact_labels_lengths=True,head_sha256=digest(head_path)))
            print('BOUNDARY_GENERATED',fs,arm,gs,len(d),flush=True)
    saved=evaluation.DOCS,evaluation.CONTROL;evaluation.DOCS=DOCS;evaluation.CONTROL={**saved[1],'generation_seeds':CFG['generation_seeds']}
    try:evaluation.evaluate(run,arm,fs)
    finally:evaluation.DOCS,evaluation.CONTROL=saved
    write(run/'COMPLETE.json',dict(generations=len(CFG['generation_seeds']),exact_labels_lengths=True,test_events_read=False))


def worker(fs,device):
    verify();start=time.monotonic();device=torch.device(device);seed(fs);wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    _,codec,frames,metas=inputs();ws,model=load_parent(fs,device);keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['gap']+'__')]
    payload=torch.load(study.foundation(fs)/'amount_head.pt',map_location='cpu',weights_only=True);caches={};checks={}
    for split in frames:
        cache,checks[split]=build_cache(model,frames[split],metas[split],codec,keys,device);caches[split]={k:v.to(device) for k,v in cache.items()}
    write(wd/'CACHE_CHECK.json',checks);masks=_translate_fixed_probs(_fix_rare_token_probs(ws.tgt_stats.read()),ws.tgt_stats.read());rows=[]
    frozen=wd/'frozen';frozen.mkdir();base=get_head(model,keys,payload)
    for s in frames:rows+=conditional(base,model,caches[s],metas[s],codec,keys,fs,'frozen',s,frozen,masks)
    for arm in CFG['arms']:
        run=wd/arm;run.mkdir();head=fit(model,caches,metas,keys,payload,arm,fs,run)
        for s in frames:rows+=conditional(head,model,caches[s],metas[s],codec,keys,fs,arm,s,run,masks)
        integration(ws,model,frames['development'],metas['development'],codec,fs,run/'boundary_head.pt',run)
    pd.DataFrame(rows).to_csv(DOCS/f'conditional_{fs}.csv',index=False)
    del caches,cache,frames,metas,model,head,base;torch.cuda.empty_cache()
    for arm in CFG['arms']:generate_arm(ws,fs,arm,wd/arm/'boundary_head.pt',device)
    verify();write(wd/'COMPLETE.json',dict(fits=2,generations=4,seconds=time.monotonic()-start,test_events_read=False));print('BOUNDARY_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--device',default='cpu');a=p.parse_args()
    prepare() if a.mode=='prepare' else worker(a.seed,a.device)
