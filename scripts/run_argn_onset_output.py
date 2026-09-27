"""Targeted rare-onset output adaptation with an equal-capacity training control."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
import run_argn_joint_preservation as study
from run_argn_amount_learning import load_parent,AmountHeads,row_nll,head_inputs,_sample,_translate_fixed_probs,_fix_rare_token_probs
from run_argn_joint_preservation import ROOT,seed,digest,write,inputs
from benchmarks.argn_onset_output import onset_output_generation

OUT=study.OUT/'onset_output';DOCS=study.DOCS/'onset_output'
CONFIG=ROOT/'configs/argn_onset_output_v1.json';CFG=json.loads(CONFIG.read_text())


def verify():
    study.verify();m=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    study.verify();assert not (OUT/'MANIFEST.json').exists()
    paths=[CONFIG,Path(__file__),ROOT/'scripts/dispatch_argn_onset_output.py',ROOT/'benchmarks/argn_onset_output.py',
           ROOT/'tests/test_argn_onset_output.py',DOCS/'PROTOCOL.md',study.OUT/'MANIFEST.json',
           study.DOCS/'ORACLE_PHASE_COMPLETE.json',ROOT/'scripts/run_argn_amount_learning.py']
    for fs in CFG['fit_seeds']:
        paths+=[ROOT/f'artifacts/argn_amount_learning_v1/worker_{fs}/{s}_cache.pt' for s in ['optimization','internal_validation','development']]
        paths += [ROOT/f'artifacts/argn_amount_learning_v1/runs/balanced_shared_{fs}/amount_head.pt',study.OUT/f'worker_{fs}/unconditional/count_head.pt']
    m=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('ONSET_OUTPUT_PREPARED',flush=True)


@torch.no_grad()
def score(head,model,cache,keys):
    head.eval();return float(row_nll(head,model,cache,keys,torch.arange(len(cache['labels']),device=model.device)).mean())


def fit(model,caches,keys,payload,fs,arm,run):
    seed(fs+100);head=AmountHeads(model,keys).to(model.device);head.load_state_dict(payload['state_dict'])
    train=caches['optimization']['onset' if arm=='onset_fit' else 'fraud'];val=caches['internal_validation']['onset']
    optimizer=torch.optim.AdamW(head.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    best=score(head,model,val,keys);initial=best;selected=0;stale=0;best_state=deepcopy(head.state_dict());rows=[]
    rows.append(dict(step=0,onset_validation_nll=best))
    for step in range(1,CFG['max_steps']+1):
        head.train();ix=torch.randint(len(train['labels']),(CFG['batch_size'],),device=model.device)
        loss=row_nll(head,model,train,keys,ix).mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),CFG['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        if step%CFG['validation_every']:continue
        nll=score(head,model,val,keys);rows.append(dict(step=step,onset_validation_nll=nll))
        if nll<best-CFG['min_delta']:best=nll;best_state=deepcopy(head.state_dict());selected=step;stale=0
        else:stale+=1
        if stale>=CFG['patience']:break
    head.load_state_dict(best_state);head.eval()
    torch.save(dict(keys=keys,kind='shared',state_dict={k:v.cpu() for k,v in head.state_dict().items()}),run/'onset_head.pt')
    pd.DataFrame(rows).to_csv(run/'learning_curve.csv',index=False)
    evidence=dict(arm=arm,fit_seed=fs,initial_validation_nll=initial,best_validation_nll=best,selected_step=selected,last_step=step,
        train_events=len(train['labels']),validation_events=len(val['labels']),parameters=sum(p.numel() for p in head.parameters()),head_sha256=digest(run/'onset_head.pt'))
    write(run/'FIT_COMPLETE.json',evidence);print('ONSET_FIT_COMPLETE',evidence,flush=True)
    return head


@torch.no_grad()
def conditional(head,model,cache,keys,codec,fs,arm,split,masks):
    values=[];ix=torch.arange(len(cache['labels']),device=model.device)
    for gs in CFG['sampling_seeds']:
        seed(gs);parts=[cache['base']];tokens={}
        for key in keys:
            y=_sample(head(torch.cat(parts,-1),key,cache['labels']).softmax(-1),fixed_probs=masks.get(key)).reshape(-1)
            tokens[key]=y.cpu().numpy();parts.append(model.embedders.get(key)(y))
        values.append(codec.numeric(tokens,'amount_or_numeric_value'))
    truth=codec.numeric({k:cache['targets'][:,i].cpu().numpy() for i,k in enumerate(keys)},'amount_or_numeric_value')
    synthetic=np.concatenate(values)
    return dict(arm=arm,fit_seed=fs,split=split,events=len(truth),actual_prefix_onset_nll=score(head,model,cache,keys),
        real_median=np.median(truth),generated_median=np.median(synthetic),amount_log_w1=wasserstein_distance(np.log1p(truth),np.log1p(synthetic)))


def worker(fs,device):
    verify();device=torch.device(device);start=time.monotonic();wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    _,codec,records,metas=inputs();del records
    ws,model=load_parent(fs,device)
    payload=torch.load(ROOT/f'artifacts/argn_amount_learning_v1/runs/balanced_shared_{fs}/amount_head.pt',weights_only=True,map_location=device)
    keys=payload['keys'];caches={}
    combined=torch.load(study.foundation(fs)/'amount_head.pt',weights_only=True,map_location='cpu')
    for k,v in payload['state_dict'].items():assert torch.equal(v.cpu(),combined['state_dict'][k])
    del combined
    for split,m in metas.items():
        raw=torch.load(ROOT/f'artifacts/argn_amount_learning_v1/worker_{fs}/{split}_cache.pt',map_location='cpu',weights_only=True,mmap=True)
        assert np.array_equal(raw['labels'].numpy(),m.label.to_numpy())
        caches[split]={}
        for group,mask in [('fraud',m.label.eq(1)),('onset',m.label.eq(1)&m.previous_label.eq(0))]:
            ix=torch.as_tensor(np.flatnonzero(mask));caches[split][group]={k:v[ix].to(device) for k,v in raw.items()}
        del raw
    base=AmountHeads(model,keys).to(device);base.load_state_dict(payload['state_dict']);base.eval()
    masks=_translate_fixed_probs(_fix_rare_token_probs(ws.tgt_stats.read()),ws.tgt_stats.read())
    metrics=[conditional(base,model,caches[s]['onset'],keys,codec,fs,'frozen',s,masks) for s in caches]
    for arm in CFG['arms']:
        run=wd/arm;run.mkdir();write(wd/'STATUS.json',dict(stage='onset_training',arm=arm))
        trained=fit(model,caches,keys,payload,fs,arm,run)
        metrics += [conditional(trained,model,caches[s]['onset'],keys,codec,fs,arm,s,masks) for s in caches]
        pd.DataFrame(metrics).to_csv(DOCS/f'conditional_{fs}.csv',index=False)
    del model,caches,base,trained,payload;torch.cuda.empty_cache()
    for arm in CFG['arms']:
        run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
        write(wd/'STATUS.json',dict(stage='free_generation',arm=arm))
        count=study.OUT/f'worker_{fs}/unconditional/count_head.pt';head=wd/arm/'onset_head.pt'
        original_manager,original_config,original_docs=study.preservation_generation,study.CFG,study.DOCS
        def manager(*args,**kwargs):return onset_output_generation(*args,onset_payload=head,**kwargs)
        study.preservation_generation=manager;study.DOCS=DOCS
        study.CFG={**original_config,'generation_arms':{arm:original_config['generation_arms']['count_hazard_no_context']}}
        try:study.free_generate(ws,run,fs,arm,str(device),count)
        finally:study.preservation_generation,study.CFG,study.DOCS=original_manager,original_config,original_docs
        for gs in CFG['generation_seeds']:
            baseline=pd.read_parquet(study.OUT/f'runs/count_hazard_no_context_{fs}/generated_validation_{gs}.parquet')
            generated=pd.read_parquet(run/f'generated_validation_{gs}.parquet')
            assert generated[['entity_id','event_index','event_is_fraud']].equals(baseline[['entity_id','event_index','event_is_fraud']])
        write(run/'COMPLETE.json',dict(arm=arm,fit_seed=fs,generated_datasets=2,exact_fixed_lengths_and_labels=True,
              onset_head_sha256=digest(head),test_events_read=False))
    verify();write(wd/'STATUS.json',dict(stage='complete'));write(wd/'COMPLETE.json',dict(fits=2,generations=4,seconds=time.monotonic()-start,test_events_read=False))
    print('ONSET_OUTPUT_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
