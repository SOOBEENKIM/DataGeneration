"""Freeze the recovered amount head; test upstream category learning factorially."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd
import torch

import run_argn_amount_learning as shared
from run_argn_amount_learning import ROOT, seed, digest, write, inputs, load_parent, build_cache, AmountHeads, Workspace, folder

OUT=ROOT/'artifacts/argn_category_amount_v1'
DOCS=ROOT/'docs/argn_category_amount_v1'
CONFIG=ROOT/'configs/argn_category_amount_v1.json'
CFG=json.loads(CONFIG.read_text())


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),CONFIG,DOCS/'PROTOCOL.md',ROOT/'scripts/dispatch_argn_category_amount.py',
        ROOT/'scripts/run_argn_amount_learning.py',ROOT/'benchmarks/argn_amount_control.py',
        ROOT/'scripts/run_argn_label_first_control.py',ROOT/'benchmarks/argn_transition_reference.py',
        ROOT/'artifacts/research_reaudit_20260927/simple_label_parameters.json',
        ROOT/'docs/argn_amount_learning_v1/condition_probe/MANIFEST.json']
    for fs in CFG['fit_seeds']:
        paths += [ROOT/f'artifacts/argn_amount_learning_v1/runs/balanced_shared_{fs}/amount_head.pt',
                  Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,
                  hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',manifest);write(DOCS/'MANIFEST.json',manifest)


def verify():
    m=json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h


@torch.no_grad()
def category_distribution(heads,cache,key,model):
    sums=torch.zeros((2,model.tgt_cardinalities[key]),device=model.device)
    actual=torch.zeros_like(sums);counts=torch.zeros(2,device=model.device)
    for lo in range(0,len(cache['labels']),8192):
        x=cache['base'][lo:lo+8192];y=cache['labels'][lo:lo+8192];target=cache['targets'][lo:lo+8192,0]
        probs=heads(x,key,y).softmax(-1)
        for label in [0,1]:
            mask=y.eq(label);sums[label]+=probs[mask].sum(0)
            actual[label]+=torch.bincount(target[mask],minlength=probs.shape[1]);counts[label]+=mask.sum()
    return [dict(label=label,events=int(counts[label]),
                 category_marginal_tv=float((sums[label]/counts[label]-actual[label]/counts[label]).abs().sum()/2)) for label in [0,1]]


def worker(fs,device):
    verify();seed(fs);device=torch.device(device);start=time.monotonic()
    wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    _,codec,frames,metas=inputs();ws,model=load_parent(fs,device)
    category=codec.prefixes['category']+'__cat'
    amount=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['amount_or_numeric_value']+'__')]
    caches={};checks={}
    for split in ['optimization','internal_validation','development']:
        cache,checks[split]=build_cache(model,frames[split],metas[split],codec,[category],device)
        caches[split]={k:v.to(device) for k,v in cache.items()}
    write(wd/'CACHE_CHECK.json',dict(positions=checks,category_future_fields_masked=True))
    del frames
    shared.CFG={**shared.CFG,**CFG,'arms':{'balanced_category':{'kind':'shared','balanced':True}}}
    shared.DOCS=DOCS
    fit_dir=wd/'category_fit';fit_dir.mkdir()
    baseline=AmountHeads(model,[category]).eval()
    distributions=[]
    for split,cache in caches.items():
        distributions += [dict(model='parent',split=split,fit_seed=fs,**r)
                          for r in category_distribution(baseline,cache,category,model)]
    write(wd/'STATUS.json',dict(stage='category_training'))
    trained=shared.fit(model,caches,[category],'balanced_category',fs,fit_dir)
    for split,cache in caches.items():
        distributions += [dict(model='balanced_category',split=split,fit_seed=fs,**r)
                          for r in category_distribution(trained,cache,category,model)]
    pd.DataFrame(distributions).to_csv(DOCS/f'category_fit_{fs}.csv',index=False)
    amount_path=ROOT/f'artifacts/argn_amount_learning_v1/runs/balanced_shared_{fs}/amount_head.pt'
    amount_payload=torch.load(amount_path,map_location='cpu',weights_only=True)
    category_payload=torch.load(fit_dir/'amount_head.pt',map_location='cpu',weights_only=True)
    assert amount_payload['kind']==category_payload['kind']=='shared'
    for mode in CFG['generation_arms']:
        run=OUT/'runs'/f'{mode}_{fs}';run.mkdir(parents=True,exist_ok=False)
        keys=[category] if mode=='category_only' else [category]+amount
        state=category_payload['state_dict'].copy()
        if mode=='category_and_amount':
            assert not (set(state)&set(amount_payload['state_dict']))
            state.update(amount_payload['state_dict'])
        # Check exact module compatibility before native generation loads it.
        combined=AmountHeads(model,keys,'shared').to(device);combined.load_state_dict(state);combined.eval()
        torch.save(dict(keys=keys,kind='shared',state_dict={k:v.cpu() for k,v in state.items()},
            category_head_sha256=digest(fit_dir/'amount_head.pt'),
            amount_head_sha256=digest(amount_path) if mode=='category_and_amount' else 'native_parent'),run/'amount_head.pt')
        write(wd/'STATUS.json',dict(stage='free_generation',arm=mode))
        shared.free_generate(ws,run,fs,mode,str(device))
        write(run/'COMPLETE.json',dict(arm=mode,fit_seed=fs,generated_datasets=2,
            head_sha256=digest(run/'amount_head.pt'),test_events_read=False))
        del combined
    verify();write(wd/'STATUS.json',dict(stage='complete'))
    write(wd/'COMPLETE.json',dict(seconds=time.monotonic()-start,fit_seed=fs,fits=1,
        generated_datasets=4,cache_checks=checks,test_events_read=False))
    print('CATEGORY_AMOUNT_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
