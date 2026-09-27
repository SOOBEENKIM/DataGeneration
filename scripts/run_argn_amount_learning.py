"""Registered frozen-representation amount learning, sampling and free generation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch
from torch.nn import functional as F

from audit_argn_fit_generalization import inputs, folder
from run_argn_state_first import ROOT, OUT as BASE, seed, digest, write
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.argn import _sample
from mostlyai.engine._tabular.generation import _fix_rare_token_probs, _translate_fixed_probs
from benchmarks.argn_label_first_control import label_first_model_class
from benchmarks.argn_state_adapter import FullHistoryCollator
from benchmarks.argn_amount_control import AmountHeads, conditional_objective, amount_generation

OUT = ROOT / 'artifacts/argn_amount_learning_v1'
DOCS = ROOT / 'docs/argn_amount_learning_v1'
CONFIG = ROOT / 'configs/argn_amount_learning_v1.json'
CFG = json.loads(CONFIG.read_text())


def sources():
    return [CONFIG, Path(__file__), ROOT/'benchmarks/argn_amount_control.py',
            ROOT/'scripts/dispatch_argn_amount_learning.py',
            ROOT/'tests/test_argn_amount_control.py', DOCS/'PROTOCOL.md',
            ROOT/'benchmarks/argn_label_first_control.py', ROOT/'benchmarks/argn_state_adapter.py',
            ROOT/'benchmarks/argn_transition_reference.py', ROOT/'scripts/audit_argn_fit_generalization.py',
            ROOT/'scripts/run_argn_label_first_control.py',
            ROOT/'artifacts/research_reaudit_20260927/simple_label_parameters.json',
            BASE/'prepared/MANIFEST.json', BASE/'prepared/validation_context.parquet']


def prepare():
    assert not (OUT/'MANIFEST.json').exists(), 'never overwrite a registered run'
    paths = sources() + [Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path
                         for fs in CFG['fit_seeds']]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=CFG,
                    hashes={str(p): digest(p) for p in paths}, test_events_read=False)
    write(OUT/'MANIFEST.json', manifest); write(DOCS/'MANIFEST.json', manifest)


def verify():
    manifest = json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():
        assert digest(p) == h, f'source/input drift: {p}'
    return manifest


def load_parent(fs, device):
    ws = Workspace(folder('B_event_label_first',fs)/'workspace')
    ts, cs = ws.tgt_stats.read(), ws.ctx_stats.read()
    model = label_first_model_class(ts)(tgt_cardinalities=get_cardinalities(ts),
        ctx_cardinalities=get_cardinalities(cs), tgt_seq_len_median=ts['seq_len']['median'],
        tgt_seq_len_max=ts['seq_len']['max'], ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
        model_size=ws.model_tabular_configs.read()['model_units'], column_order=None, device=device)
    load_model_weights(model=model, path=ws.model_tabular_weights_path, device=device)
    return ws, model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def build_cache(model, records, meta, codec, keys, device):
    """Full real prefix through frozen encoder; no amount digits in common base."""
    collator = FullHistoryCollator(True,None,device)
    cloned = AmountHeads(model,keys).eval()
    base, targets, labels = [], [], []
    checked = 0
    for lo in range(0,len(records),2):
        batch = collator(records[lo:lo+2]); captured = {}
        def capture(module,args):
            if args[1] in keys:
                captured[args[1]] = args[0]
        handle = model.regressors.register_forward_pre_hook(capture)
        try:
            logits,_ = model(batch,mode='trn')
        finally:
            handle.remove()
        for key in keys:
            xs = torch.cat(captured[key],-1)
            replay = cloned(xs,key,torch.zeros(xs.shape[:-1],device=device,dtype=torch.long))
            torch.testing.assert_close(replay,logits[key],rtol=1e-5,atol=2e-5)
        common = torch.cat(captured[keys[0]][:2],-1)
        lk = codec.prefixes['event_is_fraud']+'__cat'
        for b,r in enumerate(records[lo:lo+2]):
            n = len(r[lk]); base.append(common[b,:n].cpu())
            targets.append(torch.cat([batch[k][b,:n] for k in keys],-1).cpu())
            labels.append(batch[lk][b,:n,0].eq(codec.codes['1']).long().cpu())
            checked += n*len(keys)
    result = dict(base=torch.cat(base), targets=torch.cat(targets), labels=torch.cat(labels))
    assert len(meta) == len(result['labels'])
    assert np.array_equal(meta.label.to_numpy(),result['labels'].numpy())
    return result, checked


def head_inputs(model,cache,keys,indices,generated=None):
    base = cache['base'][indices]
    tokens = cache['targets'][indices] if generated is None else generated
    parts = [base]
    for j,key in enumerate(keys):
        yield key, torch.cat(parts,-1)
        if j<len(keys)-1:
            parts.append(model.embedders.get(key)(tokens[:,j]))


def row_nll(heads, model, cache, keys, indices):
    labels = cache['labels'][indices]
    terms=[]
    for j,(key,x) in enumerate(head_inputs(model,cache,keys,indices)):
        terms.append(F.cross_entropy(heads(x,key,labels),cache['targets'][indices,j],reduction='none'))
    return torch.stack(terms,-1).sum(-1)


@torch.no_grad()
def evaluate_nll(heads,model,cache,keys):
    heads.eval(); sums=torch.zeros(2,device=model.device); counts=torch.zeros_like(sums)
    for lo in range(0,len(cache['labels']),8192):
        ix=torch.arange(lo,min(lo+8192,len(cache['labels'])),device=model.device)
        losses=row_nll(heads,model,cache,keys,ix);y=cache['labels'][ix]
        for label in [0,1]:
            sums[label]+=losses[y.eq(label)].sum();counts[label]+=y.eq(label).sum()
    assert counts.min()>0
    means=(sums/counts).cpu().tolist()
    return dict(normal_nll=means[0],fraud_nll=means[1],macro_nll=sum(means)/2,
                normal_events=int(counts[0]),fraud_events=int(counts[1]))


def fit(model,caches,keys,arm,fs,run):
    spec=CFG['arms'][arm];seed(fs+100)
    heads=AmountHeads(model,keys,spec['kind']).to(model.device)
    optimizer=torch.optim.Adam(heads.parameters(),lr=CFG['learning_rate'])
    train=caches['optimization'];val=caches['internal_validation']
    pools=[torch.where(train['labels'].eq(y))[0] for y in [0,1]]
    prior=float(train['labels'].float().mean());rng=torch.Generator(device=model.device).manual_seed(fs+200)
    best=float('inf');best_step=0;stale=0;rows=[];start=time.monotonic()
    for step in range(CFG['max_steps']+1):
        if step:
            heads.train()
            ix=torch.cat([pool[torch.randint(len(pool),(CFG['batch_size']//2,),device=model.device,generator=rng)] for pool in pools])
            losses=row_nll(heads,model,train,keys,ix)
            objective=conditional_objective(losses,train['labels'][ix],prior,spec['balanced'])
            optimizer.zero_grad(set_to_none=True);objective.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(),CFG['gradient_clip']);optimizer.step()
        if step%CFG['validation_every']:
            continue
        metrics=evaluate_nll(heads,model,val,keys)
        row=dict(step=step,seconds=time.monotonic()-start,**metrics,
                 training_objective=float(objective) if step else None)
        rows.append(row);pd.DataFrame(rows).to_csv(run/'learning_curve.csv',index=False)
        if metrics['macro_nll'] < best-CFG['min_delta']:
            best=metrics['macro_nll'];best_step=step;stale=0
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in heads.state_dict().items()},
                keys=keys,kind=spec['kind'],arm=arm,fit_seed=fs,selected_step=step),run/'amount_head.pt')
        else:
            stale+=1
        print('AMOUNT_FIT',arm,fs,step,metrics,flush=True)
        if stale>=CFG['patience']:
            break
    payload=torch.load(run/'amount_head.pt',map_location=model.device,weights_only=True)
    heads.load_state_dict(payload['state_dict']);heads.eval()
    result=dict(arm=arm,fit_seed=fs,selected_step=best_step,last_step=step,
        trainable_parameters=sum(p.numel() for p in heads.parameters()),seconds=time.monotonic()-start,
        validation=evaluate_nll(heads,model,val,keys),
        optimization=evaluate_nll(heads,model,train,keys),head_sha256=digest(run/'amount_head.pt'))
    write(run/'FIT_COMPLETE.json',result)
    return heads


@torch.no_grad()
def conditional_samples(heads,model,cache,meta,codec,keys,split,fs,arm,run,masks):
    rng=np.random.default_rng(CFG['selection_seed']);fraud=np.flatnonzero(meta.label.to_numpy()==1)
    selected=np.r_[fraud,rng.choice(np.flatnonzero(meta.label.to_numpy()==0),len(fraud),replace=False)]
    # Same customer/event selection for every arm and both parent fits.
    selected=np.sort(selected);rows=[]
    for gs in CFG['sampling_seeds']:
        seed(gs)
        for lo in range(0,len(selected),4096):
            chosen=selected[lo:lo+4096];ix=torch.as_tensor(chosen,device=model.device)
            y=cache['labels'][ix];base=cache['base'][ix];parts=[base];tokens={}
            for key in keys:
                logits=heads(torch.cat(parts,-1),key,y)
                value=_sample(logits.softmax(-1),fixed_probs=masks.get(key))
                value=value.reshape(-1);tokens[key]=value.cpu().numpy()
                parts.append(model.embedders.get(key)(value))
            synthetic=codec.numeric(tokens,'amount_or_numeric_value')
            truth=codec.numeric({k:cache['targets'][ix,j].cpu().numpy() for j,k in enumerate(keys)},'amount_or_numeric_value')
            row=meta.iloc[chosen][['record','event_index','label']].copy()
            row['real_amount']=truth;row['generated_amount']=synthetic;row['sampling_seed']=gs;rows.append(row)
    raw=pd.concat(rows,ignore_index=True);raw.to_parquet(run/f'conditional_{split}.parquet',index=False)
    results=[]
    for label,g in raw.groupby('label'):
        results.append(dict(arm=arm,fit_seed=fs,split=split,label=int(label),positions=len(g)//len(CFG['sampling_seeds']),
            real_median=float(g.real_amount.median()),generated_median=float(g.generated_amount.median()),
            real_p90=float(g.real_amount.quantile(.9)),generated_p90=float(g.generated_amount.quantile(.9)),
            log_w1=float(wasserstein_distance(np.log1p(g.real_amount),np.log1p(g.generated_amount)))))
    return results


def free_generate(ws,run,fs,arm,device):
    parameters=json.loads((ROOT/'artifacts/research_reaudit_20260927/simple_label_parameters.json').read_text())
    shutil.copytree(ws.workspace_dir if hasattr(ws,'workspace_dir') else folder('B_event_label_first',fs)/'workspace',
        run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    weight=Workspace(run/'workspace').model_tabular_weights_path;before=digest(weight)
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    with amount_generation(ws.tgt_stats.read(),parameters,run/'amount_head.pt'):
        for gs in CFG['generation_seeds']:
            seed(gs);started=time.monotonic();generate(ctx_data=context,device=device,workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            frame=raw.rename(columns={'customer_id':'entity_id'}).copy()
            frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
            frame.loc[frame.event_index.eq(0),'gap']=np.nan
            dest=run/f'generated_validation_{gs}.parquet';frame.to_parquet(dest,index=False)
            assert digest(weight)==before
            write(run/f'GENERATION_{gs}.json',dict(seconds=time.monotonic()-started,rows=len(frame),
                customers=frame.entity_id.nunique(),sha256=digest(dest),head_sha256=digest(run/'amount_head.pt')))
            print('AMOUNT_GENERATED',arm,fs,gs,len(frame),flush=True)
    import run_argn_label_first_control as evaluation
    saved=evaluation.DOCS;evaluation.DOCS=DOCS
    try:evaluation.evaluate(run,arm,fs)
    finally:evaluation.DOCS=saved


def worker(fs,device):
    manifest=verify();started=time.monotonic();seed(fs);device=torch.device(device)
    worker_dir=OUT/f'worker_{fs}';worker_dir.mkdir(parents=True,exist_ok=False)
    write(worker_dir/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
          device=str(device),started_utc=datetime.now(timezone.utc).isoformat()))
    shared,codec,frames,metas=inputs();ws,model=load_parent(fs,device)
    assert ws.tgt_stats.read()==shared.tgt_stats.read()
    keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['amount_or_numeric_value']+'__')]
    caches={};checked={}
    for split in ['optimization','internal_validation','development']:
        cache,checked[split]=build_cache(model,frames[split],metas[split],codec,keys,device)
        torch.save(cache,worker_dir/f'{split}_cache.pt')
        caches[split]={k:v.to(device) for k,v in cache.items()}
        print('AMOUNT_CACHE',fs,split,len(cache['labels']),cache['base'].shape,flush=True)
    write(worker_dir/'CACHE_COMPLETE.json',dict(replayed_digit_positions=checked,
        hashes={s:digest(worker_dir/f'{s}_cache.pt') for s in caches},
        frozen_representation=True,test_events_read=False))
    del frames
    masks=_translate_fixed_probs(_fix_rare_token_probs(ws.tgt_stats.read()),ws.tgt_stats.read())
    results=[]
    # Parent conditional generation with exactly the same sampling code/seeds.
    parent_run=worker_dir/'parent';parent_run.mkdir()
    baseline=AmountHeads(model,keys).to(device).eval()
    for split in ['optimization','development']:
        results.extend(conditional_samples(baseline,model,caches[split],metas[split],codec,keys,split,fs,'parent',parent_run,masks))
    del baseline
    for arm in CFG['arms']:
        run=OUT/'runs'/f'{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
        write(worker_dir/'STATUS.json',dict(stage='training',arm=arm,fit_seed=fs))
        heads=fit(model,caches,keys,arm,fs,run)
        for split in ['optimization','development']:
            results.extend(conditional_samples(heads,model,caches[split],metas[split],codec,keys,split,fs,arm,run,masks))
        pd.DataFrame(results).to_csv(DOCS/f'conditional_{fs}.csv',index=False)
        write(worker_dir/'STATUS.json',dict(stage='free_generation',arm=arm,fit_seed=fs))
        free_generate(ws,run,fs,arm,str(device))
        write(run/'COMPLETE.json',dict(arm=arm,fit_seed=fs,generated_datasets=2,
              head_sha256=digest(run/'amount_head.pt'),test_events_read=False))
        del heads
    verify()
    write(worker_dir/'STATUS.json',dict(stage='complete',fit_seed=fs))
    write(worker_dir/'COMPLETE.json',dict(fit_seed=fs,seconds=time.monotonic()-started,
          fits=4,generated_datasets=8,parent_unchanged=True,test_events_read=False))
    print('AMOUNT_WORKER_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker'])
    p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--device',default='cpu')
    a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
