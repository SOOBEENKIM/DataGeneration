"""Frozen-output joint-count / exposed onset-hazard factorial controls."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch

from run_argn_gap_episode import (ROOT, BASE, seed, digest, write, inputs, load_parent,
    Workspace, folder, OLD, get_cardinalities, get_ctx_sequence_length, load_model_weights,
    _translate_fixed_probs, _fix_rare_token_probs)
from run_argn_gap_episode import OUT as PRIOR, verify as verify_prior
from benchmarks.argn_joint_preservation import JointCount,fit_horizon_hazard,preservation_generation
from benchmarks.argn_episode_control import decode_length,episode_metadata,episode_probabilities
from benchmarks.argn_joint_preservation import horizon_probability

OUT=ROOT/'artifacts/argn_joint_preservation_v1'
DOCS=ROOT/'docs/argn_joint_preservation_v1'
CONFIG=ROOT/'configs/argn_joint_preservation_v1.json'
CFG=json.loads(CONFIG.read_text())


def foundation(fs):return PRIOR/f'runs/gap_and_transition_{fs}'


def verify():
    verify_prior();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    verify_prior();assert not (OUT/'MANIFEST.json').exists()
    _,_,_,metas=inputs();hazard=fit_horizon_hazard(metas['optimization'])
    write(OUT/'hazard_parameters.json',hazard);write(DOCS/'hazard_parameters.json',hazard)
    parameters=json.loads((PRIOR/'episode_parameters.json').read_text());rows=[]
    for split,meta in metas.items():
        m=episode_metadata(meta);base=episode_probabilities(meta,parameters,True)
        mask=(m.previous_label.eq(0)&m.ever_fraud.eq(0)).to_numpy()
        q=horizon_probability(m.event_index.to_numpy(),m.planned_length.to_numpy(),hazard['rates'])
        for name,p in [('current',base),('horizon',q)]:
            v=m.label.to_numpy()[mask];p=p[mask]
            rows.append(dict(split=split,mode=name,risk_events=len(v),onsets=int(v.sum()),
                actual_rate=v.mean(),predicted_rate=p.mean(),nll=-(v*np.log(p)+(1-v)*np.log1p(-p)).mean(),
                brier=np.square(p-v).mean(),scope='observed length diagnostic, not online forecasting'))
    pd.DataFrame(rows).to_csv(DOCS/'hazard_teacher.csv',index=False)
    paths=[CONFIG,Path(__file__),ROOT/'benchmarks/argn_joint_preservation.py',
        ROOT/'scripts/dispatch_argn_joint_preservation.py',ROOT/'tests/test_argn_joint_preservation.py',
        DOCS/'PROTOCOL.md',ROOT/'benchmarks/argn_episode_control.py',ROOT/'scripts/run_argn_gap_episode.py',
        PRIOR/'MANIFEST.json',PRIOR/'episode_parameters.json',OUT/'hazard_parameters.json',
        BASE/'prepared/MANIFEST.json',BASE/'prepared/validation_context.parquet',OLD,
        ROOT/'benchmarks/argn_amount_control.py',ROOT/'benchmarks/argn_past_state.py',
        ROOT/'scripts/run_argn_label_first_control.py',DOCS/'DIAGNOSIS_COMPLETE.json']
    for fs in CFG['fit_seeds']:
        paths += [foundation(fs)/'amount_head.pt',Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path]
        paths += [foundation(fs)/f'generated_validation_{gs}.parquet' for gs in CFG['generation_seeds']]
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,
                hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',record);write(DOCS/'MANIFEST.json',record)
    print('JOINT_PRESERVATION_PREPARED',flush=True)


@torch.no_grad()
def context_cache(model,records,meta):
    ctx={k:torch.as_tensor(np.array([r[k] for r in records]),device=model.device).reshape(len(records),-1)
         for k in model.context_compressor.ctxflt_cardinalities}
    flat,seq,masks=model.context_compressor(ctx)
    assert not seq and not masks
    x=torch.cat(flat,-1)
    first=meta.groupby('record',sort=False).first();length=meta.groupby('record',sort=False).size()
    assert len(x)==len(first)==len(records)
    assert np.array_equal(first.index.to_numpy(),np.arange(len(records)))
    return dict(x=x.detach(),length=torch.as_tensor(length.to_numpy(),device=model.device),
                label=torch.as_tensor(first.label.to_numpy(),device=model.device))


def fit_count(cache,stats,fs,conditioned,run):
    seed(fs);x=cache['optimization'];val=cache['internal_validation'];lo,hi=stats['min'],stats['max']
    head=JointCount(x['x'].shape[-1],CFG['components'],conditioned).to(x['x'].device)
    head.initialize(**x,minimum=lo,maximum=hi,seed=fs)
    optimizer=torch.optim.AdamW(head.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    @torch.no_grad()
    def score(c):
        head.eval();loss=head.nll(**c,minimum=lo,maximum=hi)
        assert torch.isfinite(loss).all()
        return float(loss.mean())
    baseline=score(val);best=baseline;best_state=deepcopy(head.state_dict());best_step=0;stale=0
    curve=[dict(step=0,train_nll=score(x),validation_nll=baseline)]
    for step in range(1,CFG['max_steps']+1):
        head.train();ix=torch.randint(len(x['x']),(CFG['batch_size'],),device=x['x'].device)
        loss=head.nll(**{k:v[ix] for k,v in x.items()},minimum=lo,maximum=hi).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),CFG['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        if step%CFG['validation_every']:continue
        current=score(val);curve.append(dict(step=step,train_nll=score(x),validation_nll=current))
        if current<best-CFG['min_delta']:
            best=current;best_state=deepcopy(head.state_dict());best_step=step;stale=0
        else:stale+=1
        if stale>=CFG['patience']:break
    head.load_state_dict(best_state);head.eval()
    payload=dict(state_dict={k:v.cpu() for k,v in head.state_dict().items()},dim=head.dim,
                 components=head.components,conditioned=conditioned,minimum=lo,maximum=hi)
    torch.save(payload,run/'count_head.pt');pd.DataFrame(curve).to_csv(run/'learning_curve.csv',index=False)
    evidence=dict(fit_seed=fs,conditioned=conditioned,steps=step,best_step=best_step,initial_validation_nll=baseline,
        best_validation_nll=best,nll={k:score(v) for k,v in cache.items()},customers={k:len(v['label']) for k,v in cache.items()},
        head_sha256=digest(run/'count_head.pt'),test_events_read=False)
    write(run/'FIT_COMPLETE.json',evidence);print('COUNT_FIT_COMPLETE',evidence,flush=True)
    return evidence


@torch.no_grad()
def integration_check(ws,fs,cache,head_path,device,run):
    """Load actual native weights and exercise first/subsequent/compacted batch steps."""
    import mostlyai.engine._tabular.generation as generation
    ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read();old=json.loads(OLD.read_text())
    episode=json.loads((PRIOR/'episode_parameters.json').read_text());hazard=json.loads((OUT/'hazard_parameters.json').read_text())
    with preservation_generation(ts,old,foundation(fs)/'amount_head.pt',episode,hazard,head_path,True):
        model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
            ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),model_size=ws.model_tabular_configs.read()['model_units'],
            column_order=None,device=device)
        assert not any(k.startswith('joint_count.') for k in model.state_dict())
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=device);model.to(device).eval()
        assert not any(p.requires_grad for p in model.joint_count.parameters())
        from benchmarks.argn_past_state import PastState
        codec=PastState(ts);lk=codec.prefixes['event_is_fraud']+'__cat'
        masks=_translate_fixed_probs(_fix_rare_token_probs(ts),ts)
        c=cache['development']['x'][:8];history=state=None;length=None;checked=0
        for step in range(3):
            fixed={} if length is None else {k:v for k,v in structural.items()}
            generated,history,state=model(None,mode='gen',batch_size=len(c),context=([c],[],[]),
                history=history,history_state=state,fixed_values=fixed,fixed_probs=masks)
            if step==0:
                from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX
                structural={k:v for k,v in generated.items() if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
                length=decode_length({k:v.reshape(-1) for k,v in structural.items()},ts['seq_len']['min'])
                label=generated[lk].reshape(-1).eq(codec.codes['1']).long()
                assert torch.equal(label,model.first_joint_label)
                assert length.ge(ts['seq_len']['min']).all() and length.le(ts['seq_len']['max']).all()
            assert len(state)==4 and all(s.shape[1]==len(c) for s in state)
            assert torch.equal(state[2][0,:,0],torch.full((len(c),),step+1,dtype=torch.float64,device=device))
            checked+=len(c)
            if step==1:
                chosen=torch.tensor([1,3,6],device=device)
                c=c[chosen];history=history[chosen];state=tuple(s[:,chosen] for s in state)
                structural={k:v[chosen] for k,v in structural.items()}
    write(run/'NATIVE_INTEGRATION.json',dict(native_checkpoint_loaded=True,steps=3,positions=checked,
        length_and_initial_label_sampled_before_rows=True,compacted_batch=True))


def free_generate(ws,run,fs,arm,device,count_path):
    from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    spec=CFG['generation_arms'][arm];old=json.loads(OLD.read_text())
    episode=json.loads((PRIOR/'episode_parameters.json').read_text());hazard=json.loads((OUT/'hazard_parameters.json').read_text())
    shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    shutil.copy2(foundation(fs)/'amount_head.pt',run/'amount_head.pt')
    assert digest(run/'amount_head.pt')==digest(foundation(fs)/'amount_head.pt')
    weight=Workspace(run/'workspace').model_tabular_weights_path;before=digest(weight)
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    with preservation_generation(ws.tgt_stats.read(),old,run/'amount_head.pt',episode,hazard,count_path,spec['hazard']):
        for gs in CFG['generation_seeds']:
            seed(gs);start=time.monotonic();generate(ctx_data=context,device=device,workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            frame=raw.rename(columns={'customer_id':'entity_id'}).copy();frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
            frame.loc[frame.event_index.eq(0),'gap']=np.nan
            dest=run/f'generated_validation_{gs}.parquet';frame.to_parquet(dest,index=False)
            parent=pd.read_parquet(foundation(fs)/f'generated_validation_{gs}.parquet')
            same=frame.groupby('entity_id').size().equals(parent.groupby('entity_id').size())
            if not spec['count']:assert same,'hazard-only changed native sampled length'
            assert digest(weight)==before
            write(run/f'GENERATION_{gs}.json',dict(seconds=time.monotonic()-start,rows=len(frame),customers=frame.entity_id.nunique(),
                sha256=digest(dest),head_sha256=digest(run/'amount_head.pt'),same_sampled_lengths=same,
                count_sha256=digest(count_path) if count_path else None))
            print('JOINT_GENERATED',arm,fs,gs,len(frame),flush=True)
    saved=evaluation.DOCS;evaluation.DOCS=DOCS
    try:evaluation.evaluate(run,arm,fs)
    finally:evaluation.DOCS=saved


def worker(fs,device):
    verify();started=time.monotonic();seed(fs);device=torch.device(device)
    wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    _,_,records,metas=inputs();ws,model=load_parent(fs,device)
    cache={k:context_cache(model,records[k],m) for k,m in metas.items()}
    del records,model;torch.cuda.empty_cache()
    for arm,conditioned in [('conditional',True),('unconditional',False)]:
        run=wd/arm;run.mkdir();write(wd/'STATUS.json',dict(stage='count_training',arm=arm))
        fit_count(cache,ws.tgt_stats.read()['seq_len'],fs,conditioned,run)
    integration_check(ws,fs,cache,wd/'conditional/count_head.pt',device,wd)
    del cache;torch.cuda.empty_cache()
    for arm,spec in CFG['generation_arms'].items():
        run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
        write(wd/'STATUS.json',dict(stage='free_generation',arm=arm))
        count_path=wd/spec['count']/'count_head.pt' if spec['count'] else None
        free_generate(ws,run,fs,arm,str(device),count_path)
        write(run/'COMPLETE.json',dict(arm=arm,fit_seed=fs,generated_datasets=2,test_events_read=False))
    verify();write(wd/'STATUS.json',dict(stage='complete'))
    write(wd/'COMPLETE.json',dict(seconds=time.monotonic()-started,count_fits=2,generated_datasets=8,test_events_read=False))
    print('JOINT_PRESERVATION_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
