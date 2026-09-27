"""Frozen category+amount baseline, gap learning and episode factorial controls."""
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

import run_argn_amount_learning as shared
from run_argn_amount_learning import (ROOT, BASE, seed, digest, write, inputs, load_parent,
    build_cache, AmountHeads, Workspace, folder, get_cardinalities, get_ctx_sequence_length,
    load_model_weights, _translate_fixed_probs, _fix_rare_token_probs, _sample)
from benchmarks.argn_episode_control import fit_episode, episode_probabilities, episode_generation
from benchmarks.argn_transition_reference import reference_table

OUT=ROOT/'artifacts/argn_gap_episode_v1'
DOCS=ROOT/'docs/argn_gap_episode_v1'
CONFIG=ROOT/'configs/argn_gap_episode_v1.json'
CFG=json.loads(CONFIG.read_text())
OLD=ROOT/'artifacts/research_reaudit_20260927/simple_label_parameters.json'


def foundation(fs):
    return ROOT/f'artifacts/argn_category_amount_v1/runs/category_and_amount_{fs}'


def verify():
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    assert manifest['config']==CFG
    for p,h in manifest['hashes'].items():
        assert digest(p)==h, p


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    _,_,_,metas=inputs()
    old=json.loads(OLD.read_text());parameters=fit_episode(metas['optimization'],old)
    write(OUT/'episode_parameters.json',parameters);write(DOCS/'episode_parameters.json',parameters)
    rows=[]
    for split,m in metas.items():
        for name in ['duration','episode','joint']:
            if name=='duration':
                p=reference_table(old,'duration')[m.previous_label.to_numpy()+1,m.prior_age.clip(upper=21).to_numpy()]
            else:
                p=episode_probabilities(m,parameters,name=='joint')
            y=m.label.to_numpy()
            for group,mask in [('all',np.ones(len(m),bool)),('first',m.event_index.eq(0).to_numpy()),
                               ('previous_normal',m.previous_label.eq(0).to_numpy()),('previous_fraud',m.previous_label.eq(1).to_numpy())]:
                q=p[mask];v=y[mask]
                rows.append(dict(split=split,mode=name,group=group,events=len(v),actual_rate=v.mean(),
                    predicted_rate=q.mean(),nll=-(v*np.log(q)+(1-v)*np.log1p(-q)).mean(),
                    brier=np.square(q-v).mean(),observed_length_oracle=name=='joint'))
    pd.DataFrame(rows).to_csv(DOCS/'transition_teacher.csv',index=False)
    paths=[CONFIG,Path(__file__),ROOT/'benchmarks/argn_episode_control.py',
           ROOT/'scripts/dispatch_argn_gap_episode.py',DOCS/'PROTOCOL.md',
           ROOT/'tests/test_argn_episode_control.py',ROOT/'scripts/run_argn_amount_learning.py',
           ROOT/'benchmarks/argn_amount_control.py',ROOT/'benchmarks/argn_transition_reference.py',
           ROOT/'scripts/run_argn_label_first_control.py',OLD,OUT/'episode_parameters.json',
           BASE/'prepared/MANIFEST.json',BASE/'prepared/validation_context.parquet']
    for fs in CFG['fit_seeds']:
        paths += [foundation(fs)/'amount_head.pt', Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path]
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,
                hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',record);write(DOCS/'MANIFEST.json',record)
    print('GAP_EPISODE_PREPARED',flush=True)


@torch.no_grad()
def sample_gap(head,model,cache,meta,codec,keys,fs,arm,split,run,masks):
    eligible=meta.event_index.gt(0).to_numpy();rng=np.random.default_rng(CFG['selection_seed'])
    fraud=np.flatnonzero(eligible & meta.label.eq(1).to_numpy())
    selected=np.sort(np.r_[fraud,rng.choice(np.flatnonzero(eligible & meta.label.eq(0).to_numpy()),len(fraud),False)])
    rows=[]
    for gs in CFG['sampling_seeds']:
        seed(gs)
        for lo in range(0,len(selected),4096):
            chosen=selected[lo:lo+4096];ix=torch.as_tensor(chosen,device=model.device)
            parts=[cache['base'][ix]];tokens={};labels=cache['labels'][ix]
            for key in keys:
                value=_sample(head(torch.cat(parts,-1),key,labels).softmax(-1),fixed_probs=masks.get(key)).reshape(-1)
                tokens[key]=value.cpu().numpy();parts.append(model.embedders.get(key)(value))
            truth=codec.numeric({k:cache['targets'][ix,j].cpu().numpy() for j,k in enumerate(keys)},'gap')
            synthetic=codec.numeric(tokens,'gap')
            g=meta.iloc[chosen][['record','event_index','label']].copy()
            g['real_gap']=truth;g['generated_gap']=synthetic;g['sampling_seed']=gs;rows.append(g)
    data=pd.concat(rows,ignore_index=True);data.to_parquet(run/f'conditional_{split}.parquet',index=False)
    return gap_summary(data,fs,arm,split,'true_category')


def gap_summary(data,fs,arm,split,case):
    rows=[]
    for label,g in data.groupby('label'):
        rows.append(dict(fit_seed=fs,arm=arm,split=split,case=case,label=int(label),positions=len(g)//4,
            real_median=g.real_gap.median(),generated_median=g.generated_gap.median(),
            log_w1=wasserstein_distance(np.log1p(g.real_gap),np.log1p(g.generated_gap))))
    return rows


def combine(path,fs,gap_path=None):
    payload=torch.load(foundation(fs)/'amount_head.pt',map_location='cpu',weights_only=True)
    if gap_path:
        gap=torch.load(gap_path,map_location='cpu',weights_only=True)
        assert not (set(payload['state_dict']) & set(gap['state_dict']))
        payload['state_dict'].update(gap['state_dict']);payload['keys']+=gap['keys']
        payload['gap_sha256']=digest(gap_path)
    payload['foundation_sha256']=digest(foundation(fs)/'amount_head.pt')
    torch.save(payload,path)


@torch.no_grad()
def category_probe(fs,device,ws,codec,records,meta,cache,keys,payload_path,arm,run):
    from benchmarks.argn_amount_control import amount_generation
    import mostlyai.engine._tabular.generation as generation
    ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read();old=json.loads(OLD.read_text())
    # Duration replacement is bypassed by a fixed actual label in this diagnostic.
    with amount_generation(ts,old,payload_path):
        model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
            ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),model_size=ws.model_tabular_configs.read()['model_units'],
            column_order=None,device=device)
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=device);model.to(device).eval()
    encoded={k:torch.as_tensor(np.concatenate([np.asarray(r[k]).reshape(-1) for r in records]),device=device).reshape(-1,1)
             for k in model.tgt_cardinalities}
    rng=np.random.default_rng(CFG['selection_seed']);eligible=meta.event_index.gt(0).to_numpy()
    fraud=np.flatnonzero(eligible & meta.label.eq(1).to_numpy())
    selected=np.sort(np.r_[fraud,rng.choice(np.flatnonzero(eligible & meta.label.eq(0).to_numpy()),len(fraud),False)])
    context_dim=model.context_compressor.dim_output;hd=model.history_dim
    assert not model.context_compressor.ctxseq_cardinalities
    masks=_translate_fixed_probs(_fix_rare_token_probs(ts),ts);rows=[];checks=0
    label_key=codec.prefixes['event_is_fraud']+'__cat';cat_key=codec.prefixes['category']+'__cat'
    structural=[k for k in encoded if k.startswith('tgt:/')]
    for case in ['true_category','sampled_category']:
        fixed_keys=structural+[label_key]+([cat_key] if case=='true_category' else [])
        for gs in CFG['sampling_seeds']:
            seed(gs)
            for lo in range(0,len(selected),768):
                chosen=selected[lo:lo+768];n=len(chosen);b=cache['base'][chosen]
                history=b[:,context_dim:context_dim+hd].unsqueeze(1);context=([b[:,:context_dim]],[],[])
                lstm=model.history_compressor.get()
                zero=tuple(torch.zeros(lstm.num_layers,n,lstm.hidden_size,device=device) for _ in range(2))
                memory=torch.zeros(1,n,9,dtype=torch.float64,device=device)
                captured={}
                def capture(module,args,out):
                    if args[1]==keys[0]:captured['logits']=out[:,0]
                handle=model.predictors.register_forward_hook(capture)
                try:
                    output,_,_=model(None,mode='gen',batch_size=n,history=history,history_state=(*zero,memory),context=context,
                        fixed_values={k:encoded[k][chosen] for k in fixed_keys},fixed_probs=masks)
                finally:handle.remove()
                if case=='true_category' and gs==CFG['sampling_seeds'][0]:
                    label=cache['labels'][chosen]
                    h=model.amount_heads if keys[0] in model.amount_heads.keys else AmountHeads(model,keys).to(device).eval()
                    expected=h(b,keys[0],label)
                    torch.testing.assert_close(captured['logits'],expected,rtol=1e-4,atol=2e-5);checks+=n
                real=codec.numeric({k:cache['targets'][chosen,j].cpu().numpy() for j,k in enumerate(keys)},'gap')
                synthetic=codec.numeric({k:v.cpu().numpy() for k,v in output.items()},'gap')
                part=meta.iloc[chosen][['record','event_index','label']].copy()
                part['case']=case;part['sampling_seed']=gs;part['real_gap']=real;part['generated_gap']=synthetic;rows.append(part)
    data=pd.concat(rows,ignore_index=True);data.to_parquet(run/'category_probe.parquet',index=False)
    write(run/'PROBE_CHECK.json',dict(first_gap_digit_replay_positions=checks))
    return [row for case,g in data.groupby('case') for row in gap_summary(g,fs,arm,'development',case)]


def free_generate(ws,run,fs,arm,device):
    from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    spec=CFG['generation_arms'][arm];old=json.loads(OLD.read_text());parameters=json.loads((OUT/'episode_parameters.json').read_text())
    shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    weight=Workspace(run/'workspace').model_tabular_weights_path;before=digest(weight)
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    with episode_generation(ws.tgt_stats.read(),old,run/'amount_head.pt',parameters,spec['transition']):
        for gs in CFG['generation_seeds']:
            seed(gs);start=time.monotonic();generate(ctx_data=context,device=device,workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            frame=raw.rename(columns={'customer_id':'entity_id'}).copy();frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
            frame.loc[frame.event_index.eq(0),'gap']=np.nan
            dest=run/f'generated_validation_{gs}.parquet';frame.to_parquet(dest,index=False)
            parent=pd.read_parquet(foundation(fs)/f'generated_validation_{gs}.parquet')
            pd.testing.assert_series_equal(frame.groupby('entity_id').size(),parent.groupby('entity_id').size())
            same=frame[['entity_id','event_index','event_is_fraud']].equals(parent[['entity_id','event_index','event_is_fraud']])
            if spec['transition']=='duration':assert same
            assert digest(weight)==before
            write(run/f'GENERATION_{gs}.json',dict(seconds=time.monotonic()-start,rows=len(frame),customers=frame.entity_id.nunique(),
                sha256=digest(dest),head_sha256=digest(run/'amount_head.pt'),same_sampled_lengths=True,same_label_sequence=same))
            print('GAP_EPISODE_GENERATED',arm,fs,gs,len(frame),flush=True)
    saved=evaluation.DOCS;evaluation.DOCS=DOCS
    try:evaluation.evaluate(run,arm,fs)
    finally:evaluation.DOCS=saved


def worker(fs,device):
    verify();started=time.monotonic();seed(fs);device=torch.device(device)
    wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    _,codec,frames,metas=inputs();ws,model=load_parent(fs,device)
    keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['gap']+'__')]
    caches={};fit_caches={};checks={}
    for split in ['optimization','internal_validation','development']:
        cache,checks[split]=build_cache(model,frames[split],metas[split],codec,keys,device)
        caches[split]={k:v.to(device) for k,v in cache.items()}
        eligible=torch.as_tensor(metas[split].event_index.gt(0).to_numpy(),device=device)
        fit_caches[split]={k:v[eligible] for k,v in caches[split].items()}
        print('GAP_CACHE',fs,split,len(cache['labels']),flush=True)
    write(wd/'CACHE_CHECK.json',dict(digit_replay_positions=checks,first_events_excluded=True))
    shared.CFG={**shared.CFG,**CFG};masks=_translate_fixed_probs(_fix_rare_token_probs(ws.tgt_stats.read()),ws.tgt_stats.read())
    results=[];probes=[];nlls=[]
    # Complete native real-prefix diagnosis BEFORE optimizing gap outputs.
    parent_dir=wd/'native_gap';parent_dir.mkdir()
    head=AmountHeads(model,keys).to(device).eval()
    for split in ['optimization','internal_validation','development']:
        nlls.append(dict(arm='native_gap',fit_seed=fs,split=split,**shared.evaluate_nll(head,model,fit_caches[split],keys)))
    for split in ['optimization','development']:
        results+=sample_gap(head,model,caches[split],metas[split],codec,keys,fs,'native_gap',split,parent_dir,masks)
    pd.DataFrame(results).to_csv(DOCS/f'conditional_{fs}.csv',index=False)
    probes+=category_probe(fs,device,ws,codec,frames['development'],metas['development'],caches['development'],keys,
                          foundation(fs)/'amount_head.pt','native_gap',parent_dir)
    pd.DataFrame(probes).to_csv(DOCS/f'category_probe_{fs}.csv',index=False)
    print('GAP_NATIVE_DIAGNOSIS_COMPLETE',fs,flush=True)
    for arm in CFG['arms']:
        run=wd/arm;run.mkdir();write(wd/'STATUS.json',dict(stage='gap_training',arm=arm))
        trained=shared.fit(model,fit_caches,keys,arm,fs,run)
        for split in ['optimization','internal_validation','development']:
            nlls.append(dict(arm=arm,fit_seed=fs,split=split,**shared.evaluate_nll(trained,model,fit_caches[split],keys)))
        for split in ['optimization','development']:
            results+=sample_gap(trained,model,caches[split],metas[split],codec,keys,fs,arm,split,run,masks)
        pd.DataFrame(results).to_csv(DOCS/f'conditional_{fs}.csv',index=False)
        pd.DataFrame(nlls).to_csv(DOCS/f'nll_{fs}.csv',index=False)
        combine(run/'combined_head.pt',fs,run/'amount_head.pt')
        probes+=category_probe(fs,device,ws,codec,frames['development'],metas['development'],caches['development'],keys,
                              run/'combined_head.pt',arm,run)
        pd.DataFrame(probes).to_csv(DOCS/f'category_probe_{fs}.csv',index=False)
    del caches,fit_caches,frames,model,head,trained;torch.cuda.empty_cache()
    for arm,spec in CFG['generation_arms'].items():
        run=OUT/'runs'/f'{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
        combine(run/'amount_head.pt',fs,wd/'balanced_gap/amount_head.pt' if spec['gap'] else None)
        write(wd/'STATUS.json',dict(stage='free_generation',arm=arm))
        free_generate(ws,run,fs,arm,str(device))
        write(run/'COMPLETE.json',dict(arm=arm,fit_seed=fs,generated_datasets=2,test_events_read=False))
    verify();write(wd/'STATUS.json',dict(stage='complete'))
    write(wd/'COMPLETE.json',dict(seconds=time.monotonic()-started,gap_fits=2,generated_datasets=8,test_events_read=False))
    print('GAP_EPISODE_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
