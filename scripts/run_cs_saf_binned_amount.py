"""One bounded D_bin fit, fixed-plan generation, and predeclared tail evaluation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizer,SAFTensorizerState
from data.cs_saf_external import TargetWindows
from models.cs_saf_binned_amount import BinnedAmountD,fit_amount_pool,pool_metadata,pool_tail_matrix
from benchmarks.cs_saf_external import evaluate
from scripts.run_cs_saf_external_port import inputs,prediction,seed,digest,write
from scripts.run_cs_saf_external_controls import OUT as PARENT,OLD

OUT=ROOT/'artifacts/cs_saf/external_binned_amount_v1'
DOC=ROOT/'docs/cs_saf/external_binned_amount_v1'
CONFIG=ROOT/'configs/cs_saf_external_binned_amount_v1.json'


def setup(name,cfg):
    inp,_,_,frames,plan,metric_state=inputs(name)
    data=torch.load(inp/'prepared.pt',map_location='cpu')
    state=SAFTensorizerState.from_dict(data['state'])
    source=PARENT/'runs'/name/cfg['reference']
    registration=json.loads((source/'START.json').read_text())
    for f,sha in registration['scientific_source_sha256'].items():
        assert digest(ROOT/f)==sha,f
    ast=json.loads((source/'amount_state.json').read_text())
    opt=cfg['amount']
    pool=fit_amount_pool(frames['fit'].amount_or_numeric_value,state.event_numeric_codecs[0][1],
                         opt['positive_equal_mass_bins'],opt['extra_positive_quantiles'])
    seed(cfg['fit_seed'])
    model=BinnedAmountD(state.gap_support,ast,pool,**state.model_config_kwargs())
    initial=torch.load(source/'initial.pt',map_location='cpu')
    keys=model.initialize_from_direct(initial)
    assert all(torch.equal(model.state_dict()[k],initial[k]) for k in keys)
    return inp,frames,plan,metric_state,data,state,pool,model,keys


def cut_values(fit):
    a=fit.amount_or_numeric_value.to_numpy(float)
    return np.r_[np.quantile(a,[.99,.999]),a.max(),10*a.max()]


def tail_generation(real,generated,thresholds,root):
    def stats(frame,cut):
        roots=frame[root].fillna('<MISSING>').astype(str)
        above=frame.amount_or_numeric_value.to_numpy(float)>cut
        return float(above.mean()),roots[above].value_counts()/len(frame)
    output=[]
    for label,cut in zip(('q99','q999','fit_max','ten_fit_max'),thresholds):
        rate,r=stats(real,cut);pred,g=stats(generated,cut)
        r,g=r.align(g,fill_value=0)
        output.append(dict(threshold=label,cut=float(cut),real_rate=rate,generated_rate=pred,
                           root_tail_l1=float(abs(r-g).sum())))
    return output


@torch.no_grad()
def teacher_tail(model,windows,pool,cuts,batch_size):
    matrix=torch.as_tensor(pool_tail_matrix(pool,cuts),device=next(model.parameters()).device,dtype=torch.float64)
    sums=np.zeros(len(cuts));n=0
    for start in range(0,len(windows),batch_size):
        batch=windows.batch(np.arange(start,min(start+batch_size,len(windows))))
        o=model.target_outputs(**batch)
        p=model.amount_parameters(o['hidden'],o['gap'],o['mark']).double().exp() @ matrix
        sums+=p.sum(0).cpu().numpy();n+=len(p)
    return (sums/n).tolist()


def preflight(device):
    cfg=json.loads(CONFIG.read_text());result=dict(passed=True,device=device,optimizer_updates=0,rows=[])
    for name in cfg['datasets']:
        inp,frames,plan,ms,data,state,pool,model,keys=setup(name,cfg)
        model=model.to(device);opts=cfg['training']
        record=dict(dataset=name,pool=pool_metadata(pool),parameters=model.architecture_contract()['parameters'],
                    shared_initial_keys=len(keys),roles={})
        for role in ('fit','check','validation'):
            w=TargetWindows(data['sequences'][role],device=device)
            counts=np.zeros(len(pool['counts']),dtype=np.int64)
            for start in range(0,len(w),100000):
                c=model.amount_codes(w.flat['numeric_value'][start:start+100000]).cpu().numpy()
                counts+=np.bincount(c,minlength=len(counts))
            if role=='fit':np.testing.assert_array_equal(counts,pool['counts'])
            starts=np.cumsum(np.r_[0,w.lengths[:-1]])
            idx=np.unique(np.r_[starts,starts+w.lengths-1,starts+np.minimum(32,w.lengths-1)])
            for first in range(0,len(idx),512):
                batch=w.batch(idx[first:first+512]);terms,o=model.terms(**batch)
                assert all(torch.isfinite(v[0]) for v in terms.values())
                if device=='cuda' and first==0:
                    loss=sum(v/c.clamp_min(1) for v,c in terms.values());loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    model.zero_grad(set_to_none=True)
            a=frames[role].amount_or_numeric_value.to_numpy(float)
            record['roles'][role]=dict(events=len(w),boundary_checks=len(idx),bin_counts=counts.tolist(),
                outside_fit_range=int(((a<pool['values'].min())|(a>pool['values'].max())).sum()))
        result['rows'].append(record)
        print('PREFLIGHT',device,name,'PASS',record['pool']['bins'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/f'preflight_{device}.json',result);write(DOC/f'preflight_{device}.json',result)


def train(name,folder,cfg):
    inp,frames,plan,metric_state,data,state,pool,model,keys=setup(name,cfg)
    source=PARENT/'runs'/name/cfg['reference']
    source_hashes={f:digest(source/f) for f in ('initial.pt','best.pt','DONE.json')}
    np.savez_compressed(folder/'pool.npz',**pool)
    write(folder/'pool.json',pool_metadata(pool));write(folder/'amount_state.json',model.amount_state)
    architecture=model.architecture_contract();architecture['shared_initial_keys']=keys
    architecture['source_initial_sha256']=source_hashes['initial.pt']
    write(folder/'architecture.json',architecture)
    model=model.cuda();torch.save(model.state_dict(),folder/'initial.pt')
    fit=TargetWindows(data['sequences']['fit'],device='cuda')
    check=TargetWindows(data['sequences']['check'],device='cuda')
    optcfg=cfg['training'];opt=torch.optim.Adam(model.parameters(),lr=optcfg['learning_rate'],weight_decay=0.)
    best=float('inf');best_epoch=0;updates=0;history=[];reason='epoch_budget';begin=time.monotonic()
    for epoch in range(1,optcfg['max_epochs']+1):
        epoch_start=time.monotonic();model.train()
        order=np.random.default_rng(cfg['fit_seed']+epoch).permutation(len(fit));sums={};counts={}
        for start in range(0,len(order),optcfg['batch_size']):
            terms,_=model.terms(**fit.batch(order[start:start+optcfg['batch_size']]))
            loss=sum(v/c.clamp_min(1) for v,c in terms.values())
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),optcfg['gradient_clip']);opt.step();updates+=1
            for k,(v,c) in terms.items():sums[k]=sums.get(k,0.)+float(v.detach());counts[k]=counts.get(k,0)+int(c)
        score,_=prediction(model,check,optcfg['batch_size'],state)
        assert np.isfinite(score['loss'])
        improved=score['loss']<best
        if improved:
            best=score['loss'];best_epoch=epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,check=score),folder/'best.pt')
        record=dict(epoch=epoch,fit_loss=sum(sums[k]/counts[k] for k in sums),check=score,
                    selected=improved,seconds=time.monotonic()-epoch_start,optimizer_updates=updates)
        history.append(record);write(folder/'history.json',history)
        print(name,'EPOCH',epoch,'fit',record['fit_loss'],'check',score['loss'],'best',best_epoch,flush=True)
        if epoch-best_epoch>=optcfg['patience']:reason='check_patience';break
        if time.monotonic()-begin>=optcfg['max_fit_seconds']:reason='time_budget';break
    seconds=time.monotonic()-begin
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch),folder/'last.pt')
    model.load_state_dict(torch.load(folder/'best.pt')['state_dict']);model.eval()
    val=TargetWindows(data['sequences']['validation'],device='cuda')
    pred,_=prediction(model,val,optcfg['batch_size'],state)
    cuts=cut_values(frames['fit']);tails=teacher_tail(model,val,pool,cuts,optcfg['batch_size'])
    tf=SAFTensorizer(state);positions=plan.fit_position.to_numpy(int);generations=[]
    for gs in cfg['generation_seeds']:
        seed(gs);begun=time.monotonic()
        sample=model.sample_fixed_lengths(plan.length.tolist(),static=fit.static[positions],
                                         static_categorical=tuple(v[positions] for v in fit.static_cat))
        frame=tf.decode_generated(entity_ids=plan.entity_id.tolist(),lengths=plan.length.tolist(),
            **{k:sample[k] for k in ('gap','receiver','numeric_value','auxiliary_categorical','auxiliary_numeric')})
        frame['amount_or_numeric_value']=np.concatenate([sample['raw_amount'][i,:n].cpu().numpy() for i,n in enumerate(plan.length)])
        assert len(frame)==int(plan.length.sum())
        path=folder/f'generated_raw_{gs}.parquet';frame.to_parquet(path,index=False)
        generations.append(dict(seed=gs,sha256=digest(path),seconds=time.monotonic()-begun,
            metrics=evaluate(frames['validation'],frame,metric_state),
            tails=tail_generation(frames['validation'],frame,cuts,metric_state['root'])))
        print(name,'GENERATED',gs,len(frame),flush=True)
    for f,h in source_hashes.items():assert digest(source/f)==h
    return dict(dataset=name,model=cfg['model'],architecture=architecture,selected_epoch=best_epoch,
                last_epoch=epoch,stop_reason=reason,optimizer_updates=updates,fit_seconds=seconds,
                reference_files_unchanged=source_hashes,best_sha256=digest(folder/'best.pt'),
                results=dict(raw=dict(prediction=pred,teacher_tail= tails,tail_thresholds=cuts.tolist(),generations=generations)))


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?');p.add_argument('--preflight',choices=['cpu','cuda'])
    a=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if a.preflight:
        preflight(a.preflight);return
    cfg=json.loads(CONFIG.read_text());assert a.dataset in cfg['datasets'] and torch.cuda.is_available()
    for dev in ('cpu','cuda'):assert json.loads((OUT/f'preflight_{dev}.json').read_text())['passed']
    torch.cuda.set_per_process_memory_fraction(.5)
    scientific=['configs/cs_saf_external_binned_amount_v1.json','models/cs_saf_binned_amount.py',
        'scripts/run_cs_saf_binned_amount.py','docs/cs_saf/external_binned_amount_v1/preregistration.md',
        'models/cs_saf_external_controls.py','models/cs_saf_external.py','models/cof_seqgen_saf.py',
        'data/cs_saf_external.py','data/cof_seqgen_saf_tensorizer.py','benchmarks/cs_saf_external.py']
    subprocess.check_call(['git','ls-files','--error-unmatch',*scientific],cwd=ROOT,stdout=subprocess.DEVNULL)
    assert not subprocess.check_output(['git','diff','HEAD','--',*scientific],cwd=ROOT)
    folder=OUT/'runs'/a.dataset;assert not folder.exists(),'never overwrite a scientific fit'
    folder.mkdir(parents=True)
    write(folder/'START.json',dict(dataset=a.dataset,config_sha256=digest(CONFIG),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scientific_source_sha256={f:digest(ROOT/f) for f in scientific},
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),fit_seed=cfg['fit_seed'],
        versions={k:importlib.metadata.version(k) for k in ('torch','numpy','pandas')},test_outcomes_accessed=False))
    begin=time.monotonic()
    try:
        result=train(a.dataset,folder,cfg)
        result.update(total_seconds=time.monotonic()-begin,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      files={f.name:digest(f) for f in folder.iterdir() if f.is_file()})
        write(folder/'DONE.json',result);print(a.dataset,'DONE',flush=True)
    except Exception as e:
        write(folder/'FAILED.json',dict(type=type(e).__name__,message=str(e),elapsed_seconds=time.monotonic()-begin));raise


if __name__=='__main__':main()
