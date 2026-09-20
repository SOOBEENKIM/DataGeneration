"""Registered, bounded external pilot. One CLI invocation owns one model run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from benchmarks.cs_saf_external import fit_state,evaluate,gap_bin
from data.cof_seqgen_saf_tensorizer import SAFTensorizer,SAFTensorizerState
from data.cs_saf_external import TargetWindows,VIEWS
from models.cs_saf_external import ExternalUG
from generators.cs_saf_external_empirical import ObservedTransition

OUT=ROOT/'artifacts/cs_saf/external_port_v1'
CONFIG=ROOT/'configs/cs_saf_external_port_v1.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def inputs(name):
    folder=OUT/'input'/name
    pre=json.loads((folder/'preflight.json').read_text())
    for f,h in pre['files'].items():assert digest(folder/f)==h,f
    events=pd.read_parquet(folder/'events.parquet');parents=pd.read_parquet(folder/'context.parquet')
    roles=pd.read_parquet(folder/'roles.parquet');plan=pd.read_parquet(folder/'plan.parquet')
    ids={r:set(roles.loc[roles.role.eq(r),'entity_id']) for r in ('fit','check','validation')}
    frames={r:events.loc[events.entity_id.isin(v)].copy() for r,v in ids.items()}
    state=fit_state(frames['fit'],name)
    return folder,parents,ids,frames,plan,state


@torch.no_grad()
def prediction(model,windows,batch_size,state,collect=False):
    model.eval();sums={};counts={};brier=0.;nrepeat=0;mae=0.;n=0;arrays=[]
    for start in range(0,len(windows),batch_size):
        terms,o=model.terms(**windows.batch(np.arange(start,min(start+batch_size,len(windows)))))
        for k,(v,c) in terms.items():sums[k]=sums.get(k,0.)+float(v);counts[k]=counts.get(k,0)+int(c)
        logr=o['logmark'].gather(1,o['previous'][:,None]).squeeze(1)
        target=o['mark'].eq(o['previous']).float();has=o['has_previous']
        brier+=float((logr.exp()[has]-target[has]).square().sum());nrepeat+=int(has.sum())
        mae+=float(abs(o['numeric']-o['location']).sum())*state.event_numeric_codecs[0][1].scale;n+=len(logr)
        if collect:
            other=o['logmark'].clone();other.scatter_(1,o['previous'][:,None],-torch.inf)
            r=logr-other.logsumexp(-1)
            arrays.append(np.column_stack([r[has].cpu().numpy(),target[has].cpu().numpy(),o['gap'][has].cpu().numpy()]))
    values={k:sums[k]/counts[k] for k in sums}
    return dict(loss=sum(values.values()),components=values,repeat_brier=brier/max(nrepeat,1),
                amount_log_mae=mae/n,events=n,transitions=nrepeat),np.concatenate(arrays) if collect else None


def calibration_fit(array,edges,cfg):
    r,y,g=array.T;bins=gap_bin(g,edges);k=len(edges)+2
    penalty=cfg['penalty']
    def objective(beta):
        z=r+beta[bins]
        val=np.mean(np.logaddexp(0,z)-y*z)+penalty*np.square(beta).sum()
        grad=np.bincount(bins,weights=expit(z)-y,minlength=k)/len(y)+2*penalty*beta
        return val,grad
    fitted=minimize(objective,np.zeros(k),jac=True,method='L-BFGS-B',bounds=[cfg['bounds']]*k,
                    options={'maxiter':cfg['maxiter'],'ftol':1e-12,'gtol':1e-8})
    if not fitted.success:raise RuntimeError('registered calibration optimization failed: '+str(fitted.message))
    return dict(edges=edges,beta=fitted.x.tolist(),penalty=penalty,fit_transitions=len(y),
                fit_objective_before=objective(np.zeros(k))[0],fit_objective_after=float(fitted.fun),success=True)


def train_ug(name,mode,folder,cfg):
    inp,parents,ids,frames,plan,metric_state=inputs(name)
    data=torch.load(inp/'prepared.pt',map_location='cpu')
    state=SAFTensorizerState.from_dict(data['state']);tf=SAFTensorizer(state)
    train=TargetWindows(data['sequences']['fit'],device='cuda')
    check=TargetWindows(data['sequences']['check'],device='cuda')
    seed(cfg['fit_seed'])
    model=ExternalUG(mode,state.gap_support,**state.model_config_kwargs()).cuda()
    torch.save(model.state_dict(),folder/'initial.pt')
    options=cfg['ug'];opt=torch.optim.Adam(model.parameters(),lr=options['learning_rate'],weight_decay=0.)
    best=float('inf');best_epoch=0;history=[];stopped='epoch_budget';updates=0
    started=time.monotonic()
    for epoch in range(1,options['max_epochs']+1):
        model.train();order=np.random.default_rng(cfg['fit_seed']+epoch).permutation(len(train))
        sums={};counts={};epoch_start=time.monotonic()
        for start in range(0,len(order),options['batch_size']):
            batch=train.batch(order[start:start+options['batch_size']])
            terms,_=model.terms(**batch)
            loss=sum(v/c.clamp_min(1) for v,c in terms.values())
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),options['gradient_clip']);opt.step();updates+=1
            for k,(v,c) in terms.items():sums[k]=sums.get(k,0.)+float(v.detach());counts[k]=counts.get(k,0)+int(c)
        check_score,_=prediction(model,check,options['batch_size'],state)
        improved=check_score['loss']<best
        if improved:
            best,best_epoch=check_score['loss'],epoch
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'check':check_score},folder/'best.pt')
        record=dict(epoch=epoch,fit_loss=sum(sums[k]/counts[k] for k in sums),check=check_score,
                    seconds=time.monotonic()-epoch_start,optimizer_updates=updates,selected=improved)
        history.append(record);write(folder/'history.json',history)
        print(name,mode,'EPOCH',epoch,'fit',record['fit_loss'],'check',best,'seconds',record['seconds'],flush=True)
        if epoch-best_epoch>=options['patience']:stopped='check_patience';break
        if time.monotonic()-started>=options['max_fit_seconds']:stopped='time_budget';break
    fit_seconds=time.monotonic()-started
    torch.save({'state_dict':model.state_dict(),'epoch':epoch},folder/'last.pt')
    model.load_state_dict(torch.load(folder/'best.pt')['state_dict'])
    _,calibration_data=prediction(model,train,options['batch_size'],state,collect=True)
    correction=calibration_fit(calibration_data,metric_state['gap_edges'],cfg['calibration'])
    write(folder/'calibration.json',correction)
    validation=TargetWindows(data['sequences']['validation'],device='cuda')
    results={}
    positions=plan.fit_position.to_numpy(int)
    for variant in ('raw','gap'):
        model.calibration=None if variant=='raw' else correction
        pred,_=prediction(model,validation,options['batch_size'],state)
        generations=[]
        for gs in cfg['generation_seeds']:
            seed(gs);start=time.monotonic()
            sample=model.sample_fixed_lengths(plan.length.tolist(),static=train.static[positions],
                        static_categorical=tuple(v[positions] for v in train.static_cat))
            output=tf.decode_generated(entity_ids=plan.entity_id.tolist(),lengths=plan.length.tolist(),
                **{k:sample[k] for k in ('gap','receiver','numeric_value','auxiliary_categorical','auxiliary_numeric')})
            output.to_parquet(folder/f'generated_{variant}_{gs}.parquet',index=False)
            score=evaluate(frames['validation'],output,metric_state)
            generations.append(dict(seed=gs,seconds=time.monotonic()-start,metrics=score))
            print(name,mode,variant,'GENERATED',gs,len(output),flush=True)
        results[variant]=dict(prediction=pred,generations=generations)
    return dict(model=mode,dataset=name,selected_epoch=best_epoch,last_epoch=epoch,stop_reason=stopped,
                optimizer_updates=updates,fit_seconds=fit_seconds,architecture=model.architecture_contract(),
                initial_sha256=digest(folder/'initial.pt'),best_sha256=digest(folder/'best.pt'),results=results)


def empirical(name,folder,cfg):
    inp,parents,ids,frames,plan,state=inputs(name);results={};search=[]
    for shrink in cfg['empirical']['smoothing_grid']:
        model=ObservedTransition(name,state['gap_edges'],shrink=shrink).fit(frames['fit'])
        search.append(dict(shrink=shrink,check_joint_categorical_nll=model.nll(frames['check'])))
    chosen=min(search,key=lambda x:x['check_joint_categorical_nll'])['shrink']
    for history in (False,True):
        key='transition' if history else 'marginal'
        model=ObservedTransition(name,state['gap_edges'],shrink=chosen,use_history=history).fit(frames['fit'])
        generations=[]
        for gs in cfg['generation_seeds']:
            start=time.monotonic();output=model.sample(plan,gs)
            output.to_parquet(folder/f'generated_{key}_{gs}.parquet',index=False)
            generations.append(dict(seed=gs,seconds=time.monotonic()-start,metrics=evaluate(frames['validation'],output,state)))
            print(name,'empirical',key,'GENERATED',gs,len(output),flush=True)
        results[key]=dict(generations=generations)
    return dict(model='empirical',dataset=name,smoothing_selection=search,selected_smoothing=chosen,results=results)


def native_frame(raw):
    required=['entity_id','gap','receiver_or_mark','amount_or_numeric_value']
    assert set(required)<=set(raw),raw.columns
    x=raw.copy();x['event_index']=x.groupby('entity_id',sort=False).cumcount()
    x.loc[x.event_index.eq(0),'gap']=np.nan
    x['timestamp']=x.gap.fillna(0).groupby(x.entity_id,sort=False).cumsum()
    return x


def train_argn(name,folder,cfg):
    from mostlyai.engine import split,analyze,encode,train,generate,set_random_state
    from mostlyai.engine._common import load_generated_data
    from mostlyai.engine._workspace import Workspace
    assert importlib.metadata.version('mostlyai-engine')==cfg['argn']['version']
    inp,parents,ids,frames,plan,state=inputs(name)
    par=parents.loc[parents.entity_id.isin(ids['fit']|ids['check'])].copy()
    child=pd.concat([frames['fit'],frames['check']]).sort_values(['entity_id','event_index'],kind='stable')
    columns=['entity_id','gap','receiver_or_mark','amount_or_numeric_value',*VIEWS[name]['auxiliary']]
    child=child[columns].copy();child['gap']=child.gap.fillna(0)
    for c in VIEWS[name]['categorical']:par[c]=par[c].astype(str)
    tgt_types={c:'TABULAR_CATEGORICAL' if c in ['receiver_or_mark',*VIEWS[name]['auxiliary']] else 'TABULAR_NUMERIC_AUTO' for c in columns[1:]}
    ctx_types={c:'TABULAR_CATEGORICAL' if c in VIEWS[name]['categorical'] else 'TABULAR_NUMERIC_AUTO' for c in par if c!='entity_id'}
    set_random_state(cfg['fit_seed']);wsdir=folder/'workspace'
    split(tgt_data=child,ctx_data=par,tgt_context_key='entity_id',ctx_primary_key='entity_id',
          tgt_encoding_types=tgt_types,ctx_encoding_types=ctx_types,workspace_dir=wsdir,
          trn_val_split=lambda keys:(keys[keys.isin(ids['fit'])],keys[keys.isin(ids['check'])]))
    analyze(value_protection=False,workspace_dir=wsdir);encode(workspace_dir=wsdir)
    options=cfg['argn'];started=time.monotonic()
    train(model=options['model'],max_epochs=options['max_epochs'],max_training_time=options['max_training_minutes'],
          batch_size=options['batch_size'],max_sequence_window=options['max_sequence_window'],
          enable_flexible_generation=False,device='cuda',workspace_dir=wsdir)
    fit_seconds=time.monotonic()-started
    ws=Workspace(wsdir);assert ws.tgt_stats.read()['is_sequential']
    weights=digest(ws.model_tabular_weights_path)
    ctx=plan[par.columns].copy()
    for c in VIEWS[name]['categorical']:ctx[c]=ctx[c].astype(str)
    generations=[]
    for gs in cfg['generation_seeds']:
        set_random_state(gs);started=time.monotonic()
        generate(ctx_data=ctx,batch_size=64,device='cuda',workspace_dir=wsdir,sampling_temperature=1.,sampling_top_p=1.)
        raw=load_generated_data(wsdir);raw.to_parquet(folder/f'native_{gs}.parquet',index=False)
        output=native_frame(raw);assert set(output.entity_id)<=set(plan.entity_id)
        output.to_parquet(folder/f'generated_raw_{gs}.parquet',index=False)
        generations.append(dict(seed=gs,seconds=time.monotonic()-started,metrics=evaluate(frames['validation'],output,state)))
        assert digest(ws.model_tabular_weights_path)==weights
        print(name,'ARGN GENERATED',gs,len(output),flush=True)
    return dict(model='ARGN',dataset=name,fit_seconds=fit_seconds,weights_sha256=weights,
                native_length=True,outer_validation_given_to_fit=False,results={'raw':dict(generations=generations)})


def train_cpar(name,folder,cfg):
    from sdv.metadata import SingleTableMetadata
    from sdv.sequential import PARSynthesizer
    from experiments.cs_saf_cpar_loss import equivalent_par_loss
    from deepecho.models.par import PARModel
    assert importlib.metadata.version('sdv')==cfg['cpar']['sdv_version']
    assert importlib.metadata.version('deepecho')==cfg['cpar']['deepecho_version']
    inp,parents,ids,frames,plan,state=inputs(name)
    par=parents.loc[parents.entity_id.isin(ids['fit'])].copy()
    columns=['entity_id','gap','receiver_or_mark','amount_or_numeric_value',*VIEWS[name]['auxiliary']]
    child=frames['fit'].sort_values(['entity_id','event_index'],kind='stable')[columns].copy()
    child['gap']=child.gap.fillna(0)
    data=child.merge(par,on='entity_id',validate='many_to_one')
    metadata=SingleTableMetadata();metadata.detect_from_dataframe(data)
    metadata.update_column('entity_id',sdtype='id');metadata.set_sequence_key('entity_id')
    for c in data:
        if c=='entity_id':continue
        categorical=c in ('receiver_or_mark',*VIEWS[name]['auxiliary'],*VIEWS[name]['categorical'])
        metadata.update_column(c,sdtype='categorical' if categorical else 'numerical')
    opts=cfg['cpar'];context_cols=list(par.columns.drop('entity_id'))
    model=PARSynthesizer(metadata,context_columns=context_cols,epochs=opts['epochs'],
                        segment_size=opts['segment_size'],sample_size=1,cuda=True,verbose=True)
    seed(cfg['fit_seed']);started=time.monotonic()
    with equivalent_par_loss():
        original=PARModel._compute_loss
        def bounded(self,*args):
            if time.monotonic()-started>opts['max_fit_seconds']:raise TimeoutError('registered CPAR time budget')
            return original(self,*args)
        PARModel._compute_loss=bounded
        try:model.fit(data)
        finally:PARModel._compute_loss=original
    fit_seconds=time.monotonic()-started;model.save(folder/'model.pkl')
    model.get_loss_values().to_csv(folder/'history.csv',index=False)
    # Same pinned context-transform route as the previously verified wrapper.
    parent=plan[par.columns].copy()
    stub=pd.DataFrame(index=parent.index,columns=data.columns)
    for c in parent:stub[c]=parent[c].to_numpy()
    processed=model._data_processor.transform(stub)[['entity_id',*context_cols]]
    generations=[]
    for gs in cfg['generation_seeds']:
        seed(gs);model._set_random_state(gs);started=time.monotonic();pieces=[]
        for length in sorted(plan.length.unique()):
            selected=plan.length.eq(length)
            pieces.append(model._sample(processed.loc[selected],sequence_length=int(length)))
        raw=pd.concat(pieces,ignore_index=True);raw.to_parquet(folder/f'native_{gs}.parquet',index=False)
        output=native_frame(raw)
        lengths=output.groupby('entity_id').size().reindex(plan.entity_id).to_numpy()
        np.testing.assert_array_equal(lengths,plan.length)
        output.to_parquet(folder/f'generated_raw_{gs}.parquet',index=False)
        generations.append(dict(seed=gs,seconds=time.monotonic()-started,metrics=evaluate(frames['validation'],output,state)))
        print(name,'CPAR GENERATED',gs,len(output),flush=True)
    return dict(model='CPAR',dataset=name,fit_seconds=fit_seconds,epochs=opts['epochs'],
                parameters=sum(p.numel() for p in model._model._model.parameters()),
                results={'raw':dict(generations=generations)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['U','G','ARGN','CPAR','empirical'])
    parser.add_argument('dataset',choices=['berka','sparkov']);a=parser.parse_args()
    cfg=json.loads(CONFIG.read_text());folder=OUT/'runs'/a.dataset/a.model
    assert not folder.exists(),'never overwrite a scientific run'
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    if a.model!='empirical':
        assert torch.cuda.is_available(),'scientific neural fits require admitted GPU'
        torch.cuda.set_per_process_memory_fraction(.88 if a.model=='CPAR' else .5)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    write(folder/'START.json',dict(config_sha256=digest(CONFIG),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=digest(OUT/'input'/a.dataset/'preflight.json'),fit_seed=cfg['fit_seed'],
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),model=a.model,dataset=a.dataset,
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','pandas')},test_outcomes_accessed=False))
    started=time.monotonic()
    try:
        if a.model in ('U','G'):result=train_ug(a.dataset,a.model,folder,cfg)
        elif a.model=='ARGN':result=train_argn(a.dataset,folder,cfg)
        elif a.model=='CPAR':result=train_cpar(a.dataset,folder,cfg)
        else:result=empirical(a.dataset,folder,cfg)
        result.update(total_seconds=time.monotonic()-started,config_sha256=digest(CONFIG),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if a.model!='empirical' else None,
                      files={p.name:digest(p) for p in folder.iterdir() if p.is_file()})
        write(folder/'DONE.json',result);print('DONE',a.dataset,a.model,flush=True)
    except Exception as exc:
        write(folder/'FAILED.json',dict(error_type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
