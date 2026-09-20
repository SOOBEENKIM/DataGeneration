"""Evaluation-only phase, locked until every registered correction is saved."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from experiments import cs_saf_rollout_calibration_v1 as run
from experiments.cs_saf_generation_metrics import CachedGroupMetrics
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from scripts.run_cs_saf_u_repeat_controls import to_frame
from scripts.cs_saf_repeat_control_common import conditional_metrics
from scripts.run_cs_saf_external_audit_v1 import digest,write

EVAL_SOURCES=['scripts/evaluate_cs_saf_rollout_calibration_v1.py','benchmarks/cs_saf_joint_oracle.py',
    'benchmarks/cs_saf_oracle.py','benchmarks/temporal_coupling_v2.py','configs/benchmark_v2/full_v2_5.yaml',
    'benchmarks/cof_seqgen_saf_metrics.py','benchmarks/semi_markov.py']


def source():
    record=run.sources();commit=record['commit']
    for f in EVAL_SOURCES:
        assert (ROOT/f).read_bytes()==subprocess.check_output(['git','show',f'{commit}:{f}'],cwd=ROOT)
    record['hashes'].update({f:digest(ROOT/f) for f in EVAL_SOURCES})
    return record


def inputs(k):
    inp=run.parent.OLD/f'kappa_{k}/input';pr=json.loads((inp/'provenance.json').read_text())
    for f,h in pr['files'].items():assert digest(inp/f)==h
    tr=pd.read_parquet(inp/'train_events.parquet');tr['event_index']=tr.groupby('entity_id').cumcount()
    tr.loc[tr.event_index==0,'gap']=np.nan
    val=pd.read_parquet(inp/'validation_canonical.parquet')
    tc=pd.read_parquet(inp/'train_context.parquet');vc=pd.read_parquet(inp/'validation_context.parquet')
    plan=pd.read_parquet(inp/'plan.parquet')
    assert not set(tc.entity_id)&set(vc.entity_id)
    assert len(plan)==run.config()['final_generation_entities']
    return tr,val,tc,vc,plan,pr


def evaluators(k,support=None,quantized=False):
    tr,val,tc,vc,plan,pr=inputs(k)
    if quantized:
        val=val.copy();mask=np.isfinite(val.gap)
        upper=np.asarray(support.upper_bounds,dtype=np.float32)
        val.loc[mask,'gap']=np.asarray(support.representatives,dtype=np.float32)[np.searchsorted(upper,val.loc[mask,'gap'],side='left')]
    result={}
    for g in ['pooled','0','1']:
        def ids(frame):return set(frame.entity_id if g=='pooled' else frame[frame.entity_label.astype(str)==g].entity_id)
        ti,vi,pi=ids(tc),ids(vc),ids(plan)
        reference=val[val.entity_id.isin(vi)]
        result[g]=(CachedGroupMetrics(tr[tr.entity_id.isin(ti)],reference),pi,reference)
    return result


def score(frame,ev,metadata):
    rows=[];curves=[]
    for g,(metric,ids,reference) in ev.items():
        part=frame[frame.entity_id.isin(ids)];scores=metric.score(part)
        a=metric.reference['repeat'];b=metric.sufficient_statistics(part)['repeat']
        assert (b.sum(1)>0).all()
        w=a.sum(1)/a.sum();real=a[:,1]/a.sum(1);fake=b[:,1]/b.sum(1)
        centered=float(w@abs((fake-w@fake)-(real-w@real)))
        rows.append(dict(**metadata,group=g,**scores,null_centered_shape_l1=centered,
            generated_repeat_level=float(w@fake),reference_repeat_level=float(w@real)))
        curve_metadata={('reference_kind' if key=='reference' else key):value for key,value in metadata.items()}
        for j in range(5):curves.append(dict(**curve_metadata,group=g,bin=j,reference=float(real[j]),generated=float(fake[j])))
    return rows,curves


def validate_frame(frame,plan):
    assert set(frame.entity_id)==set(plan.entity_id)
    np.testing.assert_array_equal(frame.groupby('entity_id').size().reindex(plan.entity_id),plan['__saf_planned_length'])
    assert np.isfinite(frame.amount_or_numeric_value).all()
    first=frame.groupby('entity_id').cumcount()==0
    assert frame.loc[first,'gap'].isna().all()
    assert np.isfinite(frame.loc[~first,'gap']).all() and (frame.loc[~first,'gap']>=0).all()


def evaluate(k,t,name):
    fitting=run.all_fitted();src=source();started=time.time()
    dest=run.folder(k,t,name)/'evaluation';dest.mkdir(exist_ok=False)
    model,payload,a,features,provenance=run.load_parent(k,t,name,'cpu')
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    controls={'A':a}
    for m in ['B','P']:controls[m]=json.loads((dest.parent/f'{m}_DONE.json').read_text())['control']
    old=run.parent.folder_for(k,t,name)
    oldmanifest=json.loads((old/'EVAL_DONE.json').read_text())
    vfpath=old/'evaluation/validation_features.parquet'
    assert digest(vfpath)==oldmanifest['files']['validation_features.parquet']
    vf=pd.read_parquet(vfpath)
    write(dest/'conditional.json',conditional_metrics(features,vf,controls))
    write(dest/'response.json',run.parent.response_diagnostic(model,payload['validation'],controls))
    tr,val,tc,vc,plan,pr=inputs(k);ev=evaluators(k)
    assert set(vf.entity_id)==set(vc.entity_id)
    positions=pd.Index(payload['train']['entity_ids']).get_indexer(plan.source_train_entity_id)
    assert (positions>=0).all()
    genplan=dict(lengths=payload['train']['lengths'][positions],codes=payload['train']['codes'][positions])
    np.testing.assert_array_equal(genplan['lengths'],plan['__saf_planned_length'])
    np.testing.assert_array_equal(genplan['codes']-3,plan.entity_label.astype(int))
    rows=[];curves=[];files=[]
    for method,control in controls.items():
        for seed in run.config()['final_generation_seeds']:
            sample=run.generate(model,genplan,seed,control,'cpu')
            frame=to_frame(sample,payload,state,plan);validate_frame(frame,plan)
            path=dest/f'generated_{method}_{seed}.parquet';frame.to_parquet(path,index=False)
            meta=dict(kappa=k,trial=t,name=name,variant=method,tape=seed)
            scores,points=score(frame,ev,meta);rows.extend(scores);curves.extend(points)
            files.append(dict(file=path.name,sha256=digest(path),**meta))
            print('EVALUATED',k,t,name,method,seed,flush=True)
    pd.DataFrame(rows).to_csv(dest/'generation_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(dest/'repeat_curves.csv',index=False)
    assert run.state_digest(model)==provenance['state_sha256']
    write(dest/'DONE.json',dict(source=src,provenance=provenance,all_fits=fitting,start_time=started,end_time=time.time(),
        generations=files,weights_unchanged=True,test_accessed=False,
        files={p.name:digest(p) for p in dest.iterdir() if p.is_file()}))


def oracle(k):
    # Imported only after checking every fit. This module is never called by fitting.
    fitting=run.all_fitted();src=source();started=time.time()
    import yaml
    from benchmarks.cs_saf_joint_oracle import sample,predict,monte_carlo_gate
    from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
    from benchmarks.temporal_coupling_v2 import BenchmarkConfig
    payload,_=run.parent.payload_for(k);state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    support=state.gap_support
    raw=yaml.safe_load((ROOT/'configs/benchmark_v2/full_v2_5.yaml').read_text())
    cfg=BenchmarkConfig.from_mapping(raw,'joint_semimarkov_v2b',k)
    truth=SemiMarkovCopyOracle(cfg,np.asarray(support.upper_bounds,dtype=np.float32).astype(float))
    gate=monte_carlo_gate(truth,support.representatives,n_per_group=2048,seed=20264901)
    dest=run.OUT/f'oracle/kappa_{k}';dest.mkdir(parents=True,exist_ok=False)
    write(dest/'sampler_gate.json',gate)
    tr,val,tc,vc,plan,pr=inputs(k);raw_ev=evaluators(k);bin_ev=evaluators(k,support,True)
    lengths=torch.tensor(plan['__saf_planned_length'].to_numpy(),dtype=torch.long)
    codes=torch.tensor(plan.entity_label.astype(int).to_numpy()+3,dtype=torch.long)
    valid=torch.arange(32)[None,:]<lengths[:,None];n=len(lengths)
    rows=[];curves=[];inventory=[];replay_errors=[]
    for mode in run.config()['oracle_modes']:
        for seed in run.config()['oracle_seeds']:
            # Independent extra streams, not observed initial marks or values.
            first_rng,value_rng=[np.random.default_rng(s) for s in np.random.SeedSequence([seed,91827]).spawn(2)]
            marks=torch.full((n,32),3,dtype=torch.long);marks[:,0]=torch.from_numpy(first_rng.integers(3,67,n))
            amount=value_rng.normal(cfg.amount_mean,cfg.amount_std,(n,32))
            data=dict(gap=torch.full((n,32),float('nan')),receiver=marks,numeric_value=torch.zeros((n,32)),
                valid_mask=valid.clone(),lengths=lengths.clone(),codes=codes.clone())
            result,online=sample(truth,data,k,support.representatives,mode,seed)
            replay=predict(truth,result,k,'BIN' if mode.endswith('BIN') else 'CONT')
            error=float(np.max(abs(online-replay)));assert error<1e-12;replay_errors.append(error)
            mask=result['valid_mask'].numpy();rr,cc=np.where(mask)
            frame=pd.DataFrame(dict(entity_id=plan.entity_id.to_numpy()[rr],event_index=cc,
                gap=result['gap'].numpy()[mask].astype(float),receiver_or_mark=state.receiver_codec.decode(result['receiver'].numpy()[mask]),
                amount_or_numeric_value=amount[mask]))
            validate_frame(frame,plan)
            path=dest/f'generated_{mode}_{seed}.parquet';frame.to_parquet(path,index=False)
            for reference,ev in [('raw',raw_ev)]+([('quantized',bin_ev)] if mode=='JOINT_BIN' else []):
                meta=dict(kappa=k,mode=mode,tape=seed,reference=reference)
                values,points=score(frame,ev,meta);rows.extend(values);curves.extend(points)
            inventory.append(dict(file=path.name,sha256=digest(path),mode=mode,tape=seed))
            print('ORACLE_EVALUATED',k,mode,seed,flush=True)
    pd.DataFrame(rows).to_csv(dest/'generation_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(dest/'repeat_curves.csv',index=False)
    write(dest/'DONE.json',dict(source=src,all_fits=fitting,start_time=started,end_time=time.time(),
        inventory=inventory,independent_first_marks_and_numeric_values=True,
        observable_filter_prediction_replay_max_error=max(replay_errors),evaluation_only=True,test_accessed=False,
        files={p.name:digest(p) for p in dest.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['model','oracle'])
    p.add_argument('--kappa',type=int,required=True);p.add_argument('--trial',type=int);p.add_argument('--model',choices=['U','G'])
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.phase=='oracle':oracle(a.kappa)
    else:evaluate(a.kappa,a.trial,a.model)
