"""Evaluation-only localization of frozen U/G+A; no fitting or new sampling."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from experiments import cs_saf_rollout_calibration_v1 as parent
from benchmarks.cs_saf_joint_oracle import predict as oracle_predict
from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from models.cs_saf_observed_repeat_control import apply_numpy, assign_bins
from scripts.evaluate_cs_saf_rollout_calibration_v1 import inputs, validate_frame
from scripts.run_cs_saf_u_repeat_controls import extract
from scripts.run_cs_saf_external_audit_v1 import digest, write

ROOT = parent.ROOT
DOC = ROOT/'docs/cs_saf/history_diagnostic_v1'
OUT = ROOT/'artifacts/cs_saf/history_diagnostic_v1'
SELF = 'experiments/cs_saf_history_diagnostic_v1.py'
PREREG = 'docs/cs_saf/history_diagnostic_v1/preregistration.md'
FIELDS = ('gap', 'receiver', 'numeric_value', 'valid_mask', 'lengths', 'codes')
BINS = {'history': [1, 5, 9, 17, 32], 'run': [1, 2, 4, 8, 33]}
LABELS = {'history': ['1-4', '5-8', '9-16', '17-31'], 'run': ['1', '2-3', '4-7', '8+']}
BOOT = 400


def source():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = list(dict.fromkeys(parent.SOURCES + [SELF, PREREG,
        'benchmarks/cs_saf_joint_oracle.py', 'benchmarks/cs_saf_oracle.py',
        'scripts/evaluate_cs_saf_rollout_calibration_v1.py', 'benchmarks/temporal_coupling_v2.py',
        'benchmarks/semi_markov.py', 'configs/benchmark_v2/full_v2_5.yaml']))
    for p in paths:
        assert (ROOT/p).read_bytes() == subprocess.check_output(['git', 'show', f'{commit}:{p}'], cwd=ROOT), p
    return dict(commit=commit, files={p:digest(ROOT/p) for p in paths})


def tensor_data(data):
    return {key:data[key] for key in FIELDS}


def from_frame(frame, plan, state):
    """Invert saved decoded events with the original frozen codecs, preserving plan order."""
    validate_frame(frame, plan)
    n = len(plan); steps = 32
    rows = pd.Index(plan.entity_id).get_indexer(frame.entity_id)
    cols = frame.event_index.to_numpy(int)
    assert (rows >= 0).all() and not frame.duplicated(['entity_id','event_index']).any()
    lengths = torch.tensor(plan['__saf_planned_length'].to_numpy(), dtype=torch.long)
    valid = torch.arange(steps)[None,:] < lengths[:,None]
    assert np.array_equal(np.bincount(rows, minlength=n), lengths.numpy())
    assert valid[rows,cols].all()
    x = dict(gap=torch.full((n,steps),float('nan')), receiver=torch.zeros((n,steps),dtype=torch.long),
        numeric_value=torch.zeros((n,steps)), valid_mask=valid, lengths=lengths,
        codes=torch.tensor(plan.entity_label.astype(int).to_numpy()+3))
    x['gap'][rows,cols] = torch.tensor(frame.gap.to_numpy(), dtype=torch.float32)
    x['receiver'][rows,cols] = torch.from_numpy(state.receiver_codec.encode(frame.receiver_or_mark))
    x['numeric_value'][rows,cols] = torch.from_numpy(state.event_numeric_codecs[0][1].encode(frame.amount_or_numeric_value))
    assert (x['receiver'][valid] >= 3).all()
    return x


def events(data):
    """State uses the strict prefix. End-of-sequence censoring adds no outcome."""
    marks = data['receiver'].numpy(); valid = data['valid_mask'].numpy().copy()
    valid[:,0] = False
    run = np.ones_like(marks)
    for t in range(1,marks.shape[1]):
        run[:,t] = np.where(marks[:,t] == marks[:,t-1],run[:,t-1]+1,1)
    i,t = np.where(valid)
    return pd.DataFrame(dict(entity=i, event_index=t, history=t, run=run[i,t-1],
        group=data['codes'].numpy()[i]-3, gap=data['gap'].numpy()[i,t],
        y=(marks[i,t] == marks[i,t-1]).astype(float)))


def slices(frame):
    for group in (0,1):
        gm = frame.group.to_numpy() == group
        yield group,'all','all',gm
        for dimension,boundaries in BINS.items():
            values = frame[dimension].to_numpy()
            for j,label in enumerate(LABELS[dimension]):
                yield group,dimension,label,gm & (values >= boundaries[j]) & (values < boundaries[j+1])


def coverage(frame):
    return [dict(group=g,dimension=d,bin=b,entities=int(frame.loc[m,'entity'].nunique()),
        transitions=int(m.sum()),adequate=bool(frame.loc[m,'entity'].nunique()>=50 and m.sum()>=200))
        for g,d,b,m in slices(frame)]


def oracle_for(k,support):
    raw = yaml.safe_load((ROOT/'configs/benchmark_v2/full_v2_5.yaml').read_text())
    cfg = BenchmarkConfig.from_mapping(raw,'joint_semimarkov_v2b',k)
    return SemiMarkovCopyOracle(cfg,np.asarray(support.upper_bounds,dtype=np.float32).astype(float))


@torch.no_grad()
def replay(model, data, control, batch_size=128):
    frame = events(data); pieces=[]; normal=0.
    for start in range(0,len(data['lengths']),batch_size):
        x={k:v[start:start+batch_size] for k,v in data.items() if k in FIELDS}
        h=model.encoder(x['gap'],x['receiver'],x['numeric_value'],x['valid_mask'],static_categorical=(x['codes'],))
        mask=x['valid_mask'].clone();mask[:,0]=False
        ctx=model.context(h,(x['codes'],))[mask]
        codes=x['codes'][:,None].expand_as(mask)[mask]
        previous=x['receiver'].roll(1,1)[mask]; actual=x['receiver'][mask]
        lp,_,_=model.mark_distribution(ctx,x['gap'][mask],previous,torch.ones_like(previous,dtype=torch.bool),static_codes=codes)
        p=lp.exp();normal=max(normal,float((p.sum(-1)-1).abs().max()))
        assert (p[:,:3]==0).all() and normal<1e-6
        gapcode=(model._support_code(x['gap'][mask])-3).clamp_min(0)
        r=p.gather(1,previous[:,None])[:,0]
        adjusted=parent.apply_torch(r,gapcode,codes-3,control)
        p=parent.redistribute(p,previous,adjusted)
        # Native inverse-CDF sampling renormalizes in float64; match it here.
        p=p.double();p=p/p.sum(-1,keepdim=True)
        pieces.append(np.stack([p.gather(1,previous[:,None])[:,0].numpy(),
            -p.gather(1,actual[:,None])[:,0].log().numpy()],1))
    values=np.concatenate(pieces)
    assert len(values)==len(frame)
    frame['p']=values[:,0];frame['mark_nll']=values[:,1]
    return frame,normal


def bootstrap_weights(frame, seed):
    """Resample complete sequences, separately within groups; common across predictors."""
    rng=np.random.default_rng(seed);result={}
    for g in (0,1):
        unique=np.unique(frame.loc[frame.group==g,'entity']);n=len(unique)
        draws=rng.integers(0,n,(BOOT,n))
        weights=np.zeros((BOOT,n),dtype=np.int16)
        np.add.at(weights,(np.arange(BOOT)[:,None],draws),1)
        result[g]=(unique,weights)
    return result


def cluster_draws(frame,mask,values,g,weights):
    unique,w=weights[g]
    positions=np.searchsorted(unique,frame.loc[mask,'entity'].to_numpy())
    counts=np.bincount(positions,minlength=len(unique))
    sums=np.bincount(positions,weights=np.asarray(values)[mask],minlength=len(unique))
    den=w@counts;num=w@sums
    return np.divide(num,den,out=np.full(BOOT,np.nan),where=den>0)


def summarize(frame,metadata,weights,prediction=False):
    rows=[];boots=[]
    for (g,d,b,m),cov in zip(slices(frame),coverage(frame)):
        n=int(m.sum());row=dict(**metadata,**cov)
        if not n:
            rows.append(row);boots.append(np.full(BOOT,np.nan));continue
        groupmask=frame.group.to_numpy()==g
        y=frame.y.to_numpy();q=frame.q.to_numpy();qalt=frame.q_alt.to_numpy()
        row.update(continuation=float(y[m].mean()),oracle_continuation=float(q[m].mean()),
            occupancy=float(n/groupmask.sum()),law_residual=float((y-q)[m].mean()),
            oracle_information_shift=float(abs(q-qalt)[m].mean()))
        cd=cluster_draws(frame,m,y,g,weights)
        row.update(continuation_lo=float(np.nanquantile(cd,.025)),continuation_hi=float(np.nanquantile(cd,.975)))
        od=cluster_draws(frame,groupmask,m.astype(float),g,weights)
        row.update(occupancy_lo=float(np.nanquantile(od,.025)),occupancy_hi=float(np.nanquantile(od,.975)))
        if prediction:
            p=frame.p.to_numpy();bias=p-q
            bd=cluster_draws(frame,m,bias,g,weights)
            row.update(bias=float(bias[m].mean()),mae=float(abs(bias[m]).mean()),
                squared_probability_error=float((bias[m]**2).mean()),
                brier=float(((p-y)[m]**2).mean()),mark_nll=float(frame.loc[m,'mark_nll'].mean()),
                sampled_residual=float((y-p)[m].mean()),mean_prediction=float(p[m].mean()),
                alternate_bias=float((p-qalt)[m].mean()),alternate_mae=float(abs(p-qalt)[m].mean()),
                bias_lo=float(np.nanquantile(bd,.025)),bias_hi=float(np.nanquantile(bd,.975)))
            assert abs(row['law_residual']-row['bias']-row['sampled_residual'])<1e-12
            boots.append(bd)
        else:boots.append(cd)
        rows.append(row)
    return rows,boots


def detailed(frame,meta,edges):
    code=frame.gap_code.to_numpy();group=frame.group.to_numpy()
    cb=assign_bins(code,group,edges);out=[]
    for g,d,b,m in slices(frame):
        if d=='all':continue
        for gapbin in range(5):
            ix=m & (cb==gapbin)
            out.append(dict(**meta,group=g,dimension=d,bin=b,gapbin=gapbin,
                entities=int(frame.loc[ix,'entity'].nunique()),transitions=int(ix.sum()),
                bias=float((frame.p-frame.q)[ix].mean()),mae=float(abs(frame.p-frame.q)[ix].mean())))
    return out


def validate_sources(k):
    inventory={}
    for t in range(3):
        for name in ('U','G'):
            dest=parent.folder(k,t,name)/'evaluation'
            done=json.loads((dest/'DONE.json').read_text())
            for entry in done['generations']:
                if entry['variant']!='A':continue
                path=dest/entry['file'];assert digest(path)==entry['sha256']
                inventory[str(path.relative_to(ROOT))]=entry['sha256']
    dest=parent.OUT/f'oracle/kappa_{k}'
    done=json.loads((dest/'DONE.json').read_text())
    for entry in done['inventory']:
        path=dest/entry['file'];assert digest(path)==entry['sha256']
        inventory[str(path.relative_to(ROOT))]=entry['sha256']
    return inventory


def audit():
    from models.cs_saf_structure import constrained_repeat_logits
    from scripts.check_cs_saf_structure_v1 import forward
    torch.set_num_threads(1);torch.manual_seed(20264600)
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    src=source();checks=[];coverage_rows=[];inventory={}
    args=(torch.randn(2,5,dtype=torch.float64,requires_grad=True),
        torch.randn(2,dtype=torch.float64,requires_grad=True),torch.randn(2,5,dtype=torch.float64,requires_grad=True))
    assert torch.autograd.gradcheck(constrained_repeat_logits,args,eps=1e-6,atol=1e-5,rtol=1e-4)
    for k in (0,1):
        inventory.update(validate_sources(k))
        for t in range(3):
            for name in ('U','G'):
                m,payload,a,features,provenance=parent.load_parent(k,t,name,'cpu')
                tr,va=payload['train'],payload['validation']
                assert not set(tr['entity_ids']) & set(va['entity_ids'])
                if t==0 and name=='U':
                    coverage_rows += [dict(kappa=k,**r) for r in coverage(events(tr))]
                ix=torch.cat([torch.where(tr['codes']==g)[0][:4] for g in (3,4)])
                x=parent.parent.batch(tr,ix,'cpu')
                hidden,ctx,prev,mask,codes,(lp,lr,lnr)=forward(m,x)
                torch.testing.assert_close(lp.exp().sum(-1),torch.ones_like(mask,dtype=lp.dtype),atol=1e-6,rtol=0)
                rr=lp.exp()[mask].gather(1,prev[mask][:,None])[:,0]
                torch.testing.assert_close(rr,lr[mask].exp(),atol=3e-7,rtol=0)
                altered={key:(v.clone() if isinstance(v,torch.Tensor) else v) for key,v in x.items()}
                altered['receiver'][:,4:]=3+(altered['receiver'][:,4:]-3+19)%64
                altered['numeric_value'][:,4:]+=17
                altered['gap'][:,5:]=m.support.representatives[-1]
                torch.testing.assert_close(lp[:,:5],forward(m,altered)[-1][0][:,:5],atol=0,rtol=0)
                plan=dict(lengths=tr['lengths'][ix],codes=tr['codes'][ix])
                # No new random generation: replay the saved generated table and strict prefixes.
                _,_,_,_,savedplan,_=inputs(k)
                state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
                path=parent.folder(k,t,name)/'evaluation/generated_A_20264401.parquet'
                frame=pd.read_parquet(path);smallplan=savedplan.iloc[:8]
                data=from_frame(frame[frame.entity_id.isin(smallplan.entity_id)],smallplan,state)
                pred,_=replay(m,data,a,batch_size=4)
                h,st=parent.initial_hidden(m,data['codes']);stream=[]
                for step in range(1,32):
                    h,st=parent.stream_event(m,data['gap'][:,step-1],data['receiver'][:,step-1],data['numeric_value'][:,step-1],data['valid_mask'][:,step-1],st)
                    active=data['valid_mask'][:,step]
                    p=parent.corrected_probabilities(m,h,data['codes'],data['gap'][:,step],data['receiver'][:,step-1],active,a).double()
                    p=p/p.sum(-1,keepdim=True)
                    stream.append(p.gather(1,data['receiver'][:,step-1,None])[:,0].numpy())
                stream=np.stack(stream,1)
                error=float(abs(pred.p.to_numpy()-stream[pred.entity,pred.event_index-1]).max())
                assert error<3e-6
                # Independently recompute old actual-history features on first 32 entities.
                fresh=extract(m,va,32)
                oldpath=parent.parent.folder_for(k,t,name)/'evaluation/validation_features.parquet'
                olddone=json.loads((oldpath.parent.parent/'EVAL_DONE.json').read_text())
                assert digest(oldpath)==olddone['files']['validation_features.parquet']
                old=pd.read_parquet(oldpath).iloc[:len(fresh)]
                for field in ('entity_id','event_index','y','group','gap_code'):
                    np.testing.assert_array_equal(fresh[field],old[field])
                ferr=float(abs(fresh.p.to_numpy()-old.p.to_numpy()).max());assert ferr<1e-6
                assert parent.state_digest(m)==provenance['state_sha256']
                # Check stored selection, fit split, and historical gate source hashes.
                report=json.loads((oldpath.parent.parent/'TRAIN_DONE.json').read_text())
                best=min(report['history'],key=lambda row:row['validation']['base_nll'])['epoch']
                assert best==report['best_epoch']
                assert a['fit_split']=='outer_train' and not a['validation_used'] and not a['oracle_used']
                checks.append(dict(kappa=k,trial=t,model=name,parameters=sum(p.numel() for p in m.parameters()),
                    stream_replay_max_error=error,stored_feature_max_error=ferr,state_unchanged=True,strict_past=True,
                    normalized=True,observable_repeat_identity=True,selection_correct=True))
    # Existing mathematical/CPU gates refer to exactly the same historical implementation.
    for gate in ('structure_v1','rollout_calibration_v1'):
        g=json.loads((ROOT/f'docs/cs_saf/{gate}/cpu_gate.json').read_text())
        for path,h in g['source_hashes'].items():assert digest(ROOT/path)==h,path
    pd.DataFrame(coverage_rows).to_csv(DOC/'train_coverage.csv',index=False)
    write(DOC/'audit.json',dict(status='PASS',source=src,checks=checks,C_implicit_gradient_check=True,
        source_inventory=inventory,no_fit=True,no_new_generation=True,train_coverage_only=True,time=time.time()))
    print('AUDIT_PASS',len(checks),'parents; input files',len(inventory),flush=True)


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    src=source();gate=json.loads((DOC/'audit.json').read_text())
    assert gate['status']=='PASS' and gate['source']['files']==src['files']
    assert not (OUT/'DONE.json').exists()
    started=time.time();write(OUT/'START.json',dict(source=src,time=started))
    predrows=[];predboots=[];genrows=[];detail=[];normal=0.;evaluations=0
    for k in (0,1):
        validate_sources(k)
        payload,_=parent.parent.payload_for(k);state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
        truth=oracle_for(k,state.gap_support);_,_,_,_,plan,_=inputs(k)
        real=tensor_data(payload['validation']);realframe=events(real)
        rr,tt=realframe.entity.to_numpy(),realframe.event_index.to_numpy()
        realframe['q']=oracle_predict(truth,real,k,'MODELINFO')[rr,tt]
        realframe['q_alt']=realframe.q
        realweights=bootstrap_weights(realframe,20264601+k)
        rows,_=summarize(realframe,dict(kappa=k,source='real',trial=-1,tape=-1),realweights)
        genrows+=rows
        for t in range(3):
            models={};controls={};provenances={}
            for name in ('U','G'):
                models[name],_,controls[name],_,provenances[name]=parent.load_parent(k,t,name,'cpu')
            sources=[('real',-1,real,realframe,realweights)]
            for name in ('U','G'):
                for tape in parent.config()['final_generation_seeds']:
                    path=parent.folder(k,t,name)/f'evaluation/generated_A_{tape}.parquet'
                    x=from_frame(pd.read_parquet(path),plan,state);f=events(x)
                    rr,tt=f.entity.to_numpy(),f.event_index.to_numpy()
                    f['q']=oracle_predict(truth,x,k,'BIN')[rr,tt]
                    f['q_alt']=oracle_predict(truth,x,k,'MODELINFO')[rr,tt]
                    seed=20264601+k*10000+t*100+10*(name=='G')+tape%10
                    weights=bootstrap_weights(f,seed)
                    sources.append((name,tape,x,f,weights))
                    rows,_=summarize(f,dict(kappa=k,source=name,trial=t,tape=tape),weights);genrows+=rows
            for hist,tape,x,f,weights in sources:
                for name,m in models.items():
                    p,err=replay(m,x,controls[name]);normal=max(normal,err);evaluations+=1
                    for field in ('entity','event_index','y'):np.testing.assert_array_equal(p[field],f[field])
                    p['q']=f.q;p['q_alt']=f.q_alt
                    p['gap_code']=np.searchsorted(np.asarray(state.gap_support.upper_bounds,dtype=np.float32),p.gap,side='left')
                    meta=dict(kappa=k,trial=t,source=hist,tape=tape,predictor=name)
                    rows,bs=summarize(p,meta,weights,prediction=True);predrows+=rows;predboots+=bs
                    detail+=detailed(p,meta,controls[name]['edges'])
                    p.to_parquet(OUT/f'events_k{k}_t{t}_{hist}_{tape}_{name}.parquet',index=False)
                    print('REPLAY',k,t,hist,tape,name,flush=True)
            for name,m in models.items():assert parent.state_digest(m)==provenances[name]['state_sha256']
        for mode in ('JOINT_BIN','JOINT_CONT'):
            for tape in parent.config()['oracle_seeds']:
                x=from_frame(pd.read_parquet(parent.OUT/f'oracle/kappa_{k}/generated_{mode}_{tape}.parquet'),plan,state)
                f=events(x);rr,tt=f.entity.to_numpy(),f.event_index.to_numpy()
                f['q']=oracle_predict(truth,x,k,'BIN' if mode=='JOINT_BIN' else 'CONT')[rr,tt];f['q_alt']=f.q
                weights=bootstrap_weights(f,20264601+k*10000+tape)
                rows,_=summarize(f,dict(kappa=k,source=mode,trial=-1,tape=tape),weights);genrows+=rows
            print('ORACLE_SUMMARY',k,mode,flush=True)
    assert evaluations==84
    pd.DataFrame(predrows).to_csv(DOC/'prediction_by_tape.csv',index=False)
    np.savez_compressed(OUT/'prediction_bootstrap.npz',bias=np.stack(predboots))
    pd.DataFrame(genrows).to_csv(DOC/'generation_by_tape.csv',index=False)
    pd.DataFrame(detail).to_csv(DOC/'gap_conditioned_prediction.csv',index=False)
    write(OUT/'DONE.json',dict(source=src,start_time=started,end_time=time.time(),predictor_evaluations=evaluations,
        stored_model_datasets=36,stored_oracle_datasets=120,normalization_max_error=normal,
        new_neural_fits=0,new_calibration_fits=0,new_generations=0,gpu_used=False,test_access=False,
        files={str(p.relative_to(ROOT)):digest(p) for p in [*OUT.glob('events_*.parquet'),OUT/'prediction_bootstrap.npz',
            DOC/'prediction_by_tape.csv',DOC/'generation_by_tape.csv',DOC/'gap_conditioned_prediction.csv']}))
    print('DIAGNOSTIC_COMPLETE',time.time()-started,flush=True)
