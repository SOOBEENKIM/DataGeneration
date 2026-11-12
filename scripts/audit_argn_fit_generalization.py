"""Read-only split-wise label fit audit and train-only finite-state references."""
import argparse
import json
import os
import pickle
from pathlib import Path
import time

from run_argn_state_first import ROOT, OUT as BASE, check_manifest, digest, write, seed
import numpy as np
import pandas as pd
import torch
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _fix_rare_token_probs, _translate_fixed_probs
from benchmarks.argn_past_state import PastState, STATE_COLUMN
from benchmarks.argn_state_adapter import FullHistoryCollator, model_class
from benchmarks.argn_label_first_control import label_first_model_class
from benchmarks.argn_frozen_probe import label_probability

OUT=ROOT/'artifacts/research_reaudit_20260927'
DOCS=ROOT/'docs/research_reaudit_20260927'
ARMS=['B','B_S','B_event_weighted','B_label_first','B_event_label_first']
SEEDS=[20260930,20261001]


def folder(arm,fs):
    group=('argn_state_first_v1' if arm in ['B','B_S'] else
           'argn_event_weight_control_v1' if arm=='B_event_weighted' else 'argn_label_first_control_v1')
    return ROOT/'artifacts'/group/'runs'/f'{arm}_{fs}'


def metadata(records,codec):
    rows=[];key=codec.prefixes['event_is_fraud']+'__cat'
    for ri,record in enumerate(records):
        y=(np.asarray(record[key]).reshape(-1)==codec.codes['1']).astype(int)
        prev=np.r_[-1,y[:-1]]; age=np.zeros(len(y),dtype=int)
        for t in range(1,len(y)): age[t]=age[t-1]+1 if t>1 and y[t-1]==y[t-2] else 1
        trans=np.where(prev<0,'first',np.where(prev==1,np.where(y==1,'continuation','termination'),
                        np.where(y==1,'onset','normal_stay')))
        rows.append(pd.DataFrame(dict(record=ri,event_index=np.arange(len(y)),label=y,
            previous_label=prev,prior_age=age,transition=trans)))
    return pd.concat(rows,ignore_index=True)


def inputs():
    check_manifest();ws=Workspace(BASE/'prepared/workspace');codec=PastState(ws.tgt_stats.read())
    frames={}
    for name,suffix in [('optimization','trn'),('internal_validation','val')]:
        path=BASE/f'prepared/workspace/OriginalData/encoded-data/part.000000-{suffix}.parquet'
        frames[name]=pd.read_parquet(path).to_dict('records')
        # Pandas reads nested Arrow lists as object arrays of arrays, whereas
        # the native training loader supplies ordinary nested Python lists.
        for record in frames[name]:
            record[STATE_COLUMN]=np.stack(record[STATE_COLUMN]).astype(np.float32)
    path=ROOT/'artifacts/argn_followup_diagnostics_v1/encoded_development.pkl'
    with path.open('rb') as f:frames['development']=pickle.load(f)['records']
    return ws,codec,frames,{k:metadata(v,codec) for k,v in frames.items()}


def groups(meta):
    yield 'all',np.ones(len(meta),dtype=bool)
    for v in [-1,0,1]:yield f'previous_{v}',meta.previous_label.eq(v).to_numpy()
    for t in ['first','onset','continuation','termination','normal_stay']:
        yield t,meta.transition.eq(t).to_numpy()


def summarize(meta,p,nll,arm,fs,split):
    rows=[]
    for group,mask in groups(meta):
        m=meta[mask];q=np.asarray(p)[mask];loss=np.asarray(nll)[mask];y=m.label.to_numpy()
        if not len(m):continue
        rows.append(dict(arm=arm,fit_seed=fs,split=split,group=group,events=len(m),
            customers=m.record.nunique(),actual_fraud_rate=float(y.mean()),
            mean_fraud_probability=float(q.mean()),brier=float(np.square(q-y).mean()),
            mean_label_nll=float(loss.mean())))
    return rows


def simple_references(metas):
    tr=metas['optimization']
    pooled={prev:(tr.loc[tr.previous_label.eq(prev),'label'].sum()+.5)/(tr.previous_label.eq(prev).sum()+1)
            for prev in [-1,0,1]}
    hazard={}
    for (prev,age),g in tr.assign(age=tr.prior_age.clip(upper=21)).groupby(['previous_label','age']):
        hazard[(int(prev),int(age))]=(g.label.sum()+.5)/(len(g)+1)
    rows=[]
    for split,m in metas.items():
        for arm in ['Markov_train_only','Duration_train_only']:
            p=np.array([pooled[prev] if arm.startswith('Markov') or prev<0 else hazard.get((prev,min(age,21)),pooled[prev])
                for prev,age in zip(m.previous_label,m.prior_age)])
            y=m.label.to_numpy();nll=-(y*np.log(p)+(1-y)*np.log1p(-p))
            rows.extend(summarize(m,p,nll,arm,None,split))
    pd.DataFrame(rows).to_csv(DOCS/'simple_label_references.csv',index=False)
    write(OUT/'simple_label_parameters.json',dict(previous_label=pooled,
        age_hazard={str(k):v for k,v in hazard.items()},fit_split='optimization',smoothing=.5))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'DIAGNOSTIC_MANIFEST.json').exists()
    ws,codec,frames,metas=inputs();simple_references(metas)
    paths=[Path(__file__),DOCS/'PROTOCOL.md',BASE/'prepared/MANIFEST.json',
        ROOT/'artifacts/argn_followup_diagnostics_v1/encoded_development.pkl']
    paths.extend(ws.model_tabular_weights_path for ws in
        [Workspace(folder(arm,fs)/'workspace') for arm in ARMS for fs in SEEDS])
    write(OUT/'DIAGNOSTIC_MANIFEST.json',dict(files={str(p):digest(p) for p in paths},
        split_events={k:len(v) for k,v in metas.items()},split_customers={k:len(v) for k,v in frames.items()},
        test_events_read=False))
    print('PREPARED', {k:len(v) for k,v in metas.items()},flush=True)


@torch.no_grad()
def worker(fs,device):
    seed(fs);started=time.monotonic()
    manifest=json.loads((OUT/'DIAGNOSTIC_MANIFEST.json').read_text())
    amendment=OUT/'AMENDMENT_01.json'
    if amendment.exists():
        amended=json.loads(amendment.read_text())
        assert manifest['files'][str(Path(__file__))]==amended['original_script_sha256']
        manifest['files'][str(Path(__file__))]=amended['corrected_script_sha256']
    for path,sha in manifest['files'].items():assert digest(path)==sha
    shared,codec,frames,metas=inputs();key=codec.prefixes['event_is_fraud']+'__cat'
    dev=torch.device(device);collator=FullHistoryCollator(True,None,dev)
    rows=[];columns=[]
    for arm in ARMS:
        ws=Workspace(folder(arm,fs)/'workspace');ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read()
        assert ts==shared.tgt_stats.read() and cs==shared.ctx_stats.read()
        cls=label_first_model_class(ts) if 'label_first' in arm else model_class(ts,arm=='B_S')
        model=cls(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
            ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
            model_size=ws.model_tabular_configs.read()['model_units'],column_order=None,device=dev)
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=dev);model.to(dev).eval()
        masks=_translate_fixed_probs(_fix_rare_token_probs(ts),ts)
        for split,records in frames.items():
            ps=[];nlls=[];sums={};means={};count=0
            for lo in range(0,len(records),2):
                batch=collator(records[lo:lo+2]);logits,_=model(batch,mode='trn')
                p=label_probability(logits[key],key,codec.codes['1'],masks)
                losses={k:-v.log_softmax(-1).gather(-1,batch[k]).squeeze(-1) for k,v in logits.items()}
                for b,record in enumerate(records[lo:lo+2]):
                    n=len(record[key]);ps.append(p[b,:n].cpu().numpy());nlls.append(losses[key][b,:n].cpu().numpy())
                    count+=n
                    for k,l in losses.items():
                        sums[k]=sums.get(k,0.)+float(l[b,:n].sum())
                        means[k]=means.get(k,0.)+float(l[b,:n].mean())
            rows.extend(summarize(metas[split],np.concatenate(ps),np.concatenate(nlls),arm,fs,split))
            for k in sums:
                columns.append(dict(arm=arm,fit_seed=fs,split=split,subcolumn=k,
                    event_mean_nll=sums[k]/count,customer_mean_nll=means[k]/len(records),
                    note='raw per-token losses; structural training masks are not applied'))
            pd.DataFrame(rows).to_csv(DOCS/f'fit_generalization_{fs}.csv',index=False)
            pd.DataFrame(columns).to_csv(DOCS/f'column_losses_{fs}.csv',index=False)
            print('FIT_AUDIT',arm,fs,split,'seconds',round(time.monotonic()-started,1),flush=True)
        del model;torch.cuda.empty_cache()
    for path,sha in manifest['files'].items():assert digest(path)==sha
    write(OUT/f'COMPLETE_{fs}.json',dict(fit_seed=fs,seconds=time.monotonic()-started,
        models=len(ARMS),weights_unchanged=True,test_events_read=False,pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:worker(a.seed,a.device)
