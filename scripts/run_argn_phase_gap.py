"""A simple all-gap conditional mixture, with frozen transaction structure."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from run_argn_boundary_gap import generation_args
from run_argn_amount_learning import ROOT,BASE,Workspace,folder,seed,write,digest
from diagnose_argn_residual import verify_registry,generation_path
from benchmarks.argn_phase_gap import phase_gap_generation,encode_values

OUT=ROOT/'artifacts/argn_phase_gap_v1';DOCS=ROOT/'docs/argn_phase_gap_v1'
CFG=dict(fit_seeds=[20260930,20261001],generation_seeds=[20261011,20261012],
    allowed_gpu_uuids=['GPU-f1d4c556-ae70-3415-adec-c1d68f9bb637','GPU-5eae2fe0-7b63-fcc9-dd31-4e915ab63073'])


def verify():
    verify_registry();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    verify_registry();assert not (OUT/'MANIFEST.json').exists();OUT.mkdir(exist_ok=True,parents=True)
    source=ROOT/'artifacts/argn_residual_v1';frames={s:pd.read_parquet(source/f'metadata_{s}.parquet') for s in ['optimization','internal_validation']}
    stats=Workspace(BASE/'prepared/workspace').tgt_stats.read()['columns']['gap']
    model=dict(phases={},limits=[min(stats['min5']),max(stats['max5'])]);rows=[]
    for phase in range(4):
        use={s:d[d.event_index.gt(0)&(2*d.previous+d.label).eq(phase)] for s,d in frames.items()}
        train=use['optimization'];val=use['internal_validation'];tr=train.sample(min(len(train),50000),random_state=20261115)
        fits=[]
        for k in ([1,2,3] if len(train)>=20 else [1]):
            m=GaussianMixture(k,random_state=20261115,n_init=5,reg_covar=1e-4).fit(np.log1p(tr.gap.to_numpy())[:,None]);nll=-m.score(np.log1p(val.gap.to_numpy())[:,None])
            rows.append(dict(phase=phase,k=k,internal_nll=nll,available_train=len(train),used_train=len(tr),internal_events=len(val)));fits.append((nll,k,m))
        _,k,m=min(fits,key=lambda x:x[0]);model['phases'][str(phase)]=dict(k=k,weights=m.weights_.tolist(),means=m.means_.reshape(-1).tolist(),variances=m.covariances_.reshape(-1).tolist())
    write(OUT/'parameters.json',model);write(DOCS/'parameters.json',model);pd.DataFrame(rows).to_csv(DOCS/'selection.csv',index=False)
    # Check the real codec on a broad deterministic grid before any generation.
    ts=Workspace(BASE/'prepared/workspace').tgt_stats.read()
    from mostlyai.engine._common import get_cardinalities
    cards=get_cardinalities(ts);prefix=f"{stats['argn_processor']}:{stats['argn_table']}/{stats['argn_column']}__";keys=[k for k in cards if k.startswith(prefix)]
    v=np.linspace(*model['limits'],10001);tokens=encode_values(v,stats,keys,cards)
    reconstructed=sum((tokens[k]+stats['min_digits'][k.rsplit('__',1)[1]])*10.**int(k.rsplit('__E',1)[1]) for k in keys)
    assert np.allclose(reconstructed,np.floor(v/10.**stats['min_decimal']+1e-8)*10.**stats['min_decimal'])
    write(DOCS/'CODEC_CHECK.json',dict(positions=len(v),roundtrip=True))
    paths=[Path(__file__),ROOT/'benchmarks/argn_phase_gap.py',ROOT/'tests/test_argn_phase_gap.py',ROOT/'scripts/dispatch_argn_phase_gap.py',DOCS/'PROTOCOL.md',
        ROOT/'scripts/run_argn_boundary_gap.py',OUT/'parameters.json']+[source/f'metadata_{s}.parquet' for s in frames]
    m=dict(config=CFG,created_utc=datetime.now(timezone.utc).isoformat(),hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('PHASE_PREPARED',flush=True)


def worker(fs,device):
    verify();from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    run=OUT/f'runs/phase_mixture_{fs}';run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    ws=Workspace(run/'workspace');before=digest(ws.model_tabular_weights_path);context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    args,kwargs=generation_args(ws,fs);p=json.loads((OUT/'parameters.json').read_text())
    with phase_gap_generation(*args,**kwargs,phase_parameters=p) as audit:
        for gs in CFG['generation_seeds']:
            seed(gs);start=time.monotonic();prior_audits=len(audit);generate(ctx_data=context,device=device,workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            d=raw.rename(columns={'customer_id':'entity_id'});d['event_index']=d.groupby('entity_id',sort=False).cumcount();d.loc[d.event_index.eq(0),'gap']=np.nan
            path=run/f'generated_validation_{gs}.parquet';d.to_parquet(path,index=False)
            baseline=pd.read_parquet(generation_path('onset_fit',fs,gs));cols=['entity_id','event_index','event_is_fraud']
            assert d[cols].equals(baseline[cols]);assert digest(ws.model_tabular_weights_path)==before
            write(run/f'GENERATION_{gs}.json',dict(rows=len(d),seconds=time.monotonic()-start,sha256=digest(path),exact_labels_lengths=True,parameters_sha256=digest(OUT/'parameters.json'),sampling_audit=audit[prior_audits:]))
            print('PHASE_GENERATED',fs,gs,len(d),flush=True)
    saved=evaluation.DOCS,evaluation.CONTROL;evaluation.DOCS=DOCS;evaluation.CONTROL={**saved[1],'generation_seeds':CFG['generation_seeds']}
    try:evaluation.evaluate(run,'phase_mixture',fs)
    finally:evaluation.DOCS,evaluation.CONTROL=saved
    verify();write(run/'COMPLETE.json',dict(new_neural_training=0,generations=2,test_events_read=False));print('PHASE_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--device',default='cpu');a=p.parse_args()
    prepare() if a.mode=='prepare' else worker(a.seed,a.device)
