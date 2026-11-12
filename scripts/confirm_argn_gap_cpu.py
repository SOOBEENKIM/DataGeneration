"""Explicit CPU alternative: matched-device reference and two frozen controls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pandas as pd
import torch
from confirm_argn_gap_controls import verify as verify_registration, head
from run_argn_boundary_gap import generation_args
from run_argn_amount_learning import ROOT,BASE,Workspace,folder,seed,write,digest
from benchmarks.argn_phase_gap import phase_gap_generation
from benchmarks.argn_boundary_gap import boundary_gap_generation
from benchmarks.argn_onset_output import onset_output_generation

OUT=ROOT/'artifacts/argn_phase_gap_v1/confirmation_cpu'
DOCS=ROOT/'docs/argn_phase_gap_v1/confirmation_cpu'
CFG=dict(fits=[20260930,20261001],draws=[20261021,20261022],arms=['frozen','boundary_fit','phase_mixture'],device='cpu')


def verify():
    verify_registration();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items(): assert digest(p)==h,p


def prepare():
    verify_registration();assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),DOCS.parent/'confirmation/CPU_AMENDMENT.md']
    m=dict(config=CFG,hashes={str(p):digest(p) for p in paths},new_training=0,test_events_read=False,created_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('CPU_CONFIRM_PREPARED',flush=True)


def worker(fs):
    verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not torch.cuda.is_available(), 'CPU confirmation must not allocate any GPU'
    from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    params=json.loads((OUT.parent/'parameters.json').read_text())
    for arm in CFG['arms']:
        run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
        shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
        ws=Workspace(run/'workspace');before=digest(ws.model_tabular_weights_path)
        args,kwargs=generation_args(ws,fs)
        if arm=='boundary_fit':cm=boundary_gap_generation(*args,**kwargs,boundary_payload=head(fs))
        elif arm=='phase_mixture':cm=phase_gap_generation(*args,**kwargs,phase_parameters=params)
        else:cm=onset_output_generation(*args,**kwargs)
        with cm as audit:
            for gs in CFG['draws']:
                seed(gs);start=time.monotonic();prior=len(audit) if audit is not None else 0
                generate(ctx_data=context,device='cpu',workspace_dir=run/'workspace')
                raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
                d=raw.rename(columns={'customer_id':'entity_id'});d['event_index']=d.groupby('entity_id',sort=False).cumcount();d.loc[d.event_index.eq(0),'gap']=float('nan')
                path=run/f'generated_validation_{gs}.parquet';d.to_parquet(path,index=False)
                if arm!='frozen':
                    cols=['entity_id','event_index','event_is_fraud'];b=pd.read_parquet(OUT/f'runs/frozen_{fs}/generated_validation_{gs}.parquet');assert d[cols].equals(b[cols])
                assert digest(ws.model_tabular_weights_path)==before
                write(run/f'GENERATION_{gs}.json',dict(rows=len(d),seconds=time.monotonic()-start,sha256=digest(path),exact_labels_lengths=arm!='frozen',device='cpu',sampling_audit=audit[prior:] if audit is not None else []))
                print('CPU_CONFIRM_GENERATED',fs,arm,gs,len(d),flush=True)
        saved=evaluation.DOCS,evaluation.CONTROL;evaluation.DOCS=DOCS;evaluation.CONTROL={**saved[1],'generation_seeds':CFG['draws']}
        try:evaluation.evaluate(run,arm,fs)
        finally:evaluation.DOCS,evaluation.CONTROL=saved
        write(run/'COMPLETE.json',dict(generations=2,new_training=0,test_events_read=False))
    verify();print('CPU_CONFIRM_COMPLETE',fs,flush=True)


def dispatch():
    verify();assert not (OUT/'LAUNCH.json').exists();jobs={};launched={};(OUT/'logs').mkdir()
    for fs in CFG['fits']:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--seed',str(fs)]
        with (OUT/'logs'/f'{fs}.log').open('x') as log:
            jobs[fs]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launched[fs]=dict(pid=jobs[fs].pid,device='cpu',command=cmd,started_utc=datetime.now(timezone.utc).isoformat())
        print('CPU_CONFIRM_LAUNCHED',fs,jobs[fs].pid,flush=True)
    write(OUT/'LAUNCH.json',launched);write(DOCS/'LAUNCH.json',launched)
    codes={fs:p.wait() for fs,p in jobs.items()}
    write(DOCS/'DISPATCH_COMPLETE.json',dict(exit_codes=codes,success=all(c==0 for c in codes.values()),completed_utc=datetime.now(timezone.utc).isoformat()))
    assert all(c==0 for c in codes.values()),codes


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','dispatch']);p.add_argument('--seed',type=int,choices=CFG['fits']);a=p.parse_args()
    prepare() if a.mode=='prepare' else worker(a.seed) if a.mode=='worker' else dispatch()
