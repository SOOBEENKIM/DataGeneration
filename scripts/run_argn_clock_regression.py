"""A registered low-dimensional conditional-mixture control, no new ARGN fit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import torch
import run_argn_time_density as temporal
from run_argn_time_density import ROOT,BASE,META,seed,digest,write,Workspace,folder,clock_features,load_parent,inputs
from benchmarks.argn_clock_regression import ClockRegression,clock_regression_generation

OUT=ROOT/'artifacts/argn_clock_regression_v1';DOCS=ROOT/'docs/argn_clock_regression_v1'
CFG=dict(fit_seeds=temporal.CFG['fit_seeds'],generation_seeds=temporal.CFG['generation_seeds'],arms=['clock_joint'],
    sampling_seeds=temporal.CFG['sampling_seeds'],allowed_gpu_uuids=temporal.CFG['allowed_gpu_uuids'][2:])


def verify():
    temporal.verify();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    temporal.verify();assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),ROOT/'benchmarks/argn_clock_regression.py',ROOT/'tests/test_argn_clock_regression.py',DOCS/'PROTOCOL.md',temporal.OUT/'MANIFEST.json']
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',manifest);write(DOCS/'MANIFEST.json',manifest)
    params=json.loads((temporal.OUT/'parameters.json').read_text());m=pd.read_parquet(META/'metadata_optimization.parquet');clock,_=clock_features(m,params['fallback'])
    phase=2*m.previous.clip(lower=0).to_numpy()+m.label.to_numpy();keep=m.event_index.gt(0).to_numpy()
    head=ClockRegression(1,'clock_joint');stats=[]
    for p in range(4):
        mask=keep&(phase==p);x=np.c_[clock[mask],np.log1p(m.gap.to_numpy()[mask])];count=len(x)
        if len(x)>50000:x=x[np.random.default_rng(20261115).choice(len(x),50000,replace=False)]
        fit=GaussianMixture(3,random_state=20261115,n_init=3,reg_covar=1e-4,covariance_type='full').fit(x)
        head.weights[p]=torch.tensor(fit.weights_);head.means[p]=torch.tensor(fit.means_);head.covariances[p]=torch.tensor(fit.covariances_)
        stats.append(dict(phase=p,positions=count,fitted_positions=len(x),converged=bool(fit.converged_),iterations=fit.n_iter_))
    ws=Workspace(folder('B_event_label_first',CFG['fit_seeds'][0])/'workspace');s=ws.tgt_stats.read()['columns']['gap'];prefix=f"{s['argn_processor']}:{s['argn_table']}/{s['argn_column']}__"
    keys=[k for k in temporal.get_cardinalities(ws.tgt_stats.read()) if k.startswith(prefix)]
    payload=dict(dim=1,arm='clock_joint',components=3,hidden=64,state_dict=head.state_dict(),fallback=params['fallback'],limits=params['limits'],quantum=params['quantum'],keys=keys)
    torch.save(payload,OUT/'time_head.pt');record=dict(statistical_phase_fits=4,neural_fits=0,head_sha256=digest(OUT/'time_head.pt'),phases=stats)
    write(OUT/'FIT_COMPLETE.json',record);write(DOCS/'FIT_COMPLETE.json',record)
    rows=[]
    for split in ['optimization','internal_validation','development']:
        m=pd.read_parquet(META/f'metadata_{split}.parquet');c,v=clock_features(m,params['fallback']);keep=m.event_index.gt(0).to_numpy()
        pair=2*m.previous.clip(lower=0).to_numpy()+m.label.to_numpy()
        data=dict(base=torch.zeros(int(keep.sum()),1),phase=torch.as_tensor(pair[keep],dtype=torch.long),clock=torch.as_tensor(c[keep],dtype=torch.float32),valid=torch.as_tensor(v[keep]),gap=torch.as_tensor(m.gap.to_numpy()[keep],dtype=torch.float32))
        rows+=temporal.conditional(head,data,m.loc[keep].reset_index(drop=True),params,0,'clock_joint',split)
    pd.DataFrame(rows).to_csv(DOCS/'conditional.csv',index=False);print('CLOCK_GMR_PREPARED',flush=True)


def worker(fs):
    verify();assert os.environ['CUDA_VISIBLE_DEVICES'] in CFG['allowed_gpu_uuids'];device=torch.device('cuda:0');seed(fs)
    record=json.loads((OUT/'FIT_COMPLETE.json').read_text());assert digest(OUT/'time_head.pt')==record['head_sha256']
    _,codec,frames,metas=inputs();ws,native=load_parent(fs,device);check=OUT/f'worker_{fs}';check.mkdir(parents=True,exist_ok=False)
    with patch.object(temporal,'time_density_generation',clock_regression_generation):
        temporal.integration(ws,native,frames['development'],metas['development'],codec,fs,OUT/'time_head.pt',check)
        del frames,metas,native;torch.cuda.empty_cache()
        with patch.object(temporal,'OUT',OUT),patch.object(temporal,'DOCS',DOCS):
            temporal.generate_arm(ws,fs,'clock_joint',OUT/'time_head.pt',device)
    verify();write(check/'COMPLETE.json',dict(generations=4,test_events_read=False));print('CLOCK_GMR_COMPLETE',fs,flush=True)


def dispatch():
    # Reuse resource selection only. Its verify callback retains the new registry.
    verify()
    with patch.object(temporal,'OUT',OUT),patch.object(temporal,'DOCS',DOCS),patch.object(temporal,'CFG',CFG),patch.object(temporal,'__file__',str(Path(__file__).resolve())),patch.object(temporal,'verify',verify_for_dispatch):
        temporal.dispatch()


def verify_for_dispatch():
    # temporal.verify itself is replaced only within dispatch; validate both manifests explicitly.
    from diagnose_argn_residual import verify_registry
    verify_registry()
    for path in [ROOT/'artifacts/argn_time_density_v1/MANIFEST.json',OUT/'MANIFEST.json']:
        m=json.loads(path.read_text())
        for p,h in m['hashes'].items():assert digest(p)==h,p


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','dispatch']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    prepare() if a.mode=='prepare' else worker(a.seed) if a.mode=='worker' else dispatch()
