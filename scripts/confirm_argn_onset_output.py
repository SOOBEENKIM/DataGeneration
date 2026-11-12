"""Confirm a fixed onset expert with fresh draws and exact label preservation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pandas as pd
import run_argn_onset_output as onset
import run_argn_joint_preservation as study
from run_argn_joint_preservation import ROOT,write,digest,Workspace,folder
from benchmarks.argn_onset_output import onset_output_generation

OUT=onset.OUT/'confirmation';DOCS=onset.DOCS/'confirmation';CFG={**onset.CFG,'generation_seeds':[20261021,20261022]}


def verify():
    onset.verify();m=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    onset.verify();assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),ROOT/'scripts/dispatch_argn_onset_confirmation.py',onset.DOCS/'CONFIRMATION_PROTOCOL.md',onset.DOCS/'COMPLETE.json']
    for fs in CFG['fit_seeds']:paths.append(onset.OUT/f'worker_{fs}/onset_fit/onset_head.pt')
    m=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,hashes={str(p):digest(p) for p in paths},new_training=0,test_events_read=False)
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('ONSET_CONFIRMATION_PREPARED',flush=True)


def worker(fs,device):
    verify();wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    import run_argn_label_first_control as evaluation
    head=onset.OUT/f'worker_{fs}/onset_fit/onset_head.pt';count=study.OUT/f'worker_{fs}/unconditional/count_head.pt'
    run=OUT/f'runs/onset_fit_{fs}';run.mkdir(parents=True,exist_ok=False)
    ws=Workspace(folder('B_event_label_first',fs)/'workspace')
    saved=(study.preservation_generation,study.CFG,study.DOCS,evaluation.CONTROL)
    def manager(*args,**kwargs):return onset_output_generation(*args,onset_payload=head,**kwargs)
    study.preservation_generation=manager;study.DOCS=DOCS
    study.CFG={**saved[1],'generation_seeds':CFG['generation_seeds'],'generation_arms':{'onset_fit':saved[1]['generation_arms']['count_hazard_no_context']}}
    evaluation.CONTROL={**saved[3],'generation_seeds':CFG['generation_seeds']}
    try:study.free_generate(ws,run,fs,'onset_fit',device,count)
    finally:study.preservation_generation,study.CFG,study.DOCS,evaluation.CONTROL=saved
    for gs in CFG['generation_seeds']:
        b=pd.read_parquet(study.OUT/f'confirmation/runs/count_hazard_no_context_{fs}/generated_validation_{gs}.parquet')
        s=pd.read_parquet(run/f'generated_validation_{gs}.parquet');cols=['entity_id','event_index','event_is_fraud'];assert b[cols].equals(s[cols])
    write(run/'COMPLETE.json',dict(generations=2,exact_labels_lengths=True,head_sha256=digest(head),test_events_read=False))
    verify();write(wd/'COMPLETE.json',dict(new_training=0,new_generations=2,test_events_read=False));print('ONSET_CONFIRMATION_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
