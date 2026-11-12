"""Frozen-generator intervention, registered before generation."""
import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import time

from audit_argn_fit_generalization import ROOT, OUT as AUDIT, DOCS, folder, inputs
from run_argn_state_first import OUT as BASE, SOURCE, seed, digest, write
import numpy as np
import pandas as pd
import torch
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace
from benchmarks.argn_transition_reference import reference_generation

OUT=AUDIT/'transition_reference'
GEN_SEEDS=[20261011,20261012]


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    paths=[ROOT/'benchmarks/argn_transition_reference.py', ROOT/'scripts/run_argn_transition_reference.py',
        DOCS/'TRANSITION_REFERENCE_PROTOCOL.md', AUDIT/'simple_label_parameters.json',
        BASE/'prepared/validation_context.parquet']
    paths += [Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path
              for fs in [20260930,20261001]]
    write(OUT/'MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hashes={str(p):digest(p) for p in paths},fit_seeds=[20260930,20261001],
        modes=['markov','duration'],generation_seeds=GEN_SEEDS,test_events_read=False))


def evaluate(run,mode,fs):
    import run_argn_label_first_control as evaluation
    saved=evaluation.DOCS
    evaluation.DOCS=DOCS/'transition_reference'
    try:evaluation.evaluate(run,'reference_'+mode,fs)
    finally:evaluation.DOCS=saved


def worker(fs,device):
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    def unchanged():
        for p,h in manifest['hashes'].items():assert digest(p)==h
    unchanged()
    parameters=json.loads((AUDIT/'simple_label_parameters.json').read_text())
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    for mode in manifest['modes']:
        run=OUT/f'{mode}_{fs}';run.mkdir(parents=True,exist_ok=False)
        shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',
                        ignore=shutil.ignore_patterns('SyntheticData'))
        ws=Workspace(run/'workspace');weight_hash=digest(ws.model_tabular_weights_path)
        write(run/'START.json',dict(parent='B_event_label_first',fit_seed=fs,mode=mode,
            pid=os.getpid(),device=device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            created_utc=datetime.now(timezone.utc).isoformat(),neural_training=False))
        with reference_generation(ws.tgt_stats.read(),parameters,mode):
            for gs in GEN_SEEDS:
                seed(gs);start=time.monotonic()
                generate(ctx_data=context,device=device,workspace_dir=run/'workspace')
                raw=pd.read_parquet(run/'workspace/SyntheticData')
                raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
                frame=raw.rename(columns={'customer_id':'entity_id'}).copy()
                frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
                frame.loc[frame.event_index.eq(0),'gap']=np.nan
                dest=run/f'generated_validation_{gs}.parquet';frame.to_parquet(dest,index=False)
                assert digest(ws.model_tabular_weights_path)==weight_hash
                write(run/f'GENERATION_{gs}.json',dict(seconds=time.monotonic()-start,
                    rows=len(frame),customers=frame.entity_id.nunique(),sha256=digest(dest),
                    parent_weights_sha256=weight_hash))
                print('REFERENCE_GENERATED',mode,fs,gs,len(frame),flush=True)
        evaluate(run,mode,fs);unchanged()
        write(run/'COMPLETE.json',dict(mode=mode,fit_seed=fs,weights_unchanged=True,
            test_events_read=False,completed_utc=datetime.now(timezone.utc).isoformat()))
    print('REFERENCE_WORKER_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker'])
    p.add_argument('--seed',type=int,choices=[20260930,20261001]);p.add_argument('--device',default='cpu')
    a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
