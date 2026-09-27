"""Resume valid saved draws after a post-generation baseline-path error."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import confirm_argn_onset_output as registered
import run_argn_onset_output as onset
import run_argn_joint_preservation as study
from run_argn_joint_preservation import ROOT,BASE,seed,digest,write,Workspace
from benchmarks.argn_onset_output import onset_output_generation

OUT=registered.OUT/'recovery_dispatch';DOCS=registered.DOCS/'recovery';CFG=registered.CFG


def verify():
    registered.verify();m=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in m['hashes'].items():assert digest(p)==h,p


def prepare():
    registered.verify();assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),ROOT/'scripts/dispatch_argn_onset_recovery.py',onset.DOCS/'RECOVERY_PROTOCOL.md',registered.OUT/'MANIFEST.json']
    for fs in CFG['fit_seeds']:paths.append(registered.OUT/f'runs/onset_fit_{fs}/generated_validation_20261021.parquet')
    m=dict(created_utc=datetime.now(timezone.utc).isoformat(),hashes={str(p):digest(p) for p in paths},config=CFG,
        reused_existing_draws=2,remaining_new_draws=2,training_changes=False,test_events_read=False)
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('ONSET_RECOVERY_PREPARED',flush=True)


def worker(fs,device):
    verify();from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    run=registered.OUT/f'runs/onset_fit_{fs}';ws=Workspace(run/'workspace')
    old=json.loads(study.OLD.read_text());episode=json.loads((study.PRIOR/'episode_parameters.json').read_text());hazard=json.loads((study.OUT/'hazard_parameters.json').read_text())
    count=study.OUT/f'worker_{fs}/unconditional/count_head.pt';head=onset.OUT/f'worker_{fs}/onset_fit/onset_head.pt'
    weight_hash=digest(ws.model_tabular_weights_path);context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    with onset_output_generation(ws.tgt_stats.read(),old,run/'amount_head.pt',episode,hazard,count,True,onset_payload=head):
        for gs in CFG['generation_seeds']:
            path=run/f'generated_validation_{gs}.parquet';reused=path.exists();started=time.monotonic()
            if not reused:
                seed(gs);generate(ctx_data=context,device=device,workspace_dir=run/'workspace')
                raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
                frame=raw.rename(columns={'customer_id':'entity_id'}).copy();frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
                frame.loc[frame.event_index.eq(0),'gap']=np.nan;frame.to_parquet(path,index=False)
            frame=pd.read_parquet(path);base=pd.read_parquet(study.OUT/f'confirmation/runs/count_hazard_no_context_{fs}/generated_validation_{gs}.parquet')
            cols=['entity_id','event_index','event_is_fraud'];assert frame[cols].equals(base[cols])
            assert digest(ws.model_tabular_weights_path)==weight_hash
            assert digest(run/'amount_head.pt')==digest(study.foundation(fs)/'amount_head.pt')
            write(run/f'GENERATION_{gs}.json',dict(rows=len(frame),customers=frame.entity_id.nunique(),sha256=digest(path),
                head_sha256=digest(run/'amount_head.pt'),count_sha256=digest(count),onset_head_sha256=digest(head),
                reused_saved_valid_draw=reused,recovery_seconds=time.monotonic()-started,exact_labels_lengths=True))
            print('ONSET_RECOVERED',fs,gs,len(frame),'reused',reused,flush=True)
    saved=evaluation.DOCS,evaluation.CONTROL;evaluation.DOCS=registered.DOCS;evaluation.CONTROL={**saved[1],'generation_seeds':CFG['generation_seeds']}
    try:evaluation.evaluate(run,'onset_fit',fs)
    finally:evaluation.DOCS,evaluation.CONTROL=saved
    verify();write(run/'COMPLETE.json',dict(generations=2,exact_labels_lengths=True,recovered=True,test_events_read=False))
    write(OUT/f'worker_{fs}/COMPLETE.json',dict(reused_draws=1,new_draws=1,test_events_read=False));print('ONSET_RECOVERY_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
