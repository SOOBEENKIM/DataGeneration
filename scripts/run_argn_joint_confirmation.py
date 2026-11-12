"""Fresh generation draws, with all fitted distributions fixed before execution."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
import run_argn_joint_preservation as study
from run_argn_joint_preservation import ROOT,BASE,PRIOR,OLD,foundation,seed,digest,write,Workspace,folder
from benchmarks.argn_joint_preservation import preservation_generation
from benchmarks.argn_episode_control import episode_generation

OUT=study.OUT/'confirmation';DOCS=study.DOCS/'confirmation'
CFG={**study.CFG,'generation_seeds':[20261021,20261022]}


def verify():
    study.verify();m=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in m['hashes'].items():assert digest(p)==h,p
    return m


def prepare():
    study.verify();assert not (OUT/'MANIFEST.json').exists()
    gates=pd.read_csv(study.DOCS/'preservation_screen.csv');choice=None
    for arm in ['count_hazard_no_context','count_hazard']:
        g=gates[gates.arm.eq(arm)]
        if len(g)==2 and g.passes_preregistered_screen.all():choice=arm;break
    assert choice is not None,'No joint candidate passed; do not extend training ad hoc.'
    spec=study.CFG['generation_arms'][choice]
    paths=[Path(__file__),ROOT/'scripts/dispatch_argn_joint_confirmation.py',study.OUT/'MANIFEST.json',
           study.DOCS/'CONFIRMATION_PROTOCOL.md',study.DOCS/'preservation_screen.csv',study.DOCS/'completion_evidence.json']
    for fs in CFG['fit_seeds']:paths.append(study.OUT/f"worker_{fs}/{spec['count']}/count_head.pt")
    m=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CFG,selected_arm=choice,
           hashes={str(p):digest(p) for p in paths},test_events_read=False,new_neural_fits=0)
    write(OUT/'MANIFEST.json',m);write(DOCS/'MANIFEST.json',m);print('CONFIRMATION_PREPARED',choice,flush=True)


def worker(fs,device):
    m=verify();choice=m['selected_arm'];device=torch.device(device);started=time.monotonic()
    wd=OUT/f'worker_{fs}';wd.mkdir(parents=True,exist_ok=False)
    write(wd/'START.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    from mostlyai.engine import generate
    import run_argn_label_first_control as evaluation
    ws=Workspace(folder('B_event_label_first',fs)/'workspace');old=json.loads(OLD.read_text())
    episode=json.loads((PRIOR/'episode_parameters.json').read_text());hazard=json.loads((study.OUT/'hazard_parameters.json').read_text())
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    count=study.OUT/f"worker_{fs}/{study.CFG['generation_arms'][choice]['count']}/count_head.pt"
    for arm in ['current',choice]:
        run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
        write(wd/'STATUS.json',dict(stage='free_generation',arm=arm))
        shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
        shutil.copy2(foundation(fs)/'amount_head.pt',run/'amount_head.pt')
        weight=Workspace(run/'workspace').model_tabular_weights_path;before=digest(weight)
        manager=(episode_generation(ws.tgt_stats.read(),old,run/'amount_head.pt',episode,'joint') if arm=='current' else
                 preservation_generation(ws.tgt_stats.read(),old,run/'amount_head.pt',episode,hazard,count,True))
        with manager:
            for gs in CFG['generation_seeds']:
                seed(gs);start=time.monotonic();generate(ctx_data=context,device=str(device),workspace_dir=run/'workspace')
                raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
                frame=raw.rename(columns={'customer_id':'entity_id'}).copy();frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
                frame.loc[frame.event_index.eq(0),'gap']=np.nan;dest=run/f'generated_validation_{gs}.parquet';frame.to_parquet(dest,index=False)
                assert digest(weight)==before and digest(run/'amount_head.pt')==digest(foundation(fs)/'amount_head.pt')
                write(run/f'GENERATION_{gs}.json',dict(seconds=time.monotonic()-start,rows=len(frame),customers=frame.entity_id.nunique(),
                    sha256=digest(dest),head_sha256=digest(run/'amount_head.pt'),count_sha256=digest(count) if arm!='current' else None))
                print('CONFIRMATION_GENERATED',arm,fs,gs,len(frame),flush=True)
        saved_docs,saved_control=evaluation.DOCS,evaluation.CONTROL
        evaluation.DOCS=DOCS;evaluation.CONTROL={**saved_control,'generation_seeds':CFG['generation_seeds']}
        try:evaluation.evaluate(run,arm,fs)
        finally:evaluation.DOCS,evaluation.CONTROL=saved_docs,saved_control
        write(run/'COMPLETE.json',dict(arm=arm,fit_seed=fs,generated_datasets=2,test_events_read=False))
    verify();write(wd/'STATUS.json',dict(stage='complete'))
    write(wd/'COMPLETE.json',dict(seconds=time.monotonic()-started,new_neural_fits=0,generated_datasets=4,test_events_read=False))
    print('CONFIRMATION_COMPLETE',fs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
