"""Train-only statistical controls and native ARGN free generation, stage 1."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import digest,write
from scripts.continue_cs_saf_argn_adequacy import CONFIG,OUT,OLD
from scripts.audit_cs_saf_argn_predictions import encode_paths,probabilities
from scripts.cs_saf_argn_control_adapter import generation_adapter,ordered_gap_code
from models.cs_saf_observed_repeat_control import fit_controls
from scripts.cs_saf_repeat_control_common import conditional_metrics


def extract(model,stats,ctx_stats,raw,parent):
    frame,ids,raw=encode_paths(raw,parent,stats,ctx_stats)
    mark_key='tgt:t1/c1__cat'
    pred,errors=probabilities(model,frame,128,mark_key,'cpu',checks=True)
    groups=dict(tuple(raw.groupby('entity_id',sort=False)));ctx=parent.set_index('entity_id').entity_label
    records=[]
    for i,(eid,(p,obs)) in enumerate(zip(ids,pred)):
        part=groups[eid];code=np.asarray(frame.iloc[i]['tgt:t0/c0__bin'])[:-1]
        assert len(code)==len(p)==len(part)
        y=part.receiver_or_mark.eq(part.receiver_or_mark.shift()).fillna(False).to_numpy(int)
        records.append(pd.DataFrame(dict(entity_id=eid,event_index=np.arange(1,len(p)),
            gap_code=ordered_gap_code(code[1:],stats),p=p[1:],obs_p=obs[1:],y=y[1:],
            group=int(ctx.loc[eid]),gap=part.gap.to_numpy(float)[1:])))
    return pd.concat(records,ignore_index=True),max(errors)



def run(kappa,seed,parent_kind,smoke=False):
    from mostlyai.engine import set_random_state
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._tabular.probability import _initialize_model
    from mostlyai.engine._common import load_generated_data
    torch.set_num_threads(1)
    c=json.loads(CONFIG.read_text());assert seed in c['argn_seeds'] and kappa in c['kappas']
    source=(OLD if parent_kind=='original' else OUT/'continuations')/f'kappa_{kappa}/seed_{seed}'
    assert (source/'DONE.json').exists()
    out=OUT/(f'argn_cpu_smoke' if smoke else f'controls/argn_{parent_kind}/kappa_{kappa}/seed_{seed}')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'workspace',out/'workspace')
    ws=Workspace(out/'workspace');original=digest(ws.model_tabular_weights_path)
    model,stats,ctx_stats,*_=_initialize_model(workspace=ws,device='cpu');model.eval()
    inp=OLD/f'kappa_{kappa}/input';parent=pd.read_parquet(inp/'train_context.parquet')
    raw=pd.read_parquet(inp/'train_events.parquet')
    if smoke:
        parent=parent.groupby('entity_label',group_keys=False).head(32)
        raw=raw[raw.entity_id.isin(parent.entity_id)]
    started=time.monotonic();features,error=extract(model,stats,ctx_stats,raw,parent)
    features.to_parquet(out/'train_features.parquet',index=False)
    controls=fit_controls(features,c)
    write(out/'fit.json',dict(controls=controls,config_sha256=digest(CONFIG),weights_sha256=original,
          training_entities=parent.entity_id.nunique(),training_transitions=len(features),
          feature_sha256=digest(out/'train_features.parquet'),invariance_error=error,
          scientific_fit=not smoke,parent_kind=parent_kind,seed=seed,kappa=kappa))
    # Calibration has been fixed and written before reading evaluation labels.
    val=pd.read_parquet(inp/'validation_canonical.parquet')
    vp=pd.read_parquet(inp/'validation_context.parquet')
    vp['__saf_planned_length']=vp.entity_id.map(val.groupby('entity_id').size())
    if smoke:
        vp=vp.groupby('entity_label',group_keys=False).head(8);val=val[val.entity_id.isin(vp.entity_id)]
    vf,val_error=extract(model,stats,ctx_stats,val,vp);vf.to_parquet(out/'validation_features.parquet',index=False)
    write(out/'conditional.json',conditional_metrics(features,vf,controls));del model
    plan=pd.read_parquet(inp/'plan.parquet').drop(columns='source_train_entity_id')
    if smoke:
        plan=plan.groupby('entity_label',group_keys=False).head(8)
    executions=[]
    for kind,control in [('raw',None)]+list(controls.items()):
        for tape in c['generation_seeds'][:1 if smoke else 3]:
            if parent_kind=='original' and kind=='raw' and not smoke:
                shutil.copyfile(source/f'generated_{tape}.parquet',out/f'generated_{kind}_{tape}.parquet')
                executions.append(dict(variant=kind,tape=tape,reused=True));continue
            set_random_state(tape)
            with generation_adapter(control,stats) as (generate,adapter,source_hash):
                generate(ctx_data=plan,batch_size=256,device='cpu',workspace_dir=out/'workspace')
                executions.append(dict(variant=kind,tape=tape,reused=False,official_source_sha256=source_hash,
                    steps=adapter.steps,corrected_predictions=adapter.corrected,
                    normalization_max_error=adapter.max_probability_sum_error))
            sample=load_generated_data(out/'workspace')
            assert set(sample.entity_id)==set(plan.entity_id)
            sample.to_parquet(out/f'generated_{kind}_{tape}.parquet',index=False)
            print('GENERATED',parent_kind,kappa,seed,kind,tape,flush=True)
    assert original==digest(ws.model_tabular_weights_path)==digest(Workspace(source/'workspace').model_tabular_weights_path)
    if parent_kind=='continued' and not smoke:
        # A fixed secondary diagnostic, not another selected/generative candidate.
        last_model,*_=_initialize_model(workspace=ws,device='cpu')
        last_path=source/'checkpoint_last.pt';last_hash=digest(last_path)
        last_model.load_state_dict(torch.load(last_path,map_location='cpu',weights_only=True),strict=True)
        last_model.eval()
        last_features,last_error=extract(last_model,stats,ctx_stats,val,vp)
        write(out/'last_conditional.json',dict(metrics=conditional_metrics(features,last_features,{}),
             invariance_error=last_error,checkpoint_sha256=last_hash,role='secondary_fixed_final_checkpoint_no_calibration_or_generation'))
        assert digest(last_path)==last_hash
    write(out/'DONE.json',dict(seconds=time.monotonic()-started,weights_sha256=original,weights_unchanged=True,
        validation_invariance_error=val_error,executions=executions,smoke=smoke,test_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kappa',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--parent',choices=['original','continued'],default='original');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.kappa,a.seed,a.parent,a.smoke)
