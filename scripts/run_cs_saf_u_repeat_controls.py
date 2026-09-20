"""Apply the same observable-repeat controls to saved U, without retraining it."""
import argparse
from contextlib import contextmanager
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continue_cs_saf_argn_adequacy import CONFIG,OUT,OLD
from scripts.run_cs_saf_external_audit_v1 import digest,write
from scripts.cs_saf_repeat_control_common import conditional_metrics
from models.cs_saf_observed_repeat_control import fit_controls,apply_torch,redistribute
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cs_saf_replication import ReplicationU
from experiments.cs_saf_pilot import batch,state_digest
from experiments.cs_saf_generation_repeats import generate_fixed_plan


@contextmanager
def u_control(model,control):
    native=model.mark_distribution
    def adjusted(self,context,gap,previous,has_previous,*,static_codes=None):
        logp,lr,lnr=native(context,gap,previous,has_previous,static_codes=static_codes)
        if control is None or not has_previous.any():return logp,lr,lnr
        p=logp.exp();mask=has_previous
        before=p[mask].gather(-1,previous[mask,None])[:,0]
        code=(self._support_code(gap)[mask]-3).clamp_min(0)
        updated=apply_torch(before,code,static_codes[mask]-3,control)
        p[mask]=redistribute(p[mask],previous[mask],updated)
        lr=lr.clone();lnr=lnr.clone();lr[mask]=updated.log();lnr[mask]=torch.log1p(-updated)
        return p.log(),lr,lnr
    try:
        model.mark_distribution=types.MethodType(adjusted,model)
        yield
    finally:
        # Remove the instance override so the original class method is restored.
        del model.mark_distribution


@torch.no_grad()
def extract(model,data,limit=None):
    records=[];total=len(data['lengths']) if limit is None else limit
    for start in range(0,total,256):
        ids=torch.arange(start,min(start+256,total));x=batch(data,ids,'cpu')
        context=model.context(model.encoder(**x),x['static_categorical'])
        valid=x['valid_mask'].clone();valid[:,0]=False
        previous=x['receiver'].roll(1,1);codes=x['static_categorical'][0][:,None].expand_as(valid)
        logp,_,_=model.mark_distribution(context,x['gap'],previous,valid,static_codes=codes)
        rows,cols=torch.where(valid)
        p=logp[valid].exp();prev=previous[valid];actual=x['receiver'][valid]
        records.append(pd.DataFrame(dict(entity_id=np.asarray(data['entity_ids'])[rows.numpy()+start],
            event_index=cols.numpy(),gap_code=(model._support_code(x['gap'][valid])-3).numpy(),
            p=p.gather(1,prev[:,None])[:,0].numpy(),obs_p=p.gather(1,actual[:,None])[:,0].numpy(),
            y=(actual==prev).numpy().astype(int),group=(codes[valid]-3).numpy(),gap=x['gap'][valid].numpy())))
    return pd.concat(records,ignore_index=True)


def load(kappa,trial):
    source=ROOT.parent/'research-cs-saf/artifacts/cs_saf'
    cache=source/f'prepared_v1/pi_0.10_kappa_{kappa}.pt'
    index=json.loads((cache.parent/'COMPLETE.json').read_text())
    assert digest(cache)==index['cells'][cache.stem]['cache_sha256']
    payload=torch.load(cache,map_location='cpu')
    folder=source/f'followup_v1/internal/pi_0.10/trial_{trial}/kappa_{kappa}/U'
    checkpoint=folder/'checkpoint_best.pt'
    manifest=json.loads((folder/'COMPLETE.json').read_text())
    assert digest(checkpoint)==manifest['artifact_sha256']['checkpoint_best.pt']
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    model=ReplicationU('CS2-U1',state.gap_support,**state.model_config_kwargs())
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model_state'],strict=True)
    model.eval();model.requires_grad_(False)
    return model,payload,state,dict(checkpoint_path=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        cache_sha256=digest(cache),state_sha256=state_digest(model))


def to_frame(sample,payload,state,plan):
    mask=sample['valid_mask'].numpy();rows,cols=np.where(mask)
    return pd.DataFrame(dict(entity_id=plan.entity_id.to_numpy()[rows],event_index=cols,
        gap=sample['gap'].numpy()[mask].astype(float),
        receiver_or_mark=state.receiver_codec.decode(sample['receiver'].numpy()[mask]),
        amount_or_numeric_value=state.event_numeric_codecs[0][1].decode(sample['numeric_value'].numpy()[mask])))


def run(kappa,trial,smoke=False):
    torch.set_num_threads(1);c=json.loads(CONFIG.read_text())
    assert trial in c['u_trials'] and kappa in c['kappas']
    out=OUT/('u_cpu_smoke' if smoke else f'controls/U/kappa_{kappa}/trial_{trial}')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    model,payload,state,source=load(kappa,trial)
    features=extract(model,payload['train'],128 if smoke else None)
    features.to_parquet(out/'train_features.parquet',index=False)
    controls=fit_controls(features,c)
    write(out/'fit.json',dict(controls=controls,source=source,config_sha256=digest(CONFIG),
        feature_sha256=digest(out/'train_features.parquet'),scientific_fit=not smoke,kappa=kappa,trial=trial))
    validation=extract(model,payload['validation'],64 if smoke else None)
    validation.to_parquet(out/'validation_features.parquet',index=False)
    write(out/'conditional.json',conditional_metrics(features,validation,controls))
    plan=pd.read_parquet(OLD/f'kappa_{kappa}/input/plan.parquet')
    if smoke:plan=plan.iloc[:16].copy()
    positions=pd.Index(payload['train']['entity_ids']).get_indexer(plan.source_train_entity_id)
    assert (positions>=0).all()
    assert np.array_equal(payload['train']['lengths'][positions].numpy(),plan['__saf_planned_length'].to_numpy())
    executions=[]
    for kind,control in [('raw',None)]+list(controls.items()):
        for tape in c['generation_seeds'][:1 if smoke else 3]:
            with u_control(model,control):
                sample=generate_fixed_plan(model,payload,positions,tape,'cpu',256)
            generated=to_frame(sample,payload,state,plan)
            generated.to_parquet(out/f'generated_{kind}_{tape}.parquet',index=False)
            executions.append(dict(variant=kind,tape=tape,events=len(generated)))
            print('GENERATED U',kappa,trial,kind,tape,flush=True)
    assert state_digest(model)==source['state_sha256']
    assert digest(source['checkpoint_path'])==source['checkpoint_sha256']
    write(out/'DONE.json',dict(seconds=time.monotonic()-started,weights_unchanged=True,smoke=smoke,
        source=source,test_accessed=False,executions=executions))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kappa',type=int,required=True);p.add_argument('--trial',type=int,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.kappa,a.trial,a.smoke)
