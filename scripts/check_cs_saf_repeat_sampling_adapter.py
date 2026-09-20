"""Identity and forced group-specific repetition checks on saved native samplers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continue_cs_saf_argn_adequacy import OUT,OLD
from scripts.run_cs_saf_external_audit_v1 import write,digest


def validate_forced(frame,plan,raw):
    groups=plan.set_index('entity_id').entity_label.astype(int)
    first=frame.groupby('entity_id',sort=False).head(1).set_index('entity_id').receiver_or_mark.sort_index()
    original=raw.groupby('entity_id',sort=False).head(1).set_index('entity_id').receiver_or_mark.sort_index()
    pd.testing.assert_series_equal(first,original,check_dtype=False)
    prior=frame.groupby('entity_id',sort=False).receiver_or_mark.shift()
    valid=frame.groupby('entity_id',sort=False).cumcount()>0
    observed=frame.receiver_or_mark.eq(prior).fillna(False).to_numpy(int)[valid]
    expected=frame.entity_id.map(groups).to_numpy(int)[valid]
    np.testing.assert_array_equal(observed,expected)
    assert frame.groupby('entity_id').size().nunique()>1, 'exercise differing native lengths'
    return dict(first_events_unchanged=True,group0_never_repeat=True,group1_always_repeat=True,
                unequal_lengths_exercised=True,transitions_checked=int(valid.sum()))


def argn_gate():
    from mostlyai.engine import generate,set_random_state
    from mostlyai.engine._common import load_generated_data
    from mostlyai.engine._workspace import Workspace
    from scripts.cs_saf_argn_control_adapter import generation_adapter
    out=OUT/'argn_sampler_gate';out.mkdir(exist_ok=False)
    source=OLD/'kappa_1/seed_20260920/workspace';shutil.copytree(source,out/'workspace')
    ws=Workspace(out/'workspace');before=digest(ws.model_tabular_weights_path);stats=ws.tgt_stats.read()
    plan=pd.read_parquet(OLD/'kappa_1/input/plan.parquet').groupby('entity_label',group_keys=False).head(12)
    plan=plan.drop(columns='source_train_entity_id')
    set_random_state(2026092911)
    generate(ctx_data=plan,batch_size=256,workspace_dir=out/'workspace',device='cpu')
    raw=load_generated_data(out/'workspace').copy()
    set_random_state(2026092911)
    with generation_adapter(None,stats) as (run,state,sha):
        run(ctx_data=plan,batch_size=256,workspace_dir=out/'workspace',device='cpu')
    identity=load_generated_data(out/'workspace')
    pd.testing.assert_frame_equal(raw,identity,check_exact=True)
    fitted=json.loads((OUT/'argn_cpu_smoke/fit.json').read_text())['controls']['direct']
    forced=dict(fitted,parameters=[[0.]*5,[1.]*5])
    set_random_state(2026092911)
    with generation_adapter(forced,stats) as (run,state,sha):
        run(ctx_data=plan,batch_size=256,workspace_dir=out/'workspace',device='cpu')
    sample=load_generated_data(out/'workspace')
    result=validate_forced(sample,plan,raw)
    assert before==digest(ws.model_tabular_weights_path)==digest(Workspace(source).model_tabular_weights_path)
    result.update(identity_sample_exact=True,weights_unchanged=True,official_generation_source_sha256=sha,
                  normalization_max_error=state.max_probability_sum_error,technical_fixture_only=True)
    write(ROOT/'docs/cs_saf/baseline_adequacy_v1/argn_sampling_cpu_gate.json',result);print(result)


def u_gate():
    from scripts.run_cs_saf_u_repeat_controls import load,u_control,to_frame
    from experiments.cs_saf_generation_repeats import generate_fixed_plan,compare_samples
    from experiments.cs_saf_pilot import state_digest
    model,payload,state,source=load(1,0)
    plan=pd.read_parquet(OLD/'kappa_1/input/plan.parquet').groupby('entity_label',group_keys=False).head(12)
    positions=pd.Index(payload['train']['entity_ids']).get_indexer(plan.source_train_entity_id)
    raw_sample=generate_fixed_plan(model,payload,positions,2026092911,'cpu')
    with u_control(model,None):
        identity=generate_fixed_plan(model,payload,positions,2026092911,'cpu')
    compare_samples(raw_sample,identity)
    fitted=json.loads((OUT/'u_cpu_smoke/fit.json').read_text())['controls']['direct']
    forced=dict(fitted,parameters=[[0.]*5,[1.]*5])
    with u_control(model,forced):
        sample=generate_fixed_plan(model,payload,positions,2026092911,'cpu')
    result=validate_forced(to_frame(sample,payload,state,plan),plan,to_frame(raw_sample,payload,state,plan))
    assert state_digest(model)==source['state_sha256']
    result.update(identity_sample_exact=True,weights_unchanged=True,technical_fixture_only=True)
    write(ROOT/'docs/cs_saf/baseline_adequacy_v1/u_sampling_cpu_gate.json',result);print(result)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('model',choices=['argn','U']);a=p.parse_args()
    argn_gate() if a.model=='argn' else u_gate()
