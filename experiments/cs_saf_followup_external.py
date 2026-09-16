"""Matched generation-only CPAR comparison; no invented CPAR oracle/copy score."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,time,importlib.metadata
import numpy as np
import pandas as pd
import torch
from experiments.cs_saf_followup import (ROOT,OUTPUT,CACHE,load_contract,load_cache,trial_config,folder_for,finish,verify_artifacts)
from experiments.cof_seqgen_saf_training import _seed_everything
from data.cof_seqgen_saf_tensorizer import load_canonical_dataset,SAFTensorizerState
from generators.cof_seqgen_saf_baselines import SharedGenerationPlan,validate_raw_generated_events
from generators.cof_seqgen_saf_external_baselines import SDVCPARWrapper,EmpiricalSequenceSampler
from benchmarks.cof_seqgen_saf_metrics import fit_metric_state,evaluate_metric_suite
from experiments.cs_saf_cpar_loss import equivalent_par_loss,PINNED_PAR_SOURCE_SHA256
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json


def canonical_and_plan(pi,kappa,trial):
    path=ROOT/f'data/cs_saf/prevalence_v1/pi_{pi:.2f}_kappa_{kappa}'
    payload=load_cache(CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt')
    if sha256(path/'cs_saf_manifest.json')!=payload['data_manifest_sha256']:raise ValueError('canonical manifest mismatch')
    manifest=json.loads((path/'cs_saf_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha256(path/name)!=expected:raise ValueError('canonical data modified')
    dataset=load_canonical_dataset(path,allowed_splits=('train','validation'))
    trial_seeds,_=trial_config(trial);seed=trial_seeds['sampling_seed']
    positions=np.random.default_rng(seed).integers(len(payload['train']['lengths']),size=2048)
    source_ids=tuple(payload['train']['entity_ids'][int(i)] for i in positions)
    if not set(source_ids)<=set(dataset.entity_ids_for_split('train')):raise ValueError('plan not train-only')
    ids=tuple(f'saf-synthetic-{i:08d}' for i in range(2048))
    static=dataset.static_context.set_index('entity_id').loc[list(source_ids)].reset_index()
    static['entity_id']=ids
    lengths=tuple(payload['train']['lengths'][positions].tolist())
    plan=SharedGenerationPlan(ids,source_ids,lengths,static,'train',seed)
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    np.testing.assert_array_equal(state.static_categorical_codecs[0][1].encode(static.entity_label),payload['train']['codes'][positions].numpy())
    return dataset,payload,plan,positions


def sample_to_frame(sample,payload,plan,positions):
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    np.testing.assert_array_equal(sample['plan_train_indices'].numpy(),positions)
    np.testing.assert_array_equal(sample['lengths'].numpy(),plan.lengths)
    np.testing.assert_array_equal(sample['static_codes'].numpy(),payload['train']['codes'][positions].numpy())
    mask=sample['valid_mask'].numpy();rows,cols=np.where(mask)
    expected=np.arange(32)[None,:]<np.asarray(plan.lengths)[:,None]
    np.testing.assert_array_equal(mask,expected)
    gap=sample['gap'].numpy()[mask].astype(np.float64)
    frame=pd.DataFrame({'entity_id':np.asarray(plan.entity_ids)[rows],'event_index':cols,'gap':gap,
          'receiver_or_mark':state.receiver_codec.decode(sample['receiver'].numpy()[mask]),
          'amount_or_numeric_value':state.event_numeric_codecs[0][1].decode(sample['numeric_value'].numpy()[mask])})
    frame['event_id']=[f'{entity}-{pos}' for entity,pos in zip(frame.entity_id,cols)]
    frame['timestamp']=frame['gap'].fillna(0).groupby(frame.entity_id,sort=False).cumsum()
    validate_raw_generated_events(frame,plan)
    return frame


def common_metrics(dataset,generated,plan):
    validate_raw_generated_events(generated,plan)
    if not set(generated.receiver_or_mark)<=set(dataset.events[dataset.events.entity_id.isin(dataset.entity_ids_for_split('train'))].receiver_or_mark):
        raise ValueError('unknown generated mark')
    result={}
    for group in ('pooled','context_0','context_1'):
        if group=='pooled':real_ids=set(dataset.static_context.entity_id);gen_ids=set(plan.entity_ids)
        else:
            label=int(group[-1]);real_ids=set(dataset.static_context.loc[dataset.static_context.entity_label==label,'entity_id'])
            gen_ids=set(plan.static_context.loc[plan.static_context.entity_label==label,'entity_id'])
        train=dataset.events[dataset.events.entity_id.isin(real_ids & set(dataset.entity_ids_for_split('train')))]
        validation=dataset.events[dataset.events.entity_id.isin(real_ids & set(dataset.entity_ids_for_split('validation')))]
        synth=generated[generated.entity_id.isin(gen_ids)]
        state=fit_metric_state(train)
        scores=evaluate_metric_suite(validation,synth,state,include_privacy=False)
        if not np.isfinite(list(scores.values())).all():raise FloatingPointError('nonfinite common generation score')
        result[group]={'generated_entities':len(gen_ids),'validation_entities':validation.entity_id.nunique(),'train_metric_state':asdict(state),'metrics':scores}
    return result


def external_job(pi,kappa,trial,device):
    c,_=load_contract();torch.set_num_threads(1)
    if not str(device).startswith('cuda'):raise ValueError('scientific CPAR uses assigned GPU')
    torch.cuda.set_device(device)
    for name,version in c['external']['package_versions'].items():
        if importlib.metadata.version(name)!=version:raise RuntimeError('pinned external version changed')
    folder=OUTPUT/f'external/pi_{pi:.2f}/trial_{trial}/kappa_{kappa}'
    folder.mkdir(parents=True,exist_ok=False)
    dataset,payload,plan,positions=canonical_and_plan(pi,kappa,trial)
    plan.static_context.assign(source_train_entity_id=plan.source_train_entity_ids,planned_length=plan.lengths).to_parquet(folder/'generation_plan.parquet',index=False)
    trial_seeds,_=trial_config(trial)
    _seed_everything(trial_seeds['model_seed'])
    started=time.monotonic()
    with equivalent_par_loss():
        wrapper=SDVCPARWrapper(epochs=c['external']['epochs'],sample_size=1,cuda=True,verbose=True).fit(dataset)
    fit_seconds=time.monotonic()-started
    wrapper.model.save(folder/'model.pkl')
    losses=wrapper.model.get_loss_values()
    losses.to_csv(folder/'loss_history.csv',index=False)
    print('CPAR FIT COMPLETE',fit_seconds,flush=True)
    _seed_everything(trial_seeds['sampling_seed'])
    started=time.monotonic();generated=wrapper.sample(plan);sample_seconds=time.monotonic()-started
    generated.to_parquet(folder/'generated.parquet',index=False)
    metrics=common_metrics(dataset,generated,plan)
    empirical=EmpiricalSequenceSampler().fit(dataset).sample(plan)
    empirical.to_parquet(folder/'empirical_generated.parquet',index=False)
    report={'prevalence':pi,'kappa':kappa,'trial':trial,'seeds':trial_seeds,'versions':c['external']['package_versions'],
        'epochs':128,'sample_size':1,'loss_implementation':'vectorized_pinned_0.8.1_equivalent','pinned_PAR_source_sha256':PINNED_PAR_SOURCE_SHA256,'network_parameters':sum(p.numel() for p in wrapper.model._model._model.parameters()),'network_device':str(next(wrapper.model._model._model.parameters()).device),'fit_seconds':fit_seconds,'sample_seconds':sample_seconds,'fit_record':asdict(wrapper.fit_record),
        'cache_sha256':sha256(CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt'),'sampling_plan_sha256':hashlib.sha256(positions.tobytes()).hexdigest(),
        'data_manifest_sha256':payload['data_manifest_sha256'],'CPAR':metrics,'empirical_copy_control':common_metrics(dataset,empirical,plan),
        'generation_entities':2048,'device':str(device),'raw_likelihood_compared':False,'conditional_oracle_TV_available':False,
        'test_accessed':False,'known_active_label_mask_used':False,'loaded_content_splits':['train','validation']}
    write_json(folder/'comparison.json',report);finish(folder)


def internal_generation_job(pi,kappa,trial,candidate):
    folder=OUTPUT/f'generation/pi_{pi:.2f}/trial_{trial}/kappa_{kappa}/{candidate}';folder.mkdir(parents=True,exist_ok=False)
    source=folder_for(pi,kappa,trial,candidate);verify_artifacts(source)
    dataset,payload,plan,positions=canonical_and_plan(pi,kappa,trial)
    sample=torch.load(source/'generated_sample.pt',map_location='cpu')['sample']
    generated=sample_to_frame(sample,payload,plan,positions)
    write_json(folder/'comparison.json',{'candidate':candidate,'prevalence':pi,'kappa':kappa,'trial':trial,
        'sample_sha256':sha256(source/'generated_sample.pt'),'sampling_plan_sha256':hashlib.sha256(positions.tobytes()).hexdigest(),
        'metrics':common_metrics(dataset,generated,plan),'test_accessed':False})
    finish(folder)
