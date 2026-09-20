"""Fixed checkpoints, fixed train plans, preregistered fresh sampling tapes."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from experiments import cs_saf_calibration_u as ucal
from experiments import cs_saf_calibration as ecal
from experiments.cs_saf_calibration_u import ROOT,parent,state_digest,frozen_source,sha256,write_json
from experiments.cof_seqgen_saf_training import _seed_everything
from experiments.cs_saf_followup_external import canonical_and_plan,sample_to_frame
from experiments.cs_saf_generation_metrics import CachedGenerationMetrics

CONFIG=ROOT/'configs/benchmark_v2/cs_saf_generation_repeats_v1.json'
CONFIG_SHA='99f139e6b650b8aec18bf284aef72262f7159fb5d455fed88d548c5f1dd38bc8'
OUTPUT=ROOT/'artifacts/cs_saf/generation_repeats_v1'


def contract():
    if sha256(CONFIG)!=CONFIG_SHA:raise ValueError('generation repeat registration changed')
    c=json.loads(CONFIG.read_text())
    for name,expected in c['frozen_sources'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('frozen source changed: '+name)
    ucal.contract()
    return c


def folder_for(pi,kappa,trial):
    return OUTPUT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}'


def finish(folder,source,**extra):
    if (folder/'COMPLETE.json').exists():raise FileExistsError('completed output is immutable')
    write_json(folder/'COMPLETE.json',dict(source_commit=source,config_sha256=CONFIG_SHA,
        status='COMPLETE',test_accessed=False,
        artifact_sha256={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()},**extra))


def verify(folder):
    m=json.loads((folder/'COMPLETE.json').read_text())
    if m['config_sha256']!=CONFIG_SHA or m['status']!='COMPLETE':raise ValueError('output contract mismatch')
    for n,h in m['artifact_sha256'].items():
        if sha256(folder/n)!=h:raise ValueError('changed output: '+str(folder/n))
    return m


def load_fixed(payload,pi,kappa,trial,name,device):
    base_name=name.replace('cal','')
    base,provenance=ucal.load_parent(payload,pi,kappa,trial,base_name,device)
    if name.endswith('cal'):
        module=ucal if name=='Ucal' else ecal
        old_folder=module.OUTPUT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/{name}'
        terminal=module.verify(old_folder)
        cp_path=old_folder/'checkpoint_calibrated.pt'
        cp=torch.load(cp_path,map_location='cpu')
        if cp['parent']['checkpoint_sha256']!=provenance['checkpoint_sha256']:
            raise ValueError('calibration parent differs')
        model=module.calibrated_model(base,payload,device)
        model.load_state_dict(cp['model_state'],strict=True)
        if ucal.base_digest(model)!=provenance['state_sha256']:raise ValueError('changed base tensors')
    else:
        model=base;old_folder=parent.folder_for(pi,kappa,trial,name)
        cp_path=old_folder/'checkpoint_best.pt';terminal=parent.verify_artifacts(old_folder)
    model.eval();model.requires_grad_(False)
    saved=torch.load(old_folder/'generated_sample.pt',map_location='cpu')
    if saved['model_state_sha256']!=state_digest(model):raise ValueError('frozen model differs from historical generation')
    source=dict(checkpoint_path=str(cp_path),checkpoint_sha256=sha256(cp_path),
        state_sha256=state_digest(model),base_state_sha256=provenance['state_sha256'],
        reference_folder=str(old_folder),reference_manifest_sha256=sha256(old_folder/'COMPLETE.json'),
        reference_source_commit=terminal['source_commit'],
        fixed_calibration=(dict(offset=model.calibration_offset.tolist(),slope=model.calibration_slope.tolist())
                           if name.endswith('cal') else None))
    return model,source,saved


@torch.no_grad()
def generate_fixed_plan(model,payload,positions,seed,device,batch_size=256):
    # Plan is passed in; the fresh seed must never resample contexts/lengths.
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    parts=[]
    for start in range(0,len(positions),batch_size):
        ids=torch.tensor(positions[start:start+batch_size])
        sample=model.sample_fixed_lengths(payload['train']['lengths'][ids].tolist(),
            static_categorical=(payload['train']['codes'][ids],),device=device)
        mask=sample['valid_mask'].clone();mask[:,0]=False
        reps=torch.tensor(model.support.representatives,device=device,dtype=sample['gap'].dtype)
        if ((~torch.isin(sample['gap'][mask],reps)).any()
            or (sample['receiver'][sample['valid_mask']]<3).any()
            or not torch.isfinite(sample['numeric_value'][sample['valid_mask']]).all()
            or not torch.isnan(sample['gap'][:,0]).all()):raise ValueError('invalid generated events')
        padded={}
        for key,values in sample.items():
            values=values.cpu()
            if values.ndim==2:
                fill=float('nan') if key=='gap' else 0
                full=torch.full((len(values),32),fill,dtype=values.dtype)
                full[:,:values.shape[1]]=values;values=full
            padded[key]=values
        parts.append(padded)
    combined={key:torch.cat([p[key] for p in parts]) for key in parts[0]}
    combined['static_codes']=payload['train']['codes'][torch.tensor(positions)]
    combined['plan_train_indices']=torch.tensor(positions)
    return combined


def compare_samples(a,b):
    if set(a)!=set(b):raise ValueError('sample keys differ')
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,equal_nan=True)


def run_cell(pi,kappa,trial,device):
    c=contract();source=frozen_source();out=folder_for(pi,kappa,trial);started=time.monotonic()
    if (out/'COMPLETE.json').exists():
        result=verify(out)
        if result['source_commit']!=source:raise ValueError('resume source differs')
        return result
    out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(1);seeds,_=parent.trial_config(trial);_seed_everything(seeds['model_seed'])
        dataset,payload,plan,positions=canonical_and_plan(pi,kappa,trial)
        evaluator=CachedGenerationMetrics(dataset,plan)
        plan_sha=hashlib.sha256(positions.tobytes()).hexdigest();inputs={};records=[]
        torch.save(dict(positions=positions,contexts=payload['train']['codes'][positions],
                        lengths=payload['train']['lengths'][positions],plan_seed=seeds['sampling_seed']),out/'plan.pt')
        for name in c['models']:
            model,provenance,original=load_fixed(payload,pi,kappa,trial,name,device)
            inputs[name]=provenance
            np.testing.assert_array_equal(original['sample']['plan_train_indices'].numpy(),positions)
            if original['sampling_seed']!=seeds['sampling_seed']:raise ValueError('historical seed differs')
            for repeat,seed in enumerate(c['generation_seeds'][str(trial)]):
                folder=out/name/f'repeat_{repeat}';folder.mkdir(parents=True,exist_ok=False)
                t0=time.monotonic();sample=generate_fixed_plan(model,payload,positions,seed,device,c['generation_batch_size'])
                sample_seconds=time.monotonic()-t0
                torch.save(dict(sample=sample,sampling_seed=seed,plan_seed=seeds['sampling_seed'],
                    model_state_sha256=provenance['state_sha256'],test_accessed=False),folder/'generated_sample.pt')
                frame=sample_to_frame(sample,payload,plan,positions);metrics=evaluator.score(frame)
                if state_digest(model)!=provenance['state_sha256']:raise ValueError('sampling changed model or calibration')
                report=dict(prevalence=pi,kappa=kappa,trial=trial,model=name,repeat=repeat,
                    sampling_seed=seed,sampling_plan_sha256=plan_sha,metrics=metrics,
                    sample_seconds=sample_seconds,seconds=time.monotonic()-t0,test_accessed=False)
                write_json(folder/'comparison.json',report);finish(folder,source)
                records.append(dict(model=name,repeat=repeat,sampling_seed=seed,
                                    sample_seconds=sample_seconds,seconds=report['seconds']))
            if sha256(Path(provenance['checkpoint_path']))!=provenance['checkpoint_sha256']:
                raise ValueError('checkpoint file changed')
            del model
        write_json(out/'inputs.json',inputs)
        write_json(out/'execution.json',dict(source_commit=source,device=str(device),torch=torch.__version__,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            seconds=time.monotonic()-started,records=records,all_states_unchanged=True,
            new_fits=0,plan_sha256=plan_sha))
        finish(out,source,generated_datasets=len(records));return verify(out)
    except Exception as exc:
        write_json(out/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),source_commit=source))
        raise


def smoke(device,name):
    c=contract();source=frozen_source();out=OUTPUT/name;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);_seed_everything(20260920)
    payload=parent.load_cache(parent.CACHE/'pi_0.05_kappa_1.pt')
    old_seed=parent.trial_config(0)[0]['sampling_seed']
    positions=np.random.default_rng(old_seed).integers(len(payload['train']['lengths']),size=2048)
    checks={}
    for model_name in c['models']:
        model,provenance,old=load_fixed(payload,.05,1,0,model_name,device)
        tiny=positions[:8]
        a=generate_fixed_plan(model,payload,tiny,2026091998,device)
        b=generate_fixed_plan(model,payload,tiny,2026091998,device);compare_samples(a,b)
        reloaded,_,_=load_fixed(payload,.05,1,0,model_name,device)
        compare_samples(a,generate_fixed_plan(reloaded,payload,tiny,2026091998,device))
        del reloaded
        changed=generate_fixed_plan(model,payload,tiny,2026091999,device)
        for key in ('lengths','static_codes','valid_mask','plan_train_indices'):
            torch.testing.assert_close(a[key],changed[key],rtol=0,atol=0)
        if torch.equal(a['numeric_value'],changed['numeric_value']):raise ValueError('seed change failed to change draws')
        if state_digest(model)!=provenance['state_sha256']:raise ValueError('gate changed model')
        torch.save(dict(sample=a),out/(model_name+'.pt'))
        compare_samples(torch.load(out/(model_name+'.pt'),map_location='cpu')['sample'],a)
        historical=False
        if str(device).startswith('cuda'):
            reproduced=generate_fixed_plan(model,payload,positions,old_seed,device)
            compare_samples(reproduced,old['sample']);historical=True
        checks[model_name]=dict(repeatable=True,changed_draw_same_plan=True,
            frozen_weights_and_calibration=True,model_reload=True,sample_reload=True,
            original_full_gpu_sample_identical=historical)
    write_json(out/'gate.json',dict(decision='PASS',source_commit=source,config_sha256=CONFIG_SHA,
        device=str(device),checks=checks))
    finish(out,source)
