"""Registered matched-structure training and immutable-checkpoint evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch

# Same explicit FP32 policy for all three structures; avoid TF32/CPU drift.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cs_saf_pilot import batch, subset, evaluate, state_digest
from experiments.cs_saf_replication import ReplicationU, initialize_banks, load_cache
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from experiments.cs_saf_generation_repeats import generate_fixed_plan
from models.cs_saf_structure import ObservedRepeatStructure
from models.cs_saf_observed_repeat_control import fit_controls, apply_torch
from scripts.run_cs_saf_u_repeat_controls import extract, u_control, to_frame
from scripts.cs_saf_repeat_control_common import conditional_metrics
from scripts.run_cs_saf_external_audit_v1 import digest, write

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/benchmark_v2/cs_saf_structure_v1.json'
OUT = ROOT / 'artifacts/cs_saf/structure_v1'
DOC = ROOT / 'docs/cs_saf/structure_v1'
OLD = ROOT / 'artifacts/cs_saf/external_audit_v1'
SOURCES = ['models/cs_saf_structure.py', 'experiments/cs_saf_structure_v1.py',
    'models/cs_saf.py', 'models/cs_saf_v2.py', 'models/cof_seqgen_saf.py',
    'experiments/cs_saf_replication.py', 'experiments/cs_saf_pilot.py',
    'experiments/cs_saf_generation_repeats.py', 'models/cs_saf_observed_repeat_control.py',
    'scripts/run_cs_saf_u_repeat_controls.py', 'scripts/cs_saf_repeat_control_common.py',
    'experiments/cs_saf_generation_metrics.py', 'data/cof_seqgen_saf_tensorizer.py',
    'configs/benchmark_v2/cs_saf_structure_v1.json']


def config():
    return json.loads(CONFIG.read_text())


def source_record():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name in SOURCES:
        if (ROOT/name).read_bytes() != subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT):
            raise RuntimeError('commit reviewed implementation before execution: '+name)
    return dict(commit=commit, hashes={name: digest(ROOT/name) for name in SOURCES})


def payload_for(kappa):
    cache = ROOT.parent / f'research-cs-saf/artifacts/cs_saf/prepared_v1/pi_0.10_kappa_{kappa}.pt'
    payload = load_cache(cache)
    return payload, dict(cache_path=str(cache), cache_sha256=digest(cache),
        tensorizer_sha256=hashlib.sha256(json.dumps(payload['tensorizer_state'], sort_keys=True).encode()).hexdigest())


def folder_for(kappa, trial, name):
    return OUT / f'runs/kappa_{kappa}/trial_{trial}/{name}'


def make_model(payload, name, trial, device='cpu'):
    c = config(); seed = c['trials'][trial]
    _seed_everything(seed['model_seed'])
    state = SAFTensorizerState.from_dict(payload['tensorizer_state'])
    if name == 'U':
        model = ReplicationU('CS2-U1', state.gap_support, **state.model_config_kwargs())
    else:
        model = ObservedRepeatStructure(name, state.gap_support, **state.model_config_kwargs())
    initialize_banks(model, seed['bank_seed'])
    assert sum(p.numel() for p in model.parameters()) == c['parameters_each']
    return model.to(device)


def train(kappa, trial, name, device, smoke_name=None):
    c = config(); assert kappa in c['kappas'] and name in c['models'] and 0 <= trial < 3
    source = source_record()
    if smoke_name is None:
        assert str(device).startswith('cuda'), 'scientific fits wait for a GPU'
        gate = json.loads((DOC/'gpu_gate.json').read_text())
        assert gate['status'] == 'PASS' and gate['source_hashes'] == source['hashes']
    torch.set_num_threads(1)
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
    payload, provenance = payload_for(kappa)
    dest = folder_for(kappa, trial, name) if smoke_name is None else OUT/'smoke'/smoke_name
    dest.mkdir(parents=True, exist_ok=False)
    write(dest/'START.json', dict(kappa=kappa, trial=trial, model=name, source=source,
        data=provenance, config_sha256=digest(CONFIG), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        smoke=smoke_name is not None, test_accessed=False))
    tr, va = payload['train'], payload['validation']
    opts = {key:c[key] for key in ['epochs','patience','batch_size','learning_rate','weight_decay','gradient_clip']}
    if smoke_name is not None:
        tr, va = subset(tr, 32), subset(va, 16)
        opts.update(epochs=25, patience=25, batch_size=16, learning_rate=.003)
    started = time.monotonic()
    try:
        model = make_model(payload, name, trial, device)
        initial = state_digest(model)
        initial_train = evaluate(model, tr, opts['batch_size'], device)['base_nll'] if smoke_name else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts['learning_rate'], weight_decay=opts['weight_decay'])
        counts = {g:int((tr['lengths'][tr['codes']==g]-1).sum()) for g in (3,4)}
        order_generator = torch.Generator().manual_seed(c['trials'][trial]['model_seed'])
        order_hash = hashlib.sha256(); history=[]; best=float('inf'); stale=0; best_epoch=-1
        for epoch in range(opts['epochs']):
            model.train(); order=torch.randperm(len(tr['lengths']), generator=order_generator)
            order_hash.update(order.numpy().tobytes()); train_sum=0.
            for start in range(0,len(order),opts['batch_size']):
                ids=order[start:start+opts['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                terms=model.loss_terms(**batch(tr,ids,device))
                loss,base,_=model.objective(terms,train_entities=len(order),transition_counts_by_code=counts)
                assert torch.equal(loss,base) and torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),opts['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();train_sum+=float(loss.detach())*len(ids)
            val=evaluate(model,va,opts['batch_size'],device)
            assert np.isfinite(list(val.values())).all()
            row=dict(epoch=epoch,train_base_nll=train_sum/len(order),validation=val,order_sha256=order_hash.hexdigest())
            history.append(row)
            with (dest/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            improved=val['base_nll']<best-1e-8
            stale=0 if improved else stale+1
            checkpoint=dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                tensorizer_state=payload['tensorizer_state'],epoch=epoch,name=name,trial=trial,kappa=kappa,
                config_sha256=digest(CONFIG),source=source,data=provenance,test_accessed=False)
            if improved:
                best=val['base_nll'];best_epoch=epoch
                _atomic_torch_save(checkpoint,dest/'checkpoint_best.pt')
            print(f'TRAIN {kappa} {trial} {name} epoch={epoch} val={val["base_nll"]:.6f}',flush=True)
            if stale>=opts['patience']:break
        _atomic_torch_save(checkpoint,dest/'checkpoint_last.pt')
        final_train = evaluate(model,tr,opts['batch_size'],device)['base_nll'] if smoke_name else None
        model.load_state_dict(torch.load(dest/'checkpoint_best.pt',map_location=device)['model_state'])
        report=dict(kappa=kappa,trial=trial,name=name,source=source,data=provenance,
            initial_state_sha256=initial,best_state_sha256=state_digest(model),
            config_sha256=digest(CONFIG),parameters=sum(p.numel() for p in model.parameters()),
            architecture=model.architecture_contract(),options=opts,best_epoch=best_epoch,best_validation_nll=best,
            epochs=len(history),history=history,seconds=time.monotonic()-started,
            checkpoint_sha256=digest(dest/'checkpoint_best.pt'),last_checkpoint_sha256=digest(dest/'checkpoint_last.pt'),
            initial_train_nll=initial_train,
            final_train_nll=final_train,
            best_train_nll=evaluate(model,tr,opts['batch_size'],device)['base_nll'] if smoke_name else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else None,
            device=str(device),torch=torch.__version__,smoke=smoke_name is not None,test_accessed=False,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        write(dest/'TRAIN_DONE.json',report)
        return model,report
    except Exception as exc:
        write(dest/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),source=source))
        raise


@torch.no_grad()
def response_diagnostic(model,data,controls):
    values={kind:{g:[] for g in (0,1)} for kind in ['raw']+list(controls)}
    preservation=0.; ng=len(model.support.representatives)
    for start in range(0,len(data['lengths']),128):
        ids=torch.arange(start,min(start+128,len(data['lengths'])))
        x=batch(data,ids,'cpu');hidden=model.encoder(**x);context=model.context(hidden,x['static_categorical'])
        mask=x['valid_mask'].clone();mask[:,0]=False
        rows,_=torch.where(mask);h=context[mask];codes=x['static_categorical'][0][rows]
        previous=x['receiver'].roll(1,1)[mask]
        if isinstance(model,ObservedRepeatStructure):
            r=model.repeat_logits(h,codes).sigmoid()
            if model.structure_mode=='C':
                a=model.gap_decoder.logits(h[:,:model.config.hidden_dim]).softmax(-1)
                preservation=max(preservation,float(((a*r).sum(-1)-model.copy_base(h).squeeze(-1).sigmoid()).abs().max()))
        else:r=model.response_curves(h,previous,static_codes=codes)[1]
        grid=torch.arange(ng).expand(len(h),-1);groups=(codes-3)[:,None].expand_as(grid)
        for kind,control in [('raw',None)]+list(controls.items()):
            corrected=apply_torch(r.reshape(-1),grid.reshape(-1),groups.reshape(-1),control).reshape_as(r)
            ranges=corrected.max(-1).values-corrected.min(-1).values
            sums=torch.zeros(len(ids)).scatter_add_(0,rows,ranges)
            counts=torch.bincount(rows,minlength=len(ids))
            entity=sums/counts
            for g in (0,1):values[kind][g].extend(entity[x['static_categorical'][0]==g+3].tolist())
    return dict(entity_mean_observable_repeat_range={kind:{str(g):float(np.mean(v)) for g,v in groups.items()}
        for kind,groups in values.items()},raw_C_marginal_max_error=preservation,
        scope='all_validation_histories_same_history_gap_grid; observational_not_causal',
        posthoc_controls_do_not_preserve_original_C_rho=True)


def evaluate_saved(kappa,trial,name):
    c=config();torch.set_num_threads(1);source=source_record()
    folder=folder_for(kappa,trial,name);report=json.loads((folder/'TRAIN_DONE.json').read_text())
    assert not report['smoke'] and source['hashes']==report['source']['hashes']
    dest=folder/'evaluation';dest.mkdir(exist_ok=False);started=time.monotonic()
    try:
        payload,provenance=payload_for(kappa);assert provenance==report['data']
        cp=folder/'checkpoint_best.pt';assert digest(cp)==report['checkpoint_sha256']
        model=make_model(payload,name,trial)
        model.load_state_dict(torch.load(cp,map_location='cpu')['model_state']);model.eval();model.requires_grad_(False)
        before=state_digest(model);assert before==report['best_state_sha256']
        state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
        features=extract(model,payload['train']);features.to_parquet(dest/'train_features.parquet',index=False)
        controls=fit_controls(features,c);controls={kind:controls[kind] for kind in c['variants'] if kind!='raw'}
        write(dest/'fit.json',dict(controls=controls,source_checkpoint_sha256=digest(cp),config_sha256=digest(CONFIG),
            feature_sha256=digest(dest/'train_features.parquet'),training_entities=len(payload['train']['lengths']),
            fitting_split='train',validation_used=False,oracle_used=False))
        vf=extract(model,payload['validation']);vf.to_parquet(dest/'validation_features.parquet',index=False)
        write(dest/'conditional.json',conditional_metrics(features,vf,controls))
        write(dest/'response.json',response_diagnostic(model,payload['validation'],controls))
        inp=OLD/f'kappa_{kappa}/input';prov=json.loads((inp/'provenance.json').read_text())
        for file,sha in prov['files'].items():assert digest(inp/file)==sha
        plan=pd.read_parquet(inp/'plan.parquet')
        positions=pd.Index(payload['train']['entity_ids']).get_indexer(plan.source_train_entity_id)
        assert (positions>=0).all()
        np.testing.assert_array_equal(payload['train']['lengths'][positions],plan['__saf_planned_length'])
        np.testing.assert_array_equal(payload['train']['codes'][positions]-3,plan.entity_label.astype(int))
        generations=[]
        for kind,control in [('raw',None)]+list(controls.items()):
            for tape in c['generation_seeds']:
                with u_control(model,control):sample=generate_fixed_plan(model,payload,positions,tape,'cpu',c['generation_batch_size'])
                generated=to_frame(sample,payload,state,plan)
                file=dest/f'generated_{kind}_{tape}.parquet';generated.to_parquet(file,index=False)
                generations.append(dict(variant=kind,tape=tape,file=file.name,sha256=digest(file)))
                print('GENERATED',kappa,trial,name,kind,tape,flush=True)
        assert state_digest(model)==before and digest(cp)==report['checkpoint_sha256']
        assert digest(provenance['cache_path'])==provenance['cache_sha256']
        done=dict(kappa=kappa,trial=trial,name=name,config_sha256=digest(CONFIG),source=source,
            checkpoint_sha256=digest(cp),weights_unchanged=True,seconds=time.monotonic()-started,
            files={p.name:digest(p) for p in dest.iterdir() if p.is_file()},generations=generations,
            test_accessed=False,training_features=len(features),validation_features=len(vf))
        write(folder/'EVAL_DONE.json',done)
        return done
    except Exception as exc:
        write(dest/'FAILED.json',dict(error=type(exc).__name__,message=str(exc)))
        raise
