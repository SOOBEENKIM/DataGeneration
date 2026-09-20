"""Registered, frozen-checkpoint train-only calibration and joint generation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments import cs_saf_followup as parent
from experiments.cs_saf_pilot import ROOT, batch, subset, state_digest, frozen_source, audit_model
from experiments.cs_saf_replication import tensor_digest
from experiments.cof_seqgen_saf_training import _seed_everything
from models.cs_saf_calibration import CalibratedCSSAF, CALIBRATION_BUFFERS, fit_repeat_calibration
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONFIG = ROOT/'configs/benchmark_v2/cs_saf_calibration_v1.json'
CONFIG_SHA = '50b947e4755d44c623fa343d5a8116c197c388e0221e7095b39f1388b558655c'
OUTPUT = ROOT/'artifacts/cs_saf/calibration_v1'


def contract():
    if sha256(CONFIG) != CONFIG_SHA:
        raise ValueError('registered calibration configuration changed')
    c = json.loads(CONFIG.read_text())
    for name, expected in c['frozen_sources'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('frozen input changed: '+name)
    parent.load_contract()
    return c


def base_digest(model):
    return tensor_digest({k:v for k,v in model.state_dict().items() if k not in CALIBRATION_BUFFERS})


def calibrated_model(base, payload, device):
    state = SAFTensorizerState.from_dict(payload['tensorizer_state'])
    model = CalibratedCSSAF(state.gap_support, base.reference_probabilities.cpu(),
                            **state.model_config_kwargs()).to(device)
    weights = dict(base.state_dict(), calibration_offset=model.calibration_offset,
                   calibration_slope=model.calibration_slope)
    model.load_state_dict(weights, strict=True)
    model.regularization_coefficient = base.regularization_coefficient
    model.eval()
    if base_digest(model) != state_digest(base):
        raise ValueError('base weights changed during adapter creation')
    return model


def load_parent(payload, pi, kappa, trial, candidate, device):
    folder = parent.folder_for(pi, kappa, trial, candidate)
    terminal = parent.verify_artifacts(folder)
    cp_path = folder/'checkpoint_best.pt'
    cp = torch.load(cp_path, map_location='cpu')
    old = json.loads((folder/'conditional_accuracy.json').read_text())['checkpoints']['best']
    model = parent.make_model(payload, candidate, device, trial)
    model.load_state_dict(cp['model_state'], strict=True)
    model.eval(); model.requires_grad_(False)
    if state_digest(model) != old['state_sha256']:
        raise ValueError('parent tensor hash mismatch')
    return model, dict(path=str(cp_path), checkpoint_sha256=sha256(cp_path),
                      state_sha256=state_digest(model), parent_manifest_sha256=sha256(folder/'COMPLETE.json'),
                      reference=old['reference_probabilities'], parent_source=terminal['source_commit'])


@torch.no_grad()
def extract_train_features(model, train, device, batch_size=512):
    fields = {k:[] for k in ('logit','fresh_previous','equality','code','entity_index','event_index')}
    for start in range(0, len(train['lengths']), batch_size):
        ids = torch.arange(start, min(start+batch_size, len(train['lengths'])))
        x = batch(train, ids, device)
        context = model.context(model.encoder(**x), x['static_categorical'])
        mask = x['valid_mask'].clone(); mask[:,0] = False
        rows, cols = torch.where(mask)
        previous = x['receiver'].roll(1, dims=1)[mask]
        codes = x['static_categorical'][0][:,None].expand_as(mask)[mask]
        c = context[mask]
        z = model.copy_logits(c, x['gap'][mask], static_codes=codes)
        fresh = model.new_mark_head(c).clone(); fresh[:,:3] = -torch.inf
        fp = fresh.softmax(-1).gather(1, previous[:,None])[:,0]
        for key, value in dict(logit=z, fresh_previous=fp,
                equality=x['receiver'][mask] == previous, code=codes,
                entity_index=rows+start, event_index=cols).items():
            fields[key].append(value.cpu().numpy())
    result = {k:np.concatenate(v) for k,v in fields.items()}
    if len(result['logit']) != int((train['lengths']-1).sum()):
        raise ValueError('training transitions were lost or duplicated')
    return result


def finish(folder, source, **extra):
    write_json(folder/'COMPLETE.json', dict(status='COMPLETE', source_commit=source,
        config_sha256=CONFIG_SHA, test_accessed=False,
        artifact_sha256={p.name:sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}, **extra))


def verify(folder):
    result = json.loads((folder/'COMPLETE.json').read_text())
    if result['config_sha256'] != CONFIG_SHA:
        raise ValueError('completed contract mismatch')
    for name, expected in result['artifact_sha256'].items():
        if sha256(folder/name) != expected:
            raise ValueError('output changed: '+str(folder/name))
    return result


def fit_job(pi, kappa, trial, candidate, device):
    c = contract(); source = frozen_source(); started = time.monotonic()
    folder = OUTPUT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/{candidate}cal'
    if (folder/'COMPLETE.json').exists():
        result = verify(folder)
        if result['source_commit'] != source: raise ValueError('resume source changed')
        return result
    folder.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(1)
        seeds, cfg = parent.trial_config(trial)
        _seed_everything(seeds['model_seed'])
        payload = parent.load_cache(parent.CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt')
        base, provenance = load_parent(payload, pi, kappa, trial, candidate, device)
        features = extract_train_features(base, payload['train'], device, c['feature_batch_size'])
        np.savez_compressed(folder/'train_features.npz', **features)
        parameters = fit_repeat_calibration(features, c['optimizer'])
        write_json(folder/'fit.json', dict(parameters, source_commit=source,
            config_sha256=CONFIG_SHA, parent=provenance,
            train_entity_ids_sha256=hashlib.sha256(json.dumps(payload['train']['entity_ids']).encode()).hexdigest(),
            feature_sha256=sha256(folder/'train_features.npz'),
            train_entities=len(payload['train']['lengths']), transitions=len(features['logit']),
            cache_sha256=sha256(parent.CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt'),
            validation_used_for_calibration=False, fit_seconds=time.monotonic()-started))
        del features
        model = calibrated_model(base, payload, device)
        model.set_calibration(parameters)
        torch.save(dict(model_state=model.state_dict(), parent=provenance,
                        parameters=parameters, source_commit=source, config_sha256=CONFIG_SHA),
                   folder/'checkpoint_calibrated.pt')
        before = state_digest(model)
        audit = audit_model(model, payload, cfg, device, sample_output=folder/'generated_sample.pt')
        if (audit['gap_support_violations'] or audit['invalid_reserved_marks'] or
                not audit['finite_generated_values'] or audit['zero_gap_control_max_range'] > 1e-8):
            raise ValueError('calibrated generation validity failed')
        write_json(folder/'intervention_audit.json', audit)
        # Oracle evaluation is isolated after the train-only fit and checkpoint save.
        from experiments.cs_saf_v3 import accuracy_audit
        summary, arrays = accuracy_audit(model, payload['validation'],
            torch.tensor(provenance['reference'], dtype=torch.float64), kappa, device)
        np.savez_compressed(folder/'validation_accuracy_arrays.npz', **arrays)
        write_json(folder/'conditional_accuracy.json', summary)
        for label in ('0','1'):
            for metric in ('copy','repeat'):
                a = summary['groups'][label]['metrics'][metric+'_range']['mean']
                b = audit['responses'][label]['mean_'+metric+'_range']
                if abs(a-b) > 1e-6: raise ValueError('independent response aggregations disagree')
        from experiments.cs_saf_followup_external import canonical_and_plan, sample_to_frame, common_metrics
        dataset, _, plan, positions = canonical_and_plan(pi,kappa,trial)
        sample = torch.load(folder/'generated_sample.pt', map_location='cpu')['sample']
        generated = sample_to_frame(sample, payload, plan, positions)
        metrics = common_metrics(dataset, generated, plan)
        write_json(folder/'comparison.json', dict(candidate=candidate+'cal', prevalence=pi,
            kappa=kappa, trial=trial, metrics=metrics,
            sampling_plan_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
            sampling_seed=seeds['sampling_seed'], test_accessed=False))
        if state_digest(model) != before or base_digest(model) != provenance['state_sha256']:
            raise RuntimeError('evaluation changed frozen weights')
        if sha256(Path(provenance['path'])) != provenance['checkpoint_sha256']:
            raise RuntimeError('parent checkpoint changed')
        write_json(folder/'execution.json', dict(device=str(device), torch=torch.__version__,
            source_commit=source, seconds=time.monotonic()-started,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deployed_parameters=model.architecture_contract(), base_unchanged=True))
        finish(folder, source, parent_checkpoint_sha256=provenance['checkpoint_sha256'])
        return verify(folder)
    except Exception as exc:
        write_json(folder/'FAILED.json', dict(error=type(exc).__name__, message=str(exc), source_commit=source))
        raise


def smoke(device, name):
    c = contract(); source = frozen_source(); folder = OUTPUT/name
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); _seed_everything(20260920)
    full = parent.load_cache(parent.CACHE/'pi_0.05_kappa_1.pt')
    train = subset(full['train'], 32); checks = {}
    for candidate in c['parents']:
        base, provenance = load_parent(full,.05,1,0,candidate,device)
        model = calibrated_model(base,full,device)
        x = batch(train,torch.arange(64),device)
        with torch.no_grad():
            a,b = base.loss_terms(**x), model.loss_terms(**x)
        if not all(torch.equal(a[k],b[k]) for k in a):
            raise AssertionError('identity adapter changed likelihood')
        features = extract_train_features(base,train,device,32)
        a = fit_repeat_calibration(features,c['optimizer'])
        b = fit_repeat_calibration(features,c['optimizer'])
        if a != b: raise AssertionError('non-deterministic fit')
        codes = (torch.tensor([3,4,3,4]),)
        _seed_everything(20260920); original = base.sample_fixed_lengths([4,5,6,7],static_categorical=codes,device=device)
        _seed_everything(20260920); identity = model.sample_fixed_lengths([4,5,6,7],static_categorical=codes,device=device)
        for k in original:
            torch.testing.assert_close(original[k],identity[k],rtol=0,atol=0,equal_nan=True)
        model.set_calibration(a)
        cp=folder/(candidate+'.pt'); torch.save(model.state_dict(),cp)
        reload=calibrated_model(base,full,device);reload.load_state_dict(torch.load(cp,map_location=device))
        if state_digest(model)!=state_digest(reload) or base_digest(model)!=provenance['state_sha256']:
            raise AssertionError('reload/frozen state mismatch')
        sample=model.sample_fixed_lengths([4,5,6,7],static_categorical=codes,device=device)
        if (sample['receiver'][sample['valid_mask']]<3).any() or not torch.isfinite(sample['numeric_value']).all():
            raise AssertionError('invalid calibrated sample')
        checks[candidate]=dict(identity_likelihood=True, identity_generation=True,
            deterministic_fit=True, base_unchanged=True, reload=True, valid_sample=True, fit=a)
    write_json(folder/'gate.json',dict(decision='PASS',source_commit=source,
        config_sha256=CONFIG_SHA, device=str(device),checks=checks))
    finish(folder,source)
    return checks
