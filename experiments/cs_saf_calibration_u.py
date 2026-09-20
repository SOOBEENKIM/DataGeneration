"""Preregistered equal-calibration U control; historical E artifacts are read-only."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments import cs_saf_calibration as prior
from experiments.cs_saf_calibration import (parent, ROOT, batch, subset, state_digest,
    frozen_source, audit_model, tensor_digest, _seed_everything, fit_repeat_calibration,
    sha256, write_json, base_digest, load_parent, extract_train_features,
    verify_response_precision)
from models.cs_saf_calibration_u import CalibratedU

CONFIG = ROOT/'configs/benchmark_v2/cs_saf_calibration_u_control_v1.json'
CONFIG_SHA = '772a1193a3336fff8364dddf8dcd6aac954d4a94818056d3c86c6744758b424e'
OUTPUT = ROOT/'artifacts/cs_saf/calibration_u_control_v1'
REUSABLE_SOURCES = set()


def contract():
    if sha256(CONFIG) != CONFIG_SHA:
        raise ValueError('U control registration changed')
    c = json.loads(CONFIG.read_text())
    for name, expected in c['frozen_sources'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('frozen U control source changed: '+name)
    previous = prior.contract()
    for key in ('optimizer','feature_batch_size','generation_entities','generation_batch_size',
                'prevalences','kappas','trials'):
        if c[key] != previous[key]:
            raise ValueError('unequal calibration/generation settings: '+key)
    return c


def calibrated_model(base, payload, device):
    state = SAFTensorizerState.from_dict(payload['tensorizer_state'])
    model = CalibratedU(state.gap_support, **state.model_config_kwargs()).to(device)
    model.load_state_dict(dict(base.state_dict(), calibration_offset=model.calibration_offset,
        calibration_slope=model.calibration_slope), strict=True)
    model.eval()
    if base_digest(model) != state_digest(base):
        raise ValueError('U weights changed or extra history parameters introduced')
    return model


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
    if candidate != 'U': raise ValueError('only preregistered U fits permitted')
    folder = OUTPUT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/{candidate}cal'
    if (folder/'COMPLETE.json').exists():
        result = verify(folder)
        if result['source_commit'] not in REUSABLE_SOURCES|{source}: raise ValueError('resume source changed')
        return result
    folder.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(1)
        seeds, cfg = parent.trial_config(trial)
        _seed_everything(seeds['model_seed'])
        payload = parent.load_cache(parent.CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt')
        base, provenance = load_parent(payload, pi, kappa, trial, candidate, device)
        # Verify the matched Ecal reference and identical sampling conditions.
        reference = prior.OUTPUT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/Ecal'
        reference_manifest = prior.verify(reference)
        reference_fit = json.loads((reference/'fit.json').read_text())
        if reference_fit['cache_sha256'] != sha256(parent.CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt'):
            raise ValueError('U and E calibrated on different data')
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
        precision=verify_response_precision(model,payload['validation'],device,audit,summary)
        write_json(folder/'response_precision_verification.json',precision)
        from experiments.cs_saf_followup_external import canonical_and_plan, sample_to_frame, common_metrics
        dataset, _, plan, positions = canonical_and_plan(pi,kappa,trial)
        sample = torch.load(folder/'generated_sample.pt', map_location='cpu')['sample']
        generated = sample_to_frame(sample, payload, plan, positions)
        metrics = common_metrics(dataset, generated, plan)
        write_json(folder/'comparison.json', dict(candidate=candidate+'cal', prevalence=pi,
            kappa=kappa, trial=trial, metrics=metrics,
            sampling_plan_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
            sampling_seed=seeds['sampling_seed'], test_accessed=False))
        ref_comparison = json.loads((reference/'comparison.json').read_text())
        if ref_comparison['sampling_plan_sha256'] != audit['sampling_plan_sha256'] or ref_comparison['sampling_seed'] != seeds['sampling_seed']:
            raise ValueError('Ucal and Ecal sampling not paired')
        write_json(folder/'paired_reference.json',dict(path=str(reference),
            manifest_sha256=sha256(reference/'COMPLETE.json'), source_commit=reference_manifest['source_commit'],
            equal_optimizer=True, equal_data=True, equal_plan=True, equal_sampling_seed=True))
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
