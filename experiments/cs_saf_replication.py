"""Preregistered fixed-model training-randomness/prevalence replication."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from experiments.cs_saf_pilot import ROOT, batch, subset, evaluate, state_digest, frozen_source, decide_pilot
from experiments.cs_saf_v4 import load_contract as load_parent_contract
from experiments.cs_saf_v3 import accuracy_audit, COLUMNS
from experiments.cs_saf_loss_control import summarize
from experiments.cs_saf_route_decomposition import reference_measure, audit_payload
from models.cs_saf_v2 import CSSAFv2
from models.cs_saf_v3 import CSSAFv3
from models.cs_saf_v4 import CSSAFv4
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONTRACT_SHA = 'ffbbfefaa7b236e6a595a149e78c6a610ec5e77f75a508398c901814fa562e0e'
CANDIDATES = ('CS2-U1', 'CS3-E1', 'CS4-ER1')
PRIMARY = 'CS4-ER1'
BANK_PARAMETERS = ('route_context_weight', 'route_context_bias', 'route_gap_weight')


def load_contract():
    path = ROOT/'configs/benchmark_v2/cs_saf_replication_v1.yaml'
    if sha256(path) != CONTRACT_SHA:
        raise ValueError('replication registration changed')
    c = yaml.safe_load(path.read_text())
    for key in ('parent_result', 'parent_contract'):
        if sha256(ROOT/c[key]) != c[key+'_sha256']:
            raise ValueError('registered parent changed: '+key)
    for name, expected in c['frozen_sources'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('frozen model/auditor changed: '+name)
    _, cfg = load_parent_contract()
    cfg['pilot_candidates'] = list(CANDIDATES)
    return c, cfg


def trial_config(trial_index):
    c, cfg = load_contract()
    if not 0 <= trial_index < len(c['trials']):
        raise ValueError('unregistered trial')
    trial = c['trials'][trial_index]
    cfg['model_seed'] = trial['model_seed']
    cfg['sampling_seed'] = trial['sampling_seed']
    return trial, cfg


def parse_cell(path):
    c, _ = load_contract()
    allowed = {f'pi_{p:.2f}_kappa_{k}': (p,k) for p in c['prevalences'] for k in c['kappas']}
    if path.stem not in allowed:
        raise ValueError('unregistered prevalence/cell')
    return allowed[path.stem]


def load_cache(path):
    c, _ = load_contract()
    pi, _ = parse_cell(path)
    index_path = path.parent/'COMPLETE.json'
    if sha256(index_path) != c['prepared_index_sha256']:
        raise ValueError('immutable prepared index changed')
    index = json.loads(index_path.read_text())
    if sha256(path) != index['cells'][path.stem]['cache_sha256']:
        raise ValueError('prepared cache changed')
    if index['oracle_gates'][f'{pi:.2f}']['decision'] != 'PASS':
        raise ValueError('train-only oracle prerequisite failed')
    payload = torch.load(path, map_location='cpu')
    if payload['data_manifest_sha256'] != index['cells'][path.stem]['data_manifest_sha256']:
        raise ValueError('data manifest mismatch')
    return payload


class ReplicationU(CSSAFv2):
    """Only stabilize the constant-zero-gap audit; all training stays inherited."""
    @torch.no_grad()
    def response_curves(self, context, previous, *, zero_gap=False, static_codes=None):
        if not zero_gap:
            return super().response_curves(context, previous, static_codes=static_codes)
        if context.ndim != 2:
            raise ValueError('response audit requires flattened histories')
        self.slots(static_codes, context.shape[:-1])
        q = self.copy_base(context).sigmoid()
        fresh = self.new_mark_head(context).clone()
        fresh[:, :3] = -torch.inf
        previous_fresh = fresh.softmax(-1).gather(1, previous[:, None])
        shape = (len(context), len(self.support.representatives))
        return q.expand(shape), (q+(1-q)*previous_fresh).expand(shape)


def initialize_banks(model, bank_seed):
    generator = torch.Generator(device='cpu').manual_seed(bank_seed)
    with torch.no_grad():
        for name, fanin in (('route_context_weight',136), ('route_context_bias',136), ('route_gap_weight',32)):
            tensor = getattr(model, name)
            if tensor.device.type != 'cpu':
                raise ValueError('initialize all banks on CPU before device transfer')
            tensor.uniform_(-1/math.sqrt(fanin), 1/math.sqrt(fanin), generator=generator)


def make_model(payload, candidate, device, trial_index):
    if candidate not in CANDIDATES:
        raise ValueError('unregistered candidate')
    trial, _ = trial_config(trial_index)
    s = SAFTensorizerState.from_dict(payload['tensorizer_state'])
    if candidate == 'CS2-U1':
        model = ReplicationU(candidate, s.gap_support, **s.model_config_kwargs())
    else:
        with torch.random.fork_rng(devices=[]):
            coder = CSSAFv2('CS2-U1', s.gap_support, **s.model_config_kwargs())
            reference, _ = reference_measure(coder, payload['train'])
        if candidate == 'CS3-E1':
            model = CSSAFv3(candidate, s.gap_support, reference, **s.model_config_kwargs())
        else:
            model = CSSAFv4(s.gap_support, reference, **s.model_config_kwargs())
    initialize_banks(model, trial['bank_seed'])
    return model.to(device)


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_initialization(model, payload, trial_index):
    trial, _ = trial_config(trial_index)
    s = SAFTensorizerState.from_dict(payload['tensorizer_state'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(trial['model_seed'])
        expected = CSSAFv2('CS2-U1', s.gap_support, **s.model_config_kwargs())
        initialize_banks(expected, trial['bank_seed'])
    common = expected.state_dict()
    if not all(torch.equal(model.state_dict()[k].cpu(), value) for k, value in common.items()):
        raise ValueError('paired fresh common initialization mismatch')
    count = sum(p.numel() for p in model.parameters())
    has_history = hasattr(model, 'history_interaction_weight')
    if count != (133581 if has_history else 133549):
        raise ValueError('frozen parameter count changed')
    if has_history and torch.count_nonzero(model.history_interaction_weight):
        raise ValueError('history initial coefficients must be zero')
    return {'parameters': count, 'common_state_sha256': tensor_digest(common),
            'bank_state_sha256': tensor_digest({k:model.state_dict()[k] for k in BANK_PARAMETERS}),
            'paired_fresh_common_initialization': True, 'trial_seeds': trial}


def train(cache, output, candidate, device, *, trial_index, cpu=False):
    commit = frozen_source()
    contract, _ = load_contract()
    trial, cfg = trial_config(trial_index)
    if candidate not in contract["candidates"]:
        raise ValueError("unregistered objective")
    payload = load_cache(cache)
    output.mkdir(parents=True, exist_ok=False)
    train_data, validation = payload["train"], payload["validation"]
    options = dict(cfg["training"])
    if cpu:
        train_data = subset(train_data, cfg["cpu_gate"]["train_entities_per_label"])
        validation = subset(validation, cfg["cpu_gate"]["validation_entities_per_label"])
        options.update({k: cfg["cpu_gate"][k] for k in ("epochs", "batch_size", "learning_rate", "patience")})
    torch.set_num_threads(1)
    _seed_everything(cfg["model_seed"])
    model = make_model(payload, candidate, device, trial_index)
    initial = state_digest(model)
    initialization = verify_initialization(model, payload, trial_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    n = len(train_data["lengths"])
    counts = {c: int((train_data["lengths"][train_data["codes"] == c]-1).sum()) for c in (3, 4)}
    generator = torch.Generator().manual_seed(cfg["model_seed"])
    order_hash = hashlib.sha256()
    history, best, best_epoch, stale = [], float("inf"), -1, 0
    started = time.monotonic()
    for epoch in range(options["epochs"]):
        model.train()
        order = torch.randperm(n, generator=generator)
        order_hash.update(order.numpy().tobytes())
        total, total_base, total_aux = 0., 0., 0.
        for start in range(0, n, options["batch_size"]):
            ids = order[start:start+options["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            terms = model.loss_terms(**batch(train_data, ids, device))
            loss, base, aux = model.objective(terms, train_entities=n, transition_counts_by_code=counts)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(ids)
            total_base += float(base.detach())*len(ids)
            total_aux += float(aux.detach())*len(ids)
        metrics = evaluate(model, validation, options["batch_size"], device)
        if not np.isfinite(metrics["base_nll"]):
            raise FloatingPointError("nonfinite validation loss")
        record = {"epoch": epoch, "train_objective": total/n, "train_base_nll": total_base/n,
                  "train_objective_diagnostic": total_aux/n,
                  "diagnostic_kind": "unused_balanced_repeat_BCE" if candidate == "CS2-U1" else "residual_RMS",
                  "regularization_coefficient": getattr(model, "regularization_coefficient", 0.), "validation": metrics,
                  "entity_order_prefix_sha256": order_hash.hexdigest()}
        history.append(record)
        with (output/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record)+"\n")
        improved = metrics["base_nll"] < best-1e-8
        stale = 0 if improved else stale+1
        checkpoint = {"version": model.architecture_contract()["implementation_version"], "candidate": candidate,
                      "model_state": model.state_dict(), "tensorizer_state": payload["tensorizer_state"],
                      "epoch": epoch, "source_commit": commit, "config_sha256": CONTRACT_SHA,
                      "data_manifest_sha256": payload["data_manifest_sha256"], "trial_index": trial_index,
                      "trial_seeds": trial, "test_accessed": False}
        if improved:
            best, best_epoch = metrics["base_nll"], epoch
            _atomic_torch_save(checkpoint, output/"checkpoint_best.pt")
        if epoch == 9:
            _atomic_torch_save(checkpoint, output/"checkpoint_epoch_9.pt")
        print(f"{cache.stem} {candidate} epoch={epoch} val={metrics['base_nll']:.6f}", flush=True)
        if stale >= options["patience"]:
            break
    checkpoint = torch.load(output/"checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    report = {"source_commit": commit, "config_sha256": CONTRACT_SHA, "candidate": candidate,
              "version": model.architecture_contract()["implementation_version"],
              "data_cell": cache.stem, "cache_sha256": sha256(cache),
              "data_manifest_sha256": payload["data_manifest_sha256"],
              "cpu_gate": cpu, "seed": cfg["model_seed"], "trial_index": trial_index, "trial_seeds": trial, "options": options,
              "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial,
              "initialization_contract": initialization, "architecture": dict(model.architecture_contract(), bank_initializer_seed=trial["bank_seed"],
                  constant_zero_gap_broadcast_audit=True),
              "best_state_sha256": state_digest(model), "best_epoch": best_epoch,
              "best_validation_base_nll": best, "history": history, "epochs_completed": len(history),
              "train_entities": n, "validation_entities": len(validation["lengths"]),
              "transition_counts_by_code": counts, "train_events": int(train_data["lengths"].sum()),
              "seconds": time.monotonic()-started, "checkpoint_sha256": sha256(output/"checkpoint_best.pt"),
              "fixed_epoch_checkpoint_sha256": sha256(output/"checkpoint_epoch_9.pt") if (output/"checkpoint_epoch_9.pt").exists() else None,
              "test_accessed": False, "loaded_content_splits": ["train", "validation"],
              "device": str(device), "torch": torch.__version__}
    write_json(output/"training_report.json", report)
    return model, dict(payload, train=train_data, validation=validation), report



def save_accuracy(model,payload,output,kappa,device,*,historical_paths=None):
    before=state_digest(model);original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    reference,_=reference_measure(model,payload['train'])
    summaries,arrays={},{}
    paths=historical_paths or {name:output/f'checkpoint_{name}.pt' for name in ('best','epoch_9')
                              if (output/f'checkpoint_{name}.pt').exists()}
    if 'best' not in paths:raise ValueError('best checkpoint required')
    for name,path in paths.items():
        file_sha=sha256(path);cp=torch.load(path,map_location=device);model.load_state_dict(cp['model_state'])
        state_before=state_digest(model)
        item={'checkpoint_path':str(path),'checkpoint_sha256':file_sha,'state_sha256':state_before,
              'reference_probabilities':reference.tolist(),'data_audit':audit_payload(model,payload),'splits':{}}
        for split in ('train','validation'):
            summary,raw=accuracy_audit(model,payload[split],reference,kappa,device)
            item['splits'][split]=summary
            arrays.update({f'{name}_{split}_{key}':value for key,value in raw.items()})
        if state_digest(model)!=state_before or sha256(path)!=file_sha:raise RuntimeError('audit changed checkpoint')
        summaries[name]=item
    model.load_state_dict(original)
    if state_digest(model)!=before:raise RuntimeError('model restore failed')
    np.savez_compressed(output/'accuracy_arrays.npz',**arrays)
    result={'checkpoints':summaries,'arrays_sha256':sha256(output/'accuracy_arrays.npz'),
            'source_commit':frozen_source(),'config_sha256':CONTRACT_SHA,'test_accessed':False,
            'missing_snapshots':[name for name in ('best','epoch_9') if name not in paths]}
    write_json(output/'conditional_accuracy.json',result)
    return result


def paired_contrasts(folders):
    pairs = (('CS4-ER1','CS3-E1'), ('CS4-ER1','CS2-U1'), ('CS3-E1','CS2-U1'))
    arrays = {name:np.load(path/'accuracy_arrays.npz', allow_pickle=False) for name,path in folders.items()}
    result = {}
    try:
        for a,b in pairs:
            values = {}
            common = sorted(set(arrays[a].files) & set(arrays[b].files))
            for key in common:
                if not key.endswith('_metrics'):
                    continue
                ids = key.replace('_metrics','_entity_ids')
                if not np.array_equal(arrays[a][ids], arrays[b][ids]):
                    raise ValueError('paired entity alignment failed')
                delta = arrays[a][key]-arrays[b][key]
                values[key] = {m:summarize(delta[:,j]) for j,m in enumerate(COLUMNS)}
            result[a+'_minus_'+b] = values
    finally:
        for data in arrays.values():data.close()
    return result


def seed_statistics(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not 1 <= len(a) <= 5 or not np.isfinite(a).all():
        raise ValueError('one to five finite paired trial effects required')
    n = len(a); mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else None
    se = sd/math.sqrt(n) if sd is not None else None
    interval = [mean-2.7764451051977987*se, mean+2.7764451051977987*se] if n == 5 else None
    return {'n_trials':n, 'values':a.tolist(), 'mean':mean, 'sample_SD':sd, 'standard_error':se,
            'descriptive_95_percent_t_interval':interval, 'negative_count':int((a<0).sum()),
            'nonpositive_count':int((a<=0).sum()), 'interval_not_multiplicity_adjusted':True}


def expansion_gate(response_decisions, accuracy_effects):
    if len(response_decisions) != 5:
        raise ValueError('all five trials required before stage decision')
    response_count = sum(d['decision']=='PASS' for d in response_decisions)
    screens = {}
    for comparator in ('CS3-E1','CS2-U1'):
        effects = accuracy_effects[comparator]
        if len(effects['null']) != 5 or len(effects['active']) != 5:
            raise ValueError('all five paired accuracy effects required')
        null = seed_statistics(effects['null']); active = seed_statistics(effects['active'])
        screens[comparator] = {'null':null, 'active':active,
            'per_trial_joint_direction_pass_count':sum(n<0 and a<=0 for n,a in zip(effects['null'],effects['active'])),
            'PASS':null['mean']<0 and active['mean']<=0}
    return {'response_pass_count':response_count, 'response_trials':response_decisions,
            'accuracy_screens':screens,
            'decision':'PASS' if response_count==5 and all(x['PASS'] for x in screens.values()) else 'FAIL',
            'not_statistical_superiority_or_noninferiority':True}


def stage_summary(jobs, stage_path, prevalence):
    contract, cfg = load_contract()
    if len(jobs) != 30:
        raise ValueError('all 30 fits required before scientific decision')
    per_trial = {}
    effects = {c:{'null':[],'active':[]} for c in ('CS3-E1','CS2-U1')}
    response_gates = {c:[] for c in CANDIDATES}
    for trial in range(5):
        paired = {}
        for kappa in (0,1):
            keys = {c:f'trial_{trial}/kappa_{kappa}/{c}' for c in CANDIDATES}
            reports = [jobs[keys[c]]['training'] for c in CANDIDATES]
            if len({r['initialization_contract']['common_state_sha256'] for r in reports}) != 1:
                raise ValueError('common initialization mismatch')
            if reports[1]['initial_state_sha256'] != reports[2]['initial_state_sha256']:
                raise ValueError('E/ER full initial states mismatch')
            for field in ('cache_sha256','options','trial_seeds','train_entities','validation_entities','transition_counts_by_code'):
                if not all(r[field]==reports[0][field] for r in reports):
                    raise ValueError('paired training mismatch: '+field)
            for epoch in range(min(len(r['history']) for r in reports)):
                if len({r['history'][epoch]['entity_order_prefix_sha256'] for r in reports}) != 1:
                    raise ValueError('paired training order mismatch')
            if len({jobs[keys[c]]['audit']['sampling_plan_sha256'] for c in CANDIDATES}) != 1:
                raise ValueError('paired generation plan mismatch')
            references = [jobs[keys[c]]['accuracy']['checkpoints']['best']['reference_probabilities'] for c in CANDIDATES]
            if not all(ref==references[0] for ref in references):
                raise ValueError('evaluation reference differs')
            paired[str(kappa)] = paired_contrasts({c:stage_path/keys[c] for c in CANDIDATES})
        local_gates = {}
        for c in CANDIDATES:
            local_gates[c] = decide_pilot({k:jobs[f'trial_{trial}/kappa_{k}/{c}']['audit'] for k in (0,1)},cfg['pilot_gate'])
            response_gates[c].append(local_gates[c])
        for comparator in effects:
            pair = PRIMARY+'_minus_'+comparator
            null = np.mean([paired[str(k)][pair][f'best_validation_label_{y}_metrics']['grid_mark_TV']['mean']
                            for k,y in ((0,0),(0,1),(1,0))])
            active = paired['1'][pair]['best_validation_label_1_metrics']['grid_mark_TV']['mean']
            effects[comparator]['null'].append(float(null))
            effects[comparator]['active'].append(active)
        per_trial[str(trial)] = {'seeds':contract['trials'][trial], 'paired_contrasts':paired,
                                 'candidate_response_gates':local_gates}
    aggregate = {}
    for kappa in ('0','1'):
        aggregate[kappa] = {}
        pairs = per_trial['0']['paired_contrasts'][kappa]
        for pair in pairs:
            aggregate[kappa][pair] = {}
            available_keys = sorted(set().union(*(per_trial[str(t)]['paired_contrasts'][kappa][pair] for t in range(5))))
            for key in available_keys:
                available_trials = [t for t in range(5) if key in per_trial[str(t)]['paired_contrasts'][kappa][pair]]
                aggregate[kappa][pair][key] = {'trial_indices':available_trials,
                    'metrics':{metric:seed_statistics([per_trial[str(t)]['paired_contrasts'][kappa][pair][key][metric]['mean']
                               for t in available_trials]) for metric in COLUMNS}}
    gate = expansion_gate(response_gates[PRIMARY], effects)
    return {'status':'COMPLETE','prevalence':prevalence, 'scientific_fits':len(jobs),
            'source_commit':frozen_source(), 'config_sha256':CONTRACT_SHA,
            'trial_results':per_trial, 'paired_seed_aggregates':aggregate,
            'candidate_response_pass_counts':{c:sum(x['decision']=='PASS' for x in response_gates[c]) for c in CANDIDATES},
            'gate':gate, 'jobs':jobs, 'matched_initialization_orders_references_sampling':True,
            'test_accessed':False, 'independent_data_confirmation':False}
