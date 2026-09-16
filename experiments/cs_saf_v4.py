"""Frozen v4 training, reusing the unchanged v3 model internals and TV auditor.

Training and accuracy-persistence bodies intentionally match v3; they resolve
v4's own factory/contract without monkeypatching or changing historical code.
"""
from __future__ import annotations
import hashlib
import json
import time
import numpy as np
import torch
import yaml
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from experiments.cs_saf_pilot import ROOT,batch,subset,evaluate,state_digest,frozen_source
from experiments.cs_saf_loss_control import load_cache, summarize
from experiments.cs_saf_route_decomposition import reference_measure, audit_payload
from models.cs_saf_v2 import CSSAFv2
from models.cs_saf_v4 import CSSAFv4,CANDIDATE
from experiments.cs_saf_v3 import (load_contract as load_parent_contract,
    verify_initialization, accuracy_audit, COLUMNS)
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONTRACT_SHA = 'c7ae36b8d9617f6940c7dca45f679f2b286d2ffcad565c8ac246953242604aa5'
CANDIDATES = (CANDIDATE,)


def load_contract():
    path = ROOT/'configs/benchmark_v2/cs_saf_revision_v4.yaml'
    if sha256(path) != CONTRACT_SHA:
        raise ValueError('v4 contract changed')
    contract = yaml.safe_load(path.read_text())
    for key in ('parent_result', 'inherited_contract'):
        if sha256(ROOT/contract[key]) != contract[key+'_sha256']:
            raise ValueError('registered parent changed: '+key)
    for filename, expected in contract['frozen_parent_files'].items():
        if sha256(ROOT/filename) != expected:
            raise ValueError('frozen parent implementation changed: '+filename)
    _, cfg = load_parent_contract()
    cfg['pilot_candidates'] = list(CANDIDATES)
    cfg['pilot_gate']['primary_candidate'] = CANDIDATE
    return contract, cfg


def make_model(payload,candidate,device):
    if candidate != CANDIDATE:
        raise ValueError('unregistered v4 candidate')
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    # This temporary untrained v2 object only supplies exact bin coding. Restore RNG.
    with torch.random.fork_rng(devices=[]):
        coder=CSSAFv2('CS2-U1',state.gap_support,**state.model_config_kwargs())
        pi,_=reference_measure(coder,payload['train'])
    return CSSAFv4(state.gap_support,pi,**state.model_config_kwargs()).to(device)


def train(cache, output, candidate, device, *, cpu=False):
    commit = frozen_source()
    contract, cfg = load_contract()
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
    model = make_model(payload, candidate, device)
    initial = state_digest(model)
    initialization = verify_initialization(model, payload, cfg["model_seed"])
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
                  "train_residual_penalty": total_aux/n,
                  "regularization_coefficient": model.regularization_coefficient, "validation": metrics,
                  "entity_order_prefix_sha256": order_hash.hexdigest()}
        history.append(record)
        with (output/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record)+"\n")
        improved = metrics["base_nll"] < best-1e-8
        stale = 0 if improved else stale+1
        checkpoint = {"version": model.architecture_contract()["implementation_version"], "candidate": candidate,
                      "model_state": model.state_dict(), "tensorizer_state": payload["tensorizer_state"],
                      "epoch": epoch, "source_commit": commit, "config_sha256": CONTRACT_SHA,
                      "data_manifest_sha256": payload["data_manifest_sha256"], "test_accessed": False}
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
              "cpu_gate": cpu, "seed": cfg["model_seed"], "options": options,
              "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial,
              "initialization_contract": initialization, "architecture": model.architecture_contract(),
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
    paths=historical_paths or {'best':output/'checkpoint_best.pt','epoch_9':output/'checkpoint_epoch_9.pt'}
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
            'source_commit':frozen_source(),'config_sha256':CONTRACT_SHA,'test_accessed':False}
    write_json(output/'conditional_accuracy.json',result)
    return result


def contrasts(folders):
    """Aligned entity contrasts, including the complete 2x2 interaction."""
    arrays = {name: np.load(path/'accuracy_arrays.npz', allow_pickle=False)
              for name, path in folders.items()}
    specifications = {
        'ER_minus_E': {CANDIDATE: 1, 'CS3-E1': -1},
        'ER_minus_U': {CANDIDATE: 1, 'CS2-U1': -1},
        'ER_minus_R': {CANDIDATE: 1, 'CS3-R1': -1},
        'ER_minus_C': {CANDIDATE: 1, 'CS3-C1': -1},
        'C_minus_E': {'CS3-C1': 1, 'CS3-E1': -1},
        'R_minus_C': {'CS3-R1': 1, 'CS3-C1': -1},
        'penalty_by_forward_interaction': {CANDIDATE: 1, 'CS3-E1': -1,
                                           'CS3-R1': -1, 'CS3-C1': 1},
    }
    result = {}
    try:
        anchor = arrays[CANDIDATE]
        for name, coefficients in specifications.items():
            item = {}
            for key in anchor.files:
                if not key.endswith('_metrics'):
                    continue
                ids = key.replace('_metrics', '_entity_ids')
                if any(not np.array_equal(anchor[ids], arrays[c][ids]) for c in coefficients):
                    raise ValueError('paired entity alignment failed')
                delta = sum(weight*arrays[c][key] for c, weight in coefficients.items())
                item[key] = {metric: summarize(delta[:, j]) for j, metric in enumerate(COLUMNS)}
            result[name] = item
    finally:
        for array in arrays.values():
            array.close()
    return result


def accuracy_screens(paired):
    screens = {}
    for comparator in ('E', 'U'):
        pair = 'ER_minus_'+comparator
        null = [paired[str(k)][pair][f'best_validation_label_{y}_metrics']['grid_mark_TV']['mean']
                for k, y in ((0, 0), (0, 1), (1, 0))]
        active = paired['1'][pair]['best_validation_label_1_metrics']['grid_mark_TV']['mean']
        screens[comparator] = {'null_cell_differences': null,
            'equal_null_mean_difference': float(np.mean(null)), 'active_difference': active,
            'PASS': float(np.mean(null)) < 0 and active <= 0}
    return screens
