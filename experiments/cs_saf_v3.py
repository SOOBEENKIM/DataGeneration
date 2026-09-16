"""Frozen eight-fit v3 comparison and observation-only conditional accuracy audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from experiments.cs_saf_pilot import ROOT,batch,subset,evaluate,state_digest,frozen_source,load_pilot_config
from experiments.cs_saf_loss_control import load_cache, summarize
from experiments.cs_saf_route_decomposition import reference_measure, audit_payload
from models.cs_saf_v2 import CSSAFv2
from models.cs_saf_v3 import CSSAFv3,CANDIDATES
from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONTRACT_SHA = '5ca902201d932a990cda77b92a79944812078d437eb4df820e3c7c523a7f6b63'
COLUMNS = ('grid_mark_TV','factual_mark_TV','grid_repeat_L1','repeat_BCE',
           'copy_range','repeat_range','residual_RMS')


def load_contract():
    path=ROOT/'configs/benchmark_v2/cs_saf_revision_v3.yaml'
    if sha256(path)!=CONTRACT_SHA:raise ValueError('v3 contract changed')
    c=yaml.safe_load(path.read_text())
    for name in ('parent_result','historical_U_result','inherited_pilot','oracle_config'):
        if sha256(ROOT/c[name])!=c[name+'_sha256']:raise ValueError('registered input changed: '+name)
    cfg,_=load_pilot_config('v1')
    cfg['pilot_candidates']=list(CANDIDATES);cfg['pilot_gate']['primary_candidate']='CS3-R1'
    return c,cfg


def make_model(payload,candidate,device):
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    # This temporary untrained v2 object only supplies exact bin coding. Restore RNG.
    with torch.random.fork_rng(devices=[]):
        coder=CSSAFv2('CS2-U1',state.gap_support,**state.model_config_kwargs())
        pi,_=reference_measure(coder,payload['train'])
    return CSSAFv3(candidate,state.gap_support,pi,**state.model_config_kwargs()).to(device)


def verify_initialization(model,payload,seed):
    s=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        old=CSSAFv2('CS2-U1',s.gap_support,**s.model_config_kwargs())
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in old.state_dict().items())
    assert torch.count_nonzero(model.history_interaction_weight)==0
    count=sum(p.numel() for p in model.parameters())
    if count!=133581:raise ValueError('registered parameter count changed')
    return {'fresh_v2_common_initialization':True,'parameters':count,
            'dormant_parameters':model.architecture_contract()['dormant_parameter_count'],
            'reference_probabilities':model.reference_probabilities.cpu().tolist()}


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



def mark_tv(q, fresh, previous, oracle_q):
    """Exact TV over observed marks, including nonuniform learned fresh mass."""
    if q.shape != oracle_q.shape or fresh.shape[0] != len(q):
        raise ValueError('conditional distribution shape mismatch')
    n,k=q.shape; categories=fresh.shape[1]
    diff=(1-q[:,:,None])*fresh[:,None,:]-(1-oracle_q[:,:,None])/categories
    diff.scatter_add_(2,previous[:,None,None].expand(n,k,1),(q-oracle_q)[:,:,None])
    return .5*diff.abs().sum(-1)


@torch.no_grad()
def accuracy_audit(model,data,reference,kappa,device,*,batch_size=256):
    contract,_=load_contract()
    raw=yaml.safe_load((ROOT/contract['oracle_config']).read_text())
    production=BenchmarkConfig.from_mapping(raw,'joint_semimarkov_v2b',kappa)
    if model.config.receiver_vocab_size-3 != production.n_receiver_categories:
        raise ValueError('controlled full-mark oracle requires the complete 64-mark vocabulary')
    # Exactly the bin boundaries used by float32 model._support_code.
    oracle=SemiMarkovCopyOracle(production,np.asarray(model.support.upper_bounds,dtype=np.float32).astype(float))
    model.eval();groups,arrays={},{}
    for label in (0,1):
        selected=torch.where(data['codes']==label+3)[0]
        blocks,names=[],[]
        pi=reference[label].to(device=device,dtype=torch.float64)
        for start in range(0,len(selected),batch_size):
            ids=selected[start:start+batch_size]; x=batch(data,ids,device)
            pred=oracle.filter_batch(data['gap'][ids].numpy(),data['receiver'][ids].numpy()-3,
                                     data['lengths'][ids].numpy(),active=bool(kappa*label))
            prior=np.zeros(tuple(x['gap'].shape),dtype=float)
            prior[pred['entity_index'],pred['event_index']]=pred['prior_burst']
            context=model.context(model.encoder(**x),x['static_categorical'])
            mask=x['valid_mask'].clone();mask[:,0]=False
            rows,_=torch.where(mask)
            codes=x['static_categorical'][0][:,None].expand_as(mask)[mask]
            previous=x['receiver'].roll(1,dims=1)[mask]
            actual=x['receiver'][mask]; bins=model._support_code(x['gap'][mask])-3
            oracle_grid=torch.tensor(oracle.copy_curve(prior[mask.cpu().numpy()],bool(kappa*label)),device=device,dtype=torch.float64)
            flat=context[mask];parts=[]
            for j in range(0,len(flat),512):
                sl=slice(j,j+512); c=flat[sl]; prev=previous[sl]
                q,_=model.response_curves(c,prev,static_codes=codes[sl]);q=q.double()
                fresh_logits=model.new_mark_head(c).clone();fresh_logits[:,:3]=-torch.inf
                fresh=fresh_logits.softmax(-1)[:,3:].double()
                fp=fresh.gather(1,(prev-3)[:,None]);repeat=q+(1-q)*fp
                oq=oracle_grid[sl]; op=oq+(1-oq)/64
                tv=mark_tv(q,fresh,prev-3,oq)
                b=bins[sl,None];factual=repeat.gather(1,b)[:,0].clamp(1e-12,1-1e-12)
                equality=actual[sl]==prev
                bce=-torch.where(equality,factual.log(),torch.log1p(-factual))
                logits=torch.logit(q.clamp(1e-12,1-1e-12))
                residual=logits-(logits*pi).sum(1,keepdim=True)
                parts.append(torch.stack([(tv*pi).sum(1),tv.gather(1,b)[:,0],
                    ((repeat-op).abs()*pi).sum(1),bce,q.max(1).values-q.min(1).values,
                    repeat.max(1).values-repeat.min(1).values,(residual.square()*pi).sum(1).sqrt()],dim=1))
            values=torch.cat(parts)
            entity=torch.zeros((len(ids),len(COLUMNS)),device=device,dtype=torch.float64)
            entity.index_add_(0,rows,values/mask.sum(1)[rows,None])
            blocks.append(entity.cpu().numpy());names.extend(data['entity_ids'][int(i)] for i in ids)
        values=np.concatenate(blocks)
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite conditional errors')
        groups[str(label)]={'entities':len(values),'metrics':{name:summarize(values[:,j]) for j,name in enumerate(COLUMNS)}}
        arrays[f'label_{label}_metrics']=values;arrays[f'label_{label}_entity_ids']=np.asarray(names,dtype=str)
    return {'groups':groups,'columns':list(COLUMNS),'entity_batch_size':batch_size,
            'oracle_uses_realized_latents':False,'test_accessed':False},arrays


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
    pairs=(('CS3-E1','CS2-U1'),('CS3-C1','CS3-E1'),('CS3-R1','CS3-C1'),('CS3-R1','CS2-U1'),('CS3-H1','CS2-U1'))
    arrays={name:np.load(path/'accuracy_arrays.npz',allow_pickle=False) for name,path in folders.items()}
    result={}
    for a,b in pairs:
        item={}
        for key in arrays[a].files:
            if not key.endswith('_metrics'):continue
            ids=key.replace('_metrics','_entity_ids')
            if not np.array_equal(arrays[a][ids],arrays[b][ids]):raise ValueError('paired entity alignment failed')
            delta=arrays[a][key]-arrays[b][key]
            item[key]={name:summarize(delta[:,j]) for j,name in enumerate(COLUMNS)}
        result[a+'_minus_'+b]=item
    return result
