"""Bounded post-failure lambda/prevalence extension; frozen historical code is unchanged."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import yaml
from experiments import cs_saf_replication as parent
from experiments.cs_saf_pilot import ROOT, batch, subset, evaluate, state_digest, frozen_source, audit_model, decide_pilot
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json
CONTRACT_SHA = 'd3d768a5b49f35bb7ee9803adf2424116de88f6f026ed4c07f6dd7cf9bf89b2e'
CANDIDATES = ('U','E','L003','ER','L030')
CACHE = ROOT/'artifacts/cs_saf/prepared_v1'
HISTORICAL = ROOT/'artifacts/cs_saf/replication_v1/gpu_v1'
OUTPUT = ROOT/'artifacts/cs_saf/followup_v1'


def load_contract():
    p=ROOT/'configs/benchmark_v2/cs_saf_followup_v1.yaml'
    if sha256(p)!=CONTRACT_SHA: raise ValueError('follow-up contract changed')
    c=yaml.safe_load(p.read_text())
    for key in ('parent_contract','parent_result'):
        if sha256(ROOT/c[key])!=c[key+'_sha256']: raise ValueError('frozen parent changed')
    for path,expected in c['frozen_sources'].items():
        if sha256(ROOT/path)!=expected: raise ValueError('frozen source changed: '+path)
    _,cfg=parent.load_contract()
    return c,cfg


trial_config=parent.trial_config
load_cache=parent.load_cache
verify_initialization=parent.verify_initialization


def make_model(payload,candidate,device,trial_index):
    c,_=load_contract(); spec=c['candidates'][candidate]
    model=parent.make_model(payload,spec['parent'],device,trial_index)
    if candidate!='U': model.regularization_coefficient=spec['lambda']
    return model


def folder_for(pi,kappa,trial,candidate):
    c,_=load_contract()
    if pi==.05 and candidate in c['internal']['reuse_pi005']:
        return HISTORICAL/f'pi_{pi:.2f}/trial_{trial}/kappa_{kappa}'/c['candidates'][candidate]['parent']
    return OUTPUT/f'internal/pi_{pi:.2f}/trial_{trial}/kappa_{kappa}/{candidate}'


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
                  "diagnostic_kind": "unused_balanced_repeat_BCE" if candidate == "U" else "residual_RMS",
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



def internal_job(pi,kappa,trial,candidate,device):
    output=folder_for(pi,kappa,trial,candidate)
    if output.is_relative_to(HISTORICAL): raise ValueError('never retrain/overwrite historical job')
    model,payload,report=train(CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt',output,candidate,device,trial_index=trial)
    _,cfg=trial_config(trial)
    with torch.no_grad():
        audit=audit_model(model,payload,cfg,device,sample_output=output/'generated_sample.pt')
    write_json(output/'intervention_audit.json',audit)
    diagnostic=parent.save_accuracy(model,payload,output,kappa,device)
    diagnostic['config_sha256']=CONTRACT_SHA
    diagnostic['auditor_parent_config_sha256']=parent.CONTRACT_SHA
    write_json(output/'conditional_accuracy.json',diagnostic)
    if state_digest(model)!=report['best_state_sha256']: raise RuntimeError('audit mutated model')
    finish(output)


def finish(folder,**extra):
    result={'status':'COMPLETE','source_commit':frozen_source(),'config_sha256':CONTRACT_SHA,
            'test_accessed':False,'artifact_sha256':{p.name:sha256(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='COMPLETE.json'},**extra}
    write_json(folder/'COMPLETE.json',result)
    return result


def verify_artifacts(folder):
    result=json.loads((folder/'COMPLETE.json').read_text())
    for name,expected in result['artifact_sha256'].items():
        if sha256(folder/name)!=expected: raise RuntimeError('artifact changed: '+str(folder/name))
    return result


def cpu_gate(output):
    source=frozen_source();output.mkdir(parents=True,exist_ok=False)
    results={}
    for candidate in CANDIDATES:
        reports=[];checks=[]
        for repeat in (1,2):
            folder=output/candidate/f'run_{repeat}'
            model,payload,report=train(CACHE/'pi_0.05_kappa_1.pt',folder,candidate,torch.device('cpu'),trial_index=0,cpu=True)
            cp=torch.load(folder/'checkpoint_best.pt',map_location='cpu')
            reload=make_model(payload,candidate,torch.device('cpu'),0)
            reload.load_state_dict(cp['model_state']);reload.eval();model.eval()
            x=batch(payload['validation'],torch.arange(len(payload['validation']['lengths'])),torch.device('cpu'))
            with torch.no_grad():
                a=model.loss_terms(**x);b=reload.loss_terms(**x)
                _,cfg=trial_config(0)
                audit=audit_model(reload,payload,dict(cfg,generation_entities=32,generation_batch_size=32),torch.device('cpu'))
            checks.append(all(torch.equal(a[k],b[k]) for k in a) and state_digest(reload)==report['best_state_sha256'] and audit['zero_gap_control_max_range']<=1e-8 and audit['gap_support_violations']==0 and audit['invalid_reserved_marks']==0 and audit['finite_generated_values'])
            reports.append(report)
        a,b=reports
        drop=1-min(h['train_objective'] for h in a['history'])/a['history'][0]['train_objective']
        results[candidate]={'deterministic_histories':a['history']==b['history'],'deterministic_state':a['best_state_sha256']==b['best_state_sha256'],'reload_and_validity':all(checks),'loss_drop':drop,'PASS':all(checks) and a['history']==b['history'] and a['best_state_sha256']==b['best_state_sha256'] and drop>=.10}
    # Existing pinned external adapter must execute before any scientific fit.
    import runpy
    _dataset=runpy.run_path(str(ROOT/"tests/test_cof_seqgen_saf_external_baselines.py"))["_dataset"]
    from generators.cof_seqgen_saf_external_baselines import SDVCPARWrapper
    from generators.cof_seqgen_saf_baselines import build_shared_generation_plan,validate_raw_generated_events
    d=_dataset();w=SDVCPARWrapper(epochs=1,cuda=False).fit(d)
    plan=build_shared_generation_plan(d,n_entities=3,seed=11)
    generated=w.sample(plan);validate_raw_generated_events(generated,plan)
    result={'decision':'PASS' if all(x['PASS'] for x in results.values()) else 'FAIL','source_commit':source,'config_sha256':CONTRACT_SHA,'candidates':results,'CPAR_CPU_adapter_smoke':'PASS','test_accessed':False}
    write_json(output/'COMPLETE.json',result)
    return result


def parameter_block(name):
    if name.startswith('encoder.'): return 'encoder'
    if name.startswith('copy_base.'): return 'copy_base'
    if name=='history_interaction_weight': return 'history_alpha'
    if name.startswith('route_context'): return 'route_context'
    if name.startswith('route_'): return 'route_gap_interaction'
    if name.startswith('gap_route.'): return 'gap_embedding'
    return 'other_heads_and_context'


def gradient_geometry(a,b):
    na=float(a.norm());nb=float(b.norm());dot=float(a@b)
    return {'a_norm':na,'b_norm':nb,'dot':dot,'cosine':dot/(na*nb) if na>0 and nb>0 else None,'b_over_a_norm':nb/na if na>0 else None}


def gradient_job(kappa,trial,device):
    output=OUTPUT/f'gradients/trial_{trial}/kappa_{kappa}';output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    payload=load_cache(CACHE/f'pi_0.05_kappa_{kappa}.pt');data=payload['train']
    results=[]
    for candidate in ('E','ER'):
        folder=folder_for(.05,kappa,trial,candidate)
        verify_artifacts(folder)
        for snapshot in ('best','epoch_9'):
            path=folder/f'checkpoint_{snapshot}.pt'
            if not path.exists(): results.append({'candidate':candidate,'snapshot':snapshot,'missing':True});continue
            file_before=sha256(path);cp=torch.load(path,map_location=device)
            model=make_model(payload,candidate,device,trial);model.load_state_dict(cp['model_state']);model.eval()
            before=state_digest(model);names,params=zip(*model.named_parameters())
            groups=sorted(set(parameter_block(n) for n in names))
            for label in (0,1):
                eligible=torch.where(data['codes']==label+3)[0].numpy()
                ids=torch.tensor(np.random.default_rng(20261301+trial).choice(eligible,size=min(256,len(eligible)),replace=False))
                x=batch(data,ids,device);terms=model.loss_terms(**x)
                mark=terms['mark_sum']/terms['mark_count']
                base=sum(terms[f'{n}_sum']/terms[f'{n}_count'] for n in ('gap','mark','value'))
                penalty=terms['residual_per_entity'].sum()/(data['lengths'][ids]-1).sum().to(device)
                def vector(value):
                    g=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
                    return [torch.zeros_like(p).flatten().double() if a is None else a.detach().flatten().double() for a,p in zip(g,params)]
                gm,gb,gp=vector(mark),vector(base),vector(penalty)
                context=model.context(model.encoder(**x),x['static_categorical'])
                codes=x['static_categorical'][0][:,None].expand_as(x['valid_mask'])
                h,raw,delta=model.components(context,codes)
                mask=x['valid_mask'].clone();mask[:,0]=False
                pi=model.reference_probabilities[label].to(device=device,dtype=raw.dtype)
                gh=vector(h[mask].mean());gmean=vector((raw*pi).sum(-1)[mask].mean())
                blocks={}
                for block in [*groups,'all']:
                    indices=[i for i,n in enumerate(names) if block=='all' or parameter_block(n)==block]
                    join=lambda vectors:torch.cat([vectors[i] for i in indices])
                    p=.01*join(gp)
                    blocks[block]={'mark_vs_scaled_penalty':gradient_geometry(join(gm),p),'base_vs_scaled_penalty':gradient_geometry(join(gb),p),'history_directional_derivative':float(-(join(gh)@p)),'raw_gap_mean_directional_derivative':float(-(join(gmean)@p))}
                results.append({'candidate':candidate,'snapshot':snapshot,'checkpoint_sha256':file_before,'state_sha256':before,'label':label,'entities':len(ids),'subset_entity_ids_sha256':hashlib.sha256(json.dumps([data['entity_ids'][i] for i in ids]).encode()).hexdigest(),'mark_NLL':float(mark),'base_NLL':float(base),'penalty':float(penalty),'blocks':blocks})
            if sha256(path)!=file_before or state_digest(model)!=before: raise RuntimeError('gradient audit changed checkpoint')
    write_json(output/'gradient_diagnostic.json',{'kappa':kappa,'trial':trial,'results':results,'optimizer_steps':0,'role':'local_train_subset_not_historical_causality','test_accessed':False})
    finish(output)
