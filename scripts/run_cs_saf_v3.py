"""Execute the separately preregistered four-arm CS-SAF v3 pilot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from experiments.cs_saf_v3 import (ROOT,CONTRACT_SHA,CANDIDATES,load_contract,train,make_model,
    save_accuracy,contrasts,frozen_source,load_cache,reference_measure,accuracy_audit)
from experiments.cs_saf_loss_control import make_model as make_historical
from experiments.cs_saf_pilot import audit_model,decide_pilot,batch,state_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json


def cpu_gate(cache_root, output):
    commit = frozen_source()
    _, cfg = load_contract()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for candidate in CANDIDATES:
        runs, audits, reload_checks = [], [], []
        for repeat in (1, 2):
            folder = output/candidate/f"run_{repeat}"
            model, payload, report = train(cache_root/"pi_0.05_kappa_1.pt", folder,
                                           candidate, torch.device("cpu"), cpu=True)
            copy = make_model(payload, candidate, torch.device("cpu")).eval()
            copy.load_state_dict(torch.load(folder/"checkpoint_best.pt", map_location="cpu")["model_state"])
            x = batch(payload["validation"], torch.arange(len(payload["validation"]["lengths"])), torch.device("cpu"))
            with torch.no_grad():
                a, b = model.loss_terms(**x), copy.loss_terms(**x)
            reload_checks.append(state_digest(copy) == report["best_state_sha256"]
                                 and all(torch.equal(a[k], b[k]) for k in a))
            audits.append(audit_model(copy, payload, dict(cfg, generation_entities=32, generation_batch_size=32),
                                      torch.device("cpu"), sample_output=folder/"generated_sample.pt"))
            runs.append(report)
        first, second = runs
        drop = (first["history"][0]["train_objective"]-min(x["train_objective"] for x in first["history"]))/abs(first["history"][0]["train_objective"])
        checks = {"training_history_identical": first["history"] == second["history"],
                  "best_model_identical": first["best_state_sha256"] == second["best_state_sha256"],
                  "loss_decrease": drop >= cfg["cpu_gate"]["minimum_relative_train_loss_decrease"],
                  "checkpoint_reload_identical": all(reload_checks),
                  "generation_support": all(a["gap_support_violations"] == 0 for a in audits),
                  "generation_valid": all(a["invalid_reserved_marks"] == 0 and a["finite_generated_values"] for a in audits),
                  "zero_gap_invariant": all(a["zero_gap_control_max_range"] <= 1e-8 for a in audits)}
        results[candidate] = {"checks": checks, "decision": "PASS" if all(checks.values()) else "FAIL",
                              "relative_loss_decrease": drop, "best_state_sha256": first["best_state_sha256"],
                              "initial_state_sha256": first["initial_state_sha256"], "audits": audits}
    paired = len({r["initial_state_sha256"] for r in results.values()}) == 1
    result = {"decision": "PASS" if paired and all(r["decision"] == "PASS" for r in results.values()) else "FAIL",
              "source_commit": commit, "config_sha256": CONTRACT_SHA, "candidates": results,
              "matched_initialization": paired, "test_accessed": False}
    write_json(output/"COMPLETE.json", result)
    print(json.dumps(result), flush=True)
    return result


def validate_cpu(path):
    cpu = json.loads(path.read_text())
    if cpu["source_commit"] != frozen_source() or cpu["config_sha256"] != CONTRACT_SHA or cpu["decision"] != "PASS":
        raise RuntimeError("same-source passing CPU gate required")



def job(cache,output,candidate,device,cpu_path):
    validate_cpu(cpu_path)
    if output.exists():raise FileExistsError('prior output is immutable')
    try:
        model,payload,report=train(cache,output,candidate,device)
        _,cfg=load_contract();kappa=int(cache.stem[-1])
        audit=audit_model(model,payload,cfg,device,sample_output=output/'generated_sample.pt')
        write_json(output/'intervention_audit.json',audit)
        diagnostic=save_accuracy(model,payload,output,kappa,device)
        for label in ('0','1'):
            for metric in ('copy','repeat'):
                detailed=diagnostic['checkpoints']['best']['splits']['validation']['groups'][label]['metrics'][metric+'_range']['mean']
                if abs(detailed-audit['responses'][label][f'mean_{metric}_range'])>1e-6:
                    raise RuntimeError('independent response aggregations disagree')
        if state_digest(model)!=report['best_state_sha256']:raise RuntimeError('analysis changed best state')
        write_json(output/'COMPLETE.json',{'status':'COMPLETE','source_commit':report['source_commit'],
            'config_sha256':CONTRACT_SHA,'test_accessed':False,
            'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    except Exception as exc:
        if output.is_dir() and not (output/'COMPLETE.json').exists():
            write_json(output/'FAILED.json',{'error':type(exc).__name__,'message':str(exc)})
        raise


def historical(cache,output,device,cpu_path):
    validate_cpu(cpu_path);contract,_=load_contract()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((ROOT/contract['historical_U_result']).read_text())['result']
    kappa=int(cache.stem[-1]);key=f'pi_0.05_kappa_{kappa}/CS2-U1'
    payload=load_cache(cache)
    if sha256(cache)!=parent['training'][key]['cache_sha256']:raise RuntimeError('historical cache mismatch')
    model=make_historical(payload,'CS2-U1',device).eval();paths={}
    for name in ('best','epoch_9'):
        path=ROOT/'artifacts/cs_saf/loss_control_v1/gpu_v1'/key/f'checkpoint_{name}.pt'
        if sha256(path)!=parent['worker_terminals'][key]['artifact_sha256'][path.name]:raise RuntimeError('historical checkpoint mismatch')
        cp=torch.load(path,map_location=device);model.load_state_dict(cp['model_state'])
        if state_digest(model)!=parent['diagnostics'][key]['checkpoints'][name]['state_sha256']:raise RuntimeError('historical tensor mismatch')
        paths[name]=path
    diagnostic=save_accuracy(model,payload,output,kappa,device,historical_paths=paths)
    # The identical groupwise 256 batch shape reproduces parent observed BCE.
    error=0.
    for name,item in diagnostic['checkpoints'].items():
        for split,summary in item['splits'].items():
            old=parent['diagnostics'][key]['checkpoints'][name]['splits'][split]
            for label in ('0','1'):
                error=max(error,abs(summary['groups'][label]['metrics']['repeat_BCE']['mean']-old['groups'][label]['metrics']['observed_repeat_BCE']['mean']))
    if error>1e-6:raise RuntimeError('historical factual likelihood mismatch')
    write_json(output/'COMPLETE.json',{'status':'COMPLETE','source_commit':frozen_source(),
        'config_sha256':CONTRACT_SHA,'new_fits':0,'historical_BCE_max_error':error,
        'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})


def run(cache_root,cpu_path,output,gpu):
    validate_cpu(cpu_path);source=frozen_source();contract,cfg=load_contract()
    output.mkdir(parents=True,exist_ok=False)
    training,audits,diagnostics,terminals,paired={},{},{},{},{}
    for kappa in (0,1):
        folders={};cell=output/f'pi_0.05_kappa_{kappa}';cell.mkdir()
        for candidate in (*CANDIDATES,'CS2-U1'):
            folder=cell/candidate;folders[candidate]=folder
            command=[sys.executable,'-m','scripts.run_cs_saf_v3',
                'historical' if candidate=='CS2-U1' else 'job','--cache',str(cache_root/f'pi_0.05_kappa_{kappa}.pt'),
                '--output',str(folder),'--device',f'cuda:{gpu}','--cpu-gate',str(cpu_path)]
            if candidate!='CS2-U1':command+=['--candidate',candidate]
            with (cell/f'{candidate}.log').open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(output/'FAILED.json',{'source_commit':source,'kappa':kappa,'candidate':candidate,'exit_code':result.returncode})
                raise RuntimeError('technical failure: preserve outputs')
            key=f'pi_0.05_kappa_{kappa}/{candidate}';t=json.loads((folder/'COMPLETE.json').read_text())
            if t['source_commit']!=source or t['config_sha256']!=CONTRACT_SHA:raise RuntimeError('source mismatch')
            for name,expected in t['artifact_sha256'].items():
                if sha256(folder/name)!=expected:raise RuntimeError('artifact changed')
            terminals[key]=t;diagnostics[key]=json.loads((folder/'conditional_accuracy.json').read_text())
            if candidate!='CS2-U1':
                training[key]=json.loads((folder/'training_report.json').read_text())
                audits[key]=json.loads((folder/'intervention_audit.json').read_text())
            print(f'Completed {key}',flush=True)
        reports=[training[f'pi_0.05_kappa_{kappa}/{c}'] for c in CANDIDATES]
        if len({r['initial_state_sha256'] for r in reports})!=1:raise RuntimeError('initial states unmatched')
        for epoch in range(min(len(r['history']) for r in reports)):
            if len({r['history'][epoch]['entity_order_prefix_sha256'] for r in reports})!=1:raise RuntimeError('training orders unmatched')
        if len({audits[f'pi_0.05_kappa_{kappa}/{c}']['sampling_plan_sha256'] for c in CANDIDATES})!=1:raise RuntimeError('sampling plans unmatched')
        paired[str(kappa)]=contrasts(folders)
    gates={c:decide_pilot({k:audits[f'pi_0.05_kappa_{k}/{c}'] for k in (0,1)},cfg['pilot_gate']) for c in CANDIDATES}
    screens={}
    for comparator in ('CS3-C1','CS2-U1'):
        pair='CS3-R1_minus_'+comparator
        null=[paired[str(k)][pair][f'best_validation_label_{y}_metrics']['grid_mark_TV']['mean'] for k,y in ((0,0),(0,1),(1,0))]
        active=paired['1'][pair]['best_validation_label_1_metrics']['grid_mark_TV']['mean']
        screens[comparator]={'null_cell_differences':null,'equal_null_mean_difference':float(np.mean(null)),
                            'active_difference':active,'PASS':float(np.mean(null))<0 and active<=0}
    eligible=gates['CS3-R1']['decision']=='PASS' and all(s['PASS'] for s in screens.values())
    result={'status':'COMPLETE','source_commit':source,'config_sha256':CONTRACT_SHA,'scientific_fits':len(training),
            'training':training,'audits':audits,'conditional_accuracy':diagnostics,'paired_contrasts':paired,
            'worker_terminals':terminals,'candidate_response_gates':gates,'accuracy_screen':screens,
            'next_registered_pilot_eligible':eligible,'new_method_success':False,
            'test_accessed':False,'later_stage_started':False,'historical_U_new_fits':0,
            'cpu_gate_sha256':sha256(cpu_path),'prepared_index_sha256':sha256(cache_root/'COMPLETE.json')}
    write_json(output/'COMPLETE.json',result)
    print(json.dumps({'status':'COMPLETE','response_gates':gates,'accuracy_screen':screens,'next_registered_pilot_eligible':eligible}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    a=sub.add_parser('cpu');a.add_argument('--cache-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    for phase in ('job','historical'):
        a=sub.add_parser(phase);a.add_argument('--cache',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        a.add_argument('--device',required=True);a.add_argument('--cpu-gate',type=Path,required=True)
        if phase=='job':a.add_argument('--candidate',choices=CANDIDATES,required=True)
    a=sub.add_parser('run');a.add_argument('--cache-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--cpu-gate',type=Path,required=True);a.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    if args.phase=='cpu':
        if cpu_gate(args.cache_root,args.output)['decision']!='PASS':raise SystemExit(2)
    elif args.phase=='job':job(args.cache,args.output,args.candidate,torch.device(args.device),args.cpu_gate)
    elif args.phase=='historical':historical(args.cache,args.output,torch.device(args.device),args.cpu_gate)
    else:run(args.cache_root,args.cpu_gate,args.output,args.gpu)


if __name__=='__main__':main()
