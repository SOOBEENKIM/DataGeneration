"""Execute the preregistered 30-fit screen and conditional 90-fit expansion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from experiments.cs_saf_replication import (
    ROOT, CONTRACT_SHA, CANDIDATES, load_contract, trial_config, parse_cell,
    load_cache, train, make_model, save_accuracy, stage_summary, frozen_source,
)
from experiments.cs_saf_pilot import audit_model, batch, state_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json


def cpu_gate(cache_root, output):
    source = frozen_source()
    _, cfg = trial_config(0)
    output.mkdir(parents=True, exist_ok=False)
    candidates = {}
    common_initial = []
    for candidate in CANDIDATES:
        reports, audits, reloads = [], [], []
        for repeat in (1, 2):
            folder = output/candidate/f'run_{repeat}'
            model, payload, report = train(cache_root/'pi_0.05_kappa_1.pt', folder,
                candidate, torch.device('cpu'), trial_index=0, cpu=True)
            copy = make_model(payload, candidate, torch.device('cpu'), 0).eval()
            copy.load_state_dict(torch.load(folder/'checkpoint_best.pt', map_location='cpu')['model_state'])
            x = batch(payload['validation'], torch.arange(len(payload['validation']['lengths'])), torch.device('cpu'))
            with torch.no_grad():
                a, b = model.loss_terms(**x), copy.loss_terms(**x)
            reloads.append(state_digest(copy)==report['best_state_sha256'] and all(torch.equal(a[k],b[k]) for k in a))
            audits.append(audit_model(copy, payload, dict(cfg,generation_entities=32,generation_batch_size=32),
                torch.device('cpu'), sample_output=folder/'generated_sample.pt'))
            reports.append(report)
        a,b = reports
        drop = (a['history'][0]['train_objective']-min(x['train_objective'] for x in a['history']))/abs(a['history'][0]['train_objective'])
        checks = {'repeat_histories':a['history']==b['history'],
            'repeat_states':a['best_state_sha256']==b['best_state_sha256'], 'reload':all(reloads),
            'loss_decrease':drop>=cfg['cpu_gate']['minimum_relative_train_loss_decrease'],
            'support':all(x['gap_support_violations']==0 for x in audits),
            'validity':all(x['invalid_reserved_marks']==0 and x['finite_generated_values'] for x in audits),
            'zero_gap':all(x['zero_gap_control_max_range']<=1e-8 for x in audits)}
        candidates[candidate] = {'checks':checks, 'decision':'PASS' if all(checks.values()) else 'FAIL',
            'relative_loss_decrease':drop,'initial_state_sha256':a['initial_state_sha256'],
            'best_state_sha256':a['best_state_sha256'],'initialization_contract':a['initialization_contract'],'audits':audits}
        common_initial.append(a['initialization_contract']['common_state_sha256'])
    matched = len(set(common_initial))==1 and candidates['CS3-E1']['initial_state_sha256']==candidates['CS4-ER1']['initial_state_sha256']
    result = {'decision':'PASS' if matched and all(x['decision']=='PASS' for x in candidates.values()) else 'FAIL',
        'source_commit':source, 'config_sha256':CONTRACT_SHA, 'matched_initialization':matched,
        'candidates':candidates,'test_accessed':False}
    write_json(output/'COMPLETE.json',result)
    print(json.dumps(result),flush=True)
    return result


def validate_cpu(path):
    result = json.loads(path.read_text())
    if result['source_commit']!=frozen_source() or result['config_sha256']!=CONTRACT_SHA or result['decision']!='PASS':
        raise RuntimeError('same-source passing CPU gate required')


def authorize_cell(cache, study_root):
    prevalence, kappa = parse_cell(cache)
    if prevalence != .05:
        stage1_path = study_root/'pi_0.05'/'SUMMARY.json'
        gate = json.loads((study_root/'STAGE1_GATE.json').read_text())
        if (gate['source_commit']!=frozen_source() or gate['config_sha256']!=CONTRACT_SHA
            or not gate['advance'] or sha256(stage1_path)!=gate['summary_sha256']):
            raise RuntimeError('unchanged passing stage-1 gate required before expansion')
    return prevalence, kappa


def job(cache, output, candidate, trial, device, cpu_path, study_root):
    validate_cpu(cpu_path)
    _, kappa = authorize_cell(cache, study_root)
    if output.exists():raise FileExistsError('existing scientific artifacts are immutable')
    try:
        model,payload,report = train(cache, output, candidate, device, trial_index=trial)
        _, cfg = trial_config(trial)
        audit = audit_model(model,payload,cfg,device,sample_output=output/'generated_sample.pt')
        write_json(output/'intervention_audit.json',audit)
        diagnostic = save_accuracy(model,payload,output,kappa,device)
        for label in ('0','1'):
            for metric in ('copy','repeat'):
                expected = diagnostic['checkpoints']['best']['splits']['validation']['groups'][label]['metrics'][metric+'_range']['mean']
                if abs(expected-audit['responses'][label][f'mean_{metric}_range'])>1e-6:
                    raise RuntimeError('independent response aggregations disagree')
        if state_digest(model)!=report['best_state_sha256']:raise RuntimeError('audit changed model')
        write_json(output/'COMPLETE.json',{'status':'COMPLETE','source_commit':report['source_commit'],
            'config_sha256':CONTRACT_SHA,'trial_index':trial,'test_accessed':False,
            'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    except Exception as exc:
        if output.is_dir() and not (output/'COMPLETE.json').exists():
            write_json(output/'FAILED.json',{'error':type(exc).__name__,'message':str(exc)})
        raise


def available_gpus(allowed):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if ',' in line}
    free = {}
    for line in rows.splitlines():
        index, uuid, memory = (x.strip() for x in line.split(','))
        if int(index) in allowed and uuid not in busy and int(memory)>=2048:
            free[int(index)]={'uuid':uuid,'free_memory_MiB':int(memory),'compute_processes_before_launch':0}
    return free


def collect_job(folder, source):
    terminal = json.loads((folder/'COMPLETE.json').read_text())
    if terminal['source_commit']!=source or terminal['config_sha256']!=CONTRACT_SHA:
        raise RuntimeError('worker source/contract mismatch')
    for name,expected in terminal['artifact_sha256'].items():
        if sha256(folder/name)!=expected:raise RuntimeError('worker artifact changed')
    return {'terminal':terminal,'terminal_sha256':sha256(folder/'COMPLETE.json'),
        'training':json.loads((folder/'training_report.json').read_text()),
        'audit':json.loads((folder/'intervention_audit.json').read_text()),
        'accuracy':json.loads((folder/'conditional_accuracy.json').read_text())}


def execute_stage(prevalence, cache_root, cpu_path, output, gpus):
    source=frozen_source(); stage=output/f'pi_{prevalence:.2f}';stage.mkdir()
    pending=[(trial,k,c) for trial in range(5) for k in (0,1) for c in CANDIDATES]
    running={}; completed={}; allocations=[]; failures=[]
    while pending or running:
        for gpu,item in list(running.items()):
            process,handle,key,folder=item
            code=process.poll()
            if code is None:continue
            handle.close();del running[gpu]
            if code:
                failures.append({'key':key,'exit_code':code,'gpu':gpu})
            else:
                try:
                    completed[key]=collect_job(folder,source)
                    print(f'pi={prevalence:.2f} complete {len(completed)}/30 {key}',flush=True)
                except Exception as exc:
                    failures.append({'key':key,'error':type(exc).__name__,'message':str(exc)})
            write_json(stage/'progress.json',{'completed':list(completed),'inflight':[v[2] for v in running.values()],
                'pending_count':len(pending),'failures':failures,'source_commit':source,'allocations':allocations})
        if failures:
            if running:time.sleep(2);continue
            write_json(stage/'FAILED.json',{'failures':failures,'completed':list(completed),'pending':pending})
            raise RuntimeError('technical failure; outputs preserved and new dispatch stopped')
        if pending:
            try:
                free=available_gpus(gpus)
            except Exception as exc:
                failures.append({'error':type(exc).__name__,'message':str(exc),'phase':'GPU_availability'})
                continue
            for gpu in gpus:
                if not pending:break
                if gpu in running or gpu not in free:continue
                trial,k,c=pending.pop(0);key=f'trial_{trial}/kappa_{k}/{c}';folder=stage/key
                folder.parent.mkdir(parents=True,exist_ok=True)
                handle=(folder.parent/f'{c}.log').open('w')
                command=[sys.executable,'-m','scripts.run_cs_saf_replication','job',
                    '--cache',str(cache_root/f'pi_{prevalence:.2f}_kappa_{k}.pt'),
                    '--output',str(folder),'--candidate',c,'--trial',str(trial),'--device',f'cuda:{gpu}',
                    '--cpu-gate',str(cpu_path),'--study-root',str(output)]
                try:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                except Exception as exc:
                    handle.close()
                    failures.append({'key':key,'error':type(exc).__name__,'message':str(exc),'phase':'worker_launch'})
                    break
                running[gpu]=(process,handle,key,folder)
                allocation={'key':key,'gpu_index':gpu,'pid':process.pid,**free[gpu]}
                allocations.append(allocation);write_json(stage/'gpu_allocations.json',allocations)
                print(f'pi={prevalence:.2f} start {key} gpu={gpu} pid={process.pid}',flush=True)
        if pending or running:time.sleep(2)
    result=stage_summary(completed,stage,prevalence)
    result['gpu_allocations']=allocations
    write_json(stage/'SUMMARY.json',result)
    print(json.dumps({'prevalence':prevalence,'gate':result['gate'],'response_pass_counts':result['candidate_response_pass_counts']}),flush=True)
    return result


def run(cache_root,cpu_path,output,gpus):
    validate_cpu(cpu_path);contract,_=load_contract();source=frozen_source()
    if not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus):raise ValueError('one to four distinct GPUs required')
    for p in contract['prevalences']:
        for k in (0,1):load_cache(cache_root/f'pi_{p:.2f}_kappa_{k}.pt')
    output.mkdir(parents=True,exist_ok=False)
    result={'source_commit':source,'config_sha256':CONTRACT_SHA,'stages':{},'scientific_fits':0,
        'test_accessed':False,'external_baselines_run':False,'independent_data_confirmation':False,
        'cpu_gate_sha256':sha256(cpu_path),'prepared_index_sha256':sha256(cache_root/'COMPLETE.json')}
    try:
        for pi in contract['prevalences']:
            stage=execute_stage(pi,cache_root,cpu_path,output,gpus)
            result['stages'][f'{pi:.2f}']={'summary_path':str(output/f'pi_{pi:.2f}'/'SUMMARY.json'),
                'summary_sha256':sha256(output/f'pi_{pi:.2f}'/'SUMMARY.json'),'gate':stage['gate'],
                'candidate_response_pass_counts':stage['candidate_response_pass_counts']}
            result['scientific_fits']+=30
            write_json(output/'progress.json',result)
            if pi==.05:
                advance=stage['gate']['decision']=='PASS'
                write_json(output/'STAGE1_GATE.json',{'source_commit':source,'config_sha256':CONTRACT_SHA,
                    'summary_sha256':result['stages']['0.05']['summary_sha256'],'advance':advance})
                if not advance:
                    result['stop_reason']='registered_stage1_scientific_gate_failed'
                    break
        result.update(status='COMPLETE',stage2_started=len(result['stages'])>1,
            all_registered_prevalences_complete=len(result['stages'])==4,
            all_prevalence_gates_pass=len(result['stages'])==4 and all(x['gate']['decision']=='PASS' for x in result['stages'].values()),
            new_method_superiority=False)
        write_json(output/'COMPLETE.json',result)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(output/'FAILED.json',dict(result,error=type(exc).__name__,message=str(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    a=sub.add_parser('cpu');a.add_argument('--cache-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('job');a.add_argument('--cache',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--candidate',choices=CANDIDATES,required=True);a.add_argument('--trial',type=int,required=True)
    a.add_argument('--device',required=True);a.add_argument('--cpu-gate',type=Path,required=True);a.add_argument('--study-root',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--cache-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--cpu-gate',type=Path,required=True);a.add_argument('--gpus',type=int,nargs='+',required=True)
    args=p.parse_args()
    if args.phase=='cpu':
        if cpu_gate(args.cache_root,args.output)['decision']!='PASS':raise SystemExit(2)
    elif args.phase=='job':job(args.cache,args.output,args.candidate,args.trial,torch.device(args.device),args.cpu_gate,args.study_root)
    else:run(args.cache_root,args.cpu_gate,args.output,args.gpus)


if __name__=='__main__':main()
