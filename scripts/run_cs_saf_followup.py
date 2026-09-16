"""Run the separately registered, unconditional post-failure extension."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
from experiments.cs_saf_followup import (ROOT,OUTPUT,CACHE,CONTRACT_SHA,CANDIDATES,load_contract,folder_for,
    frozen_source,cpu_gate,internal_job,gradient_job,verify_artifacts)
from scripts.run_cs_saf_replication import available_gpus
from scripts.materialize_cs_saf_prevalence import write_json
from scripts.audit_cs_saf_oracle import sha256


def task_path(task):
    phase,pi,k,t,c=task
    if phase=='internal':return folder_for(pi,k,t,c)
    if phase=='gradient':return OUTPUT/f'gradients/trial_{t}/kappa_{k}'
    if phase=='external':return OUTPUT/f'external/pi_{pi:.2f}/trial_{t}/kappa_{k}'
    return OUTPUT/f'generation/pi_{pi:.2f}/trial_{t}/kappa_{k}/{c}'


def tasks():
    c,_=load_contract()
    internal=[('internal',pi,k,t,a) for pi in c['prevalences'] for t in c['trials'] for k in c['kappas'] for a in CANDIDATES if not (pi==.05 and a in c['internal']['reuse_pi005'])]
    gradient=[('gradient',.05,k,t,'none') for t in range(5) for k in (0,1)]
    external=[('external',pi,k,t,'CPAR') for pi in c['prevalences'] for t in range(5) for k in (0,1)]
    generation=[('generation',pi,k,t,a) for pi in c['prevalences'] for t in range(5) for k in (0,1) for a in CANDIDATES]
    assert len(internal)==170 and len(external)==40 and len(generation)==200
    return internal[:20]+gradient+internal[20:]+external,generation


def validate_cpu():
    path=OUTPUT/'cpu_v1/COMPLETE.json';r=json.loads(path.read_text())
    if r['decision']!='PASS' or r['source_commit']!=frozen_source() or r['config_sha256']!=CONTRACT_SHA:raise RuntimeError('same-source CPU gate required')
    return r


def execute(queue,label,*,cpu=False):
    source=frozen_source();logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
    pending=[];completed=[]
    for task in queue:
        folder=task_path(task)
        if (folder/'COMPLETE.json').exists():
            result=verify_artifacts(folder)
            if result['config_sha256']!=CONTRACT_SHA:raise ValueError('resume contract mismatch')
            completed.append({'task':task,'source_commit':result['source_commit'],'resumed':True})
        elif folder.exists():raise RuntimeError('incomplete artifacts require documented repair/new attempt: '+str(folder))
        else:pending.append(task)
    running={};allocations=[];failures=[]
    def progress():
        write_json(OUTPUT/f'{label}_progress.json',{'completed':completed,'running':[v[2] for v in running.values()],
            'pending':pending,'failures':failures,'allocations':allocations,'source_commit':source,'config_sha256':CONTRACT_SHA})
    while pending or running:
        for slot,item in list(running.items()):
            proc,handle,task=item;code=proc.poll()
            if code is None:continue
            handle.close();del running[slot]
            if code:failures.append({'task':task,'exit_code':code})
            else:
                try:
                    terminal=verify_artifacts(task_path(task))
                    if terminal['source_commit']!=source:raise RuntimeError('worker source changed')
                    completed.append({'task':task,'source_commit':source,'terminal_sha256':sha256(task_path(task)/'COMPLETE.json')})
                    print(f'{label} {len(completed)}/{len(queue)} complete {task}',flush=True)
                except Exception as exc:failures.append({'task':task,'error':str(exc)})
            progress()
        if failures:
            if running:time.sleep(2);continue
            progress();raise RuntimeError('technical failure: stopped new dispatch; artifacts preserved')
        for slot in range(4):
            if not pending:break
            if slot in running:continue
            free={slot:{'cpu_worker':True}} if cpu else available_gpus([slot])
            if slot not in free:continue
            task=pending.pop(0);phase,pi,k,t,c=task
            log=logs/f'{phase}_pi_{pi:.2f}_k_{k}_trial_{t}_{c}.log'
            handle=log.open('x');env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            if not cpu:env['CUDA_VISIBLE_DEVICES']=free[slot]['uuid']
            else:env['CUDA_VISIBLE_DEVICES']=''
            command=[sys.executable,'-m','scripts.run_cs_saf_followup','worker','--kind',phase,'--pi',str(pi),'--kappa',str(k),'--trial',str(t),'--candidate',c,'--device','cpu' if cpu else 'cuda:0']
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            running[slot]=(proc,handle,task)
            allocations.append({'task':task,'slot':slot,'pid':proc.pid,**free[slot]});progress()
            print(f'{label} start {task} slot={slot} pid={proc.pid}',flush=True)
        if pending or running:time.sleep(2)
    progress()
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['cpu','run','worker'])
    p.add_argument('--kind',choices=['internal','gradient','external','generation']);p.add_argument('--pi',type=float)
    p.add_argument('--kappa',type=int);p.add_argument('--trial',type=int);p.add_argument('--candidate');p.add_argument('--device',default='cpu')
    args=p.parse_args();load_contract();OUTPUT.mkdir(exist_ok=True)
    if args.phase=='cpu':
        r=cpu_gate(OUTPUT/'cpu_v1');print(json.dumps(r),flush=True)
        if r['decision']!='PASS':raise SystemExit(2)
    elif args.phase=='run':
        validate_cpu();gpu,cpu=tasks()
        a=execute(gpu,'gpu');b=execute(cpu,'generation',cpu=True)
        write_json(OUTPUT/'COMPLETE.json',{'status':'COMPLETE','source_commit':frozen_source(),'config_sha256':CONTRACT_SHA,
            'new_internal_fits':170,'reused_internal_fits':30,'CPAR_fits':40,'gradient_jobs':10,'common_internal_generation_evaluations':200,
            'test_accessed':False,'independent_data_confirmation':False,'original_replication_gate':'FAIL',
            'gpu_progress_sha256':sha256(OUTPUT/'gpu_progress.json'),'generation_progress_sha256':sha256(OUTPUT/'generation_progress.json')})
    else:
        validate_cpu();task=(args.kind,args.pi,args.kappa,args.trial,args.candidate)
        gpu,cpu=tasks()
        if task not in gpu+cpu:raise ValueError('unregistered worker task')
        try:
            if args.kind=='internal':internal_job(args.pi,args.kappa,args.trial,args.candidate,torch.device(args.device))
            elif args.kind=='gradient':gradient_job(args.kappa,args.trial,torch.device(args.device))
            else:
                from experiments.cs_saf_followup_external import external_job,internal_generation_job
                if args.kind=='external':external_job(args.pi,args.kappa,args.trial,args.device)
                else:internal_generation_job(args.pi,args.kappa,args.trial,args.candidate)
        except Exception as exc:
            folder=task_path(task)
            if folder.exists() and not (folder/'COMPLETE.json').exists():write_json(folder/'FAILED.json',{'error':type(exc).__name__,'message':str(exc),'source_commit':frozen_source()})
            raise


if __name__=='__main__':main()
