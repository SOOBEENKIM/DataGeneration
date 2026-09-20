"""Run same-source gates and the registered 40 frozen-checkpoint calibrations."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import torch

from experiments.cs_saf_calibration_u import (ROOT, OUTPUT, CONFIG_SHA, contract,
    fit_job, frozen_source, smoke, verify, REUSABLE_SOURCES)
from scripts.materialize_cs_saf_prevalence import write_json


def available_gpus():
    query = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    busy = set()
    for line in apps.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)>=3 and fields[2].split('/')[-1] != 'nvidia-cuda-mps-server':
            busy.add(fields[0])
    result=[]
    for line in query.splitlines():
        index,uuid,memory,util=[x.strip() for x in line.split(',')]
        if uuid not in busy and int(memory)<1024 and int(util)<=5:
            result.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),utilization=int(util),
                               compute_snapshot=apps,observed_at=time.time()))
    return result


def check_gate(name,source):
    verify(OUTPUT/name)
    gate=json.loads((OUTPUT/name/'gate.json').read_text())
    if gate['decision']!='PASS' or gate['source_commit']!=source or gate['config_sha256']!=CONFIG_SHA:
        raise RuntimeError('same-source passing gate required: '+name)


def run():
    c=contract();source=frozen_source()
    check_gate('cpu_gate',source);check_gate('gpu_gate',source)
    tasks=[(pi,k,t,a) for pi in c['prevalences'] for t in c['trials'] for k in c['kappas'] for a in c['parents']]
    pending=[];completed=[];running={};failures=[];allocations=[]
    for task in tasks:
        pi,k,t,a=task;folder=OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/{a}cal'
        if (folder/'COMPLETE.json').exists():
            item=verify(folder)
            if item['source_commit'] not in REUSABLE_SOURCES|{source}:raise ValueError('resume source differs')
            completed.append(list(task))
        elif folder.exists():
            raise RuntimeError('incomplete evidence requires a documented new attempt: '+str(folder))
        else:pending.append(task)
    logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
    def progress():
        write_json(OUTPUT/'progress.json',dict(source_commit=source,config_sha256=CONFIG_SHA,
            completed=completed,pending=pending,running=[dict(task=x[2],pid=x[0].pid,gpu=u) for u,x in running.items()],
            failures=failures,allocations=allocations,total=len(tasks)))
    while pending or running:
        for uuid,(proc,handle,task) in list(running.items()):
            status=proc.poll()
            if status is None:continue
            handle.close();del running[uuid]
            if status:failures.append(dict(task=task,exit_code=status))
            else:
                pi,k,t,a=task;item=verify(OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/{a}cal')
                if item['source_commit']!=source:raise RuntimeError('worker source mismatch')
                completed.append(list(task));print(f'completed {len(completed)}/{len(tasks)} {task}',flush=True)
            progress()
        if failures:
            if running:time.sleep(2);continue
            progress();raise RuntimeError('technical failure; no further tasks dispatched')
        if pending and len(running)<c['max_workers']:
            for gpu in available_gpus():
                uuid=gpu['uuid']
                if uuid in running:continue
                if not pending or len(running)>=c['max_workers']:break
                task=pending.pop(0);pi,k,t,a=task
                handle=(logs/f'pi_{pi:.2f}_k{k}_t{t}_{a}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command=[sys.executable,'-m','scripts.run_cs_saf_calibration_u','worker',
                    '--pi',str(pi),'--kappa',str(k),'--trial',str(t),'--candidate',a,'--device','cuda:0']
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[uuid]=(proc,handle,task)
                allocations.append(dict(task=task,pid=proc.pid,**gpu));progress()
                print(f'start {task} GPU {gpu["index"]} pid={proc.pid}',flush=True)
        progress()
        if pending or running:time.sleep(3)
    write_json(OUTPUT/'GRID_COMPLETE.json',dict(source_commit=source,config_sha256=CONFIG_SHA,
        completed=completed,new_calibration_fits=len(completed),base_model_fits=0,test_accessed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['cpu','gpu','run','worker'])
    p.add_argument('--device',default='cpu');p.add_argument('--pi',type=float)
    p.add_argument('--kappa',type=int);p.add_argument('--trial',type=int)
    p.add_argument('--candidate',choices=['U'])
    a=p.parse_args();contract();OUTPUT.mkdir(parents=True,exist_ok=True)
    if a.device.startswith('cuda'):
        torch.cuda.set_device(a.device)
        torch.cuda.set_per_process_memory_fraction(.20,a.device)
    if a.phase=='cpu':
        if a.device!='cpu':raise ValueError('CPU gate must use CPU')
        smoke(torch.device(a.device),'cpu_gate')
    elif a.phase=='gpu':
        if not a.device.startswith('cuda'):raise ValueError('GPU gate must use GPU')
        check_gate('cpu_gate',frozen_source());smoke(torch.device(a.device),'gpu_gate')
    elif a.phase=='worker':
        source=frozen_source();check_gate('cpu_gate',source);check_gate('gpu_gate',source)
        c=contract()
        if a.pi not in c['prevalences'] or a.kappa not in c['kappas'] or a.trial not in c['trials']:
            raise ValueError('unregistered cell')
        fit_job(a.pi,a.kappa,a.trial,a.candidate,torch.device(a.device))
    else:run()


if __name__=='__main__':main()
