"""Bounded queues: one admitted GPU training job and two one-thread CPU evaluators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cs_saf_structure_v1 import config,OUT,DOC,folder_for

C=config();ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
JOBS=[(k,t,m) for k in C['kappas'] for t in range(3) for m in C['models']]


def available():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    candidates=[]
    for line in raw.strip().splitlines():
        i,free,use=map(int,line.split(','))
        if free>=C['gpu_min_free_mib'] and use<=C['gpu_max_utilization_at_admission']:
            candidates.append((use,-free,i,free))
    return min(candidates) if candidates else None


def execute(action,job,gpu=None):
    k,t,m=job;log=OUT/'logs'/f'{action}_{k}_{t}_{m}.log'
    args=[sys.executable,'scripts/run_cs_saf_structure_v1.py',action,'--kappa',str(k),'--trial',str(t),'--model',m]
    env=dict(ENV)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    print(action.upper()+'_START',*job,'GPU',gpu,flush=True)
    with log.open('w') as f:subprocess.run(['nice','-n','10']+args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(action.upper()+'_DONE',*job,flush=True)


def gpu_queue():
    assert json.loads((DOC/'gpu_gate.json').read_text())['status']=='PASS'
    for job in JOBS:
        folder=folder_for(*job)
        if folder.exists():
            assert (folder/'TRAIN_DONE.json').exists(), 'preserve unfinished training; no automatic retry'
            continue
        admitted=available()
        while admitted is None:
            print('WAIT_GPU',*job,flush=True);time.sleep(30);admitted=available()
        use,_,gpu,free=admitted
        with (OUT/'admissions.jsonl').open('a') as f:
            f.write(json.dumps(dict(job=job,physical_gpu=gpu,free_mib=free,utilization=use,timestamp=time.time()))+'\n')
        execute('train',job,gpu)
    print('ALL_18_TRAINING_FITS_COMPLETE',flush=True)


def cpu_queue():
    pending=list(JOBS);active={}
    with ThreadPoolExecutor(max_workers=C['max_cpu_evaluation_workers']) as pool:
        while pending or active:
            for future in list(active):
                if future.done():future.result();del active[future]
            for job in list(pending):
                folder=folder_for(*job)
                if (folder/'FAILED.json').exists() or (folder/'evaluation/FAILED.json').exists():
                    raise RuntimeError('technical failure preserved: '+str(folder))
                if (folder/'EVAL_DONE.json').exists():pending.remove(job);continue
                if len(active)<C['max_cpu_evaluation_workers'] and (folder/'TRAIN_DONE.json').exists():
                    assert not (folder/'evaluation').exists(), 'preserve unfinished evaluation'
                    active[pool.submit(execute,'evaluate',job)]=job;pending.remove(job)
            if pending or active:time.sleep(5)
    print('ALL_18_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('queue',choices=['gpu','cpu']);a=p.parse_args()
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    gpu_queue() if a.queue=='gpu' else cpu_queue()
