"""Run only the frozen job list, with one worker per idle workstation GPU."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cs_saf/external_controls_v1'
PY='/home/finx_sbk/.conda/envs/cofseq/bin/python3'
JOBS={0:[('berka',m) for m in ('U_amount','G_amount','U_both','D_both')],
      1:[('sparkov',m) for m in ('U_amount','G_amount','U_both','D_both')],
      2:[('berka',m) for m in ('U_action','G_action','G_both')],
      3:[('sparkov',m) for m in ('U_action','G_action','G_both')]}


def worker(gpu,jobs):
    outcomes=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for name,model in jobs:
        while True:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'],text=True)
            cards={int(x.split(',')[0]):tuple(map(int,x.split(',')[1:])) for x in text.splitlines()}
            used,util=cards[gpu]
            if used<512 and util<5:break
            time.sleep(15)
        logfile=OUT/'logs'/f'{name}_{model}.log'
        with logfile.open('x') as log:
            proc=subprocess.Popen([PY,str(ROOT/'scripts/run_cs_saf_external_controls.py'),model,name],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (OUT/'logs'/f'{name}_{model}_admission.json').write_text(json.dumps(dict(
                gpu=gpu,memory_used_before=used,utilization_before=util,pid=proc.pid,started_unix=time.time()))+'\n')
            print('START',name,model,'GPU',gpu,'PID',proc.pid,flush=True)
            code=proc.wait()
            outcomes.append(dict(dataset=name,model=model,exit_code=code,gpu=gpu))
            print('FINISHED',name,model,'exit',code,flush=True)
            # Other registered jobs are independent; never retry the failed job.
    return outcomes


def main():
    cfg=json.loads((ROOT/'configs/cs_saf_external_controls_v1.json').read_text())
    jobs=[(d,m) for rows in JOBS.values() for d,m in rows]
    assert len(jobs)==14 and set(jobs)=={(d,m) for d in cfg['datasets'] for m in cfg['new_models']}
    assert (OUT/'preflight.json').exists() and (OUT/'gpu_preflight.json').exists()
    (OUT/'logs').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(worker,gpu,rows) for gpu,rows in JOBS.items()]
        outcomes=[row for f in futures for row in f.result()]
    (OUT/'dispatch_done.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    assert all(r['exit_code']==0 for r in outcomes), 'registered failures recorded; no automatic retry'


if __name__=='__main__':main()
