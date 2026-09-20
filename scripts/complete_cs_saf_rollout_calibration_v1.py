"""After the frozen fit manifest, run at most two single-thread CPU evaluators."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cs_saf_rollout_calibration_v1 as run


def execute(args):
    label='_'.join(args);logs=run.OUT/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (logs/f'{label}.log').open('x') as output:
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_cs_saf_rollout_calibration_v1.py'),*args],
            cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    print('EVALUATION_JOB_DONE',label,flush=True)


if __name__=='__main__':
    while not (run.OUT/'ALL_FITS_DONE.json').exists():
        failures=list(run.OUT.glob('runs/**/*FAILED.json'))
        if failures:raise RuntimeError('preserved fitting failure: '+str(failures))
        time.sleep(20)
    run.all_fitted()
    jobs=[['model','--kappa',str(k),'--trial',str(t),'--model',name]
        for k in run.config()['kappas'] for t in run.config()['trials'] for name in run.config()['models']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute,jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute,[['oracle','--kappa',str(k)] for k in run.config()['kappas']]))
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_cs_saf_rollout_calibration_v1.py')],cwd=ROOT,check=True)
    print('ALL_EVALUATION_AND_VERIFICATION_COMPLETE',flush=True)
