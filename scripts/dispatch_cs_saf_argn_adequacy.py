"""One bounded continuation at a time; never stop another user's GPU work."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/benchmark_v2/cs_saf_baseline_adequacy_v1.json').read_text())
OUT=ROOT/'artifacts/cs_saf/baseline_adequacy_v1'
PYTHON=ROOT/'external/cs_saf_external_audit/runtime/bin/python'


def available():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    candidates=[]
    for line in result.strip().splitlines():
        index,free,usage=map(int,line.split(','))
        if free>=C['gpu_min_free_mib'] and usage<=C['gpu_max_utilization_at_admission']:
            candidates.append((usage,-free,index))
    return min(candidates)[2] if candidates else None


def main():
    assert json.loads((OUT/'continuation_cpu_smoke/DONE.json').read_text())['originals_unchanged']
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for k in C['kappas']:
        for seed in C['argn_seeds']:
            target=OUT/f'continuations/kappa_{k}/seed_{seed}'
            if target.exists():
                assert (target/'DONE.json').exists(), 'preserve failed/in-progress run; no automatic retry'
                continue
            gpu=available()
            while gpu is None:
                print('WAIT_GPU',k,seed,flush=True);time.sleep(30);gpu=available()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            print('START',k,seed,'physical_gpu',gpu,flush=True)
            with (logs/f'continue_kappa_{k}_seed_{seed}.log').open('w') as log:
                subprocess.run(['nice','-n','10',str(PYTHON),'scripts/continue_cs_saf_argn_adequacy.py',
                    '--kappa',str(k),'--seed',str(seed),'--device','cuda'],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',k,seed,flush=True)
    print('ALL_REGISTERED_CONTINUATIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
