"""Two CPU workers for registered immutable-parent controls, including queued fits."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/benchmark_v2/cs_saf_baseline_adequacy_v1.json').read_text())
OUT=ROOT/'artifacts/cs_saf/baseline_adequacy_v1'
ARGN_PY=ROOT/'external/cs_saf_external_audit/runtime/bin/python'
U_PY='/home/finx_sbk/.conda/envs/cofseq/bin/python3'


def execute(job):
    family,k,index=job
    if family=='U':
        target=OUT/f'controls/U/kappa_{k}/trial_{index}'
        command=[U_PY,'scripts/run_cs_saf_u_repeat_controls.py','--kappa',str(k),'--trial',str(index)]
    else:
        target=OUT/f'controls/argn_{family}/kappa_{k}/seed_{index}'
        command=[str(ARGN_PY),'scripts/run_cs_saf_argn_repeat_controls.py','--kappa',str(k),'--seed',str(index),'--parent',family]
    if target.exists():
        assert (target/'DONE.json').exists(), f'preserve unfinished run: {target}'
        return
    print('CONTROL_START',*job,flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (OUT/f'logs/control_{family}_{k}_{index}.log').open('w') as log:
        subprocess.run(['nice','-n','10']+command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('CONTROL_DONE',*job,flush=True)


def main():
    for name in ['argn_sampling_cpu_gate.json','u_sampling_cpu_gate.json']:
        gate=json.loads((ROOT/'docs/cs_saf/baseline_adequacy_v1'/name).read_text())
        assert gate['identity_sample_exact'] and gate['weights_unchanged']
    (OUT/'logs').mkdir(exist_ok=True)
    pending=[(family,k,i) for family in ('original','U','continued') for k in C['kappas']
             for i in (C['u_trials'] if family=='U' else C['argn_seeds'])]
    active={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while pending or active:
            for future in list(active):
                if future.done():
                    future.result();del active[future]
            for job in list(pending):
                if len(active)>=2:break
                family,k,index=job
                if family=='continued' and not (OUT/f'continuations/kappa_{k}/seed_{index}/DONE.json').exists():
                    if (OUT/f'continuations/kappa_{k}/seed_{index}/FAILED.json').exists():
                        raise RuntimeError('registered continuation failed; no automatic replacement')
                    continue
                active[pool.submit(execute,job)]=job;pending.remove(job)
            if pending or active:time.sleep(5)
    print('ALL_TWELVE_PARENT_CONTROLS_COMPLETE',flush=True)


if __name__=='__main__':main()
