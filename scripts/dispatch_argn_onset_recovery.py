"""Reuse the idle-GPU dispatcher with the registered onset confirmation recovery study entry point."""
from datetime import datetime, timezone
import os
import subprocess
import sys
import time

from recover_argn_onset_confirmation import ROOT, OUT, DOCS, CFG, verify, write, digest
from dispatch_argn_label_first_control import gpu_snapshot


def main():
    verify();(OUT/'logs').mkdir(exist_ok=True)
    assert not (OUT/'LAUNCH.json').exists()
    pending=list(CFG['fit_seeds']);running={};done={};idle={};launches={}
    while pending or running:
        for fs,job in list(running.items()):
            code=job['process'].poll()
            if code is not None:
                done[fs]=dict(pid=job['process'].pid,exit_code=code);del running[fs]
        for gpu in gpu_snapshot():
            uuid=gpu['uuid']
            free=(uuid in CFG['allowed_gpu_uuids'] and gpu['memory_mib']<500
                  and gpu['utilization']<=5 and not gpu['has_compute_process'])
            idle[uuid]=idle.get(uuid,0)+1 if free else 0
            if not pending or idle[uuid]<2:continue
            now=next(g for g in gpu_snapshot() if g['uuid']==uuid)
            if now['memory_mib']>=500 or now['utilization']>5 or now['has_compute_process']:continue
            fs=pending.pop(0)
            command=[sys.executable,str(ROOT/'scripts/recover_argn_onset_confirmation.py'),'worker','--seed',str(fs),'--device','cuda:0']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (OUT/'logs'/f'{fs}.log').open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[fs]=dict(process=process,gpu=uuid)
            launches[fs]=dict(pid=process.pid,gpu=gpu,command=command,
                started_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'LAUNCH.json',launches);write(DOCS/'LAUNCH.json',launches)
            print('ONSET_RECOVERY_LAUNCHED',fs,process.pid,gpu['index'],flush=True)
        write(OUT/'STATUS.json',dict(pending=pending,running={k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in running.items()},finished=done))
        if pending or running:time.sleep(10)
    result=dict(success=len(done)==2 and all(x['exit_code']==0 for x in done.values()),jobs=done,
                completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'DISPATCH_COMPLETE.json',result)
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
