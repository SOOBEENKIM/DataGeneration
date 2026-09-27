"""Run the registered four statistical-control generations on idle GPU 2/3."""
from datetime import datetime,timezone
import os
import subprocess
import sys
import time
from run_argn_phase_gap import ROOT,OUT,DOCS,CFG,verify,write,digest
from dispatch_argn_label_first_control import gpu_snapshot


def main():
    verify();assert not (OUT/'LAUNCH.json').exists()
    (OUT/'logs').mkdir(exist_ok=True);pending=list(CFG['fit_seeds']);running={};finished={};launched={};idle={}
    while pending or running:
        for fs,job in list(running.items()):
            code=job['process'].poll()
            if code is not None:finished[fs]=dict(pid=job['process'].pid,exit_code=code);del running[fs]
        for gpu in gpu_snapshot():
            uuid=gpu['uuid'];occupied=any(v['gpu']==uuid for v in running.values())
            free=uuid in CFG['allowed_gpu_uuids'] and gpu['memory_mib']<500 and gpu['utilization']<=5 and not gpu['has_compute_process'] and not occupied
            idle[uuid]=idle.get(uuid,0)+1 if free else 0
            if not pending or idle[uuid]<2:continue
            now=next(g for g in gpu_snapshot() if g['uuid']==uuid)
            if now['memory_mib']>=500 or now['utilization']>5 or now['has_compute_process']:continue
            fs=pending.pop(0);cmd=[sys.executable,str(ROOT/'scripts/run_argn_phase_gap.py'),'worker','--seed',str(fs),'--device','cuda:0']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (OUT/'logs'/f'{fs}.log').open('x') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[fs]=dict(process=proc,gpu=uuid)
            launched[fs]=dict(pid=proc.pid,gpu=uuid,command=cmd,started_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'LAUNCH.json',launched);write(DOCS/'LAUNCH.json',launched);print('PHASE_LAUNCHED',fs,proc.pid,gpu['index'],flush=True)
        write(OUT/'STATUS.json',dict(pending=pending,running={k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in running.items()},finished=finished))
        if pending or running:time.sleep(5)
    result=dict(success=len(finished)==2 and all(x['exit_code']==0 for x in finished.values()),jobs=finished,
                source_sha256=digest(__file__),completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'DISPATCH_COMPLETE.json',result)
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
