"""Two seed workers, each running all amount controls, on verified idle GPUs."""
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

from run_argn_amount_learning import ROOT, OUT, DOCS, CFG, verify, write
from dispatch_argn_label_first_control import gpu_snapshot


def main():
    verify();(OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'DISPATCH_STARTED.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()),f)
    pending=list(CFG['fit_seeds']);running={};finished={};idle={};launches={}
    while pending or running:
        for fs,job in list(running.items()):
            code=job['process'].poll()
            if code is not None:
                finished[fs]=dict(exit_code=code,pid=job['process'].pid,gpu_uuid=job['uuid'])
                del running[fs]
        snapshot=gpu_snapshot()
        for g in snapshot:
            free=(g['uuid'] in CFG['allowed_gpu_uuids'] and g['memory_mib']<500
                  and g['utilization']<=5 and not g['has_compute_process'])
            idle[g['uuid']]=idle.get(g['uuid'],0)+1 if free else 0
        busy={v['uuid'] for v in running.values()}
        for g in snapshot:
            if not pending or len(running)>=2:break
            if idle.get(g['uuid'],0)<2 or g['uuid'] in busy:continue
            now=next(x for x in gpu_snapshot() if x['uuid']==g['uuid'])
            if now['has_compute_process'] or now['memory_mib']>=500 or now['utilization']>5:continue
            fs=pending.pop(0)
            command=[sys.executable,str(ROOT/'scripts/run_argn_amount_learning.py'),
                     'worker','--seed',str(fs),'--device','cuda:0']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],PYTHONUNBUFFERED='1',
                     OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (OUT/'logs'/f'{fs}.log').open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[fs]=dict(process=process,uuid=g['uuid']);busy.add(g['uuid'])
            launches[fs]=dict(pid=process.pid,gpu=g,command=command,
                started_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'LAUNCH.json',launches);write(DOCS/'LAUNCH.json',launches)
            print('LAUNCHED_AMOUNT',fs,'gpu',g['index'],'pid',process.pid,flush=True)
        write(OUT/'DISPATCH_STATUS.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            pending=pending,running={k:dict(pid=v['process'].pid,gpu_uuid=v['uuid']) for k,v in running.items()},
            finished=finished,last_gpu_state=snapshot))
        if pending or running:time.sleep(15)
    result=dict(success=len(finished)==2 and all(v['exit_code']==0 for v in finished.values()),
        jobs=finished,completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'DISPATCH_COMPLETE.json',result);write(DOCS/'DISPATCH_COMPLETE.json',result)
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
