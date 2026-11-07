"""Lease idle GPUs only; pack up to four of our own registered workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from run_stratified import ROOT, OUT, DOCS, CFG, verify, write
from dispatch_argn_label_first_control import gpu_snapshot


def processes():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in result.splitlines():
        fields=[v.strip() for v in line.split(',')]
        if len(fields)!=4:continue
        uuid,pid,name,memory=fields
        if 'nvidia-cuda-mps-server' in name and float(memory)<=32:continue
        rows.append(dict(uuid=uuid,pid=int(pid),memory=float(memory)))
    return rows


def main():
    verify();assert (OUT/'FIT_COMPLETE.json').exists() and not (OUT/'LAUNCH.json').exists()
    pending=[(fs,arm) for arm in CFG['arms'] for fs in CFG['fit_seeds']]
    running={};finished={};launched={};leases=set();idle={}
    (OUT/'logs').mkdir(exist_ok=True)
    while pending or running:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is not None:
                finished[key]=dict(pid=job['pid'],exit_code=code);del running[key]
                print('FINISHED',key,code,flush=True)
        own={v['pid'] for v in running.values()};active=processes()
        for gpu in gpu_snapshot():
            uuid=gpu['uuid']
            if uuid not in CFG['allowed_gpu_uuids']:continue
            if any(p['uuid']==uuid and p['pid'] not in own for p in active):
                idle[uuid]=0;continue
            jobs=sum(j['gpu']==uuid for j in running.values())
            if uuid not in leases:
                free=not gpu['has_compute_process'] and gpu['memory_mib']<500 and gpu['utilization']<=5
                idle[uuid]=idle.get(uuid,0)+1 if free else 0
                if idle[uuid]<2:continue
                leases.add(uuid)
            if not pending or jobs>=CFG['maximum_own_workers_per_gpu'] or gpu['memory_mib']>=12000:continue
            if any(p['uuid']==uuid and p['pid'] not in own for p in processes()):continue
            fs,arm=pending.pop(0);key=f'{arm}_{fs}'
            command=[sys.executable,str(OUT/'local_source/run_stratified.py'),'worker','--seed',str(fs),'--arm',arm]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (OUT/'logs'/f'{key}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            running[key]=dict(pid=proc.pid,gpu=uuid,process=proc);own.add(proc.pid)
            launched[key]=dict(pid=proc.pid,gpu=uuid,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'LAUNCH.json',launched);print('LAUNCHED',key,proc.pid,gpu['index'],flush=True)
        write(OUT/'STATUS.json',dict(pending=pending,running={k:dict(pid=v['pid'],gpu=v['gpu']) for k,v in running.items()},finished=finished))
        if pending or running:time.sleep(5)
    result=dict(success=all(j['exit_code']==0 for j in finished.values()),jobs=finished,
        completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'DISPATCH_COMPLETE.json',result);assert result['success'],result
    print('DISPATCH_COMPLETE',flush=True)


if __name__=='__main__':main()
