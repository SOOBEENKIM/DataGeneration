"""Resource-only continuation: own workers can share an initially idle GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from run_adaptive import ROOT, OUT, DOCS, CFG, verify, write, digest
from dispatch_argn_label_first_control import gpu_snapshot


def processes():
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in text.splitlines():
        fields=[v.strip() for v in line.split(',')]
        if len(fields)!=4:continue
        uuid,pid,name,memory=fields
        if 'nvidia-cuda-mps-server' in name and float(memory)<=32:continue
        rows.append(dict(uuid=uuid,pid=int(pid),memory=float(memory)))
    return rows


def main():
    verify()
    amend=json.loads((OUT/'SCHEDULING_AMENDMENT.json').read_text())
    assert digest(__file__)==amend['source_sha256']
    dispatcher=amend['previous_dispatcher_pid']
    cmd=Path(f'/proc/{dispatcher}/cmdline').read_bytes().replace(b'\x00',b' ')
    assert b'run_adaptive.py dispatch' in cmd,cmd
    os.kill(dispatcher,signal.SIGTERM)  # only this experiment's scheduler; workers stay alive
    for _ in range(50):
        if not Path(f'/proc/{dispatcher}').exists():break
        time.sleep(.1)
    assert not Path(f'/proc/{dispatcher}').exists()
    status=json.loads((OUT/'STATUS.json').read_text())
    launched=json.loads((OUT/'LAUNCH.json').read_text())
    write(OUT/'SERIAL_DISPATCH_PRESERVED.json',dict(status=status,launched=launched))
    pending=[tuple(x) for x in status['pending']]
    running={k:{**v,'process':None} for k,v in status['running'].items()}
    finished=status['finished']; leases={v['gpu'] for v in running.values()}
    idle={};limit=amend['maximum_own_workers_per_gpu']
    print('ADOPTED_WORKERS',running,flush=True)
    while pending or running:
        for key,job in list(running.items()):
            process=job['process'];pid=job['pid']
            if process is not None:
                code=process.poll()
            else:
                fs=int(key.rsplit('_',1)[1]);arm=key.rsplit('_',1)[0]
                completed=OUT/f'worker_{fs}/{arm}/COMPLETE.json'
                if Path(f'/proc/{pid}').exists():continue
                code=0 if completed.exists() else 1
            if code is not None:
                finished[key]=dict(pid=pid,exit_code=code);del running[key]
                print('FINISHED',key,code,flush=True)
        own={v['pid'] for v in running.values()}
        active=processes()
        for gpu in gpu_snapshot():
            uuid=gpu['uuid']
            if uuid not in CFG['allowed_gpu_uuids']:continue
            external=[p for p in active if p['uuid']==uuid and p['pid'] not in own]
            if external:idle[uuid]=0;continue
            jobs=sum(j['gpu']==uuid for j in running.values())
            if uuid not in leases:
                free=not gpu['has_compute_process'] and gpu['memory_mib']<500 and gpu['utilization']<=5
                idle[uuid]=idle.get(uuid,0)+1 if free else 0
                if idle[uuid]<2:continue
                leases.add(uuid)
            if not pending or jobs>=limit or gpu['memory_mib']>=12000:continue
            # Recheck foreign ownership immediately before each additional launch.
            if any(p['uuid']==uuid and p['pid'] not in own for p in processes()):continue
            fs,arm=pending.pop(0);key=f'{arm}_{fs}'
            command=[sys.executable,str(OUT/'local_source/run_adaptive.py'),'worker','--seed',str(fs),'--arm',arm]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (OUT/'logs'/f'{key}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            running[key]=dict(pid=proc.pid,gpu=uuid,process=proc);own.add(proc.pid)
            launched[key]=dict(pid=proc.pid,gpu=uuid,command=command,started_utc=datetime.now(timezone.utc).isoformat(),scheduler='packed_own_jobs_only')
            write(OUT/'LAUNCH.json',launched);write(DOCS/'LAUNCH.json',launched)
            print('LAUNCHED',key,proc.pid,gpu['index'],flush=True)
        write(OUT/'STATUS.json',dict(pending=pending,running={k:dict(pid=v['pid'],gpu=v['gpu']) for k,v in running.items()},finished=finished))
        if pending or running:time.sleep(5)
    result=dict(success=all(j['exit_code']==0 for j in finished.values()),jobs=finished,
                scheduler='packed_own_jobs_only',completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'DISPATCH_COMPLETE.json',result)
    assert result['success'],result
    print('PACKED_DISPATCH_COMPLETE',flush=True)


if __name__=='__main__':main()
