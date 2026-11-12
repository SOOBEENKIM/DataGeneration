"""Run the four registered jobs only on idle GPU 2/3, then publish local reports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/argn_label_first_control_v1'
CONFIG = ROOT/'configs/argn_label_first_control_v1.json'


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)


def active_workload_gpus(process_text):
    occupied = set()
    for line in process_text.splitlines():
        if not line.strip(): continue
        uuid, pid, name, memory = [s.strip() for s in line.split(',')]
        # This host has a persistent 28 MiB MPS daemon on every device.
        # Total memory/utilization are still checked before every dispatch.
        idle_daemon = (name.rsplit('/', 1)[-1] == 'nvidia-cuda-mps-server'
                       and memory.isdigit() and int(memory) <= 32)
        if not idle_daemon: occupied.add(uuid)
    return occupied


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    occupied = active_workload_gpus(processes.stdout)
    rows=[]
    for line in result.stdout.splitlines():
        index,uuid,memory,utilization=[s.strip() for s in line.split(',')]
        rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),
                         utilization=int(utilization),has_compute_process=uuid in occupied))
    return rows


def main():
    config=json.loads(CONFIG.read_text()); manifest=json.loads((OUT/'MANIFEST.json').read_text())
    assert manifest['config']==config
    for path,sha in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    (OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'DISPATCH_STARTED.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()),handle)
    # Complete both seeds of event-weighted order control first; keep the
    # customer-weighted replication in the same pre-registered queue.
    pending=[(arm,fs) for arm in ['B_event_label_first','B_label_first'] for fs in config['fit_seeds']]
    running={}; finished={}; idle={}; launches={}; snapshot=[]; report_errors=[]
    while pending or running:
        for name,entry in list(running.items()):
            code=entry['process'].poll()
            if code is None: continue
            finished[name]=dict(exit_code=code,pid=entry['process'].pid,gpu_uuid=entry['uuid'])
            del running[name]
            with (OUT/'logs/report.log').open('a') as log:
                report=subprocess.run([sys.executable,str(ROOT/'scripts/report_argn_label_first_control.py'),'--partial'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if report.returncode: report_errors.append(dict(after=name,exit_code=report.returncode))
        if pending:
            try:
                snapshot=gpu_snapshot()
            except (subprocess.SubprocessError,ValueError) as error:
                idle.clear(); snapshot=[]
                print('GPU_QUERY_FAILED',repr(error),flush=True)
            for gpu in snapshot:
                free=(gpu['uuid'] in config['allowed_gpu_uuids'] and gpu['memory_mib']<500
                      and gpu['utilization']<=5 and not gpu['has_compute_process'])
                idle[gpu['uuid']]=idle.get(gpu['uuid'],0)+1 if free else 0
            busy={entry['uuid'] for entry in running.values()}
            for gpu in snapshot:
                if not pending or len(running)>=config['max_concurrent_workers']: break
                if idle.get(gpu['uuid'],0)<2 or gpu['uuid'] in busy: continue
                # Recheck immediately before launch; two periodic idle observations
                # alone must not authorize starting onto a newly occupied GPU.
                now=next(x for x in gpu_snapshot() if x['uuid']==gpu['uuid'])
                if now['has_compute_process'] or now['memory_mib']>=500 or now['utilization']>5:
                    idle[gpu['uuid']]=0; continue
                arm,fs=pending.pop(0); name=f'{arm}_{fs}'
                command=[sys.executable,str(ROOT/'scripts/run_argn_label_first_control.py'),
                         'run','--arm',arm,'--seed',str(fs),'--device','cuda:0']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONUNBUFFERED='1',
                         OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                with (OUT/'logs'/f'{name}.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[name]=dict(process=process,uuid=gpu['uuid']); busy.add(gpu['uuid'])
                launches[name]=dict(pid=process.pid,gpu=gpu,command=command,
                                   started_utc=datetime.now(timezone.utc).isoformat())
                write(OUT/'LAUNCH.json',launches)
                print('LAUNCHED',name,'pid',process.pid,'gpu',gpu['index'],flush=True)
            for arm,fs in pending:
                write(OUT/f'queue_{arm}_{fs}.json',dict(stage='waiting_for_idle_gpu',arm=arm,fit_seed=fs,
                                                       training_started=False))
        write(OUT/'DISPATCH_STATUS.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            pending=pending,running={k:dict(pid=v['process'].pid,gpu_uuid=v['uuid']) for k,v in running.items()},
            finished=finished,last_gpu_state=snapshot,report_errors=report_errors))
        if pending or running: time.sleep(20)
    with (OUT/'logs/report.log').open('a') as log:
        report=subprocess.run([sys.executable,str(ROOT/'scripts/report_argn_label_first_control.py')],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    success=all(v['exit_code']==0 for v in finished.values()) and report.returncode==0
    write(OUT/'DISPATCH_COMPLETE.json',dict(success=success,jobs=finished,report_exit_code=report.returncode,
                                          completed_utc=datetime.now(timezone.utc).isoformat()))
    if not success: raise SystemExit(1)


if __name__=='__main__': main()
