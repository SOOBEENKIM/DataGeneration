"""Durable idle-GPU-only dispatcher; independent worker failures do not kill siblings."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/comparison_completion_v1'
def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def idle():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True);busy=set()
    for line in apps.strip().splitlines():
        pid,gpu=[x.strip() for x in line.split(',')]
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        except (FileNotFoundError,PermissionError):cmd=b'unknown'
        if b'nvidia-cuda-mps-server' not in cmd:busy.add(gpu)
    return [x[1].strip() for x in [r.split(',') for r in rows.strip().splitlines()] if int(x[2])<500 and int(x[3])<=5 and x[1].strip() not in busy]
def main():
    lock=(OUT/'CPAR_LOCK').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[20261022,20261023];running={};done=[]
    assert not (OUT/'CPAR_STATUS_v2.json').exists(),'Do not duplicate a previous dispatcher'
    while pending or running:
        for seed,(p,gpu,h) in list(running.items()):
            rc=p.poll()
            if rc is not None:
                h.close();rec=dict(seed=seed,pid=p.pid,gpu=gpu,exit_code=rc,complete=(OUT/f'cpar/{seed}/COMPLETE.json').exists())
                write(OUT/f'cpar/{seed}/PROCESS_EXIT_v2.json',rec);done.append(rec);del running[seed]
        free=[g for g in idle() if g not in [v[1] for v in running.values()]]
        for gpu in free[:len(pending)]:
            if gpu not in idle():continue
            seed=pending.pop(0);h=(OUT/f'logs/cpar_{seed}_v2.log').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=f'{ROOT}:{ROOT}/scripts',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            p=subprocess.Popen(['/home/finx_sbk/.conda/envs/cofseq/bin/python',str(OUT/'local_source/cpar_refit.py'),str(seed)],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
            running[seed]=(p,gpu,h);write(OUT/f'cpar/{seed}/LAUNCH_v2.json',dict(seed=seed,pid=p.pid,gpu=gpu))
        write(OUT/'CPAR_STATUS_v2.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pending=pending,running=[dict(seed=k,pid=v[0].pid,gpu=v[1]) for k,v in running.items()],done=done))
        if pending or running:time.sleep(5)
    write(OUT/'CPAR_PROCESS_COMPLETE_v2.json',dict(done=done,all_success=all(v['exit_code']==0 and v['complete'] for v in done)))
if __name__=='__main__':main()
