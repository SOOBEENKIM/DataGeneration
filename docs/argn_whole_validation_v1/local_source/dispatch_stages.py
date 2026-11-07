"""Persistent dependency dispatcher. Launch only on GPUs without external workers."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from stages import ROOT,OUT,DOCS,SEEDS,ARMS,verify,write,now

SCRIPT=OUT/'local_source/stages.py'

def snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    busy=set();processes=[]
    for row in apps.strip().splitlines():
        pid,uuid=[v.strip() for v in row.split(',')]
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError):cmd='unknown'
        processes.append(dict(pid=pid,uuid=uuid,command=cmd))
        if 'nvidia-cuda-mps-server' not in cmd:busy.add(uuid)
    return [dict(index=int(v[0]),uuid=v[1].strip(),memory=int(v[2]),utilization=int(v[3]),external=v[1].strip() in busy)
        for v in [row.split(',') for row in rows.strip().splitlines()]],processes

def jobs():
    out=[]
    def add(key,mode,kind,deps,done,*args):
        out.append(dict(key=key,mode=mode,kind=kind,deps=[str(p) for p in deps],done=str(done),args=list(args)))
    for fs in SEEDS:
        foundation=OUT/f'foundation/{fs}/FIT_COMPLETE.json'
        timehead=OUT/f'time_fits/{fs}/COMPLETE.json'
        add(f'foundation_{fs}','foundation','gpu',[OUT/f'backbones/{fs}/FIT.json'],foundation,'--seed',str(fs))
        add(f'history_{fs}','history','gpu',[foundation],timehead,'--seed',str(fs))
        for arm in ARMS:
            if arm!='history_only':
                cal=OUT/f'calibration/{fs}/{arm}/COMPLETE.json'
                add(f'cal_{arm}_{fs}','calibrate','cpu',[timehead],cal,'--seed',str(fs),'--arm',arm)
            deps=[timehead,OUT/'EVALUATION_READY.json'] if arm=='history_only' else [cal,OUT/f'runs/history_only_{fs}/COMPLETE.json']
            add(f'gen_{arm}_{fs}','generate','gpu',deps,OUT/f'runs/{arm}_{fs}/COMPLETE.json','--seed',str(fs),'--arm',arm)
    return out

def main():
    lock=(OUT/'STAGES_PIPELINE.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();pending=jobs();running={};finished={};handles={};launches=[];idle={}
    assert not (OUT/'STAGES_LAUNCH.json').exists(),'Explicit recovery is required; no blind restart.'
    write(OUT/'STAGES_LAUNCH.json',dict(pid=os.getpid(),started_utc=now(),jobs=pending))
    while pending or running:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            handles.pop(key).close();finished[key]=dict(exit_code=code,completed=Path(job['spec']['done']).exists(),finished_utc=now())
            del running[key]
            write(OUT/'STAGES_PROCESS_EXIT.json',finished)
            if code!=0 or not finished[key]['completed']:
                write(OUT/'STAGES_FAILED.json',dict(job=key,**finished[key],others_left_running=list(running)))
                # Existing sibling fits are allowed to finish; never kill them.
                raise RuntimeError(f'Job failed: {key}')
        gpus,apps=snapshot();used={j['gpu'] for j in running.values() if j['gpu']}
        for gpu in gpus:
            free=not gpu['external'] and gpu['uuid'] not in used and gpu['memory']<500 and gpu['utilization']<=5
            idle[gpu['uuid']]=idle.get(gpu['uuid'],0)+1 if free else 0
        for spec in list(pending):
            if not all(Path(p).exists() for p in spec['deps']):continue
            gpu=None
            if spec['kind']=='cpu':
                if sum(j['spec']['kind']=='cpu' for j in running.values())>=2:continue
            else:
                candidates=[g for g in gpus if idle.get(g['uuid'],0)>=2 and g['uuid'] not in used]
                if not candidates:continue
                candidate=candidates[0]
                fresh,_=snapshot();candidate=next(g for g in fresh if g['uuid']==candidate['uuid'])
                if candidate['external'] or candidate['memory']>=500 or candidate['utilization']>5:continue
                gpu=candidate['uuid'];used.add(gpu);idle[gpu]=0
            key=spec['key'];env=dict(os.environ,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source',
                OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=gpu or '')
            cmd=[sys.executable,str(SCRIPT),spec['mode'],*spec['args']]
            handle=(OUT/f'logs/{key}.log').open('x');handles[key]=handle
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            running[key]=dict(process=proc,spec=spec,gpu=gpu);pending.remove(spec)
            launches.append(dict(job=key,pid=proc.pid,gpu=gpu,command=cmd,started_utc=now()));write(OUT/'STAGES_LAUNCHES.json',launches)
            print('LAUNCH',key,proc.pid,gpu,flush=True)
        write(OUT/'STAGES_STATUS.json',dict(updated_utc=now(),pending=[s['key'] for s in pending],
            running={k:dict(pid=j['process'].pid,gpu=j['gpu'],kind=j['spec']['kind']) for k,j in running.items()},
            finished=finished,gpus=gpus))
        if pending or running:time.sleep(5)
    write(OUT/'STAGES_COMPLETE.json',dict(jobs=len(finished),success=True,finished_utc=now()))
    subprocess.run([sys.executable,str(OUT/'local_source/evaluate_whole.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
