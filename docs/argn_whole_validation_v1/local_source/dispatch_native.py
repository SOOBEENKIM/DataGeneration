"""Persistent native-fit phase; independent sources for later phases."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from native import ROOT,OUT,SEEDS,verify,write,now
sys.path.insert(0,str(ROOT/'artifacts/argn_init_validation_v1/local_source'))
# Importing pipeline only provides its read-only GPU probe; its main is not invoked.
from pipeline import snapshot

def main():
    lock=(OUT/'NATIVE_PIPELINE.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();assert not (OUT/'NATIVE_LAUNCH.json').exists(),'Explicit recovery required'
    pending=list(SEEDS);running={};finished={};handles={};launches=[];idle={}
    write(OUT/'NATIVE_LAUNCH.json',dict(utc=now(),pid=os.getpid(),seeds=SEEDS))
    while pending or running:
        for fs,(proc,gpu) in list(running.items()):
            rc=proc.poll()
            if rc is None:continue
            handles.pop(fs).close();finished[fs]=dict(exit_code=rc,complete=(OUT/f'backbones/{fs}/FIT.json').exists(),utc=now())
            del running[fs];write(OUT/'NATIVE_PROCESS_EXIT.json',finished)
            if rc or not finished[fs]['complete']:
                write(OUT/'NATIVE_FAILED.json',dict(seed=fs,**finished[fs]));raise RuntimeError(f'Failed seed {fs}')
        gpus,apps=snapshot();used={gpu for proc,gpu in running.values()}
        for gpu in gpus:
            free=not gpu['external'] and gpu['uuid'] not in used and gpu['memory']<500 and gpu['utilization']<=5
            idle[gpu['uuid']]=idle.get(gpu['uuid'],0)+1 if free else 0
        for fs in list(pending):
            available=[g for g in gpus if idle.get(g['uuid'],0)>=2 and g['uuid'] not in used]
            if not available:break
            gpu=available[0]['uuid'];fresh,_=snapshot();g=next(x for x in fresh if x['uuid']==gpu)
            if g['external'] or g['memory']>=500 or g['utilization']>5:continue
            env=dict(os.environ,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source',CUDA_VISIBLE_DEVICES=gpu,
              OMP_NUM_THREADS='4',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            h=(OUT/f'logs/native_{fs}.log').open('x');handles[fs]=h
            proc=subprocess.Popen([sys.executable,str(OUT/'local_source/native.py'),'--seed',str(fs)],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
            running[fs]=(proc,gpu);pending.remove(fs);used.add(gpu);idle[gpu]=0
            launches.append(dict(seed=fs,pid=proc.pid,gpu=gpu,utc=now()));write(OUT/'NATIVE_LAUNCHES.json',launches)
            print('LAUNCH',fs,proc.pid,gpu,flush=True)
        write(OUT/'NATIVE_STATUS.json',dict(utc=now(),pending=pending,running={s:dict(pid=p.pid,gpu=g) for s,(p,g) in running.items()},finished=finished,gpus=gpus))
        if pending or running:time.sleep(5)
    write(OUT/'NATIVE_COMPLETE.json',dict(fits=len(finished),utc=now(),success=True))
if __name__=='__main__':main()
