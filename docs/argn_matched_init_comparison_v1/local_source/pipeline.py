"""Run the registered matched comparison without occupying another user's GPU."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from matched_study import ROOT,OUT,DOCS,SEEDS,verify,write,now
from dispatch_stages import snapshot

def main():
    lock=(OUT/'PIPELINE.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();assert not (OUT/'LAUNCH.json').exists(),'Explicit recovery required.'
    pending=[dict(key='existing_costs',kind='cpu',deps=[],done=str(DOCS/'existing_costs/COMPLETE.json'),
                  cmd=[sys.executable,str(OUT/'local_source/evaluate_matched.py'),'--existing-only'])]
    for fs in SEEDS:
        done=OUT/f'fits/statistics_{fs}/HEAD_PREFLIGHT.json'
        pending.append(dict(key=f'fit_{fs}',kind='cpu',deps=[],done=str(done),
            cmd=[sys.executable,str(OUT/'local_source/matched_study.py'),'fit','--seed',str(fs)]))
        pending.append(dict(key=f'generate_{fs}',kind='gpu',deps=[str(done)],
            done=str(OUT/f'runs/init_only_{fs}/COMPLETE.json'),
            cmd=[sys.executable,str(OUT/'local_source/matched_study.py'),'generate','--seed',str(fs)]))
    write(OUT/'LAUNCH.json',dict(pid=os.getpid(),utc=now(),jobs=pending))
    active={};exits={};launches=[];idle={}
    while pending or active:
        failed=False
        for key,j in list(active.items()):
            code=j['p'].poll()
            if code is None:continue
            j['handle'].close();del active[key]
            exits[key]=dict(exit_code=code,completed=Path(j['spec']['done']).exists(),utc=now())
            write(OUT/'PROCESS_EXIT.json',exits)
            if code!=0 or not exits[key]['completed']:
                failed=True;write(OUT/f'FAILED_{key}.json',dict(job=key,**exits[key]))
        # Let running siblings finish after failure. No new dependent job is launched.
        failed=failed or any(OUT.glob('FAILED_*.json'))
        gpus,apps=snapshot();used={j['gpu'] for j in active.values() if j['gpu']}
        for g in gpus:
            free=not g['external'] and g['uuid'] not in used and g['memory']<500 and g['utilization']<=5
            idle[g['uuid']]=idle.get(g['uuid'],0)+1 if free else 0
        if not failed:
            for spec in list(pending):
                if not all(Path(p).exists() for p in spec['deps']):continue
                gpu=None
                if spec['kind']=='cpu':
                    if sum(j['spec']['kind']=='cpu' for j in active.values())>=2:continue
                else:
                    free=[g for g in gpus if idle.get(g['uuid'],0)>=2 and g['uuid'] not in used]
                    if not free:continue
                    fresh,_=snapshot();g=next(x for x in fresh if x['uuid']==free[0]['uuid'])
                    if g['external'] or g['memory']>=500 or g['utilization']>5:continue
                    gpu=g['uuid'];used.add(gpu);idle[gpu]=0
                env=dict(os.environ,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source',
                    OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=gpu or '')
                h=(OUT/f'logs/{spec["key"]}.log').open('x')
                p=subprocess.Popen(spec['cmd'],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                active[spec['key']]=dict(p=p,handle=h,spec=spec,gpu=gpu);pending.remove(spec)
                launches.append(dict(job=spec['key'],pid=p.pid,gpu=gpu,command=spec['cmd'],utc=now()))
                write(OUT/'LAUNCHES.json',launches);print('LAUNCH',spec['key'],p.pid,gpu,flush=True)
        write(OUT/'STATUS.json',dict(utc=now(),pending=[s['key'] for s in pending],
            running={k:dict(pid=j['p'].pid,gpu=j['gpu'],kind=j['spec']['kind']) for k,j in active.items()},
            finished=exits,gpus=gpus,failed=failed))
        if failed and not active:raise RuntimeError('Failure recorded; explicit recovery required.')
        if pending or active:time.sleep(5)
    with (OUT/'logs/final_evaluation.log').open('x') as h:
        subprocess.run([sys.executable,str(OUT/'local_source/evaluate_matched.py')],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,check=True)
    write(OUT/'PIPELINE_COMPLETE.json',dict(jobs=len(exits),utc=now(),success=True))

if __name__=='__main__':main()
