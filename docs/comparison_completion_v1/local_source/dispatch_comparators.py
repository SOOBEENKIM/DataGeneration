"""Run full MeLD and CPAR count controls only on genuinely idle GPUs."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from dispatch_cpar import idle,write,ROOT,OUT
ARGN=ROOT.parent/'research-reporting/argn-state-first-2026-09-27/runtime/bin/python'
CPAR=Path('/home/finx_sbk/.conda/envs/cofseq/bin/python')
def main():
    lock=(OUT/'COMPARATOR_LOCK').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'COMPARATOR_STATUS.json').exists(),'Inspect existing state before recovery'
    assert (OUT/'MELD_MANIFEST.json').exists() and (OUT/'meld/LOSS_REDUCTION_PARITY.json').exists()
    jobs=[dict(name=f'meld_{s}',seed=s,script='meld_train.py',python=str(ARGN),dependency=OUT/'meld/PREPARED.json',complete=OUT/f'meld/fit_{s}/COMPLETE.json') for s in [20261021,20261022]]
    for s in [20261021,20261022,20261023]:
        source=ROOT/'artifacts/cpar_adequacy_v1' if s==20261021 else OUT/f'cpar/{s}'
        jobs.append(dict(name=f'cpar_length_{s}',seed=s,script='cpar_length.py',python=str(CPAR),dependency=source/'COMPLETE.json',complete=OUT/f'cpar_length/{s}/COMPLETE.json'))
    pending=jobs[:];running={};done=[]
    while pending or running:
        for name,(p,j,gpu,h) in list(running.items()):
            if p.poll() is not None:
                h.close();record=dict(name=name,pid=p.pid,gpu=gpu,exit_code=p.returncode,complete=j['complete'].exists())
                write(OUT/f'logs/{name}_EXIT.json',record);done.append(record);del running[name]
        ready=[j for j in pending if j['dependency'].exists()]
        free=[g for g in idle() if g not in [v[2] for v in running.values()]]
        for gpu,j in zip(free,ready):
            if gpu not in idle():continue
            h=(OUT/f'logs/{j["name"]}.log').open('x');pending.remove(j)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCHDYNAMO_DISABLE='1')
            p=subprocess.Popen([j['python'],str(OUT/'local_source'/j['script']),str(j['seed'])],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
            running[j['name']]=(p,j,gpu,h);write(OUT/f'logs/{j["name"]}_LAUNCH.json',dict(pid=p.pid,gpu=gpu,job=j['name']))
        # A failed native prerequisite must not leave an unexplained infinite wait.
        for j in pending[:]:
            failure=j['dependency'].parent/'PROCESS_EXIT_v2.json'
            if failure.exists() and json.loads(failure.read_text())['exit_code']!=0:
                pending.remove(j);done.append(dict(name=j['name'],stage='dependency_failed',complete=False))
        write(OUT/'COMPARATOR_STATUS.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pending=[j['name'] for j in pending],running=[dict(name=k,pid=v[0].pid,gpu=v[2]) for k,v in running.items()],done=done))
        if pending or running:time.sleep(5)
    write(OUT/'COMPARATOR_EXECUTION_COMPLETE.json',dict(done=done,all_complete=all(x['complete'] for x in done),test_reads=0))
    # The independent evaluator is registered separately; never read sealed test here.
    evaluator=OUT/'local_source/evaluate_comparators.py'
    if evaluator.exists():
        env=dict(os.environ,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        with (OUT/'logs/comparator_evaluation.log').open('x') as h:
            rc=subprocess.call([str(ARGN),str(evaluator)],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
        write(OUT/'EVALUATION_EXIT.json',dict(exit_code=rc))
if __name__=='__main__':main()
