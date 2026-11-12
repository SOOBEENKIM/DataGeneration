"""Take over only recorded MeLD workers at an exact atomic epoch boundary."""
import fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/meld_500_pilot_v1'
OLD=ROOT/'artifacts/comparison_completion_v1'
PYTHON=ROOT.parent/'research-reporting/argn-state-first-2026-09-27/runtime/bin/python'
sys.path.insert(0,str(OLD/'local_source'))
from dispatch_cpar import idle
import torch
import hashlib

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,default=str)+'\n');q.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),start_ticks=stat[19],state=stat[0])
    except FileNotFoundError:return None
def matches(bound):
    live=identity(bound['pid'])
    return live is not None and live['start_ticks']==bound['start_ticks'] and live['cmd']==bound['cmd'] and live['state']!='Z'
def send(bound,sig):
    assert matches(bound),f'Process identity changed: {bound}'
    os.kill(bound['pid'],sig)
def stopped(bound):
    live=identity(bound['pid']);return matches(bound) and live['state'] in ('T','t')
def stop(bound):
    send(bound,signal.SIGSTOP)
    for _ in range(200):
        if stopped(bound):return
        time.sleep(.05)
    raise RuntimeError(f'Could not suspend {bound}')
def terminate(bound):
    send(bound,signal.SIGTERM)
    if stopped(bound):send(bound,signal.SIGCONT)
    for _ in range(100):
        if not matches(bound):return
        time.sleep(.1)
    raise RuntimeError(f'Worker did not exit after SIGTERM {bound}')

def snapshot_files(src,dst,expected=500):
    last=torch.load(src/'vae_last.pt',map_location='cpu',weights_only=False)
    best=torch.load(src/'vae_best.pt',map_location='cpu',weights_only=False)
    assert last['epoch']==expected,(last['epoch'],expected)
    assert last['best_epoch']==best['epoch']<=expected
    assert abs(last['best']-best['internal']['reconstruction'])<1e-10
    rows=[json.loads(x) for x in (src/'VAE_CURVE.jsonl').read_text().splitlines()]
    assert rows[-1]['epoch']==expected,rows[-1]
    vals=[r for r in rows if 'internal' in r and r['epoch']<=expected]
    chosen=min(vals,key=lambda r:r['internal']['reconstruction'])
    assert chosen['epoch']==best['epoch']
    dst.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in ['vae_last.pt','vae_best.pt','VAE_CURVE.jsonl','STATUS.json','START.json']:
        p=src/name;q=dst/name;shutil.copy2(p,q)
        assert digest(p)==digest(q);hashes[str(q)]=digest(q)
    write(dst/'SNAPSHOT.json',dict(epoch=expected,best_epoch=best['epoch'],
          internal_best=last['best'],hashes=hashes,optimizer_and_rng_preserved=True,
          original_source=str(src),preliminary=True))
    return json.loads((dst/'SNAPSHOT.json').read_text())

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'SUPERVISOR_LOCK').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'STATUS.json').exists(),'Inspect existing supervisor before recovery'
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    config=json.loads((OUT/'CONTROL.json').read_text());dispatcher=config['dispatcher']
    bindings=config['workers'];torch.set_num_threads(1)
    assert matches(dispatcher)
    for b in bindings:
        assert matches(b)
        assert 'meld_train.py' in b['cmd'] and str(b['seed']) in b['cmd']
        s=json.loads((OLD/f'meld/fit_{b["seed"]}/STATUS.json').read_text())
        assert s['stage']=='VAE_training' and s['epoch']<500
    stop(dispatcher)
    write(OUT/'TAKEOVER.json',dict(dispatcher=dispatcher,suspended=True,workers_continue=True,
          target_epoch=500,reason='User changed 600 to 500; preserve old longer-budget run.',utc=time.time()))
    write(OLD/'CONTROLLED_BY_500_PILOT.json',dict(supervisor_pid=os.getpid(),study=str(OUT),
          original_dispatcher_suspended=True,target_epoch=500,final_evaluation_deferred=True))
    waiting={b['seed']:b for b in bindings};pending=[];running={};done=[];retired=False;last_write=0.
    try:
      while waiting or pending or running:
        for fs,b in list(waiting.items()):
            assert matches(b),f'Original VAE unexpectedly exited for {fs}'
            src=OLD/f'meld/fit_{fs}';s=json.loads((src/'STATUS.json').read_text())
            assert s['stage']=='VAE_training',f'Original VAE ended before boundary {fs}: {s}'
            if s['epoch']<500:continue
            last=torch.load(src/'vae_last.pt',map_location='cpu',weights_only=False)
            if last['epoch']<500:continue  # STATUS is written just before the atomic checkpoint.
            stop(b)
            record=snapshot_files(src,OUT/'snapshots'/str(fs))
            write(OUT/'snapshots'/str(fs)/'BOUNDARY_PROCESS.json',dict(binding=b,paused_after_saved_epoch=500,
                 discarded_unsaved_next_epoch_possible=True,original_longer_run_resumable=True))
            terminate(b)
            write(OUT/'snapshots'/str(fs)/'PROCESS_STOPPED.json',dict(expected=True,reason='requested_500_epoch_boundary',pid=b['pid']))
            del waiting[fs];pending.append(fs)
            print('CAPTURED_AND_STOPPED',fs,record['epoch'],record['best_epoch'],flush=True)
        for fs,(p,gpu,h) in list(running.items()):
            rc=p.poll()
            if rc is not None:
                h.close();rec=dict(seed=fs,pid=p.pid,gpu=gpu,exit_code=rc,
                      complete=(OUT/f'runs/{fs}/COMPLETE.json').exists(),adequacy_block=(OUT/f'runs/{fs}/ADEQUACY_BLOCK.json').exists())
                write(OUT/f'logs/pilot_{fs}_EXIT.json',rec);done.append(rec);del running[fs]
        if not waiting and not retired:
            terminate(dispatcher);retired=True
            old=json.loads((OLD/'COMPARATOR_STATUS.json').read_text())
            write(OUT/'ORIGINAL_STATUS_AT_HANDOFF.json',old)
            old.update(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),running=[],
              original_long_run_paused=True,superseded_by=str(OUT),reason='User-requested 500-epoch preliminary branch; not full comparison completion')
            write(OLD/'COMPARATOR_STATUS.json',old)
            write(OUT/'DISPATCHER_RETIRED.json',dict(pid=dispatcher['pid'],expected=True))
        if pending:
            free=[g for g in idle() if g not in [v[1] for v in running.values()]]
            for gpu,fs in zip(free,pending[:]):
                if gpu not in idle():continue
                pending.remove(fs);h=(OUT/f'logs/pilot_{fs}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source:{OLD}/local_source',
                    OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCHDYNAMO_DISABLE='1')
                p=subprocess.Popen([str(PYTHON),str(OUT/'local_source/pilot_train.py'),str(fs)],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
                running[fs]=(p,gpu,h);write(OUT/f'logs/pilot_{fs}_LAUNCH.json',dict(seed=fs,pid=p.pid,gpu=gpu))
        if time.monotonic()-last_write>2:
            write(OUT/'STATUS.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stage='waiting_vae_500' if waiting else 'pilot_training_or_evaluation',
              waiting_vae=[dict(seed=fs,pid=b['pid'],epoch=json.loads((OLD/f'meld/fit_{fs}/STATUS.json').read_text())['epoch']) for fs,b in waiting.items()],
              pending=pending,running=[dict(seed=fs,pid=x[0].pid,gpu=x[1]) for fs,x in running.items()],done=done,
              target_vae_epoch=500,dit_cap=2000,preliminary=True,final_comparison_complete=False))
            last_write=time.monotonic()
        if waiting or pending or running:time.sleep(.25 if waiting else 2)
      from evaluate_pilot import summary
      summary()
      for p,h in manifest['hashes'].items():assert digest(p)==h,p
      write(OUT/'EXECUTION_COMPLETE.json',dict(done=done,all_complete=all(x['complete'] for x in done),
              preliminary=True,final_comparison_complete=False,test_reads=0))
      status=json.loads((OUT/'STATUS.json').read_text());status.update(stage='complete' if all(x['complete'] for x in done) else 'requires_inspection',waiting_vae=[],pending=[],running=[],done=done)
      write(OUT/'STATUS.json',status)
    except BaseException:
      # Preserve exact process identity and checkpoints; do not let the old 1000/100000
      # process silently bypass the requested budget after a controller error.
      for b in waiting.values():
          if matches(b) and not stopped(b):stop(b)
      write(OUT/'FAILED.json',dict(error=traceback.format_exc(),waiting=list(waiting),
             original_workers_suspended=True,running_pilot=[x[0].pid for x in running.values()]))
      raise

if __name__=='__main__':main()
