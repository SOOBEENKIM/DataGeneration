"""Finish the authorized study once CPU fits and idle GPU capacity are ready."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')

def main():
    assert not (OUT/'PIPELINE_STARTED.json').exists()
    save('PIPELINE_STARTED.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    while not (OUT/'FIT_COMPLETE.json').exists():
        log=OUT/'logs/prepare.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Preparation failed; see logs/prepare.log')
        save('PIPELINE_STATUS.json',dict(stage='waiting_for_cpu_fits',updated_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(5)
    for name in ['dispatch','report_feedback']:
        save('PIPELINE_STATUS.json',dict(stage=name,updated_utc=datetime.now(timezone.utc).isoformat()))
        print('PIPELINE_STAGE',name,flush=True)
        with (OUT/f'logs/{name}.log').open('x') as log:
            subprocess.run([sys.executable,str(OUT/f'local_source/{name}.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    save('PIPELINE_STATUS.json',dict(stage='completed',updated_utc=datetime.now(timezone.utc).isoformat()))
    print('PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('PIPELINE_ERROR.json',dict(error=repr(error),utc=datetime.now(timezone.utc).isoformat()))
        raise
