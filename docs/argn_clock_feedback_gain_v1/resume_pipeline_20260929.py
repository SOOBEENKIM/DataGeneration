"""Resume the unchanged registered generation/evaluation in a persistent tmux session."""
from datetime import datetime, timezone
import fcntl, hashlib, json, os
from pathlib import Path
import shutil, subprocess, sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def write(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    with (OUT/'PIPELINE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (OUT/'FIT_COMPLETE.json').exists()
        assert not (OUT/'LAUNCH.json').exists(), 'Existing launches must be audited before resuming'
        archive=OUT/'report/restart_20260929'
        archive.mkdir(exist_ok=False)
        for name in ['PIPELINE_STARTED.json','PIPELINE_STATUS.json','STATUS.json','logs/pipeline.log','logs/dispatch.log']:
            source=OUT/name
            if source.exists():
                dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        write(archive/'RESTART.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            previous_pipeline_pid=1907476,previous_process_absent=True,previous_last_queue_update='2026-09-28T15:08:26Z',
            previous_exit_reason='not established; no traceback or completion record',
            session='argn-feedback-20260929',model_protocol_and_fits_unchanged=True))
        for stage in ['dispatch','report_feedback']:
            write(OUT/'PIPELINE_STATUS.json',dict(stage=stage,pid=os.getpid(),session='argn-feedback-20260929',updated_utc=datetime.now(timezone.utc).isoformat()))
            print('RESTART_STAGE',stage,flush=True)
            with (OUT/f'logs/{stage}_20260929.log').open('x') as log:
                subprocess.run([sys.executable,str(OUT/f'local_source/{stage}.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(OUT/'PIPELINE_STATUS.json',dict(stage='completed',pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat()))
        print('RESTART_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'PIPELINE_RESTART_ERROR.json',dict(error=repr(error),utc=datetime.now(timezone.utc).isoformat()))
        raise
