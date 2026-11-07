import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from study_cpar import ROOT,OUT,verify,write
# Use simple NVIDIA read-only probe, independent of ARGN runtime.
def available():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True);busy=set()
 for line in apps.strip().splitlines():
  pid,uuid=[x.strip() for x in line.split(',')]
  try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  except (FileNotFoundError,PermissionError):cmd=b'unknown'
  if b'nvidia-cuda-mps-server' not in cmd:busy.add(uuid)
 return [x[1].strip() for x in [r.split(',') for r in rows.strip().splitlines()] if int(x[2])<500 and int(x[3])<=5 and x[1].strip() not in busy]
def main():
 lock=(OUT/'LOCK').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();assert not (OUT/'LAUNCH.json').exists()
 consecutive={}
 while True:
  free=available()
  for gpu in free:consecutive[gpu]=consecutive.get(gpu,0)+1
  consecutive={g:n for g,n in consecutive.items() if g in free}
  ready=[g for g,n in consecutive.items() if n>=2]
  # Descending choice avoids the ascending ARGN dispatcher when several devices free.
  if ready:
   gpu=ready[-1]
   if gpu not in available():continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
   with (OUT/'logs/worker.log').open('x') as h:
    p=subprocess.Popen([sys.executable,str(OUT/'local_source/study_cpar.py')],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
    write(OUT/'LAUNCH.json',dict(pid=p.pid,gpu=gpu,executable=sys.executable))
    rc=p.wait()
   write(OUT/'PROCESS_EXIT.json',dict(exit_code=rc,complete=(OUT/'COMPLETE.json').exists()));return
  write(OUT/'STATUS.json',dict(stage='waiting_for_idle_gpu',pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
  time.sleep(5)
if __name__=='__main__':main()
