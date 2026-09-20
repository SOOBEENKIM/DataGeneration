"""One process per empty GPU; wait without altering anybody else's processes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cs_saf/external_port_v1'
PY='/home/finx_sbk/.conda/envs/cofseq/bin/python3'
ARGN=ROOT/'external/cs_saf_external_audit/runtime/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('jobs',nargs='+')
    a=p.parse_args();(OUT/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='4')
    for spec in a.jobs:
        model,name=spec.split(':')
        while True:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            cards={int(x.split(',')[0]):tuple(map(int,x.split(',')[1:])) for x in lines.splitlines()}
            used,util=cards[a.gpu]
            if used<512 and util<5:break
            print('WAITING',a.gpu,used,util,flush=True);time.sleep(15)
        logfile=OUT/'logs'/f'{name}_{model}.log'
        with logfile.open('x') as log:
            print('START',spec,'GPU',a.gpu,flush=True)
            proc=subprocess.Popen([str(ARGN) if model=='ARGN' else PY,str(ROOT/'scripts/run_cs_saf_external_port.py'),model,name],
                                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            admission=OUT/'logs'/f'{name}_{model}_admission.json'
            admission.write_text(json.dumps(dict(gpu=a.gpu,memory_used_before=used,utilization_before=util,pid=proc.pid))+'\n')
            code=proc.wait()
            print('FINISHED',spec,'exit',code,flush=True)
            if code:print('Failure recorded; no automatic retry or setting change',flush=True)


if __name__=='__main__':main()
