"""Same-source CPU/GPU gates and one idle-GPU replay worker."""
import argparse,json,os,time
import torch
from experiments.cs_saf_calibrated_replay import (ROOT,OUTPUT,CONFIG_SHA,contract,smoke,
    run_cell,folder_for,verify,frozen_source,write_json)
from experiments.cof_seqgen_saf_training import _seed_everything
from scripts.run_cs_saf_generation_repeats import available_gpus


def check_gate(name,source):
    verify(OUTPUT/name);g=json.loads((OUTPUT/name/'gate.json').read_text())
    if g['decision']!='PASS' or g['source_commit']!=source:raise ValueError('same-source gate required')


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cpu','gpu','run']);p.add_argument('--device',default='cpu')
    a=p.parse_args();c=contract();source=frozen_source();OUTPUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);_seed_everything(20260920)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.phase=='cpu':
        if a.device!='cpu':raise ValueError('CPU gate device mismatch')
        smoke(torch.device('cpu'),'cpu_gate');return
    check_gate('cpu_gate',source)
    if a.phase=='run':check_gate('gpu_gate',source)
    # The caller selects an idle physical GPU. Recheck immediately before CUDA use.
    idle=available_gpus();index=int(a.device.split(':')[-1])
    if not a.device.startswith('cuda:') or index not in [v['index'] for v in idle]:raise RuntimeError('selected GPU is not idle')
    allocation=next(v for v in idle if v['index']==index)
    torch.cuda.set_device(a.device);torch.cuda.set_per_process_memory_fraction(.20,a.device)
    os.nice(10)
    if a.phase=='gpu':smoke(torch.device(a.device),'gpu_gate');return
    started=time.time();completed=[]
    for pi in c['prevalences']:
        for k in c['kappas']:
            for t in c['trials']:
                run_cell(pi,k,t,torch.device(a.device));completed.append([pi,k,t])
                write_json(OUTPUT/'progress.json',dict(source_commit=source,completed=completed,total=40,
                    allocation=allocation,pid=os.getpid(),elapsed_seconds=time.time()-started))
    write_json(OUTPUT/'GRID_COMPLETE.json',dict(source_commit=source,config_sha256=CONFIG_SHA,
        completed=completed,source_paths=400,predictor_path_evaluations=800,new_fits=0,new_sequences=0,
        allocation=allocation,seconds=time.time()-started))

if __name__=='__main__':main()
