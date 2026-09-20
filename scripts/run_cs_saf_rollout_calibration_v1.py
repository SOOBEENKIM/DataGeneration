"""Technical gates and one-GPU queue; never selects on evaluation outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cs_saf_rollout_calibration_v1 import *


def gate(device):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(config()['gpu_memory_fraction'])
    src=sources();reports=[]
    for name in ['U','G']:
        model,payload,a,features,provenance=load_parent(0,0,name,device)
        fitplan,selectionplan=plans(payload['train'],0)
        assert not set(fitplan['positions'])&set(selectionplan['positions'])
        assert set(np.asarray(payload['train']['entity_ids'])[fitplan['positions']])<=set(features.entity_id)
        target=TrainingTarget(features,a)
        frame=features.copy();frame['bin']=assign_bins(frame.gap_code.to_numpy(),frame.group.to_numpy(),a['edges'])
        independent=np.array([pd.crosstab(frame[frame.group==g]['bin'],frame[frame.group==g].y).reindex(
            index=range(5),columns=[0,1],fill_value=0).to_numpy().reshape(-1)/sum(frame.group==g) for g in [0,1]])
        np.testing.assert_allclose(target.target,independent,atol=1e-15)
        z=np.asarray(a['parameters']);assert target.feasible(z)
        altered=z+2;assert not target.feasible(altered)
        p0=apply_numpy(features.p,features.gap_code,features.group,a)
        p1=apply_numpy(features.p,features.gap_code,features.group,control_at(a,altered))
        old=features.p.to_numpy().clip(1e-9,1-1e-9);y=features.y.to_numpy()
        full0=np.where(y,p0,features.obs_p*(1-p0)/(1-old))
        full1=np.where(y,p1,features.obs_p*(1-p1)/(1-old))
        for g in [0,1]:
            ix=features.group.to_numpy()==g
            np.testing.assert_allclose(np.mean(-np.log(full1[ix])+np.log(full0[ix])),
                (target.costs(altered)-target.base)[g,1],atol=1e-12)
        tiny={key:(v[[0,1,1024,1025]] if key!='positions' else v[[0,1,1024,1025]]) for key,v in fitplan.items()}
        sample=generate(model,tiny,812,a,device,check_prefix=True)
        check=sample.pop('prefix_check');again=generate(model,tiny,812,a,device)
        for key in sample:torch.testing.assert_close(sample[key],again[key],rtol=0,atol=0,equal_nan=True)
        changed=generate(model,tiny,812,control_at(a,z-2),device)
        assert not torch.equal(sample['receiver'],changed['receiver'])
        assert not torch.allclose(sample['gap'],changed['gap'],equal_nan=True)
        assert not torch.equal(sample['numeric_value'],changed['numeric_value'])
        assert state_digest(model)==provenance['state_sha256'] and all(not p.requires_grad for p in model.parameters())
        # Independent full-prefix next-event probabilities on the same histories.
        cpu=parent.make_model(payload,name,0,'cpu');cpu.load_state_dict(model.state_dict());cpu.eval()
        h,_=initial_hidden(model,tiny['codes'].to(device));hc,_=initial_hidden(cpu,tiny['codes'])
        pp=corrected_probabilities(model,h,tiny['codes'].to(device),sample['gap'][:,0].to(device),
            torch.ones(4,dtype=torch.long,device=device),torch.zeros(4,dtype=torch.bool,device=device),a).cpu()
        pc=corrected_probabilities(cpu,hc,tiny['codes'],sample['gap'][:,0],torch.ones(4,dtype=torch.long),
            torch.zeros(4,dtype=torch.bool),a)
        error=float((pp-pc).abs().max());assert error<1e-5
        # Stronger same-generated-history CPU/GPU comparison after every event.
        hd,sd=initial_hidden(model,tiny['codes'].to(device));hh,sh=initial_hidden(cpu,tiny['codes'])
        for t in range(1,32):
            args=[sample[key][:,t-1] for key in ['gap','receiver','numeric_value','valid_mask']]
            hd,sd=stream_event(model,*[x.to(device) for x in args],sd);hh,sh=stream_event(cpu,*args,sh)
            pr=sample['receiver'][:,t-1];ac=sample['valid_mask'][:,t]
            pdv=corrected_probabilities(model,hd,tiny['codes'].to(device),sample['gap'][:,t].to(device),pr.to(device),ac.to(device),a).cpu()
            pcv=corrected_probabilities(cpu,hh,tiny['codes'],sample['gap'][:,t],pr,ac,a)
            error=max(error,float((pdv-pcv).abs().max()))
        assert error<1e-5
        reports.append(dict(name=name,prefix=check,cpu_device_probability_error=error,
            target_counts_verified=True,full_mark_nll_delta_verified=True,feedback_gap_and_value_changed=True,
            frozen_weights=True,reproducible_tapes=True,disjoint_train_plans=True))
    counts=[0,0]
    def objective(x):counts[0]+=1;return float(np.sum((x-.6)**2))
    def selection(x):counts[1]+=1;return float(np.sum((x-.4)**2))
    result,trace=pattern_search(np.zeros(10),objective,selection,lambda x:bool((x<=.4+1e-12).all()),0)
    assert counts==[61,4] and np.max(result)<=.4+1e-12 and trace['selected_endpoint']==1
    report=dict(status='PASS',source_hashes=src['hashes'],source_commit=src['commit'],checks=reports,
        pattern_search_calls=counts,device=device,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        timestamp=time.time(),scientific_outcomes_evaluated=False)
    if device.startswith('cuda'):
        start=time.monotonic();generate(model,fitplan,20264201,a,device)
        report.update(full_plan_sample_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    path=DOC/('gpu_gate.json' if device.startswith('cuda') else 'cpu_gate.json')
    assert not path.exists(),'preserve earlier gate before retry'
    write(path,report);print(json.dumps(report),flush=True)


def admit():
    c=config()
    while True:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        rows=[tuple(int(x.strip()) for x in line.split(',')) for line in output.strip().splitlines()]
        free=[r for r in rows if r[1]>=c['gpu_min_free_mib'] and r[2]<=c['gpu_max_utilization_at_admission']]
        if free:
            chosen=sorted(free,key=lambda r:(r[2],-r[1],r[0]))[0]
            print('GPU_ADMISSION',chosen,flush=True);return str(chosen[0])
        print('WAIT_GPU',rows,flush=True);time.sleep(30)


def dispatch():
    src=sources();OUT.mkdir(parents=True,exist_ok=True)
    for k in config()['kappas']:
        for t in config()['trials']:
            for name in config()['models']:
                if all((folder(k,t,name)/f'{m}_DONE.json').exists() for m in ['B','P']):continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=admit(),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                cmd=[sys.executable,__file__,'fit','--kappa',str(k),'--trial',str(t),'--model',name]
                subprocess.run(cmd,env=env,cwd=ROOT,check=True)
    write(OUT/'ALL_FITS_DONE.json',dict(fits=all_fitted(),timestamp=time.time(),source=src))
    print('ALL_24_FITS_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cpu-gate','gpu-gate','fit','dispatch'])
    p.add_argument('--kappa',type=int);p.add_argument('--trial',type=int);p.add_argument('--model',choices=['U','G'])
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.phase.endswith('gate'):gate('cpu' if a.phase=='cpu-gate' else 'cuda:0')
    elif a.phase=='dispatch':dispatch()
    else:
        torch.cuda.set_per_process_memory_fraction(config()['gpu_memory_fraction'])
        fit(a.kappa,a.trial,a.model,'cuda:0')
