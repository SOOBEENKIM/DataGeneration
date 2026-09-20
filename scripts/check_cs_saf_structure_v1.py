"""Mathematical, data-alignment, deterministic-training and runtime gates."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cs_saf_structure_v1 import (config,source_record,payload_for,make_model,train,
    DOC,OUT,batch,state_digest,evaluate)
from models.cs_saf_structure import constrained_repeat_logits
from models.cs_saf_observable_coupling import observable_gap_repeat_joint
from scripts.run_cs_saf_external_audit_v1 import write
from experiments.cs_saf_generation_repeats import generate_fixed_plan
from scripts.run_cs_saf_u_repeat_controls import u_control


def forward(model,x):
    hidden=model.encoder(**x);context=model.context(hidden,x['static_categorical'])
    mask=x['valid_mask'].clone();mask[:,0]=False
    previous=x['receiver'].roll(1,1);previous[:,0]=1
    codes=x['static_categorical'][0][:,None].expand_as(mask)
    out=model.mark_distribution(context,x['gap'],previous,mask,static_codes=codes)
    return hidden,context,previous,mask,codes,out


def cpu_gate():
    torch.set_num_threads(1);source=source_record();torch.manual_seed(13)
    args=(torch.randn(2,5,dtype=torch.float64,requires_grad=True),
          torch.randn(2,dtype=torch.float64,requires_grad=True),
          torch.randn(2,5,dtype=torch.float64,requires_grad=True))
    assert torch.autograd.gradcheck(constrained_repeat_logits,args,eps=1e-6,atol=1e-5,rtol=1e-4)
    z=constrained_repeat_logits(*args)
    joint=observable_gap_repeat_joint(*args)
    equivalent=float((z.sigmoid()-joint[...,1]/joint.sum(-1)).abs().max())
    assert equivalent<1e-12
    extreme=(torch.randn(2,31,dtype=torch.float64,requires_grad=True),
        torch.tensor([-25.,25.],dtype=torch.float64,requires_grad=True),
        torch.randn(2,31,dtype=torch.float64,requires_grad=True)*3)
    ez=constrained_repeat_logits(*extreme)
    grads=torch.autograd.grad(F.logsigmoid(ez).sum(),extreme)
    assert all(torch.isfinite(g).all() for g in grads)
    payload,_=payload_for(1)
    x=batch(payload['train'],torch.cat([torch.where(payload['train']['codes']==g)[0][:4] for g in (3,4)]),'cpu')
    models={name:make_model(payload,name,0) for name in ('U','G','C')}
    initial={name:state_digest(m) for name,m in models.items()}
    assert len(set(initial.values()))==1
    assert state_digest(make_model(payload,'G',1))!=initial['G']
    outputs={name:forward(m,x)[-1][0] for name,m in models.items()}
    torch.testing.assert_close(outputs['G'],outputs['C'],rtol=1e-6,atol=1e-6)
    checks={};max_constraint=0.
    for name,m in models.items():
        with torch.no_grad():m.route_interaction_weight.fill_(.4)
        hidden,h,previous,mask,codes,(lp,lr,lnr)=forward(m,x)
        torch.testing.assert_close(lp.exp().sum(-1),torch.ones_like(mask,dtype=lp.dtype),atol=3e-7,rtol=0)
        assert (lp.exp()[...,:3]==0).all()
        if name!='U':
            torch.testing.assert_close(lp.exp()[mask].gather(-1,previous[mask][:,None])[:,0],lr[mask].exp())
            logits=m.new_mark_head(h).clone();logits[...,:3]=-torch.inf
            nonrepeat=F.log_softmax(logits.scatter(-1,previous[...,None],-torch.inf),-1)
            target=x['receiver'];repeated=(target==previous)
            conditional=torch.where(repeated,lr,lnr+nonrepeat.gather(-1,target[...,None]).squeeze(-1))
            torch.testing.assert_close(lp.gather(-1,target[...,None]).squeeze(-1)[mask],conditional[mask])
            if name=='C':
                r=m.repeat_logits(h,codes).sigmoid();a=m.gap_decoder.logits(hidden).softmax(-1)
                max_constraint=float(((a*r).sum(-1)-m.copy_base(h).squeeze(-1).sigmoid()).abs().max())
                assert max_constraint<1e-6
        altered={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in x.items()}
        altered['receiver'][:,3:]=3+(altered['receiver'][:,3:]-2)%64
        altered['numeric_value'][:,3:]+=10
        altered['gap'][:,4:]=m.support.representatives[-1]
        changed=forward(m,altered)
        torch.testing.assert_close(lp[:,:4],changed[-1][0][:,:4],rtol=0,atol=0)
        terms=m.loss_terms(**x);loss,base,_=m.objective(terms,train_entities=8,transition_counts_by_code={3:80,4:80})
        assert torch.equal(loss,base);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        # Same-seed sampler and reload, plus direct controls through actual generated history.
        m.eval();snapshot={k:v.detach().clone() for k,v in m.state_dict().items()}
        lengths=[5,9,6,11];static=(torch.tensor([3,4,3,4]),)
        torch.manual_seed(101);sample=m.sample_fixed_lengths(lengths,static_categorical=static)
        reloaded=make_model(payload,name,0);reloaded.load_state_dict(snapshot);reloaded.eval()
        torch.manual_seed(101);again=reloaded.sample_fixed_lengths(lengths,static_categorical=static)
        for key in sample:torch.testing.assert_close(sample[key],again[key],rtol=0,atol=0,equal_nan=True)
        control=dict(kind='direct',edges=[[1,2,3,4]]*2,parameters=[[0.]*5,[1.]*5])
        with u_control(m,control):
            torch.manual_seed(102);forced=m.sample_fixed_lengths(lengths,static_categorical=static)
        for row in range(4):
            marks=forced['receiver'][row,:lengths[row]]
            assert bool((marks[1:]==marks[:-1]).all()) if row%2 else bool((marks[1:]!=marks[:-1]).all())
        checks[name]=dict(normalized=True,first_step_uses_common_head=True,strict_past=True,
            nll_factorization=True,finite_gradients=True,deterministic_reload_generation=True,
            forced_repeat_group_alignment=True,parameters=sum(p.numel() for p in m.parameters()))
    training={}
    for name in ('U','G','C'):
        results=[]
        for repeat in ('a','b'):
            _,r=train(1,0,name,'cpu',f'{name}_{repeat}')
            assert r['final_train_nll']<.9*r['initial_train_nll']
            assert r['best_epoch']==min(r['history'],key=lambda row:row['validation']['base_nll'])['epoch']
            results.append(r)
        assert results[0]['best_state_sha256']==results[1]['best_state_sha256']
        assert results[0]['history']==results[1]['history']
        training[name]=dict(initial_nll=results[0]['initial_train_nll'],best_nll=results[0]['best_train_nll'],
            final_train_nll=results[0]['final_train_nll'],best_checkpoint_selection_verified=True,
            exact_repeated_training=True,best_state_sha256=results[0]['best_state_sha256'])
    write(DOC/'cpu_gate.json',dict(status='PASS',source_hashes=source['hashes'],source_commit=source['commit'],
        common_initial_state_sha256=initial,gradient_check=True,old_coupling_probability_max_error=equivalent,
        C_fixed_history_marginal_max_error=max_constraint,extreme_logits_gradient_finite=True,
        checks=checks,tiny_training=training,scientific_fits=0))
    print('CPU_GATE_PASS',flush=True)


def gpu_gate():
    torch.set_num_threads(1);c=config();source=source_record()
    cpu=json.loads((DOC/'cpu_gate.json').read_text());assert cpu['status']=='PASS' and cpu['source_hashes']==source['hashes']
    torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
    payload,_=payload_for(1);x=batch(payload['train'],torch.arange(c['batch_size']),'cuda')
    checks={}
    for name in ('U','G','C'):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        m=make_model(payload,name,0,'cuda');opt=torch.optim.AdamW(m.parameters(),lr=c['learning_rate'])
        started=time.monotonic()
        for _ in range(3):
            opt.zero_grad(set_to_none=True);terms=m.loss_terms(**x)
            loss,_,_=m.objective(terms,train_entities=512,transition_counts_by_code={3:10000,4:1000})
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();seconds=(time.monotonic()-started)/3
        m.eval()
        with torch.no_grad():
            _,_,_,_,_,out=forward(m,x)
            cpu_model=make_model(payload,name,0);cpu_model.load_state_dict({k:v.cpu() for k,v in m.state_dict().items()});cpu_model.eval()
            xc={k:(v.cpu() if isinstance(v,torch.Tensor) else tuple(t.cpu() for t in v)) for k,v in x.items()}
            other=forward(cpu_model,xc)[-1]
            error=float((out[0].exp().cpu()-other[0].exp()).abs().max())
            assert error<1e-5
        checks[name]=dict(seconds_per_training_batch=seconds,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cpu_gpu_probability_max_error=error)
        del m,opt,cpu_model,out,other
    write(DOC/'gpu_gate.json',dict(status='PASS',source_hashes=source['hashes'],source_commit=source['commit'],checks=checks,
        device=torch.cuda.get_device_name(),scientific_fits=0))
    print('GPU_GATE_PASS',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','gpu']);a=p.parse_args()
    cpu_gate() if a.device=='cpu' else gpu_gate()
