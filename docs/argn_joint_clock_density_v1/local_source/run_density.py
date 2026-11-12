"""Registered 2x2 residual output x time-rollout objective screening."""
import argparse, copy, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(Path(__file__).parent)]
from residual_density import ResidualDensity,InitialClock,implicit_sample,clock_tensor,residual_generation
from benchmarks.argn_time_density import PastClock
from run_argn_amount_learning import digest,write,seed

OUT=ROOT/'artifacts/argn_joint_clock_density_v1';DOCS=OUT/'report'
BASELINE=ROOT/'artifacts/argn_clock_initial_coupling_v1/init_only/time_head.pt'
ARMS=['mean_marginal','mean_joint','mixture_marginal','mixture_joint']
CFG=dict(arms=ARMS,parent_seed=20260930,fit_seed=20261330,hidden=64,components=3,
    learning_rate=.0003,weight_decay=.0001,nll_steps=400,rollout_steps=160,
    validation_every=40,prefix_refresh_every=80,chunk_length=64,chunks=16,nll_batch=512,
    quantiles=128,gradient_clip=1.,nll_weight=.05,joint_weight=.25,
    validation_noises=[20261401,20261402],training_noise=20261400,
    generation_seeds=[20261011,20261012],fit_seeds=[20260930],
    full_length_replay=True,backbone_in_replay='frozen_real_prefix',
    native_generation='all_fields_recursive',no_automatic_promotion=True)
BINS=[(6,50),(50,100),(100,200),(200,500),(500,1024),(1024,2048),(2048,None)]

def verify():
    m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p

def load_data(split,device):
    d=torch.load(OUT/f'cache/{split}.pt',map_location='cpu',weights_only=True)
    d={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in d.items()}
    d['pools']=[torch.where((d['phase']==p)&(d['position']>5))[0] for p in range(4)]
    return d

def new_head(d,arm,device):
    p=torch.load(BASELINE,map_location='cpu',weights_only=True)
    head=ResidualDensity(d['base'].shape[-1],arm).to(device)
    missing,unexpected=head.load_state_dict(p['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('base_','residual.')) for k in missing)
    with torch.no_grad():
        head.base_center.copy_(d['base'].mean(0));head.base_scale.copy_(d['base'].std(0).clamp(min=.05))
    return head

def empirical_w1(x,y,ids=None):
    """Differentiable weighted quantile approximation, explicitly 128 points."""
    if len(x)==0:return x.new_tensor(0.)
    q=(torch.arange(CFG['quantiles'],device=x.device,dtype=x.dtype)+.5)/CFG['quantiles']
    if ids is None:
        return (torch.quantile(x,q)-torch.quantile(y,q)).abs().mean()
    _,inverse,counts=torch.unique(ids,return_inverse=True,return_counts=True)
    weights=1./counts[inverse].to(x.dtype)
    def quantile(v):
        values,ix=v.sort();cdf=weights[ix].cumsum(0)/weights.sum()
        return values[torch.searchsorted(cdf.contiguous(),q).clamp(max=len(values)-1)]
    return (quantile(x)-quantile(y)).abs().mean()

def measures(real_z,real_clock,generated_z,generated_clock,phase,pos,ids):
    normal=phase.remainder(2).eq(0);nn=phase.eq(0)&pos.ge(6)
    valid=pos.ge(6)
    relative_real=torch.nn.functional.softplus(real_z-real_clock)
    relative_gen=torch.nn.functional.softplus(generated_z-generated_clock)
    # This training relative coordinate is log(1+(1+gap)/(1+clock_gap)),
    # not the headline raw-gap ratio. Exact raw-gap metrics are separate.
    out={}
    def dist(mask,a=real_z,b=generated_z,customer=True):
        return empirical_w1(a[mask],b[mask],ids[mask] if customer else None)
    out['normal']=dist(normal)
    out['event_normal']=dist(normal,customer=False)
    out['clock']=dist(nn,real_clock,generated_clock)
    out['personal']=dist(nn,relative_real,relative_gen)
    out['onset']=dist(phase.eq(1)&valid,relative_real,relative_gen)
    out['fraud']=dist(phase.remainder(2).eq(1))
    terms=[];joint=[]
    for lo,hi in BINS:
        position=pos.ge(lo)&(pos.lt(hi) if hi is not None else torch.ones_like(pos,dtype=torch.bool))
        mask=nn&position
        if mask.any():
            val=dist(mask);out[f'bin_{lo}']=val;terms.append(val)
        for p in range(4):
            mask=phase.eq(p)&position
            if mask.sum()<4:continue
            for a,b in [(1.,0.),(0.,1.),(2**-.5,2**-.5),(2**-.5,-2**-.5)]:
                joint.append(dist(mask,a*real_clock+b*real_z,a*generated_clock+b*generated_z))
    out['bins']=torch.stack(terms).mean() if terms else out['normal']*0
    out['joint']=torch.stack(joint).mean() if joint else out['normal']*0
    return out

def objective(m,reference,joint):
    value=.5*(m['normal']+m['bins'])+.25*m['clock']
    for k,tol in [('personal',.01),('onset',.01),('fraud',.02)]:
        value=value+4*torch.relu(m[k]-reference[k]-tol)
    if joint:
        value=value+CFG['joint_weight']*m['joint']
        guards=[torch.relu(v-reference[k]-.02) for k,v in m.items() if k.startswith('bin_')]
        guards.append(torch.relu(m['event_normal']-reference['event_normal']-.01))
        # Worst violating group cannot be traded off by averaging other bins.
        value=value+4*torch.stack(guards).max()
    return value

def make_noise(h,n,noise,device):
    g=torch.Generator(device=device).manual_seed(noise)
    return torch.rand(h,n,device=device,generator=g),torch.randn(h,n,device=device,generator=g)

@torch.no_grad()
def full_replay(head,d,noise):
    grid=d['grid'];h,n=grid.shape;device=grid.device
    u,z=make_noise(h,n,noise,device)
    history=torch.full((n,20),float('nan'),device=device)
    gaps=torch.zeros(h,n,device=device);clocks=torch.zeros(h,n,device=device)
    for t in range(1,h):
        ix=grid[t].clamp(min=0);pos=torch.full((n,),t,device=device)
        clock,valid=clock_tensor(history,pos,d['fallback']);clocks[t]=clock[:,0]
        logits,mu,sd=head(d['base'][ix],d['phase'][ix],clock,valid)
        gap=implicit_sample(logits,mu,sd,u[t],z[t]);gaps[t]=gap
        history=torch.cat([history[:,1:],gap[:,None]],1)
    return gaps,clocks

def full_measures(d,gaps,clocks):
    keep=d['position']>0;t=d['position'][keep];j=d['customer'][keep]
    return measures(torch.log1p(d['gap'][keep]),d['clock'][keep,0],
        torch.log1p(gaps[t,j]),clocks[t,j],d['phase'][keep],t,j)

def choose_chunks(d,rng):
    grid=d['grid'].cpu().numpy();length=(grid>=0).sum(0);customers=[];starts=[]
    for i in range(CFG['chunks']):
        if i>=CFG['chunks']//2:
            pool=d['pools'][1];index=int(pool[int(rng.integers(len(pool)))])
            j=int(d['customer'][index]);anchor=int(d['position'][index])
            start=max(1,anchor-int(rng.integers(CFG['chunk_length'])))
        else:
            lo,hi=BINS[i%len(BINS)];eligible=np.flatnonzero(length>lo)
            if not len(eligible):eligible=np.flatnonzero(length>6);lo=6
            j=int(rng.choice(eligible));end=min(int(length[j])-1,hi-1 if hi else int(length[j])-1)
            anchor=int(rng.integers(lo,end+1));start=max(1,anchor-CFG['chunk_length']//2)
        customers.append(j);starts.append(start)
    return torch.tensor(customers,device=d['grid'].device),torch.tensor(starts,device=d['grid'].device)

def chunk_replay(head,d,prefix,customers,starts,u,z):
    n=len(customers);device=starts.device
    offsets=torch.arange(-20,0,device=device)
    history_pos=starts[:,None]+offsets[None]
    history=prefix[history_pos.clamp(min=0),customers[:,None]]
    history=torch.where(history_pos>0,history,torch.full_like(history,float('nan'))).detach()
    columns=[]
    for k in range(CFG['chunk_length']):
        t=starts+k;exists=t<d['grid'].shape[0];ix=d['grid'][t.clamp(max=d['grid'].shape[0]-1),customers]
        keep=exists&(ix>=0);ix=ix.clamp(min=0)
        clock,valid=clock_tensor(history,t,d['fallback'])
        logits,mu,sd=head(d['base'][ix],d['phase'][ix],clock,valid)
        gap=implicit_sample(logits,mu,sd,u[k],z[k])
        columns.append((ix[keep],gap[keep],clock[keep,0],t[keep],customers[keep]))
        history=torch.cat([history[:,1:],gap[:,None]],1)
    ix,gap,clock,pos,ids=[torch.cat([row[j] for row in columns]) for j in range(5)]
    return measures(torch.log1p(d['gap'][ix]),d['clock'][ix,0],torch.log1p(gap),clock,d['phase'][ix],pos,ids)

def conditional_loss(head,d):
    ix=torch.cat([p[torch.randint(len(p),(CFG['nll_batch']//4,),device=p.device)] for p in d['pools']])
    ls=head.nll(d['base'][ix],d['phase'][ix],d['clock'][ix],d['valid'][ix],d['gap'][ix],[0.,1045691.],1.)
    population=torch.stack([len(p)*torch.ones((),device=ix.device) for p in d['pools']]);population/=population.sum()
    phase_means=ls.reshape(4,-1).mean(1)
    # Explicit .5 natural event likelihood + .5 phase-macro likelihood.
    return .5*(phase_means.mean()+(population*phase_means).sum())

@torch.no_grad()
def internal(head,d,refs,joint):
    rows=[]
    for noise in CFG['validation_noises']:
        gaps,clock=full_replay(head,d,noise);m=full_measures(d,gaps,clock)
        rows.append(dict(noise=noise,objective=float(objective(m,refs[noise],joint)),
                         **{k:float(v) for k,v in m.items()}))
    return float(np.mean([r['objective'] for r in rows])),rows

def fit(arm):
    verify();seed(CFG['fit_seed']);device=torch.device('cuda:0');assert torch.cuda.is_available()
    run=OUT/arm;run.mkdir(exist_ok=False);started=time.monotonic()
    d=load_data('optimization',device);val=load_data('internal_validation',device)
    head=new_head(d,arm,device);frozen=copy.deepcopy(head).eval().requires_grad_(False)
    params=[p for n,p in head.named_parameters() if n.startswith('residual.')]
    optimizer=torch.optim.AdamW(params,lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    joint=arm.endswith('_joint');rng=np.random.default_rng(CFG['fit_seed']);curve=[]
    write(run/'START.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),arm=arm,
        started_utc=datetime.now(timezone.utc).isoformat(),trainable_parameters=sum(p.numel() for p in params)))
    references={}
    for noise in CFG['validation_noises']:
        g,c=full_replay(frozen,val,noise);references[noise]=full_measures(val,g,c)
    best,rows=internal(head,val,references,joint);best_state=copy.deepcopy(head.state_dict());selected=0
    curve.append(dict(step=0,stage='identity',seconds=time.monotonic()-started,score=best,internal=rows))
    print('DENSITY_IDENTITY',arm,curve[-1],flush=True)
    for step in range(1,CFG['nll_steps']+1):
        loss=conditional_loss(head,d);assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,CFG['gradient_clip'],error_if_nonfinite=True);optimizer.step()
    # NLL warm start is common to both objectives and is evaluated, not
    # automatically adopted. The identity remains a selectable checkpoint.
    score,rows=internal(head,val,references,joint)
    curve.append(dict(step=CFG['nll_steps'],stage='likelihood',seconds=time.monotonic()-started,score=score,internal=rows))
    if score<best:best,selected,best_state=score,CFG['nll_steps'],copy.deepcopy(head.state_dict())
    print('DENSITY_NLL',arm,curve[-1],flush=True)
    initial_prefix,_=full_replay(frozen,d,CFG['training_noise'])
    for update in range(1,CFG['rollout_steps']+1):
        if update==1 or (update-1)%CFG['prefix_refresh_every']==0:
            prefix,_=full_replay(head,d,CFG['training_noise'])
        customers,starts=choose_chunks(d,rng)
        u,z=make_noise(CFG['chunk_length'],len(customers),CFG['training_noise']+update,device)
        with torch.no_grad():reference=chunk_replay(frozen,d,initial_prefix,customers,starts,u,z)
        m=chunk_replay(head,d,prefix,customers,starts,u,z)
        loss=objective(m,reference,joint)+CFG['nll_weight']*conditional_loss(head,d)
        assert torch.isfinite(loss),(arm,update)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,CFG['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        if update%CFG['validation_every']:continue
        score,rows=internal(head,val,references,joint);step=CFG['nll_steps']+update
        curve.append(dict(step=step,stage='recursive_time',seconds=time.monotonic()-started,score=score,
                          training_loss=float(loss.detach()),internal=rows))
        if score<best:best,selected,best_state=score,step,copy.deepcopy(head.state_dict())
        write(run/'LEARNING_CURVE.json',curve);print('DENSITY_FIT',arm,curve[-1],flush=True)
    head.load_state_dict(best_state)
    original=torch.load(BASELINE,map_location='cpu',weights_only=True)
    for k,v in original['state_dict'].items():torch.testing.assert_close(head.state_dict()[k].cpu(),v,rtol=0,atol=0)
    payload={**original,'dim':head.dim,'arm':arm,'state_dict':{k:v.cpu() for k,v in head.state_dict().items()},
             'parent_seed':CFG['parent_seed'],'fit_seed':CFG['fit_seed'],'selected_step':selected}
    torch.save(payload,run/'time_head.pt');write(run/'LEARNING_CURVE.json',curve)
    write(run/'FIT_COMPLETE.json',dict(selected_step=selected,selected_identity=selected==0,
          internal_objective=best,seconds=time.monotonic()-started,sha256=digest(run/'time_head.pt'),
          original_buffers_exact=True,new_native_argn_fits=0,independent_head_fits=1,test_events_read=False))
    verify();print('DENSITY_FIT_COMPLETE',arm,selected,best,flush=True)

def generate(arm):
    verify();import run_argn_time_density as temporal
    from run_argn_amount_learning import inputs,load_parent
    device=torch.device('cuda:0');fs=CFG['parent_seed'];run=OUT/arm
    assert (run/'FIT_COMPLETE.json').exists()
    _,codec,frames,metas=inputs();ws,native=load_parent(fs,device)
    with patch.object(temporal,'time_density_generation',residual_generation):
        temporal.integration(ws,native,frames['development'],metas['development'],codec,fs,run/'time_head.pt',run)
        del native,frames,metas;torch.cuda.empty_cache()
        with patch.object(temporal,'OUT',OUT),patch.object(temporal,'DOCS',DOCS),patch.object(temporal,'CFG',CFG):
            temporal.generate_arm(ws,fs,arm,run/'time_head.pt',device)
    # Inherited runner has a historical hard-coded count; explicit correction
    # is stored separately instead of treating that field as evidence.
    write(run/'NATIVE_COMPLETE.json',dict(generations=len(CFG['generation_seeds']),parent_seed=fs,
          inherited_complete_count_is_legacy=True,test_events_read=False))
    verify()

def checks():
    d=torch.load(OUT/'cache/internal_validation.pt',map_location='cpu',weights_only=True)
    old=torch.load(BASELINE,map_location='cpu',weights_only=True)
    native=InitialClock(1,'init_only').double();native.load_state_dict(old['state_dict'])
    head=new_head(d,'mixture_joint','cpu').double()
    ix=torch.arange(512);base=d['base'][ix].double();phase=d['phase'][ix];clock=d['clock'][ix].double();valid=d['valid'][ix].double()
    for a,b in zip(native(base,phase,clock,valid),head(base,phase,clock,valid)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Verify clock parity for even medians, clipping, zero intervals and empty
    # history, and check nonzero gradients reach logits and scale parameters.
    codec=PastClock(old['fallback']);memory=codec.empty(4);history=torch.full((4,20),float('nan'),dtype=torch.double)
    rng=np.random.default_rng(27);count=0
    memory=codec.advance(memory,np.zeros(4))
    for t in range(1,65):
        want,wv=codec.features(memory)
        got,gv=clock_tensor(history,torch.full((4,),t),old['fallback'])
        np.testing.assert_allclose(got[:,0].numpy(),want,atol=2e-15,rtol=0);np.testing.assert_array_equal(gv.numpy(),wv)
        gap=rng.choice([0.,1.,20.,1e5,1e6],4);memory=codec.advance(memory,gap)
        history=torch.cat([history[:,1:],torch.tensor(gap)[:,None]],1);count+=4
    logits=torch.tensor([[.1,-.4,.3]],dtype=torch.double,requires_grad=True)
    mu=torch.tensor([[7.,9.,11.]],dtype=torch.double,requires_grad=True)
    sd=torch.tensor([[.8,.6,1.]],dtype=torch.double,requires_grad=True)
    sampled=implicit_sample(logits,mu,sd,torch.tensor([.31]),torch.tensor([.2]),False)
    derivatives=torch.autograd.grad(torch.log1p(sampled).sum(),(logits,mu,sd))
    z0=torch.log1p(sampled).detach().item()
    def cdf(z,l,m,s):return float((l.softmax(-1)*torch.special.ndtr((z-m)/s)).sum())
    from scipy.optimize import brentq
    u=cdf(z0,logits.detach(),mu.detach(),sd.detach());errors=[]
    for group in range(3):
        for k in range(3):
            plus=[x.detach().clone() for x in [logits,mu,sd]];minus=[x.clone() for x in plus]
            plus[group][0,k]+=1e-5;minus[group][0,k]-=1e-5
            delta=(brentq(lambda z:cdf(z,*plus)-u,-20,40)-brentq(lambda z:cdf(z,*minus)-u,-20,40))/2e-5
            errors.append(abs(delta-float(derivatives[group][0,k])))
    assert max(errors)<2e-6,errors
    with torch.no_grad():head.residual[-1].bias.fill_(.3)
    early=d['position'][ix].le(5)
    for a,b in zip(native(base,phase,clock,valid),head(base,phase,clock,valid)):
        torch.testing.assert_close(a[early],b[early],rtol=0,atol=0)
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(identity_law_exact=True,clock_checks=count,
          first5_laws_preserved=True,implicit_gradient_max_error=max(errors),quantization_gradient_is_surrogate=True,
          native_history_in_training_is_teacher_forced=True))
    print('IMPLEMENTATION_CHECK',max(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['checks','fit','generate']);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    checks() if a.mode=='checks' else fit(a.arm) if a.mode=='fit' else generate(a.arm)
