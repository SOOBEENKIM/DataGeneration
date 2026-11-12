"""Scientific adapter parity/deletion checks, independent of fit outcomes."""
import copy
import numpy as np
import torch
from experiment import ROOT,OUT,DOCS,write,parameters,initial_parameters
from validation_head import factory,HistoryClock
from initial_clock import InitialClock
from benchmarks.argn_time_density import TimeDensity

def main():
    torch.manual_seed(919)
    p=torch.load(ROOT/'artifacts/argn_clock_initial_coupling_v1/init_only/time_head.pt',map_location='cpu',weights_only=True)
    old=InitialClock(p['dim'],p['arm']).double();old.load_state_dict(p['state_dict'])
    new=factory(p['dim'],p['arm']).double();new.load_state_dict(p['state_dict'])
    phase=torch.arange(400)%4;n=torch.arange(400)%100
    x=torch.stack([torch.linspace(5,14,400),n.double().log1p()],1);v=n.ge(6).double();base=torch.randn(400,1,dtype=torch.double)
    a=old(base,phase,x,v);b=new(base,phase,x,v)
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    no_init=copy.deepcopy(new);no_init.use_initial.fill_(False)
    changed=no_init(base,phase,x,v);untouched=~(phase.eq(0)&n.ge(1)&n.le(5))
    for aa,bb in zip(a,changed):torch.testing.assert_close(aa[untouched],bb[untouched],rtol=0,atol=0)
    no_cal=copy.deepcopy(new);no_cal.correction.zero_();changed=no_cal(base,phase,x,v)
    for aa,bb in zip(a,changed):torch.testing.assert_close(aa[phase.ne(0)],bb[phase.ne(0)],rtol=0,atol=0)
    hist=TimeDensity(13,'history_only').double();wrapper=HistoryClock(13,'history_only').double()
    wrapper.history_head.load_state_dict(hist.state_dict());base=torch.randn(400,13,dtype=torch.double)
    a=hist(base,phase,x[:,0],v);b=wrapper(base,phase,x,v)
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    wrapper.correction.copy_(torch.linspace(-.3,.3,6));wrapper.clock_limits.copy_(torch.tensor([7.,12.]))
    c=wrapper(base,phase,x,v)
    for i in [0,2]:torch.testing.assert_close(b[i],c[i],rtol=0,atol=0)
    torch.testing.assert_close(b[1][phase.ne(0)],c[1][phase.ne(0)],rtol=0,atol=0)
    assert (c[2]>0).all() and all(torch.isfinite(t).all() for t in c)
    write(DOCS/'PREFLIGHT.json',dict(frozen_init_laws_exact=True,history_wrapper_identity_exact=True,
        initial_removal_other_laws_exact=True,correction_removal_other_phases_exact=True,
        history_calibration_preserves_weights_scales=True,positions=400,new_fits=0))
    print('PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
