import copy
import numpy as np
import torch
from data.cof_seqgen_saf_tensorizer import NumericCodec
from data.cs_saf_external import TargetWindows
from models.cs_saf_binned_amount import BinnedAmountD,fit_amount_pool,pool_tail_matrix
from tests.test_cs_saf_external_controls import support,KWARGS,make,sequences

torch.set_num_threads(1)


def fixture():
    raw=np.r_[0.,0.,np.repeat(1.,10),np.arange(2.,102.),1000.,10000.]
    codec=NumericCodec.fit(raw,transform='signed_log1p_zscore')
    pool=fit_amount_pool(raw,codec)
    ast=dict(codec_mean=codec.mean,codec_scale=codec.scale,log_mean=1.,log_scale=2.,
             initial_locations=[-1.,0.,1.],zero_rate=2/len(raw))
    model=BinnedAmountD(support(),ast,pool,**KWARGS)
    return raw,codec,pool,model


def test_bins_pool_boundaries_zero_duplicates_and_known_expectation():
    raw,codec,pool,m=fixture()
    codes=m.amount_codes(torch.from_numpy(codec.encode(raw))).numpy()
    assert np.all(codes[raw==0]==0) and np.all(codes[raw>0]>0)
    assert (pool['counts']>0).all()
    for k,(s,n) in enumerate(zip(pool['starts'],pool['counts'])):
        np.testing.assert_array_equal(np.sort(raw[codes==k]),np.sort(pool['values'][s:s+n]))
    cuts=np.array([0.,10.,100.,1000.,100000.])
    actual=pool['counts']/len(raw) @ pool_tail_matrix(pool,cuts)
    np.testing.assert_allclose(actual,(raw[:,None]>cuts).mean(0),atol=1e-15)
    p=torch.tensor(np.log(pool['counts']/len(raw)),dtype=torch.float32)[None,:]
    mean=m.amount_point(p).item()*codec.scale+codec.mean
    assert abs(mean-np.log1p(raw).mean())<1e-6
    # Outside fit support is mapped to edge bins for bin CE only, not given raw
    # probability mass by the empirical sampler.
    assert int(m.amount_codes(torch.from_numpy(codec.encode([1e9]))))==len(pool['counts'])-1


def test_sampling_matches_empirical_fit_mass_and_exact_history_values():
    raw,codec,pool,m=fixture();n=60000
    torch.manual_seed(9)
    z,a=m.sample_amount(torch.zeros(n,16),torch.ones(n),torch.full((n,),3))
    assert set(a.tolist())<=set(raw.tolist())
    assert abs(float(a.eq(0).double().mean())-np.mean(raw==0))<.003
    assert abs(float((a>100).double().mean())-np.mean(raw>100))<.003
    np.testing.assert_array_equal(z.numpy(),codec.encode(a.numpy()))


def test_shared_initial_nonamount_outputs_and_strict_past_gradient():
    raw,codec,pool,m=fixture();parent=make('D')
    keys=m.initialize_from_direct(parent.state_dict())
    assert all(torch.equal(m.state_dict()[k],parent.state_dict()[k]) for k in keys)
    batch=TargetWindows(sequences()).batch([0,1,31,36,37,101])
    before=m.target_outputs(**batch);old=parent.target_outputs(**batch)
    torch.testing.assert_close(before['logmark'],old['logmark'],rtol=0,atol=0)
    changed=copy.deepcopy(batch);row=torch.arange(len(batch['target_position']));pos=batch['target_position']
    changed['numeric_value'][row,pos]=123
    changed['auxiliary_categorical'][0][row,pos]=1
    after=m.target_outputs(**changed)
    torch.testing.assert_close(before['location'],after['location'],rtol=0,atol=0)
    m.compute_loss(**batch)['loss'].backward()
    assert m.value_head.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_positive_constant_pool_and_boundary_rounding():
    raw=np.r_[np.repeat(.01,10),np.repeat(4.,100),np.repeat(4.00000001,10)]
    c=NumericCodec.fit(raw,transform='signed_log1p_zscore');p=fit_amount_pool(raw,c)
    assert not p['zero_bin'] and (p['counts']>0).all()
    p=fit_amount_pool(np.ones(100),NumericCodec.fit(np.ones(100),transform='signed_log1p_zscore'))
    assert len(p['counts'])==1 and p['counts'][0]==100


def test_long_rollout_uses_each_sampled_amount_in_next_history():
    _,codec,pool,m=fixture();m.eval();seen=[];returned=[]
    original=m.sample_amount
    def sample(h,gap,mark):
        z,a=original(h,gap,mark);returned.append((z.detach().clone(),a.detach().clone()));return z,a
    def hook(module,args):
        seen.append(tuple(a.detach().clone() if isinstance(a,torch.Tensor) else a for a in args[:4]))
    m.sample_amount=sample
    handle=m.encoder.register_forward_pre_hook(hook)
    torch.manual_seed(31)
    out=m.sample_fixed_lengths([41,37],static=torch.tensor([[0.],[1.]]),static_categorical=(torch.tensor([3,4]),))
    handle.remove()
    for step,(g,mark,value,valid) in enumerate(seen):
        rows=np.array([i for i,n in enumerate((41,37)) if step<n])
        start=max(0,step-31)
        if step:
            expected=out['numeric_value'][rows,start:step]
            torch.testing.assert_close(value[:,:-1],expected,rtol=0,atol=0)
        torch.testing.assert_close(out['raw_amount'][rows,step],returned[step][1],rtol=0,atol=0)
    assert out['receiver'].shape==(2,41)
