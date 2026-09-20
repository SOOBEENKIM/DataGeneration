"""Check diagnostic semantics against hand examples, not fitted scientific outcomes."""
import numpy as np
import pandas as pd
import torch

from experiments.cs_saf_history_diagnostic_v1 import events, bootstrap_weights, cluster_draws, summarize


def example():
    marks=torch.tensor([[3,4,4,4,5,5],[7,7,7,0,0,0]])
    lengths=torch.tensor([6,3]);valid=torch.arange(6)[None,:]<lengths[:,None]
    return dict(receiver=marks,gap=torch.zeros((2,6)),codes=torch.tensor([3,4]),valid_mask=valid,lengths=lengths)


def test_prefix_run_and_censoring_by_hand():
    x=example();f=events(x)
    assert f.run.tolist()==[1,1,2,3,1,1,2]
    assert f.history.tolist()==[1,2,3,4,5,1,2]
    assert f.y.tolist()==[0,1,1,0,1,1,1]
    assert len(f)==7 # no made-up nonrepeat when second sequence ends at run 3


def test_current_mark_and_future_do_not_define_current_run():
    x=example();before=events(x)
    x['receiver'][:,3:]=11
    after=events(x)
    pd.testing.assert_frame_equal(before.loc[before.event_index<=3,['history','run']],
        after.loc[after.event_index<=3,['history','run']])


def test_cluster_resampling_preserves_a_whole_sequence():
    f=pd.DataFrame(dict(entity=[0,0,1,2,3],group=[0,0,0,1,1]))
    w=bootstrap_weights(f,3)
    assert np.all(w[0][1].sum(1)==2)
    actual=cluster_draws(f,f.group.to_numpy()==0,np.array([0.,0.,1.,0.,0.]),0,w)
    weights=w[0][1]
    expected=weights[:,1]/(2*weights[:,0]+weights[:,1])
    np.testing.assert_array_equal(actual,expected)


def test_signed_error_identity_and_empty_cells_stay_missing():
    f=events(example());f['q']=.3;f['q_alt']=.31;f['p']=.4;f['mark_nll']=1.
    rows,boot=summarize(f,{},bootstrap_weights(f,7),True)
    for row in rows:
        if row['transitions']:
            assert abs(row['law_residual']-row['bias']-row['sampled_residual'])<1e-12
            assert abs(row['bias']-.1)<1e-12
            assert abs(row['squared_probability_error']-.01)<1e-12
        else:assert 'bias' not in row
    assert all(not r['adequate'] for r in rows)
