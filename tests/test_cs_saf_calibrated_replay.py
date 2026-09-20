"""Mechanism checks, no new scientific outcome selection."""
import numpy as np
import pytest
import torch
from experiments.cs_saf_calibrated_replay import (load_models,load_path,selected,probabilities,
    evaluate_pair,check_replay,bin_means,decomposition,GROUPS)


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(1)
    models,inputs=load_models(.05,1,0,'cpu');data,_,_=load_path(.05,1,0,'Ecal',0,inputs)
    ids=torch.cat([torch.where(data['codes']==s)[0][:3] for s in (3,4)])
    return models,selected(data,ids)


def test_strict_past_current_mark_value_and_future_cannot_change_current_prediction(case):
    models,data=case
    for model in models.values():
        before,mask,_=probabilities(model,data)
        altered={k:v.clone() for k,v in data.items()}
        # Current gap is held, current mark/value and all later events change.
        step=1
        altered['receiver'][:,step:]=3+(altered['receiver'][:,step:]-3+17)%64
        altered['numeric_value'][:,step:]+=13.
        altered['gap'][:,step+1:]=.2
        after,_,_=probabilities(model,altered)
        rows,steps=torch.where(mask)
        torch.testing.assert_close(before[steps==step],after[steps==step],atol=0,rtol=0)


def test_full_probability_streaming_prefix_and_copy_formula_agree(case):
    models,data=case
    _,error,_=check_replay(models,data,'cpu',prefix=True)
    assert error<3e-6


def test_identical_predictors_have_zero_tv_and_repeat_difference(case):
    models,data=case
    same={'Ucal':models['Ucal'],'Ecal':models['Ucal']}
    out=evaluate_pair(same,data,'cpu',batch_size=2)
    np.testing.assert_array_equal(out['repeat'][0],out['repeat'][1])
    assert out['mark_tv'].max()==0


def test_chunking_preserves_predictions_and_tv_contraction(case):
    models,data=case
    a=evaluate_pair(models,data,'cpu',batch_size=2);b=evaluate_pair(models,data,'cpu',batch_size=6)
    np.testing.assert_allclose(a['repeat'],b['repeat'],atol=3e-6,rtol=0)
    np.testing.assert_allclose(a['mark_tv'],b['mark_tv'],atol=3e-6,rtol=0)
    assert np.max(np.abs(a['repeat'][1].astype(float)-a['repeat'][0])-a['mark_tv'])<3e-6


def test_empty_bin_convention_and_reference_weighting():
    counts,means=bin_means(np.array([0,0,2]),np.array([.2,.6,1.]),4)
    np.testing.assert_array_equal(counts,[2,0,1,0]);np.testing.assert_allclose(means,[.4,0,1.,0])
    assert np.array([.2,.3,.4,.1])@abs(means-np.array([.5,.5,.5,.5]))==pytest.approx(.42)


def test_nonlinear_decomposition_preserves_cancellation_and_signed_bins():
    # Large opposite-signed mapping/history terms must not become causal percentages.
    def block(own,cu,ce,emp,pu,pe):
        return dict(empirical_L1=own,expected_curve_L1={'Ucal':cu,'Ecal':ce},
            empirical_repeat=emp,predicted_repeat={'Ucal':pu,'Ecal':pe})
    scores={'Ucal':{g:block(.12,.10,.30,[.3,.4],[.2,.4],[.5,.6]) for g in GROUPS},
            'Ecal':{g:block(.14,.02,.11,[.4,.4],[.1,.4],[.2,.5]) for g in GROUPS}}
    d=decomposition(scores)['pooled']
    assert d['predictor_F']==pytest.approx(.145)
    assert d['source_H']==pytest.approx(-.135)
    assert d['empirical_remainder_S']==pytest.approx(.01)
    assert d['native_difference']==pytest.approx(.02)
