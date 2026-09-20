import numpy as np
import pandas as pd
import pytest
from experiments.cs_saf_generation_metrics import CachedGroupMetrics,conditional_tv
from experiments.cs_saf_generation_repeat_stats import repeat_statistics
from benchmarks.cof_seqgen_saf_metrics import evaluate_metric_suite


def events(offset=0,constant=False):
    rows=[]
    for e in range(6):
        for i in range(4):
            rows.append(dict(entity_id=f'e{e}',event_index=i,gap=np.nan if i==0 else (0 if constant else (e+i+offset)%5),
                receiver_or_mark='rare' if constant else f'm{(e+2*i+offset)%4}',
                amount_or_numeric_value=2. if constant else float((i+e+offset)%7)))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('constant',[False,True])
def test_cached_suite_matches_original_for_missing_bins_and_mark_groups(constant):
    train=events();reference=events(2);candidate=events(1,constant)
    cache=CachedGroupMetrics(train,reference)
    expected=evaluate_metric_suite(reference,candidate,cache.state,include_privacy=False)
    actual=cache.score(candidate)
    assert actual.keys()==expected.keys()
    for m in actual:assert actual[m]==pytest.approx(expected[m],abs=1e-12)


def test_empty_candidate_condition_has_full_tv_cost():
    assert conditional_tv(np.array([[2,0],[0,2]]),np.array([[2,0],[0,0]]))==.5


def test_nested_statistics_keep_five_units_and_separate_randomness():
    base=np.arange(5.,dtype=float)
    jitter=np.array([-2.,-1.,0.,1.,2.])
    data=base[:,None]+jitter[None,:]
    out=repeat_statistics(data)
    assert out['n_trials']==5
    np.testing.assert_array_equal(out['values'],base)
    assert out['mean']==2.
    assert out['mean_within_tape_variance']==2.5
    assert out['observed_between_trial_mean_variance']==2.5
    assert out['between_trial_variance_raw']==2.
    assert out['conditional_MC_standard_error']==pytest.approx(np.sqrt(.1))
    assert out['conditional_MC_degrees_of_freedom']==pytest.approx(20.)
    # A different order of tapes cannot change the trial summary.
    other=repeat_statistics(data[:,::-1])
    assert other['values']==out['values']


def test_zero_sampling_variance_does_not_erase_between_trial_uncertainty():
    out=repeat_statistics(np.repeat(np.arange(5.)[:,None],5,axis=1))
    assert out['conditional_MC_standard_error']==0
    assert out['conditional_MC_95_percent_interval']==[2.,2.]
    assert out['descriptive_95_percent_t_interval'][0]<2.
    assert out['descriptive_95_percent_t_interval'][1]>2.


def test_no_pseudoreplication_or_nonfinite_values():
    with pytest.raises(ValueError):repeat_statistics(np.zeros((25,1)))
    with pytest.raises(ValueError):repeat_statistics(np.full((5,5),np.nan))
