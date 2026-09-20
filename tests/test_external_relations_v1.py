import numpy as np
import pandas as pd
from experiments.cs_saf_external_relations_v1 import (
    ClusterSummary, EmpiricalDiagnostics, gap_codes, make_edges, partition, sequence_features,
)


def frame():
    return pd.DataFrame(dict(entity_id=[1]*6+[2]*2,event_id=list(range(8)),
        timestamp=[0,1,1,2,3,4,0,1],mark=['a','a','b','a','a','a','b','b'],
        amount=[1.]*8,category=['cat']*8))


def test_no_cross_entity_or_future_repeat():
    x=sequence_features(frame(),[1.])
    assert not x.loc[0,'has_previous'] and not x.loc[6,'has_previous']
    assert x.loc[7,'prior_run']==1 and x.loc[7,'repeat']==1
    assert x.loc[5,'prior_run']==2


def test_tied_endpoints_are_both_excluded():
    x=sequence_features(frame(),[1.])
    assert x.index[x.unambiguous].tolist()==[4,5,7]
    y=sequence_features(frame(),[1.],'reverse')
    assert x.loc[x.unambiguous,'repeat'].tolist()==y.loc[y.unambiguous,'repeat'].tolist()


def test_gap_edges_missing_zero_and_duplicate_quantiles():
    edges=make_edges([np.nan,0,1,1,1,2],[.2,.4,.6,.8])
    assert np.array_equal(edges,np.unique(edges))
    assert gap_codes([np.nan,0,1,20],[1]).tolist()==[-1,0,1,2]


def test_partition_disjoint_complete_deterministic():
    a,b=partition(range(20),'berka',42,.8)
    c,d=partition(reversed(range(20)),'berka',42,.8)
    assert (a,b)==(c,d) and not a&b and a|b==set(range(20)) and len(a)==16


def test_cluster_uses_entity_resampling_and_event_weighting():
    c=ClusterSummary([1,1,2],500,1)
    result=c.mean([1,1,2],[0.,0.,1.])
    assert result['value']==1/3 and result['entity_mean']==.5
    assert result['entities']==2 and result['ci_low']<=1/3<=result['ci_high']


def test_hierarchical_tables_unknown_backoff_and_no_label_input():
    x=sequence_features(frame(),[1.])
    x=x.loc[x.unambiguous].copy()
    m=EmpiricalDiagnostics(['a','b'],3).fit(x)
    expected=(3+.5)/(3+3*.5) # three current marks: a,a,b -> use explicit counts below
    assert np.isclose(m.p0[1],(2+.5)/(3+3*.5))
    np.testing.assert_allclose(m.p1[0],m.p0)
    np.testing.assert_allclose(m.p2[0,0],m.p1[0])
    x['event_is_fraud']=1
    a=m.losses(x)
    x['event_is_fraud']=0
    b=m.losses(x)
    for k in a: np.testing.assert_array_equal(a[k],b[k])
    x['mark']='unseen';x['previous_mark']='unseen'
    assert all(np.isfinite(v).all() for v in m.losses(x).values())


def test_current_action_does_not_change_prior_run():
    x=frame();a=sequence_features(x,[1.])
    x.loc[5,'mark']='b';b=sequence_features(x,[1.])
    assert a.loc[5,'prior_run']==b.loc[5,'prior_run']==2
    assert a.loc[5,'repeat']!=b.loc[5,'repeat']
