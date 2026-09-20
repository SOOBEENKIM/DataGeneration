import numpy as np
import pandas as pd
import pytest
from benchmarks.cs_saf_external import fit_state,evaluate,gap_bin
from generators.cs_saf_external_empirical import ObservedTransition
from scripts.run_cs_saf_external_port import calibration_fit


def sample(name):
    rows=[]
    for i in range(8):
        for t,stamp in enumerate([0,0,1,2,3,3,4,5]):
            rows.append(dict(entity_id=i,event_index=t,timestamp=stamp,
                gap=np.nan if t==0 else stamp-[0,0,1,2,3,3,4,5][t-1],
                receiver_or_mark=['A','B','A','A','B','B','A','B'][t],
                category=['cat1','cat2','cat1','cat1','cat2','cat2','cat1','cat2'][t],
                transaction_type='in' if t%2 else 'out',amount_or_numeric_value=t+1.))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('name',['berka','sparkov'])
def test_identity_metrics_and_true_zero_gap(name):
    frame=sample(name);state=fit_state(frame,name);score=evaluate(frame,frame,state)
    for k,v in score.items():
        if k.endswith(('_tv','_ks')):assert v==0
    assert gap_bin([np.nan,-1,0,.01,1],[1,2]).tolist()==[-1,-2,0,1,1]


def test_berka_daily_metric_ignores_unknown_within_day_order():
    frame=sample('berka');state=fit_state(frame,'berka')
    changed=frame.sort_values(['entity_id','timestamp','event_index'],ascending=[True,True,False]).copy()
    changed['event_index']=changed.groupby('entity_id').cumcount()
    changed['gap']=changed.groupby('entity_id').timestamp.diff()
    score=evaluate(frame,changed,state)
    assert score['daily_gap_mark_joint_tv']==0 and score['daily_gap_count_joint_tv']==0
    assert score['gap_root_transition_joint_tv']>0


@pytest.mark.parametrize('name',['berka','sparkov'])
def test_empirical_model_probabilities_and_full_length_feedback(name):
    frame=sample(name);state=fit_state(frame,name)
    model=ObservedTransition(name,state['gap_edges']).fit(frame)
    np.testing.assert_allclose(model.p2.sum(-1),1)
    np.testing.assert_allclose(model.emission.sum(-1),1)
    unknown=frame.copy();unknown.loc[1,'receiver_or_mark']='unseen'
    assert np.isfinite(model.nll(unknown))
    plan=pd.DataFrame({'entity_id':['x','y'],'length':[65,101]})
    a=model.sample(plan,5);b=model.sample(plan,5)
    pd.testing.assert_frame_equal(a,b)
    assert a.groupby('entity_id').size().tolist()==[65,101]
    assert a.loc[a.event_index.eq(0),'gap'].isna().all()
    np.testing.assert_allclose(a.groupby('entity_id').timestamp.diff(),a.gap,equal_nan=True)


def test_bounded_calibration_improves_fit_objective():
    a=np.column_stack([np.full(100,-2.),np.r_[np.zeros(60),np.ones(40)],np.ones(100)])
    r=calibration_fit(a,[.5,1.5,2,3],{'penalty':.001,'maxiter':100,'bounds':[-5,5]})
    assert r['success'] and r['fit_objective_after']<r['fit_objective_before']
    assert len(r['beta'])==6 and max(abs(np.array(r['beta'])))<=5
