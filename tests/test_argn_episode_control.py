import numpy as np
import pandas as pd
import torch
from benchmarks.argn_episode_control import episode_metadata, fit_episode, episode_probabilities, first_probability, decode_length
from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX


def fixture():
    rows=[]
    for i in range(40):
        labels=([1,1,1] if i<20 else [0,0,1,1,0,0,0,0])
        age=0
        for t,y in enumerate(labels):
            age=0 if t==0 else (age+1 if t>1 and labels[t-1]==labels[t-2] else 1)
            rows.append(dict(record=i,event_index=t,label=y,previous_label=labels[t-1] if t else -1,prior_age=age))
    return pd.DataFrame(rows)


def parameters():
    return dict(previous_label={'-1':.5,'0':.1,'1':.8},age_hazard={})


def test_past_episode_features_do_not_see_current_or_future_label():
    m=fixture();a=episode_metadata(m);changed=m.copy()
    changed.loc[(changed.record==20)&(changed.event_index>=2),'label']=0
    b=episode_metadata(changed)
    mask=(m.record==20)&(m.event_index<=2)
    assert a.loc[mask,['ever_fraud','initial_fraud']].equals(b.loc[mask,['ever_fraud','initial_fraud']])
    assert a.loc[(m.record==0)&(m.event_index==0),'initial_fraud'].iloc[0]==0


def test_transition_is_not_hard_one_episode_and_boundary_is_not_termination():
    m=fixture();p=fit_episode(m,parameters());q=episode_probabilities(m,p,True)
    assert ((q>0)&(q<1)).all()
    # Zero repeat onsets remain possible with smoothing; no hard zero.
    after=m.record.ge(20)&m.event_index.ge(5)
    assert (q[after]>0).all() and (q[after]<.1).all()
    counts=p['exposure_counts']
    # No fictitious outcome after the last recorded event.
    assert sum(x['events'] for x in counts)==len(m)-m.record.nunique()
    assert sum(x['events']-x['frauds'] for x in counts if x['previous']==1 and x['initial']==1)==0


def test_initial_state_depends_on_learned_length_and_removal_is_pooled():
    m=fixture();p=fit_episode(m,parameters())
    q=first_probability(np.array([3,8]),p['first_tree'])
    assert q[0]>.9 and q[1]<.1
    removed=episode_probabilities(m,p,False)
    assert np.all(removed[m.event_index.eq(0)]==.5)


def test_structural_length_decoding_and_minimum_match_engine():
    torch.testing.assert_close(decode_length({SLEN_SUB_COLUMN_PREFIX+'E0':torch.tensor([5,0]),
        SLEN_SUB_COLUMN_PREFIX+'E1':torch.tensor([2,0])},1),torch.tensor([25,1]))
