from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from evaluate_sparkov_argn_control import extended,ratio_ci,numeric_metrics

STATE={'gap_edges':[0,5,60,300],'amount_edges':[10,100,1000],'position_edges':[0,31,99,499]}

def frame(labels):
    rows=[]
    for customer,values in enumerate(labels):
        for i,value in enumerate(values):
            rows.append(dict(entity_id=customer,event_index=i,gap=np.nan if i==0 else 10.,receiver_or_mark='m',amount_or_numeric_value=50.,category='c',event_is_fraud=str(value)))
    return pd.DataFrame(rows)

def test_onset_continuation_never_cross_customer_boundaries():
    d=extended(frame([[1,1,0],[0,1],[0]]),STATE)
    assert len(d[(d.fraud==1)&(d.previous_fraud==0)])==1
    assert len(d[(d.fraud==1)&(d.previous_fraud==1)])==1
    assert d.loc[d.event_index.eq(0),'previous_fraud'].eq(-2).all()
    assert d.loc[d.event_index.eq(0),'seen_merchant'].eq(0).all()

def test_future_amount_does_not_enter_past_history_features():
    original=frame([[0]*30]);changed=original.copy();changed.loc[20,'amount_or_numeric_value']=1e8
    a=extended(original,STATE);b=extended(changed,STATE)
    pd.testing.assert_frame_equal(a.loc[:19,['amount_vs_history','seen_merchant']],b.loc[:19,['amount_vs_history','seen_merchant']])
    assert b.loc[20,'amount_vs_history']==2

def test_customer_bootstrap_keeps_whole_correlated_groups():
    d=extended(frame([[0]*1000,[1]]),STATE)
    result=ratio_ci(d,[0,1],repeats=500)
    assert result['fraud_rate']==1/1001
    # Whole-customer resampling can select only either customer. IID row
    # resampling would incorrectly produce a narrow interval around 1/1001.
    assert result['ci_low']==0 and result['ci_high']==1

def test_raw_scale_detects_distortion_hidden_inside_one_quantile_bin():
    original=frame([[1]*12]);changed=original.copy()
    original['amount_or_numeric_value']=2000.;changed['amount_or_numeric_value']=12000.
    a=extended(original,STATE);b=extended(changed,STATE)
    assert a.amount_bin.equals(b.amount_bin)
    result=[r for r in numeric_metrics(a,b,'fixture') if r['field']=='amount' and r['label']=='fraud'][0]
    assert result['wasserstein_raw']==10000. and result['ks_distance']==1.

def test_fraud_run_ages_reset_after_normal_and_at_customer_boundary():
    from diagnose_sparkov_fraud_runs import fraud_runs
    d=fraud_runs(extended(frame([[1,1,0,1],[1,1],[0,1,0]]),STATE))
    assert d.fraud_age.tolist()==[1,2,1,1,2,1]
    assert d.groupby('run_id').size().tolist()==[2,1,2,1]
    assert d.loc[d.last_in_customer,['entity_id','event_index']].values.tolist()==[[0,3],[1,1]]
