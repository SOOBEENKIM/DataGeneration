import numpy as np
import pandas as pd
from benchmarks.argn_fraud_audit import features, summaries, metric_state, risk_table, position_curves


def example():
    return pd.DataFrame(dict(entity_id=[1]*7+[2]*7,event_index=list(range(7))*2,
        gap=[np.nan,5,5,60,300,7200,5]*2,receiver_or_mark=['A','B','A','B','C','A','D']*2,
        category=['food','fuel','food','fuel','travel','food','travel']*2,
        amount_or_numeric_value=[10,10,20,10,50,40,100]*2,
        event_is_fraud=['0','0','0','0','1','0','1']*2))


def state(d):
    return metric_state(d,dict(gap_edges_seconds=[0,5,60,300,1800,7200,86400],position_edges=[0,31,99,499]))


def test_future_values_cannot_change_past_history():
    d=example();s=state(d);a=features(d,s)
    d.loc[d.event_index.eq(6),'amount_or_numeric_value']=999999
    d.loc[d.event_index.eq(6),'receiver_or_mark']='FUTURE'
    b=features(d,s)
    pd.testing.assert_frame_equal(a.loc[a.event_index.lt(6)],b.loc[b.event_index.lt(6)])
    # Rolling history uses previous values, excluding the current amount.
    assert a.loc[5,'amount_vs_history']==2
    assert a.loc[7,'seen_merchant']==0 and a.loc[7,'previous_category']=='<START>'


def test_labels_are_counted_not_string_concatenated_and_invalids_retained():
    d=example();s=state(d);out=summaries(d,d,s)
    assert out['generated_frauds']==4 and out['label_tv']==0 and out['merchant_category_tv']==0
    modified=d.copy();modified.loc[0,'event_is_fraud']='_RARE_'
    out=summaries(d,modified,s)
    assert out['invalid_generated_label_rate']==1/14 and out['label_tv']>0
    tab=risk_table(modified,s,'test')
    assert tab.events.sum()==14 and tab.invalid_labels.sum()==1


def test_native_nullable_numeric_dtype_matches_plain_numeric():
    d=example();s=state(d);native=d.copy()
    native['gap']=native.gap.astype('Float64')
    native['amount_or_numeric_value']=native.amount_or_numeric_value.astype('Float64')
    assert summaries(d,native,s)==summaries(d,d,s)


def test_first_event_and_customer_boundaries_and_pair_support():
    d=example();s=state(d);out=features(d,s)
    assert out.loc[out.event_index.eq(0),'gap_bin'].eq(-1).all()
    modified=d.copy();modified.loc[0,'category']='unseen'
    curves=position_curves(d,modified,d,s)
    first=next(x for x in curves if x['position_band']==0)
    assert first['real_events']==2 and first['generated_pair_absent_from_fit']==.5
    assert first['real_pair_absent_from_fit']==0
