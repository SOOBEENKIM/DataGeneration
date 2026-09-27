import numpy as np
import pandas as pd
from diagnose_argn_residual import decorate, standardization, AMOUNT


def example():
    return pd.DataFrame(dict(entity_id=[0]*9+[1]*2,event_index=list(range(9))+[0,1],
        event_is_fraud=[0,0,0,0,0,1,1,0,0,1,1],gap=[1]*11,
        amount_or_numeric_value=[10,10,10,10,10,100,200,20,30,90,80],category=['a']*11))


def test_strict_past_and_customer_boundaries():
    d=decorate(example())
    assert d.loc[5,'phase']=='onset' and d.loc[6,'phase']=='continuation'
    assert d.loc[7,'phase']=='return_normal' and d.loc[8,'phase']=='normal_stay'
    assert d.loc[9:10,'phase'].tolist()==['left_fraud','left_fraud']
    assert d.loc[5,'past_amount']==10 and d.loc[5,'past_normal_amount']==10
    assert d.loc[7,'past_normal_amount']==10 and np.isnan(d.loc[9,'past_normal_amount'])
    assert np.isnan(d.loc[0,'gap']) and np.isnan(d.loc[9,'gap'])
    changed=example();changed.loc[5:,AMOUNT]=500
    q=decorate(changed)
    assert q.loc[:5,'past_amount'].equals(d.loc[:5,'past_amount'])
    assert q.loc[:5,'past_normal_amount'].equals(d.loc[:5,'past_normal_amount'])


def test_standardization_reports_missing_phase_mass():
    r=decorate(example());s=r[~r.phase.eq('onset')].copy()
    for d in [r,s]:
        d['past_amount_band']=0;d['past_gap_band']=0
    cells,rows=standardization(r,s,{})
    gap=next(x for x in rows if x['field']=='gap' and x['conditioning']=='phase')
    assert 0<gap['real_coverage']<1
    assert gap['common_support_standardized_w1']==0
    assert gap['supported_real_mass']==0
