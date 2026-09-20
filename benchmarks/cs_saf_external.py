"""Common external empirical-distribution metrics, with no oracle targets."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def gap_bin(values, edges):
    v=np.asarray(values,float)
    return np.where(~np.isfinite(v),-1,np.where(v<0,-2,np.where(v==0,0,np.searchsorted(edges,v,side='left')+1)))


def daily_view(frame):
    # Calendar-day aggregation is order invariant, not a repair of native order.
    d=frame.copy();d['day']=np.floor(d.timestamp)
    days=d[['entity_id','day']].drop_duplicates().sort_values(['entity_id','day'])
    days['day_gap']=days.groupby('entity_id').day.diff()
    days['day_count']=d.groupby(['entity_id','day']).size().reindex(pd.MultiIndex.from_frame(days[['entity_id','day']])).to_numpy()
    return d.merge(days,on=['entity_id','day'],validate='many_to_one'),days


def fit_state(train,name):
    root='category' if name=='sparkov' else 'receiver_or_mark'
    positive=train.loc[train.gap.gt(0),'gap']
    daily,days=daily_view(train)
    state=dict(dataset=name,root=root,gap_edges=np.unique(positive.quantile([.2,.4,.6,.8])).tolist(),
        amount_edges=np.unique(np.log1p(train.amount_or_numeric_value).quantile(np.arange(.1,1,.1))).tolist(),
        roots=sorted(train[root].fillna('<MISSING>').astype(str).unique()),
        marks=sorted(train.receiver_or_mark.fillna('<MISSING>').astype(str).unique()),
        daily_gap_edges=np.unique(days.day_gap.dropna().quantile([.2,.4,.6,.8])).tolist(),
        daily_count_edges=np.unique(days.day_count.quantile([.2,.4,.6,.8])).tolist())
    return state


def features(frame,state):
    d=frame.sort_values(['entity_id','event_index'],kind='stable').copy()
    root=d[state['root']].fillna('<MISSING>').astype(str)
    mark=d.receiver_or_mark.fillna('<MISSING>').astype(str)
    d['root_code']=root.map({k:i+1 for i,k in enumerate(state['roots'])}).fillna(0).astype(int)
    d['mark_code']=mark.map({k:i+1 for i,k in enumerate(state['marks'])}).fillna(0).astype(int)
    d['previous_root']=d.groupby('entity_id').root_code.shift().fillna(-1).astype(int)
    d['gap_bin']=gap_bin(d.gap,state['gap_edges'])
    # Negative/nonfinite amounts stay visible as an invalid bin, never clipped.
    amounts=d.amount_or_numeric_value.to_numpy(float)
    good=np.isfinite(amounts)&(amounts>=0)
    transformed=np.zeros(len(d));transformed[good]=np.log1p(amounts[good])
    d['amount_bin']=np.where(good,np.searchsorted(state['amount_edges'],transformed,side='left'),-1)
    d['has_previous']=d.event_index.gt(0)
    ties=d.groupby(['entity_id','timestamp']).entity_id.transform('size')
    d['unambiguous']=d.has_previous & ties.eq(1) & ties.groupby(d.entity_id).shift().eq(1)
    return d


def tv(a,b,keys):
    if len(a)==0 or len(b)==0:return None
    x=a.groupby(keys,dropna=False).size()/len(a)
    y=b.groupby(keys,dropna=False).size()/len(b)
    x,y=x.align(y,fill_value=0)
    return float(abs(x-y).sum()/2)


def evaluate(reference,generated,state):
    real,syn=features(reference,state),features(generated,state)
    rp,sp=real.loc[real.has_previous],syn.loc[syn.has_previous]
    result={
        'gap_tv':tv(rp,sp,['gap_bin']), 'mark_tv':tv(real,syn,['mark_code']),
        'root_tv':tv(real,syn,['root_code']),
        'root_amount_joint_tv':tv(real,syn,['root_code','amount_bin']),
        'merchant_amount_joint_tv':tv(real,syn,['mark_code','amount_bin']),
        'gap_root_transition_joint_tv':tv(rp,sp,['gap_bin','previous_root','root_code']),
        'unambiguous_gap_root_transition_tv':tv(real.loc[real.unambiguous],syn.loc[syn.unambiguous],['gap_bin','previous_root','root_code']),
        'amount_ks':float(ks_2samp(real.amount_or_numeric_value,syn.amount_or_numeric_value,method='asymp').statistic),
        'length_ks':float(ks_2samp(real.groupby('entity_id').size(),syn.groupby('entity_id').size(),method='asymp').statistic),
        'generated_events':len(syn),'generated_entities':int(syn.entity_id.nunique()),
        'generated_max_length':int(syn.groupby('entity_id').size().max()),
        'unknown_mark_rate':float(syn.mark_code.eq(0).mean()),
        'unknown_root_rate':float(syn.root_code.eq(0).mean()),
        'invalid_amount_rate':float((syn.amount_or_numeric_value.lt(0)|~np.isfinite(syn.amount_or_numeric_value)).mean()),
        'invalid_gap_rate':float((sp.gap.lt(0)|~np.isfinite(sp.gap)).mean()),
        'noninteger_gap_rate':float((abs(sp.gap-sp.gap.round())>1e-6).mean()),
        'generated_unambiguous_fraction':float(sp.unambiguous.mean()),
    }
    if state['dataset']=='berka':
        rd,rday=daily_view(real);sd,sday=daily_view(syn)
        for frame in (rd,sd):frame['daily_gap_bin']=gap_bin(frame.day_gap,state['daily_gap_edges'])
        for frame in (rday,sday):
            frame['daily_gap_bin']=gap_bin(frame.day_gap,state['daily_gap_edges'])
            frame['daily_count_bin']=np.searchsorted(state['daily_count_edges'],frame.day_count,side='left')
        result['daily_gap_mark_joint_tv']=tv(rd.loc[rd.day_gap.notna()],sd.loc[sd.day_gap.notna()],['daily_gap_bin','mark_code'])
        result['daily_gap_count_joint_tv']=tv(rday.loc[rday.day_gap.notna()],sday.loc[sday.day_gap.notna()],['daily_gap_bin','daily_count_bin'])
    return result
