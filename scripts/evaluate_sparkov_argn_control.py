"""Direct distribution checks with customer-level uncertainty, never a detector."""
import argparse
from run_sparkov_argn_control import OUT,DOCS,CFG,CORE,LABEL,MERCHANT,AMOUNT,pd,np,write,digest
from benchmarks.argn_fraud_audit import features,summaries,position_curves,risk_table,tv
import json
from scipy.stats import ks_2samp,wasserstein_distance

def extended(frame,state):
    d=features(frame,state)
    d['previous_fraud']=d.groupby('entity_id').fraud.shift().fillna(-2).astype(int)
    d['sequence_length']=d.groupby('entity_id').entity_id.transform('size')
    prior=d.groupby('entity_id',sort=False)[AMOUNT].transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    d['amount_history_ratio_raw']=d[AMOUNT]/prior.where(prior.gt(0))
    return d

def numeric_metrics(real,synthetic,name):
    rows=[]
    for field,column,minimum_position in [('amount',AMOUNT,0),('gap_seconds','gap',1),('amount_history_ratio','amount_history_ratio_raw',5)]:
        for label,value in [('all',None),('normal',0),('fraud',1)]:
            arrays=[];row=dict(run=name,field=field,label=label)
            for prefix,frame in [('real',real),('generated',synthetic)]:
                use=frame.event_index.ge(minimum_position)
                if value is not None:use &=frame.fraud.eq(value)
                raw=frame.loc[use,column].to_numpy(dtype=float);valid=np.isfinite(raw)&(raw>=0);x=raw[valid];arrays.append(x)
                row[prefix+'_eligible']=len(raw);row[prefix+'_valid']=len(x);row[prefix+'_invalid_or_missing']=int((~valid).sum())
                for stat,number in [('mean',float(x.mean()) if len(x) else None),('median',float(np.median(x)) if len(x) else None),('p90',float(np.quantile(x,.9)) if len(x) else None),('p99',float(np.quantile(x,.99)) if len(x) else None)]:row[prefix+'_'+stat]=number
            a,b=arrays
            row['ks_distance']=float(ks_2samp(a,b,method='asymp').statistic) if len(a) and len(b) else None
            row['wasserstein_raw']=float(wasserstein_distance(a,b)) if len(a) and len(b) else None
            row['wasserstein_log1p']=float(wasserstein_distance(np.log1p(a),np.log1p(b))) if len(a) and len(b) else None
            rows.append(row)
    return rows

def additional_relations(real,synthetic):
    result={}
    for label in [0,1]:
        rr=real[real.fraud.eq(label)];ss=synthetic[synthetic.fraud.eq(label)]
        result[f'class_{label}_merchant_tv']=tv(rr,ss,[MERCHANT])
        result[f'class_{label}_merchant_category_tv']=tv(rr,ss,[MERCHANT,'category'])
        result[f'class_{label}_gap_category_amount_history_tv']=tv(rr[rr.event_index.gt(0)],ss[ss.event_index.gt(0)],['gap_bin','category','amount_bin','seen_merchant','amount_vs_history'])
    return result

def ratio_ci(frame,all_ids,repeats=500):
    counts=frame.groupby('entity_id').fraud.agg(events='size',frauds=lambda x:x.eq(1).sum()).reindex(all_ids,fill_value=0)
    rng=np.random.default_rng(20260928)
    weights=rng.multinomial(len(all_ids),np.full(len(all_ids),1/len(all_ids)),size=repeats)
    events=weights@counts.events.to_numpy();frauds=weights@counts.frauds.to_numpy()
    rates=np.divide(frauds,events,out=np.full(repeats,np.nan),where=events>0)
    ok=rates[np.isfinite(rates)]
    return dict(events=len(frame),customers=frame.entity_id.nunique(),frauds=int(frame.fraud.eq(1).sum()),
        fraud_rate=float(frame.fraud.eq(1).mean()) if len(frame) else None,
        ci_low=float(np.quantile(ok,.025)) if len(ok) else None,ci_high=float(np.quantile(ok,.975)) if len(ok) else None,
        bootstrap_nonempty=len(ok),adequate_support=bool(frame.entity_id.nunique()>=10 and frame.fraud.eq(1).sum()>=20))

def conditions(d):
    subsequent=d.event_index.gt(0)
    return {'all':np.ones(len(d),bool),'positive_gap_le_5s':subsequent&d.gap.gt(0)&d.gap.le(5),
        'positive_gap_le_300s':subsequent&d.gap.gt(0)&d.gap.le(300),
        'seen_merchant':d.seen_merchant.eq(1),'new_merchant':d.seen_merchant.eq(0),
        'amount_gt_3x_prior20median':d.amount_vs_history.eq(2),
        'previous_fraud':subsequent&d.previous_fraud.eq(1),'previous_normal':subsequent&d.previous_fraud.eq(0),
        'short_seen':subsequent&d.gap.gt(0)&d.gap.le(300)&d.seen_merchant.eq(1),
        'short_new':subsequent&d.gap.gt(0)&d.gap.le(300)&d.seen_merchant.eq(0),
        'first_transaction':d.event_index.eq(0),
        'short_sequence_le_100_RETROSPECTIVE':d.sequence_length.le(100),
        'long_sequence_gt_100_RETROSPECTIVE':d.sequence_length.gt(100)}

def evaluate(supported=False):
    base=OUT/'prepared';state=json.loads((base/'metric_state.json').read_text())
    train=pd.read_parquet(base/'train.parquet');val=pd.read_parquet(base/'validation.parquet')
    prefix='supported_' if supported else ''
    ids_supported=set(pd.read_parquet(base/'supported_validation_context.parquet').customer_id) if supported else None
    if supported:val=val[val.entity_id.isin(ids_supported)]
    training_lengths=train.groupby('entity_id').size()
    low_max=int(training_lengths[training_lengths.le(100)].max())
    high_min=int(training_lengths[training_lengths.gt(100)].min())
    vr=extended(val,state)
    paths={'real_train':base/'train.parquet','real_validation':base/'validation.parquet','codec_roundtrip':base/'validation_roundtrip.parquet'}
    generated=sorted((OUT/'runs').glob('*/generated_*.parquet'))
    if supported:
        generated=[p for p in generated if 'generated_supported_validation_' in p.name or ('generated_validation_' in p.name and not p.parent.name.startswith('cpar_'))]
    else:
        generated=[p for p in generated if 'generated_supported_' not in p.name and not p.parent.name.startswith('cpar_')]
    paths.update({f'{p.parent.name}/{p.stem}':p for p in generated})
    metrics=[];positions=[];rates=[];risk=[];lengths=[];numeric=[]
    for name,path in paths.items():
        raw=pd.read_parquet(path)
        if supported and name!='real_train':
            raw=raw[raw.entity_id.isin(ids_supported)]
            assert set(raw.entity_id)==ids_supported
        d=extended(raw,state)
        row=dict(run=name,**summaries(val,raw,state))
        row.update(additional_relations(vr,d))
        row['previous_current_fraud_tv']=tv(vr[vr.event_index.gt(0)],d[d.event_index.gt(0)],['previous_fraud','fraud'])
        row['fraud_previous_category_current_category_tv']=tv(vr[(vr.fraud==1)&vr.event_index.gt(0)],d[(d.fraud==1)&d.event_index.gt(0)],['previous_category','category'])
        for phase,previous in [('onset',0),('continuation',1)]:
            rr=vr[(vr.fraud==1)&(vr.previous_fraud==previous)];ss=d[(d.fraud==1)&(d.previous_fraud==previous)]
            row[f'fraud_{phase}_events']=len(ss)
            row[f'real_fraud_{phase}_events']=len(rr)
            row[f'fraud_{phase}_category_amount_tv']=tv(rr,ss,['category','amount_bin'])
            row[f'fraud_{phase}_history_amount_tv']=tv(rr,ss,['gap_bin','seen_merchant','amount_vs_history'])
        metrics.append(row)
        numeric.extend(numeric_metrics(vr,d,name))
        positions.extend([dict(run=name,**x) for x in position_curves(val,raw,train,state)])
        ids=d.entity_id.unique()
        for condition,mask in conditions(d).items():rates.append(dict(run=name,condition=condition,**ratio_ci(d.loc[mask],ids,CFG['bootstrap_repeats'])))
        risk.append(risk_table(raw,state,name))
        ll=d.groupby('entity_id').size()
        by_customer=d.groupby('entity_id').fraud.agg(fraud_rate=lambda x:x.eq(1).mean(),frauds=lambda x:x.eq(1).sum())
        lengths.append(dict(run=name,customers=len(ll),events=len(d),minimum=int(ll.min()),p10=float(ll.quantile(.1)),median=float(ll.median()),p90=float(ll.quantile(.9)),maximum=int(ll.max()),customer_weighted_fraud_rate=float(by_customer.fraud_rate.mean()),all_fraud_customers=int(by_customer.fraud_rate.eq(1).sum()),customers_with_any_fraud=int(by_customer.frauds.gt(0).sum()),short_customers_le_100=int(ll.le(100).sum()),training_length_gap_low_exclusive=low_max,training_length_gap_high_exclusive=high_min,customers_in_training_length_gap=int((ll.gt(low_max)&ll.lt(high_min)).sum())))
        print('EVALUATED',name,flush=True)
    pd.DataFrame(metrics).to_csv(DOCS/f'{prefix}metrics.csv',index=False)
    pd.DataFrame(positions).to_csv(DOCS/f'{prefix}positions.csv',index=False)
    pd.DataFrame(rates).to_csv(DOCS/f'{prefix}conditional_rates_customer_bootstrap.csv',index=False)
    pd.concat(risk).to_csv(DOCS/f'{prefix}risk_cells.csv',index=False)
    pd.DataFrame(lengths).to_csv(DOCS/f'{prefix}lengths.csv',index=False)
    pd.DataFrame(numeric).to_csv(DOCS/f'{prefix}numeric_metrics.csv',index=False)
    write(DOCS/f'{prefix}evaluation_manifest.json',dict(files={name:digest(p) for name,p in paths.items()},test_events_loaded=False,script_sha256=digest(__file__),bootstrap_unit='customer',bootstrap_repeats=CFG['bootstrap_repeats'],ci_scope='within one output draw, not fit-seed uncertainty',validation_customers=val.entity_id.nunique(),cohort='native categorical context common support' if supported else 'all requested validation contexts',context_support_sha256=digest(base/'supported_validation_context.parquet') if supported else None))

def reference(supported=False):
    base=OUT/'prepared';state=json.loads((base/'metric_state.json').read_text());train=pd.read_parquet(base/'train.parquet');val=pd.read_parquet(base/'validation.parquet')
    prefix='supported_' if supported else ''
    if supported:val=val[val.entity_id.isin(pd.read_parquet(base/'supported_validation_context.parquet').customer_id)]
    count=val.entity_id.nunique()
    groups={key:d for key,d in train.groupby('entity_id')};ids=list(groups);rng=np.random.default_rng(20260928);rows=[];vr=extended(val,state);numeric=[]
    for repeat in range(12):
        parts=[]
        for new,key in enumerate(rng.choice(ids,count,replace=True)):
            d=groups[key].copy();d.entity_id=new;parts.append(d)
        sampled=pd.concat(parts,ignore_index=True)
        row=dict(repeat=repeat,**summaries(val,sampled,state));ss=extended(sampled,state)
        row.update(additional_relations(vr,ss))
        numeric.extend(numeric_metrics(vr,ss,f'reference_{repeat}'))
        for phase,previous in [('onset',0),('continuation',1)]:
            real=vr[(vr.fraud==1)&(vr.previous_fraud==previous)];syn=ss[(ss.fraud==1)&(ss.previous_fraud==previous)]
            row[f'fraud_{phase}_category_amount_tv']=tv(real,syn,['category','amount_bin'])
            row[f'fraud_{phase}_history_amount_tv']=tv(real,syn,['gap_bin','seen_merchant','amount_vs_history'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(DOCS/f'{prefix}training_customer_resampling_reference.csv',index=False)
    pd.DataFrame(numeric).to_csv(DOCS/f'{prefix}numeric_reference.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['evaluate','reference','evaluate-supported','reference-supported']);args=p.parse_args()
    (evaluate if args.command.startswith('evaluate') else reference)(supported=args.command.endswith('-supported'))
