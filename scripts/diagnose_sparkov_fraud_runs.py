"""Observed fraud-run composition; never label repair or a causal intervention."""
import json
import argparse
from run_sparkov_argn_control import OUT,DOCS,AMOUNT,pd,np,write,digest
from evaluate_sparkov_argn_control import extended
from scipy.stats import wasserstein_distance

def fraud_runs(frame):
    d=frame.copy()
    boundary=d.entity_id.ne(d.entity_id.shift())|d.fraud.ne(d.fraud.shift())
    d['run_id']=boundary.cumsum()
    d['fraud_age']=d.groupby('run_id').cumcount()+1
    d['fraud_age_band']=np.searchsorted([1,5,10,20],d.fraud_age,side='left')
    d['last_in_customer']=~d.entity_id.eq(d.entity_id.shift(-1))
    return d[d.fraud.eq(1)].copy()

def valid_ratios(d):
    x=d.amount_history_ratio_raw
    return d[d.event_index.ge(5)&np.isfinite(x)&x.ge(0)]

def diagnose(supported=False):
    base=OUT/'prepared';state=json.loads((base/'metric_state.json').read_text())
    prefix='supported_' if supported else ''
    ids=set(pd.read_parquet(base/'supported_validation_context.parquet').customer_id) if supported else None
    validation=pd.read_parquet(base/'validation.parquet')
    if supported:validation=validation[validation.entity_id.isin(ids)]
    real=fraud_runs(extended(validation,state));rv=valid_ratios(real)
    rshare=rv.fraud_age_band.value_counts(normalize=True)
    paths={'real_train':base/'train.parquet','real_validation':base/'validation.parquet'}
    generated=sorted((OUT/'runs').glob('*/generated_validation_*.parquet'))
    generated=[p for p in generated if not p.parent.name.startswith('cpar_')]
    if supported:generated+=sorted((OUT/'runs').glob('cpar_*/generated_supported_validation_*.parquet'))
    paths.update({f'{p.parent.name}/{p.stem}':p for p in generated})
    summary=[];bands=[]
    for name,path in paths.items():
        raw=pd.read_parquet(path)
        if supported and name!='real_train':raw=raw[raw.entity_id.isin(ids)]
        d=fraud_runs(extended(raw,state));v=valid_ratios(d)
        runs=d.groupby('run_id').agg(length=('fraud_age','max'),first_position=('event_index','min'),right_boundary=('last_in_customer','any'))
        lengths=runs.length.to_numpy();row=dict(run=name,events=len(d),runs=len(runs),runs_start_at_customer_boundary=int(runs.first_position.eq(0).sum()),runs_end_at_customer_boundary=int(runs.right_boundary.sum()),events_after_twentieth_fraud=int(d.fraud_age.gt(20).sum()),event_share_after_twentieth_fraud=float(d.fraud_age.gt(20).mean()) if len(d) else None,mean_length=float(lengths.mean()) if len(lengths) else None,p90_length=float(np.quantile(lengths,.9)) if len(lengths) else None,max_length=int(lengths.max()) if len(lengths) else None)
        weights=np.zeros(len(v));coverage=0.
        for band in range(5):
            a=rv[rv.fraud_age_band.eq(band)];b=v[v.fraud_age_band.eq(band)]
            share=float(rshare.get(band,0));mask=v.fraud_age_band.eq(band).to_numpy()
            if len(b):weights[mask]=share/len(b);coverage+=share
            bands.append(dict(run=name,age_band=band,events=int(d.fraud_age_band.eq(band).sum()),valid_ratio_events=len(b),real_valid_ratio_events=len(a),valid_event_share=len(b)/len(v) if len(v) else None,real_valid_event_share=share,amount_mean=float(b[AMOUNT].mean()) if len(b) else None,ratio_median=float(b.amount_history_ratio_raw.median()) if len(b) else None,ratio_log_w1=float(wasserstein_distance(np.log1p(a.amount_history_ratio_raw),np.log1p(b.amount_history_ratio_raw))) if len(a) and len(b) else None))
        row['standardization_real_band_coverage']=coverage
        row['ratio_log_w1']=float(wasserstein_distance(np.log1p(rv.amount_history_ratio_raw),np.log1p(v.amount_history_ratio_raw))) if len(v) else None
        row['age_standardized_ratio_log_w1']=float(wasserstein_distance(np.log1p(rv.amount_history_ratio_raw),np.log1p(v.amount_history_ratio_raw),v_weights=weights)) if len(v) and np.isclose(coverage,1.) else None
        summary.append(row)
    pd.DataFrame(summary).to_csv(DOCS/f'{prefix}fraud_run_summary.csv',index=False)
    pd.DataFrame(bands).to_csv(DOCS/f'{prefix}fraud_run_age_bands.csv',index=False)
    write(DOCS/f'{prefix}fraud_run_manifest.json',dict(protocol_sha256=digest(DOCS/'FRAUD_RUN_DIAGNOSTIC_PROTOCOL.md'),script_sha256=digest(__file__),inputs={name:digest(path) for name,path in paths.items()},causal_intervention=False,generated_outputs_modified=False,validation_customers=validation.entity_id.nunique(),context_support_sha256=digest(base/'supported_validation_context.parquet') if supported else None))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--supported',action='store_true');diagnose(parser.parse_args().supported)
