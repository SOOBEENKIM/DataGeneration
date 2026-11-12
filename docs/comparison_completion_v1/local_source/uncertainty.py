"""Exploratory paired whole-fit and customer uncertainty on frozen outputs."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance,t
from diagnose_argn_residual import decorate
from run_argn_state_first import SOURCE
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/comparison_completion_v1'
OLD=ROOT/'artifacts/argn_whole_validation_v1';FITS=[20261021,20261022,20261023];DRAWS=[20261031,20261032]
ARMS=['history_only','history_position6','history_calibration_only']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def w(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(wasserstein_distance(np.log1p(a),np.log1p(b))) if len(a) and len(b) else float('nan')
def selected(d,key):
    if key=='normal':return d[d.label.eq(0)&d.event_index.gt(0)&d.gap.notna()]
    return d[d.phase.eq(key)&d.past_gap.gt(0)]
def values(d,key):return d.gap if key=='normal' else d.gap/d.past_gap
def main():
    manifest=json.loads((OUT/'UNCERTAINTY_MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    real=decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet'))
    table=pd.read_csv(OLD/'report/ALL_DRAWS.csv');checks=[];rows=[];supports=[]
    for fs in FITS:
      for gs in DRAWS:
        for arm in ARMS:
          path=OLD/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'
          rec=json.loads((path.parent/f'GENERATION_{gs}.json').read_text());assert digest(path)==rec['sha256']
          d=decorate(pd.read_parquet(path));saved=table[(table.arm==arm)&(table.fit_seed==fs)&(table.generation_seed==gs)].iloc[0]
          for key,metric in [('normal','normal_gap_seconds_log_w1'),('onset','onset_personal_gap_w1'),('normal_stay','normal_stay_personal_gap_w1')]:
            a,b=selected(real,key),selected(d,key);err=w(values(a,key),values(b,key));diff=abs(err-saved[metric]);assert diff<1e-10,(metric,diff)
            checks.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,metric=metric,error=diff))
            supports.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,metric=metric,real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique()))
            if key=='onset':continue # population onset is NOT paired actual event prediction
            ga={cid:q for cid,q in a.groupby('entity_id')};gb={cid:q for cid,q in b.groupby('entity_id')}
            for cid in sorted(set(real.entity_id)):
                aa=ga.get(cid,a.iloc[:0]);bb=gb.get(cid,b.iloc[:0])
                rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,customer_id=cid,metric=metric,real_events=len(aa),generated_events=len(bb),w1=w(values(aa,key),values(bb,key))))
          print('RAW_CHECKED',fs,gs,arm,flush=True)
    pd.DataFrame(checks).to_csv(OUT/'report/UNCERTAINTY_RAW_CHECKS.csv',index=False)
    pd.DataFrame(supports).to_csv(OUT/'report/UNCERTAINTY_SUPPORT.csv',index=False)
    data=pd.DataFrame(rows);data.to_csv(OUT/'report/CUSTOMER_ERRORS.csv',index=False)
    keys=[k for k in pd.read_csv(OLD/'report/MEANS.csv').columns if k!='arm']
    fit=table.groupby(['arm','fit_seed'])[keys].mean();stats=[]
    for control in ARMS[:2]:
      for key in keys:
        delta=(fit.loc['history_calibration_only',key]-fit.loc[control,key]).to_numpy()
        se=np.std(delta,ddof=1)/np.sqrt(3);half=t.ppf(.975,2)*se
        perm=np.array([np.mean(delta*np.asarray(s)) for s in itertools.product([-1,1],repeat=3)])
        stats.append(dict(control=control,metric=key,n_independent_fits=3,delta=float(delta.mean()),ci95_low=float(delta.mean()-half),ci95_high=float(delta.mean()+half),lower_fits=int((delta<0).sum()),sign_flip_p=float((np.abs(perm)>=abs(delta.mean())-1e-14).mean()),leave_one_fit_min=float(min(np.delete(delta,i).mean() for i in range(3))),leave_one_fit_max=float(max(np.delete(delta,i).mean() for i in range(3)))))
    stats=pd.DataFrame(stats);stats.to_csv(OUT/'report/WHOLE_FIT_UNCERTAINTY.csv',index=False)
    rng=np.random.default_rng(20261041);boot=[];cost=[]
    for control in ARMS[:2]:
      p=data[data.arm.isin([control,'history_calibration_only'])].pivot(index=['fit_seed','generation_seed','customer_id','metric'],columns='arm',values='w1')
      p['delta']=p['history_calibration_only']-p[control]
      for metric,q in p.reset_index().groupby('metric'):
        grouped=q.groupby('customer_id').delta.agg(['mean','count']);matched=grouped[grouped['count'].eq(6)]
        vals=matched['mean'].to_numpy();sample=vals[rng.integers(0,len(vals),size=(10000,len(vals)))].mean(axis=1)
        boot.append(dict(control=control,metric=metric,estimand='mean matched-customer W1 difference, conditional on fixed trained outputs; NOT pooled W1',requested_customers=147,complete_six_pair_support=len(vals),incomplete_or_missing=147-len(vals),delta=float(vals.mean()),bootstrap95_low=float(np.quantile(sample,.025)),bootstrap95_high=float(np.quantile(sample,.975)),improved_customers=int((vals<0).sum()),worsened_customers=int((vals>0).sum())))
        for cid,r in grouped.iterrows():cost.append(dict(control=control,metric=metric,customer_id=cid,delta=r['mean'],available_pairs=int(r['count'])))
    pd.DataFrame(boot).to_csv(OUT/'report/CUSTOMER_BOOTSTRAP.csv',index=False);pd.DataFrame(cost).to_csv(OUT/'report/CUSTOMER_EFFECTS.csv',index=False)
    previous=json.loads((ROOT/'artifacts/argn_calibration_specificity_v1/PREVIOUS_HASHES.json').read_text())
    for p,h in previous.items():assert digest(p)==h,p
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    result=dict(complete=True,raw_checks=len(checks),max_error=max(x['error'] for x in checks),prior_hashes=len(previous),independent_fits=3,new_fits=0,new_generations=0,test_reads=0,confirmatory=False,model_replacement=False)
    (OUT/'report/UNCERTAINTY_COMPLETE.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result),flush=True)
    print(stats[(stats.control=='history_position6')&stats.metric.isin(['normal_gap_seconds_log_w1','onset_personal_gap_w1'])].to_string(index=False),flush=True)
if __name__=='__main__':main()
