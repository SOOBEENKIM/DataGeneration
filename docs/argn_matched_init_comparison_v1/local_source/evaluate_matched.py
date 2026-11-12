"""Frozen global metrics plus preregistered descriptive customer/position costs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from matched_study import ROOT,OUT,OLD,DOCS,SEEDS,DRAWS,verify,digest,write,now,head_path
import evaluate_whole as existing
from diagnose_argn_residual import decorate
from run_argn_state_first import SOURCE
ARMS=['history_only','history_position6','history_calibration_only','init_only']
METRICS=existing.old.KEYS
POSITIONS=[(1,6,'1_5'),(6,50,'6_49'),(50,100,'50_99'),(100,200,'100_199'),(200,500,'200_499'),(500,1024,'500_1023'),(1024,np.inf,'1024_plus')]

def all_metrics(real,raw):
    from benchmarks.argn_fraud_audit import tv,MERCHANT
    values,support=existing.old.metrics(real,raw)
    a=real[pd.to_numeric(real.event_is_fraud).eq(0)]
    b=raw[pd.to_numeric(raw.event_is_fraud).eq(0)]
    values['class_0_merchant_tv']=tv(a,b,[MERCHANT])
    values['median_unique_merchants_per_customer']=float(raw.groupby('entity_id')[MERCHANT].nunique().median())
    assert all(k in values for k in METRICS)
    return values,support

def source(arm,fs,gs):
    return (OUT if arm=='init_only' else OLD)/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'

def check():
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');fs,gs=SEEDS[0],DRAWS[0]
    values,_=all_metrics(real,pd.read_parquet(source('history_calibration_only',fs,gs)))
    table=pd.read_csv(OLD/'report/ALL_DRAWS.csv')
    expected=table[(table.arm=='history_calibration_only')&(table.fit_seed==fs)&(table.generation_seed==gs)].iloc[0]
    errors={k:abs(values[k]-expected[k]) for k in METRICS};assert max(errors.values())<1e-10
    write(OUT/'EVALUATION_READY.json',dict(source_sha256=digest(__file__),checks=errors,passed=True,new_output_reads=0))

def wd(a,b):
    if not len(a) or not len(b):return np.nan
    return float(wasserstein_distance(np.log1p(np.asarray(a,dtype=float)),np.log1p(np.asarray(b,dtype=float))))

def metric_rows(r,g,info):
    rows=[]
    for key in ['normal_absolute','normal_relative','onset_relative']:
        if key=='normal_absolute':
            a=r[r.label.eq(0)&r.event_index.gt(0)&r.gap.notna()];b=g[g.label.eq(0)&g.event_index.gt(0)&g.gap.notna()]
            av,bv=a.gap,b.gap
        else:
            phase='normal_stay' if key=='normal_relative' else 'onset'
            a=r[r.phase.eq(phase)&r.past_gap.gt(0)];b=g[g.phase.eq(phase)&g.past_gap.gt(0)]
            av,bv=a.gap/a.past_gap,b.gap/b.past_gap
        rows.append({**info,'metric':key,'w1':wd(av,bv),'real_events':len(a),'generated_events':len(b),
          'real_customers':a.entity_id.nunique(),'generated_customers':b.entity_id.nunique(),
          'mean_log_bias':float(np.log1p(bv).mean()-np.log1p(av).mean()) if len(av) and len(bv) else np.nan})
    return rows

def analyze(only_existing=False):
    verify();assert digest(__file__)==json.loads((OUT/'EVALUATION_READY.json').read_text())['source_sha256']
    directory=DOCS/('existing_costs' if only_existing else 'matched');directory.mkdir(exist_ok=False)
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');r=decorate(real)
    optimization=pd.read_parquet(ROOT/'artifacts/argn_residual_v1/metadata_optimization.parquet')
    normal=optimization[optimization.label.eq(0)&optimization.event_index.gt(0)]
    med=normal.groupby('entity_id').gap.median();edges=np.quantile(med,[1/3,2/3]);assert edges[0]<edges[1]
    mapping=pd.DataFrame(index=sorted(r.entity_id.unique()))
    mapping['length']=r.groupby('entity_id').size()
    mapping['normal_median_gap']=r[r.label.eq(0)&r.event_index.gt(0)].groupby('entity_id').gap.median()
    mapping['speed']=pd.cut(mapping.normal_median_gap,[-np.inf,*edges,np.inf],labels=['fast','middle','slow']).astype(object).fillna('unknown')
    mapping['length_group']=pd.cut(mapping.length,[0,199,1023,np.inf],labels=['short','medium','long']).astype(str)
    mapping.to_csv(directory/'CUSTOMER_STRATA.csv',index_label='customer_id')
    write(directory/'STRATA_DEFINITION.json',dict(optimization_median_gap_cutoffs=edges.tolist(),
        development_classification='observed full real record, descriptive only, never supplied to generation',unknown_speed=int(mapping.speed.eq('unknown').sum())))
    rows=[];subgroups=[];customers=[];checks=[];supports=[];armset=ARMS[:3] if only_existing else ARMS
    saved=pd.read_csv(OLD/'report/ALL_DRAWS.csv');total=0
    for fs in SEEDS:
      for gs in DRAWS:
        baseline=pd.read_parquet(source('history_only',fs,gs));cols=['entity_id','event_index','event_is_fraud']
        for arm in armset:
          path=source(arm,fs,gs);rec=json.loads((path.parent/f'GENERATION_{gs}.json').read_text())
          assert digest(path)==rec['sha256'];assert (path.parent/'COMPLETE.json').exists()
          raw=pd.read_parquet(path);assert raw[cols].equals(baseline[cols]);total+=len(raw)
          info=dict(arm=arm,fit_seed=fs,generation_seed=gs)
          values,support=all_metrics(real,raw)
          if arm!='init_only':
            expected=saved[(saved.arm==arm)&(saved.fit_seed==fs)&(saved.generation_seed==gs)].iloc[0]
            err={k:abs(values[k]-expected[k]) for k in METRICS};assert max(err.values())<1e-10
          else:
            assert digest(head_path(arm,fs))==rec['head_sha256']
            reference=pd.read_csv(DOCS/f'evaluation/metrics_init_only_{fs}.csv')
            expected=reference[reference.generation_seed==gs].iloc[0]
            err={k:abs(values[k]-expected[k]) for k in values if k in expected};assert max(err.values())<1e-10
          checks.append({**info,'metric_checks':len(err),'max_error':max(err.values()),'labels_lengths_exact':True})
          rows.append({**info,**values,'events':len(raw)});supports.extend([{**info,**x} for x in support])
          d=decorate(raw)
          for lo,hi,label in POSITIONS:
            a=r[r.event_index.ge(lo)&r.event_index.lt(hi)];b=d[d.event_index.ge(lo)&d.event_index.lt(hi)]
            subgroups.extend(metric_rows(a,b,{**info,'axis':'position','group':label}))
          for axis,column in [('customer_speed','speed'),('customer_length','length_group')]:
            for label,subset in mapping.groupby(column,observed=True):
              ids=set(subset.index);a=r[r.entity_id.isin(ids)];b=d[d.entity_id.isin(ids)]
              subgroups.extend(metric_rows(a,b,{**info,'axis':axis,'group':label}))
          gr={cid:q for cid,q in r.groupby('entity_id')};gd={cid:q for cid,q in d.groupby('entity_id')}
          for cid in mapping.index:
            for x in metric_rows(gr[cid],gd[cid],{**info,'customer_id':cid}):
              if x['metric']!='onset_relative':customers.append(x)
          print('EVALUATED',arm,fs,gs,len(raw),flush=True)
    rawtable=pd.DataFrame(rows);rawtable.to_csv(directory/'ALL_DRAWS.csv',index=False)
    f=rawtable.groupby(['arm','fit_seed'])[METRICS].mean();f.to_csv(directory/'FIT_MEANS.csv');means=f.groupby('arm').mean();means.to_csv(directory/'MEANS.csv')
    pd.DataFrame(checks).to_csv(directory/'CHECKS.csv',index=False);pd.DataFrame(supports).to_csv(directory/'SUPPORT.csv',index=False)
    sg=pd.DataFrame(subgroups);sg.to_csv(directory/'SUBGROUP_DRAWS.csv',index=False)
    customer=pd.DataFrame(customers);customer.to_csv(directory/'CUSTOMER_DRAWS.csv',index=False)
    comparisons=[];sf=sg.groupby(['arm','fit_seed','axis','group','metric']).w1.mean();cr=[]
    for control in [x for x in armset if x!='history_calibration_only']:
      for metric in METRICS:
        delta=f.loc['history_calibration_only',metric]-f.loc[control,metric]
        comparisons.append(dict(control=control,metric=metric,mean_delta_C_minus_control=float(delta.mean()),lower_fits=int((delta<0).sum()),fits=len(delta)))
      paired=(sf.loc['history_calibration_only']-sf.loc[control]).rename('delta').reset_index()
      agg=paired.groupby(['axis','group','metric']).delta.agg(['mean','count',lambda x:int((x<0).sum())]).reset_index()
      agg.columns=['axis','group','metric','mean_delta_C_minus_control','fits','lower_fits'];agg['control']=control
      cr.append(agg)
    pd.DataFrame(comparisons).to_csv(directory/'PAIRED_EFFECTS.csv',index=False);pd.concat(cr,ignore_index=True).to_csv(directory/'SUBGROUP_EFFECTS.csv',index=False)
    cp=customer.pivot(index=['fit_seed','generation_seed','customer_id','metric'],columns='arm',values='w1');crows=[]
    for control in [x for x in armset if x!='history_calibration_only']:
      delta=(cp.history_calibration_only-cp[control]).rename('delta').reset_index()
      for (cid,metric),q in delta.groupby(['customer_id','metric']):
        crows.append(dict(control=control,customer_id=cid,metric=metric,delta=q.delta.mean(),available_pairs=int(q.delta.notna().sum())))
    pd.DataFrame(crows).to_csv(directory/'CUSTOMER_EFFECTS.csv',index=False)
    # All old comparison files remain fixed; subgroup summaries are descriptive only.
    verify()
    audit=dict(utc=now(),outputs=len(rows),rows=total,new_generations=0 if only_existing else 6,
      reused_generations=18,metric_checks=sum(x['metric_checks'] for x in checks),max_error=max(x['max_error'] for x in checks),
      native_argn_new_fits=0,matched_backbones=3,draws_per_arm_per_backbone=2,labels_lengths_exact=True,
      test_reads=0,push=False,model_promotion=False,subgroups_confirmatory=False)
    write(directory/'COMPLETE.json',audit)
    if not only_existing:
      write(DOCS/'FINAL_AUDIT.json',audit)
      write(DOCS/'MODEL_DECISION.json',dict(preserve_C_P6_H_init=True,automatic_promotion=False,
           new_architecture_established=False,latest_generator_superiority=False,fit_effects=comparisons))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--existing-only',action='store_true');a=p.parse_args()
    check() if a.check else analyze(a.existing_only)
