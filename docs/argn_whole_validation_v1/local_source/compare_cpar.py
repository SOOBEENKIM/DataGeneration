"""Common-supported-cohort full-pipeline comparison; missing support remains visible."""
import json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from stages import ROOT,OUT,DOCS,SEEDS,ARMS,DRAWS,digest,write,now
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate
CPAR=ROOT/'artifacts/cpar_adequacy_v1'
def w(a,b,wa=None,wb=None):
 a=np.asarray(a);b=np.asarray(b)
 return float(wasserstein_distance(np.log1p(a),np.log1p(b),u_weights=wa,v_weights=wb)) if len(a) and len(b) else np.nan

def metrics(real,raw):
 r,g=decorate(real),decorate(raw);values={};support=[]
 for lab,name in [(0,'normal'),(1,'fraud')]:
  for field,column,minpos in [('gap','gap',1),('amount','amount_or_numeric_value',0)]:
   a=r[r.label.eq(lab)&r.event_index.ge(minpos)];b=g[g.label.eq(lab)&g.event_index.ge(minpos)]
   values[f'{name}_{field}_log_w1']=w(pd.to_numeric(a[column]),pd.to_numeric(b[column]))
   support.append(dict(group=f'{name}_{field}',real_events=len(a),generated_events=len(b)))
 for phase in ['onset','normal_stay','continuation','return_normal']:
  a=r[r.phase.eq(phase)&r.past_gap.gt(0)];b=g[g.phase.eq(phase)&g.past_gap.gt(0)]
  values[f'{phase}_personal_gap_w1']=w(a.gap/a.past_gap,b.gap/b.past_gap)
  support.append(dict(group=phase,real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique()))
 for lo,hi,name in [(6,50,'early'),(100,200,'mid'),(1024,np.inf,'tail')]:
  a=r[r.label.eq(0)&r.event_index.ge(lo)&r.event_index.lt(hi)];b=g[g.label.eq(0)&g.event_index.ge(lo)&g.event_index.lt(hi)]
  values[f'normal_{name}_gap_log_w1']=w(a.gap,b.gap)
  support.append(dict(group=name,real_events=len(a),generated_events=len(b)))
 values.update(real_fraud_rate=float(r.label.mean()),generated_fraud_rate=float(g.label.mean()),
  real_fraud_customers=int(r.groupby('entity_id').label.sum().gt(0).sum()),generated_fraud_customers=int(g.groupby('entity_id').label.sum().gt(0).sum()),
  customer_length_log_w1=w(r.groupby('entity_id').size(),g.groupby('entity_id').size()),events=len(raw),customers=raw.entity_id.nunique())
 return values,support

def main():
 for flag in [OUT/'report/FINAL_AUDIT.json',CPAR/'COMPLETE.json']:
  while not flag.exists():
   if (OUT/'STAGES_FAILED.json').exists() or ((CPAR/'PROCESS_EXIT.json').exists() and not (CPAR/'COMPLETE.json').exists()):
    write(DOCS/'COMPARISON_WAIT.json',dict(stage='blocked_by_execution_failure',utc=now()));return
   time.sleep(10)
 chosen=json.loads((CPAR/'SELECTION.json').read_text())['chosen_epochs'];p=CPAR/f'budget_{chosen}'
 ids=json.loads((p/'development_SUPPORT.json').read_text())['supported'];real=pd.read_parquet(SOURCE/'prepared/validation.parquet');real=real[real.entity_id.isin(ids)]
 assert real.entity_id.nunique()==len(ids)==143
 rows=[];supports=[]
 cases=[('CPAR',20261021,gs,p/f'generated_development_{gs}.parquet') for gs in DRAWS]
 cases += [(arm,fs,gs,OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet') for fs in SEEDS for arm in ARMS for gs in DRAWS]
 for model,fs,gs,path in cases:
  raw=pd.read_parquet(path);raw=raw[raw.entity_id.isin(ids)];assert set(raw.entity_id)==set(ids)
  v,s=metrics(real,raw);info=dict(model=model,fit_seed=fs,generation_seed=gs,cohort='native_common_supported143')
  rows.append({**info,**v});supports += [{**info,**r} for r in s]
 pd.DataFrame(rows).to_csv(DOCS/'CPAR_COMMON143_DRAWS.csv',index=False);pd.DataFrame(supports).to_csv(DOCS/'CPAR_COMMON143_SUPPORT.csv',index=False)
 metrics_cols=[k for k in rows[0] if k not in ['model','fit_seed','generation_seed','cohort']]
 f=pd.DataFrame(rows).groupby(['model','fit_seed'])[metrics_cols].mean();f.to_csv(DOCS/'CPAR_COMMON143_FIT_MEANS.csv')
 write(DOCS/'CPAR_COMPARISON_COMPLETE.json',dict(utc=now(),datasets=len(rows),common_customers=len(ids),argn_independent_fits=3,cpar_independent_fits=1,cpar_selected_epochs=chosen,
  original147retained=True,MeLD_trained_comparison=False,sealed_test_reads=0,qualification='one CPAR trajectory; no balanced multi-fit SOTA or final contribution claim'))
if __name__=='__main__':main()
