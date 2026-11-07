"""Matched 128/256 native optimizer epochs, fixed split and internal selection."""
import copy,json,os,sys,time,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_sparkov_argn_control import ROOT,CORE,STATIC,LABEL,MERCHANT,AMOUNT,seed,write,digest,native_frame
BASE=ROOT/'artifacts/argn_state_first_v1'
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
from experiments.cs_saf_cpar_loss import equivalent_par_loss
from experiments.sparkov_cpar_dense import equivalent_microbatch_training
from experiments.sparkov_cpar_sampling import equivalent_sampling
OUT=ROOT/'artifacts/cpar_adequacy_v1';SEED=20261021;DRAWS=[20261031,20261032]
def verify():
 m=json.loads((OUT/'MANIFEST.json').read_text())
 for p,h in m['hashes'].items():assert digest(p)==h,p
 assert importlib.metadata.version('sdv')=='1.38.0' and importlib.metadata.version('deepecho')=='0.8.1'
def raw_split(suffix):
 path=BASE/'prepared/workspace/OriginalData'
 d=pd.read_parquet(path/f'tgt-data/part.000000-{suffix}.parquet');c=pd.read_parquet(path/f'ctx-data/part.000000-{suffix}.parquet')
 assert set(d.customer_id)==set(c.customer_id)
 return d,c
def sample(model,parent,out,label):
 stub=pd.DataFrame(index=parent.index,columns=['customer_id',*CORE,*STATIC])
 for c in parent:stub[c]=parent[c].to_numpy()
 processed=model._data_processor.transform(stub)[['customer_id',*model.context_columns]]
 supported=[];failures=[]
 for row in processed.itertuples(index=False,name=None):
  try:model._model._context_to_tensor(list(row[1:]));supported.append(row[0])
  except KeyError as e:failures.append(dict(customer_id=row[0],reason=str(e)))
 write(out/f'{label}_SUPPORT.json',dict(requested=len(parent),supported=supported,excluded=failures))
 processed=processed[processed.customer_id.isin(supported)];results=[]
 for gs in DRAWS:
  seed(gs);model._data_processor.reset_sampling();start=time.monotonic()
  with equivalent_sampling():raw=model._sample(processed,sequence_length=None)
  raw.to_parquet(out/f'native_{label}_{gs}.parquet',index=False);d=native_frame(raw);p=out/f'generated_{label}_{gs}.parquet';d.to_parquet(p,index=False)
  write(out/f'GENERATION_{label}_{gs}.json',dict(rows=len(d),customers=d.entity_id.nunique(),seconds=time.monotonic()-start,sha256=digest(p),future_lengths_supplied=False))
  results.append(d);print('CPAR_GENERATED',out.name,label,gs,len(d),flush=True)
 return results,supported

def metrics(real,raw):
 from scipy.stats import wasserstein_distance
 def features(d):
  d=d.sort_values(['entity_id','event_index']).copy();d['label']=pd.to_numeric(d.event_is_fraud)
  d['prev']=d.groupby('entity_id').label.shift();d['past']=d.groupby('entity_id').gap.transform(lambda s:pd.to_numeric(s).shift().rolling(20,min_periods=5).median())
  return d
 r,g=features(real),features(raw);out={}
 for field,label in [('normal',0),('fraud',1)]:
  a=r.loc[r.label.eq(label)&r.event_index.gt(0),'gap'];b=g.loc[g.label.eq(label)&g.event_index.gt(0),'gap']
  out[f'{field}_gap_seconds_log_w1']=float(wasserstein_distance(np.log1p(a),np.log1p(b))) if len(a) and len(b) else np.nan
 a=r[r.label.eq(1)&r.prev.eq(0)&r.past.gt(0)];b=g[g.label.eq(1)&g.prev.eq(0)&g.past.gt(0)]
 out['onset_personal_gap_w1']=float(wasserstein_distance(np.log1p(a.gap/a.past),np.log1p(b.gap/b.past))) if len(a) and len(b) else np.nan
 out.update(real_onsets=len(a),generated_onsets=len(b),real_fraud_rate=float(r.label.mean()),generated_fraud_rate=float(g.label.mean()))
 return out

def main():
 verify();assert torch.cuda.is_available();torch.set_num_threads(2);run=OUT/'run';run.mkdir(parents=True,exist_ok=False)
 d,c=raw_split('trn');val,vctx=raw_split('val');assert not set(c.customer_id)&set(vctx.customer_id)
 data=d[['customer_id',*CORE]].copy();data['gap']=data.gap.fillna(0);data=data.merge(c,on='customer_id',validate='many_to_one')
 cats=[MERCHANT,'category',LABEL,'cardholder_gender','cardholder_state']
 for name in cats:data[name]=data[name].astype(str)
 metadata=SingleTableMetadata();metadata.detect_from_dataframe(data);metadata.update_column('customer_id',sdtype='id');metadata.set_sequence_key('customer_id')
 for name in data.columns.drop('customer_id'):metadata.update_column(name,sdtype='categorical' if name in cats else 'numerical')
 model=PARSynthesizer(metadata,context_columns=STATIC,epochs=256,segment_size=None,sample_size=1,cuda=True,verbose=True)
 seed(SEED);start=time.monotonic();write(run/'START.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seed=SEED,training_customers=len(c),training_events=len(d),internal_customers=len(vctx),epochs=256))
 def progress(par,optimizer,epoch):
  if epoch in [128,256]:torch.save({k:v.detach().cpu() for k,v in par._model.state_dict().items()},run/f'epoch_{epoch}.pt')
  if epoch%8==0:par.loss_values.to_csv(run/'LOSS.csv',index=False)
  write(run/'PROGRESS.json',dict(epoch=epoch,seconds=time.monotonic()-start))
 with equivalent_par_loss(),equivalent_microbatch_training(progress):model.fit(data)
 model.save(run/'native_256.pkl');write(run/'FIT.json',dict(seconds=time.monotonic()-start,sha256=digest(run/'native_256.pkl'),epochs=256,independent_trajectories=1))
 real=native_frame(val);rows=[]
 for budget in [128,256]:
  model._model._model.load_state_dict(torch.load(run/f'epoch_{budget}.pt',map_location=model._model.device,weights_only=True))
  out=OUT/f'budget_{budget}';out.mkdir();model.save(out/'model.pkl')
  samples,supported=sample(model,vctx,out,'internal')
  actual=real[real.entity_id.isin(supported)]
  for gs,syn in zip(DRAWS,samples):
   values=metrics(actual,syn);rows.append(dict(epochs=budget,generation_seed=gs,**values))
  pd.DataFrame(rows).to_csv(OUT/'report/INTERNAL_DRAWS.csv',index=False)
 scores={}
 for budget in [128,256]:
  r=pd.DataFrame(rows);r=r[r.epochs==budget]
  v=r[['normal_gap_seconds_log_w1','onset_personal_gap_w1']].to_numpy()
  scores[budget]=float(v.mean()) if np.isfinite(v).all() else float('inf')
 chosen=min(scores,key=lambda e:(scores[e],e));write(OUT/'SELECTION.json',dict(chosen_epochs=chosen,scores={k:(v if np.isfinite(v) else None) for k,v in scores.items()},criterion='mean normal/onset personal log W1 across two internal draws; missing phase => infinity, tie=>128',test_reads=0))
 model._model._model.load_state_dict(torch.load(run/f'epoch_{chosen}.pt',map_location=model._model.device,weights_only=True))
 dev=pd.read_parquet(BASE/'prepared/validation_context.parquet');samples,support=sample(model,dev,OUT/f'budget_{chosen}','development')
 verify();write(OUT/'COMPLETE.json',dict(selected_epochs=chosen,fit_trajectories=1,internal_generations=4,development_generations=2,development_supported_customers=len(support),test_reads=0))
if __name__=='__main__':main()
