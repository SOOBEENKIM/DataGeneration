import csv,json,hashlib,itertools
from pathlib import Path
import numpy as np
from experiments.cs_saf_gap_calibration import ROOT,OUTPUT,CONFIG_SHA,contract,folder_for
from experiments.cs_saf_generation_repeat_stats import repeat_statistics
from experiments.cs_saf_replication import seed_statistics
c=contract();rp=ROOT/'docs/cs_saf/gap_calibration_v1_result.json';r=json.loads(rp.read_text());cp=ROOT/r['all_metrics_csv']['path']
rows=list(csv.DictReader(cp.open()));assert len(rows)==2400
fields=[k for k in rows[0] if k not in ['prevalence','kappa','trial','repeat','model','group']];assert len(fields)==13
index={(float(x['prevalence']),int(x['kappa']),int(x['trial']),int(x['repeat']),x['model'],x['group']):x for x in rows};assert len(index)==2400
assert hashlib.sha256(cp.read_bytes()).hexdigest()==r['all_metrics_csv']['sha256']
nulls=[(0,'0'),(0,'1'),(1,'0')];models=['Ucal','Ugap','Ecal','Egap'];pairs=[('Egap','Ugap'),('Ugap','Ucal'),('Egap','Ecal'),('Ecal','Ucal')]
checked=0;maxerr=0.
def same(got,expected):
 global checked,maxerr
 for key in ['mean','values','descriptive_95_percent_t_interval']:
  a=np.array(got[key]);b=np.array(expected[key]);error=float(np.max(abs(a-b)));assert error<1e-12;maxerr=max(maxerr,error)
 assert got['negative_count']==expected['negative_count'];checked+=1
for pi in c['prevalences']:
 pk=f'{pi:.2f}';data={}
 for model in models:
  v={}
  for field in fields:
   v['active_'+field]=np.array([[float(index[(pi,1,t,z,model,'context_1')][field]) for z in c['repeats']] for t in c['trials']])
   v['three_null_generation_'+field]=np.array([[np.mean([float(index[(pi,k,t,z,model,'context_'+label)][field]) for k,label in nulls]) for z in c['repeats']] for t in c['trials']])
  conditional={}
  for k in c['kappas']:
   for t in c['trials']:
    new=model if model.endswith('gap') else c['variants'][model];prefix='' if model.endswith('gap') else 'control_'
    conditional[(k,t)]=json.loads((folder_for(pi,k,t,new)/(prefix+'conditional_accuracy.json')).read_text())['groups']
  for metric in conditional[(1,0)]['1']['metrics']:
   v['active_conditional_'+metric]=np.array([conditional[(1,t)]['1']['metrics'][metric]['mean'] for t in c['trials']])
   for k,label in nulls:v[f'null_k{k}_c{label}_'+metric]=np.array([conditional[(k,t)][label]['metrics'][metric]['mean'] for t in c['trials']])
   ns=np.stack([v[f'null_k{k}_c{label}_'+metric] for k,label in nulls]);v['three_null_'+metric]=ns.mean(0)
   if metric in ['copy_range','repeat_range']:
    v['mean_trial_worst_null_'+metric]=ns.max(0)
    assert abs(float(ns.mean(1).max())-r['worst_null_cell_mean_ranges'][pk][model][metric]['mean'])<1e-12
  data[model]=v
  for metric,values in v.items():same(r['summaries'][pk][model][metric],repeat_statistics(values) if values.ndim==2 else seed_statistics(values))
 for a,b in pairs:
  for metric,values in data[a].items():
   z=values-data[b][metric];same(r['contrasts'][pk][a+'_minus_'+b][metric],repeat_statistics(z) if z.ndim==2 else seed_statistics(z))
 for metric in data['Egap']:
  z=data['Egap'][metric]-data['Ecal'][metric]-data['Ugap'][metric]+data['Ucal'][metric]
  same(r['contrasts'][pk]['factorial_interaction'][metric],repeat_statistics(z) if z.ndim==2 else seed_statistics(z))
for a,b in pairs:
 pair=a+'_minus_'+b;good=[];cost=[];joint=[]
 for pi in c['prevalences']:
  pk=f'{pi:.2f}';d=r['contrasts'][pk][pair];m=d['active_short_gap_repeat_curve_l1']
  if m['mean']<0 and sum(x<0 for x in m['values'])>=4:good.append(pk)
  if m['mean']>0 and sum(x>0 for x in m['values'])>=4:cost.append(pk)
  if pk in good and all(d[x]['mean']<=0 for x in r['joint_screen_endpoints']):joint.append(pk)
 s=r['diagnostic_screens'][pair];assert s['primary_prevalences']==good and s['cost_prevalences']==cost and s['joint_prevalences']==joint
 assert s['primary_consistent_improvement']==(len(good)>=3) and s['primary_consistent_cost']==(len(cost)>=3) and s['joint_lead']==(len(joint)>=3)
assert all(g['source_commit']==r['source_commit'] for g in r['gates'].values())
assert all(x['parameters_identical'] and x['checkpoint_identical'] and x['conditional_endpoints_identical'] and x['first_saved_generation_identical'] for x in r['technical_attempt_01']['rerun_identity'])
print(json.dumps(dict(status='PASS',statistics_reconstructed=checked,max_difference=maxerr,csv_rows=len(rows),native_scalar_values=len(rows)*len(fields),all_four_comparison_screens_verified=True,result_sha256=hashlib.sha256(rp.read_bytes()).hexdigest()),indent=2))
