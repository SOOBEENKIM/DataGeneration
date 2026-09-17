"""Verify all frozen artifacts and summarize every registered arm, without selecting winners."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.cs_saf_followup import OUTPUT,CACHE,CANDIDATES,CONTRACT_SHA,load_contract,folder_for,verify_artifacts,parent
from experiments.cs_saf_pilot import decide_pilot
from experiments.cs_saf_loss_control import summarize
from scripts.audit_cs_saf_oracle import sha256
c,cfg=load_contract();terminal=json.loads((OUTPUT/'COMPLETE.json').read_text());assert terminal['status']=='COMPLETE'
assert terminal['config_sha256']==CONTRACT_SHA
for key,name in [('cpu_gate_sha256',terminal['cpu_gate_path']),('gpu_progress_sha256',OUTPUT/'gpu_progress.json'),('generation_progress_sha256',OUTPUT/'generation_progress.json')]:
 assert sha256(Path(name))==terminal[key]
assert terminal['new_internal_fits']==170 and terminal['CPAR_fits']==40
cpu=json.loads(Path(terminal['cpu_gate_path']).read_text());assert cpu['decision']=='PASS' and cpu['source_commit']==terminal['source_commit']
files=[];states=0;arrays=0;generated_entities=0;missing=[];rows=[]
def verify(folder):
 r=verify_artifacts(folder)
 files.append({'path':str(folder/'COMPLETE.json'),'sha256':sha256(folder/'COMPLETE.json')})
 for name,expected in r['artifact_sha256'].items():files.append({'path':str(folder/name),'sha256':expected})
 return r

def statistics(values):
 if not values:
  return {'n_trials':0,'values':[],'mean':None,'sample_SD':None,'standard_error':None,'descriptive_95_percent_t_interval':None,'negative_count':0,'nonpositive_count':0,'interval_not_multiplicity_adjusted':True,'missing':True}
 s=parent.seed_statistics(values)
 a=np.asarray(values,float)
 if len(a)==5:
  np.testing.assert_allclose(s['descriptive_95_percent_t_interval'],a.mean()+np.array([-1,1])*2.7764451051977987*a.std(ddof=1)/np.sqrt(5),atol=1e-14,rtol=0)
 return s

jobs={};metrics={};responses={};pairs={};gates={};all_aggregates={}
for pi in c['prevalences']:
 jobs[pi]={};metrics[pi]={};responses[pi]={}
 payloads={k:parent.load_cache(CACHE/f'pi_{pi:.2f}_kappa_{k}.pt') for k in (0,1)}
 for t in range(5):
  for k in (0,1):
   reports=[];plans=[];refs=[]
   for a in CANDIDATES:
    folder=folder_for(pi,k,t,a);done=verify(folder)
    report=json.loads((folder/'training_report.json').read_text());audit=json.loads((folder/'intervention_audit.json').read_text());acc=json.loads((folder/'conditional_accuracy.json').read_text())
    assert report['trial_seeds']==parent.trial_config(t)[0] and not report['test_accessed']
    assert report['parameters']==(133549 if a=='U' else 133581)
    if a!='U':assert report['architecture']['regularization_coefficient']==c['candidates'][a]['lambda']
    best=float('inf');epoch=-1
    for h in report['history']:
     if h['validation']['base_nll']<best-1e-8:best=h['validation']['base_nll'];epoch=h['epoch']
    assert epoch==report['best_epoch'] and best==report['best_validation_base_nll']
    if acc['missing_snapshots']:missing.append({'pi':pi,'trial':t,'kappa':k,'candidate':a,'missing':acc['missing_snapshots']})
    loaded=np.load(folder/'accuracy_arrays.npz',allow_pickle=False)
    for sn,meta in acc['checkpoints'].items():
     cp=torch.load(meta['checkpoint_path'],map_location='cpu')
     assert sha256(Path(meta['checkpoint_path']))==meta['checkpoint_sha256']
     assert parent.tensor_digest(cp['model_state'])==meta['state_sha256'];states+=1
     if sn=='best':assert meta['state_sha256']==report['best_state_sha256']
     for split,summary in meta['splits'].items():
      for y,g in summary['groups'].items():
       prefix=f'{sn}_{split}_label_{y}'
       m=loaded[prefix+'_metrics'];ids=loaded[prefix+'_entity_ids']
       selected=torch.where(payloads[k][split]['codes']==int(y)+3)[0]
       expected=[payloads[k][split]['entity_ids'][int(i)] for i in selected]
       assert ids.tolist()==expected and np.isfinite(m).all()
       for j,col in enumerate(summary['columns']):assert summarize(m[:,j])==g['metrics'][col]
       metrics[pi][t,k,a,sn,split,int(y)]=m.copy();arrays+=1
    loaded.close()
    sample=torch.load(folder/'generated_sample.pt',map_location='cpu');assert sample['model_state_sha256']==report['best_state_sha256']
    sample=sample['sample'];expected=np.random.default_rng(parent.trial_config(t)[0]['sampling_seed']).integers(len(payloads[k]['train']['lengths']),size=2048)
    np.testing.assert_array_equal(sample['plan_train_indices'],expected)
    np.testing.assert_array_equal(sample['lengths'],payloads[k]['train']['lengths'][expected])
    np.testing.assert_array_equal(sample['static_codes'],payloads[k]['train']['codes'][expected])
    mask=sample['valid_mask'].clone();mask[:,0]=False
    reps=torch.tensor(payloads[k]['tensorizer_state']['gap_support']['representatives'],dtype=sample['gap'].dtype)
    assert torch.isin(sample['gap'][mask],reps).all() and torch.isnan(sample['gap'][:,0]).all()
    assert torch.isfinite(sample['numeric_value'][sample['valid_mask']]).all()
    assert ((sample['receiver'][sample['valid_mask']]>=3)&(sample['receiver'][sample['valid_mask']]<67)).all()
    generated_entities+=len(sample['lengths'])
    jobs[pi][t,k,a]={'training':report,'audit':audit,'accuracy':acc,'terminal':done}
    reports.append(report);plans.append(audit['sampling_plan_sha256']);refs.append(acc['checkpoints']['best']['reference_probabilities'])
   assert len({r['initialization_contract']['common_state_sha256'] for r in reports})==1
   assert len({r['initial_state_sha256'] for r in reports[1:]})==1
   assert len(set(plans))==1 and all(ref==refs[0] for ref in refs)
   for field in ('cache_sha256','options','trial_seeds','train_entities','validation_entities','transition_counts_by_code'):
    assert all(r[field]==reports[0][field] for r in reports)
   for epoch in range(min(r['epochs_completed'] for r in reports)):
    assert len({r['history'][epoch]['entity_order_prefix_sha256'] for r in reports})==1
 for a in CANDIDATES:
  responses[pi][a]=[decide_pilot({k:jobs[pi][t,k,a]['audit'] for k in (0,1)},cfg['pilot_gate']) for t in range(5)]
 all_aggregates[str(pi)]={}
 for a in CANDIDATES:
  out={}
  for sn in ('best','epoch_9'):
   out[sn]={}
   for split in ('train','validation'):
    out[sn][split]={}
    for cell in ('null','active','k0_y0','k0_y1','k1_y0','k1_y1'):
     values=[]
     cells=[(0,0),(0,1),(1,0)] if cell=='null' else ([(1,1)] if cell=='active' else [(int(cell[1]),int(cell[-1]))])
     for t in range(5):
      if all((t,k,a,sn,split,y) in metrics[pi] for k,y in cells):values.append(np.mean([metrics[pi][t,k,a,sn,split,y].mean(axis=0) for k,y in cells],axis=0).tolist())
     out[sn][split][cell]={name:statistics([v[j] for v in values]) for j,name in enumerate(parent.COLUMNS)}
  all_aggregates[str(pi)][a]=out
 pairs[str(pi)]={}
 for a,b in [('E','U')]+[(a,b) for a in ('L003','ER','L030') for b in ('E','U')]:
  pair={}
  for sn in ('best','epoch_9'):
   pair[sn]={}
   for split in ('train','validation'):
    pair[sn][split]={}
    for cell,cells in [('null',[(0,0),(0,1),(1,0)]),('active',[(1,1)]),('k0_y0',[(0,0)]),('k0_y1',[(0,1)]),('k1_y0',[(1,0)]),('k1_y1',[(1,1)])]:
     values=[]
     for t in range(5):
      if all((t,k,x,sn,split,y) in metrics[pi] for x in (a,b) for k,y in cells):
       values.append(np.mean([(metrics[pi][t,k,a,sn,split,y]-metrics[pi][t,k,b,sn,split,y]).mean(axis=0) for k,y in cells],axis=0).tolist())
     pair[sn][split][cell]={name:statistics([v[j] for v in values]) for j,name in enumerate(parent.COLUMNS)}
  pairs[str(pi)][f'{a}-{b}']=pair
 gates[str(pi)]={}
 for a in ('L003','ER','L030'):
  effects={parentname:{cell:pairs[str(pi)][f'{a}-{b}']['best']['validation'][cell]['grid_mark_TV']['values'] for cell in ('null','active')} for b,parentname in [('E','CS3-E1'),('U','CS2-U1')]}
  gates[str(pi)][a]=parent.expansion_gate(responses[pi][a],effects)
 print('Verified internal pi',pi,flush=True)

external={};external_pairs={};generation_internal_pairs={};external_run_metadata=[]
for pi in c['prevalences']:
 raw={}
 for t in range(5):
  for k in (0,1):
   folder=OUTPUT/f'external/pi_{pi:.2f}/trial_{t}/kappa_{k}';verify(folder)
   r=json.loads((folder/'comparison.json').read_text());assert r['epochs']==128 and not r['test_accessed']
   external_run_metadata.append({key:value for key,value in r.items() if key not in ('CPAR','empirical_copy_control')})
   losses=pd.read_csv(folder/'loss_history.csv');assert len(losses)==128 and np.isfinite(losses['Loss']).all()
   raw[t,k,'CPAR']=r['CPAR'];raw[t,k,'empirical_copy_control']=r['empirical_copy_control']
   for a in CANDIDATES:
    folder=OUTPUT/f'generation/pi_{pi:.2f}/trial_{t}/kappa_{k}/{a}';verify(folder)
    q=json.loads((folder/'comparison.json').read_text());assert q['sampling_plan_sha256']==r['sampling_plan_sha256']
    raw[t,k,a]=q['metrics']
   generated_entities+=r['generation_entities']
 external[str(pi)]={};external_pairs[str(pi)]={}
 for model in [*CANDIDATES,'CPAR','empirical_copy_control']:
  external[str(pi)][model]={}
  for k in (0,1):
   external[str(pi)][model][str(k)]={g:{metric:statistics([raw[t,k,model][g]['metrics'][metric] for t in range(5)]) for metric in raw[0,k,model][g]['metrics']} for g in ('pooled','context_0','context_1')}
 for model in CANDIDATES:
  external_pairs[str(pi)][model+'-CPAR']={str(k):{g:{metric:statistics([raw[t,k,model][g]['metrics'][metric]-raw[t,k,'CPAR'][g]['metrics'][metric] for t in range(5)]) for metric in raw[0,k,model][g]['metrics']} for g in ('pooled','context_0','context_1')} for k in (0,1)}
 generation_internal_pairs[str(pi)]={}
 for a,b in [('E','U')]+[(a,'E') for a in ('L003','ER','L030')]:
  generation_internal_pairs[str(pi)][a+'-'+b]={str(k):{g:{metric:statistics([raw[t,k,a][g]['metrics'][metric]-raw[t,k,b][g]['metrics'][metric] for t in range(5)]) for metric in raw[0,k,a][g]['metrics']} for g in ('pooled','context_0','context_1')} for k in (0,1)}
 print('Verified external pi',pi,flush=True)

gradients=[]
for t in range(5):
 for k in (0,1):
  folder=OUTPUT/f'gradients/trial_{t}/kappa_{k}';verify(folder)
  r=json.loads((folder/'gradient_diagnostic.json').read_text());assert r['optimizer_steps']==0
  gradients.append(r)
allocations=json.loads((OUTPUT/'gpu_progress.json').read_text())['allocations']
prior_progress=json.loads((OUTPUT/'technical_failure_1/gpu_progress.json').read_text())
allocations+=prior_progress['allocations']
performance_progress=json.loads((OUTPUT/'performance_amendment_2/gpu_progress.json').read_text())
allocations+=performance_progress['allocations']
assert all(x['compute_processes_before_launch']==0 for x in allocations)
assert len(allocations)==223 and len(prior_progress['failures'])==3
assert len({tuple(x['task']) for x in allocations})==220
result={'schema_version':'cs-saf-followup-v1-result','contract_sha256':CONTRACT_SHA,'execution':terminal,'cpu':cpu,
 'verification':{'status':'PASS','artifact_checks':len(files),'checkpoint_states':states,'aligned_entity_arrays':arrays,'generated_entities_internal_plus_CPAR':generated_entities,'missing_snapshots':missing,'matched_initialization_order_plans':True,'files':files},
 'internal':all_aggregates,'paired_internal':pairs,'response_gates':{str(pi):r for pi,r in responses.items()},'diagnostic_gates_not_reclassifying_original':gates,
 'external_generation':external,'external_run_metadata':external_run_metadata,'paired_external_generation':external_pairs,'paired_internal_generation':generation_internal_pairs,'gradient_diagnostics':gradients,
 'technical_failures_preserved':prior_progress['failures'],'gpu_allocations_including_failed_diagnostics':allocations,
 'limitations':['same_DGP_seed42','validation_reused','no_independent_new_data_confirmation','no_test_or_realdata','one_external_neural_family','CPAR_default_budget_not_compute_matched','generation_metrics_not_oracle_conditional_TV','no_multiplicity_adjustment','pinned_CPAR_continuous_target_tail_alignment_preserved','CPAR_loss_vectorization_changes_floating_point_reduction_order','combined_training_bank_and_generation_trial_variation']}
# The runner has finished: write the reviewed compact evidence to git only now.
p=ROOT/'docs/cs_saf/followup_v1_result.json';p.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print('VERIFIED',len(files),states,arrays,generated_entities,flush=True)
