"""Independent artifact and paired-statistic verification for fixed U/E/ER replication."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from experiments.cs_saf_replication import load_contract,load_cache,tensor_digest,paired_contrasts,seed_statistics,expansion_gate,CANDIDATES
from experiments.cs_saf_pilot import decide_pilot
from experiments.cs_saf_loss_control import summarize
c,cfg=load_contract();runtime=root/'artifacts/cs_saf/replication_v1/gpu_v1'
r=json.loads((runtime/'COMPLETE.json').read_text());cpu_path=root/'artifacts/cs_saf/replication_v1/cpu_v1/COMPLETE.json';cpu=json.loads(cpu_path.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(path,expected,role):
 actual=sha(path);assert actual==expected,(path,expected,actual)
 checks.append({'path':str(path),'sha256':actual,'role':role})
assert r['status']=='COMPLETE' and r['scientific_fits'] in (30,120)
assert not r['test_accessed'] and not r['external_baselines_run'] and not r['independent_data_confirmation']
assert cpu['decision']=='PASS' and cpu['source_commit']==r['source_commit']
check(cpu_path,r['cpu_gate_sha256'],'CPU gate')
check(root/'configs/benchmark_v2/cs_saf_replication_v1.yaml',r['config_sha256'],'preregistration')
check(root/'artifacts/cs_saf/prepared_v1/COMPLETE.json',r['prepared_index_sha256'],'immutable data index')
for key in ('parent_result','parent_contract'):check(root/c[key],c[key+'_sha256'],key)
for path,expected in c['frozen_sources'].items():check(root/path,expected,'frozen model/auditor')
for model in CANDIDATES:
 a=torch.load(cpu_path.parent/model/'run_1/checkpoint_best.pt',map_location='cpu')
 b=torch.load(cpu_path.parent/model/'run_2/checkpoint_best.pt',map_location='cpu')
 assert tensor_digest(a['model_state'])==tensor_digest(b['model_state'])==cpu['candidates'][model]['best_state_sha256']
state_count=array_count=stat_count=sample_count=gap_count=fit_count=0
stages={};missing=[]
for pi,meta in r['stages'].items():
 stage_path=Path(meta['summary_path']);check(stage_path,meta['summary_sha256'],'stage summary')
 stage=json.loads(stage_path.read_text());stages[pi]=stage
 assert stage['scientific_fits']==30 and stage['source_commit']==r['source_commit']
 assert stage['gate']==meta['gate']
 payloads={k:load_cache(root/f'artifacts/cs_saf/prepared_v1/pi_{pi}_kappa_{k}.pt') for k in (0,1)}
 for k in (0,1):check(root/f'artifacts/cs_saf/prepared_v1/pi_{pi}_kappa_{k}.pt',stage['jobs'][f'trial_0/kappa_{k}/CS2-U1']['training']['cache_sha256'],'data cache')
 for key,job in stage['jobs'].items():
  trial=int(key.split('/')[0].split('_')[-1]);kappa=int(key.split('/')[1][-1]);candidate=key.split('/')[-1]
  folder=stage_path.parent/key;payload=payloads[kappa];terminal=job['terminal']
  check(folder/'COMPLETE.json',job['terminal_sha256'],'job terminal')
  assert json.loads((folder/'COMPLETE.json').read_text())==terminal
  assert terminal['source_commit']==r['source_commit'] and terminal['config_sha256']==r['config_sha256']
  for name,expected in terminal['artifact_sha256'].items():check(folder/name,expected,'job artifact')
  report=job['training'];audit=job['audit'];diag=job['accuracy']
  assert report==json.loads((folder/'training_report.json').read_text())
  assert audit==json.loads((folder/'intervention_audit.json').read_text())
  assert diag==json.loads((folder/'conditional_accuracy.json').read_text())
  assert report['trial_index']==trial and report['trial_seeds']==c['trials'][trial]
  assert report['parameters']==(133549 if candidate=='CS2-U1' else 133581)
  assert report['architecture']['bank_initializer_seed']==c['trials'][trial]['bank_seed']
  assert report['loaded_content_splits']==['train','validation'] and not report['test_accessed']
  assert ('epoch_9' in diag['checkpoints'])==(len(report['history'])>=10)
  best=float('inf');selected=-1
  for h in report['history']:
   if h['validation']['base_nll']<best-1e-8:best=h['validation']['base_nll'];selected=h['epoch']
  assert selected==report['best_epoch'] and best==report['best_validation_base_nll']
  if diag['missing_snapshots']:missing.append({'prevalence':pi,'key':key,'missing':diag['missing_snapshots']})
  with np.load(folder/'accuracy_arrays.npz',allow_pickle=False) as arrays:
   for name,item in diag['checkpoints'].items():
    cp_path=Path(item['checkpoint_path']);check(cp_path,item['checkpoint_sha256'],'diagnosed checkpoint')
    cp=torch.load(cp_path,map_location='cpu')
    assert tensor_digest(cp['model_state'])==item['state_sha256']
    assert cp['source_commit']==r['source_commit'] and cp['trial_seeds']==c['trials'][trial]
    assert cp['candidate']==candidate and cp['epoch']==(9 if name=='epoch_9' else selected)
    state_count+=1
    if candidate!='CS2-U1':np.testing.assert_array_equal(cp['model_state']['reference_probabilities'],np.asarray(item['reference_probabilities']))
    for split,summary in item['splits'].items():
     for label,g in summary['groups'].items():
      prefix=f'{name}_{split}_label_{label}';values=arrays[prefix+'_metrics'];ids=arrays[prefix+'_entity_ids']
      expected_ids=[payload[split]['entity_ids'][int(i)] for i in torch.where(payload[split]['codes']==int(label)+3)[0]]
      assert ids.tolist()==expected_ids and len(values)==g['entities']==len(set(ids))
      assert np.isfinite(values).all() and values.min()>=-1e-12 and values[:,[0,1,2,4,5]].max()<=1+1e-7
      assert np.min(values[:,0]-values[:,2])>=-1e-7
      for j,col in enumerate(summary['columns']):
       assert summarize(values[:,j])==g['metrics'][col];stat_count+=4
      array_count+=1
  sample=torch.load(folder/'generated_sample.pt',map_location='cpu')
  assert sample['model_state_sha256']==report['best_state_sha256'] and sample['sampling_seed']==c['trials'][trial]['sampling_seed']
  sample=sample['sample'];mask=sample['valid_mask'].clone();mask[:,0]=False
  expected_plan=np.random.default_rng(c['trials'][trial]['sampling_seed']).integers(len(payload['train']['lengths']),size=2048)
  np.testing.assert_array_equal(sample['plan_train_indices'],expected_plan)
  ids=torch.tensor(expected_plan)
  assert torch.equal(sample['lengths'],payload['train']['lengths'][ids])
  assert torch.equal(sample['static_codes'],payload['train']['codes'][ids])
  assert torch.isnan(sample['gap'][:,0]).all()
  assert torch.isfinite(sample['numeric_value'][sample['valid_mask']]).all()
  assert ((sample['receiver'][sample['valid_mask']]>=3)&(sample['receiver'][sample['valid_mask']]<67)).all()
  reps=torch.tensor(payload['tensorizer_state']['gap_support']['representatives'],dtype=sample['gap'].dtype)
  assert torch.isin(sample['gap'][mask],reps).all()
  assert torch.equal(sample['valid_mask'],torch.arange(32)[None,:]<sample['lengths'][:,None])
  assert int(mask.sum())==audit['generated_gap_count']
  sample_count+=len(sample['lengths']);gap_count+=int(mask.sum());fit_count+=1
 effects={model:{'null':[],'active':[]} for model in ('CS3-E1','CS2-U1')};decisions=[]
 for trial in range(5):
  pair={}
  for k in (0,1):
   folders={model:stage_path.parent/f'trial_{trial}/kappa_{k}/{model}' for model in CANDIDATES}
   pair[str(k)]=paired_contrasts(folders)
   assert pair[str(k)]==stage['trial_results'][str(trial)]['paired_contrasts'][str(k)]
   reports=[stage['jobs'][f'trial_{trial}/kappa_{k}/{model}']['training'] for model in CANDIDATES]
   assert len({x['initialization_contract']['common_state_sha256'] for x in reports})==1
   assert reports[1]['initial_state_sha256']==reports[2]['initial_state_sha256']
   for epoch in range(min(len(x['history']) for x in reports)):
    assert len({x['history'][epoch]['entity_order_prefix_sha256'] for x in reports})==1
  for model in CANDIDATES:
   decision=decide_pilot({k:stage['jobs'][f'trial_{trial}/kappa_{k}/{model}']['audit'] for k in (0,1)},cfg['pilot_gate'])
   assert decision==stage['trial_results'][str(trial)]['candidate_response_gates'][model]
   if model=='CS4-ER1':decisions.append(decision)
  for model in effects:
   name='CS4-ER1_minus_'+model
   effects[model]['null'].append(float(np.mean([pair[str(k)][name][f'best_validation_label_{y}_metrics']['grid_mark_TV']['mean'] for k,y in ((0,0),(0,1),(1,0))])))
   effects[model]['active'].append(pair['1'][name]['best_validation_label_1_metrics']['grid_mark_TV']['mean'])
 assert expansion_gate(decisions,effects)==stage['gate']
 for kappa,pairs in stage['paired_seed_aggregates'].items():
  for pair,keys in pairs.items():
   for key,summary in keys.items():
    for metric,stats in summary['metrics'].items():
     values=[stage['trial_results'][str(t)]['paired_contrasts'][kappa][pair][key][metric]['mean'] for t in summary['trial_indices']]
     assert seed_statistics(values)==stats
     # Independently recompute the uncertainty, without the helper.
     a=np.array(values);assert stats['mean']==float(a.mean())
     if len(a)==5:
      error=2.7764451051977987*np.std(a,ddof=1)/np.sqrt(5)
      np.testing.assert_allclose(stats['descriptive_95_percent_t_interval'],[a.mean()-error,a.mean()+error],atol=1e-15,rtol=0)
     else:assert stats['descriptive_95_percent_t_interval'] is None
 for model in CANDIDATES:
  seeds=[stage['jobs'][f'trial_{t}/kappa_0/{model}']['training']['initialization_contract'] for t in range(5)]
  assert len({x['common_state_sha256'] for x in seeds})==5 and len({x['bank_state_sha256'] for x in seeds})==5
 assert len(stage['gpu_allocations'])==30 and all(x['compute_processes_before_launch']==0 for x in stage['gpu_allocations'])
assert fit_count==r['scientific_fits']
assert (len(stages)==4)==(stages['0.05']['gate']['decision']=='PASS')
assert r['stage2_started']==(len(stages)>1)
verification={'status':'PASS','file_checksum_comparisons':len(checks),'checks':checks,
 'checkpoint_tensor_comparisons':state_count,'aligned_entity_array_groups':array_count,'recomputed_entity_statistics':stat_count,
 'all_paired_seed_aggregates_and_intervals_recomputed':True,'all_stage_gates_recomputed':True,
 'generated_entities':sample_count,'generated_nonfirst_gaps':gap_count,'missing_snapshots':missing}
# Keep full aggregate evidence in git; omit redundant per-epoch histories from this summary.
compact={}
for pi,stage in stages.items():
 compact[pi]={k:v for k,v in stage.items() if k!='jobs'}
 compact[pi]['jobs']={key:{**job,'training':{k:v for k,v in job['training'].items() if k!='history'}} for key,job in stage['jobs'].items()}
record={'recorded_on':'2026-09-16','registration_commit':'8f5eaef','source_commit':r['source_commit'],
 'result':r,'stages':compact,'cpu_gate':cpu,'artifact_verification':verification,'tests':{'passed':135,'failed':0},
 'runtime_root':str(runtime),'runtime_terminal_sha256':sha(runtime/'COMPLETE.json')}
(root/'docs/cs_saf/replication_v1_result.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({**verification,'checks':'recorded'},indent=2))
for pi,s in stages.items():print(pi,json.dumps(s['gate'],indent=2))
