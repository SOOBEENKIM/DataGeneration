"""Verify and preserve the completed v4 pilot and read-only comparator evidence."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cs_saf_v4 import load_contract,summarize,contrasts,load_cache,accuracy_screens,CANDIDATE
from experiments.cs_saf_pilot import decide_pilot
root=Path(__file__).resolve().parents[1];contract,cfg=load_contract();runtime=root/'artifacts/cs_saf/revision_v4/gpu_v1'
r=json.loads((runtime/'COMPLETE.json').read_text())
cpu_path=root/'artifacts/cs_saf/revision_v4/cpu_v1/COMPLETE.json';cpu=json.loads(cpu_path.read_text())
parent=json.loads((root/contract['parent_result']).read_text());old=parent['result'];oldroot=root/contract['execution']['saved_control_root']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(p,expected,role):
 actual=sha(p);assert actual==expected,(p,actual,expected)
 checks.append({'path':str(p),'sha256':actual,'role':role})
assert r['status']=='COMPLETE' and r['scientific_fits']==2 and not r['test_accessed'] and not r['later_stage_started']
assert r['saved_comparator_new_fits']==0 and r['matched_saved_controls']
assert cpu['decision']=='PASS' and cpu['source_commit']==r['source_commit']
check(cpu_path,r['cpu_gate_sha256'],'CPU gate')
check(root/'artifacts/cs_saf/prepared_v1/COMPLETE.json',r['prepared_index_sha256'],'prepared index')
check(root/'configs/benchmark_v2/cs_saf_revision_v4.yaml',r['config_sha256'],'preregistration')
for key in ('parent_result','inherited_contract'):
 check(root/contract[key],contract[key+'_sha256'],key)
for p,h in contract['frozen_parent_files'].items():check(root/p,h,'frozen inherited implementation')
check(oldroot/'COMPLETE.json',parent['runtime_terminal_sha256'],'saved v3 terminal')
payloads={}
for k in (0,1):
 path=root/f'artifacts/cs_saf/prepared_v1/pi_0.05_kappa_{k}.pt'
 check(path,r['training'][f'pi_0.05_kappa_{k}/{CANDIDATE}']['cache_sha256'],'prepared cache')
 payloads[k]=load_cache(path)
state_checks=0;array_groups=0;statistics=0;samples=0;gaps=0
entries={key:(runtime/key,t,r['conditional_accuracy'][key],True) for key,t in r['worker_terminals'].items()}
entries.update({key:(oldroot/key,e['terminal'],e['diagnostic'],False) for key,e in r['reused_controls'].items()})
for key,(folder,terminal,diag,is_new) in entries.items():
 payload=payloads[int(key.split('/')[0][-1])]
 assert json.loads((folder/'COMPLETE.json').read_text())==terminal
 assert terminal['source_commit']==(r['source_commit'] if is_new else old['source_commit'])
 for name,expected in terminal['artifact_sha256'].items():check(folder/name,expected,'new artifact' if is_new else 'read-only comparator artifact')
 assert json.loads((folder/'conditional_accuracy.json').read_text())==diag
 with np.load(folder/'accuracy_arrays.npz',allow_pickle=False) as data:
  for name,item in diag['checkpoints'].items():
   cp_path=Path(item['checkpoint_path']);check(cp_path,item['checkpoint_sha256'],'diagnosed checkpoint')
   cp=torch.load(cp_path,map_location='cpu');state=cp['model_state'];d=hashlib.sha256()
   for sk,t in sorted(state.items()):d.update(sk.encode());d.update(t.contiguous().numpy().tobytes())
   assert d.hexdigest()==item['state_sha256'];state_checks+=1
   if is_new:
    assert cp['source_commit']==r['source_commit'] and cp['config_sha256']==r['config_sha256']
    assert cp['candidate']==CANDIDATE
    np.testing.assert_array_equal(state['reference_probabilities'].numpy(),np.asarray(item['reference_probabilities']))
   for split,s in item['splits'].items():
    for label,g in s['groups'].items():
     prefix=f'{name}_{split}_label_{label}'
     values=data[prefix+'_metrics'];ids=data[prefix+'_entity_ids']
     expected_ids=[payload[split]['entity_ids'][int(i)] for i in torch.where(payload[split]['codes']==int(label)+3)[0]]
     assert ids.tolist()==expected_ids
     assert len(values)==g['entities']==len(set(ids)) and np.isfinite(values).all()
     assert np.min(values)>=-1e-12 and np.max(values[:,[0,1,2,4,5]])<=1+1e-7
     assert np.min(values[:,0]-values[:,2])>=-1e-7
     for j,col in enumerate(s['columns']):
      actual=summarize(values[:,j]);assert actual==g['metrics'][col];statistics+=len(actual)
     array_groups+=1
 if is_new:
  report=r['training'][key];assert report==json.loads((folder/'training_report.json').read_text())
  assert report['parameters']==133581 and report['seed']==20260930 and not report['test_accessed']
  assert report['initialization_contract']['fresh_v2_common_initialization']
  assert report['architecture']['candidate']==CANDIDATE and not report['architecture']['centered']
  assert report['architecture']['regularization_coefficient']==.01
  saved=torch.load(folder/'generated_sample.pt',map_location='cpu')
  assert saved['model_state_sha256']==report['best_state_sha256']
  sample=saved['sample'];mask=sample['valid_mask'].clone();mask[:,0]=False
  samples+=len(sample['lengths']);gaps+=int(mask.sum())
  assert len(sample['lengths'])==2048 and torch.isfinite(sample['numeric_value'][sample['valid_mask']]).all()
  assert torch.isnan(sample['gap'][:,0]).all()
  assert ((sample['receiver'][sample['valid_mask']]>=3)&(sample['receiver'][sample['valid_mask']]<67)).all()
  reps=torch.tensor(payload['tensorizer_state']['gap_support']['representatives'],dtype=sample['gap'].dtype)
  assert torch.isin(sample['gap'][mask],reps).all()
  assert torch.equal(sample['valid_mask'],torch.arange(32)[None,:]<sample['lengths'][:,None])
  oldkey=key.replace(CANDIDATE,'CS3-E1')
  oldsample=torch.load(oldroot/oldkey/'generated_sample.pt',map_location='cpu')['sample']
  assert torch.equal(sample['lengths'],oldsample['lengths'])
  for field in ('initial_state_sha256','cache_sha256','seed','options','parameters','initialization_contract'):
   assert report[field]==old['training'][oldkey][field]
  # CPU uses kappa=1; the reference buffer differs across data cells.
  if key.startswith('pi_0.05_kappa_1/'):
   assert report['initial_state_sha256']==cpu['candidates'][CANDIDATE]['initial_state_sha256']
for k in (0,1):
 folders={c:oldroot/f'pi_0.05_kappa_{k}'/c for c in contract['saved_comparators']}
 folders[CANDIDATE]=runtime/f'pi_0.05_kappa_{k}'/CANDIDATE
 assert contrasts(folders)==r['paired_contrasts'][str(k)]
 # Independent algebra checks against separately recomputed direct contrasts.
 for key,metrics in r['paired_contrasts'][str(k)]['penalty_by_forward_interaction'].items():
  for m,s in metrics.items():
   expected=r['paired_contrasts'][str(k)]['ER_minus_E'][key][m]['mean']-r['paired_contrasts'][str(k)]['R_minus_C'][key][m]['mean']
   assert abs(s['mean']-expected)<1e-12
assert accuracy_screens(r['paired_contrasts'])==r['accuracy_screen']
gate=decide_pilot({k:r['audits'][f'pi_0.05_kappa_{k}/{CANDIDATE}'] for k in (0,1)},cfg['pilot_gate'])
assert gate==r['response_gate']
assert r['next_registered_pilot_eligible']==(gate['decision']=='PASS' and all(s['PASS'] for s in r['accuracy_screen'].values()))
oldcpu=json.loads((root/'artifacts/cs_saf/revision_v3/cpu_v2/COMPLETE.json').read_text())
assert cpu['candidates'][CANDIDATE]['initial_state_sha256']==oldcpu['candidates']['CS3-E1']['initial_state_sha256']
verification={'status':'PASS','file_checksum_comparisons':len(checks),'checks':checks,
 'checkpoint_tensor_comparisons':state_checks,'entity_array_groups':array_groups,'recomputed_summary_statistics':statistics,
 'paired_contrasts_and_factorial_interaction_recomputed':True,'CPU_initial_state_matches_saved_E':True,
 'saved_sample_support_reserved_codes_lengths_and_values_rechecked':True,'generated_entities':samples,'generated_nonfirst_gaps':gaps}
record={'recorded_on':'2026-09-16','registration_commit':'642b878','result':r,'cpu_gate':cpu,
 'artifact_verification':verification,'tests':{'passed':123,'failed':0,'commands':[
 'python3 -m pytest -q tests/test_cs_saf_v4.py tests/test_cs_saf_v3.py tests/test_cs_saf_v2.py tests/test_cs_saf_model.py tests/test_cs_saf_oracle.py tests/test_cs_saf_prevalence.py tests/test_cs_saf_loss_control.py tests/test_cs_saf_forensics.py tests/test_cs_saf_route_decomposition.py',
 'python3 -m pytest -q tests/test_cof_seqgen_saf_model.py tests/test_cof_seqgen_saf_tensorizer.py tests/test_cof_seqgen_saf_contract.py']},
 'runtime_root':str(runtime),'runtime_terminal_sha256':sha(runtime/'COMPLETE.json'),
 'GPU_allocation':{'index':2,'uuid':'GPU-f1d4c556-ae70-3415-adec-c1d68f9bb637','observed_before_start_memory_MiB':15,'observed_before_start_utilization_percent':0,'observed_before_start_compute_processes':0,'other_busy_indices_avoided':[0,1,3],'first_worker_pid':321594}}
(root/'docs/cs_saf/v4_pilot_v1_result.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(verification|{'checks':'recorded'},indent=2))
print(json.dumps({'response_gate':gate,'accuracy_screen':r['accuracy_screen'],'eligible':r['next_registered_pilot_eligible']},indent=2))
for k in (0,1):
 for c in ('CS2-U1','CS3-E1','CS3-C1','CS3-R1',CANDIDATE):
  key=f'pi_0.05_kappa_{k}/{c}'
  d=r['conditional_accuracy'][key] if c==CANDIDATE else old['conditional_accuracy'][key]
  for label in ('0','1'):
   g=d['checkpoints']['best']['splits']['validation']['groups'][label]['metrics']
   print(k,c,label,{m:round(g[m]['mean'],9) for m in ['grid_mark_TV','factual_mark_TV','grid_repeat_L1','repeat_BCE','copy_range','repeat_range','residual_RMS']})
