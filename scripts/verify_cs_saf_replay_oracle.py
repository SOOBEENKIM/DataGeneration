"""Independent arithmetic checks of saved cross-history/oracle evidence."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cs_saf_rollout_audit as parent
from experiments.cs_saf_followup import folder_for, verify_artifacts
from experiments.cs_saf_pilot import state_digest
from benchmarks import cs_saf_joint_oracle as joint
from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
import yaml

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def verify(path):
    m=read(path/'COMPLETE.json')
    for name,h in m['artifact_sha256'].items(): assert digest(path/name)==h, str(path/name)
    return m

def metrics(data, prob, ref, edges, label):
    valid=data['valid_mask'].numpy().copy(); valid[:,0]=False
    valid &= (data['codes'].numpy()==label+3)[:,None]
    row, t=np.nonzero(valid)
    gap=data['gap'].numpy()[valid]; mark=data['receiver'].numpy()
    event=(mark[row,t]==mark[row,t-1]); bins=np.digitize(gap,edges,right=False)
    counts=np.array([(bins==b).sum() for b in range(5)])
    empirical=np.array([event[bins==b].mean() for b in range(5)])
    predicted=np.array([prob[valid][bins==b].mean() for b in range(5)])
    target=np.array(ref['observed_repeat']); w=np.array(ref['counts'])/sum(ref['counts'])
    table=np.array([[(event[bins==b]==v).sum() for v in (False,True)] for b in range(5)],float)
    table/=table.sum(); marg=table.sum(1)[:,None]*table.sum(0)[None,:]; use=table>0
    mi=(table[use]*np.log(table[use]/marg[use])).sum()
    return {'counts':counts, 'observed_repeat':empirical,'model_repeat':predicted,
            'expected_repeat_L1':w@abs(predicted-target), 'empirical_repeat_L1':w@abs(empirical-target),
            'mi_error':abs(mi-ref['mi'])}

def check_stats(actual, values):
    v=np.array(values); mean=v.mean(); half=2.7764451051977987*v.std(ddof=1)/np.sqrt(5)
    np.testing.assert_allclose(actual['values'],v,atol=1e-12,rtol=0)
    np.testing.assert_allclose([actual['mean'],*actual['ci95_descriptive']],[mean,mean-half,mean+half],atol=1e-12,rtol=0)
    assert actual['negative_trials']==int((v<0).sum()) and actual['positive_trials']==int((v>0).sum())

c=read(ROOT/'configs/benchmark_v2/cs_saf_replay_oracle_v1.json'); out=ROOT/c['output']; result=read(out/'compact_result.json')
assert digest(Path(result['full_evidence']['path']))==result['full_evidence']['sha256']
counts={'output_checksums':0,'parent_checksums':0,'checkpoint_file_and_state_identities':0,'replay_group_metric_checks':0,'oracle_group_metric_checks':0,'oracle_replay_arrays':0,'decomposition_statistic_groups':0,'oracle_statistic_groups':0}
max_metric_error=0.; reports={}; controls={}
for pi in c['prevalences']:
  for k in c['kappas']:
    payload,edges,panel,ids=parent.load_cell(pi,k)
    support=SAFTensorizerState.from_dict(payload['tensorizer_state']).gap_support
    config=BenchmarkConfig.from_mapping(yaml.safe_load((ROOT/'configs/benchmark_v2/full_v2_5.yaml').read_text()),'joint_semimarkov_v2b',k)
    oracle=SemiMarkovCopyOracle(config,np.asarray(support.upper_bounds,np.float32).astype(float))
    for trial in c['trials']:
      path=out/f'pi_{pi:.2f}/kappa_{k}/trial_{trial}'; m=verify(path); counts['output_checksums']+=len(m['artifact_sha256'])
      assert m==result['output_manifests'][f'{pi:.2f}/{k}/{trial}']
      rep=read(path/'replay.json'); ctr=read(path/'oracle.json'); reports[pi,k,trial]=rep; controls[pi,k,trial]=ctr
      assert rep['panel_entity_ids']==ids
      sources={}
      for candidate in c['models']:
        p=ROOT/f'artifacts/cs_saf/rollout_audit_v1/pi_{pi:.2f}/kappa_{k}/trial_{trial}/{candidate}'
        parent_m=verify(p); assert parent_m==rep['parent_manifests'][candidate]
        counts['parent_checksums']+=len(parent_m['artifact_sha256'])
        sources[candidate]=torch.load(p/'panel_rollouts.pt',map_location='cpu',weights_only=False)['samples']
        original=folder_for(pi,k,trial,candidate); manifest=verify_artifacts(original)
        assert manifest==rep['provenance'][candidate]['input_manifest']
        checkpoint=original/'checkpoint_best.pt'; assert digest(checkpoint)==rep['provenance'][candidate]['checkpoint_sha256']
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        # Same canonical tensor digest convention, without loading a model.
        class State:
          def state_dict(self): return cp['model_state']
        assert state_digest(State())==rep['provenance'][candidate]['state_sha256']
        counts['checkpoint_file_and_state_identities']+=1
      with np.load(path/'replay_probabilities.npz') as arrays:
        assert set(arrays.files)==set(rep['matrices'])
        for key, scores in rep['matrices'].items():
          mode,seedpart,targetpart,sourcepart=key.split('/'); source=sourcepart.removeprefix('source_'); seed=seedpart.removeprefix('seed_')
          data=sources[source][mode+'_'+seed]
          for label in (0,1):
            computed=metrics(data,arrays[key],rep['reference'][str(label)],edges[str(label)],label)
            for metric,value in computed.items():
              error=float(np.max(abs(np.asarray(value)-scores[str(label)][metric]))); max_metric_error=max(max_metric_error,error)
              assert error<1e-12, (key,metric,error)
            counts['replay_group_metric_checks']+=1
      paths=torch.load(path/'oracle_samples.pt',map_location='cpu',weights_only=False)
      assert paths['panel_entity_ids']==ids and set(paths['samples'])==set(ctr['controls'])
      for key,data in paths['samples'].items():
        mode=key.rsplit('_',1)[0]; info='MODELINFO' if mode=='FIX_MODELINFO' else ('BIN' if mode.endswith('BIN') else 'CONT')
        probability=data['predicted_repeat'].numpy()
        np.testing.assert_allclose(joint.predict(oracle,{field:data[field] for field in parent.FIELDS},k,info),probability,atol=1e-12,rtol=0)
        counts['oracle_replay_arrays']+=1
        for field in ('numeric_value','codes','lengths','valid_mask'):
          torch.testing.assert_close(data[field],panel[field],equal_nan=True,atol=0,rtol=0)
        torch.testing.assert_close(data['receiver'][:,0],panel['receiver'][:,0],atol=0,rtol=0)
        if mode in ('FIX_CONT','FIX_MODELINFO'): torch.testing.assert_close(data['gap'],panel['gap'],equal_nan=True,atol=0,rtol=0)
        for reference in ('raw_reference','matched_reference'):
          ref=rep['quantized_reference'] if reference=='matched_reference' and info=='BIN' else rep['reference']
          for label in (0,1):
            for metric,value in metrics(data,probability,ref[str(label)],edges[str(label)],label).items():
              error=float(np.max(abs(np.asarray(value)-ctr['controls'][key][reference][str(label)][metric]))); max_metric_error=max(max_metric_error,error)
              assert error<1e-12,(key,reference,metric,error)
            counts['oracle_group_metric_checks']+=1

for pi in c['prevalences']:
  for k in c['kappas']:
    for label in (0,1):
      cell=f'kappa_{k}_context_{label}'
      for mode in c['replay_modes']:
        for a,b in c['pairs']:
          vectors=[]
          for target,source in ((a,a),(a,b),(b,a),(b,b)):
            v=[]
            for trial in c['trials']:
              vals=[s[str(label)]['expected_repeat_L1'] for key,s in reports[pi,k,trial]['matrices'].items() if key.startswith(mode+'/') and key.endswith(f'/target_{target}/source_{source}')]
              assert len(vals)==3; v.append(np.mean(vals))
            vectors.append(np.array(v))
          aa,ab,ba,bb=vectors
          effects={'predictor_F':(aa-ba+ab-bb)/2,'source_history_H':(aa-ab+ba-bb)/2,'diagonal_total':aa-bb}
          for name,values in effects.items():
            check_stats(result['replay_summary'][f'{pi:.2f}'][cell][mode]['effects'][a+'_minus_'+b][name],values)
            counts['decomposition_statistic_groups']+=1
      for mode in c['oracle_modes']:
        for ref in ('raw_reference','matched_reference'):
          for metric in ('expected_repeat_L1','empirical_repeat_L1','mi_error'):
            values=[np.mean([s[ref][str(label)][metric] for key,s in controls[pi,k,trial]['controls'].items() if key.startswith(mode+'_')]) for trial in c['trials']]
            check_stats(result['oracle_summary'][f'{pi:.2f}'][cell][mode][ref][metric],values)
            counts['oracle_statistic_groups']+=1
# Audit every published bin summary from raw per-tape probabilities, and the
# retrospective cost calculation against the immutable native-generation result.
binwise=read(out/'binwise_result.json'); bin_stats=0
for pi in c['prevalences']:
  for k in c['kappas']:
    for label in ('0','1'):
      cell=binwise['cells'][f'pi_{pi:.2f}/kappa_{k}'][label]
      for group in ('replay','oracle'):
        for key,summary in cell[group].items():
          all_values=[]
          if group=='replay':
            mode,target,source=key.split('/')
            for trial in c['trials']:
              entries=[s[label]['model_repeat'] for name,s in reports[pi,k,trial]['matrices'].items() if name.startswith(mode+'/') and name.endswith('/'+target+'/'+source)]
              assert len(entries)==3; all_values.append(np.mean(entries,axis=0))
            reference=cell['raw_reference']['observed_repeat']
          else:
            mode,ref=key.split('/')
            for trial in c['trials']:
              entries=[s[ref][label]['model_repeat'] for name,s in controls[pi,k,trial]['controls'].items() if name.startswith(mode+'_')]
              assert len(entries)==3; all_values.append(np.mean(entries,axis=0))
            reference=cell['quantized_reference' if ref=='matched_reference' and mode.endswith('BIN') else 'raw_reference']['observed_repeat']
          values=np.array(all_values)
          for b in range(5):
            check_stats(summary['predicted_repeat'][b],values[:,b])
            check_stats(summary['signed_bias'][b],values[:,b]-reference[b]);bin_stats+=2
counts['binwise_statistic_groups']=bin_stats
parent_result=read(ROOT/'docs/cs_saf/rollout_audit_v1_result.json')
raw={(item['pi'],item['kappa'],item['trial'],item['candidate']):item for item in parent_result['raw_reports']}
for pi in c['prevalences']:
  tradeoff=result['tradeoff_frontier'][f'{pi:.2f}']
  for metric in ('empirical_repeat_L1','mi_error'):
    e=np.array([raw[pi,1,t,'E']['native']['1'][metric] for t in c['trials']])
    l=np.array([raw[pi,1,t,'L003']['native']['1'][metric] for t in c['trials']]); delta=l-e
    item=tradeoff[metric];check_stats(item['difference'],delta)
    np.testing.assert_allclose([item['E_mean'],item['L003_mean'],item['required_mean_tolerance'],item['relative_cost_percent_ratio_of_means']], [e.mean(),l.mean(),max(0.,delta.mean()),100*delta.mean()/e.mean()],atol=1e-12,rtol=0)
    assert item['required_descriptive_upper_tolerance']==max(0.,item['difference']['ci95_descriptive'][1])
    assert item['justified_acceptance_margin'] is None and item['retroactive_PASS'] is False
  null=[]
  for trial in c['trials']:
    pair=[]
    for candidate in ('L003','E'):
      cells=[read(folder_for(pi,k,trial,candidate)/'intervention_audit.json')['responses'][str(label)]['mean_copy_range'] for k,label in ((0,0),(0,1),(1,0))]
      pair.append(np.mean(cells))
    null.append(pair[0]-pair[1])
  check_stats(tradeoff['equal_three_null_copy_range_change'],null)
  check_stats(tradeoff['active_grid_TV_change'],[raw[pi,1,t,'L003']['TF_all']['1']['conditional']['grid_mark_TV']-raw[pi,1,t,'E']['TF_all']['1']['conditional']['grid_mark_TV'] for t in c['trials']])
counts['tradeoff_statistic_groups']=16
payload,_,_,_=parent.load_cell(.05,1)
support=SAFTensorizerState.from_dict(payload['tensorizer_state']).gap_support
config=BenchmarkConfig.from_mapping(yaml.safe_load((ROOT/'configs/benchmark_v2/full_v2_5.yaml').read_text()),'joint_semimarkov_v2b',1)
oracle=SemiMarkovCopyOracle(config,np.asarray(support.upper_bounds,np.float32).astype(float))
mc=joint.monte_carlo_gate(oracle,support.representatives,n_per_group=c['oracle_mc_gate_entities_per_group'],seed=c['oracle_mc_gate_seed'])
verified={'PASS':True,'config_sha256':digest(ROOT/'configs/benchmark_v2/cs_saf_replay_oracle_v1.json'),'source_commit':result['execution']['source_commit'],'compact_result_sha256':digest(out/'compact_result.json'),'verifier_sha256':digest(Path(__file__)),'binwise_result_sha256':digest(out/'binwise_result.json'),'joint_sampler_support_prevalence':.05,'counts':counts,'maximum_independent_metric_error':max_metric_error,'joint_sampler_moment_checks':mc,'scope':'All stored arrays/metrics and decompositions; independent histogram/MI arithmetic, same filter replay plus separate dense-recursion/factorization CPU tests. No new fitting.'}
(out/'independent_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
print(json.dumps(verified,indent=2))
