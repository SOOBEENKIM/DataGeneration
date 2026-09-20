"""Independent fit/conditional/native checks for every new gap-calibrated model."""
import json
from pathlib import Path
import numpy as np
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cs_saf_gap_calibration import ROOT,OUTPUT,CONFIG_SHA,contract,folder_for,verify,sha256,write_json,repeats
from experiments.cs_saf_replication import tensor_digest
from scripts.verify_cs_saf_calibration_u_native import sample_table,scores


def main():
    c=contract();torch.set_num_threads(1);terminal=json.loads((OUTPUT/'GRID_COMPLETE.json').read_text())
    if terminal['calibration_fits']!=80 or terminal['new_generated_datasets']!=400:raise ValueError('incomplete grid')
    if terminal['config_sha256']!=CONFIG_SHA:raise ValueError('terminal contract mismatch')
    for gate in ('cpu_gate','gpu_gate'):
        gm=verify(OUTPUT/gate);g=json.loads((OUTPUT/gate/'gate.json').read_text())
        if gm['source_commit']!=terminal['source_commit'] or g['source_commit']!=terminal['source_commit'] or g['decision']!='PASS':raise ValueError('gate source mismatch')
    previous=json.loads((ROOT/'docs/cs_saf/generation_repeats_v1_result.json').read_text())
    if sha256(repeats.OUTPUT/'native_verification.json')!=previous['native_verification_sha256']:raise ValueError('old reference verification changed')
    refs={}
    for r in json.loads((repeats.OUTPUT/'native_verification.json').read_text())['records']:
        key=(r['pi'],r['kappa'],r['group']);value=r['reference_counts']
        if key in refs and refs[key]!=value:raise ValueError('inconsistent reference')
        refs[key]=value
    max_native=0.;max_cond=0.;max_fit=0.;n=0;fits=0;cond=0;records=[];maps={};max_historical=0.
    for pi in c['prevalences']:
        for k in c['kappas']:
            payload=repeats.parent.load_cache(repeats.parent.CACHE/f'pi_{pi:.2f}_kappa_{k}.pt');train=payload['train']
            expected_mask=train['valid_mask'].clone();expected_mask[:,0]=False
            expected_rows,expected_cols=torch.where(expected_mask)
            support=SAFTensorizerState.from_dict(payload['tensorizer_state']).gap_support
            expected_support=np.searchsorted(np.asarray(support.upper_bounds[:-1],dtype=np.float32),
                train['gap'][expected_mask].numpy(),side='left')
            reps=np.asarray(support.representatives,dtype=np.float32).astype(float)
            all_codes=train['codes'][expected_rows].numpy();expected_maps=[];expected_edges=[]
            for code in (3,4):
                edges=np.quantile(reps[expected_support[all_codes==code]],[.2,.4,.6,.8],method='linear')
                expected_edges.append(edges.tolist());expected_maps.append(np.searchsorted(edges,reps,side='right').tolist())
            for t in c['trials']:
                plan=torch.load(repeats.folder_for(pi,k,t)/'plan.pt',map_location='cpu')
                for parent,name in c['variants'].items():
                    folder=folder_for(pi,k,t,name);m=verify(folder)
                    if m['source_commit']!=terminal['source_commit']:raise ValueError('mixed sources')
                    fit=json.loads((folder/'fit.json').read_text());b=json.loads((folder/'train_bins.json').read_text())
                    if (pi,k) in maps and maps[(pi,k)]!=b:raise ValueError('unequal U/E/trial bin definitions')
                    maps[(pi,k)]=b;f=np.load(folder/'train_features.npz');cp=torch.load(folder/'checkpoint_gap_calibrated.pt',map_location='cpu')
                    old={a:v for a,v in cp['model_state'].items() if not a.startswith('gap_bin_')}
                    if tensor_digest(old)!=fit['parent']['state_sha256']:raise ValueError('old neural/affine tensors changed')
                    if sha256(Path(fit['parent']['checkpoint_path']))!=fit['parent']['checkpoint_sha256']:raise ValueError('old checkpoint changed')
                    rows,cols=f['entity_index'],f['event_index']
                    np.testing.assert_array_equal(rows,expected_rows.numpy())
                    np.testing.assert_array_equal(cols,expected_cols.numpy())
                    np.testing.assert_array_equal(f['support_bin'],expected_support)
                    if b['mapping']!=expected_maps or b['edges']!=expected_edges:raise ValueError('independent train bin reconstruction differs')
                    np.testing.assert_array_equal(f['bin'],np.asarray(b['mapping'])[f['code'].astype(int)-3,f['support_bin']])
                    np.testing.assert_array_equal(cp['model_state']['gap_bin_map'].numpy(),np.asarray(b['mapping']))
                    np.testing.assert_allclose(cp['model_state']['gap_bin_weights'].numpy(),np.asarray(b['weights']),rtol=0,atol=0)
                    precision=json.loads((folder/'response_precision.json').read_text())
                    if precision['decision']!='PASS' or not precision['tf32_flags_restored']:raise ValueError('response precision gate failed')
                    response=json.loads((folder/'intervention_audit.json').read_text())
                    if response['zero_gap_control_max_range']>1e-8:raise ValueError('zero-gap control failed')
                    np.testing.assert_array_equal(f['equality'],(train['receiver'][rows,cols]==train['receiver'][rows,cols-1]).numpy())
                    np.testing.assert_array_equal(f['code'],train['codes'][rows].numpy())
                    if len(rows)!=int((train['lengths']-1).sum()) or (cols<1).any():raise ValueError('incorrect fitting transitions')
                    for s in (0,1):
                        ix=f['code']==s+3;counts=np.bincount(f['bin'][ix],minlength=5);w=counts/counts.sum();np.testing.assert_array_equal(counts,b['counts'][s])
                        d=np.array(fit['delta'][s]);z=f['logit'][ix].astype(float)+d[f['bin'][ix]];fp=f['fresh_previous'][ix].astype(float)
                        lr=np.logaddexp(-np.logaddexp(0.,-z),-np.logaddexp(0.,z)+np.log(fp))
                        ln=-np.logaddexp(0.,z)+np.log1p(-fp);nll=float(-np.where(f['equality'][ix],lr,ln).mean())
                        penalty=.5*c['lambda_ridge']*float(w@d**2)
                        block=fit['groups'][str(s)];err=max(abs(nll-block['after_nll']),abs(nll+penalty-block['after_objective']))
                        max_fit=max(max_fit,err)
                        if err>1e-10 or abs(w@d)>1e-8 or abs(d).max()>.5+1e-8 or nll+penalty>block['before_nll']+1e-10:raise ValueError('independent fitting check failed')
                        if (np.abs(cp['model_state']['gap_bin_delta'][s].numpy()-d)>3e-8).any():raise ValueError('deployed offset differs')
                    fits+=1
                    for prefix in ('','control_'):
                        summary=json.loads((folder/(prefix+'conditional_accuracy.json')).read_text());a=np.load(folder/(prefix+'conditional_arrays.npz'))
                        for label in ('0','1'):
                            values=a[f'label_{label}_metrics'];ids=a[f'label_{label}_entity_ids']
                            expected_ids=np.array(payload['validation']['entity_ids'])[payload['validation']['codes'].numpy()==int(label)+3]
                            np.testing.assert_array_equal(ids,expected_ids)
                            if not np.isfinite(values).all():raise ValueError('nonfinite conditional values')
                            for j,metric in enumerate(summary['columns']):
                                err=abs(float(values[:,j].mean())-summary['groups'][label]['metrics'][metric]['mean']);max_cond=max(max_cond,err);cond+=1
                                if err>1e-12:raise ValueError('conditional mean mismatch')
                    diffs=json.loads((folder/'historical_conditional_differences.json').read_text())
                    max_historical=max(max_historical,max(abs(v) for group in diffs.values() for v in group.values()))
                    modelsha=tensor_digest(cp['model_state'])
                    for r in c['repeats']:
                        p=folder/f'repeat_{r}';saved=torch.load(p/'generated_sample.pt',map_location='cpu');sample=saved['sample']
                        if saved['model_state_sha256']!=modelsha or saved['sampling_seed']!=c['generation_seeds'][str(t)][r]:raise ValueError('sample identity mismatch')
                        for key,value in [('static_codes',plan['contexts']),('lengths',plan['lengths']),('plan_train_indices',torch.as_tensor(plan['positions']))]:torch.testing.assert_close(sample[key],value,rtol=0,atol=0)
                        out=json.loads((p/'comparison.json').read_text())
                        for g,label in [('pooled',None),('context_0',0),('context_1',1)]:
                            block=out['metrics'][g];edges=np.array(block['train_metric_state']['gap_bin_edges']);table=sample_table(sample,label,edges)
                            actual=scores(np.array(refs[(pi,k,g)]),table);expected=[block['metrics']['short_gap_repeat_curve_l1'],block['metrics']['gap_repeat_mi_error']]
                            err=float(np.max(abs(np.array(actual)-expected)));max_native=max(max_native,err);n+=1
                            if err>1e-12:raise ValueError('native metric mismatch')
                            records.append(dict(pi=pi,kappa=k,trial=t,model=name,repeat=r,group=g,L1=actual[0],MI=actual[1]))
                print(f'verified {pi} {k} {t}',flush=True)
    result=dict(status='PASS',source_commit=terminal['source_commit'],fits_checked=fits,old_states_unchanged=fits,
        conditional_means_checked=cond,native_groups_checked=n,native_values_checked=2*n,
        exact_train_transition_order_checked=fits,response_precision_checks=fits,zero_gap_checks=fits,
        max_native_error=max_native,max_conditional_error=max_cond,max_fit_error=max_fit,
        max_historical_control_mean_difference=max_historical,verifier_sha256=sha256(Path(__file__)),records=records)
    write_json(OUTPUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
