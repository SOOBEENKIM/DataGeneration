"""Prospective decision, complete inventory, and independently recomputed metrics."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cs_saf_rollout_calibration_v1 as run
from scripts.evaluate_cs_saf_rollout_calibration_v1 import inputs,evaluators,validate_frame
from scripts.summarize_cs_saf_structure_v1 import basic
from scripts.run_cs_saf_external_audit_v1 import digest,write
from benchmarks.cof_seqgen_saf_metrics import evaluate_metric_suite


def verify_evaluation_source(source):
    for file,expected in source['hashes'].items():
        snapshot=subprocess.check_output(['git','show',f'{source["commit"]}:{file}'],cwd=ROOT)
        assert hashlib.sha256(snapshot).hexdigest()==expected
        # The preserved model evaluations precede the oracle-only curve-column
        # naming fix. Verify that evaluator against its committed snapshot.
        # Every training/model/metric/oracle dependency must still match now.
        if file!='scripts/evaluate_cs_saf_rollout_calibration_v1.py':assert digest(ROOT/file)==expected


def decision(cond,means,responses):
    c=run.config();keys=['kappa','trial','name','variant','group']
    cm=cond.set_index(keys);gm=means.set_index(keys);rm=responses.set_index(keys)
    costs=[];decisions=[]
    for name in c['models']:
        for method in ['B','P']:
            local=[]
            for k in c['kappas']:
                for t in c['trials']:
                    for group in ['0','1']:
                        key=(k,t,name,method,group);basekey=(k,t,name,'A',group)
                        ca,cb=cm.loc[key],cm.loc[basekey];ga,gb=gm.loc[key],gm.loc[basekey]
                        brier=float(ca.repeat_brier-cb.repeat_brier);nll=float(ca.mark_nll-cb.mark_nll)
                        pc=c['prediction_cost_limits'];gc=c['generation_cost_limits']
                        delta={m:float(ga[m]-gb[m]) for m in ['gap_ks','mark_sparse_tv','amount_ks','length_scaled_w1']}
                        null=k==0 or group=='0';range_delta=float(rm.loc[key,'repeat_range']-rm.loc[basekey,'repeat_range'])
                        row=dict(kappa=k,trial=t,name=name,variant=method,group=group,
                            delta_repeat_brier=brier,delta_mark_nll=nll,delta_fixed_history_range=range_delta,
                            prediction_cost_pass=bool(brier<=pc['max_brier_increase'] and nll<=pc['max_mark_nll_increase']),
                            basic_prediction_pass=bool(all(basic(pd.Series(dict(row,kappa=k,group=group)),c['basic_prediction']) for row in [ca,cb])),
                            distribution_pass=bool(all(delta[m]<=gc['max_'+m+'_increase'] for m in delta)),
                            null_generation_pass=bool(not null or (ga.short_gap_repeat_curve_l1<=gc['max_null_repeat_curve_L1'] and ga.null_centered_shape_l1<=gc['max_null_centered_shape_L1'])),
                            null_response_pass=bool(not null or range_delta<=c['max_null_fixed_history_range_increase']),
                            **{'delta_'+m:v for m,v in delta.items()})
                        local.append(row);costs.append(row)
            av=[float(gm.loc[(1,t,name,'A','1'),'short_gap_repeat_curve_l1']) for t in c['trials']]
            cv=[float(gm.loc[(1,t,name,method,'1'),'short_gap_repeat_curve_l1']) for t in c['trials']]
            differences=np.array(cv)-av;gain=float(np.mean(av)-np.mean(cv));relative=gain/float(np.mean(av))
            gate=c['primary_generation_improvement']
            gp=bool(gain>=gate['minimum_absolute_mean_reduction'] and relative>=gate['minimum_relative_mean_reduction'] and
                sum(differences<0)>=gate['minimum_improving_training_seeds'] and max(differences)<=gate['maximum_single_seed_L1_increase'] and
                max(cv)<=gate['maximum_candidate_L1_each_seed'])
            screens={key:bool(all(row[key] for row in local)) for key in ['prediction_cost_pass','basic_prediction_pass',
                'distribution_pass','null_generation_pass','null_response_pass']}
            decisions.append(dict(name=name,variant=method,A_values=av,candidate_values=cv,
                paired_differences=differences.tolist(),absolute_reduction=gain,relative_reduction=relative,
                improving_seeds=int(sum(differences<0)),generation_improvement_pass=gp,**screens,joint_pass=bool(gp and all(screens.values()))))
    attribution={}
    for name in c['models']:
        b=next(x for x in decisions if x['name']==name and x['variant']=='B')
        p=next(x for x in decisions if x['name']==name and x['variant']=='P')
        attribution[name]=dict(protection_supported_in_this_pilot=bool(p['joint_pass'] and b['generation_improvement_pass'] and not b['prediction_cost_pass']),
            B_sufficient=bool(b['joint_pass']),P_passes=bool(p['joint_pass']))
    pd.DataFrame(costs).to_csv(run.DOC/'costs_by_parent.csv',index=False)
    write(run.DOC/'decision.json',dict(comparisons=decisions,attribution=attribution,
        thresholds_changed=False,previous_C_failure_unchanged=True,independent_data=False,
        no_statistical_superiority_claim=True,no_external_generalization_claim=True))
    return decisions


def independent_repeat(frame,reference,edges):
    def table(f):
        f=f.sort_values(['entity_id','event_index']).copy()
        f['repeat']=f.receiver_or_mark.eq(f.groupby('entity_id').receiver_or_mark.shift())
        f=f[np.isfinite(f.gap)].copy();f['bin']=np.searchsorted(edges,f.gap.to_numpy(),side='right')
        return f.groupby('bin').repeat.agg(['mean','size']).reindex(range(5))
    a=table(reference);b=table(frame);w=a['size']/a['size'].sum()
    l1=float((w*(a['mean']-b['mean']).abs()).sum())
    shape=float((w*((a['mean']-(w*a['mean']).sum())-(b['mean']-(w*b['mean']).sum())).abs()).sum())
    return l1,shape


def main():
    fitting=run.all_fitted();c=run.config();cond=[];gen=[];curves=[];responses=[];fits=[];inventory=[];traces=[]
    conditional_errors=[];repeat_errors=[];native_errors=[];oracle_rows=[]
    latest_fit=max(x['end_time'] for x in fitting)
    for k in c['kappas']:
        tr,val,tc,vc,plan,pr=inputs(k);ev=evaluators(k)
        for t in c['trials']:
            for name in c['models']:
                dest=run.folder(k,t,name);evaluation=dest/'evaluation'
                done=json.loads((evaluation/'DONE.json').read_text());assert done['start_time']>=latest_fit
                assert done['all_fits']==fitting
                assert done['weights_unchanged'] and not done['test_accessed']
                for f,h in done['files'].items():assert digest(evaluation/f)==h
                verify_evaluation_source(done['source'])
                for f,h in c['parents'][f'{k}/{t}/{name}'].items():assert digest(run.parent.folder_for(k,t,name)/f)==h
                old=run.parent.folder_for(k,t,name)/'evaluation'
                a=json.loads((old/'fit.json').read_text())['controls']['gap'];controls={'A':a}
                tf=pd.read_parquet(old/'train_features.parquet');vf=pd.read_parquet(old/'validation_features.parquet')
                assert set(tf.entity_id)==set(tc.entity_id) and set(vf.entity_id)==set(vc.entity_id)
                target=run.TrainingTarget(tf,a)
                for m in ['B','P']:
                    fit=json.loads((dest/f'{m}_DONE.json').read_text());controls[m]=fit['control']
                    assert fit['sample_calls']==130 and not fit['validation_used'] and not fit['oracle_used']
                    assert not set(fit['fit_positions'])&set(fit['selection_positions'])
                    np.testing.assert_allclose(target.target,fit['target'],atol=1e-15)
                    np.testing.assert_allclose(target.costs(fit['control']['parameters'])-target.base,fit['training_cost_delta'],atol=1e-15)
                    assert len(np.asarray(fit['control']['parameters']).reshape(-1))==10
                    assert fit['control']['edges']==a['edges']
                    assert np.max(np.abs(fit['control']['parameters']))<=8
                    search=fit['search'];assert len(search['trace'])==60 and len(search['selections'])==4
                    if m=='P':
                        assert target.feasible(fit['control']['parameters'])
                        for row in search['trace']+search['selections']:assert target.feasible(row['parameters'])==row['eligible']
                    else:assert all(x['eligible'] for x in search['trace']+search['selections'])
                    expected=min((r for r in search['selections'] if r['eligible']),key=lambda r:(r['objective'],r['endpoint']))
                    selected=search['selections'][search['selected_endpoint']]
                    assert selected['eligible'] and selected['objective']<=expected['objective']+c['tie_improvement']
                    np.testing.assert_array_equal(np.asarray(fit['control']['parameters']).reshape(-1),selected['parameters'])
                    fits.append(dict(kappa=k,trial=t,name=name,variant=m,seconds=fit['seconds'],sample_calls=fit['sample_calls'],
                        selected_endpoint=search['selected_endpoint'],training_guard_satisfied=fit['training_guard_satisfied'],
                        peak_reserved_bytes=fit['peak_reserved_bytes'],control=fit['control'],training_cost_delta=fit['training_cost_delta']))
                    traces.append(dict(kappa=k,trial=t,name=name,variant=m,source=fit['source'],
                        initial_control=fit['initial_control'],target=fit['target'],search=fit['search'],
                        start_manifest_sha256=digest(dest/f'{m}_START.json'),
                        done_manifest_sha256=digest(dest/f'{m}_DONE.json')))
                rows=json.loads((evaluation/'conditional.json').read_text())
                for row in rows:
                    if row['variant']=='raw':continue
                    cond.append(dict(kappa=k,trial=t,name=name,**row))
                    ix=np.ones(len(vf),bool) if row['group']=='pooled' else vf.group.to_numpy()==int(row['group'])
                    z=vf.loc[ix].copy();control=controls[row['variant']]
                    b=np.array([np.searchsorted(control['edges'][int(g)],code,side='right') for code,g in zip(z.gap_code,z.group)])
                    delta=np.asarray(control['parameters'])[z.group.to_numpy(int),b]
                    oldp=z.p.to_numpy(float).clip(1e-9,1-1e-9);p=expit(logit(oldp)+delta);y=z.y.to_numpy()
                    obs=np.where(y==1,p,z.obs_p.to_numpy(float)*(1-p)/(1-oldp))
                    metrics=dict(repeat_brier=float(np.mean((p-y)**2)),repeat_mean_bias=float(abs(p.mean()-y.mean())),
                        mark_nll=float(-np.log(obs.clip(1e-12,1)).mean()))
                    for key,v in metrics.items():
                        error=abs(v-row[key]);assert error<2e-6,(key,error);conditional_errors.append(error)
                resp=json.loads((evaluation/'response.json').read_text())['entity_mean_observable_repeat_range']
                for m in c['methods']:
                    for g in ['0','1']:responses.append(dict(kappa=k,trial=t,name=name,variant=m,group=g,repeat_range=resp[m][g]))
                generated=pd.read_csv(evaluation/'generation_metrics.csv',dtype={'group':str});gen.extend(generated.to_dict('records'))
                curves.extend(pd.read_csv(evaluation/'repeat_curves.csv',dtype={'group':str}).to_dict('records'))
                for f in done['generations']:
                    frame=pd.read_parquet(evaluation/f['file']);validate_frame(frame,plan)
                    for g,(metric,ids,reference) in ev.items():
                        part=frame[frame.entity_id.isin(ids)];r=generated[(generated.variant==f['variant'])&(generated.tape==f['tape'])&(generated.group==g)].iloc[0]
                        independent=independent_repeat(part,reference,metric.state.gap_bin_edges)
                        for x,yv in zip(independent,[r.short_gap_repeat_curve_l1,r.null_centered_shape_l1]):
                            error=abs(x-yv);assert error<1e-12;repeat_errors.append(error)
                        # Every family/condition/trial: full independent native suite on one shared tape and A.
                        if f['variant']=='A' and f['tape']==c['final_generation_seeds'][0]:
                            scores=evaluate_metric_suite(reference,part,metric.state,include_privacy=False)
                            for key,x in scores.items():
                                error=abs(x-r[key]);assert error<1e-10,(key,error);native_errors.append(error)
                inventory.append(dict(kappa=k,trial=t,name=name,evaluation_manifest_sha256=digest(evaluation/'DONE.json'),
                    path=str(evaluation.relative_to(ROOT)),files=done['files']))
        oracle_path=run.OUT/f'oracle/kappa_{k}';done=json.loads((oracle_path/'DONE.json').read_text())
        assert done['start_time']>=latest_fit and done['evaluation_only'] and len(done['inventory'])==60
        assert done['all_fits']==fitting
        for f,h in done['files'].items():assert digest(oracle_path/f)==h
        verify_evaluation_source(done['source'])
        o=pd.read_csv(oracle_path/'generation_metrics.csv',dtype={'group':str});oracle_rows.extend(o.to_dict('records'))
        # All oracle raw-reference repeat scores independently recomputed.
        for f in done['inventory']:
            frame=pd.read_parquet(oracle_path/f['file']);validate_frame(frame,plan)
            for g,(metric,ids,reference) in ev.items():
                r=o[(o['mode']==f['mode'])&(o.tape==f['tape'])&(o.group==g)&(o.reference=='raw')].iloc[0]
                values=independent_repeat(frame[frame.entity_id.isin(ids)],reference,metric.state.gap_bin_edges)
                for x,y in zip(values,[r.short_gap_repeat_curve_l1,r.null_centered_shape_l1]):
                    error=abs(x-y);assert error<1e-12;repeat_errors.append(error)
        inventory.append(dict(oracle_kappa=k,path=str(oracle_path.relative_to(ROOT)),files=done['files'],
            manifest_sha256=digest(oracle_path/'DONE.json')))
    conditional=pd.DataFrame(cond);generation=pd.DataFrame(gen);response=pd.DataFrame(responses);oracle=pd.DataFrame(oracle_rows)
    assert len(fits)==24 and len(generation)==108*3 and len(conditional)==12*3*3
    assert len(oracle)==2*30*3*3
    conditional.to_csv(run.DOC/'conditional_metrics.csv',index=False);generation.to_csv(run.DOC/'generation_metrics.csv',index=False)
    response.to_csv(run.DOC/'fixed_history_response.csv',index=False);pd.DataFrame(curves).to_csv(run.DOC/'repeat_curves.csv',index=False)
    oracle.to_csv(run.DOC/'oracle_metrics.csv',index=False)
    keys=['kappa','trial','name','variant','group']
    means=generation.drop(columns=['tape']).groupby(keys).mean(numeric_only=True).reset_index()
    means.to_csv(run.DOC/'generation_by_parent.csv',index=False)
    conditional.drop(columns=['trial']).groupby(['kappa','name','variant','group']).mean(numeric_only=True).reset_index().to_csv(run.DOC/'conditional_means.csv',index=False)
    means.drop(columns=['trial']).groupby(['kappa','name','variant','group']).mean(numeric_only=True).reset_index().to_csv(run.DOC/'generation_means.csv',index=False)
    og=oracle.groupby(['kappa','mode','reference','group']).short_gap_repeat_curve_l1
    summary=og.agg(['mean','std','min','max']).join(og.quantile([.05,.5,.95]).unstack().rename(columns={.05:'q05',.5:'q50',.95:'q95'})).reset_index()
    summary.to_csv(run.DOC/'oracle_repeat_summary.csv',index=False)
    decisions=decision(conditional,means,response)
    write(run.DOC/'calibration_runs.json',fits)
    write(run.DOC/'optimization_traces.json',traces)
    write(run.DOC/'verification.json',dict(status='PASS',all_24_fits_precede_all_evaluation=True,
        source_hashes=done['source']['hashes'],
        original_weights_and_inputs_hashes_verified=True,all_generation_plans_and_hashes_verified=True,
        ten_parameters_unchanged_capacity=True,train_only_targets_recomputed=True,
        guard_eligibility_and_selection_verified=True,conditional_independent_values=len(conditional_errors),
        conditional_max_error=max(conditional_errors),repeat_independent_values=len(repeat_errors),repeat_max_error=max(repeat_errors),
        native_suite_independent_values=len(native_errors),native_max_error=max(native_errors),
        no_changed_thresholds=True,previous_C_failure_preserved=True))
    write(run.DOC/'execution_summary.json',dict(status='COMPLETE',neural_fits=0,calibration_fits=24,
        fit_sample_calls=24*130,fit_generated_sequences=24*130*2048,final_datasets=108,oracle_datasets=120,
        final_and_oracle_sequences=228*2048,fit_seconds=sum(f['seconds'] for f in fits),
        max_reserved_bytes=max(f['peak_reserved_bytes'] for f in fits),config_sha256=digest(run.CONFIG),inventory=inventory,
        fitting_manifests=fitting,
        independent_data=False,test_accessed=False,scientific_fitting_retries=0,
        technical_oracle_retry_cells=2,technical_failure_events=1))
    print(json.dumps(decisions,indent=2),flush=True)


if __name__=='__main__':main()
