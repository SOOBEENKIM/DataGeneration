"""All registered cells, primary decisions and independent artifact/fit checks."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cs_saf_structure_v1 import config,CONFIG,ROOT,OUT,DOC,OLD,folder_for
from experiments.cs_saf_generation_metrics import CachedGroupMetrics
from models.cs_saf_observed_repeat_control import assign_bins
from scripts.run_cs_saf_external_audit_v1 import digest,write


def basic(row,c):
    return bool(row.repeat_mean_bias<=c['max_absolute_repeat_mean_bias'] and
        row.repeat_brier<=row.train_mean_brier+c['max_brier_above_train_mean_control'] and
        row.mark_nll<=row.uniform_mark_nll-c['min_mark_nll_improvement_over_uniform'] and
        (row.kappa!=1 or row.group!='1' or row.curve_l1<=c['active_curve_l1_max']))


def decide(cond,gen,c):
    keys=['kappa','trial','name','variant','group']
    means=gen.drop(columns=['tape']).groupby(keys).mean(numeric_only=True).reset_index()
    means.to_csv(DOC/'generation_by_parent.csv',index=False)
    cm=cond.set_index(keys);gm=means.set_index(keys)
    rows=[];costs=[];eligibility=[]
    for k in c['kappas']:
        for t in range(3):
            for g in ['0','1']:
                for name,variant in [('C','raw'),('U','gap')]:
                    row=cm.loc[(k,t,name,variant,g)]
                    eligibility.append(dict(kappa=k,trial=t,name=name,variant=variant,group=g,
                        passes_basic=basic(pd.Series(dict(row,kappa=k,group=g)),c['candidate_and_Ugap_basic_prediction'])))
                ca=cm.loc[(k,t,'C','raw',g)];ga=gm.loc[(k,t,'C','raw',g)]
                null=(k==0 or g=='0')
                for name,variant in [('G','raw'),('U','gap')]:
                    cb=cm.loc[(k,t,name,variant,g)];gb=gm.loc[(k,t,name,variant,g)]
                    dc={key:float(ga[key]-gb[key]) for key in ['gap_ks','mark_sparse_tv','amount_ks','length_scaled_w1']}
                    brier=float(ca.repeat_brier-cb.repeat_brier);nll=float(ca.mark_nll-cb.mark_nll)
                    cl=c['conditional_cost_limits_each_seed_and_group_against_both_comparators']
                    gl=c['generation_cost_limits_against_both_comparators_each_seed_group']
                    cp=(brier<=cl['max_brier_increase'] and nll<=cl['max_mark_nll_increase'])
                    gp=all(dc[key]<=gl['max_'+key+'_increase'] for key in dc)
                    npass=(not null or (ga.short_gap_repeat_curve_l1<=gl['max_null_repeat_curve_L1'] and
                        ga.null_centered_shape_l1<=gl['max_null_centered_shape_L1']))
                    costs.append(dict(kappa=k,trial=t,group=g,comparator=f'{name}/{variant}',
                        delta_repeat_brier=brier,delta_mark_nll=nll,passes_conditional_cost=bool(cp),
                        passes_distribution_cost=bool(gp),passes_null=bool(npass),
                        **{'delta_'+key:value for key,value in dc.items()}))
    for variant,comparator,role in [('raw',('G','raw'),'primary'),('raw',('U','gap'),'primary'),
                                  ('gap',('G','gap'),'secondary'),('gap',('U','gap'),'secondary')]:
        differences=[];cand=[];ref=[]
        for t in range(3):
            av=float(gm.loc[(1,t,'C',variant,'1')].short_gap_repeat_curve_l1)
            bv=float(gm.loc[(1,t,*comparator,'1')].short_gap_repeat_curve_l1)
            differences.append(av-bv);cand.append(av);ref.append(bv)
        gate=c['primary_generation_improvement'];reduction=float(np.mean(ref)-np.mean(cand))
        relative=reduction/float(np.mean(ref))
        passed=(reduction>=gate['minimum_absolute_mean_reduction'] and relative>=gate['minimum_relative_mean_reduction'] and
            sum(x<0 for x in differences)>=gate['minimum_improving_training_seeds'] and
            max(differences)<=gate['maximum_single_seed_L1_increase'] and max(cand)<=gate['maximum_candidate_L1_each_seed'])
        rows.append(dict(candidate=f'C/{variant}',comparator='/'.join(comparator),role=role,
            candidate_values=cand,comparator_values=ref,paired_differences=differences,
            absolute_mean_reduction=reduction,relative_mean_reduction=relative,
            improving_seeds=sum(x<0 for x in differences),passes_generation_improvement=bool(passed)))
    cost=pd.DataFrame(costs);eligible=pd.DataFrame(eligibility)
    cost.to_csv(DOC/'primary_costs.csv',index=False);eligible.to_csv(DOC/'basic_prediction_screens.csv',index=False)
    passed=all(row['passes_generation_improvement'] for row in rows if row['role']=='primary') and bool(
        cost[['passes_conditional_cost','passes_distribution_cost','passes_null']].all().all()) and bool(eligible.passes_basic.all())
    write(DOC/'decision.json',dict(primary_pass=passed,advance=passed,generation_contrasts=rows,
        conditional_cost_all_pass=bool(cost.passes_conditional_cost.all()),distribution_cost_all_pass=bool(cost.passes_distribution_cost.all()),
        null_all_pass=bool(cost.passes_null.all()),basic_prediction_all_pass=bool(eligible.passes_basic.all()),
        thresholds_changed=False,secondary_cannot_rescue_primary=True,not_statistical_superiority=True))
    return passed


def main():
    c=config();cond=[];generated=[];curves=[];inventory=[];fits=[];responses=[];training=[]
    optimum_errors=[];init={};orders={}
    for k in c['kappas']:
        inp=OLD/f'kappa_{k}/input';provenance=json.loads((inp/'provenance.json').read_text())
        for file,sha in provenance['files'].items():assert digest(inp/file)==sha
        tr=pd.read_parquet(inp/'train_events.parquet');tr['event_index']=tr.groupby('entity_id').cumcount()
        tr.loc[tr.event_index==0,'gap']=np.nan
        val=pd.read_parquet(inp/'validation_canonical.parquet');tc=pd.read_parquet(inp/'train_context.parquet')
        vc=pd.read_parquet(inp/'validation_context.parquet');plan=pd.read_parquet(inp/'plan.parquet')
        evaluators={}
        for group in ['pooled','0','1']:
            def ids(frame):return set(frame.entity_id if group=='pooled' else frame[frame.entity_label.astype(str)==group].entity_id)
            ti,vi,pi=ids(tc),ids(vc),ids(plan)
            evaluators[group]=(CachedGroupMetrics(tr[tr.entity_id.isin(ti)],val[val.entity_id.isin(vi)]),pi)
        for trial in range(3):
            for name in c['models']:
                folder=folder_for(k,trial,name);done=json.loads((folder/'EVAL_DONE.json').read_text());ev=folder/'evaluation'
                trained=json.loads((folder/'TRAIN_DONE.json').read_text())
                assert done['weights_unchanged'] and not done['test_accessed'] and not trained['smoke']
                assert trained['parameters']==c['parameters_each'] and done['config_sha256']==digest(CONFIG)
                for file,sha in done['files'].items():assert digest(ev/file)==sha
                assert digest(folder/'checkpoint_best.pt')==done['checkpoint_sha256']==trained['checkpoint_sha256']
                assert digest(folder/'checkpoint_last.pt')==trained['last_checkpoint_sha256']
                assert digest(trained['data']['cache_path'])==trained['data']['cache_sha256']
                for file,sha in done['source']['hashes'].items():assert digest(ROOT/file)==sha
                init.setdefault((k,trial),[]).append(trained['initial_state_sha256'])
                orders.setdefault((k,trial),[]).append([r['order_sha256'] for r in trained['history']])
                training.append(dict(kappa=k,trial=trial,name=name,epochs=trained['epochs'],best_epoch=trained['best_epoch'],
                    best_validation_nll=trained['best_validation_nll'],seconds=trained['seconds'],parameters=trained['parameters'],
                    peak_reserved_bytes=trained['peak_reserved_bytes']))
                tf=pd.read_parquet(ev/'train_features.parquet');vf=pd.read_parquet(ev/'validation_features.parquet')
                assert set(tf.entity_id)==set(tc.entity_id) and set(vf.entity_id)==set(vc.entity_id)
                assert not set(tf.entity_id)&set(vf.entity_id)
                assert len(tf)==len(tr)-len(tc) and len(vf)==len(val)-len(vc)
                fit=json.loads((ev/'fit.json').read_text());assert fit['feature_sha256']==digest(ev/'train_features.parquet')
                for kind,control in fit['controls'].items():
                    edges=[np.quantile(tf[tf.group==g].gap_code,c['coarse_quantiles']).tolist() for g in (0,1)]
                    np.testing.assert_array_equal(edges,control['edges'])
                    bins=assign_bins(tf.gap_code.to_numpy(),tf.group.to_numpy(),edges)
                    for group in (0,1):
                        for b in range(5):
                            mask=(tf.group.to_numpy()==group)&(bins==b);y=tf.y.to_numpy(float)[mask]
                            z=logit(tf.p.to_numpy(float)[mask].clip(1e-9,1-1e-9));saved=control['parameters'][group][b]
                            if kind=='direct':expected=(y.sum()+.5)/(len(y)+1)
                            else:
                                def der(delta):return float(np.mean(expit(z+delta)-y)+c['ridge']*delta)
                                bound=c['offset_bound'];expected=-bound if der(-bound)>=0 else bound if der(bound)<=0 else brentq(der,-bound,bound,xtol=1e-12)
                            error=abs(saved-expected);optimum_errors.append(error);assert error<1e-4
                    fits.append(dict(kappa=k,trial=trial,name=name,variant=kind,control=control))
                for row in json.loads((ev/'conditional.json').read_text()):cond.append(dict(kappa=k,trial=trial,name=name,**row))
                response=json.loads((ev/'response.json').read_text())
                assert response['raw_C_marginal_max_error']<1e-6
                responses.append(dict(kappa=k,trial=trial,name=name,**response))
                for run in done['generations']:
                    frame=pd.read_parquet(ev/run['file']);assert digest(ev/run['file'])==run['sha256']
                    assert set(frame.entity_id)==set(plan.entity_id)
                    assert np.isfinite(frame.amount_or_numeric_value).all()
                    np.testing.assert_array_equal(frame.groupby('entity_id').size().reindex(plan.entity_id),plan['__saf_planned_length'])
                    assert set(frame.receiver_or_mark)<=set(tr.receiver_or_mark)
                    nonfirst=frame.groupby('entity_id').cumcount()>0
                    assert frame.loc[~nonfirst,'gap'].isna().all() and np.isfinite(frame.loc[nonfirst,'gap']).all()
                    assert (frame.loc[nonfirst,'gap']>=0).all()
                    for group,(e,pi) in evaluators.items():
                        part=frame[frame.entity_id.isin(pi)];metrics=e.score(part);s=e.sufficient_statistics(part)
                        a=e.reference['repeat'];b=s['repeat'];w=a.sum(1)/a.sum();real=a[:,1]/a.sum(1);fake=b[:,1]/b.sum(1)
                        assert np.isfinite(fake).all()
                        centered=float(w@abs((fake-w@fake)-(real-w@real)))
                        generated.append(dict(kappa=k,trial=trial,name=name,variant=run['variant'],tape=run['tape'],group=group,
                            **metrics,null_centered_shape_l1=centered,generated_repeat_level=float(w@fake),reference_repeat_level=float(w@real)))
                        for j in range(5):curves.append(dict(kappa=k,trial=trial,name=name,variant=run['variant'],tape=run['tape'],group=group,
                            bin=j,reference=float(real[j]),generated=float(fake[j])))
                inventory.append(dict(kappa=k,trial=trial,name=name,path=str(folder.relative_to(ROOT)),
                    training_manifest_sha256=digest(folder/'TRAIN_DONE.json'),evaluation_manifest_sha256=digest(folder/'EVAL_DONE.json'),
                    checkpoint_sha256=done['checkpoint_sha256'],files=done['files']))
    assert len(inventory)==18 and len(generated)==162*3
    for values in init.values():assert len(set(values))==1
    for values in orders.values():
        n=min(map(len,values));assert all(v[:n]==values[0][:n] for v in values)
    conditional=pd.DataFrame(cond);gen=pd.DataFrame(generated)
    conditional.to_csv(DOC/'conditional_metrics.csv',index=False);gen.to_csv(DOC/'generation_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(DOC/'repeat_curves.csv',index=False)
    pd.DataFrame(training).to_csv(DOC/'training_runs.csv',index=False)
    conditional.drop(columns=['trial']).groupby(['kappa','name','variant','group']).mean(numeric_only=True).reset_index().to_csv(DOC/'conditional_means.csv',index=False)
    gen.drop(columns=['trial','tape']).groupby(['kappa','name','variant','group']).mean(numeric_only=True).reset_index().to_csv(DOC/'generation_means.csv',index=False)
    passed=decide(conditional,gen,c)
    write(DOC/'response_diagnostics.json',responses);write(DOC/'calibration_coefficients.json',fits)
    write(DOC/'verification.json',dict(status='PASS',source_hashes=done['source']['hashes'],
        all_common_initial_states_identical=True,all_shared_epoch_orders_identical=True,
        original_cache_and_checkpoint_hashes_match=True,all_generation_plans_and_hashes_match=True,
        independent_calibration_optimum_max_error=max(optimum_errors),all_validation_entities_disjoint=True,
        C_fixed_history_constraints_verified=True))
    write(DOC/'execution_summary.json',dict(status='COMPLETE',fits=18,generated_datasets=162,generated_sequences=162*2048,
        config_sha256=digest(CONFIG),primary_pass=passed,independent_data=False,test_accessed=False,
        additional_external_training=False,inventory=inventory))
    print('COMPLETE; primary_pass=',passed,flush=True)


if __name__=='__main__':main()
