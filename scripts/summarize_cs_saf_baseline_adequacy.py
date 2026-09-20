"""All registered stage-one cells, native metrics and predefined screening gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import digest,write
from scripts.continue_cs_saf_argn_adequacy import CONFIG,OUT,OLD
from experiments.cs_saf_generation_metrics import CachedGroupMetrics


def evaluate(partial=False):
    c=json.loads(CONFIG.read_text());report=ROOT/'docs/cs_saf/baseline_adequacy_v1'
    metric_rows=[];conditional=[];last=[];inventory=[];curve_rows=[]
    old_metrics=pd.read_csv(ROOT/'docs/cs_saf/external_audit_v1/metrics.csv')
    reused_differences=[]
    for k in c['kappas']:
        inp=OLD/f'kappa_{k}/input';provenance=json.loads((inp/'provenance.json').read_text())
        for name,sha in provenance['files'].items():assert digest(inp/name)==sha
        train=pd.read_parquet(inp/'train_events.parquet');train['event_index']=train.groupby('entity_id').cumcount()
        train.loc[train.event_index==0,'gap']=np.nan
        parent=pd.read_parquet(inp/'train_context.parquet');vp=pd.read_parquet(inp/'validation_context.parquet')
        val=pd.read_parquet(inp/'validation_canonical.parquet');plan=pd.read_parquet(inp/'plan.parquet')
        groups={}
        for group in ('pooled','0','1'):
            def ids(p):return set(p.entity_id if group=='pooled' else p.loc[p.entity_label.astype(str)==group,'entity_id'])
            ti,vi,gi=ids(parent),ids(vp),ids(plan)
            groups[group]=(CachedGroupMetrics(train[train.entity_id.isin(ti)],val[val.entity_id.isin(vi)]),gi)
        jobs=[(family,index) for family in ('argn_original','argn_continued','U')
              for index in (c['u_trials'] if family=='U' else c['argn_seeds'])]
        for family,index in jobs:
            folder=OUT/f'controls/{family}/kappa_{k}/{"trial" if family=="U" else "seed"}_{index}'
            if partial and not (folder/'DONE.json').exists():continue
            done=json.loads((folder/'DONE.json').read_text());fit=json.loads((folder/'fit.json').read_text())
            assert done['weights_unchanged'] and not done['smoke'] and fit['config_sha256']==digest(CONFIG)
            assert fit['feature_sha256']==digest(folder/'train_features.parquet')
            tf=pd.read_parquet(folder/'train_features.parquet');vf=pd.read_parquet(folder/'validation_features.parquet')
            assert not set(tf.entity_id)&set(vf.entity_id)
            assert set(tf.entity_id)==set(parent.entity_id) and set(vf.entity_id)==set(vp.entity_id)
            assert len(tf)==len(train)-len(parent) and len(vf)==len(val)-len(vp)
            for row in json.loads((folder/'conditional.json').read_text()):
                conditional.append(dict(family=family,kappa=k,index=index,**row))
            if (folder/'last_conditional.json').exists():
                for row in json.loads((folder/'last_conditional.json').read_text())['metrics']:
                    last.append(dict(family=family,kappa=k,index=index,**row))
            files={}
            for variant in c['variants']:
                for tape in c['generation_seeds']:
                    file=folder/f'generated_{variant}_{tape}.parquet'
                    generated=pd.read_parquet(file)
                    assert set(generated.entity_id)==set(plan.entity_id)
                    assert not (set(generated.receiver_or_mark)-set(train.receiver_or_mark))
                    ix=generated.groupby('entity_id',sort=False).cumcount()
                    if 'event_index' in generated:assert np.array_equal(generated.event_index,ix)
                    generated['event_index']=ix;generated.loc[ix==0,'gap']=np.nan
                    assert np.isfinite(generated.loc[ix>0,'gap']).all() and (generated.loc[ix>0,'gap']>=0).all()
                    assert np.isfinite(generated.amount_or_numeric_value).all()
                    for group,(metric,ids) in groups.items():
                        part=generated[generated.entity_id.isin(ids)]
                        values=metric.score(part);s=metric.sufficient_statistics(part)
                        a=metric.reference['repeat'];b=s['repeat'];w=a.sum(1)/a.sum()
                        reference=a[:,1]/a.sum(1);curve=b[:,1]/b.sum(1)
                        assert np.isfinite(curve).all()
                        centered=float(w@np.abs((curve-w@curve)-(reference-w@reference)))
                        row=dict(family=family,kappa=k,index=index,variant=variant,tape=tape,group=group,
                            entities=part.entity_id.nunique(),events=len(part),**values,
                            repeat_level_reference_weighted=float(w@curve),
                            reference_repeat_level=float(w@reference),null_centered_shape_l1=centered)
                        metric_rows.append(row)
                        for bin_id in range(len(w)):
                            curve_rows.append(dict(family=family,kappa=k,index=index,variant=variant,tape=tape,
                                group=group,bin=bin_id,observed_repeat=float(reference[bin_id]),generated_repeat=float(curve[bin_id])))
                        if family=='argn_original' and variant=='raw':
                            old=old_metrics[(old_metrics.kappa==k)&(old_metrics.seed==index)&(old_metrics.tape==tape)&
                                (old_metrics.kind=='generated')&(old_metrics.group==('pooled' if group=='pooled' else 'context_'+group))].iloc[0]
                            reused_differences += [abs(float(old[name])-value) for name,value in values.items()]
                    files[file.name]=digest(file)
            inventory.append(dict(family=family,kappa=k,index=index,path=str(folder.relative_to(ROOT)),done=done,
                fit_sha256=digest(folder/'fit.json'),train_feature_sha256=fit['feature_sha256'],
                validation_feature_sha256=digest(folder/'validation_features.parquet'),generated_sha256=files))
    if not partial:assert len(inventory)==12
    assert inventory
    metrics=pd.DataFrame(metric_rows);cond=pd.DataFrame(conditional)
    metrics.to_csv(report/'generation_metrics.csv',index=False);cond.to_csv(report/'conditional_metrics.csv',index=False)
    pd.DataFrame(last).to_csv(report/'final_checkpoint_conditional.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(report/'repeat_curves.csv',index=False)
    metrics.drop(columns=['index','tape']).groupby(['family','kappa','variant','group']).mean(numeric_only=True).reset_index().to_csv(report/'generation_means.csv',index=False)
    cond.drop(columns=['index']).groupby(['family','kappa','variant','group']).mean(numeric_only=True).reset_index().to_csv(report/'conditional_means.csv',index=False)
    gate=c['conditional_gate']
    cond['passes_basic_prediction']=(cond.repeat_mean_bias<=gate['max_absolute_repeat_mean_bias'])&(
        cond.repeat_brier<=cond.train_mean_brier+gate['max_brier_above_train_mean_control'])&(
        cond.mark_nll<=cond.uniform_mark_nll-gate['min_mark_nll_improvement_over_uniform'])
    cond['passes_conditional_screen']=cond.passes_basic_prediction&(
        ~((cond.kappa==1)&(cond.group=='1'))|(cond.curve_l1<=gate['active_curve_l1_max']))
    cond.to_csv(report/'conditional_screens.csv',index=False)
    group_cols=['family','kappa','index','group']
    parent_means=metrics.groupby(group_cols+['variant']).mean(numeric_only=True).reset_index()
    raw=parent_means[parent_means.variant=='raw'].set_index(group_cols)
    screens=[];g=c['generation_screen']
    for row in parent_means.to_dict('records'):
        native=raw.loc[tuple(row[x] for x in group_cols)]
        deltas={key:float(row[key]-native[key]) for key in ['gap_ks','mark_sparse_tv','amount_ks','length_scaled_w1']}
        relation=row['short_gap_repeat_curve_l1']<=g['repeat_curve_l1_max']
        null=(row['kappa']==0 or row['group']=='0')
        shape=not null or row['null_centered_shape_l1']<=g['null_centered_shape_l1_max']
        cost=all(deltas[key]<=g['max_gap_mark_amount_distribution_increase'] for key in ['gap_ks','mark_sparse_tv','amount_ks'])
        cost=cost and deltas['length_scaled_w1']<=g['max_length_scaled_w1_increase']
        screens.append({**{key:row[key] for key in group_cols+['variant']},'passes_relation':relation,
            'passes_null_shape':shape,'passes_distribution_cost':cost,'passes_generation_screen':relation and shape and cost,
            **{'delta_'+key:value for key,value in deltas.items()}})
    pd.DataFrame(screens).to_csv(report/'generation_screens.csv',index=False)
    write(report/'execution_summary.json',dict(status='COMPLETE' if len(inventory)==12 else 'PARTIAL',
        config_sha256=digest(CONFIG),completed_parent_controls=len(inventory),registered_parent_controls=12,
        generation_conditions=len(metrics)//3,registered_generation_conditions=144,
        reused_native_metrics_max_difference=max(reused_differences,default=0),
        new_method_neural_training=False,independent_data=False,test_accessed=False,
        screening_not_significance_or_convergence_proof=True,parents=inventory))
    assert max(reused_differences,default=0)<1e-12
    print(cond.groupby(['family','kappa','variant','group'])[['repeat_brier','mark_nll','curve_l1']].mean().to_string())
    print(parent_means.groupby(['family','kappa','variant','group'])[['short_gap_repeat_curve_l1','mark_sparse_tv']].mean().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();evaluate(a.partial)
