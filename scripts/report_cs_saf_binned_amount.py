"""Apply the fixed last-control decision and report all costs, without retuning."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/cs-saf-binned-amount-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.run_cs_saf_binned_amount import OUT,DOC,CONFIG,OLD,PARENT,inputs,cut_values,tail_generation,write


def main():
    cfg=json.loads(CONFIG.read_text());limits=cfg['screens']
    assert json.loads((DOC/'verification.json').read_text())['passed']
    diagnostic=pd.read_csv(ROOT/'docs/cs_saf/external_amount_diagnostic_v1/tail_summary.csv')
    rows=[];pred=[];tails=[];teacher=[];budgets=[];traces={};archives=[];coverage=[];mark_rates=[]
    for name in cfg['datasets']:
        _,_,_,frames,_,_=inputs(name);cuts=cut_values(frames['fit'])
        support=np.unique(frames['fit'].amount_or_numeric_value.to_numpy(float))
        for role in ('fit','check','validation'):
            values=frames[role].amount_or_numeric_value.to_numpy(float)
            coverage.append(dict(dataset=name,role=role,events=len(values),
                outside_fit_range=int(((values<support.min())|(values>support.max())).sum()),
                exact_amount_not_in_fit=int((~np.isin(values,support)).sum())))
        if name=='berka':
            counts=frames['validation'].receiver_or_mark.value_counts(dropna=False)/len(frames['validation'])
            mark_rates.extend(dict(model='observed_validation',seed=0,mark=str(k),rate=float(v)) for k,v in counts.items())
        for label,folder,reused in [('D',PARENT/'runs'/name/cfg['reference'],True),
                                    ('D_bin',OUT/'runs'/name,False)]:
            done=json.loads((folder/'DONE.json').read_text());start=json.loads((folder/'START.json').read_text())
            result=done['results']['raw'];p=result['prediction']
            pred.append(dict(dataset=name,model=label,amount_score_type='mixed_log_density' if reused else 'bin_cross_entropy',
                             **{k:v for k,v in p.items() if k!='components'},
                             **{k+'_nll':v for k,v in p['components'].items()}))
            budgets.append(dict(dataset=name,model=label,reused=reused,parameters=done['architecture']['parameters'],
                selected_epoch=done['selected_epoch'],last_epoch=done['last_epoch'],stop_reason=done['stop_reason'],
                fit_seconds=done['fit_seconds'],optimizer_updates=done['optimizer_updates'],source_commit=start['source_commit']))
            traces[f'{name}_{label}']=json.loads((folder/'history.json').read_text())
            archives.append(dict(reused=reused,start=start,done=done))
            for item in result['generations']:
                gs=item['seed'];frame=pd.read_parquet(folder/f'generated_raw_{gs}.parquet')
                if name=='berka':
                    counts=frame.receiver_or_mark.value_counts(dropna=False)/len(frame)
                    mark_rates.extend(dict(model=label,seed=gs,mark=str(k),rate=float(v)) for k,v in counts.items())
                rows.append(dict(dataset=name,model=label,seed=gs,**item['metrics']))
                analysis=tail_generation(frames['validation'],frame,cuts,'category' if name=='sparkov' else 'receiver_or_mark')
                for r in analysis:tails.append(dict(dataset=name,model=label,seed=gs,**r))
            for j,threshold in enumerate(('q99','q999','fit_max','ten_fit_max')):
                real_rate=float((frames['validation'].amount_or_numeric_value>cuts[j]).mean())
                if label=='D':
                    key={'q99':'fit_q99','q999':'fit_q999'}.get(threshold,threshold)
                    t=diagnostic.loc[diagnostic.dataset.eq(name)&diagnostic.case.eq('validation')&diagnostic.threshold.eq(key)].iloc[0]
                    assert abs(t.amount_threshold-cuts[j])<1e-8
                    expected=float(t.expected_exceedance)
                else:expected=result['teacher_tail'][j]
                teacher.append(dict(dataset=name,model=label,threshold=threshold,cut=float(cuts[j]),
                    actual=real_rate,predicted=expected,absolute_error=abs(expected-real_rate)))
    gen=pd.DataFrame(rows);prediction=pd.DataFrame(pred);tail=pd.DataFrame(tails);tf=pd.DataFrame(teacher)
    means=gen.groupby(['dataset','model']).mean(numeric_only=True).drop(columns=['seed'])
    ts=tail.groupby(['dataset','model','threshold']).mean(numeric_only=True)
    ps=prediction.set_index(['dataset','model']);tt=tf.set_index(['dataset','model','threshold'])
    screens=[]
    def check(ds,name,value,limit,comparison='<=',reference=None):
        passed=bool(value<=limit) if comparison=='<=' else bool(value>=limit)
        screens.append(dict(dataset=ds,criterion=name,value=float(value),comparison=comparison,
                            limit=float(limit),reference=None if reference is None else float(reference),passed=passed))
    for ds in cfg['datasets']:
        new,old=means.loc[(ds,'D_bin')],means.loc[(ds,'D')]
        pn,po=ps.loc[(ds,'D_bin')],ps.loc[(ds,'D')]
        relation='daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv'
        check(ds,'time_action_tv_cost',new[relation]-old[relation],limits['other_joint_tv_max_cost'],reference=old[relation])
        for metric in ('gap_tv','mark_tv','root_tv','amount_ks'):
            check(ds,metric+'_cost',new[metric]-old[metric],limits['marginal_max_cost'],reference=old[metric])
        check(ds,'mark_nll_cost',pn.mark_nll-po.mark_nll,limits['mark_nll_max_cost'],reference=po.mark_nll)
        check(ds,'repeat_brier_cost',pn.repeat_brier-po.repeat_brier,limits['repeat_brier_max_cost'],reference=po.repeat_brier)
        check(ds,'amount_log_mae_cost',pn.amount_log_mae-po.amount_log_mae,
              max(limits['amount_log_mae_abs_cost'],limits['amount_log_mae_relative_cost']*po.amount_log_mae),reference=po.amount_log_mae)
        for key in ('invalid_amount_rate','invalid_gap_rate'):
            check(ds,key,new[key],0.)
        for key in ('q99','q999'):
            n,b=ts.loc[(ds,'D_bin',key)],ts.loc[(ds,'D',key)]
            ratio=n.generated_rate/n.real_rate
            check(ds,key+'_generated_tail_min_ratio',ratio,limits['generated_tail_rate_ratio_min'],'>=')
            check(ds,key+'_generated_tail_max_ratio',ratio,limits['generated_tail_rate_ratio_max'])
            check(ds,key+'_root_tail_l1_cost',n.root_tail_l1-b.root_tail_l1,limits[key+'_tail_relation_max_cost'],reference=b.root_tail_l1)
            n,b=tt.loc[(ds,'D_bin',key)],tt.loc[(ds,'D',key)]
            if ds=='berka':
                check(ds,key+'_teacher_tail_error',n.absolute_error,limits['berka_teacher_tail_error_ratio_max']*b.absolute_error,reference=b.absolute_error)
            else:
                check(ds,key+'_teacher_tail_error_cost',n.absolute_error-b.absolute_error,limits['sparkov_'+key+'_teacher_tail_error_max_cost'],reference=b.absolute_error)
        if ds=='berka':
            gain=old.root_amount_joint_tv-new.root_amount_joint_tv
            check(ds,'amount_relation_gain',gain,max(limits['berka_amount_tv_abs_gain'],limits['berka_amount_tv_relative_gain']*old.root_amount_joint_tv),'>=',old.root_amount_joint_tv)
        else:
            check(ds,'amount_relation_cost',new.root_amount_joint_tv-old.root_amount_joint_tv,limits['other_joint_tv_max_cost'],reference=old.root_amount_joint_tv)
    screen=pd.DataFrame(screens)
    passed={ds:bool(screen.loc[screen.dataset.eq(ds),'passed'].all()) for ds in cfg['datasets']}
    decision=('adopt_D_bin_as_fixed_working_model' if all(passed.values()) else
              'berka_only_candidate_no_common_adoption' if passed['berka'] else 'reject_amount_change_keep_D_reference')
    conclusion=dict(decision=decision,passed_by_dataset=passed,
        failed_checks={ds:screen.loc[screen.dataset.eq(ds)&~screen.passed,'criterion'].tolist() for ds in cfg['datasets']},
        new_fits=2,new_generations=4,new_calibrations=0,automatic_followup=False,
        novel_method_claim=False,independent_validation=False)
    for label,frame in [('generation_metrics',gen),('generation_summary',means.reset_index()),('prediction_metrics',prediction),
                        ('tail_metrics',tail),('teacher_tail',tf),('training_budgets',pd.DataFrame(budgets)),
                        ('support_coverage_descriptive',pd.DataFrame(coverage)),('screens',screen)]:
        frame.to_csv(DOC/f'{label}.csv',index=False)
    write(DOC/'decision.json',conclusion);write(DOC/'training_traces.json',traces);write(DOC/'results.json',archives)
    pd.DataFrame(mark_rates).to_csv(DOC/'berka_operation_rates_descriptive.csv',index=False)
    external=pd.read_csv(ROOT/'docs/cs_saf/external_controls_v1/generation_summary.csv')
    comparison=[]
    for ds in cfg['datasets']:
        relation='daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv'
        for label in ('D','D_bin'):
            row=means.loc[(ds,label)]
            comparison.append(dict(dataset=ds,model=label,time_action_tv=row[relation],action_amount_tv=row.root_amount_joint_tv))
        for label in ('Transition','Marginal','ARGN','CPAR_tail'):
            row=external[external.dataset.eq(ds)&external.model.eq(label)&external.variant.eq('raw')].iloc[0]
            comparison.append(dict(dataset=ds,model=label,time_action_tv=row[relation],action_amount_tv=row.root_amount_joint_tv))
    comparison=pd.DataFrame(comparison);comparison.to_csv(DOC/'reference_comparison.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    order=['D','D_bin','Transition']
    for i,ds in enumerate(cfg['datasets']):
        c=comparison[comparison.dataset.eq(ds)].set_index('model').loc[order]
        for j,metric in enumerate(('time_action_tv','action_amount_tv')):
            ax=axes[i,j];ax.bar(order,c[metric],color=['#5487b4','#d38e42','#589e87'])
            for k,label in enumerate(('D','D_bin')):
                key=('daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv') if j==0 else 'root_amount_joint_tv'
                vals=gen.loc[gen.dataset.eq(ds)&gen.model.eq(label),key].to_numpy()
                ax.scatter(np.full(2,k),vals,color='black',s=15,zorder=3)
            ax.set_title(ds.title()+': '+('time-action' if j==0 else 'action-amount'));ax.set_ylabel('TV (lower is better)')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('One fixed amount-output change; two generation draws\nDots are individual draws, not independent training runs')
    fig.savefig(DOC/'relations.png',dpi=170);fig.savefig(DOC/'relations.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,ds in zip(axes,cfg['datasets']):
        labels=['Real\nvalidation','D\nreal history','D_bin\nreal history','D\ngeneration','D_bin\ngeneration']
        values=[tt.loc[(ds,'D','q999')].actual,tt.loc[(ds,'D','q999')].predicted,
                tt.loc[(ds,'D_bin','q999')].predicted,ts.loc[(ds,'D','q999')].generated_rate,
                ts.loc[(ds,'D_bin','q999')].generated_rate]
        ax.bar(labels,np.array(values)*100,color=['#777777','#5487b4','#d38e42','#5487b4','#d38e42'])
        ax.set_title(ds.title());ax.set_ylabel('Above fit q99.9 (%)');ax.tick_params(axis='x',labelsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Large-amount frequency must be preserved, not removed')
    fig.savefig(DOC/'upper_amount.png',dpi=170);fig.savefig(DOC/'upper_amount.pdf');plt.close(fig)
    print(comparison.to_string(index=False));print(prediction.to_string(index=False))
    print(tf.to_string(index=False));print(screen.loc[~screen.passed].to_string(index=False))
    print(json.dumps(conclusion,indent=2))


if __name__=='__main__':main()
